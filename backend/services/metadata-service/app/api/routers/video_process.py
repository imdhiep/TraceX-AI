"""Video processing pipeline for metadata-service — SOTA 2026 AI.

Production single-video pipeline (`_process_video_sync`):
  1. VideoFrameSampler (6 fps default, Laplacian sharpness scoring)
  2. RT-DETR R50 person detection (primary) / Grounding DINO 1.6 (fallback)
  3. BoT-SORT tracker (Kalman + IoU, ReID disabled, GMC disabled) —
     legacy BodyPartAdaptiveTracker available via TRACER_BACKEND=adaptive
  4. TrackletQualityScorer (min_frames, density, duration, Laplacian)
  5. SigLIP 2-So400m (1152-dim) — multi-frame pool-avg per fragment
  6. TrackletFragmentMerger (cosine ≥ 0.85, max_gap ≤ 60 s, Union-Find)
  7. Qwen2.5-VL-7B-Instruct open-vocabulary attribute captioning
     (only on MERGED tracklets, batch with OOM-aware backoff)
  8. VideoMAE V2 action classification
     (per-frame bbox crops, Kinetics-400 → TraceX taxonomy)
  9. BEV projection + crop save + DB write

DB embedding = L2-normalized pool-avg of multi-frame SigLIP features across
all fragments in the merged group (DINOv2 has been removed).

Cross-camera batch pipeline (`process_batch`):
  - Stage 1-3: parallel per video
  - Stage 4: ONE MCBLT call across all cameras
  - Stage 5-7: per unified tracklet (uses legacy `_caption_crop_vlm` path)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ...services.model_warmup import get_model
from .video_process_schemas import (
    BatchProcessRequest,
    BatchProcessResponse,
    BatchVideoEntry,
    ProcessVideoRequest,
    ProcessVideoResponse,
    ProcessVideoStreamRequest,
    TrackletResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video"])


def _get_positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
        return default

#test thử coolify

DEFAULT_SAMPLE_INTERVAL = 15
DEFAULT_MIN_BBOX_AREA = 400
DEFAULT_BEV_MAX_DIST = 1.5
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))
VLM_BATCH_SIZE = _get_positive_env_int("VLM_BATCH_SIZE", 4)
VLM_BATCH_MAX_SIZE = _get_positive_env_int("VLM_BATCH_MAX_SIZE", 8)
VLM_BATCH_GROW_STEP = _get_positive_env_int("VLM_BATCH_GROW_STEP", 1)
VLM_BATCH_STABLE_STEPS = _get_positive_env_int("VLM_BATCH_STABLE_STEPS", 3)
# Per-crop budget for the FULL schema (~30 fields, of which 6 are free-text:
# upper_clothing_desc, lower_clothing_desc, shoes_desc, bag_desc, hat_desc,
# appearance_summary). Token count for a complete object is typically 330-380;
# 512 gives enough headroom that the trailing fields (often `appearance_summary`
# or `hair_color`) don't get truncated, which would drop the entire crop's
# attrs to defaults via the JSON-regex fallback.
VLM_BATCH_MAX_NEW_TOKENS_PER_CROP = _get_positive_env_int(
    "VLM_BATCH_MAX_NEW_TOKENS_PER_CROP", 448
)
# Bench cam_0002 + production review: 0.85 đang merge nhầm các tracklet
# có đồng phục giống nhau (nhân viên/hồ sơ bệnh nhân tương tự). Nâng lên 0.90
# để chỉ merge khi appearance thực sự rất gần. Hệ quả: số merged tracklets
# tăng (kỳ vọng 30-50%), nhưng độ tinh khiết group tăng đáng kể.
# Component margin 0.03 → 0.05 → floor = 0.85 (cùng giá trị threshold cũ),
# vẫn cho phép expand group qua các fragment trung gian.
# 2026-05-17: tightened after audit showed hospital-uniform false-positive
# merges (group span 500-600s on 10-min videos, 18 fragments → 1 candidate).
#   SIM_THRESHOLD 0.89 → 0.93 : cosine 0.89-0.92 is ambiguous for uniformed
#     subjects (e.g. nurses in similar scrubs); 0.93 excludes the ambiguous band.
#   MAX_GAP_SECONDS 180 → 30 : long-gap appearance bridges across 2-3 minutes
#     are almost always a different person passing through, not the same person
#     returning. BoT-SORT already covers short-gap re-entry inside track_buffer.
#   MAX_SPEED_PX_PER_S 800 → 300 : 800 px/s overstates hospital walking speed
#     (~150-300 px/s) and made the spatial gate ineffective. 300 px/s rejects
#     cross-frame appearance bridges with implausible displacement.
FRAGMENT_MERGE_SIM_THRESHOLD = _get_env_float("FRAGMENT_MERGE_SIM_THRESHOLD", 0.93)
FRAGMENT_MERGE_MAX_GAP_SECONDS = _get_env_float("FRAGMENT_MERGE_MAX_GAP_SECONDS", 30.0)
FRAGMENT_MERGE_COMPONENT_MARGIN = _get_env_float("FRAGMENT_MERGE_COMPONENT_MARGIN", 0.02)
FRAGMENT_MERGE_MAX_SPEED_PX_PER_S = _get_env_float("FRAGMENT_MERGE_MAX_SPEED_PX_PER_S", 300.0)
FRAGMENT_MERGE_SPATIAL_BYPASS_MARGIN = _get_env_float("FRAGMENT_MERGE_SPATIAL_BYPASS_MARGIN", 0.05)
# 2026-05-17: hard cap on the appearance-pass spatial budget. Without this,
# at gap=30s with max_speed=300 the budget reaches 9000 px (wider than any
# FOV) → spatial gate becomes a no-op. 400 px ≈ 1/5 of 1080p width, matching
# plausible cross-frame travel for one continuous appearance. 0 disables.
FRAGMENT_MERGE_MAX_SPATIAL_DIST_PX = _get_env_float("FRAGMENT_MERGE_MAX_SPATIAL_DIST_PX", 400.0)
# Motion-merge appearance floor — block "two different people crossing at the
# same point" (dist≈0px but cos far below sim_thresh). Stays below sim_thresh
# so SigLIP-borderline same-person rescues remain possible.
FRAGMENT_MERGE_MOTION_MIN_SIM = _get_env_float("FRAGMENT_MERGE_MOTION_MIN_SIM", 0.85)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda")


def _video_to_frames(video_path: str, max_frames: int = 500) -> tuple[list[np.ndarray], float]:
    """Load frames from video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _sample_frames_uniform(
    frames: list[np.ndarray], fps: float, sample_interval: int = 15
) -> list[tuple[int, np.ndarray]]:
    """Sample frames at uniform intervals with their indices."""
    sampled = []
    for frame_idx in range(0, len(frames), sample_interval):
        sampled.append((frame_idx, frames[frame_idx]))
    return sampled


# ---------------------------------------------------------------------------
# Stage 2: Person Detection (RT-DETR primary / GDINO legacy fallback)
# ---------------------------------------------------------------------------

def _detect_persons(frame: np.ndarray, threshold: float = 0.3) -> list[dict]:
    """Legacy single-frame GDINO detection — only called from batch error fallback."""
    model = get_model("gdino16")
    processor = get_model("gdino16_processor")
    if model is None or processor is None:
        logger.warning("Grounding DINO not available, returning empty detections")
        return []

    device = _get_device()
    dtype = torch.float16
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, text="person.", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    autocast_ctx = torch.autocast("cuda", dtype=dtype)
    with torch.no_grad(), autocast_ctx:
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=threshold,
        text_threshold=threshold,
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        if score < threshold:
            continue
        if label.lower() != "person":
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(score),
            "label": label,
        })

    return detections


def _detect_persons_rtdetr(
    frames: list[np.ndarray],
    threshold: float = 0.4,
    batch_size: int = 128,
) -> list[list[dict]] | None:
    """RT-DETR R50 person detection — primary detector when loaded.
    Returns None if not available (caller falls back to GDINO).
    batch_size=128 on A100 80GB (256 causes OOM).

    Has two preprocessing paths:
      • O3b GPU path (default): stack numpy → torch GPU resize + rescale, no PIL
        or HF processor on the hot path. Cuts preprocess_wait ~36 ms/frame → ~3 ms.
      • Legacy CPU path: HF processor with PIL. Used as fallback when
        RTDETR_GPU_PREPROCESS=0 or processor config doesn't match expectations.

    O1 (debug): logs cumulative time spent in each sub-stage. Toggle via env
    var RTDETR_PROFILE=1.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    model = get_model("rtdetr")
    processor = get_model("rtdetr_processor")
    if model is None or processor is None:
        return None

    person_ids: set = get_model("rtdetr_person_ids") or {0, 1}
    device = _get_device()
    dtype = torch.float16
    profile_on = os.environ.get("RTDETR_PROFILE", "0") == "1"
    use_gpu_pre = os.environ.get("RTDETR_GPU_PREPROCESS", "1") == "1"

    # O3b — read processor config once. If the processor config differs from
    # what _preprocess_gpu can replicate (e.g. it wants padding or BGR), fall
    # back to the legacy CPU path silently.
    tgt_h = processor.size.get("height") if isinstance(processor.size, dict) else None
    tgt_w = processor.size.get("width") if isinstance(processor.size, dict) else None
    pre_compatible = (
        use_gpu_pre
        and tgt_h is not None and tgt_w is not None
        and bool(getattr(processor, "do_resize", True))
        and not bool(getattr(processor, "do_pad", False))
        and bool(getattr(processor, "do_rescale", True))
    )
    if not pre_compatible and use_gpu_pre:
        logger.warning(
            "[rtdetr] GPU preprocess incompatible with processor config "
            "(size=%s do_resize=%s do_pad=%s do_rescale=%s) — falling back to CPU path",
            getattr(processor, "size", None),
            getattr(processor, "do_resize", None),
            getattr(processor, "do_pad", None),
            getattr(processor, "do_rescale", None),
        )

    # Pre-stage normalize tensors on GPU (only used when do_normalize=True).
    do_normalize = bool(getattr(processor, "do_normalize", False))
    rescale_factor = float(getattr(processor, "rescale_factor", 1.0 / 255.0))
    if do_normalize:
        _mean_t = torch.tensor(processor.image_mean, device=device,
                               dtype=dtype).view(1, 3, 1, 1)
        _std_t = torch.tensor(processor.image_std, device=device,
                              dtype=dtype).view(1, 3, 1, 1)
    else:
        _mean_t = _std_t = None

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    # O1 — cumulative timers (seconds)
    t_pre_wall = 0.0
    t_h2d = 0.0
    t_fwd = 0.0
    t_post = 0.0
    n_pre_batches = 0

    def _preprocess_cpu(batch: list) -> tuple:
        """Legacy CPU path — exact HF processor behavior."""
        sizes = [(f.shape[0], f.shape[1]) for f in batch]
        pil_imgs = [Image.fromarray(f[:, :, ::-1]) for f in batch]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        return inputs, sizes

    def _preprocess_gpu(batch: list) -> tuple:
        """O3b path — stack on CPU, do resize + rescale on GPU.

        Mirrors HF RTDetrImageProcessor exactly when:
          do_resize=True, do_pad=False, do_rescale=True,
          square resize, BGR input → convert to RGB.
        Reads mean/std/rescale_factor from the processor at call time so
        we follow whatever the loaded model expects.
        """
        sizes = [(f.shape[0], f.shape[1]) for f in batch]
        # CPU stack only (fast, no per-frame Python overhead)
        stacked = np.stack(batch, axis=0)                 # [B, H, W, 3] uint8 BGR
        t = torch.from_numpy(stacked).to(device, non_blocking=True)
        # BGR → RGB then HWC → CHW
        t = t[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        # uint8 → float (rescale 1/255)
        t = t.to(dtype) * rescale_factor
        # Resize to model input
        t = torch.nn.functional.interpolate(
            t, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=True,
        )
        # Optional normalize
        if do_normalize and _mean_t is not None:
            t = (t - _mean_t) / _std_t
        # Already on device & in inference dtype — return as the dict the model expects
        return {"pixel_values": t}, sizes

    _preprocess = _preprocess_gpu if pre_compatible else _preprocess_cpu

    all_dets: list[list[dict]] = []
    autocast_ctx = torch.autocast("cuda", dtype=dtype)

    def _infer_one(batch_frames: list, inputs_raw, sizes, depth: int = 0) -> list[list[dict]]:
        """Run inference on one (sub-)batch. On CUDA OOM, split in half and recurse."""
        nonlocal t_h2d, t_fwd, t_post
        try:
            if profile_on:
                torch.cuda.synchronize()
                t0 = time.time()
            # GPU preprocess already returns tensors on device/dtype. CPU
            # preprocess returns CPU tensors that still need a copy + cast.
            first_val = next(iter(inputs_raw.values()))
            if first_val.is_cuda:
                inputs = inputs_raw
            else:
                inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in inputs_raw.items()}
            if profile_on:
                torch.cuda.synchronize()
                t_h2d += time.time() - t0
                t0 = time.time()
            with torch.no_grad(), autocast_ctx:
                outputs = model(**inputs)
            if profile_on:
                torch.cuda.synchronize()
                t_fwd += time.time() - t0
                t0 = time.time()
            results = processor.post_process_object_detection(
                outputs, threshold=threshold,
                target_sizes=torch.tensor(sizes, device=device),
            )
            if profile_on:
                torch.cuda.synchronize()
                t_post += time.time() - t0
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            n = len(batch_frames)
            if depth >= 3 or n <= 4:
                logger.error("[rtdetr] OOM at batch_size=%d depth=%d, dropping: %s", n, depth, exc)
                return [[] for _ in batch_frames]
            mid = n // 2
            logger.warning("[rtdetr] OOM at batch_size=%d depth=%d, splitting to %d/%d",
                           n, depth, mid, n - mid)
            # Release any GPU tensors the failed batch was holding before
            # recursing so we don't double-allocate during the split.
            del inputs_raw
            torch.cuda.empty_cache()
            left_inputs, left_sizes = _preprocess(batch_frames[:mid])
            right_inputs, right_sizes = _preprocess(batch_frames[mid:])
            return (_infer_one(batch_frames[:mid], left_inputs, left_sizes, depth + 1) +
                    _infer_one(batch_frames[mid:], right_inputs, right_sizes, depth + 1))
        except Exception as exc:
            logger.warning("[rtdetr] batch failed (non-OOM): %s", exc)
            return [[] for _ in batch_frames]

        out: list[list[dict]] = []
        for res in results:
            dets = []
            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                if label.item() not in person_ids:
                    continue
                x1, y1, x2, y2 = box.tolist()
                dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                             "score": float(score), "label": "person"})
            out.append(dets)
        return out

    t_total = time.time() if profile_on else 0.0

    with ThreadPoolExecutor(max_workers=2) as ex:
        # Submit first batch preprocessing
        futures = [ex.submit(_preprocess, b) for b in batches[:2]]

        for idx, batch in enumerate(batches):
            # Prefetch next+1 batch while current is on GPU
            if idx + 2 < len(batches):
                futures.append(ex.submit(_preprocess, batches[idx + 2]))

            if profile_on:
                t0 = time.time()
            inputs_raw, sizes = futures[idx].result()
            if profile_on:
                t_pre_wall += time.time() - t0
                n_pre_batches += 1
            all_dets.extend(_infer_one(batch, inputs_raw, sizes))

    if profile_on:
        t_total = time.time() - t_total
        n_frames = sum(len(b) for b in batches)
        logger.warning(
            "[rtdetr-profile] batches=%d  frames=%d  batch_size=%d  total=%.1fs (%.1fms/frame)  "
            "preprocess_wait=%.1fs (%.1fms/frame)  h2d=%.1fs (%.1fms/frame)  "
            "forward=%.1fs (%.1fms/frame)  post=%.1fs (%.1fms/frame)",
            len(batches), n_frames, batch_size, t_total, 1000 * t_total / max(n_frames, 1),
            t_pre_wall, 1000 * t_pre_wall / max(n_frames, 1),
            t_h2d, 1000 * t_h2d / max(n_frames, 1),
            t_fwd, 1000 * t_fwd / max(n_frames, 1),
            t_post, 1000 * t_post / max(n_frames, 1),
        )

    return all_dets


_NMS_IOU_THRESH = _get_env_float("PERSON_NMS_IOU_THRESH", 0.65)


def _sanitize_dets_inplace(
    dets_per_frame: list[list[dict]],
    frames: list[np.ndarray],
    # Balanced defaults for AICity-style demo data. min_w=14 still rejects
    # detector-noise slivers while keeping narrow side-view profiles; min_h=18
    # rejects truly tiny boxes that ReID can't handle. Should be moved to a
    # per-camera config once we have hospital-scene calibration data.
    min_w: int = 12,
    min_h: int = 16,
) -> list[list[dict]]:
    """Clip bboxes into frame bounds, drop bboxes smaller than min_w/min_h, then
    apply class-agnostic NMS at PERSON_NMS_IOU_THRESH (default 0.55) to suppress
    duplicate person boxes that survived the detector's internal NMS.

    Conservative: NMS threshold is relatively high (0.55) so that genuinely
    adjacent persons are kept, only truly overlapping duplicates are removed.
    Per-frame numpy/torch vectorization keeps overhead negligible.
    """
    try:
        from torchvision.ops import nms as _tv_nms
    except Exception:
        _tv_nms = None

    out: list[list[dict]] = []
    for dets, frame in zip(dets_per_frame, frames):
        if not dets:
            out.append(dets)
            continue
        H, W = frame.shape[:2]
        arr = np.asarray([d["bbox"] for d in dets], dtype=np.float32)        # [N, 4]
        arr[:, 0] = np.clip(arr[:, 0], 0.0, W - 1)
        arr[:, 2] = np.clip(arr[:, 2], 0.0, W)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, H - 1)
        arr[:, 3] = np.clip(arr[:, 3], 0.0, H)
        keep_size = ((arr[:, 2] - arr[:, 0]) >= min_w) & ((arr[:, 3] - arr[:, 1]) >= min_h)
        if not bool(keep_size.all()):
            dets = [d for d, k in zip(dets, keep_size) if k]
            arr  = arr[keep_size]
            if not dets:
                out.append(dets)
                continue
        for d, row in zip(dets, arr):
            d["bbox"] = row.tolist()

        # Class-agnostic NMS on person boxes (priority = score).
        if _tv_nms is not None and len(dets) > 1:
            try:
                boxes_t  = torch.from_numpy(arr)
                scores_t = torch.tensor(
                    [float(d.get("score", 0.0)) for d in dets], dtype=torch.float32,
                )
                keep_idx = _tv_nms(boxes_t, scores_t, _NMS_IOU_THRESH).tolist()
                if len(keep_idx) < len(dets):
                    keep_idx_sorted = sorted(keep_idx)
                    dets = [dets[i] for i in keep_idx_sorted]
            except Exception:
                pass
        out.append(dets)
    return out


def _detect_persons_batch(frames: list[np.ndarray], threshold: float = 0.22) -> list[list[dict]]:
    """Person detection: RT-DETR primary (fast), GDINO fallback."""
    rtdetr_threshold = float(os.getenv("RTDETR_PERSON_THRESHOLD", str(min(threshold, 0.22))))
    rtdetr_result = _detect_persons_rtdetr(frames, threshold=rtdetr_threshold)
    if rtdetr_result is not None:
        rtdetr_result = _sanitize_dets_inplace(rtdetr_result, frames)
        n_dets = sum(len(d) for d in rtdetr_result)
        logger.debug("[detect] RT-DETR: %d frames → %d detections", len(frames), n_dets)
        return rtdetr_result

    # GDINO fallback
    model = get_model("gdino16")
    processor = get_model("gdino16_processor")
    if model is None or processor is None:
        logger.error(
            "[detect] RT-DETR unavailable AND GDINO not loaded — "
            "returning 0 detections for %d frames. "
            "Check GPU OOM or model warmup logs.",
            len(frames),
        )
        return [[] for _ in frames]

    device = _get_device()
    dtype = torch.float16
    autocast_ctx = torch.autocast("cuda", dtype=dtype)

    def _preprocess(batch_frames: list[np.ndarray]):
        pil_imgs = []
        sizes = []
        for f in batch_frames:
            h, w = f.shape[:2]
            sizes.append((h, w))
            pil_imgs.append(Image.fromarray(f[:, :, ::-1]))
        texts = ["person."] * len(pil_imgs)
        inputs = processor(images=pil_imgs, text=texts, return_tensors="pt", padding=True)
        return inputs, sizes

    BATCH_SIZE = 64
    batches = [frames[i: i + BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]
    if not batches:
        return []

    all_dets: list[list[dict]] = []
    prefetch_exec = ThreadPoolExecutor(max_workers=1)

    prefetch_fut = prefetch_exec.submit(_preprocess, batches[0])
    try:
        for b_idx, batch in enumerate(batches):
            inputs_cpu, sizes = prefetch_fut.result()
            if b_idx + 1 < len(batches):
                prefetch_fut = prefetch_exec.submit(_preprocess, batches[b_idx + 1])

            try:
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs_cpu.items()}
                target_sizes = torch.tensor(sizes, dtype=torch.int64).to(device)
                with torch.no_grad(), autocast_ctx:
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    box_threshold=threshold, text_threshold=threshold,
                    target_sizes=target_sizes,
                )
            except Exception:
                results = None

            if results is None:
                for f in batch:
                    all_dets.append(_detect_persons(f, threshold))
                continue

            for res in results:
                dets = []
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    if float(score) < threshold or label.lower() != "person":
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                 "score": float(score), "label": label})
                all_dets.append(dets)
    finally:
        prefetch_exec.shutdown(wait=False)

    return _sanitize_dets_inplace(all_dets, frames)


# ---------------------------------------------------------------------------
# Stage 3: BEV Projection
# ---------------------------------------------------------------------------

def _project_to_bev_single(
    detections: list[dict],
    camera_id: str,
    cal_path: str | None,
) -> list[dict]:
    """Project 2D bboxes to BEV for a single camera."""
    from shared.core.geometry import BEVProjector

    try:
        projector = BEVProjector(camera_id, cal_path)
    except Exception as exc:
        logger.warning("BEVProjector failed for %s: %s", camera_id, exc)
        for d in detections:
            d["bev_x"] = 0.0
            d["bev_y"] = 0.0
        return detections

    for det in detections:
        bbox = det["bbox"]
        bev_x, bev_y = projector.bbox_bottom_center_to_bev(*bbox)
        det["bev_x"] = bev_x
        det["bev_y"] = bev_y
        det["camera_id"] = camera_id

    return detections


# ---------------------------------------------------------------------------
# Per-video pipeline (stages 1-3: load → detect → BEV)
# ---------------------------------------------------------------------------

def _process_single_video(entry: BatchVideoEntry) -> dict:
    """Run stages 1-3 for one video: load frames, detect, project to BEV."""
    video_path = entry.video_path
    camera_id = entry.camera_id or f"cam_{entry.video_id.split('_')[0]}"

    try:
        frames, fps = _video_to_frames(video_path, max_frames=500)
    except Exception as exc:
        logger.warning("Failed to load video %s: %s", video_path, exc)
        return {
            "video_id": entry.video_id,
            "camera_id": camera_id,
            "detections": [],
            "error": str(exc),
        }

    if not frames:
        return {
            "video_id": entry.video_id,
            "camera_id": camera_id,
            "detections": [],
            "error": "No frames extracted",
        }

    sampled = _sample_frames_uniform(frames, fps, entry.sample_interval)
    all_detections: list[dict] = []

    sampled_imgs = [frame for _, frame in sampled]
    batch_dets = _detect_persons_batch(sampled_imgs, threshold=0.22)
    for (frame_idx, _frame), dets in zip(sampled, batch_dets):
        for det in dets:
            det["frame_idx"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
            det["video_id"] = entry.video_id
        all_detections.extend(dets)

    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    all_detections = _project_to_bev_single(all_detections, camera_id, cal_path)

    return {
        "video_id": entry.video_id,
        "camera_id": camera_id,
        "detections": all_detections,
        "fps": fps,
        "n_frames": len(frames),
    }


# ---------------------------------------------------------------------------
# Stage 4: MCBLT Cross-Camera Association (ONE call across all cameras)
# ---------------------------------------------------------------------------

def _mcblt_associate(
    detections_by_camera: dict[str, list[dict]],
    max_dist: float = 1.5,
) -> list[list[dict]]:
    """MCBLT cross-camera association using Hungarian matching."""
    from shared.core.mcblt import associate_cross_camera

    return associate_cross_camera(detections_by_camera, max_dist_meters=max_dist)


# ---------------------------------------------------------------------------
# Stage 5: DINOv2 Appearance Embedding
# ---------------------------------------------------------------------------

def _generate_dinov2_embeddings(
    frames: list[np.ndarray],
    bboxes: list[list[float]],
    tracklet_id: str,
) -> Optional[list[float]]:
    """Generate DINOv2 ViT-L/14 appearance embeddings (1024-dim)."""
    model = get_model("dinov2")
    processor = get_model("dinov2_processor")
    if model is None or processor is None:
        return None

    device = _get_device()
    dtype = torch.float16

    if not frames or not bboxes:
        return None

    n = min(5, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    pil_crops = []
    for idx in indices:
        frame = frames[idx]
        bbox = bboxes[min(idx, len(bboxes) - 1)]
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        max_dim = max(crop_h, crop_w)
        top = (max_dim - crop_h) // 2
        bottom = max_dim - crop_h - top
        left = (max_dim - crop_w) // 2
        right = max_dim - crop_w - left
        square = cv2.copyMakeBorder(
            crop, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        resized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
        pil_crops.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    if not pil_crops:
        return None

    with torch.no_grad():
        inputs = processor(images=pil_crops, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        feats = model(**inputs).pooler_output.float()  # [N, 1024] — stays on GPU

    avg = feats.mean(0)
    norm = avg.norm()
    if norm > 0:
        avg = avg / norm
    return avg.cpu().tolist()


# ---------------------------------------------------------------------------
# Stage 6: Legacy SigLIP 2 label maps (kept for reference / _legacy_ function only)
# ---------------------------------------------------------------------------

_AGE_RANGE_MAP: dict[str, str] = {
    "child person":       "child",
    "teenage person":     "teenager",
    "young adult person": "young_adult",
    "middle-aged person": "middle_aged",
    "elderly person":     "elderly",
}
_BAG_TYPE_MAP: dict[str, str] = {
    "person with backpack":     "backpack",
    "person with handbag":      "handbag",
    "person with shoulder bag": "shoulder_bag",
    "person with suitcase":     "suitcase",
    "person without bag":       "none",
}
_HAT_COLOR_MAP: dict[str, str] = {
    "person with red hat":    "red",
    "person with blue hat":   "blue",
    "person with black hat":  "black",
    "person with white hat":  "white",
    "person with gray hat":   "gray",
    "person with yellow hat": "yellow",
    "person without hat":     "none",
}
_HAIR_STYLE_MAP: dict[str, str] = {
    "person with short hair": "short",
    "person with long hair":  "long",
    "person with ponytail":   "ponytail",
    "person with tied hair":  "tied",
    "bald person":            "bald",
}
_HAIR_COLOR_MAP: dict[str, str] = {
    "person with black hair":  "black",
    "person with brown hair":  "brown",
    "person with blonde hair": "blonde",
    "person with gray hair":   "gray",
    "person with white hair":  "white",
}

def _legacy_run_siglip2_label_attributes(
    frames: list[np.ndarray],
    bbox: list[float],
) -> dict[str, Any]:
    """Legacy label-based SigLIP 2 attribute tagging — kept for debug/fallback only.
    Main pipeline uses _caption_crop_vlm() instead."""
    model = get_model("siglip2")
    processor = get_model("siglip2_processor")
    if model is None or processor is None:
        return _default_attributes()

    device = _get_device()
    dtype = torch.float16

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frames[0].shape[:2] if frames else (1080, 1920)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frames[0][y1:y2, x1:x2] if frames else np.zeros((1, 1, 3), dtype=np.uint8)
    if crop.size == 0:
        return _default_attributes()

    crop_h, crop_w = crop.shape[:2]
    max_dim = max(crop_h, crop_w)
    top = (max_dim - crop_h) // 2
    bottom = max_dim - crop_h - top
    left = (max_dim - crop_w) // 2
    right = max_dim - crop_w - left
    square = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    resized = cv2.resize(square, (384, 384), interpolation=cv2.INTER_LINEAR)
    pil_crop = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    label_groups = {
        "upper_color": [
            "red shirt", "blue shirt", "green shirt", "white shirt", "black shirt",
            "yellow shirt", "orange shirt", "purple shirt", "gray shirt", "brown shirt",
        ],
        "lower_color": [
            "black pants", "blue jeans", "gray pants", "white pants", "brown pants",
            "black shorts", "gray shorts",
        ],
        "gender":         ["man", "woman"],
        "bag_presence":   ["person carrying a bag", "person not carrying a bag"],
        "hat_presence":   ["person wearing a hat", "person not wearing a hat"],
        "age_range":      list(_AGE_RANGE_MAP.keys()),
        "hat_color":      list(_HAT_COLOR_MAP.keys()),
        "bag_type":       list(_BAG_TYPE_MAP.keys()),
        "mask_presence":  ["person wearing face mask", "person not wearing face mask"],
        "hair_style":     list(_HAIR_STYLE_MAP.keys()),
        "hair_color":     list(_HAIR_COLOR_MAP.keys()),
    }

    attributes: dict[str, str] = {}
    for attr_type, labels in label_groups.items():
        try:
            inputs = processor(
                text=labels, images=pil_crop,
                return_tensors="pt", padding=True
            )
            siglip_dtype = torch.float16
            inputs = {k: v.to(device, dtype=siglip_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            siglip_ctx = torch.autocast("cuda", dtype=siglip_dtype)
            with torch.no_grad(), siglip_ctx:
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image).squeeze()

            best_idx = int(probs.argmax())
            best_label = labels[best_idx]
            best_prob = float(probs[best_idx])

            if attr_type == "upper_color":
                attributes["upper_color"] = best_label.split()[0]
            elif attr_type == "lower_color":
                attributes["lower_color"] = best_label.split()[0]
            elif attr_type == "gender":
                attributes["gender"] = best_label.split()[0] if best_prob > 0.6 else "unknown"
            elif attr_type == "bag_presence":
                attributes["bag_presence"] = "yes" if "not" not in best_label else "no"
            elif attr_type == "hat_presence":
                attributes["hat_presence"] = "yes" if "not" not in best_label else "no"
            elif attr_type == "age_range":
                attributes["age_range"] = _AGE_RANGE_MAP.get(best_label, "unknown")
            elif attr_type == "hat_color":
                attributes["hat_color"] = _HAT_COLOR_MAP.get(best_label, "unknown")
            elif attr_type == "bag_type":
                attributes["bag_type"] = _BAG_TYPE_MAP.get(best_label, "unknown")
            elif attr_type == "mask_presence":
                attributes["mask_presence"] = "yes" if best_label == "person wearing face mask" else "no"
            elif attr_type == "hair_style":
                attributes["hair_style"] = _HAIR_STYLE_MAP.get(best_label, "unknown")
            elif attr_type == "hair_color":
                attributes["hair_color"] = _HAIR_COLOR_MAP.get(best_label, "unknown")

        except Exception as exc:
            logger.warning("SigLIP2 attribute failed for %s: %s", attr_type, exc)

    for key in ["upper_color", "lower_color", "gender", "bag_presence", "hat_presence",
                "age_range", "hat_color", "bag_type", "mask_presence", "hair_style", "hair_color"]:
        if key not in attributes:
            attributes[key] = "unknown"

    return attributes


# ---------------------------------------------------------------------------
# Stage 7: VideoMAE V2 Action Classification
# ---------------------------------------------------------------------------

# Mapping from Kinetics-400 labels → TraceX simplified action taxonomy.
# Used by VideoMAE Kinetics fine-tuned model (400 classes) → 4TraceX classes.
_KINETICS_TO_SIMPLIFIED: dict[str, str] = {
    # Standing / static
    "standing": [
        "looking at person", "shaking hands", "applauding", "brushing teeth",
        "combing hair", "dancing", "fidgeting", "headbutting", "head massage",
        "holding baby", "hugging person", "kissing", "laughing", "looking at phone",
        "marching", "parade", "playing harmonica", "playing organ", "playing piano",
        "playing recorder", "playing violin", "playing accordion", "playing guitar",
        "playing drums", "playing cello", "singing", "tapping pen", "texting",
        "waving", "whistling", "wrestling", "yawning",
    ],
    # Walking
    "walking": [
        "walking the dog", "walking on stilts", "crossing street",
        "drumming fingers", "golf putting", "hula hooping", "juggling balls",
        "kicking soccer ball", "massaging back", "massaging feet", "massaging legs",
        "moving car", "moving trolley", "pushing car", "pushing cart",
        "pushing wheelchair", "shuffling cards", "sled dog racing",
        "sneaking", "snowkiting", "snowmobiling", "somersaulting",
        "speed walking", "strumming guitar", "surfing crowd", "tai chi",
        "tapping guitar", "tasting beer", "tasting food", "tasting wine",
        "throwing ball", "throwing discus", "throwing axe",
        "tickling", "tobogganing", "tossing salad", "towel snapping",
        "trapeze", "unboxing", "vault", "waiting in line", "walking on beam",
    ],
    # Running
    "running": [
        "running on treadmill", "sprinting", "jogging", "dribbling",
        "basketball", "burpee", "cartwheeling", "catching baseball",
        "catching cricket ball", "catching/disc throwing", "celebrating",
        "chopping wood", "climbing", "climbing rope", "climbing tree",
        "contact juggling", "crawling", "cricket batting", "croquet",
        "cutting pineapple", "diving", "dodgeball", "doing capoeira",
        "doing jigsaw puzzle", "dribbling basketball", "drop kicking",
        "exercising arm", "exercising with exercise ball", "faceplanting",
        "falling off bike", "falling off chair", "fencing", "flying disc",
        "freediving", "front raises", "golf driving", "hammering",
        "hand car wash", "hand washing", "headstands", "high kick",
        "hitball", "hitball with rake", "hockey", "horse race",
    ],
    # Sitting
    "sitting": [
        "sitting", "sitting on bed", "sitting on chair", "sitting on floor",
        "sitting on stairs", "sitting with something", "lying", "lying down",
        "sleeping", "taking a shower", "using computer", "typing",
        "writing", "reading", "drinking coffee", "drinking beer",
        "eating", "eating burger", "eating cake", "eating carrots",
        "eating chips", "eating corn", "eating doughnuts", "eating grapes",
        "eating hotdog", "eating ice cream", "eating spaghetti",
    ],
    "bending": [
        "bending back", "bending metal", "bowling", "clean and press",
        "clean and jerk", "cleaning floor", "cleaning gutters",
        "crouching", "curling (exercise)", "cutting nails",
        "cutting paper", "deadlifting", "digging", "dunking basketball",
    ],
    "carrying": [
        "carrying baby", "carrying cradles", "carrying rifle",
        "carrying something", "carrying water", "catching something",
    ],
    "pushing_pulling": [
        "pulling car", "pulling cart", "pulling rope", "pushing car",
        "pushing cart", "pushing wheelchair", "shovelling",
        "sweeping", "washing dishes", "washing windows",
    ],
    "sports": [
        "archery", "backflip", "badminton", "baseball batting",
        "basketball shooting", "bench pressing", "biking on trail",
        "billiards", "blowdrying hair", "blowing glass", "blowing leaves",
        "bobsledding", "body flying", "bodyweight stretching",
        "bouncing basketball", "bouncing on bouncy castle",
        "bouncing on trampoline", "breakdancing", "bungee jumping",
        "camel riding", "canoeing", "capoeira", "carrying baby",
        "catching/throwing baseball", "catching/flying disc",
        "chest press machine", "cheerleading", "chestfeeding",
        "chinning", "choirs", "chopping vegetables", "clapping",
        "climbing ladder", "climbing tree", "climbing wall",
        "counting money", "couple dancing", "cracking back",
        "cracking knuckles", "cricket bowling", "curling hair",
    ],
}

# Flatten mapping: kinetics_label → tracex_action
_KINETICS_MAP: dict[str, str] = {}
for tracex_action, kinetics_list in _KINETICS_TO_SIMPLIFIED.items():
    for k_label in kinetics_list:
        _KINETICS_MAP[k_label.lower()] = tracex_action

# Fallback: if no match, guess from logits distribution
_SIMPLIFIED_ACTIONS = ["standing", "walking", "running", "sitting", "bending", "carrying", "pushing_pulling", "sports"]


def _map_kinetics_to_tracex_action(kinetics_label: str) -> str:
    """Map a Kinetics-400 label to TraceX simplified action."""
    return _KINETICS_MAP.get(kinetics_label.lower(), "standing")


def _run_videomae_actions(
    frames: list[np.ndarray],
    bbox: list[float],
) -> str:
    """VideoMAE V2 action classification from tracklet frames.

    Uses VideoMAE fine-tuned on Kinetics-400 (400 classes).
    Maps Kinetics labels → TraceX simplified taxonomy:
      standing, walking, running, sitting, bending,
      carrying, pushing_pulling, sports
    """
    model = get_model("videomae")
    processor = get_model("videomae_processor")
    if model is None or processor is None:
        return "standing"

    device = _get_device()
    dtype = torch.float16

    if len(frames) < 2:
        return "standing"

    n = 16
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    crops = []
    for idx in indices:
        frame = frames[idx]
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops.append(resized)

    try:
        inputs = processor(crops, return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        vmae_ctx = torch.autocast("cuda", dtype=dtype)
        with torch.no_grad(), vmae_ctx:
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5, dim=-1)

        # Try to map top predictions to TraceX actions
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            for prob, idx in zip(top_probs[0].cpu(), top_indices[0].cpu()):
                kinetics_label = id2label.get(int(idx), "")
                tracex_action = _map_kinetics_to_tracex_action(kinetics_label)
                if tracex_action not in ("sports",):
                    return tracex_action
            return "sports"  # default fallback
        else:
            # No label mapping available — return from simplified
            return "standing"
    except Exception as exc:
        logger.warning("VideoMAE action failed: %s", exc)
        return "standing"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_attributes() -> dict[str, Any]:
    """All-unknown attrs used when VLM fails / falls back. Confidences are None,
    not 0.0 — None means 'not yet extracted', 0.0 means 'extracted, very unsure'."""
    return {
        "gender": "unknown",          "gender_conf": None,
        "age_range": "unknown",       "age_range_conf": None,
        "upper_color": "unknown",     "upper_type": "unknown",
        "upper_desc": None,           "upper_conf": None, "upper_desc_conf": None,
        "lower_color": "unknown",     "lower_type": "unknown",
        "lower_desc": None,           "lower_conf": None, "lower_desc_conf": None,
        "shoes_color": "unknown",     "shoes_type": "unknown",
        "shoes_desc": None,           "shoes_conf": None, "shoes_desc_conf": None,
        "bag_presence": "unknown",    "bag_type": "unknown",
        "bag_desc": None,             "bag_conf": None,   "bag_desc_conf": None,
        "hat_presence": "unknown",    "hat_color": "unknown", "hat_type": "unknown",
        "hat_desc": None,             "hat_conf": None,   "hat_desc_conf": None,
        "mask_presence": "unknown",   "mask_conf": None,
        "hair_style": "unknown",      "hair_style_conf": None,
        "hair_color": "unknown",      "hair_color_conf": None,
        "appearance_summary": "person", "appearance_summary_conf": None,
    }


def _build_tracklet_result(
    *,
    tracklet_id: str,
    video_id: str,
    camera_id: str,
    track_idx: int,
    start_time: float,
    end_time: float,
    quality_score: float,
    attrs: dict,
    summary: str,
    rep_bbox: Any,
    bev_x: float,
    bev_y: float,
    siglip_embedding: list[float],
    action: str,
    action_confidence: float,
    kinetics_label: str,
    crop_url: str,
    observations: list[dict] | None = None,
) -> "TrackletResult":
    """Single source-of-truth for building a TrackletResult from a parsed attrs dict.

    Used by all 3 pipeline entry points (sync, single-video legacy, cross-camera
    batch) so a schema change only touches this one helper.
    """
    return TrackletResult(
        tracklet_id=tracklet_id,
        video_id=video_id,
        camera_id=camera_id,
        track_id=str(track_idx),
        start_time=float(start_time or 0.0),
        end_time=float(end_time or 0.0),
        quality_score=float(quality_score or 0.0),

        gender=attrs.get("gender", "unknown"),
        gender_conf=attrs.get("gender_conf"),
        age_range=attrs.get("age_range", "unknown"),
        age_range_conf=attrs.get("age_range_conf"),

        upper_color=attrs.get("upper_color"),
        upper_type=attrs.get("upper_type"),
        upper_desc=attrs.get("upper_desc"),
        upper_conf=attrs.get("upper_conf"),
        upper_desc_conf=attrs.get("upper_desc_conf"),

        lower_color=attrs.get("lower_color"),
        lower_type=attrs.get("lower_type"),
        lower_desc=attrs.get("lower_desc"),
        lower_conf=attrs.get("lower_conf"),
        lower_desc_conf=attrs.get("lower_desc_conf"),

        shoes_color=attrs.get("shoes_color"),
        shoes_type=attrs.get("shoes_type"),
        shoes_desc=attrs.get("shoes_desc"),
        shoes_conf=attrs.get("shoes_conf"),
        shoes_desc_conf=attrs.get("shoes_desc_conf"),

        bag_presence=attrs.get("bag_presence"),
        bag_type=attrs.get("bag_type"),
        bag_desc=attrs.get("bag_desc"),
        bag_conf=attrs.get("bag_conf"),
        bag_desc_conf=attrs.get("bag_desc_conf"),

        hat_presence=attrs.get("hat_presence"),
        hat_color=attrs.get("hat_color"),
        hat_type=attrs.get("hat_type"),
        hat_desc=attrs.get("hat_desc"),
        hat_conf=attrs.get("hat_conf"),
        hat_desc_conf=attrs.get("hat_desc_conf"),

        mask_presence=attrs.get("mask_presence", "unknown"),
        mask_conf=attrs.get("mask_conf"),
        hair_style=attrs.get("hair_style", "unknown"),
        hair_style_conf=attrs.get("hair_style_conf"),
        hair_color=attrs.get("hair_color", "unknown"),
        hair_color_conf=attrs.get("hair_color_conf"),

        appearance_summary=summary,
        appearance_summary_conf=attrs.get("appearance_summary_conf"),

        representative_bbox=[int(x) for x in (rep_bbox or [0, 0, 0, 0])],
        bev_x=float(bev_x or 0.0),
        bev_y=float(bev_y or 0.0),
        crop_url=crop_url,
        siglip_embedding=siglip_embedding or [],

        action=action,
        action_confidence=float(action_confidence or 0.0),
        kinetics_label=kinetics_label,

        observations=observations or [],
    )


# Qwen no longer self-reports confidence — confidence is computed from the
# token-level logprobs of the generated value spans (geometric mean of
# P(token | context)). The prompt requests only the values themselves.
_VLM_PROMPT = """You are analyzing a person crop from a surveillance camera.

**Critical rules:**
- Describe visible appearance from the crop, and make limited attribute inferences ONLY from visible cues inside the crop.
- Use "unknown" only when visible cues are insufficient, occluded, cut off by the bbox, blurry, or not present (or "no" for *_presence fields when clearly absent).
- Do NOT infer from scene/camera/location context, surrounding people, or assumptions outside the crop.

**Field formats (when visible):**
- gender: "man" or "woman" when visible cues in the crop support the inference; otherwise "unknown"
- age_range: "child | teenager | young_adult | middle_aged | elderly" estimated only from visible body/face/hair/posture/clothing cues in the crop; otherwise "unknown"
- *_color: a single concrete color word in English, lowercase (dominant color if multiple); use "unknown" only when the item is not visible enough
- upper_type: garment noun for the upper body (e.g. "shirt", "t-shirt", "jacket", "hoodie", "sweater", "blouse", "dress", "robe", "gown", "ao_dai", "jumpsuit")
- lower_type: garment noun for the legs (e.g. "pants", "jeans", "shorts", "skirt"). For one-piece outfits see ONE-PIECE rule below.
- shoes_type: footwear noun (e.g. "sneakers", "boots", "sandals", "heels", "slippers")
- bag_type: bag noun (e.g. "backpack", "handbag", "shoulder_bag", "suitcase", "tote", "none")
- hat_type: head-covering noun — see HEAD COVERING rule below
- *_desc: 2-3 word literal description of what is visible
- *_presence (bag/hat/mask): "yes" if clearly visible, "no" if clearly not present, "unknown" if uncertain
- hair_style: "short | long | ponytail | bald | bun | covered" (use "covered" when hat/scarf hides hair)
- hair_color: concrete color word, "unknown" if hair not visible
- appearance_summary: one short sentence stating only directly visible features

**ONE-PIECE OUTFITS (dress, robe, hospital gown, jumpsuit, áo dài, overall):**
When the person wears a single garment covering both upper and lower body:
- upper_type = the one-piece kind ("dress", "robe", "gown", "jumpsuit", "overall", "ao_dai")
- upper_color = the garment's dominant color
- upper_desc = short description (e.g. "long red dress", "white hospital gown")
- lower_type = repeat the SAME one-piece kind as upper_type
- lower_color = same color as upper_color
- lower_desc = "" (empty string — do not duplicate upper_desc)

**HEAD COVERING (hat_*):**
hat_presence = "yes" for ANY head covering visible. Choose hat_type from what you see:
- rigid hats → "cap", "beanie", "hat", "helmet"
- hood attached to a garment → "hood"
- cloth wrapped around the head → "headscarf" (covers hijab, turban, head wrap, religious veil, cloth tied around hair)
- If the head covering extends down to cover shoulders or part of the torso, note it in hat_desc (e.g. "headscarf covering shoulders"). Still describe the visible part of the inner clothing in upper_*.

**LAYERED CLOTHING:**
If there is an outer layer (coat, cardigan, shawl, cape, vest) over the inner top:
- upper_type / upper_color describe the OUTERMOST visible layer.
- upper_desc may mention the inner layer if visible (e.g. "black coat over white shirt").

Return ONLY a JSON object with EXACTLY these fields and no others:

{
  "gender": "...",
  "age_range": "...",
  "upper_color": "...", "upper_type": "...", "upper_desc": "...",
  "lower_color": "...", "lower_type": "...", "lower_desc": "...",
  "shoes_color": "...", "shoes_type": "...", "shoes_desc": "...",
  "bag_presence": "...", "bag_type": "...", "bag_desc": "...",
  "hat_presence": "...", "hat_color": "...", "hat_type": "...", "hat_desc": "...",
  "mask_presence": "...",
  "hair_style": "...", "hair_color": "...",
  "appearance_summary": "..."
}

Do NOT include any confidence fields — the system computes confidence from
your token logits, not from your self-report.
Return only the JSON object, no surrounding text."""


_GENDER_NORM   = {"male": "man", "man": "man", "female": "woman", "woman": "woman"}
_AGE_NORM      = {
    "young adult": "young_adult", "young_adult": "young_adult",
    "middle aged": "middle_aged", "middle-aged": "middle_aged", "middle_aged": "middle_aged",
    "teen": "teenager", "teenager": "teenager",
    "elder": "elderly", "elderly": "elderly",
    "child": "child",
}
_PRESENCE_NORM = {"yes": "yes", "no": "no", "true": "yes", "false": "no", "none": "no"}

# Mapping of VLM JSON key → (canonical attr key, conf key).
# Used by _parse_vlm_attrs + _attach_logit_confs to keep everything in sync.
_ATTR_CONF_MAP: list[tuple[str, str, str | None]] = [
    # (vlm_key, conf_key, normaliser)
    ("gender",             "gender_conf",              "GENDER"),
    ("age_range",          "age_range_conf",           "AGE"),
    ("upper_color",        "upper_conf",               None),
    ("upper_type",         "upper_conf",               None),    # shares with upper_color
    ("upper_desc",         "upper_desc_conf",          None),
    ("lower_color",        "lower_conf",               None),
    ("lower_type",         "lower_conf",               None),
    ("lower_desc",         "lower_desc_conf",          None),
    ("shoes_color",        "shoes_conf",               None),
    ("shoes_type",         "shoes_conf",               None),
    ("shoes_desc",         "shoes_desc_conf",          None),
    ("bag_presence",       "bag_conf",                 "PRESENCE"),
    ("bag_type",           "bag_conf",                 None),
    ("bag_desc",           "bag_desc_conf",            None),
    ("hat_presence",       "hat_conf",                 "PRESENCE"),
    ("hat_color",          "hat_conf",                 None),
    ("hat_type",           "hat_conf",                 None),
    ("hat_desc",           "hat_desc_conf",            None),
    ("mask_presence",      "mask_conf",                "PRESENCE"),
    ("hair_style",         "hair_style_conf",          None),
    ("hair_color",         "hair_color_conf",          None),
    ("appearance_summary", "appearance_summary_conf",  None),
]


def _parse_vlm_attrs(parsed: dict) -> dict:
    """Normalise a raw VLM JSON dict into the canonical attrs dict.

    Confidence fields are set to None here. They are filled in afterwards by
    _attach_logit_confs() using token-level logprobs from generate().
    """
    def _s(key: str, fallback: str = "unknown") -> str:
        v = parsed.get(key)
        return str(v).strip().lower() if v not in (None, "", "null") else fallback

    def _norm(val: str, mapping: dict) -> str:
        return mapping.get(val.lower().strip(), val) if val else "unknown"

    return {
        "gender":           _norm(_s("gender"), _GENDER_NORM),       "gender_conf": None,
        "age_range":        _norm(_s("age_range"), _AGE_NORM),       "age_range_conf": None,
        "upper_color":      _s("upper_color"),
        "upper_type":       _s("upper_type"),
        "upper_desc":       parsed.get("upper_desc"),
        "upper_conf":       None,                                    "upper_desc_conf": None,
        "lower_color":      _s("lower_color"),
        "lower_type":       _s("lower_type"),
        "lower_desc":       parsed.get("lower_desc"),
        "lower_conf":       None,                                    "lower_desc_conf": None,
        "shoes_color":      _s("shoes_color"),
        "shoes_type":       _s("shoes_type"),
        "shoes_desc":       parsed.get("shoes_desc"),
        "shoes_conf":       None,                                    "shoes_desc_conf": None,
        "bag_presence":     _norm(_s("bag_presence"), _PRESENCE_NORM),
        "bag_type":         _s("bag_type"),
        "bag_desc":         parsed.get("bag_desc"),
        "bag_conf":         None,                                    "bag_desc_conf": None,
        "hat_presence":     _norm(_s("hat_presence"), _PRESENCE_NORM),
        "hat_color":        _s("hat_color"),
        "hat_type":         _s("hat_type"),
        "hat_desc":         parsed.get("hat_desc"),
        "hat_conf":         None,                                    "hat_desc_conf": None,
        "mask_presence":    _norm(_s("mask_presence"), _PRESENCE_NORM),
        "mask_conf":        None,
        "hair_style":       _s("hair_style"),                        "hair_style_conf": None,
        "hair_color":       _s("hair_color"),                        "hair_color_conf": None,
        "appearance_summary":      parsed.get("appearance_summary") or "person",
        "appearance_summary_conf": None,
    }


def _compute_value_logprob_confs(
    generated_token_ids: "torch.Tensor",   # 1-D tensor of token IDs (already detached, cpu)
    scores: list,                          # list[Tensor] of per-step logits (length = len(generated_token_ids))
    tokenizer: Any,
) -> dict[str, float]:
    """Compute per-attribute confidence from token logprobs.

    Strategy:
      1. Decode the generated tokens into a string.
      2. Tokenize the string with offset_mapping (or scan token-by-token) to find
         the character span of each value token.
      3. For each `"<key>": "<value>"` pair, locate the tokens belonging to
         the value span, gather their per-token probabilities, and compute the
         geometric mean. That is the "real" model confidence for that key.

    Returns a dict mapping vlm_key (e.g. "gender", "upper_color") → conf in [0, 1].
    Keys not found in the output are simply absent from the returned dict.
    """
    import torch as _torch

    # Per-token probability of the actually-sampled token.
    # scores[i] has shape [vocab]; sampled token = generated_token_ids[i].
    token_probs: list[float] = []
    for step, score_tensor in enumerate(scores):
        if step >= len(generated_token_ids):
            break
        try:
            logp = _torch.log_softmax(score_tensor.float(), dim=-1)
            tok_id = int(generated_token_ids[step])
            token_probs.append(float(logp[..., tok_id].squeeze().item()))  # log-prob
        except Exception:
            token_probs.append(0.0)  # safe default

    # Walk the decoded string to find "key": "value" pairs and their char spans.
    decoded = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    # Build character → token index mapping by decoding each prefix.
    # This avoids reliance on offset_mapping which isn't always exposed by
    # the processor. Acceptable cost: |tokens| decodes.
    cum_char_to_tok: list[int] = []  # cum_char_to_tok[char_idx] = token index containing that char
    running = ""
    for tok_idx in range(len(generated_token_ids)):
        piece = tokenizer.decode(generated_token_ids[tok_idx:tok_idx + 1], skip_special_tokens=True)
        for _ in piece:
            cum_char_to_tok.append(tok_idx)
        running += piece
        if len(running) >= len(decoded):
            break
    # Pad if rounding caused a mismatch
    while len(cum_char_to_tok) < len(decoded):
        cum_char_to_tok.append(len(generated_token_ids) - 1)

    pattern = re.compile(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"')
    confs: dict[str, float] = {}

    for match in pattern.finditer(decoded):
        key = match.group(1)
        val_start = match.start(2)
        val_end   = match.end(2)
        if val_end <= val_start:
            continue
        tok_lo = cum_char_to_tok[val_start] if val_start < len(cum_char_to_tok) else 0
        tok_hi = cum_char_to_tok[val_end - 1] if (val_end - 1) < len(cum_char_to_tok) else tok_lo
        span = token_probs[tok_lo:tok_hi + 1]
        if not span:
            continue
        # Geometric mean of P(token) = exp(mean(log P))
        mean_logp = sum(span) / len(span)
        confs[key] = max(0.0, min(1.0, float(_torch.tensor(mean_logp).exp().item())))

    return confs



def _attach_logit_confs(attrs: dict, logit_confs: dict[str, float]) -> dict:
    """Merge logit-derived confidences into a parsed attrs dict.

    Shared-conf fields (e.g. upper_color + upper_type → upper_conf) take the
    MIN of the per-value confidences, so a shaky color doesn't get masked by
    a confident type.
    """
    shared_min: dict[str, float] = {}
    for vlm_key, conf_key, _norm in _ATTR_CONF_MAP:
        if vlm_key not in logit_confs:
            continue
        c = float(logit_confs[vlm_key])
        if conf_key in shared_min:
            shared_min[conf_key] = min(shared_min[conf_key], c)
        else:
            shared_min[conf_key] = c
    for conf_key, c in shared_min.items():
        attrs[conf_key] = c
    return attrs


def _extract_json_object(raw: str) -> dict:
    """Extract the first complete JSON object from model output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"JSON object not found: {cleaned[:200]}")

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                payload = cleaned[start:idx + 1]
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
                return parsed

    raise ValueError(f"JSON object incomplete: {cleaned[:200]}")


def _caption_crop_vlm(crop: "Image.Image") -> dict:
    """Generate open-vocabulary appearance attributes via Qwen2.5-VL-7B-Instruct (single crop).

    Returns attrs with logit-derived confidence fields filled in (None for
    values the model didn't emit / didn't match the JSON pattern).
    """
    model = get_model("qwen25vl")
    processor = get_model("qwen25vl_processor")
    if model is None or processor is None:
        return _default_attributes()

    device = _get_device()
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": _VLM_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[crop], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=VLM_BATCH_MAX_NEW_TOKENS_PER_CROP,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                return_dict_in_generate=True,
                output_scores=True,
            )
        output_ids = gen.sequences
        scores = list(gen.scores or [])
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0][input_len:]
        raw = processor.decode(gen_ids, skip_special_tokens=True).strip()

        attrs = _parse_vlm_attrs(_extract_json_object(raw))

        # Token-level confidences from the model's own logits.
        try:
            tokenizer = getattr(processor, "tokenizer", processor)
            logit_confs = _compute_value_logprob_confs(gen_ids.detach().cpu(), scores, tokenizer)
            _attach_logit_confs(attrs, logit_confs)
        except Exception as conf_exc:
            logger.warning("[vlm] logit-conf compute failed (single): %s", conf_exc)

        return attrs

    except Exception as exc:
        logger.warning("[vlm] _caption_crop_vlm failed: %s", exc)
        return _default_attributes()


def _vlm_progress_bar(done: int, total: int, width: int = 25) -> str:
    filled = int(width * done / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * done / total) if total else 0
    return f"[{bar}] {done}/{total} ({pct}%)"


def _caption_crops_vlm_batch(crops: list, batch_size: int = VLM_BATCH_SIZE, tag: str = "") -> list:
    """Adaptive Qwen2.5-VL captioning with independent per-crop prompts.

    Each batch item has its own single-image prompt and its own generated JSON
    object. This keeps GPU batching benefits without asking Qwen to reason over
    multiple people in one long JSON-array prompt, which can mix attributes
    between crops and makes parsing more fragile.
    """
    model = get_model("qwen25vl")
    processor = get_model("qwen25vl_processor")
    if model is None or processor is None:
        return [_default_attributes() for _ in crops]

    device = _get_device()
    results: list = []
    total = len(crops)
    log_every = max(1, total // 20)  # log every ~5%
    _vlm_t0 = time.perf_counter()

    start_batch_size = max(1, min(batch_size, VLM_BATCH_MAX_SIZE, total or 1))
    current_batch_size = start_batch_size
    stable_windows = 0
    i = 0

    def _log_progress(done: int, batch_used: int) -> None:
        if done == 1 or done % log_every == 0 or done == total:
            elapsed = time.perf_counter() - _vlm_t0
            eta = (elapsed / done * (total - done)) if done else 0
            logger.info(
                "[vlm] %s %s  %.0fs elapsed  ETA %.0fs  batch=%d current=%d start=%d max=%d tokens/crop=%d mode=independent",
                tag,
                _vlm_progress_bar(done, total),
                elapsed,
                eta,
                batch_used,
                current_batch_size,
                start_batch_size,
                VLM_BATCH_MAX_SIZE,
                VLM_BATCH_MAX_NEW_TOKENS_PER_CROP,
            )

    def _maybe_grow_batch() -> None:
        nonlocal current_batch_size, stable_windows
        if stable_windows >= VLM_BATCH_STABLE_STEPS and current_batch_size < VLM_BATCH_MAX_SIZE:
            current_batch_size = min(
                VLM_BATCH_MAX_SIZE,
                current_batch_size + VLM_BATCH_GROW_STEP,
            )
            stable_windows = 0

    while i < total:
        batch = crops[i:i + current_batch_size]
        n = len(batch)

        if n == 1:
            results.append(_caption_crop_vlm(batch[0]))
            i += 1
            stable_windows += 1
            _log_progress(len(results), n)
            _maybe_grow_batch()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        inputs = None
        output_ids = None
        scores = []
        try:
            texts = []
            for crop in batch:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": _VLM_PROMPT},
                ]}]
                texts.append(processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))

            inputs = processor(
                text=texts,
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=VLM_BATCH_MAX_NEW_TOKENS_PER_CROP,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            output_ids = gen.sequences
            scores = list(gen.scores or [])

            input_len = inputs["input_ids"].shape[1]
            tokenizer = getattr(processor, "tokenizer", processor)
            batch_attrs: list[dict | None] = []
            fallback_rows: list[int] = []

            for row_idx, crop in enumerate(batch):
                gen_ids = output_ids[row_idx][input_len:]
                raw = processor.decode(gen_ids, skip_special_tokens=True).strip()

                try:
                    attrs = _parse_vlm_attrs(_extract_json_object(raw))
                except Exception as parse_exc:
                    logger.warning(
                        "[vlm] %s output parse failed at %d/%d (batch=%d row=%d): %s",
                        tag,
                        i + row_idx,
                        total,
                        n,
                        row_idx,
                        parse_exc,
                    )
                    batch_attrs.append(None)
                    fallback_rows.append(row_idx)
                    continue

                try:
                    row_scores = [
                        score_tensor[row_idx] if getattr(score_tensor, "ndim", 0) > 1 else score_tensor
                        for score_tensor in scores
                    ]
                    logit_confs = _compute_value_logprob_confs(
                        gen_ids.detach().cpu(), row_scores, tokenizer,
                    )
                    _attach_logit_confs(attrs, logit_confs)
                except Exception as conf_exc:
                    logger.warning(
                        "[vlm] logit-conf compute failed (batch=%d row=%d): %s",
                        n,
                        row_idx,
                        conf_exc,
                    )

                batch_attrs.append(attrs)

            had_fallback_rows = bool(fallback_rows)
            if had_fallback_rows:
                logger.warning(
                    "[vlm] %s batch(%d) at offset %d needs %d single-crop parse fallback(s)",
                    tag,
                    n,
                    i,
                    len(fallback_rows),
                )
                try:
                    del gen
                    del inputs
                    del output_ids
                    del scores
                except Exception:
                    pass
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for row_idx in fallback_rows:
                    batch_attrs[row_idx] = _caption_crop_vlm(batch[row_idx])
            else:
                logger.debug("[vlm] independent batch(%d) OK at offset %d", n, i)

            results.extend(attrs if attrs is not None else _default_attributes() for attrs in batch_attrs)
            i += n
            if had_fallback_rows:
                stable_windows = 0
                if len(fallback_rows) == n and current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
            else:
                stable_windows += 1
            _log_progress(len(results), n)
            if not had_fallback_rows:
                _maybe_grow_batch()

        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and current_batch_size > 1:
                next_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "[vlm] %s OOM at %d/%d (batch=%d) -> retry batch=%d",
                    tag,
                    i,
                    total,
                    current_batch_size,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
                stable_windows = 0
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            if current_batch_size > 1:
                next_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "[vlm] %s batch(%d) failed at %d/%d: %s -> retry batch=%d",
                    tag,
                    n,
                    i,
                    total,
                    exc,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
                stable_windows = 0
                continue
            logger.warning("[vlm] single crop failed at %d/%d: %s", i, total, exc)
            results.append(_default_attributes())
            i += 1
            _log_progress(len(results), 1)

        except Exception as exc:
            if current_batch_size > 1:
                next_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "[vlm] %s batch(%d) failed at %d/%d: %s -> retry batch=%d",
                    tag,
                    n,
                    i,
                    total,
                    exc,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
                stable_windows = 0
                continue

            logger.warning("[vlm] single crop failed at %d/%d: %s", i, total, exc)
            results.append(_default_attributes())
            i += 1
            _log_progress(len(results), 1)

        finally:
            try:
                del inputs
                del output_ids
                del scores
            except Exception:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def _build_appearance_summary(attrs: dict) -> str:
    # Prefer VLM-generated summary (open-vocabulary, accurate)
    vlm_summary = attrs.get("appearance_summary")
    if vlm_summary and str(vlm_summary).strip() and str(vlm_summary).strip().lower() != "person":
        return str(vlm_summary).strip()
    # Fallback: compose from VLM desc fields
    parts = []
    gender = (attrs.get("gender") or "").strip()
    upper = (attrs.get("upper_desc") or "").strip()
    lower = (attrs.get("lower_desc") or "").strip()
    if gender and gender != "unknown":
        parts.append(gender)
    if upper and upper != "unknown":
        parts.append(upper)
    if lower and lower != "unknown":
        parts.append(lower)
    return " ".join(parts) or "person"


def _extract_crop_for_vlm(frame: np.ndarray, bbox: list[float]) -> Optional[np.ndarray]:
    """Extract a square-padded 384×384 crop from frame for VLM input."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    max_dim = max(crop.shape[0], crop.shape[1])
    top = (max_dim - crop.shape[0]) // 2
    bottom = max_dim - crop.shape[0] - top
    left = (max_dim - crop.shape[1]) // 2
    right = max_dim - crop.shape[1] - left
    padded = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return cv2.resize(padded, (384, 384), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Per-video endpoint (single video, existing behaviour)
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessVideoResponse)
def process_video(req: ProcessVideoRequest) -> ProcessVideoResponse:
    """Process a single video (single-camera tracking pipeline)."""
    start = time.time()
    logger.info("Processing video: id=%s camera=%s", req.video_id, req.camera_id)

    video_path = req.video_path
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    camera_id = req.camera_id or "Camera_0000"

    try:
        frames, fps = _video_to_frames(video_path, max_frames=500)
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read video frames")

    if not frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    logger.info("Loaded %d frames (fps=%.1f)", len(frames), fps)

    sample_interval = req.sample_interval or DEFAULT_SAMPLE_INTERVAL
    sampled = _sample_frames_uniform(frames, fps, sample_interval)
    all_detections: list[dict] = []

    sampled_imgs = [frame for _, frame in sampled]
    batch_dets = _detect_persons_batch(sampled_imgs, threshold=0.22)
    for (frame_idx, _frame), dets in zip(sampled, batch_dets):
        for det in dets:
            det["frame_idx"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
            det["video_id"] = req.video_id
        all_detections.extend(dets)

    logger.info("Detected %d person detections", len(all_detections))

    if not all_detections:
        return ProcessVideoResponse(
            video_id=req.video_id,
            camera_id=camera_id,
            tracklets=[],
            total_detections=0,
            processing_time_s=time.time() - start,
        )

    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    all_detections = _project_to_bev_single(all_detections, camera_id, cal_path)

    detections_by_camera = {camera_id: all_detections}
    groups = _mcblt_associate(detections_by_camera, max_dist=req.bev_max_dist or DEFAULT_BEV_MAX_DIST)
    logger.info("MCBLT formed %d tracklet groups", len(groups))

    tracklets: list[TrackletResult] = []
    for group_idx, group in enumerate(groups):
        if len(group) < 1:
            continue

        group = sorted(group, key=lambda d: d.get("frame_idx", 0))
        start_frame = group[0].get("frame_idx", 0)
        end_frame = group[-1].get("frame_idx", len(frames) - 1)
        step = max(1, (end_frame - start_frame) // 16)
        tracklet_frames = frames[start_frame:end_frame + 1:step]
        if not tracklet_frames:
            tracklet_frames = [frames[min(start_frame, len(frames) - 1)]]

        mid_det = group[len(group) // 2]
        rep_bbox = mid_det["bbox"]
        rep_bev_x = mid_det.get("bev_x", 0.0)
        rep_bev_y = mid_det.get("bev_y", 0.0)

        embedding = _generate_dinov2_embeddings(
            tracklet_frames, [rep_bbox] * len(tracklet_frames),
            f"{req.video_id}_{group_idx}"
        )
        mid_frame = tracklet_frames[len(tracklet_frames) // 2]
        rep_crop_cv = _extract_crop_for_vlm(mid_frame, rep_bbox)
        rep_crop_pil = Image.fromarray(cv2.cvtColor(rep_crop_cv, cv2.COLOR_BGR2RGB)) if rep_crop_cv is not None \
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        attributes = _caption_crop_vlm(rep_crop_pil)
        action = _run_videomae_actions(tracklet_frames, rep_bbox)
        summary = _build_appearance_summary(attributes)

        tracklets.append(_build_tracklet_result(
            tracklet_id=f"{req.video_id}_{camera_id}_{group_idx}",
            video_id=req.video_id,
            camera_id=camera_id,
            track_idx=group_idx,
            start_time=group[0].get("timestamp", 0),
            end_time=group[-1].get("timestamp", 0),
            quality_score=float(mid_det.get("score", 0.5)),
            attrs=attributes,
            summary=summary,
            rep_bbox=rep_bbox,
            bev_x=rep_bev_x,
            bev_y=rep_bev_y,
            siglip_embedding=[],
            action=action if isinstance(action, str) else str(action),
            action_confidence=0.0,
            kinetics_label="",
            crop_url="",
        ))

    elapsed = time.time() - start
    logger.info("Processed %s: %d tracklets in %.1fs", req.video_id, len(tracklets), elapsed)

    return ProcessVideoResponse(
        video_id=req.video_id,
        camera_id=camera_id,
        tracklets=tracklets,
        total_detections=len(all_detections),
        processing_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# STREAMING endpoint — accepts video bytes directly (no disk download)
# ---------------------------------------------------------------------------

@router.post("/process/stream", response_model=ProcessVideoResponse)
async def process_video_stream(
    video_id: str = Form(...),
    camera_id: str | None = Form(None),
    source_filename: str | None = Form(None),
    sample_interval: int = Form(15),
    bev_max_dist: float = Form(1.5),
    video: UploadFile = File(...),
) -> ProcessVideoResponse:
    """
    Accept video bytes via multipart upload, write to a temp file,
    process, then delete the temp file.

    This lets ingest_service stream bytes directly from Google Drive
    to the GPU pipeline without caching the file on disk permanently.
    """
    import tempfile

    tmp_path: Path | None = None
    try:
        suffix = Path(source_filename or "video").suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await video.read(1024 * 1024 * 50)  # 50 MB chunks
                if not chunk:
                    break
                tmp.write(chunk)

        logger.info("[stream] Received %s (%s), saved to %s", video_id, source_filename, tmp_path)

        # Delegate to the sync process handler (uses cv2 which is sync-only)
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            _process_video_sync,
            str(tmp_path),
            video_id,
            camera_id,
            sample_interval,
            bev_max_dist,
        )
        return result

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.info("[stream] Cleaned up temp file: %s", tmp_path)
            except OSError:
                pass


def _best_observation(obs: list):
    """Pick the observation that maximizes frame quality for SigLIP/Qwen crops.

    Score = 0.5 * bbox_area_ratio + 0.3 * laplacian_norm + 0.2 * confidence
    where bbox_area_ratio and laplacian_norm are normalised to [0, 1] across obs.
    Falls back to median frame if scoring fails.
    """
    if not obs:
        return None
    if len(obs) == 1:
        return obs[0]
    try:
        areas = []
        for o in obs:
            x1, y1, x2, y2 = o.bbox
            areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
        laps = [float(getattr(o, "laplacian_score", 0.0) or 0.0) for o in obs]
        confs = [float(getattr(o, "confidence", 0.0) or 0.0) for o in obs]

        max_area = max(areas) or 1.0
        max_lap  = max(laps)  or 1.0

        best, best_score = obs[len(obs) // 2], -1.0
        for o, area, lap, conf in zip(obs, areas, laps, confs):
            score = 0.5 * (area / max_area) + 0.3 * (lap / max_lap) + 0.2 * conf
            if score > best_score:
                best_score = score
                best = o
        return best
    except Exception:
        return obs[len(obs) // 2]


def _batch_siglip_embeddings(
    t_data: list,
    video_id: str,
    frame_lookup: dict | None = None,
) -> tuple[list, list, list]:
    """
    Batch SigLIP2 embeddings only. DINOv2 removed — SigLIP2 is the sole embedding model.
    Returns: (siglip_multi_feats, all_rep_crops, all_siglip_embeddings)
      siglip_multi_feats: multi-frame pool-avg per fragment (for fragment merge)
      all_rep_crops: PIL Images 384×384 from best-quality frame (for Qwen/storage)
      all_siglip_embeddings: single-crop embedding per fragment (for DB)

    frame_lookup: optional {frame_index -> image} mapping. When provided, the
    multi-frame SigLIP path crops each frame with that observation's own bbox
    instead of a frozen rep_bbox — critical for moving subjects.
    """
    if frame_lookup is None:
        frame_lookup = {}
    device = _get_device()
    dtype = torch.float16

    def _extract_crop(frame, bbox, size):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        max_dim = max(crop.shape[0], crop.shape[1])
        pad = cv2.copyMakeBorder(
            crop,
            (max_dim - crop.shape[0]) // 2, max_dim - crop.shape[0] - (max_dim - crop.shape[0]) // 2,
            (max_dim - crop.shape[1]) // 2, max_dim - crop.shape[1] - (max_dim - crop.shape[1]) // 2,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return cv2.resize(pad, (size, size), interpolation=cv2.INTER_LINEAR)

    def _select_siglip_observation_entries(lt, *, min_frames: int = 5, max_frames: int = 8) -> list[dict]:
        """Pick quality-aware, temporally diverse observations for identity embedding."""
        entries: list[dict] = []
        for order, o in enumerate(lt.observations):
            frame = frame_lookup.get(o.frame_index)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in o.bbox)
            bw = max(0.0, min(float(w), x2) - max(0.0, x1))
            bh = max(0.0, min(float(h), y2) - max(0.0, y1))
            if bw <= 1.0 or bh <= 1.0:
                continue
            edge_clear = min(
                max(0.0, x1),
                max(0.0, y1),
                max(0.0, float(w) - x2),
                max(0.0, float(h) - y2),
            )
            edge_norm = min(1.0, edge_clear / max(1.0, 0.04 * min(h, w)))
            edge_touch = x1 <= 1.0 or y1 <= 1.0 or x2 >= float(w - 1) or y2 >= float(h - 1)
            entries.append({
                "order": order,
                "obs": o,
                "frame": frame,
                "bbox": [float(v) for v in o.bbox],
                "area": bw * bh,
                "lap": float(getattr(o, "laplacian_score", 0.0) or 0.0),
                "conf": float(getattr(o, "confidence", 0.0) or 0.0),
                "edge_norm": edge_norm,
                "edge_touch": edge_touch,
                "low_quality": bool(getattr(o, "is_low_quality_crop", False)),
            })
        if not entries:
            return []

        max_area = max((e["area"] for e in entries), default=1.0) or 1.0
        max_lap = max((e["lap"] for e in entries), default=1.0) or 1.0
        for e in entries:
            area_norm = e["area"] / max_area
            lap_norm = min(1.0, e["lap"] / max_lap)
            score = (
                0.38 * lap_norm
                + 0.30 * area_norm
                + 0.20 * e["conf"]
                + 0.12 * e["edge_norm"]
            )
            if e["low_quality"]:
                score -= 0.35
            if e["edge_touch"]:
                score -= 0.20
            e["score"] = score

        if len(entries) <= min_frames:
            return entries

        target = min(max_frames, len(entries))
        target = max(min_frames, target)
        chosen: set[int] = set()
        boundaries = np.linspace(0, len(entries), target + 1, dtype=int)
        for i in range(target):
            start_i, end_i = int(boundaries[i]), int(boundaries[i + 1])
            if end_i <= start_i:
                continue
            best_i = max(range(start_i, end_i), key=lambda idx: entries[idx]["score"])
            chosen.add(best_i)

        if len(chosen) < target:
            for idx in sorted(range(len(entries)), key=lambda i: entries[i]["score"], reverse=True):
                chosen.add(idx)
                if len(chosen) >= target:
                    break

        return [entries[idx] for idx in sorted(chosen)]

    # ── Representative crops (384×384) from best-quality frame ───────────────
    # best_obs is determined once here; the same crop feeds both SigLIP and Qwen.
    all_rep_crops: list[Image.Image] = []
    for lt, t_idx, rep_bbox, t_frames in t_data:
        best_obs = _best_observation(lt.observations)
        if best_obs is not None:
            rep_bbox = [float(x) for x in best_obs.bbox]
            # t_frames is indexed by position; find the frame matching best_obs
            best_frame_idx = next(
                (i for i, o in enumerate(lt.observations) if o is best_obs), len(t_frames) // 2
            )
            best_frame = t_frames[best_frame_idx] if best_frame_idx < len(t_frames) else t_frames[len(t_frames) // 2]
        else:
            best_frame = t_frames[len(t_frames) // 2] if t_frames else None
        c = _extract_crop(best_frame, rep_bbox, 384) if best_frame is not None else None
        all_rep_crops.append(
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) if c is not None
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        )

    model_sip = get_model("siglip2")
    proc_sip = get_model("siglip2_processor")
    _t0 = time.perf_counter()

    siglip_init_batch = _get_positive_env_int("SIGLIP_BATCH_INIT", 128)
    siglip_max_batch = _get_positive_env_int("SIGLIP_BATCH_MAX", 256)
    siglip_batch_step = _get_positive_env_int("SIGLIP_BATCH_GROW_STEP", 16)

    def _encode_siglip_images_adaptive(images: list[Image.Image], phase: str) -> torch.Tensor | None:
        if not images or model_sip is None or proc_sip is None:
            return None
        batch = max(1, min(siglip_init_batch, siglip_max_batch, len(images)))
        feats_chunks: list[torch.Tensor] = []
        idx = 0
        stable_windows = 0

        while idx < len(images):
            end = min(len(images), idx + batch)
            window = images[idx:end]
            try:
                inputs = proc_sip(images=window, return_tensors="pt", padding=True)
                inputs = {
                    k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                    for k, v in inputs.items()
                }
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.no_grad(), amp_ctx:
                    feats = model_sip.get_image_features(
                        **{k: v for k, v in inputs.items() if k in ["pixel_values"]}
                    )
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_chunks.append(feats.detach().cpu().float())
                idx = end
                stable_windows += 1

                # Slowly increase after stable windows to keep speed for easy videos.
                if stable_windows >= 3 and batch < siglip_max_batch:
                    batch = min(siglip_max_batch, batch + siglip_batch_step, len(images) - idx or batch)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg and batch > 1:
                    new_batch = max(1, batch // 2)
                    logger.warning(
                        "[pipeline] %s: SigLIP %s OOM at %d/%d (batch=%d) -> retry batch=%d",
                        video_id,
                        phase,
                        idx,
                        len(images),
                        batch,
                        new_batch,
                    )
                    batch = new_batch
                    stable_windows = 0
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise
            finally:
                try:
                    del inputs
                except Exception:
                    pass
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if not feats_chunks:
            return None
        return torch.cat(feats_chunks, dim=0)

    # ── SigLIP2 multi-frame pool-avg — for fragment merge ────────────────────
    # Use quality-aware temporal buckets and PER-OBSERVATION bbox (not the
    # frozen rep_bbox), so selected crops are sharp, less edge-clipped, and
    # still spread across the tracklet.
    siglip_multi_feats = [[] for _ in t_data]
    if model_sip and proc_sip and t_data:
        try:
            siglip_crops, siglip_slices = [], []
            for lt, t_idx, rep_bbox, t_frames in t_data:
                selected_entries = _select_siglip_observation_entries(lt) if frame_lookup else []

                if selected_entries:
                    start = len(siglip_crops)
                    for entry in selected_entries:
                        c = _extract_crop(entry["frame"], entry["bbox"], 384)
                        if c is not None:
                            siglip_crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                    siglip_slices.append((start, len(siglip_crops)))
                else:
                    # Legacy fallback when no frame_lookup is provided: sample
                    # t_frames uniformly with the (frozen) rep_bbox. Less
                    # accurate for moving subjects but keeps callers without
                    # a frame_lookup working.
                    n = min(5, len(t_frames))
                    indices = np.linspace(0, len(t_frames) - 1, n, dtype=int)
                    start = len(siglip_crops)
                    for idx in indices:
                        c = _extract_crop(t_frames[idx], rep_bbox, 384)
                        if c is not None:
                            siglip_crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                    siglip_slices.append((start, len(siglip_crops)))

            if siglip_crops:
                feats = _encode_siglip_images_adaptive(siglip_crops, phase="multi-frame")
                if feats is not None:
                    for t_i, (s, e) in enumerate(siglip_slices):
                        if e > s:
                            avg = feats[s:e].mean(0)
                            siglip_multi_feats[t_i] = (avg / avg.norm()).cpu().float().tolist()
        except Exception as exc:
            logger.warning("[pipeline] SigLIP2 multi-frame encoding failed: %s", exc)
            siglip_multi_feats = [[] for _ in t_data]

    # ── SigLIP2 single-crop embedding — for DB storage ───────────────────────
    img_feats = None
    if model_sip and proc_sip and t_data:
        try:
            img_feats = _encode_siglip_images_adaptive(all_rep_crops, phase="single-crop")
        except Exception as exc:
            logger.warning("[pipeline] SigLIP image encoding failed: %s", exc)
            img_feats = None

    all_siglip_embeddings: list[list[float]] = []
    try:
        if img_feats is not None:
            all_siglip_embeddings = [img_feats[i].cpu().float().tolist() for i in range(len(t_data))]
        else:
            all_siglip_embeddings = [[]] * len(t_data)
    except Exception:
        all_siglip_embeddings = [[]] * len(t_data)

    logger.info("[pipeline] %s: SigLIP2 done in %.1fs — %d fragments | merge_emb=%d | DB_emb=%d",
                video_id, time.perf_counter() - _t0, len(t_data),
                sum(1 for e in siglip_multi_feats if e),
                sum(1 for e in all_siglip_embeddings if e))
    return siglip_multi_feats, all_rep_crops, all_siglip_embeddings


def _process_video_sync(
    video_path: str,
    video_id: str,
    camera_id: str | None,
    sample_interval: int,
    bev_max_dist: float,
    presampled_frames=None,
) -> ProcessVideoResponse:
    """
    Sync video processing pipeline.

    Tracker backend is selected by env var `TRACER_BACKEND`:
      - `botsort` (default) — BoT-SORT via boxmot, no ReID, no GMC.
      - `adaptive`         — legacy BodyPartAdaptiveTracker (kept for benchmarks).

    presampled_frames: pre-decoded frames from background thread (skips Stage 1).
    """
    import time
    from .tracking_pipeline import (
        VideoFrameSampler, BodyPartAdaptiveTracker, TrackletQualityScorer,
        TrackletFragmentMerger, FrameDetection, _crop_from_bbox,
        _crop_laplacian_variance, _is_low_quality_crop,
    )
    from .botsort_tracker import BotSortTracker, is_botsort_backend
    start = time.time()
    camera_id = camera_id or "Camera_0000"

    # Stage 1: Sample frames at 6fps (skip if pre-decoded externally).
    # BoT-SORT relies on Kalman + IoU, so it benefits from smaller inter-frame
    # motion than the legacy adaptive tracker. Keep PIPELINE_SAMPLE_FPS tunable
    # for per-camera latency/accuracy trade-offs.
    sample_fps = int(os.environ.get("PIPELINE_SAMPLE_FPS", "6"))
    if presampled_frames is not None:
        sampled_frames = presampled_frames
    else:
        sampler = VideoFrameSampler(sample_fps=sample_fps)
        try:
            sampled_frames = sampler.sample(video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read video: {e}")

    if not sampled_frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    logger.warning("[pipeline] %s: %d frames sampled at %dfps", video_id, len(sampled_frames), sample_fps)

    # Stage 2: Batch detect — RT-DETR primary, GDINO fallback
    from .tracking_pipeline import _crop_from_bbox as _tcrop
    t_det_start = time.time()
    detector = "RT-DETR" if get_model("rtdetr") is not None else "GDINO-fallback"
    logger.info("[pipeline] %s: running %s detection on %d frames...", video_id, detector, len(sampled_frames))
    all_batch_dets = _detect_persons_batch([sf.image for sf in sampled_frames], threshold=0.22)
    logger.warning("[pipeline] %s: %s done in %.1fs", video_id, detector, time.time() - t_det_start)

    # B1: build (sf, det) work items, run crop + crop-Laplacian in parallel.
    # cv2 releases the GIL inside cvtColor + Laplacian + slicing, so threads
    # actually run concurrently. Saves ~2-5s/video on dense scenes.
    work_items: list[tuple[int, int, np.ndarray, tuple, int, int, float]] = []
    for sf_idx, (sf, raw_dets) in enumerate(zip(sampled_frames, all_batch_dets)):
        frame_h, frame_w = sf.image.shape[:2]
        for d_idx, d in enumerate(raw_dets):
            bbox = tuple(int(x) for x in d["bbox"])
            work_items.append((sf_idx, d_idx, sf.image, bbox, frame_h, frame_w, float(d["score"])))

    def _build_one(item):
        sf_idx, d_idx, image, bbox, frame_h, frame_w, score = item
        crop = _tcrop(image, bbox)
        crop_lap = _crop_laplacian_variance(crop)
        low_q = _is_low_quality_crop(bbox, frame_h, frame_w)
        return sf_idx, d_idx, bbox, crop, crop_lap, low_q, score

    # Bucket results back per source frame. Preserve original det order via d_idx.
    per_frame: dict[int, list] = {}
    if work_items:
        with ThreadPoolExecutor(max_workers=min(8, len(work_items))) as ex:
            for sf_idx, d_idx, bbox, crop, crop_lap, low_q, score in ex.map(_build_one, work_items):
                per_frame.setdefault(sf_idx, []).append((d_idx, bbox, crop, crop_lap, low_q, score))

    detections_by_frame: dict[int, list[FrameDetection]] = {}
    total_raw = 0
    total_low_quality = 0
    for sf_idx, sf in enumerate(sampled_frames):
        items = per_frame.get(sf_idx) or []
        if items:
            items.sort(key=lambda x: x[0])  # restore original detection order
        frame_dets: list[FrameDetection] = []
        for _d_idx, bbox, crop, crop_lap, low_q, score in items:
            if low_q:
                total_low_quality += 1
            # B1 (legacy comment): laplacian on the person crop, not the full frame.
            # B3 (legacy comment): is_low_quality_crop flags half-body / edge-clipped
            # crops; tracker still receives them so it can keep an ID through occlusion.
            frame_dets.append(FrameDetection(
                frame_index=sf.frame_index,
                timestamp_second=sf.timestamp_second,
                bbox=bbox,
                confidence=score,
                laplacian_score=crop_lap,
                crop_bgr=crop,
                is_low_quality_crop=low_q,
            ))
        detections_by_frame[sf.frame_index] = frame_dets
        total_raw += len(frame_dets)
    if total_raw:
        logger.warning("[pipeline] %s: %d/%d detections flagged low-quality crop (%.1f%%)",
                       video_id, total_low_quality, total_raw,
                       100.0 * total_low_quality / total_raw)

    logger.warning("[pipeline] %s: %d detections across %d sampled frames", video_id, total_raw, len(detections_by_frame))

    # ── Diagnostic: detection density + confidence distribution ──────────────
    # Helps answer "is tracking losing the person, or is the detector?".
    # frames_with_no_det = sampled frames where RT-DETR found nothing —
    #   if this is high, the gap is in detection, not tracking.
    # conf histogram shows how many dets are stuck in the low-conf band
    #   (0.22, 0.30) which BoT-SORT cannot use to start a new track.
    _det_diag = {
        "frames_with_no_det": sum(1 for v in detections_by_frame.values() if not v),
        "det_conf_lt_030": 0,
        "det_conf_030_050": 0,
        "det_conf_050_080": 0,
        "det_conf_ge_080": 0,
        "det_lap_lt_25": 0,
    }
    for v in detections_by_frame.values():
        for d in v:
            c = d.confidence
            if c < 0.30: _det_diag["det_conf_lt_030"] += 1
            elif c < 0.50: _det_diag["det_conf_030_050"] += 1
            elif c < 0.80: _det_diag["det_conf_050_080"] += 1
            else: _det_diag["det_conf_ge_080"] += 1
            if d.laplacian_score < 25.0:
                _det_diag["det_lap_lt_25"] += 1
    logger.info(
        "[diag-det] %s: empty_frames=%d/%d  conf<0.30=%d  0.30-0.50=%d  0.50-0.80=%d  >=0.80=%d  lap<25=%d",
        video_id,
        _det_diag["frames_with_no_det"], len(detections_by_frame),
        _det_diag["det_conf_lt_030"], _det_diag["det_conf_030_050"],
        _det_diag["det_conf_050_080"], _det_diag["det_conf_ge_080"],
        _det_diag["det_lap_lt_25"],
    )

    import torch as _torch

    if total_raw == 0:
        logger.warning("[pipeline] %s: no persons detected", video_id)
        return ProcessVideoResponse(
            video_id=video_id, camera_id=camera_id,
            tracklets=[], total_detections=0,
            processing_time_s=time.time() - start,
        )

    # Stage 3: Tracking.
    # Default backend is BoT-SORT (boxmot, no ReID, no GMC). Set
    # TRACER_BACKEND=adaptive to fall back to the legacy BodyPartAdaptiveTracker
    # (kept in tracking_pipeline.py for camera_0002 benchmarks).
    _fps_ratio_self = sample_fps / 4.0             # 3fps→0.75, 4fps→1.0, 6fps→1.5
    # BoT-SORT adapter accepts track_buffer as effective sampled-frame lost
    # buffer and converts it to BoxMOT's 30-FPS-scaled constructor argument.
    # At the 6 FPS default this is 30 sampled frames ≈ 5 seconds.
    _botsort_track_buffer = max(int(round(5.0 * sample_fps)), 8)
    # Legacy adaptive tracker used 20 frames at 4 FPS, also ≈ 5 seconds.
    _adaptive_track_buffer = max(int(round(20 * _fps_ratio_self)), 8)
    if is_botsort_backend():
        tracker = BotSortTracker(
            track_thresh=0.30,
            low_thresh=0.10,
            new_track_threshold=0.30,
            # match_thresh in boxmot = upper bound on (1 - IoU) cost. 0.85
            # → accept match when IoU ≥ 0.15, 0.80 → IoU ≥ 0.20 (tight).
            # Loosened from 0.80 → 0.85 to address "track id jumps mid-FOV while
            # person walks normally": at 6 FPS with fast walkers near the camera,
            # adjacent-frame IoU can drop below 0.20 → match fails → BoT-SORT
            # spawns a new id. 0.85 is a middle ground: looser than 0.80 so the
            # tracker survives fast walkers, tighter than 0.90 so it does not
            # accept cross-person matches when two people pass close together.
            match_thresh=0.85,
            track_buffer=_botsort_track_buffer,
            frame_rate=max(sample_fps, 1),
            min_track_frames=2,
            min_track_density=0.03,
            max_center_jump_ratio=1.60,
            max_speed_px_per_s=800.0,
            min_short_gap_iou=0.02,
        )
        logger.warning("[pipeline] %s: tracker backend = BoT-SORT (no-ReID, no-GMC)", video_id)
    else:
        # Pixel-per-frame thresholds scale with FPS (lower FPS → people move more
        # between frames → larger gate). Buffer counts scale to keep wall-clock
        # seconds constant. Defaults below correspond to the legacy 4 FPS config.
        # min_track_frames=2 (the legacy value) — benchmark on camera_0002 showed
        # G6 (raising to 3) does not improve purity once G4 post-hoc split is on.
        _fps_ratio_4 = 4.0 / max(sample_fps, 1)    # 3fps→1.33, 4fps→1.0, 6fps→0.67
        tracker = BodyPartAdaptiveTracker(
            track_thresh=0.30,
            low_thresh=0.10,
            new_track_threshold=0.30,
            min_track_frames=2,
            min_track_density=0.03,
            # Bench cam_0002 (sweep margin 0.10 → 0.30 trên 25 GT person):
            #   0.10 → impurity 4.81%, contam 19.87%, IDS 580, frag/GT 83.12
            #   0.20 → impurity 2.01%, contam 10.58%, IDS 265, frag/GT 82.44  ★
            #   0.30 → impurity 0.59% nhưng max track length -83%, concurrency -51%
            # 0.20 là sweet spot: giảm merge nhầm 58%, IDS 54%, frag/GT gần như
            # giữ nguyên, không phá vỡ track dài (max length giữ ở 648 frame).
            discriminative_margin=0.15,
            max_head_center_distance=120.0 * _fps_ratio_4,
            max_foot_distance=150.0 * _fps_ratio_4,
            max_predicted_distance=180.0 * _fps_ratio_4,
            max_center_jump_ratio=2.0 * _fps_ratio_4,
            min_active_iou_short_gap=0.0,
            max_phantom_frames=1,
            track_buffer=_adaptive_track_buffer,
            max_buffer_frames=max(int(round(300 * _fps_ratio_self)), 100),
        )
        logger.warning("[pipeline] %s: tracker backend = BodyPartAdaptiveTracker (legacy)", video_id)
    local_tracklets = tracker.track(video_id, camera_id, detections_by_frame)
    logger.warning("[pipeline] %s: %d raw tracklets from tracker", video_id, len(local_tracklets))

    # ── Diagnostic: tracklet length/gap distribution from the tracker ────────
    # If many tracklets are short (2-4 obs) the tracker is breaking IDs mid-FOV.
    # split_ids = tracklets whose id ends with "_sN" (post-association split guard
    # broke a BoT-SORT join). High count means the guard is firing a lot.
    if local_tracklets:
        _lengths = [len(t.observations) for t in local_tracklets]
        _spans = [
            max(t.observations[-1].frame_index - t.observations[0].frame_index + 1, 1)
            for t in local_tracklets
        ]
        _split_ids = sum(1 for t in local_tracklets if "_s" in t.track_id)
        _short = sum(1 for L in _lengths if L <= 4)
        _med_len = sorted(_lengths)[len(_lengths) // 2]
        _med_span_s = sorted(_spans)[len(_spans) // 2] / max(sample_fps, 1)
        logger.info(
            "[diag-track] %s: median_len=%d  median_span=%.1fs  short(<=4obs)=%d/%d  split_fragments=%d",
            video_id, _med_len, _med_span_s, _short, len(local_tracklets), _split_ids,
        )

    # Stage 4: Quality filter — siết để loại tracklet stub trước khi vào
    # SigLIP merge và Qwen captioning. Trước: 1893 vào → 1892 ra (filter
    # không tồn tại). Sau: kỳ vọng loại được 30-40% các tracklet 2-3 obs
    # noise/false-positive.
    scorer = TrackletQualityScorer(
        min_confidence=0.35,
        min_frames=2,           # require at least two detections before embedding
        min_density=0.15,       # 1 obs / 6.6 frames = ~2.2s — tracklet liên tục
        min_duration_s=0.4,     # drop very short FP detection bursts
        min_laplacian=25.0,
    )
    quality_results = {t.track_id: scorer.score(t) for t in local_tracklets}
    accepted = [t for t in local_tracklets if quality_results[t.track_id].accepted]
    rejected_reasons: dict = {}
    # Per-reason length / Laplacian / duration stats so we can see whether the
    # rejected tracklets are obviously bad (≤2 obs, lap<5) or borderline real
    # (15 obs, lap 22) — borderline rejections suggest the gate is too tight.
    _rej_detail: dict[str, list[tuple[int, float, float]]] = {}
    for t in local_tracklets:
        q = quality_results[t.track_id]
        if not q.accepted:
            r = q.rejection_reason or "unknown"
            rejected_reasons[r] = rejected_reasons.get(r, 0) + 1
            _rej_detail.setdefault(r, []).append(
                (q.frame_count, q.average_laplacian, q.duration_seconds)
            )
    logger.warning("[pipeline] %s: %d accepted, %d rejected %s",
                   video_id, len(accepted), len(local_tracklets) - len(accepted), rejected_reasons)
    for reason, samples in _rej_detail.items():
        if not samples:
            continue
        ns = [s[0] for s in samples]; laps = [s[1] for s in samples]; durs = [s[2] for s in samples]
        logger.info(
            "[diag-stage4] %s: reason=%s  n=%d  median_obs=%d  median_lap=%.1f  median_dur=%.2fs",
            video_id, reason, len(samples),
            sorted(ns)[len(ns) // 2],
            sorted(laps)[len(laps) // 2],
            sorted(durs)[len(durs) // 2],
        )

    # Stage 5-7: TRUE batch feature extraction — 1 GPU call per model for ALL tracklets
    frame_lookup = {sf.frame_index: sf.image for sf in sampled_frames}

    t_data = []
    for t_idx, lt in enumerate(accepted):
        best = _best_observation(lt.observations)
        rep_bbox_float = [float(x) for x in best.bbox]
        t_frames = [frame_lookup[o.frame_index] for o in lt.observations if o.frame_index in frame_lookup] or [sampled_frames[0].image]
        t_data.append((lt, t_idx, rep_bbox_float, t_frames))

    siglip_multi_feats, all_rep_crops, all_siglip_embeddings = _batch_siglip_embeddings(
        t_data, video_id, frame_lookup=frame_lookup,
    )

    # Stage 8: Post-hoc fragment merging via SigLIP2 cosine similarity.
    import numpy as _np

    _n_raw = len(accepted)
    logger.info(
        "[merge] %s: %s  0/%d — merging fragments (emb=SigLIP2, threshold=%.2f, max_gap=%.0fs)",
        video_id, _vlm_progress_bar(0, _n_raw), _n_raw,
        FRAGMENT_MERGE_SIM_THRESHOLD, FRAGMENT_MERGE_MAX_GAP_SECONDS,
    )

    _merger = TrackletFragmentMerger(
        similarity_threshold=FRAGMENT_MERGE_SIM_THRESHOLD,
        max_gap_seconds=FRAGMENT_MERGE_MAX_GAP_SECONDS,
        component_similarity_margin=FRAGMENT_MERGE_COMPONENT_MARGIN,
        max_speed_px_per_s=FRAGMENT_MERGE_MAX_SPEED_PX_PER_S,
        spatial_bypass_margin=FRAGMENT_MERGE_SPATIAL_BYPASS_MARGIN,
        max_spatial_dist_px=FRAGMENT_MERGE_MAX_SPATIAL_DIST_PX,
        motion_merge_min_appearance_sim=FRAGMENT_MERGE_MOTION_MIN_SIM,
    )
    _orig_accepted = list(accepted)
    _t_merge = time.perf_counter()
    accepted, _groups = _merger.merge(list(accepted), siglip_multi_feats)
    _merge_elapsed = time.perf_counter() - _t_merge

    _n_merged = sum(len(g) - 1 for g in _groups if len(g) > 1)
    _multi_groups = [g for g in _groups if len(g) > 1]
    logger.info(
        "[merge] %s: %s  %d/%d → %d tracklets (%d fragments joined, %d groups, %.2fs)",
        video_id, _vlm_progress_bar(_n_raw, _n_raw), _n_raw, _n_raw,
        len(accepted), _n_merged, len(_multi_groups), _merge_elapsed,
    )
    # Enriched per-group log: covers track_ids + total time-span. A group whose
    # time-span is much wider than any plausible single appearance, or whose
    # track_ids include known different people, is a likely over-merge.
    # `_orig_accepted` holds the pre-merge tracklets aligned with _groups
    # indices.
    for _gi, _g in enumerate(_multi_groups):
        members = [_orig_accepted[i] for i in _g if 0 <= i < len(_orig_accepted)]
        if not members:
            logger.info("[merge] %s:   group %d: %d fragments → 1 (empty)", video_id, _gi + 1, len(_g))
            continue
        members.sort(key=lambda t: t.observations[0].timestamp_second if t.observations else 0.0)
        starts = [m.observations[0].timestamp_second for m in members if m.observations]
        ends = [m.observations[-1].timestamp_second for m in members if m.observations]
        span_s = (max(ends) - min(starts)) if starts and ends else 0.0
        tids = ",".join(str(m.track_id) for m in members[:12])
        if len(members) > 12:
            tids += f"...(+{len(members) - 12})"
        logger.info(
            "[merge] %s:   group %d: %d fragments → 1  span=%.1fs  tids=%s",
            video_id, _gi + 1, len(_g), span_s, tids,
        )

    def _pool_avg_normalized(vecs: list) -> list:
        """Average a list of (possibly already-normalized) vectors, then L2-normalize.
        Re-normalize because mean(unit_vectors) has norm < 1 when fragments diverge."""
        valid = [v for v in vecs if v]
        if not valid:
            return []
        arr = _np.asarray(valid, dtype=_np.float32).mean(axis=0)
        n = float(_np.linalg.norm(arr))
        if n < 1e-8:
            return arr.tolist()
        return (arr / n).tolist()

    # Pool the quality-aware multi-frame SigLIP features (5-8 crops/fragment
    # @ 384px) for the DB-stored embedding, instead of the single-crop variant.
    all_siglip_embeddings = [
        _pool_avg_normalized([siglip_multi_feats[i] for i in g]) for g in _groups
    ]

    # Rebuild t_data aligned to merged tracklets — use _best_observation for rep frame
    t_data = []
    for new_idx, mt in enumerate(accepted):
        best = _best_observation(mt.observations)
        rep_bbox_float = [float(x) for x in best.bbox]
        t_frames = (
            [frame_lookup[o.frame_index] for o in mt.observations if o.frame_index in frame_lookup]
            or [sampled_frames[0].image]
        )
        t_data.append((mt, new_idx, rep_bbox_float, t_frames))

    # Build all_rep_crops for Qwen from the best-quality frame of each merged
    # tracklet (post-merge). _best_observation() scores observations across ALL
    # fragments in the merged set, so the chosen crop is the sharpest/biggest
    # of the whole identity, not just the longest fragment.
    def _make_rep_crop_384(frame, bbox):
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
        crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
        if crop.size == 0:
            crop = frame
        max_dim = max(crop.shape[0], crop.shape[1])
        top = (max_dim - crop.shape[0]) // 2
        sq = cv2.copyMakeBorder(
            crop, top, max_dim - crop.shape[0] - top,
            (max_dim - crop.shape[1]) // 2, max_dim - crop.shape[1] - (max_dim - crop.shape[1]) // 2,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return Image.fromarray(cv2.cvtColor(cv2.resize(sq, (384, 384), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))

    all_rep_crops = []
    for mt, new_idx, rep_bbox_float, t_frames in t_data:
        best = _best_observation(mt.observations)
        best_frame = frame_lookup.get(best.frame_index, t_frames[0] if t_frames else sampled_frames[0].image)
        all_rep_crops.append(_make_rep_crop_384(best_frame, best.bbox))

    # ── VLM: Qwen2.5-VL-7B trên ~25 merged tracklets ───────────────────────────
    logger.info(
        "[vlm] %s: captioning %d merged tracklets (independent_batch_start=%d, max_batch=%d, tokens/crop=%d)",
        video_id,
        len(all_rep_crops),
        VLM_BATCH_SIZE,
        VLM_BATCH_MAX_SIZE,
        VLM_BATCH_MAX_NEW_TOKENS_PER_CROP,
    )
    all_attributes: list[dict] = _caption_crops_vlm_batch(
        all_rep_crops, batch_size=VLM_BATCH_SIZE, tag=video_id,
    )
    # Confidence fields are now embedded directly in attrs (filled by
    # _attach_logit_confs) — no separate `all_attr_confs` mapping needed.

    # ── VideoMAE true-batch trên ~25 merged tracklets ────────────────────────
    # Use PER-FRAME bbox from each observation (not a frozen rep_bbox), so the
    # crop tracks the person as they move across the tracklet. Frames are aligned
    # to observations through frame_lookup; obs without a sampled frame are skipped.
    device = _get_device()
    all_actions: list = []
    model_vmae = get_model("videomae")
    proc_vmae  = get_model("videomae_processor")

    def _build_vmae_clip(mt, rep_bbox_fallback: list[float]) -> list[np.ndarray]:
        """Return 16 frames at 224×224, each cropped by that frame's own bbox."""
        pairs: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        for o in mt.observations:
            frame = frame_lookup.get(o.frame_index)
            if frame is None:
                continue
            pairs.append((frame, tuple(int(v) for v in o.bbox)))
        if not pairs:
            fb_bbox = tuple(int(v) for v in rep_bbox_fallback)
            pairs = [(sampled_frames[0].image, fb_bbox)]

        n = len(pairs)
        idxs = np.linspace(0, n - 1, min(16, n), dtype=int)
        clip: list[np.ndarray] = []
        for fi in idxs:
            f, (x1, y1, x2, y2) = pairs[fi]
            h, w = f.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = f[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else f
            if crop.size == 0:
                crop = f
            clip.append(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR))
        # Pad to 16 by repeating the last frame
        while len(clip) < 16:
            clip.append(clip[-1] if clip else np.zeros((224, 224, 3), dtype=np.uint8))
        return clip[:16]

    if model_vmae and proc_vmae and t_data:
        try:
            all_clips = [_build_vmae_clip(mt, rep_bbox) for (mt, _, rep_bbox, _) in t_data]
            inputs = proc_vmae(all_clips, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=torch.float16) if v.is_floating_point() else v.to(device)
                      for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model_vmae(**inputs)
            logits = outputs.logits.float()
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(1, dim=-1)
            id2label = getattr(model_vmae.config, "id2label", {})
            for idx, conf in zip(top_indices.squeeze(1).tolist(), top_probs.squeeze(1).tolist()):
                label  = id2label.get(idx, "")
                action = _map_kinetics_to_tracex_action(label) if label else "unknown"
                all_actions.append((action, float(conf), label))
            logger.info("[pipeline] %s: VideoMAE done — %d actions", video_id, len(all_actions))
        except Exception as exc:
            logger.warning("[pipeline] VideoMAE true-batch failed: %s — per-tracklet fallback", exc)
            all_actions = []
            for (mt, _, rep_bbox, t_frames) in t_data:
                # _run_videomae_actions returns a plain str (mapped TraceX action).
                # Wrap it into (action, conf, kinetics_label) so downstream parsing
                # in TrackletResult is uniform with the true-batch path. The
                # previous code wrapped without conversion, which kept the string
                # intact but lost confidence — accepted, since the slow path is
                # already a degraded fallback.
                action = _run_videomae_actions(t_frames, rep_bbox)
                all_actions.append((str(action), 0.0, ""))
    else:
        all_actions = [("unknown", 0.0, "")] * len(t_data)

    # Project representative bboxes to BEV coordinates
    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    _bev_inputs = [{"bbox": rep_bbox_float} for _, _, rep_bbox_float, _ in t_data]
    _bev_inputs = _project_to_bev_single(_bev_inputs, camera_id, cal_path)

    _CROPS_DIR = Path("/workspace/storage/crops")
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tracklets: list[TrackletResult] = []
    for t_idx, (lt, _, rep_bbox_float, t_frames) in enumerate(t_data):
        obs = lt.observations
        attrs = all_attributes[t_idx]
        siglip_emb = all_siglip_embeddings[t_idx] if t_idx < len(all_siglip_embeddings) else []
        action_tuple = all_actions[t_idx]
        if isinstance(action_tuple, tuple) and len(action_tuple) >= 3:
            action, action_conf, kinetics_raw = str(action_tuple[0]), float(action_tuple[1]), str(action_tuple[2])
        elif isinstance(action_tuple, tuple):
            action, action_conf, kinetics_raw = str(action_tuple[0]), float(action_tuple[1]), ""
        else:
            action, action_conf, kinetics_raw = str(action_tuple), 0.0, ""
        summary = _build_appearance_summary(attrs)

        # Save representative crop image
        crop_url = ""
        try:
            rep_crop = all_rep_crops[t_idx]
            crop_filename = f"{video_id}_{camera_id}_{t_idx}.jpg"
            crop_path = _CROPS_DIR / crop_filename
            rep_crop.save(str(crop_path), "JPEG", quality=85)
            crop_url = f"/static/crops/{crop_filename}"
        except Exception as exc:
            logger.warning("[pipeline] Failed to save crop for %s_%s_%d: %s", video_id, camera_id, t_idx, exc)

        quality = quality_results[lt.track_id]
        # Serialize per-frame observations for the bbox timeline (trace-service
        # uses these to render evidence clips with a moving bbox).
        obs_payload = [
            {
                "frame_index": int(o.frame_index),
                "timestamp_second": float(o.timestamp_second),
                "bbox": [int(v) for v in o.bbox],
                "confidence": float(getattr(o, "confidence", 0.0) or 0.0),
            }
            for o in obs
        ]
        tracklets.append(_build_tracklet_result(
            tracklet_id=f"{video_id}_{camera_id}_{t_idx}",
            video_id=video_id,
            camera_id=camera_id,
            track_idx=t_idx,
            start_time=obs[0].timestamp_second,
            end_time=obs[-1].timestamp_second,
            quality_score=quality.average_confidence,
            attrs=attrs,
            summary=summary,
            rep_bbox=rep_bbox_float,
            bev_x=_bev_inputs[t_idx].get("bev_x", 0.0),
            bev_y=_bev_inputs[t_idx].get("bev_y", 0.0),
            siglip_embedding=siglip_emb,
            action=action,
            action_confidence=action_conf,
            kinetics_label=kinetics_raw,
            crop_url=crop_url,
            observations=obs_payload,
        ))

    elapsed = time.time() - start
    logger.warning("[pipeline] %s: %d tracklets saved in %.1fs", video_id, len(tracklets), elapsed)

    return ProcessVideoResponse(
        video_id=video_id,
        camera_id=camera_id,
        tracklets=tracklets,
        total_detections=total_raw,
        processing_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# BATCH endpoint — cross-camera processing (STAGES 1-7 across all cameras)
# ---------------------------------------------------------------------------

@router.post("/batch/process", response_model=BatchProcessResponse)
def process_batch(req: BatchProcessRequest) -> BatchProcessResponse:
    """
    Batch cross-camera processing pipeline.

    Takes up to 100 videos from different cameras (same timestamp),
    runs per-video detection + BEV in parallel, then ONE MCBLT call
    across all cameras, then DINOv2 + Qwen2.5-VL + VideoMAE per unified tracklet.

    Returns unified cross-camera tracklets with global IDs.
    """
    start = time.time()
    n_videos = len(req.videos)

    if n_videos == 0:
        raise HTTPException(status_code=400, detail="Empty batch: no videos provided")

    batch_id = req.batch_id or f"batch_{int(time.time())}"

    logger.info(
        "[%s] Batch processing %d videos: %s",
        batch_id,
        n_videos,
        [v.camera_id or v.video_id for v in req.videos],
    )

    # ---- Stage 1-3: Parallel per-video detection + BEV projection ----
    per_video_results: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n_videos)) as executor:
        futures = {
            executor.submit(_process_single_video, entry): entry.video_id
            for entry in req.videos
        }
        for future in as_completed(futures):
            result = future.result()
            per_video_results.append(result)
            if result.get("error"):
                errors.append(f"{result['video_id']}: {result['error']}")

    # Collect detections by camera_id for MCBLT
    detections_by_camera: dict[str, list[dict]] = {}
    camera_stats: dict[str, int] = {}
    all_frames: dict[str, list[np.ndarray]] = {}  # video_id -> frames
    all_fps: dict[str, float] = {}

    for result in per_video_results:
        cam_id = result.get("camera_id", "unknown")
        dets = result.get("detections", [])

        if cam_id not in detections_by_camera:
            detections_by_camera[cam_id] = []

        # Attach per-video frames for later embedding extraction
        video_id = result["video_id"]
        try:
            frames, fps = _video_to_frames(
                next(e.video_path for e in req.videos if e.video_id == video_id),
                max_frames=500,
            )
            all_frames[video_id] = frames
            all_fps[video_id] = fps
        except Exception:
            all_frames[video_id] = []
            all_fps[video_id] = 30.0

        detections_by_camera[cam_id].extend(dets)
        camera_stats[cam_id] = len(dets)

    total_detections = sum(len(d) for d in detections_by_camera.values())
    logger.info(
        "[%s] Detection complete: %d total detections across %d cameras. Stats: %s",
        batch_id, total_detections, len(detections_by_camera), camera_stats,
    )

    if not detections_by_camera or total_detections == 0:
        return BatchProcessResponse(
            batch_id=batch_id,
            n_videos=n_videos,
            n_cameras=len(detections_by_camera),
            total_detections=0,
            n_tracklets=0,
            tracklets=[],
            processing_time_s=time.time() - start,
            camera_stats=camera_stats,
        )

    # ---- Stage 4: ONE MCBLT call across all cameras ----
    max_dist = 1.5
    for entry in req.videos:
        if entry.bev_max_dist:
            max_dist = entry.bev_max_dist
            break

    groups = _mcblt_associate(detections_by_camera, max_dist=max_dist)
    logger.info("[%s] MCBLT formed %d cross-camera groups", batch_id, len(groups))

    # ---- Stages 5-7: Per unified tracklet → DINOv2 + Qwen2.5-VL + VideoMAE ----
    tracklets: list[TrackletResult] = []
    tracklet_id_prefix = f"{batch_id}_tracklet"

    for group_idx, group in enumerate(groups):
        if len(group) < 1:
            continue

        # Collect frames from all contributing videos
        contributing_vids = list(set(d.get("video_id", "") for d in group))
        contributing_cams = list(set(d.get("camera_id", "") for d in group))

        # Gather all relevant frames
        all_track_frames: list[np.ndarray] = []
        all_track_bboxes: list[list[float]] = []

        for vid in contributing_vids:
            frames = all_frames.get(vid, [])
            if not frames:
                continue
            for det in group:
                if det.get("video_id") != vid:
                    continue
                fidx = det.get("frame_idx", 0)
                if 0 <= fidx < len(frames):
                    all_track_frames.append(frames[fidx])
                    all_track_bboxes.append(det["bbox"])

        if not all_track_frames:
            continue

        # Representative detection: highest score across group
        rep_det = max(group, key=lambda d: d.get("score", 0))
        rep_bbox = rep_det["bbox"]
        rep_bev_x = rep_det.get("bev_x", 0.0)
        rep_bev_y = rep_det.get("bev_y", 0.0)
        rep_cam = rep_det.get("camera_id", "unknown")
        rep_vid = rep_det.get("video_id", contributing_vids[0] if contributing_vids else "unknown")

        # VLM open-vocabulary attribute captioning
        mid_frame = all_track_frames[len(all_track_frames) // 2]
        rep_crop_cv = _extract_crop_for_vlm(mid_frame, rep_bbox)
        rep_crop_pil = (
            Image.fromarray(cv2.cvtColor(rep_crop_cv, cv2.COLOR_BGR2RGB))
            if rep_crop_cv is not None
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        )
        attributes = _caption_crop_vlm(rep_crop_pil)

        # VideoMAE action
        action = _run_videomae_actions(all_track_frames, rep_bbox)

        # Time range from group
        timestamps = [d.get("timestamp", 0) for d in group if d.get("timestamp") is not None]
        start_time = min(timestamps) if timestamps else 0.0
        end_time = max(timestamps) if timestamps else 0.0

        summary = _build_appearance_summary(attributes)
        global_tracklet_id = f"{tracklet_id_prefix}_{group_idx}"

        tracklets.append(_build_tracklet_result(
            tracklet_id=global_tracklet_id,
            video_id=rep_vid,
            camera_id=rep_cam,
            track_idx=group_idx,
            start_time=start_time,
            end_time=end_time,
            quality_score=float(rep_det.get("score", 0.5)),
            attrs=attributes,
            summary=summary,
            rep_bbox=rep_bbox,
            bev_x=rep_bev_x,
            bev_y=rep_bev_y,
            siglip_embedding=[],
            action=action if isinstance(action, str) else str(action),
            action_confidence=0.0,
            kinetics_label="",
            crop_url="",
        ))

    elapsed = time.time() - start
    logger.info(
        "[%s] Batch complete: %d tracklets (from %d detections, %d cameras) in %.1fs",
        batch_id, len(tracklets), total_detections, len(detections_by_camera), elapsed,
    )

    if errors:
        logger.warning("[%s] %d video(s) had errors: %s", batch_id, len(errors), errors)

    return BatchProcessResponse(
        batch_id=batch_id,
        n_videos=n_videos,
        n_cameras=len(detections_by_camera),
        total_detections=total_detections,
        n_tracklets=len(tracklets),
        tracklets=tracklets,
        processing_time_s=elapsed,
        camera_stats=camera_stats,
    )
