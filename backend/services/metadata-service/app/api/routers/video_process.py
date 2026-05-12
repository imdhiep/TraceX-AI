"""Video processing pipeline for metadata-service.

Per-video pipeline:
  1. Sample frames
  2. RT-DETR R50 person detection
  3. Track + merge per-video person fragments
  4. SigLIP2 appearance embeddings (multi-frame pool-avg) — Stage 5
  5. Fragment merge (SigLIP2 cosine sim) — Stage 6
  6. Qwen2-VL captioning + VideoMAE action on merged tracklets — Stage 7

Batch pipeline:
  - Run the per-video pipeline in parallel
  - Aggregate tracklets across videos without BEV/cross-camera raw association
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


DEFAULT_SAMPLE_INTERVAL = 15
DEFAULT_MIN_BBOX_AREA = 400
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))
VLM_BATCH_SIZE = _get_positive_env_int("VLM_BATCH_SIZE", 4)
VLM_BATCH_MAX_SIZE = _get_positive_env_int("VLM_BATCH_MAX_SIZE", 8)
VLM_BATCH_GROW_STEP = _get_positive_env_int("VLM_BATCH_GROW_STEP", 1)
VLM_BATCH_STABLE_STEPS = _get_positive_env_int("VLM_BATCH_STABLE_STEPS", 3)
VLM_BATCH_MAX_NEW_TOKENS_PER_CROP = _get_positive_env_int(
    "VLM_BATCH_MAX_NEW_TOKENS_PER_CROP", 512
)
FRAGMENT_MERGE_SIM_THRESHOLD = _get_env_float("FRAGMENT_MERGE_SIM_THRESHOLD", 0.85)
FRAGMENT_MERGE_MAX_GAP_SECONDS = _get_env_float("FRAGMENT_MERGE_MAX_GAP_SECONDS", 60.0)
FRAGMENT_MERGE_COMPONENT_MARGIN = _get_env_float("FRAGMENT_MERGE_COMPONENT_MARGIN", 0.03)


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
# Stage 2: Person Detection (RT-DETR)
# ---------------------------------------------------------------------------


def _detect_persons_rtdetr(
    frames: list[np.ndarray],
    threshold: float = 0.4,
    batch_size: int = 128,
) -> list[list[dict]] | None:
    """RT-DETR R50 person detection — primary detector.
    Returns None if model not loaded.
    batch_size=128 on A100 80GB (256 causes OOM).
    CPU preprocessing is pipelined with GPU inference via ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor
    model = get_model("rtdetr")
    processor = get_model("rtdetr_processor")
    if model is None or processor is None:
        return None

    person_ids: set = get_model("rtdetr_person_ids") or {0, 1}
    device = _get_device()
    dtype = torch.float16

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    def _preprocess(batch: list) -> tuple:
        sizes = [(f.shape[0], f.shape[1]) for f in batch]
        pil_imgs = [Image.fromarray(f[:, :, ::-1]) for f in batch]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        return inputs, sizes

    all_dets: list[list[dict]] = []
    autocast_ctx = torch.autocast("cuda", dtype=dtype)

    def _infer_one(batch_frames: list, inputs_raw, sizes, depth: int = 0) -> list[list[dict]]:
        """Run inference on one (sub-)batch. On CUDA OOM, split in half and recurse."""
        try:
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                      for k, v in inputs_raw.items()}
            with torch.no_grad(), autocast_ctx:
                outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs, threshold=threshold,
                target_sizes=torch.tensor(sizes, device=device),
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            n = len(batch_frames)
            if depth >= 3 or n <= 4:
                logger.error("[rtdetr] OOM at batch_size=%d depth=%d, dropping: %s", n, depth, exc)
                return [[] for _ in batch_frames]
            mid = n // 2
            logger.warning("[rtdetr] OOM at batch_size=%d depth=%d, splitting to %d/%d",
                           n, depth, mid, n - mid)
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

    with ThreadPoolExecutor(max_workers=2) as ex:
        # Submit first batch preprocessing
        futures = [ex.submit(_preprocess, b) for b in batches[:2]]

        for idx, batch in enumerate(batches):
            # Prefetch next+1 batch while current is on GPU
            if idx + 2 < len(batches):
                futures.append(ex.submit(_preprocess, batches[idx + 2]))

            inputs_raw, sizes = futures[idx].result()
            all_dets.extend(_infer_one(batch, inputs_raw, sizes))

    return all_dets


def _sanitize_dets_inplace(
    dets_per_frame: list[list[dict]],
    frames: list[np.ndarray],
    min_w: int = 6,
    min_h: int = 12,
) -> list[list[dict]]:
    """Clip bboxes into frame bounds and drop bboxes smaller than min_w/min_h.

    Conservative: only removes clearly broken bboxes (out-of-frame or pixel-noise sized).
    No aspect-ratio or score-based filter — those risk dropping crouching/sitting persons.
    Per-frame numpy vectorization keeps overhead negligible.
    """
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
        keep = ((arr[:, 2] - arr[:, 0]) >= min_w) & ((arr[:, 3] - arr[:, 1]) >= min_h)
        if bool(keep.all()):
            for d, row in zip(dets, arr):
                d["bbox"] = row.tolist()
            out.append(dets)
            continue
        kept = []
        for d, row, k in zip(dets, arr, keep):
            if not k:
                continue
            d["bbox"] = row.tolist()
            kept.append(d)
        out.append(kept)
    return out


def _detect_persons_batch(frames: list[np.ndarray], threshold: float = 0.25) -> list[list[dict]]:
    """Person detection: RT-DETR primary (fast), GDINO fallback."""
    rtdetr_threshold = float(os.getenv("RTDETR_PERSON_THRESHOLD", str(max(threshold, 0.4))))
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
# Per-video helpers
# ---------------------------------------------------------------------------

def _process_single_video(entry: BatchVideoEntry) -> ProcessVideoResponse:
    """Run the full per-video pipeline for one batch entry."""
    return _process_video_sync(
        entry.video_path,
        entry.video_id,
        entry.camera_id,
        entry.sample_interval,
    )


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
    return {
        "gender": "unknown", "age_range": "unknown",
        "upper_clothing_desc": None, "upper_clothing_color": "unknown",
        "upper_clothing_type": "unknown", "upper_clothing_conf": None,
        "lower_clothing_desc": None, "lower_clothing_color": "unknown",
        "lower_clothing_type": "unknown", "lower_clothing_conf": None,
        "shoes_desc": None, "shoes_color": "unknown", "shoes_conf": None,
        "bag_presence": "unknown", "bag_type": "unknown",
        "bag_desc": None, "bag_conf": None,
        "hat_presence": "unknown", "hat_color": "unknown",
        "hat_type": "unknown", "hat_desc": None, "hat_conf": None,
        "is_wearing_mask": "unknown", "mask_conf": None,
        "hair_style": "unknown", "hair_color": "unknown", "hair_conf": None,
        "appearance_summary": "person",
        "bag": "unknown", "hat": "unknown",
    }


_VLM_PROMPT = """You are analyzing a person crop from a surveillance camera.
Describe this person's visible appearance accurately using ALL visible cues.

GENDER INFERENCE RULES (important):
- Infer gender from ANY combination of: clothing style (dress/skirt → woman), hair length, body silhouette, accessories, overall appearance
- Use "man" or "woman" whenever you can make a reasonable inference — do NOT default to "unknown" if there are visible cues
- Only use "unknown" if the person is completely obscured, facing away with no distinguishing features, or truly ambiguous
- A person in a dress/skirt: "woman". A person in a suit/tie: likely "man". Long hair + feminine clothing: "woman". Etc.

AGE INFERENCE RULES:
- Estimate from body size, posture, hair color, clothing style
- Use "young_adult" (18-35), "middle_aged" (35-55), "elderly" (55+), "teenager" (13-17), "child" (<13)
- Prefer a guess with lower confidence over "unknown"

Return ONLY a valid JSON object with these exact fields:

{
  "gender": "man or woman or unknown",
  "gender_conf": 0.0,
  "age_range": "child or teenager or young_adult or middle_aged or elderly or unknown",
  "age_range_conf": 0.0,
  "upper_clothing_desc": "free text e.g. black suit jacket",
  "upper_clothing_color": "dominant color or unknown",
  "upper_clothing_type": "e.g. suit jacket or hoodie or t-shirt or vest or unknown",
  "upper_clothing_conf": 0.0,
  "lower_clothing_desc": "free text e.g. black formal trousers",
  "lower_clothing_color": "dominant color or unknown",
  "lower_clothing_type": "e.g. jeans or formal trousers or shorts or skirt or unknown",
  "lower_clothing_conf": 0.0,
  "shoes_desc": "free text or unknown",
  "shoes_color": "color or unknown",
  "shoes_conf": 0.0,
  "bag_presence": "yes or no or unknown",
  "bag_type": "e.g. backpack or handbag or suitcase or none or unknown",
  "bag_desc": "free text or none",
  "bag_conf": 0.0,
  "hat_presence": "yes or no or unknown",
  "hat_color": "color or none or unknown",
  "hat_type": "e.g. cap or hat or helmet or hood or none or unknown",
  "hat_desc": "free text or none",
  "hat_conf": 0.0,
  "is_wearing_mask": "yes or no or unknown",
  "mask_conf": 0.0,
  "hair_style": "short or long or ponytail or tied or bald or unknown",
  "hair_color": "color or unknown",
  "hair_conf": 0.0,
  "appearance_summary": "one concise sentence describing the person"
}

IMPORTANT INSTRUCTIONS:
- Set gender to "unknown" ONLY if the face/head is not visible or ambiguous. If head/upper body is visible, infer gender from features.
- Set age_range to "unknown" ONLY if the face is not visible. Body shape and clothing can indicate approximate age.
- For bag/hat/mask: if you can see the upper body clearly and the item is NOT present, respond "no" (not "unknown").
- Hair style/color: respond "unknown" only if head/top of person is not in the frame.
- Replace 0.0 placeholders with your actual confidence (0.0-1.0). If you provide a value OTHER than "unknown" for a field, you MUST provide conf > 0.0."""

_VLM_BATCH_PROMPT_TEMPLATE = """You are analyzing {n} person crops from surveillance cameras.
The images above show persons labeled (1) to ({n}) in order.
Describe each person's visible appearance using ALL visible cues. Use free text for clothing descriptions.

GENDER: infer from clothing style (dress/skirt→woman), hair, body silhouette — do NOT default to "unknown" if cues are visible.
AGE: estimate from body, posture, hair — prefer a guess with low confidence over "unknown".

Return ONLY a valid JSON array with exactly {n} objects in order (index 0 = person 1).
Each object must have the same fields as below:

{{
  "gender": "man or woman or unknown",
  "gender_conf": 0.0,
  "age_range": "child or teenager or young_adult or middle_aged or elderly or unknown",
  "age_range_conf": 0.0,
  "upper_clothing_desc": "free text",
  "upper_clothing_color": "dominant color or unknown",
  "upper_clothing_type": "e.g. suit jacket or hoodie or t-shirt or unknown",
  "upper_clothing_conf": 0.0,
  "lower_clothing_desc": "free text",
  "lower_clothing_color": "dominant color or unknown",
  "lower_clothing_type": "e.g. jeans or formal trousers or shorts or unknown",
  "lower_clothing_conf": 0.0,
  "shoes_desc": "free text or unknown",
  "shoes_color": "color or unknown",
  "shoes_conf": 0.0,
  "bag_presence": "yes or no or unknown",
  "bag_type": "backpack or handbag or none or unknown",
  "bag_desc": "free text or none",
  "bag_conf": 0.0,
  "hat_presence": "yes or no or unknown",
  "hat_color": "color or none or unknown",
  "hat_type": "cap or hat or helmet or none or unknown",
  "hat_desc": "free text or none",
  "hat_conf": 0.0,
  "is_wearing_mask": "yes or no or unknown",
  "mask_conf": 0.0,
  "hair_style": "short or long or ponytail or tied or bald or unknown",
  "hair_color": "color or unknown",
  "hair_conf": 0.0,
  "appearance_summary": "one concise sentence describing the person"
}}

IMPORTANT:
- Set gender to "unknown" ONLY if face/head is not visible. If upper body is visible, infer from features.
- For bag/hat/mask: if you can see the upper body clearly and the item is NOT present, respond "no" (not "unknown").
- Hair: respond "unknown" only if top of person is not in frame.
- If you provide a value OTHER than "unknown" for a field, you MUST provide conf > 0.0.
Return only the JSON array, no surrounding text."""


_GENDER_NORM   = {"male": "man", "man": "man", "female": "woman", "woman": "woman"}
_AGE_NORM      = {
    "young adult": "young_adult", "young_adult": "young_adult",
    "middle aged": "middle_aged", "middle-aged": "middle_aged", "middle_aged": "middle_aged",
    "teen": "teenager", "teenager": "teenager",
    "elder": "elderly", "elderly": "elderly",
    "child": "child",
}
_PRESENCE_NORM = {"yes": "yes", "no": "no", "true": "yes", "false": "no", "none": "no"}


def _parse_vlm_attrs(parsed: dict) -> dict:
    """Normalise a raw VLM JSON dict into the canonical attrs dict.

    Post-processes confidence scores and repairs missing fields from appearance_summary.
    """
    def _s(key: str, fallback: str = "unknown") -> str:
        v = parsed.get(key)
        return str(v).strip().lower() if v not in (None, "", "null") else fallback

    def _f(key: str) -> float | None:
        try:
            val = float(parsed[key])
            return val if val > 0.0 else None  # 0.0 = VLM placeholder, treat as missing
        except (KeyError, TypeError, ValueError):
            return None

    def _norm(val: str, mapping: dict) -> str:
        return mapping.get(val.lower().strip(), val) if val else "unknown"

    summary = (parsed.get("appearance_summary") or "").lower()

    # Repair presence fields from summary when VLM said "unknown"
    bag_raw = _s("bag_presence")
    bag_pres = _norm(bag_raw, _PRESENCE_NORM)
    if bag_pres == "unknown" and "bag" not in summary and "backpack" not in summary and "handbag" not in summary:
        bag_pres = "no"

    hat_raw = _s("hat_presence")
    hat_pres = _norm(hat_raw, _PRESENCE_NORM)
    if hat_pres == "unknown" and "hat" not in summary and "cap" not in summary and "helmet" not in summary:
        hat_pres = "no"

    mask_raw = _s("is_wearing_mask")
    mask_pres = _norm(mask_raw, _PRESENCE_NORM)
    if mask_pres == "unknown" and "mask" not in summary:
        mask_pres = "no"

    gender_val = _norm(_s("gender"), _GENDER_NORM)
    gender_conf = _f("gender_conf")
    if gender_val == "unknown":
        gender_conf = None

    age_val = _norm(_s("age_range"), _AGE_NORM)
    age_conf = _f("age_range_conf")
    if age_val == "unknown":
        age_conf = None

    return {
        "gender":               gender_val,
        "gender_conf":          gender_conf,
        "age_range":            age_val,
        "age_range_conf":       age_conf,
        "upper_clothing_desc":  parsed.get("upper_clothing_desc"),
        "upper_clothing_color": _s("upper_clothing_color"),
        "upper_clothing_type":  _s("upper_clothing_type"),
        "upper_clothing_conf":  _f("upper_clothing_conf"),
        "lower_clothing_desc":  parsed.get("lower_clothing_desc"),
        "lower_clothing_color": _s("lower_clothing_color"),
        "lower_clothing_type":  _s("lower_clothing_type"),
        "lower_clothing_conf":  _f("lower_clothing_conf"),
        "shoes_desc":           parsed.get("shoes_desc"),
        "shoes_color":          _s("shoes_color"),
        "shoes_conf":           _f("shoes_conf"),
        "bag_presence":         bag_pres,
        "bag_type":             _s("bag_type"),
        "bag_desc":             parsed.get("bag_desc"),
        "bag_conf":             _f("bag_conf"),
        "hat_presence":         hat_pres,
        "hat_color":            _s("hat_color"),
        "hat_type":             _s("hat_type"),
        "hat_desc":             parsed.get("hat_desc"),
        "hat_conf":             _f("hat_conf"),
        "is_wearing_mask":      mask_pres,
        "mask_conf":            _f("mask_conf"),
        "hair_style":           _s("hair_style"),
        "hair_color":           _s("hair_color"),
        "hair_conf":            _f("hair_conf"),
        "appearance_summary":    parsed.get("appearance_summary") or "person",
        "bag": "no_bag"      if bag_pres == "no"  else ("carrying_bag" if bag_pres == "yes" else "unknown"),
        "hat": "no_hat"      if hat_pres == "no"  else ("wearing_hat"  if hat_pres == "yes" else "unknown"),
    }


def _extract_json_array(raw: str) -> list[dict]:
    """Extract the first complete JSON array from model output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1).strip()

    start = cleaned.find("[")
    if start == -1:
        raise ValueError(f"JSON array not found: {cleaned[:200]}")

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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                payload = cleaned[start:idx + 1]
                parsed = json.loads(payload)
                if not isinstance(parsed, list):
                    raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
                return parsed

    raise ValueError(f"JSON array incomplete: {cleaned[:200]}")


def _caption_crop_vlm(crop: "Image.Image") -> dict:
    """Generate open-vocabulary appearance attributes via Qwen2-VL-7B-Instruct (single crop)."""
    model = get_model("qwen2vl")
    processor = get_model("qwen2vl_processor")
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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=VLM_BATCH_MAX_NEW_TOKENS_PER_CROP,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("[vlm] JSON not found in output: %s", raw[:200])
            return _default_attributes()

        return _parse_vlm_attrs(json.loads(json_match.group()))

    except Exception as exc:
        logger.warning("[vlm] _caption_crop_vlm failed: %s", exc)
        return _default_attributes()


def _vlm_progress_bar(done: int, total: int, width: int = 25) -> str:
    filled = int(width * done / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * done / total) if total else 0
    return f"[{bar}] {done}/{total} ({pct}%)"


def _caption_crops_vlm_batch(crops: list, batch_size: int = VLM_BATCH_SIZE, tag: str = "") -> list:
    """Adaptive Qwen2-VL captioning with OOM-aware batch backoff."""
    model = get_model("qwen2vl")
    processor = get_model("qwen2vl_processor")
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
                "[vlm] %s %s  %.0fs elapsed  ETA %.0fs  batch=%d current=%d start=%d max=%d tokens/crop=%d",
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
        try:
            prompt = _VLM_BATCH_PROMPT_TEMPLATE.format(n=n)
            content = [{"type": "image", "image": c} for c in batch]
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=batch, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=VLM_BATCH_MAX_NEW_TOKENS_PER_CROP * n,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            input_len = inputs["input_ids"].shape[1]
            raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

            parsed_list = _extract_json_array(raw)
            if not isinstance(parsed_list, list) or len(parsed_list) != n:
                raise ValueError(f"Expected {n} objects, got {len(parsed_list) if isinstance(parsed_list, list) else type(parsed_list)}")

            logger.debug("[vlm] batch(%d) OK at offset %d", n, i)
            for obj in parsed_list:
                results.append(_parse_vlm_attrs(obj))
            i += n
            stable_windows += 1
            _log_progress(len(results), n)
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
    upper = (attrs.get("upper_clothing_desc") or "").strip()
    lower = (attrs.get("lower_clothing_desc") or "").strip()
    if gender and gender != "unknown":
        parts.append(gender)
    if upper and upper != "unknown":
        parts.append(upper)
    if lower and lower != "unknown":
        parts.append(lower)
    return " ".join(parts) or "person"


def _extract_crop_for_vlm(frame: np.ndarray, bbox: list[float], margin_pct: float = 0.15) -> Optional[np.ndarray]:
    """Extract a square-padded 384×384 crop from frame for VLM input.

    Args:
        frame: BGR frame from video.
        bbox: [x1, y1, x2, y2] detector bounding box.
        margin_pct: Expand bbox by this fraction of its width/height to capture context (head/feet).
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_pct)
    my = int(bh * margin_pct)
    x1_e = max(0, int(x1) - mx)
    y1_e = max(0, int(y1) - my)
    x2_e = min(frame.shape[1], int(x2) + mx)
    y2_e = min(frame.shape[0], int(y2) + my)
    if x2_e <= x1_e or y2_e <= y1_e:
        return None
    crop = frame[y1_e:y2_e, x1_e:x2_e]
    if crop.size == 0:
        return None
    max_dim = max(crop.shape[0], crop.shape[1])
    top = (max_dim - crop.shape[0]) // 2
    bottom = max_dim - crop.shape[0] - top
    left = (max_dim - crop.shape[1]) // 2
    right = max_dim - crop.shape[1] - left
    padded = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)  # gray — less confusing for VLM than black
    )
    return cv2.resize(padded, (384, 384), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Per-video endpoint (single video, existing behaviour)
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessVideoResponse)
def process_video(req: ProcessVideoRequest) -> ProcessVideoResponse:
    """Process a single video (single-camera tracking pipeline)."""
    video_path = req.video_path
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    return _process_video_sync(
        video_path,
        req.video_id,
        req.camera_id,
        req.sample_interval or DEFAULT_SAMPLE_INTERVAL,
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
) -> tuple[list, list, list]:
    """
    Batch SigLIP2 embeddings only. DINOv2 removed — SigLIP2 is the sole embedding model.
    Returns: (siglip_multi_feats, all_rep_crops, all_siglip_embeddings)
      siglip_multi_feats: multi-frame pool-avg per fragment (for fragment merge)
      all_rep_crops: PIL Images 384×384 from best-quality frame (for Qwen/storage)
      all_siglip_embeddings: single-crop embedding per fragment (for DB)
    """
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
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return cv2.resize(pad, (size, size), interpolation=cv2.INTER_LINEAR)

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
    siglip_multi_feats = [[] for _ in t_data]
    if model_sip and proc_sip and t_data:
        try:
            siglip_crops, siglip_slices = [], []
            for lt, t_idx, rep_bbox, t_frames in t_data:
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
    presampled_frames=None,
) -> ProcessVideoResponse:
    """
    Sync video processing pipeline using BodyPartAdaptiveTracker + 4fps sampling.
    presampled_frames: pre-decoded frames from background thread (skips Stage 1).
    """
    import time
    from .tracking_pipeline import (
        VideoFrameSampler, BodyPartAdaptiveTracker, TrackletQualityScorer,
        TrackletFragmentMerger, FrameDetection, _crop_from_bbox,
    )
    start = time.time()
    camera_id = camera_id or "Camera_0000"

    # Stage 1: Sample frames at 4fps (skip if pre-decoded externally)
    if presampled_frames is not None:
        sampled_frames = presampled_frames
    else:
        sampler = VideoFrameSampler(sample_fps=4)
        try:
            sampled_frames = sampler.sample(video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read video: {e}")

    if not sampled_frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    logger.warning("[pipeline] %s: %d frames sampled at 4fps", video_id, len(sampled_frames))

    # Stage 2: Batch detect — RT-DETR
    from .tracking_pipeline import _crop_from_bbox as _tcrop
    t_det_start = time.time()
    logger.info("[pipeline] %s: running RT-DETR detection on %d frames...", video_id, len(sampled_frames))
    all_batch_dets = _detect_persons_batch([sf.image for sf in sampled_frames], threshold=0.25)
    logger.warning("[pipeline] %s: RT-DETR done in %.1fs", video_id, time.time() - t_det_start)

    detections_by_frame: dict[int, list[FrameDetection]] = {}
    total_raw = 0
    for sf, raw_dets in zip(sampled_frames, all_batch_dets):
        frame_dets: list[FrameDetection] = []
        for d in raw_dets:
            bbox = tuple(int(x) for x in d["bbox"])
            crop = _tcrop(sf.image, bbox)
            frame_dets.append(FrameDetection(
                frame_index=sf.frame_index,
                timestamp_second=sf.timestamp_second,
                bbox=bbox,
                confidence=float(d["score"]),
                laplacian_score=sf.laplacian_score,
                crop_bgr=crop,
            ))
        if frame_dets:
            detections_by_frame[sf.frame_index] = frame_dets
            total_raw += len(frame_dets)

    logger.warning("[pipeline] %s: %d detections across %d frames", video_id, total_raw, len(detections_by_frame))

    import torch as _torch

    if not detections_by_frame:
        logger.warning("[pipeline] %s: no persons detected", video_id)
        return ProcessVideoResponse(
            video_id=video_id, camera_id=camera_id,
            tracklets=[], total_detections=0,
            processing_time_s=time.time() - start,
        )

    # Stage 3: Track with BodyPartAdaptiveTracker
    tracker = BodyPartAdaptiveTracker(
        track_thresh=0.30,
        low_thresh=0.10,
        new_track_threshold=0.30,
        min_track_frames=2,
        min_track_density=0.03,
    )
    local_tracklets = tracker.track(video_id, camera_id, detections_by_frame)
    logger.warning("[pipeline] %s: %d raw tracklets from tracker", video_id, len(local_tracklets))

    # Stage 4: Quality filter
    scorer = TrackletQualityScorer(
        min_confidence=0.25,
        min_frames=2,
        min_density=0.03,
        min_duration_s=0.25,
        min_laplacian=5.0,
    )
    quality_results = {t.track_id: scorer.score(t) for t in local_tracklets}
    accepted = [t for t in local_tracklets if quality_results[t.track_id].accepted]
    rejected_reasons = {}
    for t in local_tracklets:
        q = quality_results[t.track_id]
        if not q.accepted:
            r = q.rejection_reason or "unknown"
            rejected_reasons[r] = rejected_reasons.get(r, 0) + 1
    logger.warning("[pipeline] %s: %d accepted, %d rejected %s",
                   video_id, len(accepted), len(local_tracklets) - len(accepted), rejected_reasons)

    # Stage 5: SigLIP2 embeddings — BEFORE fragment merge
    frame_lookup = {sf.frame_index: sf.image for sf in sampled_frames}

    t_data_raw = []
    for t_idx, lt in enumerate(accepted):
        best = _best_observation(lt.observations)
        rep_bbox_float = [float(x) for x in best.bbox]
        t_frames = [frame_lookup[o.frame_index] for o in lt.observations if o.frame_index in frame_lookup] or [sampled_frames[0].image]
        t_data.append((lt, t_idx, rep_bbox_float, t_frames))

    siglip_multi_feats, all_rep_crops, all_siglip_embeddings = _batch_siglip_embeddings(t_data, video_id)

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
    )
    _orig_accepted = list(accepted)
    _t_merge = time.perf_counter()
    accepted, _groups = _merger.merge(list(accepted), siglip_multi_feats)
    _merge_elapsed = time.perf_counter() - _t_merge

    _n_merged = sum(len(g) - 1 for g in _groups if len(g) > 1)
    if _n_merged > 0:
        logger.info(
            "[pipeline] %s: fragment merger joined %d fragment(s) → %d merged tracklets",
            video_id, _n_merged, len(accepted),
        )

    # Pool-avg SigLIP embeddings across all fragments in each group
    def _pool_avg(vecs: list) -> list:
        valid = [v for v in vecs if v]
        if not valid:
            return []
        return _np.array(valid, dtype=_np.float32).mean(axis=0).tolist()

    def _richest(g: list[int]) -> int:
        return max(g, key=lambda i: len(_orig_accepted[i].observations))

    all_siglip_embeddings = [_pool_avg([all_siglip_embeddings[i] for i in g]) for g in _groups]
    all_rep_crops         = [all_rep_crops[_richest(g)]          for g in _groups]

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

    # Rebuild all_rep_crops for Qwen from best-quality frame of each merged tracklet.
    # _richest() above picks the fragment with most obs, but _best_observation() within
    # the merged tracklet's observations is the correct frame for appearance captioning.
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

    # ── VLM: Qwen2-VL-7B trên ~25 merged tracklets ───────────────────────────
    logger.info(
        "[vlm] %s: captioning %d merged tracklets (start_batch=%d, max_batch=%d, tokens/crop=%d)",
        video_id,
        len(all_rep_crops),
        VLM_BATCH_SIZE,
        VLM_BATCH_MAX_SIZE,
        VLM_BATCH_MAX_NEW_TOKENS_PER_CROP,
    )
    all_attributes: list[dict] = _caption_crops_vlm_batch(
        all_rep_crops, batch_size=VLM_BATCH_SIZE, tag=video_id,
    )
    all_attr_confs: list[dict] = []
    for attrs in all_attributes:
        all_attr_confs.append({
            "gender_conf":       attrs.get("gender_conf"),
            "age_range_conf":    attrs.get("age_range_conf"),
            "top_color_conf":    attrs.get("upper_clothing_conf"),
            "bottom_color_conf": attrs.get("lower_clothing_conf"),
            "shoes_conf":        attrs.get("shoes_conf"),
            "accessory_conf":    max(
                attrs.get("bag_conf") or 0.0,
                attrs.get("hat_conf") or 0.0,
            ) or None,
            "hat_color_conf":    attrs.get("hat_conf"),
            "bag_type_conf":     attrs.get("bag_conf"),
            "mask_conf":         attrs.get("mask_conf"),
            "hair_style_conf":   attrs.get("hair_conf"),
            "hair_color_conf":   attrs.get("hair_conf"),
        })

    # ── VideoMAE true-batch trên ~25 merged tracklets ────────────────────────
    device = _get_device()
    all_actions: list = []
    model_vmae = get_model("videomae")
    proc_vmae  = get_model("videomae_processor")
    if model_vmae and proc_vmae and t_data:
        try:
            all_clips: list[list[np.ndarray]] = []
            for lt, t_idx, rep_bbox, t_frames in t_data:
                x1, y1, x2, y2 = (max(0, int(v)) for v in rep_bbox)
                n_f = len(t_frames)
                idx_list = np.linspace(0, n_f - 1, min(16, n_f), dtype=int)
                frames_224 = []
                for fi in idx_list:
                    f = t_frames[fi]
                    h, w = f.shape[:2]
                    x2c, y2c = min(w, x2), min(h, y2)
                    crop = f[y1:y2c, x1:x2c] if x2c > x1 and y2c > y1 else f
                    frames_224.append(cv2.resize(
                        crop if crop.size > 0 else f, (224, 224), interpolation=cv2.INTER_LINEAR))
                while len(frames_224) < 16:
                    frames_224.append(frames_224[-1] if frames_224 else np.zeros((224, 224, 3), dtype=np.uint8))
                all_clips.append(frames_224[:16])
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
            logger.warning("[pipeline] VideoMAE true-batch failed: %s — fallback", exc)
            all_actions = [(_run_videomae_actions(t[3], t[2]), 0.0, "") for t in t_data]
    else:
        all_actions = [("unknown", 0.0, "")] * len(t_data)

    # Project representative bboxes to BEV coordinates
    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    _bev_inputs = [{"bbox": rep_bbox_float} for _, _, rep_bbox_float, _ in t_data]
    _bev_inputs = _project_to_bev_single(_bev_inputs, camera_id, cal_path)

    _CROPS_DIR = Path("/workspace/storage/crops")
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tracklets: list[TrackletResult] = []
    for t_idx, (lt, _, rep_bbox_float, t_frames) in enumerate(t_data_merged):
        obs = lt.observations
        attributes = all_attributes[t_idx]
        siglip_emb = all_siglip_embeddings[t_idx] if t_idx < len(all_siglip_embeddings) else []
        action_tuple = all_actions[t_idx]
        if isinstance(action_tuple, tuple) and len(action_tuple) >= 3:
            action, action_conf, kinetics_raw = str(action_tuple[0]), float(action_tuple[1]), str(action_tuple[2])
        elif isinstance(action_tuple, tuple):
            action, action_conf, kinetics_raw = str(action_tuple[0]), float(action_tuple[1]), ""
        else:
            action, action_conf, kinetics_raw = str(action_tuple), 0.0, ""
        summary = _build_appearance_summary(attributes)
        attr_conf = all_attr_confs[t_idx]

        crop_url = ""
        try:
            rep_crop = rep_crops_merged[t_idx]
            crop_filename = f"{video_id}_{camera_id}_{t_idx}.jpg"
            crop_path = _CROPS_DIR / crop_filename
            rep_crop.save(str(crop_path), "JPEG", quality=85)
            crop_url = f"/static/crops/{crop_filename}"
        except Exception as exc:
            logger.warning("[pipeline] Failed to save crop for %s_%s_%d: %s", video_id, camera_id, t_idx, exc)

        quality = quality_results[lt.track_id]
        tracklets.append(TrackletResult(
            tracklet_id=f"{video_id}_{camera_id}_{t_idx}",
            video_id=video_id,
            camera_id=camera_id,
            track_id=t_idx,
            start_time=obs[0].timestamp_second,
            end_time=obs[-1].timestamp_second,
            quality_score=quality.average_confidence,
            gender=attributes.get("gender", "unknown"),
            age_range=attributes.get("age_range", "unknown"),
            upper_clothing_color=attributes.get("upper_clothing_color", "unknown"),
            lower_clothing_color=attributes.get("lower_clothing_color", "unknown"),
            shoes_color=attributes.get("shoes_color", "unknown"),
            hat_color=attributes.get("hat_color", "unknown"),
            bag_type=attributes.get("bag_type", "unknown"),
            is_wearing_mask=attributes.get("is_wearing_mask", "unknown"),
            hair_style=attributes.get("hair_style", "unknown"),
            hair_color=attributes.get("hair_color", "unknown"),
            appearance_summary=summary,
            crop_url=crop_url,
            upper_clothing_desc=attributes.get("upper_clothing_desc"),
            upper_clothing_color=attributes.get("upper_clothing_color"),
            upper_clothing_type=attributes.get("upper_clothing_type"),
            upper_clothing_conf=attributes.get("upper_clothing_conf"),
            lower_clothing_desc=attributes.get("lower_clothing_desc"),
            lower_clothing_color=attributes.get("lower_clothing_color"),
            lower_clothing_type=attributes.get("lower_clothing_type"),
            lower_clothing_conf=attributes.get("lower_clothing_conf"),
            shoes_desc=attributes.get("shoes_desc"),
            shoes_type=attributes.get("shoes_type"),
            bag_presence=attributes.get("bag_presence"),
            bag_desc=attributes.get("bag_desc"),
            bag_conf=attributes.get("bag_conf"),
            hat_presence=attributes.get("hat_presence"),
            hat_type=attributes.get("hat_type"),
            hat_desc=attributes.get("hat_desc"),
            hat_conf=attributes.get("hat_conf"),
            representative_bbox=[int(x) for x in rep_bbox_float],
            bev_x=_bev_inputs[t_idx].get("bev_x", 0.0),
            bev_y=_bev_inputs[t_idx].get("bev_y", 0.0),
            embedding_vector=[],
            siglip_embedding=siglip_emb or [],
            action=action,
            action_confidence=action_conf,
            kinetics_label=kinetics_raw,
            occlusion_score=0.0,
            gender_conf=attr_conf.get("gender_conf"),
            upper_clothing_conf=attr_conf.get("upper_clothing_conf"),
            lower_clothing_conf=attr_conf.get("lower_clothing_conf"),
            shoes_conf=attr_conf.get("shoes_conf"),
            accessory_conf=attr_conf.get("accessory_conf"),
            age_range_conf=attr_conf.get("age_range_conf"),
            hat_color_conf=attr_conf.get("hat_color_conf"),
            bag_type_conf=attr_conf.get("bag_type_conf"),
            mask_conf=attr_conf.get("mask_conf"),
            hair_style_conf=attr_conf.get("hair_style_conf"),
            hair_color_conf=attr_conf.get("hair_color_conf"),
            contributing_cameras=[camera_id],
            contributing_video_ids=[video_id],
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
# BATCH endpoint — aggregate per-video results
# ---------------------------------------------------------------------------

@router.post("/batch/process", response_model=BatchProcessResponse)
def process_batch(req: BatchProcessRequest) -> BatchProcessResponse:
    """
    Batch processing pipeline.

    Takes up to 100 videos from different cameras (same timestamp),
    runs the full per-video pipeline in parallel, then aggregates the
    resulting tracklets into a single batch response.
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

    per_video_results: list[ProcessVideoResponse] = []
    errors: list[str] = []
    camera_stats: dict[str, int] = {
        (entry.camera_id or f"cam_{entry.video_id.split('_')[0]}"): 0
        for entry in req.videos
    }

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n_videos)) as executor:
        futures = {
            executor.submit(_process_single_video, entry): entry.video_id
            for entry in req.videos
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except HTTPException as exc:
                errors.append(f"{futures[future]}: {exc.detail}")
                continue
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
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

        tracklets.append(TrackletResult(
            tracklet_id=global_tracklet_id,
            video_id=rep_vid,
            camera_id=rep_cam,
            track_id=group_idx,
            start_time=start_time,
            end_time=end_time,
            quality_score=float(rep_det.get("score", 0.5)),
            gender=attributes.get("gender", "unknown"),
            age_range=attributes.get("age_range", "unknown"),
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color=attributes.get("shoes_color", "unknown"),
            hat_color=attributes.get("hat_color", "unknown"),
            bag_type=attributes.get("bag_type", "unknown"),
            is_wearing_mask=attributes.get("is_wearing_mask", "unknown"),
            hair_style=attributes.get("hair_style", "unknown"),
            hair_color=attributes.get("hair_color", "unknown"),
            appearance_summary=summary,
            upper_clothing_desc=attributes.get("upper_clothing_desc"),
            upper_clothing_color=attributes.get("upper_clothing_color"),
            upper_clothing_type=attributes.get("upper_clothing_type"),
            upper_clothing_conf=attributes.get("upper_clothing_conf"),
            lower_clothing_desc=attributes.get("lower_clothing_desc"),
            lower_clothing_color=attributes.get("lower_clothing_color"),
            lower_clothing_type=attributes.get("lower_clothing_type"),
            lower_clothing_conf=attributes.get("lower_clothing_conf"),
            shoes_desc=attributes.get("shoes_desc"),
            shoes_type=attributes.get("shoes_type"),
            bag_presence=attributes.get("bag_presence"),
            bag_desc=attributes.get("bag_desc"),
            bag_conf=attributes.get("bag_conf"),
            hat_presence=attributes.get("hat_presence"),
            hat_type=attributes.get("hat_type"),
            hat_desc=attributes.get("hat_desc"),
            hat_conf=attributes.get("hat_conf"),
            gender_conf=attributes.get("gender_conf"),
            age_range_conf=attributes.get("age_range_conf"),
            mask_conf=attributes.get("mask_conf"),
            hair_style_conf=attributes.get("hair_conf"),
            hair_color_conf=attributes.get("hair_conf"),
            representative_bbox=[int(x) for x in rep_bbox],
            bev_x=rep_bev_x,
            bev_y=rep_bev_y,
            embedding_vector=[],
            action=action,
            occlusion_score=0.0,
            contributing_cameras=sorted(set(contributing_cams)),
            contributing_video_ids=sorted(contributing_vids),
        ))

    elapsed = time.time() - start
    logger.info(
        "[%s] Batch complete: %d tracklets aggregated from %d videos (%d detections, %d cameras) in %.1fs",
        batch_id, len(tracklets), len(per_video_results), total_detections, len(camera_stats), elapsed,
    )

    if errors:
        logger.warning("[%s] %d video(s) had errors: %s", batch_id, len(errors), errors)

    return BatchProcessResponse(
        batch_id=batch_id,
        n_videos=n_videos,
        n_cameras=len(camera_stats),
        total_detections=total_detections,
        n_tracklets=len(tracklets),
        tracklets=tracklets,
        processing_time_s=elapsed,
        camera_stats=camera_stats,
    )
