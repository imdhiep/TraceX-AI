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


DEFAULT_SAMPLE_INTERVAL = 15
DEFAULT_MIN_BBOX_AREA = 400
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))
VLM_BATCH_SIZE = _get_positive_env_int("VLM_BATCH_SIZE", 1)
VLM_BATCH_MAX_NEW_TOKENS_PER_CROP = _get_positive_env_int(
    "VLM_BATCH_MAX_NEW_TOKENS_PER_CROP", 512
)


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

    with ThreadPoolExecutor(max_workers=2) as ex:
        # Submit first batch preprocessing
        futures = [ex.submit(_preprocess, b) for b in batches[:2]]

        for idx, batch in enumerate(batches):
            # Prefetch next+1 batch while current is on GPU
            if idx + 2 < len(batches):
                futures.append(ex.submit(_preprocess, batches[idx + 2]))

            inputs_raw, sizes = futures[idx].result()
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                      for k, v in inputs_raw.items()}

            try:
                with torch.no_grad(), autocast_ctx:
                    outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    outputs, threshold=threshold,
                    target_sizes=torch.tensor(sizes, device=device),
                )
            except Exception as exc:
                logger.warning("[rtdetr] batch failed: %s", exc)
                all_dets.extend([[] for _ in batch])
                continue

            for res in results:
                dets = []
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    if label.item() not in person_ids:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                 "score": float(score), "label": "person"})
                all_dets.append(dets)

    return all_dets


def _detect_persons_batch(frames: list[np.ndarray], threshold: float = 0.25) -> list[list[dict]]:
    """Person detection via RT-DETR. Raises RuntimeError if model is unavailable."""
    rtdetr_result = _detect_persons_rtdetr(frames, threshold=max(threshold, 0.4))
    if rtdetr_result is not None:
        n_dets = sum(len(d) for d in rtdetr_result)
        logger.debug("[detect] RT-DETR: %d frames → %d detections", len(frames), n_dets)
        return rtdetr_result

    raise RuntimeError(
        "[detect] RT-DETR unavailable — model not loaded. "
        "Check GPU OOM or model warmup logs."
    )


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
Describe this person's visible appearance accurately.
Do NOT choose from a fixed label list — use free text for clothing descriptions.
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
  "appearance_summary": "one concise sentence"
}

IMPORTANT INSTRUCTIONS:
- Set gender to "unknown" ONLY if the face/head is not visible or ambiguous. If head/upper body is visible, infer gender from features.
- Set age_range to "unknown" ONLY if the face is not visible. Body shape and clothing can indicate approximate age.
- For bag/hat/mask: if you can see the upper body clearly and the item is NOT present, respond "no" (not "unknown").
- Hair style/color: respond "unknown" only if head/top of person is not in the frame.
- Replace 0.0 placeholders with your actual confidence (0.0-1.0). If you provide a value OTHER than "unknown" for a field, you MUST provide conf > 0.0."""

_VLM_BATCH_PROMPT_TEMPLATE = """You are analyzing {n} person crops from surveillance cameras.
The images above show persons labeled (1) to ({n}) in order.
Describe each person's visible appearance accurately. Use free text for clothing descriptions.
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
  "appearance_summary": "one concise sentence"
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
                max_new_tokens=512,
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


def _caption_crops_vlm_batch(crops: list, batch_size: int = VLM_BATCH_SIZE) -> list:
    """Batch Qwen2-VL captioning — sends up to batch_size crops per call (~3-4x faster).

    Retries failed batches with smaller sub-batches before falling back to singles.
    """
    model = get_model("qwen2vl")
    processor = get_model("qwen2vl_processor")
    if model is None or processor is None:
        return [_default_attributes() for _ in crops]

    batch_size = max(1, batch_size)
    device = _get_device()
    results: list = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        n = len(batch)

        if n == 1:
            results.append(_caption_crop_vlm(batch[0]))
            continue

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

        except Exception as exc:
            next_batch_size = min(max(1, batch_size // 2), max(1, (n + 1) // 2))
            if n > 1 and next_batch_size < n:
                logger.warning(
                    "[vlm] batch(%d) failed: %s — retrying with smaller batches of %d",
                    n,
                    exc,
                    next_batch_size,
                )
                results.extend(_caption_crops_vlm_batch(batch, batch_size=next_batch_size))
                continue

            logger.warning("[vlm] batch(%d) failed: %s — falling back to single crops", n, exc)
            for crop in batch:
                results.append(_caption_crop_vlm(crop))

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


def _batch_siglip_embeddings(
    t_data: list,
) -> tuple[list, list, list]:
    """
    Batch SigLIP2 GPU inference — runs BEFORE fragment merge.
    Encodes 5 evenly-spaced frames per raw fragment → pool average for merge similarity.
    Also encodes single representative crop for DB storage.

    Args:
        t_data: list of (local_tracklet, t_idx, rep_bbox_float, t_frames)

    Returns: (siglip_multi_feats, all_rep_crops, all_siglip_embeddings)
      siglip_multi_feats: list of 1152-dim pool-avg embeddings per fragment (for merge)
      all_rep_crops: list of PIL Images (384×384) per fragment
      all_siglip_embeddings: list of 1152-dim single-crop embeddings per fragment (for DB)
    """


    device = _get_device()
    dtype = torch.float16

    # ── Helper: crop + pad to square ─────────────────────────────────────────
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

    # ── Build representative crops (384×384) ──────────────────────────────────
    all_rep_crops: list[Image.Image] = []
    for lt, t_idx, rep_bbox, t_frames in t_data:
        mid_idx = len(t_frames) // 2
        c = _extract_crop(t_frames[mid_idx], rep_bbox, 384) if t_frames else None
        all_rep_crops.append(
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) if c is not None
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        )

    # ── SigLIP2 multi-frame pool-avg — for fragment merge ─────────────────────
    siglip_multi_feats = [[] for _ in t_data]
    model_sip = get_model("siglip2")
    proc_sip = get_model("siglip2_processor")
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
                inputs = proc_sip(images=siglip_crops, return_tensors="pt", padding=True)
                inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in inputs.items()}
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                    feats = model_sip.get_image_features(**{k: v for k, v in inputs.items()
                                                                 if k in ["pixel_values"]})
                feats = feats / feats.norm(dim=-1, keepdim=True)
                # Pool-average per tracklet using slices
                siglip_multi_feats = [[] for _ in t_data]
                cur = 0
                for t_i, (lt, t_idx, rep_bbox, t_frames) in enumerate(t_data):
                    n = min(5, len(t_frames))
                    if cur + n <= len(feats):
                        avg = feats[cur:cur + n].mean(0)
                        siglip_multi_feats[t_i] = avg.cpu().float().tolist()
                    cur += n
        except Exception as exc:
            logger.warning("[pipeline] SigLIP2 multi-frame encoding failed: %s", exc)
            siglip_multi_feats = [[] for _ in t_data]

    # ── SigLIP2 single-crop — representative crop for DB storage ───────────────
    img_feats = None
    if model_sip and proc_sip and t_data:
        try:
            img_inputs = proc_sip(images=all_rep_crops, return_tensors="pt", padding=True)
            img_inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in img_inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                img_feats = model_sip.get_image_features(**{k: v for k, v in img_inputs.items()
                                                             if k in ["pixel_values"]})
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, D]
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

    logger.info("[pipeline] SigLIP2 done: %d fragments | merge_emb=%d | DB_emb=%d",
                len(t_data), sum(1 for e in siglip_multi_feats if e),
                sum(1 for e in all_siglip_embeddings if e))
    return siglip_multi_feats, all_rep_crops, all_siglip_embeddings


def _batch_caption_and_classify(
    t_data: list,
    all_rep_crops: list[Image.Image],
) -> tuple[list[dict], list[dict], list]:
    """
    Batch Qwen2-VL + VideoMAE GPU inference — runs AFTER fragment merge.
    Attribute captioning (Qwen2-VL) and action classification (VideoMAE) on merged tracklets.

    Args:
        t_data: list of (merged_tracklet, t_idx, rep_bbox_float, t_frames)
        all_rep_crops: list of PIL Images (384×384) per merged tracklet

    Returns: (all_attributes, all_attr_confs, all_actions)
      all_attributes[i]: dict with keys gender, age_range, upper_clothing_color, ...
      all_attr_confs[i]: dict with per-field confidence scores
      all_actions[i]: tuple (action_str, confidence, kinetics_label)
    """
    device = _get_device()
    dtype = torch.float16

    # ── Qwen2-VL-7B-Instruct: open-vocabulary attribute captioning ────────────
    logger.info(
        "[vlm] captioning %d representative crops with batch_size=%d",
        len(all_rep_crops),
        VLM_BATCH_SIZE,
    )
    all_attributes: list[dict] = _caption_crops_vlm_batch(
        all_rep_crops,
        batch_size=VLM_BATCH_SIZE,
    )
    all_attr_confs: list[dict] = []
    for attrs in all_attributes:
        all_attr_confs.append({
            "gender_conf":       attrs.get("gender_conf"),
            "age_range_conf":    attrs.get("age_range_conf"),
            "upper_clothing_conf": attrs.get("upper_clothing_conf"),
            "lower_clothing_conf": attrs.get("lower_clothing_conf"),
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

    # ── VideoMAE TRUE BATCH: stack all merged tracklet clips → 1 forward pass ──
    all_actions: list[tuple] = []
    model_vmae = get_model("videomae")
    proc_vmae = get_model("videomae_processor")
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
                    frames_224.append(cv2.resize(crop if crop.size > 0 else f,
                                                  (224, 224), interpolation=cv2.INTER_LINEAR))
                while len(frames_224) < 16:
                    frames_224.append(frames_224[-1] if frames_224 else np.zeros((224, 224, 3), dtype=np.uint8))
                all_clips.append(frames_224[:16])

            vmae_dtype = torch.float16
            inputs = proc_vmae(all_clips, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=vmae_dtype) if v.is_floating_point() else v.to(device)
                       for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=vmae_dtype):
                outputs = model_vmae(**inputs)
            logits = outputs.logits.float()  # [N, num_classes]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(1, dim=-1)
            top_idx = top_indices.squeeze(1).tolist()
            top_conf = top_probs.squeeze(1).tolist()
            id2label = getattr(model_vmae.config, "id2label", {})
            for idx, conf in zip(top_idx, top_conf):
                label = id2label.get(idx, "")
                action = _map_kinetics_to_tracex_action(label) if label else "unknown"
                all_actions.append((action, float(conf), label))
        except Exception as exc:
            logger.warning("[pipeline] VideoMAE batch failed on merged tracklets: %s", exc)
            all_actions = [(_run_videomae_actions(t[3], t[2]), 0.0, "") for t in t_data]
    else:
        all_actions = [("unknown", 0.0, "")] * len(t_data)

    logger.info("[pipeline] Qwen2-VL + VideoMAE done: %d merged tracklets | actions=%d",
                len(t_data), sum(1 for a in all_actions if a and a[0] != "unknown"))
    return all_attributes, all_attr_confs, all_actions
    model_sip = get_model("siglip2")
    proc_sip = get_model("siglip2_processor")
    if model_sip and proc_sip and t_data:
        try:
            img_inputs = proc_sip(images=all_rep_crops, return_tensors="pt", padding=True)
            img_inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in img_inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                img_feats = model_sip.get_image_features(**{k: v for k, v in img_inputs.items()
                                                             if k in ["pixel_values"]})
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, D]
        except Exception as exc:
            logger.warning("[pipeline] SigLIP image encoding failed: %s", exc)
            img_feats = None

    # ── Qwen2-VL-7B-Instruct: open-vocabulary attribute captioning ────────────
    logger.info(
        "[vlm] captioning %d representative crops with batch_size=%d",
        len(all_rep_crops),
        VLM_BATCH_SIZE,
    )
    all_attributes: list[dict] = _caption_crops_vlm_batch(
        all_rep_crops,
        batch_size=VLM_BATCH_SIZE,
    )
    all_attr_confs: list[dict] = []
    for attrs in all_attributes:
        all_attr_confs.append({
            "gender_conf":       attrs.get("gender_conf"),
            "age_range_conf":    attrs.get("age_range_conf"),
            "upper_clothing_conf": attrs.get("upper_clothing_conf"),
            "lower_clothing_conf": attrs.get("lower_clothing_conf"),
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

    # ── VideoMAE TRUE BATCH: stack all tracklet clips → 1 forward pass ───────
    all_actions = []
    model_vmae = get_model("videomae")
    proc_vmae = get_model("videomae_processor")
    if model_vmae and proc_vmae and t_data:
        try:
            # Build 16-frame clip for every tracklet
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
                    frames_224.append(cv2.resize(crop if crop.size > 0 else f,
                                                  (224, 224), interpolation=cv2.INTER_LINEAR))
                while len(frames_224) < 16:
                    frames_224.append(frames_224[-1] if frames_224 else np.zeros((224, 224, 3), dtype=np.uint8))
                all_clips.append(frames_224[:16])

            # Batch all clips: proc_vmae expects list-of-frames per video
            # Stack into [N, 16, H, W, C] then process
            vmae_dtype = torch.float16
            inputs = proc_vmae(all_clips, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=vmae_dtype) if v.is_floating_point() else v.to(device)
                       for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=vmae_dtype):
                outputs = model_vmae(**inputs)
            logits = outputs.logits.float()  # [N, num_classes]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(1, dim=-1)
            top_idx = top_indices.squeeze(1).tolist()
            top_conf = top_probs.squeeze(1).tolist()
            id2label = getattr(model_vmae.config, "id2label", {})
            for idx, conf in zip(top_idx, top_conf):
                label = id2label.get(idx, "")
                action = _map_kinetics_to_tracex_action(label) if label else "unknown"
                all_actions.append((action, float(conf), label))
        except Exception as exc:
            logger.warning("[pipeline] VideoMAE true-batch failed: %s — fallback", exc)
            all_actions = [(_run_videomae_actions(t[3], t[2]), 0.0, "") for t in t_data]
    else:
        all_actions = [("unknown", 0.0, "")] * len(t_data)

    # Capture SigLIP2 image embeddings (already computed above as img_feats)
    # These are in the same embedding space as SigLIP2 text queries → usable for text search
    all_siglip_embeddings: list[list[float]] = []
    try:
        if img_feats is not None:
            all_siglip_embeddings = [img_feats[i].cpu().float().tolist() for i in range(len(t_data))]
        else:
            all_siglip_embeddings = [[]] * len(t_data)
    except Exception:
        all_siglip_embeddings = [[]] * len(t_data)

    logger.info("[pipeline] batch features done: %d tracklets | SigLIP2 merge_emb=%d | SigLIP2 DB=%d | VideoMAE=%d",
                len(t_data), sum(1 for e in siglip_multi_feats if e),
                sum(1 for e in all_siglip_embeddings if e),
                sum(1 for a in all_actions if a and a != "unknown"))
    return siglip_multi_feats, all_attributes, all_attr_confs, all_actions, all_rep_crops, all_siglip_embeddings


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
        obs = lt.observations
        mid = obs[len(obs) // 2]
        rep_bbox_float = [float(x) for x in mid.bbox]
        t_frames = [frame_lookup[o.frame_index] for o in obs if o.frame_index in frame_lookup] or [sampled_frames[0].image]
        t_data_raw.append((lt, t_idx, rep_bbox_float, t_frames))

    siglip_multi_feats, all_rep_crops, all_siglip_embeddings = _batch_siglip_embeddings(t_data_raw)

    # Stage 6: Post-hoc fragment merging via SigLIP2 cosine similarity.
    # SigLIP embeddings from Stage 5 → decide which raw fragments belong to same person.
    import numpy as _np

    _merger = TrackletFragmentMerger(similarity_threshold=0.85, max_gap_seconds=60.0)
    _orig_accepted = list(accepted)
    accepted, _groups = _merger.merge(list(accepted), siglip_multi_feats)

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

    siglip_multi_feats_merged = [_pool_avg([siglip_multi_feats[i] for i in g]) for g in _groups]
    siglip_embeddings_merged  = [_pool_avg([all_siglip_embeddings[i] for i in g]) for g in _groups]

    # Rebuild t_data for merged tracklets — pick best frame by detection confidence for Qwen/VMAE
    def _best_obs_for_merged(g: list[int]):
        """Return FrameDetection with highest detection confidence from all fragments in group."""
        best = None
        best_conf = -1.0
        for frag_idx in g:
            for obs in _orig_accepted[frag_idx].observations:
                if obs.confidence > best_conf:
                    best_conf = obs.confidence
                    best = obs
        if best is None and g and _orig_accepted[g[0]].observations:
            best = _orig_accepted[g[0]].observations[0]
        return best

    t_data_merged = []
    rep_crops_merged = []
    for new_idx, mt in enumerate(accepted):
        group_indices = _groups[new_idx]
        best_obs = _best_obs_for_merged(group_indices)
        if best_obs is None:
            logger.warning("[pipeline] %s: merged group %d has no observations", video_id, new_idx)
            best_obs = _orig_accepted[group_indices[0]].observations[0]
        rep_bbox_float = [float(x) for x in best_obs.bbox]
        # Collect all frames from all fragments in this group (up to 16 for VideoMAE)
        all_frames = []
        for frag_idx in group_indices:
            frag_frames = [
                frame_lookup[o.frame_index]
                for o in _orig_accepted[frag_idx].observations
                if o.frame_index in frame_lookup
            ]
            all_frames.extend(frag_frames)
        # Deduplicate while preserving order
        seen, unique_frames = set(), []
        for f in all_frames:
            fid = id(f)
            if fid not in seen:
                seen.add(fid)
                unique_frames.append(f)
        t_data_merged.append((mt, new_idx, rep_bbox_float, unique_frames))
        # Representative crop from best_obs frame
        best_frame = frame_lookup.get(best_obs.frame_index, sampled_frames[0].image)
        x1, y1 = max(0, int(best_obs.bbox[0])), max(0, int(best_obs.bbox[1]))
        x2, y2 = min(best_frame.shape[1], int(best_obs.bbox[2])), min(best_frame.shape[0], int(best_obs.bbox[3]))
        crop = best_frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = best_frame
        max_dim = max(crop.shape[0], crop.shape[1])
        top = (max_dim - crop.shape[0]) // 2
        bottom = max_dim - crop.shape[0] - top
        left = (max_dim - crop.shape[1]) // 2
        right = max_dim - crop.shape[1] - left
        square = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        resized = cv2.resize(square, (384, 384), interpolation=cv2.INTER_LINEAR)
        rep_crops_merged.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    # Stage 7: Qwen2-VL + VideoMAE — AFTER fragment merge, on merged tracklets
    all_attributes, all_attr_confs, all_actions = _batch_caption_and_classify(t_data_merged, rep_crops_merged)

    _CROPS_DIR = Path("/workspace/storage/crops")
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tracklets: list[TrackletResult] = []
    for t_idx, (lt, _, rep_bbox_float, t_frames) in enumerate(t_data_merged):
        obs = lt.observations
        attributes = all_attributes[t_idx]
        siglip_emb = siglip_embeddings_merged[t_idx] if t_idx < len(siglip_embeddings_merged) else []
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

            per_video_results.append(result)
            camera_stats[result.camera_id] = camera_stats.get(result.camera_id, 0) + int(
                result.total_detections or 0
            )

    tracklets: list[TrackletResult] = []
    total_detections = 0
    for result in per_video_results:
        total_detections += int(result.total_detections or 0)
        tracklets.extend(result.tracklets or [])

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
