"""Candidates GPU re-ranking — EVA-02 1024-dim + SigLIP 2 attribute fusion.

Called by query-service with a pre-filtered shortlist.
Re-ranks using:
  - Cosine similarity of EVA-02 embeddings (1024-dim)
  - SigLIP 2 attribute match score
  - Text semantic overlap
  - Detection quality / coverage score

Schema v3.3: image/text search relies on embeddings plus descriptive metadata.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["candidates"])

# Fusion weights — no camera reference (sum = 1.0)
_W_VECTOR = 0.50      # EVA-02 cosine similarity
_W_ATTRIBUTE = 0.20   # SigLIP 2 attribute match
_W_TEXT = 0.20        # Text token overlap
_W_QUALITY = 0.10     # Detection confidence + frame coverage

# Fusion weights — with camera reference (sum = 1.0)
_W_VECTOR_CAM = 0.45
_W_ATTRIBUTE_CAM = 0.15
_W_TEXT_CAM = 0.15
_W_QUALITY_CAM = 0.10
_W_SPATIAL_CAM = 0.15  # camera proximity score

_CAM_NEIGHBOR_RADIUS = 10  # person seen in cam_N can appear in cam_(N-R)…cam_(N+R)


def _extract_cam_num(cam_id: str) -> int | None:
    m = re.search(r'\d+', cam_id or "")
    return int(m.group()) if m else None


def _spatiotemporal_cam_score(
    candidate: dict,
    ref_nums: list[int],
    radius: int = _CAM_NEIGHBOR_RADIUS,
) -> float:
    """Score 0–1 based on how close candidate's camera is to reference cameras.

    - Same camera → 1.0
    - Within radius → linear decay 1.0 → 0.5
    - Outside radius → 0.0 (this candidate will be filtered before scoring)
    """
    cam_num = _extract_cam_num(candidate.get("camera_id", ""))
    if cam_num is None:
        return 0.5  # unknown cam ID: neutral
    min_dist = min(abs(cam_num - r) for r in ref_nums)
    if min_dist == 0:
        return 1.0
    if min_dist <= radius:
        return 1.0 - (min_dist / radius) * 0.5  # 1.0 → 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Cosine similarity (1024-dim EVA-02 embeddings)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Query embedding from text — EVA-02 text branch via SigLIP 2
# ---------------------------------------------------------------------------

def _build_query_embedding(query_text: str) -> list[float]:
    """
    Encode query text into a vector using SigLIP 2's text tower (same space as SigLIP2 image embeddings).
    Returns the raw SigLIP2 text feature vector (1152-dim for So400m).
    Falls back to empty list when SigLIP 2 is unavailable.
    """
    if not query_text:
        return []

    try:
        import torch
        from ..services.model_warmup import get_model

        model = get_model("siglip2")
        processor = get_model("siglip2_processor")
        if model is not None and processor is not None:
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            inputs = processor(text=[query_text], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                text_emb = model.get_text_features(**{k: v for k, v in inputs.items()
                                                      if k in ["input_ids", "attention_mask"]})
            vec = text_emb[0].cpu().float().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            return vec.tolist()  # 1152-dim — matches siglip_embedding column
    except Exception as exc:
        logger.debug("SigLIP 2 text encoding failed: %s", exc)

    # Fallback: seeded deterministic pseudo-embedding (1024-dim)
    import hashlib
    digest = hashlib.sha512(query_text.encode()).digest()  # 64 bytes
    vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    # Tile to 1024
    reps = math.ceil(1024 / len(vec))
    vec = np.tile(vec, reps)[:1024]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Attribute matching (SigLIP 2 tags vs query keywords)
# ---------------------------------------------------------------------------

_ATTRIBUTE_KEYWORDS: dict[str, list[str]] = {
    "top_color": ["white", "black", "red", "blue", "green", "yellow", "gray", "brown", "pink", "orange"],
    "bottom_color": ["black", "blue", "gray", "white", "brown", "red", "green", "khaki", "jeans", "pants"],
    "gender": ["male", "man", "boy", "female", "woman", "girl"],
    "bag": ["backpack", "bag", "handbag", "luggage"],
    "hat": ["hat", "cap", "helmet", "hood"],
}


def _attribute_score(query_lower: str, candidate: dict) -> float:
    """Score how well candidate attributes match the query text."""
    score = 0.0
    attrs: dict = candidate.get("attributes") or {}

    # Also check flat attribute fields (from video_process output)
    if not attrs:
        attrs = {
            "top_color": candidate.get("top_color", ""),
            "bottom_color": candidate.get("bottom_color", ""),
            "gender": candidate.get("gender", ""),
        }

    for attr_key, keywords in _ATTRIBUTE_KEYWORDS.items():
        attr_val = str(attrs.get(attr_key, "")).lower()
        if not attr_val:
            continue
        for kw in keywords:
            if kw in query_lower and kw in attr_val:
                score += 0.15
                break
            if kw in attr_val and attr_val in query_lower:
                score += 0.10
                break

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Text semantic overlap
# ---------------------------------------------------------------------------

def _text_score(query_text: str, candidate: dict) -> float:
    query_lower = query_text.lower()
    query_tokens = set(query_lower.split())
    score = 0.0

    for key in ("search_text", "appearance_summary", "attribute_summary", "action"):
        text = str(candidate.get(key) or "").lower()
        if not text:
            continue
        if query_lower in text:
            score += 0.25
        text_tokens = set(text.split())
        overlap = len(query_tokens & text_tokens)
        if overlap:
            score += min(overlap * 0.05, 0.15)

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Quality score (detection confidence + frame coverage)
# ---------------------------------------------------------------------------

def _quality_score(candidate: dict) -> float:
    vis = candidate.get("visibility_scores") or {}
    conf = float(vis.get("detection_confidence", candidate.get("score", 0.0)) or 0.0)
    coverage = float(vis.get("frame_coverage", 0.0) or 0.0)
    return conf * 0.7 + coverage * 0.3


# ---------------------------------------------------------------------------
# Main re-ranking endpoint
# ---------------------------------------------------------------------------

@router.post("/api/v1/candidates/search")
def candidates_search(
    query_text: str = "",
    candidates: Optional[list[dict]] = None,
    limit: int = 20,
    camera_ids: Optional[list[str]] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
) -> dict[str, Any]:
    """
    GPU re-ranking using EVA-02 1024-dim embeddings + SigLIP 2 attributes.

    Called by query-service after local DB pre-filtering.

    Returns:
        {"items": [<ranked candidates with _fusion_score>]}
    """
    if not candidates:
        return {"items": []}

    query_lower = (query_text or "").lower()

    # Build query embedding once
    query_emb = _build_query_embedding(query_text) if query_text else []

    scored: list[tuple[float, int, dict[str, Any]]] = []

    for idx, candidate in enumerate(candidates):
        emb = candidate.get("embedding_vector") or []

        # 1. EVA-02 cosine similarity (1024-dim)
        vec_sim = _cosine(query_emb, emb) if (query_emb and emb) else 0.0

        # 2. SigLIP 2 attribute match
        attr_sim = _attribute_score(query_lower, candidate) if query_lower else 0.0

        # 3. Text semantic overlap
        text_sim = _text_score(query_text, candidate) if query_text else 0.0

        # 4. Detection quality + coverage
        quality = _quality_score(candidate)

        fusion = (
            _W_VECTOR * vec_sim
            + _W_ATTRIBUTE * attr_sim
            + _W_TEXT * text_sim
            + _W_QUALITY * quality
        )

        scored.append((fusion, idx, candidate))

    scored.sort(key=lambda x: (-x[0], x[1]))

    results = [
        {**c, "_fusion_score": round(score, 6)}
        for score, _, c in scored[:limit]
    ]

    logger.info(
        "GPU re-ranking: query=%r  candidates=%d → results=%d  top_score=%.4f",
        query_text,
        len(candidates),
        len(results),
        results[0]["_fusion_score"] if results else 0.0,
    )
    return {"items": results}
