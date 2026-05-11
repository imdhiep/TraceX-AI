"""Candidates GPU re-ranking — SigLIP 2 attribute fusion.

Called by query-service with a pre-filtered shortlist.
Re-ranks using:
  - SigLIP 2 attribute match score
  - Text semantic overlap
  - Detection quality / coverage score

Note: EVA-02 embedding_vector (DINOv2 1024-dim) was removed from the pipeline.
SigLIP 2 image-text search uses siglip_embedding (1152-dim) stored in DB.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["candidates"])

# Fusion weights (sum = 1.0)
_W_ATTRIBUTE = 0.50   # SigLIP 2 attribute match
_W_TEXT = 0.30        # Text token overlap
_W_QUALITY = 0.20    # Detection confidence + frame coverage


# ---------------------------------------------------------------------------
# Cosine similarity helper
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
# Query embedding from text — SigLIP 2 text tower
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
    "upper_clothing_color": ["white", "black", "red", "blue", "green", "yellow", "gray", "brown", "pink", "orange"],
    "lower_clothing_color": ["black", "blue", "gray", "white", "brown", "red", "green", "khaki", "jeans", "pants"],
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
            "upper_clothing_color": candidate.get("upper_clothing_color", ""),
            "lower_clothing_color": candidate.get("lower_clothing_color", ""),
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
    GPU re-ranking using SigLIP 2 text-image search + attribute fusion.

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
        emb = candidate.get("siglip_embedding") or []
        vec_sim = _cosine(query_emb, emb) if (query_emb and emb) else 0.0

        attr_sim = _attribute_score(query_lower, candidate) if query_lower else 0.0
        text_sim = _text_score(query_text, candidate) if query_text else 0.0
        quality = _quality_score(candidate)

        fusion = (
            0.50 * vec_sim
            + 0.20 * attr_sim
            + 0.20 * text_sim
            + 0.10 * quality
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
