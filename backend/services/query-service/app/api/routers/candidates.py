"""Candidates search router for query-service.

Flow:
1. metadata-service forwards the normalized search request here
2. text-only keeps the historical text/SigLIP path
3. image-only retrieves the full gallery by PersonViT ReID similarity only
4. image+text unions PersonViT retrieval with SigLIP/metadata recall, while image keeps rank priority
5. Union-find merge, persist QueryCandidate rows, then format results
"""

from __future__ import annotations

import heapq
import io
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session


class SearchRequest(BaseModel):
    query: str = ""
    text: str | None = None  # alias
    top_k: int = 20
    offset: int = 0
    camera_ids: list[str] | None = None
    time_from: str | None = None
    time_to: str | None = None
    query_image_url: str | None = None  # URL of uploaded query image (for history display)
    query_image_path: str | None = None  # local path for image encoding
    persist_query: bool = False
    user_id: int | None = None  # overwritten from JWT by the public runtime
    # When the frontend paginates (offset > 0) it can pass back the qid returned by
    # the first call so we reuse the same query_history row instead of creating a
    # new one per page. The full ranked set is persisted for history, while each
    # response still returns only the requested page.
    query_id: str | None = None

from shared.database import SessionLocal
from shared.models import (
    QueryCandidate, QueryCandidateTracklet, QueryHistory,
    Tracklet, TrackletAction, TrackletEmbedding, Video,
)
from shared.tracklet_time import tracklet_time_seconds, tracklet_time_window
from app.config import settings
from app.services.auth_runtime import get_current_user_id
from app.services.translation import (
    detect_vietnamese,
    translate_to_english,
    translate_to_vietnamese_cached,
    warmup as warmup_translation,
)
from app.services.query_metadata_parse import (
    parse_query_metadata,
    ParsedQueryMetadata,
    ACTION_CONFIDENCE_FLOOR,
    ACTION_BONUS_SCORE,
    METADATA_BONUS_PER_MATCH,
    METADATA_MAX_BONUS,
)
from sqlalchemy import func, or_, select
from sqlalchemy.orm import contains_eager, joinedload
import re

_CAM_NEIGHBOR_RADIUS = 10


def _extract_cam_num(cam_id: str) -> int | None:
    m = re.search(r'\d+', cam_id or "")
    return int(m.group()) if m else None


def _expand_camera_range(camera_ids: list[str], radius: int = _CAM_NEIGHBOR_RADIUS) -> list[str]:
    """Expand camera_ids to include all cams within ±radius of each cam number.

    cam_20 with radius=10 → cam_10 … cam_30.
    Cams with non-numeric IDs are kept as-is without expansion.
    """
    expanded: set[str] = set()
    for cam_id in camera_ids:
        num = _extract_cam_num(cam_id)
        if num is None:
            expanded.add(cam_id)
            continue
        # preserve prefix ("cam_") and zero-padding width ("01" → width=2)
        digits = re.search(r'\d+', cam_id).group()
        prefix = cam_id[: cam_id.index(digits)]
        width = len(digits)
        for i in range(max(1, num - radius), num + radius + 1):
            expanded.add(f"{prefix}{i:0{width}d}")
    return list(expanded)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

_translation_warmed_up = False


def _get_positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


_QUERY_LOG_TOP_N = _get_positive_env_int("QUERY_LOG_TOP_N", 10)
_QUERY_LOG_GROUP_N = _get_positive_env_int("QUERY_LOG_GROUP_N", 10)
_QUERY_LOG_MEMBER_N = _get_positive_env_int("QUERY_LOG_MEMBER_N", 8)


def _short_text(value: object, limit: int = 160) -> str:
    text_value = " ".join(str(value or "").split())
    if len(text_value) <= limit:
        return text_value
    return text_value[: max(0, limit - 3)] + "..."


def _parsed_query_log_dict(parsed: ParsedQueryMetadata) -> dict[str, list[str]]:
    return {
        "gender": parsed.gender,
        "upper_color": parsed.upper_color,
        "lower_color": parsed.lower_color,
        "shoes_color": parsed.shoes_color,
        "hat_color": parsed.hat_color,
        "bag_type": parsed.bag_type,
        "actions": parsed.actions,
        "unbound_colors": parsed.unbound_colors,
    }


def _tracklet_log_item(
    tracklet: Tracklet,
    *,
    text_score: float | None = None,
    vector_score: float | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "tracklet_id": tracklet.tracklet_id,
        "camera_id": tracklet.camera_id,
        "track_id": tracklet.track_id,
        "time": [
            round(float(tracklet.start_time or 0.0), 2),
            round(float(tracklet.end_time or 0.0), 2),
        ],
        "gender": tracklet.gender,
        "upper": tracklet.upper_color,
        "lower": tracklet.lower_color,
        "shoes": tracklet.shoes_color,
        "quality": round(float(tracklet.quality_score or 0.0), 4),
        "summary": _short_text(tracklet.appearance_summary),
    }
    if text_score is not None:
        item["text"] = round(float(text_score), 4)
    if vector_score is not None:
        item["vec"] = round(float(vector_score), 4)
    return item


def _score_stats(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0}
    nonzero = [s for s in scores if s > 0.0]
    return {
        "count": len(scores),
        "nonzero": len(nonzero),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "avg": round(sum(scores) / len(scores), 4),
    }


def _jdump(obj: Any) -> str:
    """JSON-encode a log payload preserving Vietnamese characters and using a
    `default=str` fallback so datetimes / Decimal / etc. don't blow up the
    logger. Output is a single line — friendlier than Python's repr() of
    nested dicts/lists in `docker logs` while still grep-able."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return repr(obj)


def _candidate_id_for_group(query_id: str, group: list[Tracklet]) -> str:
    """Stable query-scoped candidate id for a merged tracklet group."""
    tracklet_ids = sorted(str(t.tracklet_id) for t in group if t.tracklet_id)
    key = "|".join(tracklet_ids) or str(len(group))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tracex:{query_id}:candidate:{key}"))


def _candidate_key_for_group(group: list[Tracklet]) -> str:
    tracklet_ids = sorted(str(t.tracklet_id) for t in group if t.tracklet_id)
    if len(tracklet_ids) == 1:
        return tracklet_ids[0][:255]
    key = "|".join(tracklet_ids) or str(len(group))
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"tracex:candidate-key:{key}").hex[:12]
    return f"{len(tracklet_ids)}-tracklets:{suffix}"


def _ensure_translation_warmed_up():
    global _translation_warmed_up
    if not _translation_warmed_up:
        try:
            warmup_translation()
        except Exception:
            pass
        _translation_warmed_up = True


def _translate_query(query: str) -> str:
    """Translate Vietnamese query to English for better matching."""
    _ensure_translation_warmed_up()
    if detect_vietnamese(query):
        english = translate_to_english(query)
        if english != query:
            logger.info("Translated query: %r → %r", query, english)
            return english
    return query



def _parse_dt(value: str | None):
    """Parse ISO datetime string to timezone-aware datetime, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            from datetime import timezone as _tz
            dt = dt.replace(tzinfo=_tz.utc)
        return dt
    except ValueError:
        return None


def _build_search_text(row: Tracklet) -> str:
    return " ".join([
        row.appearance_summary or "",
        row.gender or "",
        row.upper_color or "",
        row.lower_color or "",
        row.shoes_color or "",
        row.age_range or "",
        row.hat_color or "",
        row.bag_type or "",
        row.mask_presence or "",
        row.hair_style or "",
        row.hair_color or "",
    ]).lower()


def _tracklet_overlaps_time_range(
    tracklet: Tracklet,
    time_from: datetime | None,
    time_to: datetime | None,
) -> bool:
    """Return True when the tracklet's real video time overlaps the query range."""
    if time_from is None and time_to is None:
        return True
    window = tracklet_time_window(tracklet)
    if window is None:
        return False
    start_dt, end_dt = window
    if time_from is not None and end_dt < time_from:
        return False
    if time_to is not None and start_dt > time_to:
        return False
    return True


def _score_search_text(search_text: str, cleaned_query: str, query_tokens: set[str]) -> float:
    score = 0.0
    if cleaned_query in search_text:
        score += 0.5
    row_tokens = set(search_text.split())
    overlap = len(query_tokens & row_tokens)
    if overlap:
        score += min(overlap * 0.1, 0.5)
    return score


def _search_text_expr():
    return func.lower(
        func.concat_ws(
            " ",
            func.coalesce(Tracklet.appearance_summary, ""),
            func.coalesce(Tracklet.gender, ""),
            func.coalesce(Tracklet.upper_color, ""),
            func.coalesce(Tracklet.lower_color, ""),
            func.coalesce(Tracklet.shoes_color, ""),
            func.coalesce(Tracklet.age_range, ""),
            func.coalesce(Tracklet.hat_color, ""),
            func.coalesce(Tracklet.bag_type, ""),
            func.coalesce(Tracklet.mask_presence, ""),
            func.coalesce(Tracklet.hair_style, ""),
            func.coalesce(Tracklet.hair_color, ""),
        )
    )


def _ilike_contains(expr, value: str):
    escaped = (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return expr.ilike(f"%{escaped}%", escape="\\")


# Gender hard-filter — applied at prefilter stage so a query for "woman" can
# never surface male tracklets (and vice-versa). The keyword vocabulary lives
# in backend/config/query_metadata_vocab.json and is parsed by
# query_metadata_parse.parse_query_metadata(); we only keep the confidence
# floor here because it's a prefilter-stage policy, not a vocabulary concern.
# Kept in sync with _CONF_THRESHOLDS["gender"] in the merge logic below.
_GENDER_FILTER_CONF_FLOOR = 0.55


def _local_prefilter(
    session: Session,
    query_text: str,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 200,
    parsed: ParsedQueryMetadata | None = None,
) -> tuple[list[Tracklet], dict[str, float]]:
    """Pre-filter tracklets using camera, time range, and text matching.

    `parsed` is the structured-metadata view of `query_text` produced once by
    the caller (search_candidates). When supplied, gender → hard filter at the
    SQL layer. Other parsed fields are applied as bonuses downstream, not as
    filters, so VLM mis-labels can't silently drop valid matches.
    """
    cleaned_query = (query_text or "").strip().lower()

    # Join Video so Python-side time filtering/grouping can use the real
    # recording timestamp: video.recorded_at or timestamp parsed from filename
    # + tracklet start/end offsets.
    statement = (
        select(Tracklet)
        .join(Video, Tracklet.video_id == Video.video_id)
        .options(
            contains_eager(Tracklet.video),   # reuse joined rows — needed for _tracklet_abs_time
            joinedload(Tracklet.embedding),
        )
        .order_by(Tracklet.created_at.desc(), Tracklet.id.desc())
    )

    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(func.lower(Tracklet.camera_id).in_(cam_lower))

    tf = _parse_dt(time_from)
    tt = _parse_dt(time_to)
    # Gender hard-filter — only when the query explicitly mentions one gender.
    # Tracklets with low gender_conf (uncertain VLM output) are kept to avoid
    # silently dropping valid matches.
    query_gender = parsed.gender[0] if (parsed and parsed.gender) else None
    if query_gender is not None:
        statement = statement.where(
            or_(
                func.lower(Tracklet.gender) == query_gender,
                Tracklet.gender_conf < _GENDER_FILTER_CONF_FLOOR,
                Tracklet.gender_conf.is_(None),
                Tracklet.gender.is_(None),
            )
        )

    if not cleaned_query:
        rows: list[Tracklet] = []
        stream = session.execute(
            statement.execution_options(stream_results=True, yield_per=256)
        ).scalars()
        for row in stream:
            if not _tracklet_overlaps_time_range(row, tf, tt):
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows, {row.tracklet_id: 0.0 for row in rows}

    query_tokens = {token for token in cleaned_query.split() if token}
    text_expr = _search_text_expr()
    broad_terms = [cleaned_query, *sorted(query_tokens)]
    broad_clauses = [_ilike_contains(text_expr, term) for term in broad_terms if term]
    prefilter_statement = statement.where(or_(*broad_clauses))

    stream = session.execute(
        prefilter_statement.execution_options(stream_results=True, yield_per=256)
    ).scalars()

    top_matches: list[tuple[float, int, Tracklet]] = []
    for row in stream:
        if not _tracklet_overlaps_time_range(row, tf, tt):
            continue
        search_text = _build_search_text(row)
        score = _score_search_text(search_text, cleaned_query, query_tokens)
        if score <= 0:
            continue

        heap_item = (score, row.id, row)
        if len(top_matches) < limit:
            heapq.heappush(top_matches, heap_item)
            continue

        if (score, row.id) > (top_matches[0][0], top_matches[0][1]):
            heapq.heapreplace(top_matches, heap_item)

    if top_matches:
        top_matches.sort(key=lambda item: (-item[0], -item[1]))
        shortlist = [row for _, _, row in top_matches]
        score_map = {row.tracklet_id: score for score, _, row in top_matches}
        return shortlist, score_map

    # No tracklet matched the query text. Returning "latest N rows" here would
    # silently flood the SigLIP re-rank stage with unrelated tracklets and
    # produce arbitrary-looking top-k — return empty so the caller surfaces
    # "no results" honestly.
    return [], {}


def _preview_url_for(tracklet) -> str:
    """Cache-busted preview URL for a tracklet's representative crop.

    Prefer the direct static crop URL so direct query-service search does not
    depend on metadata-service just to render thumbnails. Fall back to the
    legacy preview endpoint only when the tracklet has no saved crop URL.
    """
    tid = getattr(tracklet, "tracklet_id", "") if tracklet is not None else ""
    if not tid:
        return ""
    ts = getattr(tracklet, "updated_at", None)
    crop_url = str(getattr(tracklet, "crop_url", "") or "").strip()
    base = crop_url or f"/candidates/{tid}/preview"
    if ts is None:
        return base
    try:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}v={int(ts.timestamp() * 1_000_000)}"
    except Exception:
        return base


# ── Identity merge: cosine similarity + temporal/camera guards + union-find ───

# 2026-05-17: default 0.9 → 0.75 after Re-ID swap DINOv2 → PersonViT-S.
# Bench on camera_0002 (894 positive + 7710 hard-negative pairs) shows
# PersonViT cosine distribution is much sharper than DINOv2/SigLIP:
#   thr 0.75 → precision 92.9%, recall 75.7%
#   thr 0.80 → precision 96.2%, recall 70.5%   ★ ingest-side default
#   thr 0.90 → precision 96.8%, recall 26.5%   ← DINOv2-era default (way too strict
#                                                 for PersonViT — under-merges hard)
# Query-side intentionally runs tighter than ingest now (0.82 vs 0.80): a
# cross-video search lump that merges two visually-similar-but-different people
# is more visible to end users than a fragment under-merge inside one video.
# 0.82 sits between the two benchmarked points; expected ≈ 96.5%/68% based on
# the smooth precision/recall curve — re-bench if drift is suspected.
_MERGE_THRESHOLD = float(os.getenv("QUERY_REID_MERGE_THRESHOLD", os.getenv("QUERY_MERGE_THRESHOLD", "0.82")))
_IDENTITY_FALLBACK_TO_SIGLIP = os.getenv("QUERY_REID_FALLBACK_TO_SIGLIP", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
# Identity grouping prefers the dedicated `reid_embedding` lane (PersonViT-S
# 384-dim). The DB was reset on 2026-05-17 via scripts/reset_db_for_personvit.py,
# so any row that still lacks reid_embedding is a fresh ingest that failed —
# we keep it as a singleton rather than silently downgrading to SigLIP semantic
# space (which uses very different cosine geometry and would mis-cluster).
# Operators can set QUERY_REID_FALLBACK_TO_SIGLIP=1 only if running a transient
# A/B where SigLIP is the primary lane — not the production path.

# Component-level guard for the union-find pass. Plain pair-wise union-find is
# transitive: A↔B 0.99, B↔C 0.99 chains C into A's group even when A↔C is
# only 0.97. Before unioning two components we require:
#   - mean cross-cosine ≥ _MERGE_THRESHOLD     (cluster is genuinely tight)
#   - min cross-cosine  ≥ _MERGE_COMPONENT_FLOOR  (no weak bridge)
# The floor is _MERGE_THRESHOLD - _MERGE_COMPONENT_MARGIN so a single
# slightly-worse pair doesn't sink an otherwise solid merge.
_MERGE_COMPONENT_MARGIN = float(os.getenv("QUERY_MERGE_COMPONENT_MARGIN", "0.01"))
_MERGE_COMPONENT_FLOOR = max(0.0, _MERGE_THRESHOLD - _MERGE_COMPONENT_MARGIN)
_MERGE_MAX_GAP_S = 86400.0   # max 24-hour gap — matches trace window
_CONF_THRESHOLD = 0.70        # fallback: below this = uncertain → don't block merge

# Per-attribute confidence thresholds: both sides must exceed to block merge.
# These are calibrated for LOGIT-DERIVED confidences (geometric mean of token
# probabilities under Qwen2.5-VL), not self-reported numbers. Logit confs are
# generally lower than self-report — categorical short values like "man"/"woman"
# typically sit around 0.5–0.9; long free-text spans drift lower.
_CONF_THRESHOLDS: dict[str, float] = {
    "gender":         0.55,
    "mask_presence":  0.55,
    "bag_presence":   0.60,
    "hat_presence":   0.60,
    "age_range":      0.50,
    "upper_color":    0.55,
    "lower_color":    0.55,
    "shoes_color":    0.55,
    "hat_color":      0.55,
    "hair_color":     0.55,
}

# "none" is a meaningful value (model confirmed absence) for these fields
_NONE_IS_VALID = frozenset({"hat_color", "bag_type", "mask_presence"})
_UNKNOWN_VALUES = frozenset({"", "unknown", "null", "n/a", "not sure"})


def _norm_meta(attr: str, value: object) -> str | None:
    """Normalize a metadata value; return None if value is ambiguous/unknown."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _UNKNOWN_VALUES:
        return None
    if s == "none" and attr not in _NONE_IS_VALID:
        return None
    return s


def _metadata_matches(t1: Tracklet, t2: Tracklet) -> bool:
    """Return False only when both tracklets have conflicting attribute values
    with sufficient confidence.

    Missing confidence (None) is treated as 0.0 (uncertain) — does not block merge.
    Confidence values are logit-derived (geometric mean of token P), so the
    thresholds here are intentionally lower than the legacy self-report regime.
    """
    checks = [
        # binary / presence fields — most reliable conflict signal
        ("gender",        "gender_conf"),
        ("mask_presence", "mask_conf"),
        ("bag_presence",  "bag_conf"),
        ("hat_presence",  "hat_conf"),
        # categorical / color fields
        ("upper_color",   "upper_conf"),
        ("lower_color",   "lower_conf"),
        ("shoes_color",   "shoes_conf"),
        ("hat_color",     "hat_conf"),
        ("hair_color",    "hair_color_conf"),
        ("age_range",     "age_range_conf"),
        # NOTE: upper_type, lower_type, *_desc intentionally excluded — free-text
        # from VLM; "blazer" ≠ "suit jacket" in string comparison.
    ]
    for attr, conf_field in checks:
        v1 = _norm_meta(attr, getattr(t1, attr, None))
        v2 = _norm_meta(attr, getattr(t2, attr, None))
        if v1 is None or v2 is None:
            continue
        if v1 == v2:
            continue
        c1 = getattr(t1, conf_field, None) or 0.0
        c2 = getattr(t2, conf_field, None) or 0.0
        threshold = _CONF_THRESHOLDS.get(attr, _CONF_THRESHOLD)
        if c1 >= threshold and c2 >= threshold:
            return False  # both sides confident about conflicting values
    return True


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


# ── SigLIP query encoder ──────────────────────────────────────────────────────
# Encodes text / image queries into the same 1152-dim space as
# tracklets_embeddings.siglip_embedding. Vectors are L2-normalized so cosine =
# dot product.

# Hard floor for image-query matches. _vec_score maps raw cosine from [-1, 1]
# into [0, 1], so the default 0.5 is roughly raw cosine >= 0.0. A separate
# MIN_FUSION_SCORE gate below filters weak final candidates across all modes.
_QUERY_IMAGE_MIN_VECTOR_SCORE = float(os.getenv("QUERY_IMAGE_MIN_VECTOR_SCORE", "0.5"))


def _encode_query_text_siglip(text_query: str) -> list[float]:
    """Run the SigLIP text tower on `text_query`. Returns an L2-normalized list
    of length 1152, or [] when the model isn't loaded / encoding fails."""
    if not text_query or not text_query.strip():
        return []
    try:
        from app.services.model_warmup import get_model, get_device
    except Exception:
        return []
    model = get_model("siglip2")
    processor = get_model("siglip2_processor")
    if model is None or processor is None:
        return []
    try:
        import torch as _torch
        device = get_device()
        inputs = processor(
            text=[text_query.strip()],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with _torch.no_grad():
            feats = model.get_text_features(**{k: v for k, v in inputs.items() if k != "pixel_values"})
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats[0].detach().cpu().float().tolist()
    except Exception as exc:
        logger.warning("[siglip-text] encode failed: %s", exc)
        return []


def _resolve_query_image_source(image_url: str) -> str:
    raw = (image_url or "").strip()
    if raw.startswith("file://"):
        return raw[len("file://"):]
    static_prefix = "/static/query-images/"
    if raw.startswith(static_prefix):
        filename = Path(raw).name
        return str(Path(os.getenv("QUERY_IMAGE_ROOT", "/workspace/storage/query-images")) / filename)
    if raw.startswith("static/query-images/"):
        filename = Path(raw).name
        return str(Path(os.getenv("QUERY_IMAGE_ROOT", "/workspace/storage/query-images")) / filename)
    return raw


_IMAGE_EXT_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_IMAGE_QUERY_REID_LIMIT = _get_positive_env_int("IMAGE_QUERY_REID_LIMIT", 500)
_IMAGE_QUERY_TEXT_VECTOR_LIMIT = _get_positive_env_int("IMAGE_QUERY_TEXT_VECTOR_LIMIT", 250)
_IMAGE_QUERY_TEXT_PREFILTER_LIMIT = _get_positive_env_int("IMAGE_QUERY_TEXT_PREFILTER_LIMIT", 200)
_IMAGE_TEXT_REID_WEIGHT = float(os.getenv("IMAGE_TEXT_REID_WEIGHT", "0.80"))
_IMAGE_TEXT_SIGLIP_WEIGHT = float(os.getenv("IMAGE_TEXT_SIGLIP_WEIGHT", "0.15"))
_IMAGE_TEXT_METADATA_WEIGHT = float(os.getenv("IMAGE_TEXT_METADATA_WEIGHT", "0.05"))


def _form_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _form_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_image_extension(filename: str | None, content_type: str | None) -> str:
    if content_type:
        ext = _IMAGE_EXT_BY_TYPE.get(content_type.lower())
        if ext:
            return ext
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


async def _save_query_image(upload: StarletteUploadFile) -> tuple[str, str]:
    content_type = (upload.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh cho query_image.")

    content = await upload.read(settings.max_query_image_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Ảnh truy vấn đang trống.")
    if len(content) > settings.max_query_image_bytes:
        raise HTTPException(status_code=413, detail="Ảnh truy vấn không được vượt quá 5MB.")

    settings.query_image_root.mkdir(parents=True, exist_ok=True)
    ext = _query_image_extension(upload.filename, content_type)
    filename = f"{uuid.uuid4().hex}{ext}"
    path = settings.query_image_root / filename
    path.write_bytes(content)
    return f"/static/query-images/{filename}", str(path)


async def _parse_search_request(request: Request) -> SearchRequest:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        camera_ids = [
            str(value).strip()
            for value in form.getlist("camera_ids")
            if str(value).strip()
        ]
        body = SearchRequest(
            query=str(form.get("query") or ""),
            text=str(form.get("text") or "") or None,
            top_k=_form_int(form.get("top_k"), 20),
            offset=_form_int(form.get("offset"), 0),
            camera_ids=camera_ids or None,
            time_from=str(form.get("time_from") or "") or None,
            time_to=str(form.get("time_to") or "") or None,
            persist_query=_form_bool(form.get("persist_query")),
            query_id=str(form.get("query_id") or "") or None,
        )
        query_image = form.get("query_image")
        if isinstance(query_image, StarletteUploadFile) and query_image.filename:
            image_url, image_path = await _save_query_image(query_image)
            body.query_image_url = image_url
            body.query_image_path = image_path
        return body

    try:
        raw_body = await request.json()
    except Exception:
        raw_body = {}
    return SearchRequest(**(raw_body or {}))


def _load_query_image(image_url: str):
    if not image_url:
        return None
    from PIL import Image

    image_source = _resolve_query_image_source(image_url)
    if image_source.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(image_source, timeout=10) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return Image.open(image_source).convert("RGB")


def _crop_query_person_image(img):
    """Return the single-person crop expected by PersonViT.

    Query uploads are currently defined as one-person images, so RT-DETR only
    chooses the highest-confidence person bbox when it can. If the detector
    misses a tight crop, using the full image is still the least surprising
    fallback under that contract.
    """
    try:
        from app.services.model_warmup import get_device, get_model
        import torch as _torch

        model = get_model("rtdetr")
        processor = get_model("rtdetr_processor")
        person_ids = get_model("rtdetr_person_ids") or {0, 1}
        if model is None or processor is None:
            return img

        device = get_device()
        dtype = _torch.float16 if device.type == "cuda" else _torch.float32
        inputs = processor(images=[img], return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        with _torch.no_grad():
            outputs = model(**inputs)
        target_sizes = _torch.tensor([[img.height, img.width]], device=device)
        result = processor.post_process_object_detection(
            outputs,
            threshold=float(os.getenv("QUERY_IMAGE_RTDETR_THRESHOLD", "0.22")),
            target_sizes=target_sizes,
        )[0]
        candidates = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            if int(label.item()) not in person_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.detach().cpu().tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((float(score.item()), (x1, y1, x2, y2)))
        if not candidates:
            logger.info("[personvit-image] RT-DETR found no person; using full image fallback")
            return img
        _, bbox = max(candidates, key=lambda item: item[0])
        return img.crop(bbox)
    except Exception as exc:
        logger.warning("[personvit-image] query crop failed; using full image fallback: %s", exc)
        return img


def _encode_query_image_personvit(image_url: str) -> list[float]:
    """Encode a single-person query image into the 384-dim ReID space."""
    if not image_url:
        return []
    try:
        from app.services.model_warmup import get_device, get_model
        import numpy as _np
        import torch as _torch

        model = get_model("personvit")
        if model is None:
            return []
        img = _load_query_image(image_url)
        if img is None:
            return []
        crop = _crop_query_person_image(img).resize((128, 256))
        arr = _np.asarray(crop, dtype=_np.float32)
        device = get_device()
        dtype = _torch.float16 if device.type == "cuda" else _torch.float32
        tensor = _torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(
            device=device,
            dtype=dtype,
        ) / 255.0
        mean = _torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = _torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with _torch.no_grad():
            out = model(tensor)
        feats = (out[0] if isinstance(out, tuple) else out.embeddings).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats[0].detach().cpu().float().tolist()
    except Exception as exc:
        logger.warning("[personvit-image] encode failed: %s", exc)
        return []


def _encode_query_image_siglip(image_url: str) -> list[float]:
    """Run the SigLIP image tower on an image URL or local path. Returns
    L2-normalized list of length 1152, or [] on failure."""
    if not image_url:
        return []
    try:
        from app.services.model_warmup import get_model, get_device
    except Exception:
        return []
    model = get_model("siglip2")
    processor = get_model("siglip2_processor")
    if model is None or processor is None:
        return []
    try:
        import io
        import torch as _torch
        from PIL import Image
        device = get_device()

        image_source = _resolve_query_image_source(image_url)
        if image_source.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(image_source, timeout=10) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            # Local path (Coolify / static-served crops)
            img = Image.open(image_source).convert("RGB")

        inputs = processor(images=[img], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
        with _torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats[0].detach().cpu().float().tolist()
    except Exception as exc:
        logger.warning("[siglip-image] encode failed: %s", exc)
        return []


def _vec_score(query_vec: list[float], tracklet_vec: list[float]) -> float:
    """Map cosine similarity into [0, 1] for fusion.

    SigLIP/SigLIP-2 text↔image cosine values are not in CLIP's "normalized
    contrastive" regime — the sigmoid training loss leaves typical positive
    matches at only ~0.05–0.2 and unrelated pairs near 0 or slightly
    negative. Clamping at 0 made every fusion fall back to text+quality.
    Linear rescale (c + 1) / 2 preserves ranking and keeps everything in
    [0, 1] without throwing information away."""
    if not query_vec or not tracklet_vec:
        return 0.0
    c = _cosine_sim(query_vec, tracklet_vec)
    return max(0.0, min(1.0, (c + 1.0) / 2.0))


def _identity_score(query_vec: list[float], tracklet_vec: list[float]) -> float:
    """PersonViT same-person score in the native cosine regime [0, 1]."""
    if not query_vec or not tracklet_vec:
        return 0.0
    return max(0.0, min(1.0, _cosine_sim(query_vec, tracklet_vec)))


def _apply_camera_filter(statement, camera_ids: list[str] | None):
    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(func.lower(Tracklet.camera_id).in_(cam_lower))
    return statement


def _scan_vector_tracklets(
    session: Session,
    *,
    query_vec: list[float],
    lane: str,
    camera_ids: list[str] | None,
    time_from: str | None,
    time_to: str | None,
    limit: int,
) -> tuple[list[Tracklet], dict[str, float]]:
    """Exact Python fallback when DB vector operators are unavailable."""
    tf = _parse_dt(time_from)
    tt = _parse_dt(time_to)
    embedding_col = (
        TrackletEmbedding.reid_embedding
        if lane == "reid"
        else TrackletEmbedding.siglip_embedding
    )
    statement = (
        select(Tracklet)
        .join(Video, Tracklet.video_id == Video.video_id)
        .join(TrackletEmbedding, Tracklet.tracklet_id == TrackletEmbedding.tracklet_id)
        .options(
            contains_eager(Tracklet.video),
            contains_eager(Tracklet.embedding),
        )
        .where(embedding_col.is_not(None))
    )
    statement = _apply_camera_filter(statement, camera_ids)
    scored: list[tuple[float, int, Tracklet]] = []
    stream = session.execute(
        statement.execution_options(stream_results=True, yield_per=256)
    ).scalars()
    for row in stream:
        if not _tracklet_overlaps_time_range(row, tf, tt):
            continue
        emb = _tracklet_reid_embedding(row) if lane == "reid" else _tracklet_siglip_embedding(row)
        score = _identity_score(query_vec, emb) if lane == "reid" else _vec_score(query_vec, emb)
        scored.append((score, row.id, row))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    top = scored[:limit]
    return [row for _, _, row in top], {row.tracklet_id: score for score, _, row in top}


def _vector_retrieve_tracklets(
    session: Session,
    *,
    query_vec: list[float],
    lane: str,
    camera_ids: list[str] | None,
    time_from: str | None,
    time_to: str | None,
    limit: int,
) -> tuple[list[Tracklet], dict[str, float]]:
    """Retrieve the best gallery rows for a vector query over the full gallery."""
    if not query_vec:
        return [], {}
    tf = _parse_dt(time_from)
    tt = _parse_dt(time_to)
    embedding_col = (
        TrackletEmbedding.reid_embedding
        if lane == "reid"
        else TrackletEmbedding.siglip_embedding
    )
    try:
        distance = embedding_col.cosine_distance(query_vec)
        statement = (
            select(Tracklet, distance.label("distance"))
            .join(Video, Tracklet.video_id == Video.video_id)
            .join(TrackletEmbedding, Tracklet.tracklet_id == TrackletEmbedding.tracklet_id)
            .options(
                contains_eager(Tracklet.video),
                contains_eager(Tracklet.embedding),
            )
            .where(embedding_col.is_not(None))
            .order_by(distance.asc(), Tracklet.id.desc())
        )
        statement = _apply_camera_filter(statement, camera_ids)
        rows: list[Tracklet] = []
        score_map: dict[str, float] = {}
        stream = session.execute(
            statement.execution_options(stream_results=True, yield_per=256)
        )
        for row, distance_value in stream:
            if not _tracklet_overlaps_time_range(row, tf, tt):
                continue
            emb = _tracklet_reid_embedding(row) if lane == "reid" else _tracklet_siglip_embedding(row)
            if lane == "reid":
                score = _identity_score(query_vec, emb)
            else:
                score = _vec_score(query_vec, emb)
            rows.append(row)
            score_map[row.tracklet_id] = score
            if len(rows) >= limit:
                break
        return rows, score_map
    except Exception as exc:
        logger.warning("[%s-retrieval] DB vector search unavailable; exact Python scan fallback: %s", lane, exc)
        return _scan_vector_tracklets(
            session,
            query_vec=query_vec,
            lane=lane,
            camera_ids=camera_ids,
            time_from=time_from,
            time_to=time_to,
            limit=limit,
        )


def _merge_tracklet_pools(*pools: list[Tracklet]) -> list[Tracklet]:
    merged: list[Tracklet] = []
    seen: set[str] = set()
    for pool in pools:
        for tracklet in pool:
            if tracklet.tracklet_id in seen:
                continue
            seen.add(tracklet.tracklet_id)
            merged.append(tracklet)
    return merged


def _tracklet_siglip_embedding(t: Tracklet) -> list[float]:
    """Return the SigLIP semantic embedding for retrieval (1152-dim), or []."""
    if not t.embedding or t.embedding.siglip_embedding is None:
        return []
    try:
        return list(t.embedding.siglip_embedding)
    except Exception:
        return []


def _tracklet_reid_embedding(t: Tracklet) -> list[float]:
    """Return the identity/Re-ID embedding for grouping, or []."""
    if not t.embedding or t.embedding.reid_embedding is None:
        return []
    try:
        return list(t.embedding.reid_embedding)
    except Exception:
        return []


def _tracklet_identity_embedding(t: Tracklet) -> list[float]:
    """Return the preferred identity vector, with optional legacy fallback."""
    reid = _tracklet_reid_embedding(t)
    if reid:
        return reid
    if _IDENTITY_FALLBACK_TO_SIGLIP:
        return _tracklet_siglip_embedding(t)
    return []


def _tracklet_abs_window(t: Tracklet) -> tuple[float, float] | None:
    """Absolute (start_ts, end_ts) in Unix seconds from video recording time."""
    return tracklet_time_seconds(t)


def _can_merge(t1: Tracklet, t2: Tracklet, *, use_metadata: bool = True) -> bool:
    """Return True if t1 and t2 can belong to the same person identity."""
    # Text-only search preserves the historical metadata guard. Image-led
    # search deliberately ignores it so imperfect VLM labels cannot fracture a
    # visually strong identity match.
    if use_metadata and not _metadata_matches(t1, t2):
        return False
    w1 = _tracklet_abs_window(t1)
    w2 = _tracklet_abs_window(t2)
    if w1 is None or w2 is None:
        return False
    s1, e1 = w1
    s2, e2 = w2
    # Time gap between end of one and start of the other
    gap = max(s1 - e2, s2 - e1, 0.0)
    if gap > _MERGE_MAX_GAP_S:
        return False
    if t1.camera_id == t2.camera_id:
        # Same camera with overlapping time → physically impossible to be same person
        if min(e1, e2) > max(s1, s2):
            return False
    return True


def _merge_by_similarity(
    ranked_tracklets: list[Tracklet],
    *,
    use_metadata: bool = True,
) -> list[list[Tracklet]]:
    """Group tracklets by embedding cosine similarity using union-find with
    a component-level guard.

    Returns groups ordered by best rank (lowest index in ranked_tracklets =
    highest score). The first element of each group is the representative
    (highest-ranked tracklet). Tracklets with no embedding are kept as
    singleton groups.

    The component guard mirrors TrackletFragmentMerger's logic at ingest:
    a candidate pair (i, j) only merges if EVERY cross-pair between the two
    components they'd join clears _MERGE_COMPONENT_FLOOR and the cross-mean
    clears _MERGE_THRESHOLD. This prevents the union-find chain effect that
    pulls in increasingly different people through one-hop edges.
    """
    n = len(ranked_tracklets)
    if n == 0:
        return []
    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    embs = [_tracklet_identity_embedding(t) for t in ranked_tracklets]

    # Build a unit-normalized matrix so cosine is a single dot product.
    # Rows for tracklets without an embedding are zero — they'll never be
    # eligible to merge (sim with anything is 0).
    emb_dim = next((len(e) for e in embs if e), 0)
    if emb_dim == 0:
        # No usable embeddings at all → every tracklet is its own group.
        return [[t] for t in ranked_tracklets]

    emb_matrix = np.zeros((n, emb_dim), dtype=np.float32)
    for idx, e in enumerate(embs):
        if e and len(e) == emb_dim:
            v = np.asarray(e, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                emb_matrix[idx] = v / norm
    sim = emb_matrix @ emb_matrix.T  # [n, n] cosine matrix

    def cross_clears_component_guard(root_a: int, root_b: int) -> bool:
        a_idx = np.fromiter(members[root_a], dtype=np.int64)
        b_idx = np.fromiter(members[root_b], dtype=np.int64)
        cross = sim[np.ix_(a_idx, b_idx)]
        if cross.size == 0:
            return False
        return (
            float(cross.mean()) >= _MERGE_THRESHOLD
            and float(cross.min())  >= _MERGE_COMPONENT_FLOOR
        )

    # Candidate edges first — only the ones that already pass the pair-wise
    # threshold are worth considering. Sorting by descending sim makes the
    # union-find pick the tightest bridges first, which keeps the
    # component-mean stable as the cluster grows.
    candidate_edges: list[tuple[float, int, int]] = []
    for i in range(n):
        if not embs[i]:
            continue
        for j in range(i + 1, n):
            if not embs[j]:
                continue
            s = float(sim[i, j])
            if s < _MERGE_THRESHOLD:
                continue
            candidate_edges.append((s, i, j))
    candidate_edges.sort(reverse=True)

    for _, i, j in candidate_edges:
        pi, pj = find(i), find(j)
        if pi == pj:
            continue
        if not _can_merge(ranked_tracklets[i], ranked_tracklets[j], use_metadata=use_metadata):
            continue
        if not cross_clears_component_guard(pi, pj):
            continue
        # Keep the earliest-rank member as the root for stable output order.
        if min(members[pj]) < min(members[pi]):
            pi, pj = pj, pi
        parent[pj] = pi
        members[pi].extend(members.pop(pj))

    root_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        root_to_idxs[find(i)].append(i)

    # Each group is sorted by rank (ascending index = best score first)
    return [
        [ranked_tracklets[i] for i in sorted(idxs)]
        for idxs in sorted(root_to_idxs.values(), key=min)
    ]


def _split_group_by_24h_windows(
    group: list[Tracklet],
    rank_index: dict[str, int],
) -> list[list[Tracklet]]:
    """Split an identity group into candidate windows anchored at first tracklet.

    A candidate covers 24 hours starting at its earliest real tracklet time. If
    another same-identity tracklet starts after that 24h window, it becomes the
    first tracklet of a new candidate.
    """
    timed: list[tuple[float, Tracklet]] = []
    unknown: list[Tracklet] = []
    for tracklet in group:
        window = _tracklet_abs_window(tracklet)
        if window is None:
            unknown.append(tracklet)
            continue
        timed.append((window[0], tracklet))

    timed.sort(key=lambda item: (item[0], rank_index.get(item[1].tracklet_id, 10**9)))

    windows: list[list[Tracklet]] = []
    current: list[Tracklet] = []
    current_start: float | None = None

    def flush_current() -> None:
        if not current:
            return
        current.sort(key=lambda t: rank_index.get(t.tracklet_id, 10**9))
        windows.append(list(current))

    for start_ts, tracklet in timed:
        if current_start is None or start_ts > current_start + _MERGE_MAX_GAP_S:
            flush_current()
            current = [tracklet]
            current_start = start_ts
            continue
        current.append(tracklet)

    flush_current()
    for tracklet in unknown:
        windows.append([tracklet])

    windows.sort(key=lambda items: min(rank_index.get(t.tracklet_id, 10**9) for t in items))
    return windows


def _split_groups_by_24h_windows(
    groups: list[list[Tracklet]],
    ranked_tracklets: list[Tracklet],
) -> list[list[Tracklet]]:
    rank_index = {
        tracklet.tracklet_id: idx
        for idx, tracklet in enumerate(ranked_tracklets)
    }
    split: list[list[Tracklet]] = []
    for group in groups:
        split.extend(_split_group_by_24h_windows(group, rank_index))
    return split


@router.post("")
async def search_candidates(
    request: Request,
    current_user_id: int = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Search candidates from JSON or multipart input with self-contained auth."""
    body = await _parse_search_request(request)
    body.user_id = current_user_id

    request_t0 = time.perf_counter()
    query = body.query or body.text or ""
    top_k = body.top_k
    offset = body.offset
    camera_ids = body.camera_ids
    time_from = body.time_from
    time_to = body.time_to

    db = SessionLocal()
    try:
        # If the frontend passes an existing query_id and it belongs to the
        # current user, reuse it — that way "Tiếp theo" pagination doesn't
        # spawn a duplicate query_history row per page. Otherwise mint a new
        # qid and create a fresh history entry.
        qh: QueryHistory | None = None
        if body.query_id:
            qh = db.query(QueryHistory).filter(
                QueryHistory.query_id == body.query_id,
                QueryHistory.user_id == body.user_id,
            ).one_or_none()

        if qh is not None:
            qid = qh.query_id
            reused_history = True
        else:
            qid = str(uuid.uuid4())
            reused_history = False

        effective_query_image_url = body.query_image_url or (qh.query_image_url if qh else None)
        effective_query_image_source = body.query_image_path or body.query_image_url or effective_query_image_url
        if qh is not None and body.query_image_url and not qh.query_image_url:
            qh.query_image_url = body.query_image_url

        logger.info(
            "[query:%s] start user_id=%s top_k=%d offset=%d camera_ids=%s "
            "time_from=%s time_to=%s image_query=%s reused_history=%s query=%r",
            qid,
            body.user_id,
            top_k,
            offset,
            camera_ids,
            time_from,
            time_to,
            bool(effective_query_image_source),
            reused_history,
            query,
        )

        # Translate if Vietnamese
        translate_t0 = time.perf_counter()
        search_query = _translate_query(query) if query else ""
        logger.info(
            "[query:%s] translation elapsed=%.3fs translated=%s search_query=%r",
            qid,
            time.perf_counter() - translate_t0,
            search_query != query,
            search_query,
        )

        # ── Luồng 20.5: Create / reuse QueryHistory record ──────────────
        if qh is None:
            qh = QueryHistory(
                query_id=qid,
                user_id=body.user_id,
                query_text=query or "",
                status="searching",
                query_image_url=effective_query_image_url or None,
            )
            db.add(qh)
            db.flush()  # FK constraint: query_candidates.query_id → query_history.query_id

        # Parse query into structured constraints once. Text-only mode keeps the
        # historical behavior; image modes only use parsed metadata when text is
        # actually present, never for image-only retrieval.
        parsed_query = parse_query_metadata(search_query)
        logger.info("[query:%s] parsed_metadata=%s", qid, _jdump(_parsed_query_log_dict(parsed_query)))
        image_query = bool(effective_query_image_source)
        has_text_query = bool(search_query.strip())
        text_has_meaning = not parsed_query.is_empty()
        if not image_query and not text_has_meaning:
            qh.status = "candidates_found"
            qh.result_count = 0
            db.commit()
            logger.info(
                "[query:%s] empty structured text query without image; returning no candidates elapsed=%.3fs",
                qid,
                time.perf_counter() - request_t0,
            )
            return {"results": [], "query_id": qid}

        active_parsed_query = parsed_query if (has_text_query and text_has_meaning) else ParsedQueryMetadata()

        # Encode independently so image+text can use both lanes at once:
        #   - PersonViT = identity / same-person retrieval
        #   - SigLIP text tower = semantic retrieval for textual constraints
        personvit_query_vec: list[float] = []
        text_query_vec: list[float] = []
        encode_t0 = time.perf_counter()
        if image_query:
            personvit_query_vec = _encode_query_image_personvit(effective_query_image_source)
            if personvit_query_vec:
                logger.info("[search] query encoded via PersonViT image lane")
        if has_text_query:
            text_query_vec = _encode_query_text_siglip(search_query)
            if text_query_vec:
                logger.info("[search] query encoded via SigLIP text tower (len=%d)", len(search_query))
        logger.info(
            "[query:%s] vector_encode personvit_dim=%d siglip_text_dim=%d elapsed=%.3fs",
            qid,
            len(personvit_query_vec),
            len(text_query_vec),
            time.perf_counter() - encode_t0,
        )

        if image_query and not personvit_query_vec and not has_text_query:
            qh.status = "candidates_found"
            qh.result_count = 0
            db.commit()
            logger.warning(
                "[query:%s] image-only query could not be encoded by PersonViT; returning no candidates elapsed=%.3fs",
                qid,
                time.perf_counter() - request_t0,
            )
            return {"results": [], "query_id": qid}

        # Stage A — candidate-pool construction.
        # text-only: preserve the existing text shortlist behavior.
        # image-only: retrieve the whole gallery by PersonViT only.
        # image+text: union PersonViT retrieval with text metadata and SigLIP
        #             retrieval so text can enrich recall without owning rank.
        prefilter_t0 = time.perf_counter()
        shortlist: list[Tracklet] = []
        text_score_map: dict[str, float] = {}
        per_tracklet_image_score: dict[str, float] = {}
        per_tracklet_text_vec_score: dict[str, float] = {}

        if image_query and personvit_query_vec and not has_text_query:
            search_mode = "image_only"
            shortlist, per_tracklet_image_score = _vector_retrieve_tracklets(
                db,
                query_vec=personvit_query_vec,
                lane="reid",
                camera_ids=camera_ids,
                time_from=time_from,
                time_to=time_to,
                limit=_IMAGE_QUERY_REID_LIMIT,
            )
            text_score_map = {row.tracklet_id: 0.0 for row in shortlist}
        elif image_query and personvit_query_vec:
            search_mode = "image_text"
            image_rows, image_scores = _vector_retrieve_tracklets(
                db,
                query_vec=personvit_query_vec,
                lane="reid",
                camera_ids=camera_ids,
                time_from=time_from,
                time_to=time_to,
                limit=_IMAGE_QUERY_REID_LIMIT,
            )
            text_rows, _ = _local_prefilter(
                db,
                search_query,
                camera_ids,
                time_from,
                time_to,
                limit=_IMAGE_QUERY_TEXT_PREFILTER_LIMIT,
                parsed=active_parsed_query,
            )
            text_vec_rows, _ = _vector_retrieve_tracklets(
                db,
                query_vec=text_query_vec,
                lane="siglip",
                camera_ids=camera_ids,
                time_from=time_from,
                time_to=time_to,
                limit=_IMAGE_QUERY_TEXT_VECTOR_LIMIT,
            ) if text_query_vec else ([], {})
            shortlist = _merge_tracklet_pools(image_rows, text_vec_rows, text_rows)
            cleaned_search_query = search_query.strip().lower()
            query_tokens = {token for token in cleaned_search_query.split() if token}
            for row in shortlist:
                per_tracklet_image_score[row.tracklet_id] = image_scores.get(
                    row.tracklet_id,
                    _identity_score(personvit_query_vec, _tracklet_reid_embedding(row)),
                )
                per_tracklet_text_vec_score[row.tracklet_id] = _vec_score(
                    text_query_vec,
                    _tracklet_siglip_embedding(row),
                ) if text_query_vec else 0.0
                text_score_map[row.tracklet_id] = _score_search_text(
                    _build_search_text(row),
                    cleaned_search_query,
                    query_tokens,
                ) if cleaned_search_query else 0.0
            shortlist.sort(
                key=lambda row: (
                    -per_tracklet_image_score.get(row.tracklet_id, 0.0),
                    -per_tracklet_text_vec_score.get(row.tracklet_id, 0.0),
                    -row.id,
                )
            )
        else:
            search_mode = "text_only"
            shortlist, text_score_map = _local_prefilter(
                db,
                search_query,
                camera_ids,
                time_from,
                time_to,
                limit=200,
                parsed=active_parsed_query,
            )
            if text_query_vec:
                for row in shortlist:
                    per_tracklet_text_vec_score[row.tracklet_id] = _vec_score(
                        text_query_vec,
                        _tracklet_siglip_embedding(row),
                    )

        logger.info(
            "[query:%s] pool mode=%s rows=%d elapsed=%.3fs text_score_stats=%s image_score_stats=%s text_vec_stats=%s",
            qid,
            search_mode,
            len(shortlist),
            time.perf_counter() - prefilter_t0,
            _score_stats([float(v) for v in text_score_map.values()]),
            _score_stats([float(v) for v in per_tracklet_image_score.values()]),
            _score_stats([float(v) for v in per_tracklet_text_vec_score.values()]),
        )
        if shortlist and logger.isEnabledFor(logging.DEBUG):
            preview = [
                _tracklet_log_item(
                    t,
                    text_score=text_score_map.get(t.tracklet_id, 0.0),
                    vector_score=(
                        per_tracklet_image_score.get(t.tracklet_id, 0.0)
                        if search_mode.startswith("image")
                        else per_tracklet_text_vec_score.get(t.tracklet_id, 0.0)
                    ),
                )
                for t in shortlist[:_QUERY_LOG_TOP_N]
            ]
            logger.debug("[query:%s] pool_top_%d=%s", qid, len(preview), _jdump(preview))

        if not shortlist:
            qh.status = "candidates_found"
            qh.result_count = 0
            db.commit()
            logger.info(
                "[query:%s] completed raw_tracklets=0 merged_candidates=0 returned=0 elapsed=%.3fs",
                qid,
                time.perf_counter() - request_t0,
            )
            return {"results": [], "query_id": qid}

        # Keep one "primary vector" map for diagnostics/member links while the
        # scoring stage below still has access to the separate image/text maps.
        per_tracklet_vec_score = (
            per_tracklet_image_score
            if search_mode.startswith("image")
            else per_tracklet_text_vec_score
        )
        # Stage C — Identity merge (union-find on dedicated Re-ID vectors).
        merge_t0 = time.perf_counter()
        identity_groups = _merge_by_similarity(
            shortlist,
            use_metadata=(search_mode == "text_only"),
        )
        groups = _split_groups_by_24h_windows(identity_groups, shortlist)
        merged_groups = [g for g in groups if len(g) > 1]
        logger.info(
            "[query:%s] merge identity_groups=%d candidate_windows=%d merged_windows=%d singletons=%d max_group_size=%d elapsed=%.3fs",
            qid,
            len(identity_groups),
            len(groups),
            len(merged_groups),
            len(groups) - len(merged_groups),
            max((len(g) for g in groups), default=0),
            time.perf_counter() - merge_t0,
        )
        if merged_groups and logger.isEnabledFor(logging.DEBUG):
            # Cosine recomputation per (rep, member) is expensive on a 1152-d
            # vector in pure Python (~3 ms / pair) so we gate the entire
            # preview behind DEBUG. INFO-level summary above is enough for
            # production health monitoring.
            group_preview = []
            for group in merged_groups[:_QUERY_LOG_GROUP_N]:
                rep_emb = _tracklet_identity_embedding(group[0])
                member_items = []
                for member in group[:_QUERY_LOG_MEMBER_N]:
                    member_item = _tracklet_log_item(
                        member,
                        text_score=text_score_map.get(member.tracklet_id, 0.0),
                        vector_score=(
                            per_tracklet_vec_score.get(member.tracklet_id, 0.0)
                            if per_tracklet_vec_score else None
                        ),
                    )
                    member_emb = _tracklet_identity_embedding(member)
                    member_item["merge_sim_to_rep"] = (
                        round(_cosine_sim(rep_emb, member_emb), 4)
                        if rep_emb and member_emb else None
                    )
                    member_items.append(member_item)
                group_preview.append({
                    "size": len(group),
                    "rep_tracklet": group[0].tracklet_id,
                    "rep_camera": group[0].camera_id,
                    "members": member_items,
                })
            logger.debug("[query:%s] merged_group_preview=%s", qid, _jdump(group_preview))

        # Stage C.5 — Per-tracklet action labels (Tier A: soft bonus, no filter).
        # Aggregated to a deduped set per candidate so a query mentioning
        # "running" can boost a candidate whose member tracklets include
        # walking + running, even when the representative tracklet was sitting.
        query_actions = set(active_parsed_query.actions)
        action_by_tracklet: dict[str, str] = {}
        if shortlist:
            tracklet_ids = [t.tracklet_id for t in shortlist]
            act_rows = db.execute(
                select(TrackletAction.tracklet_id, TrackletAction.action_label)
                .where(TrackletAction.tracklet_id.in_(tracklet_ids))
                .where(TrackletAction.confidence >= ACTION_CONFIDENCE_FLOOR)
            ).all()
            action_by_tracklet = {tid: label for tid, label in act_rows}
        logger.info(
            "[query:%s] actions query_actions=%s loaded_tracklet_actions=%d confidence_floor=%.2f",
            qid,
            sorted(query_actions),
            len(action_by_tracklet),
            ACTION_CONFIDENCE_FLOOR,
        )

        # Stage D — Score each candidate.
        # text_only keeps the historical fusion. Image modes are deliberately
        # separate:
        #   image_only -> PersonViT similarity only
        #   image_text -> 0.80 PersonViT + 0.15 SigLIP(text) + 0.05 metadata
        # This makes image identity the gravitational center whenever an image
        # is supplied, while still letting text break near-ties usefully.
        score_t0 = time.perf_counter()
        merged: list[dict] = []
        filtered_by_query_vector = 0
        filtered_by_fusion_score = 0
        for group in groups:
            rep = group[0]
            candidate_id = _candidate_id_for_group(qid, group)
            text_score = float(text_score_map.get(rep.tracklet_id, 0.0))
            quality_score = float(rep.quality_score or 0.0)

            member_image_scores = [
                per_tracklet_image_score.get(m.tracklet_id, 0.0) for m in group
            ]
            image_top_n = sorted(member_image_scores, reverse=True)[:3]
            image_score = sum(image_top_n) / len(image_top_n) if image_top_n else 0.0

            member_text_vec_scores = [
                per_tracklet_text_vec_score.get(m.tracklet_id, 0.0) for m in group
            ]
            text_vec_top_n = sorted(member_text_vec_scores, reverse=True)[:3]
            text_vec_score = (
                sum(text_vec_top_n) / len(text_vec_top_n) if text_vec_top_n else 0.0
            )

            # Candidate actions = union over member tracklets, deduped.
            candidate_actions: set[str] = set()
            for member in group:
                label = action_by_tracklet.get(member.tracklet_id)
                if label:
                    candidate_actions.add(label)
            matched_actions = candidate_actions & query_actions if query_actions else set()
            action_bonus = ACTION_BONUS_SCORE if matched_actions else 0.0

            # Metadata signal aggregated across the entire candidate identity.
            matched_metadata: dict[str, list[str]] = {}
            metadata_bonus = 0.0

            def _candidate_field_values(field_name: str) -> set[str]:
                vals: set[str] = set()
                for m in group:
                    v = getattr(m, field_name, None)
                    if v is None:
                        continue
                    s = str(v).strip().lower()
                    if s and s not in {"unknown", "none", "null", "n/a"}:
                        vals.add(s)
                return vals

            field_specs = [
                ("upper_color", active_parsed_query.upper_color),
                ("lower_color", active_parsed_query.lower_color),
                ("shoes_color", active_parsed_query.shoes_color),
                ("hat_color",   active_parsed_query.hat_color),
            ]
            for col_name, requested in field_specs:
                if not requested:
                    continue
                cand_vals = _candidate_field_values(col_name)
                hits = [v for v in requested if v in cand_vals]
                if hits:
                    matched_metadata[col_name] = hits
                    metadata_bonus += METADATA_BONUS_PER_MATCH * len(hits)

            if active_parsed_query.unbound_colors:
                upper_vals = _candidate_field_values("upper_color")
                lower_vals = _candidate_field_values("lower_color")
                unbound_hits = [
                    v for v in active_parsed_query.unbound_colors
                    if v in upper_vals or v in lower_vals
                ]
                if unbound_hits:
                    matched_metadata["unbound_color"] = unbound_hits
                    metadata_bonus += METADATA_BONUS_PER_MATCH * len(unbound_hits)

            if active_parsed_query.gender:
                cand_genders = _candidate_field_values("gender")
                if any(g in cand_genders for g in active_parsed_query.gender):
                    matched_metadata["gender"] = list(active_parsed_query.gender)
                    metadata_bonus += METADATA_BONUS_PER_MATCH

            metadata_bonus_applied = min(metadata_bonus, METADATA_MAX_BONUS) if metadata_bonus > 0.0 else 0.0

            if search_mode == "image_only":
                fusion_score = image_score
                score_mode = "personvit_image"
                vector_score = image_score
            elif search_mode == "image_text":
                metadata_den = max(1e-9, METADATA_MAX_BONUS + ACTION_BONUS_SCORE)
                metadata_signal = min(
                    1.0,
                    (metadata_bonus_applied + action_bonus) / metadata_den,
                )
                fusion_score = (
                    _IMAGE_TEXT_REID_WEIGHT * image_score
                    + _IMAGE_TEXT_SIGLIP_WEIGHT * text_vec_score
                    + _IMAGE_TEXT_METADATA_WEIGHT * metadata_signal
                )
                score_mode = "personvit_image_siglip_metadata"
                vector_score = image_score
            elif text_query_vec:
                fusion_score = (
                    0.65 * text_vec_score + 0.20 * text_score + 0.15 * quality_score
                )
                fusion_score += action_bonus + metadata_bonus_applied
                score_mode = "vector_text_quality"
                vector_score = text_vec_score
            else:
                fusion_score = 0.7 * text_score + 0.3 * quality_score
                fusion_score += action_bonus + metadata_bonus_applied
                score_mode = "text_quality"
                vector_score = None

            fusion_score = round(fusion_score, 4)

            # Image retrieval is a ranked search problem, not a binary accept/
            # reject problem; always return the best available candidates. The
            # legacy min_fusion gate remains only for text-only search.
            if (
                search_mode == "text_only"
                and settings.min_fusion_score > 0.0
                and fusion_score < settings.min_fusion_score
            ):
                filtered_by_fusion_score += 1
                continue

            member_links = []
            for member in group:
                if search_mode.startswith("image"):
                    match_score = per_tracklet_image_score.get(member.tracklet_id, 0.0)
                elif text_query_vec:
                    match_score = per_tracklet_text_vec_score.get(member.tracklet_id, 0.0)
                else:
                    match_score = float(text_score_map.get(member.tracklet_id, 0.0))
                member_links.append({
                    "tracklet_id": member.tracklet_id,
                    "match_score": round(match_score, 4),
                })

            merged.append({
                "candidate_id": candidate_id,
                "group": group,
                "rep": rep,
                "fusion_score": fusion_score,
                "text_score": text_score,
                "vector_score": round(vector_score, 4) if vector_score is not None else None,
                "image_score": round(image_score, 4) if search_mode.startswith("image") else None,
                "text_vector_score": round(text_vec_score, 4) if text_query_vec else None,
                "quality_score": round(quality_score, 4),
                "score_mode": score_mode,
                "action_bonus": round(action_bonus, 4),
                "metadata_bonus": round(metadata_bonus_applied, 4),
                "member_links": member_links,
                "actions": sorted(candidate_actions),
                "matched_actions": sorted(matched_actions),
                "matched_metadata": matched_metadata,
            })

        merged.sort(
            key=lambda item: (
                -item["fusion_score"],
                -(item.get("image_score") or 0.0),
                -item["rep"].id,
            )
        )
        logger.info(
            "[query:%s] score candidates=%d filtered_vector=%d filtered_fusion=%d min_fusion=%.3f elapsed=%.3fs mode=%s",
            qid,
            len(merged),
            filtered_by_query_vector,
            filtered_by_fusion_score,
            settings.min_fusion_score,
            time.perf_counter() - score_t0,
            search_mode,
        )

        # Persist the top MAX_CANDIDATES ranked set for history, but return only
        # the requested page to the active search UI. The history page has its
        # own pagination, so users can come back later and browse every saved
        # candidate without forcing the search page to render them all at once.
        ranked_candidates = merged[: settings.max_candidates]
        paged = ranked_candidates[offset:offset + top_k]
        for rank_idx, item in enumerate(ranked_candidates, start=1):
            rep = item["rep"]
            candidate_id = item["candidate_id"]
            candidate_key = _candidate_key_for_group(item["group"])
            candidate_insert = pg_insert(QueryCandidate).values(
                query_id=qid,
                candidate_id=candidate_id,
                fusion_score=item["fusion_score"],
                vector_score=item["vector_score"],
                text_score=item["text_score"],
                rank_position=rank_idx,
                primary_camera_id=rep.camera_id or "",
                appearance_summary=rep.appearance_summary or "",
                gender=rep.gender or "unknown",
                top_color=rep.upper_color or "unknown",
                bottom_color=rep.lower_color or "unknown",
                candidate_key=candidate_key,
                preview_url=_preview_url_for(rep),
            )
            db.execute(candidate_insert.on_conflict_do_update(
                index_elements=["candidate_id"],
                set_={
                    "query_id": qid,
                    "fusion_score": item["fusion_score"],
                    "vector_score": item["vector_score"],
                    "text_score": item["text_score"],
                    "rank_position": rank_idx,
                    "primary_camera_id": rep.camera_id or "",
                    "appearance_summary": rep.appearance_summary or "",
                    "gender": rep.gender or "unknown",
                    "top_color": rep.upper_color or "unknown",
                    "bottom_color": rep.lower_color or "unknown",
                    "candidate_key": candidate_key,
                    "preview_url": _preview_url_for(rep),
                },
            ))

            for link in item["member_links"]:
                link_insert = pg_insert(QueryCandidateTracklet).values(
                    candidate_id=candidate_id,
                    tracklet_id=link["tracklet_id"],
                    match_score=link["match_score"],
                    match_type="vector",
                )
                db.execute(link_insert.on_conflict_do_update(
                    index_elements=["candidate_id", "tracklet_id"],
                    set_={
                        "match_score": link["match_score"],
                        "match_type": "vector",
                    },
                ))

        qh.status = "candidates_found"
        qh.result_count = len(ranked_candidates)
        db.commit()
        # INFO-level: one compact line per ranked candidate so health/QA can
        # spot-check ranking without parsing the per-member DEBUG payload.
        top_summary = [
            {
                "rank": offset + idx + 1,
                "cand": mc["candidate_id"],
                "fusion": mc["fusion_score"],
                "vec": mc["vector_score"],
                "text": round(float(mc["text_score"] or 0.0), 4),
                "qual": mc["quality_score"],
                "act_bonus": mc["action_bonus"],
                "meta_bonus": mc["metadata_bonus"],
                "matched_actions": mc["matched_actions"],
                "matched_metadata": mc["matched_metadata"],
                "n_tracklets": len(mc["group"]),
                "rep_cam": mc["rep"].camera_id,
            }
            for idx, mc in enumerate(paged)
        ]
        logger.info("[query:%s] top_%d=%s", qid, len(top_summary), _jdump(top_summary))

        if paged and logger.isEnabledFor(logging.DEBUG):
            top_detail = []
            for rank_idx, mc in enumerate(paged, start=offset + 1):
                rep = mc["rep"]
                member_by_id = {m.tracklet_id: m for m in mc["group"]}
                top_member_links = sorted(
                    mc["member_links"],
                    key=lambda link: link["match_score"],
                    reverse=True,
                )[:_QUERY_LOG_MEMBER_N]
                top_members = []
                for link in top_member_links:
                    member = member_by_id.get(link["tracklet_id"])
                    if member is None:
                        top_members.append(link)
                        continue
                    member_item = _tracklet_log_item(
                        member,
                        text_score=text_score_map.get(member.tracklet_id, 0.0),
                        vector_score=(
                            per_tracklet_vec_score.get(member.tracklet_id, 0.0)
                            if per_tracklet_vec_score else None
                        ),
                    )
                    member_item["match_score"] = link["match_score"]
                    top_members.append(member_item)
                top_detail.append({
                    "rank": rank_idx,
                    "candidate_id": mc["candidate_id"],
                    "fusion": mc["fusion_score"],
                    "vector": mc["vector_score"],
                    "text": round(float(mc["text_score"] or 0.0), 4),
                    "quality": mc["quality_score"],
                    "mode": mc["score_mode"],
                    "action_bonus": mc["action_bonus"],
                    "metadata_bonus": mc["metadata_bonus"],
                    "tracklet_count": len(mc["group"]),
                    "rep": _tracklet_log_item(
                        rep,
                        text_score=text_score_map.get(rep.tracklet_id, 0.0),
                        vector_score=(
                            per_tracklet_vec_score.get(rep.tracklet_id, 0.0)
                            if per_tracklet_vec_score else None
                        ),
                    ),
                    "matched_actions": mc["matched_actions"],
                    "matched_metadata": mc["matched_metadata"],
                    "members": top_members,
                })
            logger.debug("[query:%s] top_detail=%s", qid, _jdump(top_detail))

        results = []
        for mc in paged:
            rep = mc["rep"]
            candidate_id = mc["candidate_id"]
            tracklet_count = len(mc["group"])
            description = translate_to_vietnamese_cached(rep.appearance_summary or "")
            tracklet_summaries = []
            for member in mc["group"]:
                window = tracklet_time_window(member)
                tracklet_summaries.append({
                    "tracklet_id": member.tracklet_id,
                    "camera_id": member.camera_id,
                    "time_start": window[0].isoformat() if window else None,
                    "time_end": window[1].isoformat() if window else None,
                })
            results.append({
                "id": candidate_id,
                "thumbnail_url": _preview_url_for(rep),
                "description": description,
                "tracklet_count": tracklet_count,
                "tracklets": tracklet_summaries,
                "actions": mc["actions"],
                "matched_actions": mc["matched_actions"],
                "matched_metadata": mc["matched_metadata"],
                "_raw": {
                    "candidate_id": candidate_id,
                    "tracklet_count": tracklet_count,
                    "camera_id": rep.camera_id,
                    "appearance_summary": rep.appearance_summary,
                    "actions": mc["actions"],
                    "matched_actions": mc["matched_actions"],
                    "matched_metadata": mc["matched_metadata"],
                },
                "query_id": qid,
            })

        logger.info(
            "[query:%s] completed raw_tracklets=%d merged_candidates=%d returned=%d elapsed=%.3fs",
            qid,
            len(shortlist),
            len(merged),
            len(results),
            time.perf_counter() - request_t0,
        )
        return {"results": results, "query_id": qid}

    finally:
        db.close()
