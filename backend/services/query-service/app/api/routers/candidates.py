"""Candidates search router for query-service.

Flow:
1. metadata-service (/api/v1/search) forwards request here
2. Local text pre-filter → shortlist candidates (sorted by text relevance)
3. Union-find merge: group tracklets by cosine similarity + metadata + temporal/camera guards
4. Save QueryCandidate + QueryCandidateTracklet rows, format and return results
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
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
    user_id: int = 1  # injected by metadata-service from JWT; fallback=1 for direct calls

from shared.database import SessionLocal
from shared.models import (
    QueryCandidate, QueryCandidateTracklet, QueryHistory,
    Tracklet, TrackletAction, Video,
)
from app.services.translation import detect_vietnamese, translate_to_english, warmup as warmup_translation
from app.services.query_metadata_parse import (
    parse_query_metadata,
    ParsedQueryMetadata,
    ACTION_CONFIDENCE_FLOOR,
    ACTION_BONUS_SCORE,
    METADATA_BONUS_PER_MATCH,
    METADATA_MAX_BONUS,
)
from sqlalchemy import func, or_, select, text
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

    # Join Video so we can filter by absolute recording timestamp.
    # Absolute tracklet time = video.recorded_at + start/end_time (seconds).
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
    if tf:
        # Tracklet still active at time_from:
        # video.recorded_at + end_time seconds >= time_from
        statement = statement.where(
            text("videos.recorded_at + (tracklets.end_time * interval '1 second') >= :tf")
            .bindparams(tf=tf)
        )
    if tt:
        # Tracklet started before time_to:
        # video.recorded_at + start_time seconds <= time_to
        statement = statement.where(
            text("videos.recorded_at + (tracklets.start_time * interval '1 second') <= :tt")
            .bindparams(tt=tt)
        )

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
        rows = session.scalars(statement.limit(limit)).all()
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



# ── Identity merge: cosine similarity + temporal/camera guards + union-find ───

_MERGE_THRESHOLD = float(os.getenv("QUERY_MERGE_THRESHOLD", "0.98"))
# SigLIP2 image embeddings of arbitrary person crops cluster very tightly
# (cosine 0.85–1.0 across different people), so the 0.85 used at ingest's
# fragment-merge stage (where temporal/spatial guards do the heavy lifting)
# collapses entire query shortlists into one candidate at search time.
# Cross-camera identity merge needs a much stricter floor — 0.97 keeps the
# same-person bridges while breaking the noisy "everyone looks like everyone"
# edges. Tunable via QUERY_MERGE_THRESHOLD env without redeploy.
_MERGE_MAX_GAP_S = 86400.0   # max 24-hour gap — matches trace window
_CONF_THRESHOLD = 0.70        # fallback: below this = uncertain → don't block merge

# Per-attribute confidence thresholds: both sides must exceed to block merge.
# These are calibrated for LOGIT-DERIVED confidences (geometric mean of token
# probabilities under Qwen2-VL), not self-reported numbers. Logit confs are
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

        if image_url.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(image_url, timeout=10) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            # Local path (Coolify / static-served crops)
            img = Image.open(image_url).convert("RGB")

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


def _tracklet_embedding(t: Tracklet) -> list[float]:
    """Return the SigLIP2 embedding for this tracklet (1152-dim), or []."""
    if not t.embedding or t.embedding.siglip_embedding is None:
        return []
    try:
        return list(t.embedding.siglip_embedding)
    except Exception:
        return []


def _tracklet_abs_window(t: Tracklet) -> tuple[float, float]:
    """Absolute (start_ts, end_ts) in Unix seconds. Uses video.recorded_at loaded via contains_eager."""
    try:
        base = t.video.recorded_at.timestamp() if (t.video and t.video.recorded_at) else 0.0
    except Exception:
        base = 0.0
    return base + float(t.start_time or 0.0), base + float(t.end_time or 0.0)


def _can_merge(t1: Tracklet, t2: Tracklet) -> bool:
    """Return True if t1 and t2 can belong to the same person identity."""
    # Check metadata first — cheapest way to reject obviously different people
    if not _metadata_matches(t1, t2):
        return False
    s1, e1 = _tracklet_abs_window(t1)
    s2, e2 = _tracklet_abs_window(t2)
    # Time gap between end of one and start of the other
    gap = max(s1 - e2, s2 - e1, 0.0)
    if gap > _MERGE_MAX_GAP_S:
        return False
    if t1.camera_id == t2.camera_id:
        # Same camera with overlapping time → physically impossible to be same person
        if min(e1, e2) > max(s1, s2):
            return False
    return True


def _merge_by_similarity(ranked_tracklets: list[Tracklet]) -> list[list[Tracklet]]:
    """Group tracklets by embedding cosine similarity using union-find.

    Returns groups ordered by best rank (lowest index in ranked_tracklets = highest score).
    The first element of each group is the representative (highest-ranked tracklet).
    Tracklets with no embedding are kept as singleton groups.
    """
    n = len(ranked_tracklets)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    embs = [_tracklet_embedding(t) for t in ranked_tracklets]

    for i in range(n):
        if not embs[i]:
            continue
        for j in range(i + 1, n):
            if find(i) == find(j) or not embs[j]:
                continue
            if _cosine_sim(embs[i], embs[j]) < _MERGE_THRESHOLD:
                continue
            if not _can_merge(ranked_tracklets[i], ranked_tracklets[j]):
                continue
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

    root_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        root_to_idxs[find(i)].append(i)

    # Each group is sorted by rank (ascending index = best score first)
    return [
        [ranked_tracklets[i] for i in sorted(idxs)]
        for idxs in sorted(root_to_idxs.values(), key=min)
    ]


@router.post("")
def search_candidates(body: SearchRequest) -> dict[str, Any]:
    """Search candidates with GPU re-ranking. Accepts JSON body."""
    request_t0 = time.perf_counter()
    query = body.query or body.text or ""
    top_k = body.top_k
    offset = body.offset
    camera_ids = body.camera_ids
    time_from = body.time_from
    time_to = body.time_to

    db = SessionLocal()
    try:
        qid = str(uuid.uuid4())
        logger.info(
            "[query:%s] start user_id=%s top_k=%d offset=%d camera_ids=%s "
            "time_from=%s time_to=%s image_query=%s query=%r",
            qid,
            body.user_id,
            top_k,
            offset,
            camera_ids,
            time_from,
            time_to,
            bool(body.query_image_url),
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

        # ── Luồng 20.5: Create QueryHistory record ──────────────────────
        qh = QueryHistory(
            query_id=qid,
            user_id=body.user_id,
            query_text=query or "",
            status="searching",
            query_image_url=body.query_image_url or None,
        )
        db.add(qh)
        db.flush()  # FK constraint: query_candidates.query_id → query_history.query_id

        # Parse query into structured constraints once (gender / colors /
        # garments / actions). Drives both the gender hard-filter at the
        # prefilter stage and the metadata bonus during rerank.
        parsed_query = parse_query_metadata(search_query)
        logger.info("[query:%s] parsed_metadata=%s", qid, _jdump(_parsed_query_log_dict(parsed_query)))

        # Stage A — Text-shortlist (SQL ILIKE on materialized attributes).
        # Recall up to 200 candidates. This is still text-based, so it can miss
        # tracklets whose appearance_summary phrasing doesn't share tokens with
        # the query — Stage B (SigLIP rerank below) compensates by re-scoring
        # ALL shortlisted items in a shared text↔image embedding space.
        prefilter_t0 = time.perf_counter()
        shortlist, text_score_map = _local_prefilter(
            db, search_query, camera_ids, time_from, time_to, limit=200,
            parsed=parsed_query,
        )
        logger.info(
            "[query:%s] prefilter rows=%d elapsed=%.3fs text_score_stats=%s",
            qid,
            len(shortlist),
            time.perf_counter() - prefilter_t0,
            _score_stats([float(v) for v in text_score_map.values()]),
        )
        if shortlist and logger.isEnabledFor(logging.DEBUG):
            preview = [
                _tracklet_log_item(t, text_score=text_score_map.get(t.tracklet_id, 0.0))
                for t in shortlist[:_QUERY_LOG_TOP_N]
            ]
            logger.debug("[query:%s] prefilter_top_%d=%s", qid, len(preview), _jdump(preview))

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

        # Stage B — Encode the query once with SigLIP. Image takes priority
        # over text when both are supplied; text is the fallback.
        query_vec: list[float] = []
        query_vec_source: str | None = None
        encode_t0 = time.perf_counter()
        if body.query_image_url:
            query_vec = _encode_query_image_siglip(body.query_image_url)
            if query_vec:
                query_vec_source = "image"
                logger.info("[search] query encoded via SigLIP image tower")
        if not query_vec and search_query:
            query_vec = _encode_query_text_siglip(search_query)
            if query_vec:
                query_vec_source = "text"
                logger.info("[search] query encoded via SigLIP text tower (len=%d)", len(search_query))
        if query_vec:
            logger.info(
                "[query:%s] vector_encode source=%s dim=%d elapsed=%.3fs",
                qid,
                query_vec_source,
                len(query_vec),
                time.perf_counter() - encode_t0,
            )
        else:
            logger.warning(
                "[query:%s] vector_encode unavailable; fallback=text_quality elapsed=%.3fs",
                qid,
                time.perf_counter() - encode_t0,
            )

        # Pre-compute vec-score per tracklet for the whole shortlist.
        # If the SigLIP encoder is unavailable or query is empty, all scores
        # default to 0 and ranking falls back to text + quality.
        per_tracklet_vec_score: dict[str, float] = {}
        if query_vec:
            vec_t0 = time.perf_counter()
            for t in shortlist:
                emb = _tracklet_embedding(t)
                per_tracklet_vec_score[t.tracklet_id] = _vec_score(query_vec, emb)
            vec_scores = list(per_tracklet_vec_score.values())
            logger.info(
                "[query:%s] vector_scores stats=%s elapsed=%.3fs",
                qid,
                _score_stats(vec_scores),
                time.perf_counter() - vec_t0,
            )
            if logger.isEnabledFor(logging.DEBUG):
                top_vec_tracklets = sorted(
                    shortlist,
                    key=lambda t: per_tracklet_vec_score.get(t.tracklet_id, 0.0),
                    reverse=True,
                )[:_QUERY_LOG_TOP_N]
                logger.debug(
                    "[query:%s] vector_top_%d=%s",
                    qid,
                    len(top_vec_tracklets),
                    _jdump([
                        _tracklet_log_item(
                            t,
                            text_score=text_score_map.get(t.tracklet_id, 0.0),
                            vector_score=per_tracklet_vec_score.get(t.tracklet_id, 0.0),
                        )
                        for t in top_vec_tracklets
                    ]),
                )

        # Stage C — Identity merge (union-find on tracklet-tracklet SigLIP sim).
        merge_t0 = time.perf_counter()
        groups = _merge_by_similarity(shortlist)
        merged_groups = [g for g in groups if len(g) > 1]
        logger.info(
            "[query:%s] merge groups=%d merged_groups=%d singletons=%d max_group_size=%d elapsed=%.3fs",
            qid,
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
                rep_emb = _tracklet_embedding(group[0])
                member_items = []
                for member in group[:_QUERY_LOG_MEMBER_N]:
                    member_item = _tracklet_log_item(
                        member,
                        text_score=text_score_map.get(member.tracklet_id, 0.0),
                        vector_score=per_tracklet_vec_score.get(member.tracklet_id, 0.0)
                        if query_vec else None,
                    )
                    member_emb = _tracklet_embedding(member)
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
        query_actions = set(parsed_query.actions)
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
        # New fusion (when query_vec available):
        #   fusion = 0.65 * vec_q          ← cosine(query, group representative-vec)
        #          + 0.20 * text_overlap   ← legacy SQL token overlap
        #          + 0.15 * quality
        # Fallback (no query_vec): 0.7 * text_overlap + 0.3 * quality (old behaviour).
        score_t0 = time.perf_counter()
        merged: list[dict] = []
        for group in groups:
            rep = group[0]
            candidate_id = rep.tracklet_id if len(group) == 1 else str(uuid.uuid4())
            text_score = float(text_score_map.get(rep.tracklet_id, 0.0))
            quality_score = float(rep.quality_score or 0.0)

            # vec_score = average of the top-3 member vec-scores w.r.t. query.
            # Identity covered by multiple high-scoring tracklets gets boosted;
            # a singleton with one weak hit gets dragged down.
            member_vec_scores = [
                per_tracklet_vec_score.get(m.tracklet_id, 0.0) for m in group
            ]
            top_n = sorted(member_vec_scores, reverse=True)[:3]
            vec_score = sum(top_n) / len(top_n) if top_n else 0.0

            if query_vec:
                fusion_score = (
                    0.65 * vec_score + 0.20 * text_score + 0.15 * quality_score
                )
            else:
                fusion_score = 0.7 * text_score + 0.3 * quality_score

            # Candidate actions = union over member tracklets, deduped.
            # Skipping members below ACTION_CONFIDENCE_FLOOR (handled in the
            # SQL load above) keeps low-confidence VideoMAE labels from
            # polluting the set.
            candidate_actions: set[str] = set()
            for member in group:
                label = action_by_tracklet.get(member.tracklet_id)
                if label:
                    candidate_actions.add(label)

            # Tier A bonus: at least one matching action → small score nudge,
            # never a hard filter. Helps surface candidates whose action
            # matches even when appearance similarity is borderline.
            matched_actions = candidate_actions & query_actions if query_actions else set()
            action_bonus = ACTION_BONUS_SCORE if matched_actions else 0.0
            if matched_actions:
                fusion_score += action_bonus

            # Per-field metadata bonus: each parsed (field, value) that matches
            # at least one member tracklet of this candidate adds
            # METADATA_BONUS_PER_MATCH. This is what disambiguates "red shirt"
            # from "red shoes" — only the candidate whose upper_color == "red"
            # gets the upper_color bonus, regardless of token overlap or vector
            # noise. Capped at METADATA_MAX_BONUS so a heavily-tagged query
            # can't dominate vec_score.
            matched_metadata: dict[str, list[str]] = {}
            metadata_bonus = 0.0

            def _candidate_field_values(field_name: str) -> set[str]:
                """Distinct non-empty lowercased values of a Tracklet column
                across all members of this candidate group."""
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
                ("upper_color", parsed_query.upper_color),
                ("lower_color", parsed_query.lower_color),
                ("shoes_color", parsed_query.shoes_color),
                ("hat_color",   parsed_query.hat_color),
            ]
            for col_name, requested in field_specs:
                if not requested:
                    continue
                cand_vals = _candidate_field_values(col_name)
                hits = [v for v in requested if v in cand_vals]
                if hits:
                    matched_metadata[col_name] = hits
                    metadata_bonus += METADATA_BONUS_PER_MATCH * len(hits)

            # Unbound colors ("red dress" with dress ambiguous, or a bare
            # color word) match if they appear in upper OR lower of any member.
            if parsed_query.unbound_colors:
                upper_vals = _candidate_field_values("upper_color")
                lower_vals = _candidate_field_values("lower_color")
                unbound_hits = [
                    v for v in parsed_query.unbound_colors
                    if v in upper_vals or v in lower_vals
                ]
                if unbound_hits:
                    matched_metadata["unbound_color"] = unbound_hits
                    metadata_bonus += METADATA_BONUS_PER_MATCH * len(unbound_hits)

            # Gender is already a hard filter at the prefilter stage, but
            # surviving tracklets with matching gender deserve a small bonus
            # to nudge them above unknown-gender tracklets that slipped past
            # the filter.
            if parsed_query.gender:
                cand_genders = _candidate_field_values("gender")
                if any(g in cand_genders for g in parsed_query.gender):
                    matched_metadata["gender"] = list(parsed_query.gender)
                    metadata_bonus += METADATA_BONUS_PER_MATCH

            metadata_bonus_applied = 0.0
            if metadata_bonus > 0.0:
                metadata_bonus_applied = min(metadata_bonus, METADATA_MAX_BONUS)
                fusion_score += metadata_bonus_applied

            fusion_score = round(fusion_score, 4)

            # member_links: per-tracklet score within the candidate (for evidence UI).
            # Now reflects the member's own vec-similarity to the query when available,
            # not the artificial rep↔member cosine.
            member_links = []
            for member in group:
                if query_vec:
                    match_score = per_tracklet_vec_score.get(member.tracklet_id, 0.0)
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
                "vector_score": round(vec_score, 4) if query_vec else None,
                "quality_score": round(quality_score, 4),
                "score_mode": "vector_text_quality" if query_vec else "text_quality",
                "action_bonus": round(action_bonus, 4),
                "metadata_bonus": round(metadata_bonus_applied, 4),
                "member_links": member_links,
                "actions": sorted(candidate_actions),
                "matched_actions": sorted(matched_actions),
                "matched_metadata": matched_metadata,
            })

        merged.sort(key=lambda item: (-item["fusion_score"], -item["rep"].id))
        logger.info(
            "[query:%s] score candidates=%d elapsed=%.3fs mode=%s",
            qid,
            len(merged),
            time.perf_counter() - score_t0,
            "vector_text_quality" if query_vec else "text_quality",
        )

        for rank_idx, item in enumerate(merged, start=1):
            rep = item["rep"]
            candidate_id = item["candidate_id"]
            db.execute(pg_insert(QueryCandidate).values(
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
            ).on_conflict_do_nothing(index_elements=["candidate_id"]))

            for link in item["member_links"]:
                db.execute(pg_insert(QueryCandidateTracklet).values(
                    candidate_id=candidate_id,
                    tracklet_id=link["tracklet_id"],
                    match_score=link["match_score"],
                    match_type="vector",
                ).on_conflict_do_nothing(index_elements=["candidate_id", "tracklet_id"]))

        paged = merged[offset:offset + top_k]
        qh.status = "candidates_found"
        qh.result_count = len(paged)
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
                        vector_score=per_tracklet_vec_score.get(member.tracklet_id, 0.0)
                        if query_vec else None,
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
                        vector_score=per_tracklet_vec_score.get(rep.tracklet_id, 0.0)
                        if query_vec else None,
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
            description = rep.appearance_summary or ""
            if tracklet_count > 1:
                description = f"[{tracklet_count} tracklets] {description}".strip()
            results.append({
                "id": candidate_id,
                "thumbnail_url": f"/candidates/{rep.tracklet_id}/preview",
                "description": description,
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
