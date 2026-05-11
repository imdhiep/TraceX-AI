"""Candidates search router for query-service.

Flow:
1. metadata-service (/api/v1/search) forwards request here
2. Vector-first recall: encode query with SigLIP 2 text tower → cosine vs siglip_embedding → top-N
   (falls back to text-only prefilter when SigLIP 2 unavailable or query_emb unavailable)
3. Python text rerank: score each candidate's metadata text vs query tokens
4. Union-find merge: group tracklets by SigLIP2 cosine + metadata + temporal/camera guards
5. Fusion: 0.5*text + 0.3*quality + 0.2*vector + merge_boost
6. Paginate → INSERT only candidates on current page → return results
"""

from __future__ import annotations

import heapq
import logging
import math
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
from shared.models import QueryCandidate, QueryCandidateTracklet, QueryHistory, Tracklet, TrackletAction, Video
from app.services.translation import detect_vietnamese, translate_to_english, warmup as warmup_translation
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
    action_label = ""
    if row.actions:
        action_label = " ".join(a.action_label or "" for a in row.actions)
    return " ".join([
        row.appearance_summary or "",
        row.gender or "",
        row.upper_clothing_color or "",
        row.lower_clothing_color or "",
        row.shoes_color or "",
        row.age_range or "",
        getattr(row, "hat_color", "") or "",
        getattr(row, "bag_type", "") or "",
        getattr(row, "is_wearing_mask", "") or "",
        getattr(row, "hair_style", "") or "",
        getattr(row, "hair_color", "") or "",
        action_label,
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
    action_subq = (
        select(TrackletAction.action_label)
        .where(TrackletAction.tracklet_id == Tracklet.tracklet_id)
        .order_by(TrackletAction.id.desc())
        .limit(1)
        .correlate(Tracklet)
        .scalar_subquery()
    )
    return func.lower(
        func.concat_ws(
            " ",
            func.coalesce(Tracklet.appearance_summary, ""),
            func.coalesce(Tracklet.gender, ""),
            func.coalesce(Tracklet.upper_clothing_color, ""),
            func.coalesce(Tracklet.lower_clothing_color, ""),
            func.coalesce(Tracklet.shoes_color, ""),
            func.coalesce(Tracklet.age_range, ""),
            func.coalesce(Tracklet.hat_color, ""),
            func.coalesce(Tracklet.bag_type, ""),
            func.coalesce(Tracklet.is_wearing_mask, ""),
            func.coalesce(Tracklet.hair_style, ""),
            func.coalesce(Tracklet.hair_color, ""),
            func.coalesce(action_subq, ""),
        )
    )


def _ilike_contains(expr, value: str):
    escaped = (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return expr.ilike(f"%{escaped}%", escape="\\")


def _vector_recall(
    session: Session,
    query_emb: list[float],
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    recall_limit: int = 500,
) -> list[Tracklet]:
    """Vector-first recall: top-N by SigLIP2 cosine similarity.

    Returns tracklets ordered by descending vector similarity to query_emb.
    Falls back to empty list if query_emb is unavailable.
    """
    if not query_emb:
        return []

    # base statement with camera/time filters
    statement = (
        select(Tracklet)
        .join(Video, Tracklet.video_id == Video.video_id)
        .options(
            contains_eager(Tracklet.video),
            joinedload(Tracklet.embedding),
            joinedload(Tracklet.actions),
        )
    )

    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(func.lower(Tracklet.camera_id).in_(cam_lower))

    tf = _parse_dt(time_from)
    tt = _parse_dt(time_to)
    if tf:
        statement = statement.where(
            text("videos.recorded_at + (tracklets.end_time * interval '1 second') >= :tf")
            .bindparams(tf=tf)
        )
    if tt:
        statement = statement.where(
            text("videos.recorded_at + (tracklets.start_time * interval '1 second') <= :tt")
            .bindparams(tt=tt)
        )

    rows: list[Tracklet] = session.scalars(statement.limit(recall_limit * 3)).all()

    scored: list[tuple[float, int, Tracklet]] = []
    for row in rows:
        emb = _tracklet_embedding(row)
        if not emb:
            continue
        score = _cosine_sim(query_emb, emb)
        scored.append((score, row.id, row))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [row for _, _, row in scored[:recall_limit]]


def _local_prefilter(
    session: Session,
    query_text: str,
    query_emb: list[float],
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 200,
) -> tuple[list[Tracklet], dict[str, float], bool]:
    """Pre-filter tracklets.

    Returns (shortlist, text_score_map, used_vector_recall).
    used_vector_recall=True means vector-first recall was used (siglip text encoding available).
    Falls back to text-only prefilter when query_emb is empty or SigLIP unavailable.
    """
    cleaned_query = (query_text or "").strip().lower()

    # ── Vector-first path ─────────────────────────────────────────────────────
    if query_emb:
        shortlist = _vector_recall(
            session, query_emb, camera_ids, time_from, time_to,
            recall_limit=max(limit * 2, 400),
        )
        if shortlist:
            # Text rerank: score each candidate on metadata text vs query tokens
            text_score_map: dict[str, float] = {}
            scored: list[tuple[float, int, Tracklet]] = []
            for row in shortlist:
                st = _build_search_text(row)
                ts = _score_search_text(st, cleaned_query, set(cleaned_query.split()))
                text_score_map[row.tracklet_id] = ts
                scored.append((ts, row.id, row))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            shortlist = [row for _, _, row in scored]
            return shortlist, text_score_map, True

    # ── Text-only fallback path ───────────────────────────────────────────────
    statement = (
        select(Tracklet)
        .join(Video, Tracklet.video_id == Video.video_id)
        .options(
            contains_eager(Tracklet.video),
            joinedload(Tracklet.embedding),
            joinedload(Tracklet.actions),
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
        statement = statement.where(
            text("videos.recorded_at + (tracklets.end_time * interval '1 second') >= :tf")
            .bindparams(tf=tf)
        )
    if tt:
        statement = statement.where(
            text("videos.recorded_at + (tracklets.start_time * interval '1 second') <= :tt")
            .bindparams(tt=tt)
        )

    if not cleaned_query:
        rows = session.scalars(statement.limit(limit)).all()
        return rows, {row.tracklet_id: 0.0 for row in rows}, False

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
        return shortlist, score_map, False

    rows = session.scalars(statement.limit(limit)).all()
    return rows, {row.tracklet_id: 0.0 for row in rows}, False



# ── Identity merge: cosine similarity + temporal/camera guards + union-find ───

_MERGE_THRESHOLD = 0.85       # SigLIP2 cosine similarity to consider same identity
_MERGE_MAX_GAP_S = 86400.0   # max 24-hour gap — matches trace window
_CONF_THRESHOLD = 0.70        # fallback: below this = uncertain → don't block merge

# Per-attribute confidence thresholds: both sides must exceed to block merge.
# Free-text fields (upper_clothing_type, *_desc) are NOT in this list — exact
# string match would incorrectly treat "blazer" vs "suit jacket" as a conflict.
_CONF_THRESHOLDS: dict[str, float] = {
    "gender":               0.70,
    "is_wearing_mask":      0.70,
    "bag_presence":         0.75,
    "hat_presence":         0.75,
    "age_range":            0.75,
    "upper_clothing_color": 0.82,
    "lower_clothing_color": 0.82,
    "shoes_color":          0.82,
    "hat_color":            0.82,
    "hair_color":           0.82,
}

# "none" is a meaningful value (model confirmed absence) for these fields
_NONE_IS_VALID = frozenset({"hat_color", "bag_type", "is_wearing_mask"})
_UNKNOWN_VALUES = frozenset({"", "unknown", "null", "n/a", "not sure"})


def _norm_meta(attr: str, value: object) -> str | None:
    """Normalize a metadata value; return None if value is ambiguous/unknown."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _UNKNOWN_VALUES:
        return None
    # "none" only counts as a real value for fields that explicitly track absence
    if s == "none" and attr not in _NONE_IS_VALID:
        return None
    return s


def _metadata_matches(t1: Tracklet, t2: Tracklet) -> bool:
    """Return False only when both tracklets have conflicting attribute values with sufficient confidence.

    Missing confidence (None) is treated as 0.0 (uncertain) — does not block merge.
    """
    checks = [
        # binary / presence fields — most reliable conflict signal
        ("gender",               "gender_conf"),
        ("is_wearing_mask",      "mask_conf"),
        ("bag_presence",         "bag_conf"),
        ("hat_presence",         "hat_conf"),
        # color fields — VLM self-reported confidence
        ("upper_clothing_color", "upper_clothing_conf"),
        ("lower_clothing_color", "lower_clothing_conf"),
        ("shoes_color",          "shoes_conf"),
        ("hat_color",            "hat_conf"),
        ("hair_color",           "hair_conf"),
        ("age_range",            "age_range_conf"),
        # NOTE: upper_clothing_type, lower_clothing_type, *_desc intentionally
        # excluded — free-text from VLM; "blazer" ≠ "suit jacket" in string
        # comparison but may refer to the same garment.
    ]
    for attr, conf_field in checks:
        v1 = _norm_meta(attr, getattr(t1, attr, None))
        v2 = _norm_meta(attr, getattr(t2, attr, None))
        if v1 is None or v2 is None:
            continue  # one side unknown → not a conflict
        if v1 == v2:
            continue
        # Values differ → check confidence; missing conf → treat as uncertain.
        # VLM often returns 0.0 as a literal placeholder when it didn't replace the
        # template value — 0.0 is indistinguishable from "no confidence" and must not
        # block a merge. Threshold is raised only when VLM clearly replaced the value.
        c1 = getattr(t1, conf_field, None) or 0.0
        c2 = getattr(t2, conf_field, None) or 0.0
        if c1 <= 0.05 or c2 <= 0.05:
            continue  # at least one side is uncertain → don't block
        threshold = _CONF_THRESHOLDS.get(attr, _CONF_THRESHOLD)
        if c1 >= threshold and c2 >= threshold:
            return False  # both sides confident about conflicting values

    # Action conflict check — VideoMAE labels
    if t1.actions and t2.actions:
        a1 = t1.actions[0].action_label
        a2 = t2.actions[0].action_label
        if a1 and a2 and a1 != a2:
            conf1 = t1.actions[0].confidence or 0.0
            conf2 = t2.actions[0].confidence or 0.0
            action_threshold = 0.80
            if conf1 >= action_threshold and conf2 >= action_threshold:
                return False  # both sides confident about different actions

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


def _build_query_embedding(query_text: str) -> list[float]:
    """
    Encode query text into a SigLIP 2 text embedding (1152-dim, same space as siglip_embedding).

    Falls back to empty list when SigLIP 2 is unavailable (no GPU / model load failed).
    """
    if not query_text:
        return []

    try:
        import numpy as np
        import torch
        from ..services.model_warmup import get_model

        model = get_model("siglip2")
        processor = get_model("siglip2_processor")
        if model is None or processor is None:
            return []

        device = next(model.parameters()).device
        inputs = processor(text=[query_text], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            text_emb = model.get_text_features(**{k: v for k, v in inputs.items()
                                                  if k in ["input_ids", "attention_mask"]})
        vec = text_emb[0].cpu().float().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()  # 1152-dim — matches siglip_embedding
    except Exception:
        return []


def _tracklet_embedding(t: Tracklet) -> list[float]:
    """Return SigLIP2 embedding for cosine similarity; empty list if unavailable."""
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
    query = body.query or body.text or ""
    top_k = body.top_k
    offset = body.offset
    camera_ids = body.camera_ids
    time_from = body.time_from
    time_to = body.time_to

    db = SessionLocal()
    try:
        # Translate if Vietnamese
        search_query = _translate_query(query) if query else ""

        # ── Luồng 20.5: Create QueryHistory record ──────────────────────
        qid = str(uuid.uuid4())
        qh = QueryHistory(
            query_id=qid,
            user_id=body.user_id,
            query_text=query or "",
            status="searching",
            query_image_url=body.query_image_url or None,
        )
        db.add(qh)
        db.flush()  # FK constraint: query_candidates.query_id → query_history.query_id

        # Encode query once using SigLIP 2 text tower (same embedding space as siglip_embedding)
        query_emb = _build_query_embedding(search_query)

        # Local pre-filter (vector-first when SigLIP 2 available, text-only fallback)
        shortlist, text_score_map, _used_vector = _local_prefilter(
            db, search_query, query_emb, camera_ids, time_from, time_to, limit=200
        )

        if not shortlist:
            qh.status = "candidates_found"
            qh.result_count = 0
            db.commit()
            return {"results": [], "query_id": qid}

        # Union-find merge: group tracklets belonging to the same person identity
        groups = _merge_by_similarity(shortlist)

        # Build candidates, compute fusion scores
        merged: list[dict] = []
        for group in groups:
            rep = group[0]
            candidate_id = rep.tracklet_id if len(group) == 1 else str(uuid.uuid4())
            rep_emb = _tracklet_embedding(rep)
            text_score = float(text_score_map.get(rep.tracklet_id, 0.0))
            quality_score = float(rep.quality_score or 0.0)
            # vector_score: cosine between query text embedding and candidate appearance embedding
            vector_score = _cosine_sim(query_emb, rep_emb) if (query_emb and rep_emb) else 0.0
            # merge_boost: multi-tracklet group signals cross-camera/cross-time identity evidence
            merge_boost = round(0.05 * (len(group) - 1), 4) if len(group) > 1 else 0.0
            fusion_score = round(
                (0.5 * text_score) + (0.3 * quality_score) + (0.2 * vector_score) + merge_boost, 4
            )

            member_links = []
            for member in group:
                if member.tracklet_id == rep.tracklet_id:
                    match_score = 1.0
                else:
                    member_emb = _tracklet_embedding(member)
                    match_score = (
                        _cosine_sim(rep_emb, member_emb)
                        if rep_emb and member_emb
                        else float(member.quality_score or 0.0)
                    )
                member_links.append({
                    "tracklet_id": member.tracklet_id,
                    "match_score": match_score,
                })

            merged.append({
                "candidate_id": candidate_id,
                "group": group,
                "rep": rep,
                "fusion_score": fusion_score,
                "text_score": text_score,
                "vector_score": vector_score,
                "merge_boost": merge_boost,
                "member_links": member_links,
            })

        merged.sort(key=lambda item: (-item["fusion_score"], -item["rep"].id))

        # Paginate BEFORE INSERT — only persist candidates on the current page
        paged = merged[offset:offset + top_k]

        for rank_idx, item in enumerate(paged, start=offset + 1):
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
            ).on_conflict_do_nothing(index_elements=["candidate_id"]))

            for link in item["member_links"]:
                db.execute(pg_insert(QueryCandidateTracklet).values(
                    candidate_id=candidate_id,
                    tracklet_id=link["tracklet_id"],
                    match_score=link["match_score"],
                    match_type="vector",
                ).on_conflict_do_nothing(index_elements=["candidate_id", "tracklet_id"]))

        qh.status = "candidates_found"
        qh.result_count = len(paged)
        db.commit()

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
                "_raw": {
                    "candidate_id": candidate_id,
                    "tracklet_count": tracklet_count,
                    "camera_id": rep.camera_id,
                    "appearance_summary": rep.appearance_summary,
                },
                "query_id": qid,
            })

        logger.debug(
            "search completed: query_id=%s raw_tracklets=%d merged_candidates=%d results=%d",
            qid, len(shortlist), len(merged), len(results),
        )
        return {"results": results, "query_id": qid}

    finally:
        db.close()
