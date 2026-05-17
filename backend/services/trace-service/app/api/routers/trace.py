"""Trace API router - select candidate, build trace, feedback."""

from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import settings
from ...core.models import (
    Camera,
    CameraEdge,
    EvidenceTracklet,
    EvidenceVideo,
    QueryCandidate,
    QueryCandidateTracklet,
    QueryHistory,
    Tracklet,
    VerifiedObject,
    VerifiedObjectTracklet,
)
from ...core.schemas import (
    BuildTraceRequest,
    BuildTraceResponse,
    CandidateDetailResponse,
    CandidateTrackletAction,
    CandidateTrackletEmbeddingInfo,
    CandidateTrackletPreview,
    ContinueTraceRequest,
    ContinueTraceResponse,
    SelectCandidateRequest,
    SelectCandidateResponse,
    TraceCandidateDetailRequest,
    TraceFeedbackRequest,
    TraceFeedbackResponse,
    TraceSegmentResponse,
    TraceStatusResponse,
    TraceTimelineResponse,
)
from ...database import get_session
from ...services.trace_service import TraceService
from backend.services.shared.tracklet_time import tracklet_time_window

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _build_trace_service(session: Session) -> TraceService:
    """Build trace service instance."""
    return TraceService(session)


def _get_query(session: Session, query_id: UUID | str) -> QueryHistory | None:
    return session.scalar(select(QueryHistory).where(QueryHistory.query_id == str(query_id)))


def _get_candidate(session: Session, candidate_id: str) -> QueryCandidate | None:
    return session.scalar(select(QueryCandidate).where(QueryCandidate.candidate_id == str(candidate_id)))


def _build_candidate_id_list(
    primary_candidate_id: str,
    candidate_ids_raw: Any,
) -> list[str]:
    candidate_ids: list[str] = []
    extra_ids = candidate_ids_raw if isinstance(candidate_ids_raw, list) else []
    for raw_id in [primary_candidate_id, *extra_ids]:
        candidate_id = str(raw_id).strip()
        if candidate_id and candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)
    return candidate_ids


@router.post("/select", response_model=SelectCandidateResponse)
def select_candidate(
    request: SelectCandidateRequest,
    session: SessionDep,
) -> SelectCandidateResponse:
    """Select a candidate as the target for tracing.

    This marks the candidate as selected and allows building a trace from it.
    """
    service = _build_trace_service(session)

    # Verify query exists
    query = _get_query(session, request.query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    # Verify candidate exists and belongs to query
    candidate = _get_candidate(session, request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.query_id != str(request.query_id):
        raise HTTPException(status_code=400, detail="Candidate does not belong to this query")

    # Deselect all other candidates for this query
    service.deselect_other_candidates(request.query_id)

    # Select this candidate
    now = datetime.now(timezone.utc)
    candidate.is_selected = True
    candidate.selected_at = now

    # Update query status
    query.selected_candidate_id = str(request.candidate_id)
    query.updated_at = now

    session.commit()

    return SelectCandidateResponse(
        success=True,
        query_id=request.query_id,
        candidate_id=request.candidate_id,
        is_selected=True,
        selected_at=now,
        message="Candidate selected successfully",
    )


@router.post("/build", response_model=BuildTraceResponse)
def build_trace(
    session: SessionDep,
    background_tasks: BackgroundTasks,
    body: dict[str, Any] = Body(...),
) -> BuildTraceResponse:
    """Build a trace from one or more selected candidates.

    Returns immediately with `video_clip_url=None` on each segment. Clip
    rendering runs in a background task; the frontend polls /trace/status
    (or /trace/timeline) until every segment has a URL. This split was
    required because rendering 40+-tracklet candidates took minutes and was
    exceeding the upstream proxy's response timeout.
    """
    try:
        request = BuildTraceRequest(**body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    service = _build_trace_service(session)

    # Verify query exists
    query = _get_query(session, request.query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    candidate_ids = _build_candidate_id_list(request.candidate_id, body.get("candidate_ids"))
    if not candidate_ids:
        raise HTTPException(status_code=400, detail="No candidates selected")

    candidates: list[QueryCandidate] = []
    for candidate_id in candidate_ids:
        candidate = _get_candidate(session, candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")
        if candidate.query_id != str(request.query_id):
            raise HTTPException(status_code=400, detail="Candidate does not belong to this query")
        candidates.append(candidate)

    primary_candidate_id = candidate_ids[0]
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}

    # Keep one trace result per query. Tracing another candidate replaces the
    # previous evidence for the same query.
    service.delete_query_evidence(request.query_id)

    # QueryCandidateTracklet already defines candidate membership. For a
    # multi-candidate trace, union those memberships, drop explicit exclusions,
    # then sort by real wall-clock time before applying the normal trace window.
    excluded_raw = body.get("excluded_tracklet_ids")
    excluded_tracklet_ids = {
        str(tid)
        for tid in (excluded_raw if isinstance(excluded_raw, list) else [])
    }
    tracklet_by_id: dict[str, Tracklet] = {}
    for candidate_id in candidate_ids:
        candidate = candidate_by_id[candidate_id]
        for tracklet in service.get_candidate_tracklets(
            candidate_id=candidate_id,
            allow_fallback=bool(candidate.preview_url),
        ):
            if tracklet.tracklet_id in excluded_tracklet_ids:
                continue
            tracklet_by_id.setdefault(tracklet.tracklet_id, tracklet)

    tracklets = service.sort_tracklets_by_time(list(tracklet_by_id.values()))

    if not tracklets:
        raise HTTPException(status_code=404, detail="No tracklets found for candidate")

    window_start, window_end = service.candidate_time_window(
        tracklets,
        hours=request.time_window_hours,
        fallback_start=query.created_at,
    )
    tracklets = service.filter_tracklets_by_time_window(
        tracklets,
        time_window_start=window_start,
        time_window_end=window_end,
    )
    if not tracklets:
        raise HTTPException(status_code=404, detail="No tracklets found for candidate in time window")

    # Build segment metadata only — clip rendering happens in a background
    # task after the response is sent.
    segments = service.build_trace_segments(
        tracklets,
        query_id=request.query_id,
        candidate_id=primary_candidate_id,
        render_clips=False,
    )

    # Calculate trace confidence
    trace_confidence = service.calculate_trace_confidence(segments)

    # Create evidence video record
    evidence = service.create_evidence_video(
        query_id=request.query_id,
        candidate_id=primary_candidate_id,
        segments=segments,
        trace_confidence=trace_confidence,
        time_window_start=window_start,
        time_window_end=window_end,
    )

    # Mark the query as completed so /history can show a "Đã truy vết" badge
    # even before the background render finishes — the evidence row already
    # exists at this point.
    now = datetime.now(timezone.utc)
    service.deselect_other_candidates(request.query_id)
    for candidate in candidates:
        candidate.is_selected = True
        candidate.selected_at = now
    query.status = "completed"
    query.selected_candidate_id = primary_candidate_id
    query.updated_at = now

    session.commit()

    # Kick off background render for every pending clip. The worker dedupes
    # in-flight evidence IDs and is safe to invoke even if all clips are
    # already cached on disk.
    from ...services.render_worker import render_evidence_clips
    background_tasks.add_task(render_evidence_clips, evidence.id)

    # Build response
    segment_responses = [
        TraceSegmentResponse(
            segment_order=seg["segment_order"],
            tracklet_id=seg["tracklet_id"],
            camera_id=seg["camera_id"],
            time_start=seg["time_start"],
            time_end=seg["time_end"],
            duration_seconds=seg["duration_seconds"],
            thumbnail_url=seg.get("thumbnail_url"),
            video_clip_url=seg.get("video_clip_url"),
            confidence=seg.get("confidence"),
        )
        for seg in segments
    ]

    total_duration = sum(s["duration_seconds"] or 0 for s in segments if s["duration_seconds"])

    return BuildTraceResponse(
        success=True,
        query_id=request.query_id,
        candidate_id=primary_candidate_id,
        evidence_id=evidence.id,
        trace_duration_ms=int(total_duration * 1000) if total_duration else None,
        trace_confidence=trace_confidence,
        segment_count=len(segments),
        total_duration_seconds=int(total_duration),
        segments=segment_responses,
        merged_video_url=(evidence.video_url or None) if request.merge_videos else None,
        time_window_start=window_start,
        time_window_end=window_end,
    )


@router.get("/status/{evidence_id}", response_model=TraceStatusResponse)
def get_trace_status(
    evidence_id: int,
    session: SessionDep,
    background_tasks: BackgroundTasks,
) -> TraceStatusResponse:
    """Get trace status by evidence ID.

    Computes progress live from `evidence_tracklets.video_clip_url` and a full
    ffmpeg decode validation of the `/static/traces/...` MP4. A non-empty URL is
    still pending if the MP4 is missing, incomplete, or not playable. As a
    self-healing measure, pending evidence is re-enqueued for rendering.
    """
    evidence = session.get(EvidenceVideo, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    service = _build_trace_service(session)
    rows = (
        session.query(EvidenceTracklet)
        .filter(EvidenceTracklet.evidence_video_id == evidence_id)
        .all()
    )
    total = len(rows)
    rendered = 0
    cleared_stale_url = False
    for row in rows:
        if service.is_clip_url_ready(row.video_clip_url):
            rendered += 1
        elif row.video_clip_url:
            row.video_clip_url = ""
            cleared_stale_url = True
    if cleared_stale_url:
        session.commit()

    if total == 0 or rendered == total:
        status_str = "completed" if total > 0 else "pending"
    elif rendered == 0:
        status_str = "pending"
    else:
        status_str = "rendering"

    if status_str != "completed":
        from ...services.render_worker import render_evidence_clips
        background_tasks.add_task(render_evidence_clips, evidence.id)

    return TraceStatusResponse(
        evidence_id=evidence.id,
        query_id=evidence.query_id,
        status=status_str,
        trace_confidence=evidence.trace_confidence,
        segment_count=evidence.segment_count,
        rendered_segments=rendered,
        total_segments=total,
        created_at=evidence.created_at,
        updated_at=evidence.created_at,
    )


@router.get("/timeline/{evidence_id}", response_model=TraceTimelineResponse)
def get_trace_timeline(
    evidence_id: int,
    session: SessionDep,
    background_tasks: BackgroundTasks,
) -> TraceTimelineResponse:
    """Get trace timeline with camera path."""
    service = _build_trace_service(session)

    evidence = session.get(EvidenceVideo, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    segments = service.get_trace_segments(evidence_id)
    if session.dirty:
        session.commit()
    if any(not seg.get("video_clip_url") for seg in segments):
        from ...services.render_worker import render_evidence_clips
        background_tasks.add_task(render_evidence_clips, evidence.id)

    # Build camera path
    camera_path = [seg["camera_id"] for seg in segments if seg["camera_id"]]
    camera_path = list(dict.fromkeys(camera_path))  # Remove duplicates while preserving order

    segment_responses = [
        TraceSegmentResponse(
            segment_order=seg["segment_order"],
            tracklet_id=seg["tracklet_id"],
            camera_id=seg["camera_id"],
            time_start=seg["time_start"],
            time_end=seg["time_end"],
            duration_seconds=seg["duration_seconds"],
            thumbnail_url=seg.get("thumbnail_url"),
            video_clip_url=seg.get("video_clip_url"),
            confidence=seg.get("confidence"),
        )
        for seg in segments
    ]

    return TraceTimelineResponse(
        segments=segment_responses,
        camera_path=camera_path,
        time_window_start=evidence.time_window_start,
        time_window_end=evidence.time_window_end,
    )


@router.post("/feedback", response_model=TraceFeedbackResponse)
def submit_feedback(
    request: TraceFeedbackRequest,
    session: SessionDep,
) -> TraceFeedbackResponse:
    """Submit feedback for a trace."""
    service = _build_trace_service(session)

    evidence = session.get(EvidenceVideo, request.evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    verified_object_id = request.verified_object_id

    if request.is_correct and not verified_object_id:
        # Create verified object — linked to candidate via evidence
        verified_obj = VerifiedObject(
            candidate_id=evidence.query_candidate_id,
            verified_by_user_id=None,  # set from auth if available
            is_correct=request.is_correct,
            notes=getattr(request, "feedback_text", None),
            verified_at=datetime.now(timezone.utc),
        )
        session.add(verified_obj)
        session.flush()
        verified_object_id = verified_obj.id

        # Link tracklets
        evidence_tracklets = session.query(EvidenceTracklet).filter(
            EvidenceTracklet.evidence_video_id == evidence.id
        ).all()

        for idx, et in enumerate(evidence_tracklets):
            if et.tracklet_id:
                link = VerifiedObjectTracklet(
                    verified_object_id=verified_object_id,
                    tracklet_id=et.tracklet_id,
                    position=idx,
                )
                session.add(link)

    session.commit()

    return TraceFeedbackResponse(
        success=True,
        evidence_id=request.evidence_id,
        verified_object_id=verified_object_id,
        message="Feedback submitted successfully",
    )


def _embedding_info(t: Tracklet) -> CandidateTrackletEmbeddingInfo:
    emb = t.embedding
    if emb is None:
        return CandidateTrackletEmbeddingInfo(has_embedding=False)

    try:
        siglip_dim = len(emb.siglip_embedding) if emb.siglip_embedding is not None else None
    except TypeError:
        siglip_dim = None
    try:
        reid_dim = len(emb.reid_embedding) if emb.reid_embedding is not None else None
    except TypeError:
        reid_dim = None

    return CandidateTrackletEmbeddingInfo(
        has_embedding=siglip_dim is not None,
        dim=siglip_dim,
        has_reid_embedding=reid_dim is not None,
        reid_dim=reid_dim,
        reid_model=emb.reid_model_version,
    )


def _tracklet_to_preview(t: Tracklet) -> CandidateTrackletPreview:
    """Flatten a Tracklet row + its embedding/actions into the wire schema.

    `time_start` / `time_end` are wall-clock anchored to the recording time
    from `recorded_at` or the camera filename plus the tracklet's offset within
    that video. This is what the popup uses to sort across tracklets that came
    from different videos.
    """
    video_id = None
    window = tracklet_time_window(t)
    if t.video is not None:
        video_id = t.video.video_id
    start_offset = float(t.start_time) if t.start_time is not None else None
    end_offset = float(t.end_time) if t.end_time is not None else None

    time_start = window[0] if window else None
    time_end = window[1] if window else None
    duration = (end_offset - start_offset) if (start_offset is not None and end_offset is not None) else None

    actions = [
        CandidateTrackletAction(
            action_label=a.action_label,
            kinetics_label=a.kinetics_label,
            confidence=float(a.confidence or 0.0),
        )
        for a in (t.actions or [])
    ]
    actions.sort(key=lambda a: a.confidence, reverse=True)

    return CandidateTrackletPreview(
        tracklet_id=t.tracklet_id,
        video_id=video_id,
        camera_id=t.camera_id,
        track_id=t.track_id,
        time_start=time_start,
        time_end=time_end,
        start_offset_seconds=start_offset,
        end_offset_seconds=end_offset,
        duration_seconds=duration,
        crop_url=t.crop_url or None,
        representative_bbox=[float(v) for v in (t.representative_bbox or [])] or None,
        quality_score=float(t.quality_score) if t.quality_score is not None else None,
        confidence=float(t.quality_score) if t.quality_score is not None else None,
        gender=t.gender,
        gender_conf=t.gender_conf,
        age_range=t.age_range,
        age_range_conf=t.age_range_conf,
        upper_color=t.upper_color,
        upper_type=t.upper_type,
        upper_desc=t.upper_desc,
        upper_conf=t.upper_conf,
        lower_color=t.lower_color,
        lower_type=t.lower_type,
        lower_desc=t.lower_desc,
        lower_conf=t.lower_conf,
        shoes_color=t.shoes_color,
        shoes_type=t.shoes_type,
        shoes_desc=t.shoes_desc,
        shoes_conf=t.shoes_conf,
        bag_presence=t.bag_presence,
        bag_type=t.bag_type,
        bag_desc=t.bag_desc,
        bag_conf=t.bag_conf,
        hat_presence=t.hat_presence,
        hat_color=t.hat_color,
        hat_type=t.hat_type,
        hat_desc=t.hat_desc,
        hat_conf=t.hat_conf,
        mask_presence=t.mask_presence,
        mask_conf=t.mask_conf,
        hair_style=t.hair_style,
        hair_style_conf=t.hair_style_conf,
        hair_color=t.hair_color,
        hair_color_conf=t.hair_color_conf,
        appearance_summary=t.appearance_summary or None,
        appearance_summary_conf=t.appearance_summary_conf,
        bev_x=float(t.bev_x) if t.bev_x is not None else None,
        bev_y=float(t.bev_y) if t.bev_y is not None else None,
        actions=actions,
        embedding=_embedding_info(t),
    )


@router.post("/candidate-detail", response_model=CandidateDetailResponse)
def get_candidate_detail(
    request: TraceCandidateDetailRequest,
    session: SessionDep,
) -> CandidateDetailResponse:
    """Full detail of a candidate + every tracklet that belongs to it.

    Returns the persisted QueryCandidateTracklet members ordered by real
    wall-clock time (video recording time from DB/filename + tracklet offset).
    Each tracklet carries its full appearance/demographic record, top VideoMAE
    actions, and embedding metadata.
    """
    service = _build_trace_service(session)

    query = _get_query(session, request.query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    candidate = _get_candidate(session, request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.query_id != str(request.query_id):
        raise HTTPException(status_code=400, detail="Candidate does not belong to this query")

    tracklets = service.get_candidate_tracklets(
        candidate_id=request.candidate_id,
        allow_fallback=bool(candidate.preview_url),
    )

    previews = [_tracklet_to_preview(t) for t in tracklets]
    # Order by wall-clock start (None last). Tie-break by tracklet_id for stability.
    previews.sort(key=lambda p: (
        p.time_start or datetime.max.replace(tzinfo=timezone.utc),
        p.tracklet_id,
    ))

    camera_path: list[str] = []
    for p in previews:
        if p.camera_id and p.camera_id not in camera_path:
            camera_path.append(p.camera_id)

    return CandidateDetailResponse(
        candidate_id=candidate.candidate_id,
        candidate_key=candidate.candidate_key,
        fusion_score=candidate.fusion_score,
        vector_score=candidate.vector_score,
        text_score=candidate.text_score,
        appearance_summary=candidate.appearance_summary,
        preview_url=candidate.preview_url,
        rank_position=candidate.rank_position,
        total_tracklets_in_window=len(previews),
        tracklets=previews,
        camera_path=camera_path,
    )


@router.post("/candidate-tracklet/remove")
def remove_candidate_tracklet(
    session: SessionDep,
    body: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Remove one persisted tracklet membership from a candidate."""
    query_id = str(body.get("query_id") or "").strip()
    candidate_id = str(body.get("candidate_id") or "").strip()
    tracklet_id = str(body.get("tracklet_id") or "").strip()
    if not query_id or not candidate_id or not tracklet_id:
        raise HTTPException(status_code=400, detail="query_id, candidate_id, and tracklet_id are required")

    query = _get_query(session, query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    candidate = _get_candidate(session, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.query_id != query_id:
        raise HTTPException(status_code=400, detail="Candidate does not belong to this query")

    membership_rows = (
        session.query(QueryCandidateTracklet)
        .filter(QueryCandidateTracklet.candidate_id == candidate_id)
        .order_by(QueryCandidateTracklet.id.asc())
        .with_for_update()
        .all()
    )
    link = next((row for row in membership_rows if row.tracklet_id == tracklet_id), None)
    if not link:
        raise HTTPException(status_code=404, detail="Tracklet is not part of this candidate")
    if len(membership_rows) <= 1:
        raise HTTPException(status_code=400, detail="Candidate must keep at least one tracklet")

    session.delete(link)
    session.flush()

    remaining_rows = [row for row in membership_rows if row.id != link.id]
    remaining_count = len(remaining_rows)

    first_tracklet_id = remaining_rows[0].tracklet_id
    candidate.preview_url = f"/candidates/{first_tracklet_id}/preview"
    first_tracklet = session.scalar(
        select(Tracklet).where(Tracklet.tracklet_id == first_tracklet_id)
    )
    if first_tracklet and first_tracklet.appearance_summary:
        candidate.appearance_summary = first_tracklet.appearance_summary

    query.updated_at = datetime.now(timezone.utc)
    session.commit()

    return {
        "success": True,
        "query_id": query_id,
        "candidate_id": candidate_id,
        "tracklet_id": tracklet_id,
        "remaining_tracklet_count": remaining_count,
        "message": "Tracklet removed from candidate",
    }


@router.post("/continue", response_model=ContinueTraceResponse)
def continue_trace(
    request: ContinueTraceRequest,
    session: SessionDep,
) -> ContinueTraceResponse:
    """Continue/retrace with a new time window."""
    service = _build_trace_service(session)

    query = _get_query(session, request.query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    candidate = _get_candidate(session, request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if not candidate.is_selected:
        raise HTTPException(status_code=400, detail="Candidate is not selected")
    if candidate.query_id != str(request.query_id):
        raise HTTPException(status_code=400, detail="Candidate does not belong to this query")

    # Get previous evidence
    previous_evidence = session.query(EvidenceVideo).filter(
        EvidenceVideo.query_id == str(request.query_id),
        EvidenceVideo.query_candidate_id == str(request.candidate_id),
    ).order_by(EvidenceVideo.created_at.desc()).first()

    if not previous_evidence:
        raise HTTPException(status_code=404, detail="No previous trace found. Please build trace first.")

    # Delete old evidence for this query; the replacement below becomes the
    # single saved trace result for the query.
    service.delete_query_evidence(request.query_id)

    # Build new trace with a window anchored at the first real tracklet time.
    base_tracklets = service.get_candidate_tracklets(
        candidate_id=request.candidate_id,
        allow_fallback=bool(candidate.preview_url),
    )
    if not base_tracklets:
        raise HTTPException(status_code=404, detail="No tracklets found for candidate")
    window_start, window_end = service.candidate_time_window(
        base_tracklets,
        hours=request.new_time_window_hours,
        fallback_start=query.created_at,
    )
    tracklets = service.filter_tracklets_by_time_window(
        base_tracklets,
        time_window_start=window_start,
        time_window_end=window_end,
    )

    segments = service.build_trace_segments(
        tracklets,
        query_id=request.query_id,
        candidate_id=request.candidate_id,
    )
    trace_confidence = service.calculate_trace_confidence(segments)

    new_evidence = service.create_evidence_video(
        query_id=request.query_id,
        candidate_id=request.candidate_id,
        segments=segments,
        trace_confidence=trace_confidence,
        time_window_start=window_start,
        time_window_end=window_end,
    )

    query.status = "completed"
    query.selected_candidate_id = str(request.candidate_id)
    query.updated_at = datetime.now(timezone.utc)

    session.commit()

    return ContinueTraceResponse(
        success=True,
        previous_evidence_id=previous_evidence.id,
        new_evidence_id=new_evidence.id,
        new_segment_count=len(segments),
        message=f"Trace continued with {request.new_time_window_hours}h window",
    )
