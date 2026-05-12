"""Trace API router - select candidate, build trace, feedback."""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
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

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _build_trace_service(session: Session) -> TraceService:
    """Build trace service instance."""
    return TraceService(session)


def _get_query(session: Session, query_id: UUID | str) -> QueryHistory | None:
    return session.scalar(select(QueryHistory).where(QueryHistory.query_id == str(query_id)))


def _get_candidate(session: Session, candidate_id: str) -> QueryCandidate | None:
    return session.scalar(select(QueryCandidate).where(QueryCandidate.candidate_id == str(candidate_id)))


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
    request: BuildTraceRequest,
    session: SessionDep,
) -> BuildTraceResponse:
    """Build a trace from the selected candidate.

    This finds all tracklets within the time window and creates evidence video.
    """
    service = _build_trace_service(session)

    # Verify query exists
    query = _get_query(session, request.query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    # Verify candidate exists and is selected
    candidate = _get_candidate(session, request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if not candidate.is_selected:
        raise HTTPException(status_code=400, detail="Candidate is not selected. Please select it first.")
    if candidate.query_id != str(request.query_id):
        raise HTTPException(status_code=400, detail="Candidate does not belong to this query")

    # Delete old evidence for this candidate (cache overwrite behavior)
    service.delete_old_evidence(request.candidate_id)

    # Get tracklets for this candidate within time window
    window_start = query.created_at - timedelta(hours=request.time_window_hours)
    window_end = query.created_at + timedelta(hours=request.time_window_hours)

    tracklets = service.get_candidate_tracklets(
        candidate_id=request.candidate_id,
        time_window_start=window_start,
        time_window_end=window_end,
    )

    if not tracklets:
        raise HTTPException(status_code=404, detail="No tracklets found for candidate in time window")

    # Build trace segments
    segments = service.build_trace_segments(
        tracklets,
        query_id=request.query_id,
        candidate_id=request.candidate_id,
    )

    # Calculate trace confidence
    trace_confidence = service.calculate_trace_confidence(segments)

    # Create evidence video record
    evidence = service.create_evidence_video(
        query_id=request.query_id,
        candidate_id=request.candidate_id,
        segments=segments,
        trace_confidence=trace_confidence,
        time_window_start=window_start,
        time_window_end=window_end,
    )

    session.commit()

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
        candidate_id=request.candidate_id,
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
) -> TraceStatusResponse:
    """Get trace status by evidence ID."""
    evidence = session.get(EvidenceVideo, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    return TraceStatusResponse(
        evidence_id=evidence.id,
        query_id=evidence.query_id,
        status="completed",
        trace_confidence=evidence.trace_confidence,
        segment_count=evidence.segment_count,
        created_at=evidence.created_at,
        updated_at=evidence.created_at,
    )


@router.get("/timeline/{evidence_id}", response_model=TraceTimelineResponse)
def get_trace_timeline(
    evidence_id: int,
    session: SessionDep,
) -> TraceTimelineResponse:
    """Get trace timeline with camera path."""
    service = _build_trace_service(session)

    evidence = session.get(EvidenceVideo, evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    segments = service.get_trace_segments(evidence_id)

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


@router.get("/candidate-detail", response_model=CandidateDetailResponse)
def get_candidate_detail(
    request: TraceCandidateDetailRequest,
    session: SessionDep,
) -> CandidateDetailResponse:
    """Get detailed information about a candidate for preview."""
    service = _build_trace_service(session)

    query = _get_query(session, request.query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    candidate = _get_candidate(session, request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.query_id != str(request.query_id):
        raise HTTPException(status_code=400, detail="Candidate does not belong to this query")

    # Get tracklets within 24h window
    window_start = query.created_at - timedelta(hours=24)
    window_end = query.created_at + timedelta(hours=24)

    tracklets = service.get_candidate_tracklets(
        candidate_id=request.candidate_id,
        time_window_start=window_start,
        time_window_end=window_end,
    )

    # Build tracklet previews
    tracklet_previews = []
    camera_ids = []

    for t in tracklets:
        video = t.video if t.video_id else None
        tracklet_previews.append(
            CandidateTrackletPreview(
                tracklet_id=t.tracklet_id,
                camera_id=t.camera_id,
                time_start=t.video.created_at if t.video else None,
                time_end=None,
                duration_seconds=t.end_time - t.start_time if t.start_time and t.end_time else None,
                appearance_summary=t.appearance_summary,
                crop_url=t.crop_url,
                confidence=t.quality_score,
            )
        )
        if t.camera_id and t.camera_id not in camera_ids:
            camera_ids.append(t.camera_id)

    return CandidateDetailResponse(
        candidate_id=candidate.candidate_id,
        candidate_key=candidate.candidate_key,
        fusion_score=candidate.fusion_score,
        vector_score=candidate.vector_score,
        text_score=candidate.text_score,
        appearance_summary=candidate.appearance_summary,
        preview_url=candidate.preview_url,
        rank_position=candidate.rank_position,
        total_tracklets_in_window=len(tracklets),
        tracklets=tracklet_previews,
        camera_path=camera_ids,
    )


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
        EvidenceVideo.query_candidate_id == str(request.candidate_id)
    ).order_by(EvidenceVideo.created_at.desc()).first()

    if not previous_evidence:
        raise HTTPException(status_code=404, detail="No previous trace found. Please build trace first.")

    # Delete old evidence
    service.delete_old_evidence(request.candidate_id)

    # Build new trace with new window
    window_start = query.created_at - timedelta(hours=request.new_time_window_hours)
    window_end = query.created_at + timedelta(hours=request.new_time_window_hours)

    tracklets = service.get_candidate_tracklets(
        candidate_id=request.candidate_id,
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

    session.commit()

    return ContinueTraceResponse(
        success=True,
        previous_evidence_id=previous_evidence.id,
        new_evidence_id=new_evidence.id,
        new_segment_count=len(segments),
        message=f"Trace continued with {request.new_time_window_hours}h window",
    )
