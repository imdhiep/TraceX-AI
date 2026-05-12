"""Trace service - business logic for building traces."""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

_CAM_NEIGHBOR_RADIUS = 10


def _extract_cam_num(cam_id: str) -> int | None:
    m = re.search(r'\d+', cam_id or "")
    return int(m.group()) if m else None


def _neighbor_cameras(primary_cam: str, radius: int = _CAM_NEIGHBOR_RADIUS) -> list[str]:
    """Return cam IDs within ±radius of primary_cam by numeric suffix.

    cam_20, radius=10 → [cam_10, cam_11, ..., cam_30]
    """
    num = _extract_cam_num(primary_cam)
    if num is None:
        return [primary_cam]
    digits = re.search(r'\d+', primary_cam).group()
    prefix = primary_cam[: primary_cam.index(digits)]
    width = len(digits)
    return [
        f"{prefix}{i:0{width}d}"
        for i in range(max(1, num - radius), num + radius + 1)
    ]

logger = logging.getLogger(__name__)

from ..config import settings
from ..core.models import (
    Camera,
    CameraEdge,
    EvidenceTracklet,
    EvidenceVideo,
    QueryCandidate,
    QueryCandidateTracklet,
    Tracklet,
)


class TraceService:
    """Service for building traces from selected candidates."""

    def __init__(self, session: Session):
        self.session = session

    def deselect_other_candidates(self, query_id: UUID) -> None:
        """Deselect all other candidates for a query."""
        self.session.query(QueryCandidate).filter(
            QueryCandidate.query_id == str(query_id),
            QueryCandidate.is_selected == True,  # noqa: E712
        ).update({"is_selected": False, "selected_at": None})

    def get_candidate_tracklets(
        self,
        candidate_id: UUID | str,
        time_window_start: datetime,
        time_window_end: datetime,
    ) -> list[Tracklet]:
        """Get all tracklets for a candidate within time window.

        Strategy:
        1. Try query_candidate_tracklets join table (populated by advanced re-ID flows).
        2. Fallback: use primary_camera_id of the candidate → expand to ±_CAM_NEIGHBOR_RADIUS
           neighbouring cameras → return all tracklets in that range.
           This covers the case where person found in cam_20 can realistically
           appear in cam_10…cam_30.
        """
        candidate = self.session.scalar(
            select(QueryCandidate).where(QueryCandidate.candidate_id == str(candidate_id))
        )
        if not candidate:
            return []

        # ── 1. Try join table ──────────────────────────────────────────────
        via_join = (
            self.session.query(Tracklet)
            .join(
                QueryCandidateTracklet,
                QueryCandidateTracklet.tracklet_id == Tracklet.tracklet_id,
            )
            .filter(QueryCandidateTracklet.candidate_id == str(candidate_id))
            .order_by(Tracklet.start_time.asc())
            .all()
        )

        if via_join:
            tracklets = via_join
        else:
            # ── 2. Fallback: neighbor-camera expansion ─────────────────────
            primary_cam = candidate.primary_camera_id or ""
            neighbor_cams = _neighbor_cameras(primary_cam)
            logger.info(
                "Trace expand: candidate %s primary_cam=%s → %d neighbour cams %s…%s",
                candidate_id, primary_cam, len(neighbor_cams),
                neighbor_cams[0], neighbor_cams[-1],
            )
            tracklets = (
                self.session.query(Tracklet)
                .filter(Tracklet.camera_id.in_(neighbor_cams))
                .order_by(Tracklet.camera_id.asc(), Tracklet.start_time.asc())
                .all()
            )

            # Soft metadata filter — loại người rõ ràng khác identity.
            # candidate (QueryCandidate) still carries legacy top_color/bottom_color
            # columns; tracklet rows now use upper_color/lower_color.
            _gender = candidate.gender
            _top    = candidate.top_color
            _bottom = candidate.bottom_color
            _UNKNOWN = {"unknown", "", None}

            def _meta_ok(t: Tracklet) -> bool:
                if _gender not in _UNKNOWN and t.gender not in _UNKNOWN:
                    if t.gender != _gender:
                        return False
                if _top not in _UNKNOWN and t.upper_color not in _UNKNOWN:
                    if t.upper_color != _top:
                        return False
                if _bottom not in _UNKNOWN and t.lower_color not in _UNKNOWN:
                    if t.lower_color != _bottom:
                        return False
                return True

            before = len(tracklets)
            tracklets = [t for t in tracklets if _meta_ok(t)]
            logger.warning(
                "Trace fallback (no QCT rows): %d → %d tracklets after metadata filter "
                "(gender=%s top=%s bottom=%s)",
                before, len(tracklets), _gender, _top, _bottom,
            )

        # ── Time-window filter ─────────────────────────────────────────────
        # Use recorded_at (actual recording time from filename) when available,
        # fall back to created_at (ingest time).
        filtered: list[Tracklet] = []
        for t in tracklets:
            video = t.video
            if video:
                base_ts = (video.recorded_at or video.created_at).timestamp()
                t_start = base_ts + (t.start_time or 0)
                t_end = base_ts + (t.end_time or 0)
                if t_start <= time_window_end.timestamp() and t_end >= time_window_start.timestamp():
                    filtered.append(t)
            else:
                filtered.append(t)  # no video metadata: include anyway

        return filtered

    def build_trace_segments(
        self,
        tracklets: list[Tracklet],
        query_id: UUID | str | None = None,
        candidate_id: UUID | str | None = None,
    ) -> list[dict[str, Any]]:
        """Build trace segments from tracklets.

        Each tracklet becomes a segment. Segments are ordered by time.
        When `query_id` + `candidate_id` are provided AND the tracklet's source
        video file is on local disk, an evidence clip is rendered with a moving
        bbox (from `tracklet_observations`) and the static URL is attached.
        """
        if not tracklets:
            return []

        # Lazy import to avoid hard dep on cv2 when only metadata flows are used.
        from .clip_render import render_tracklet_clip

        segments = []
        for idx, tracklet in enumerate(tracklets):
            video = tracklet.video if tracklet.video_id else None

            duration = None
            if tracklet.start_time is not None and tracklet.end_time is not None:
                duration = tracklet.end_time - tracklet.start_time

            time_start = None
            time_end = None
            if video:
                base_dt = video.recorded_at or video.created_at
                time_start = base_dt + timedelta(seconds=tracklet.start_time or 0)
                time_end   = base_dt + timedelta(seconds=tracklet.end_time   or 0)

            # Try to render an evidence clip with moving bbox. Falls back to the
            # legacy synthetic URL when (a) we lack query/candidate context, or
            # (b) the source video isn't on local disk, or (c) render fails.
            clip_url = None
            if query_id and candidate_id and video and video.storage_path:
                obs_payload = [
                    {
                        "frame_index": o.frame_index,
                        "timestamp_second": o.timestamp_second,
                        "bbox": list(o.bbox) if o.bbox else [],
                        "confidence": o.confidence,
                    }
                    for o in (tracklet.observations or [])
                ]
                try:
                    clip_url = render_tracklet_clip(
                        source_video_path=str(video.storage_path),
                        observations=obs_payload,
                        start_time=float(tracklet.start_time or 0.0),
                        end_time=float(tracklet.end_time or 0.0),
                        query_id=str(query_id),
                        candidate_id=str(candidate_id),
                        tracklet_id=tracklet.tracklet_id,
                        draw_bbox=bool(obs_payload),
                    )
                except Exception as exc:
                    logger.warning(
                        "[trace] clip render failed for %s: %s",
                        tracklet.tracklet_id, exc,
                    )

            if clip_url is None:
                clip_url = self._get_video_clip_url(tracklet)

            segment = {
                "segment_order": idx + 1,
                "tracklet_id": tracklet.tracklet_id,
                "camera_id": tracklet.camera_id,
                "time_start": time_start,
                "time_end": time_end,
                "duration_seconds": duration,
                "thumbnail_url": tracklet.crop_url,
                "video_clip_url": clip_url,
                "confidence": tracklet.quality_score,
            }
            segments.append(segment)

        # Sort by time
        segments.sort(key=lambda s: (s["time_start"] or datetime.min, s["segment_order"]))

        # Re-index segment order after sorting
        for idx, seg in enumerate(segments):
            seg["segment_order"] = idx + 1

        return segments

    def calculate_trace_confidence(
        self,
        segments: list[dict[str, Any]],
    ) -> float | None:
        """Calculate overall trace confidence.

        Based on:
        - Individual segment confidences
        - Camera path continuity
        - Number of segments

        Args:
            segments: List of trace segments

        Returns:
            Confidence score between 0 and 1
        """
        if not segments:
            return None

        confidences = [s.get("confidence") for s in segments if s.get("confidence") is not None]

        if not confidences:
            return 0.5  # Default confidence

        # Base confidence is average of segment confidences
        avg_confidence = sum(confidences) / len(confidences)

        # Penalize for gaps in camera path
        camera_ids = [s["camera_id"] for s in segments if s["camera_id"]]
        unique_cameras = len(set(camera_ids))

        # Bonus for multiple cameras (more complete trace)
        camera_bonus = min(0.1, unique_cameras * 0.02)

        # Bonus for more segments (more complete trace)
        segment_bonus = min(0.1, len(segments) * 0.01)

        final_confidence = min(1.0, avg_confidence + camera_bonus + segment_bonus)
        return round(final_confidence, 3)

    def create_evidence_video(
        self,
        query_id: UUID,
        candidate_id: UUID | str,
        segments: list[dict[str, Any]],
        trace_confidence: float | None,
        time_window_start: datetime,
        time_window_end: datetime,
    ) -> EvidenceVideo:
        """Create evidence video record.

        Args:
            query_id: Query ID
            candidate_id: Selected candidate ID
            segments: List of trace segments
            trace_confidence: Overall trace confidence
            time_window_start: Start of trace window
            time_window_end: End of trace window

        Returns:
            Created EvidenceVideo record
        """
        total_duration = sum(s["duration_seconds"] or 0 for s in segments if s["duration_seconds"])

        evidence = EvidenceVideo(
            query_id=query_id,
            query_candidate_id=candidate_id,
            # Per-tracklet clips are generated now and stored on EvidenceTracklet.
            # A real merged video can be filled here later by a stitcher.
            video_url="",
            total_duration=total_duration,
            segment_count=len(segments),
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            trace_confidence=trace_confidence or 0.0,
        )
        self.session.add(evidence)
        self.session.flush()

        # Create evidence tracklets
        for seg in segments:
            evidence_tracklet = EvidenceTracklet(
                evidence_video_id=evidence.id,
                tracklet_id=seg["tracklet_id"],
                segment_order=seg["segment_order"],
                camera_id=seg["camera_id"],
                time_range={
                    "start": seg["time_start"].isoformat() if seg["time_start"] else None,
                    "end": seg["time_end"].isoformat() if seg["time_end"] else None,
                },
                video_clip_url=seg.get("video_clip_url") or "",
                thumbnail_url=seg.get("thumbnail_url"),
                confidence=seg.get("confidence"),
            )
            self.session.add(evidence_tracklet)

        logger.debug(
            "create_evidence_video: evidence_id=%s query_id=%s candidate_id=%s segments=%d",
            evidence.id, query_id, candidate_id, len(segments),
        )
        return evidence

    def get_trace_segments(self, evidence_id: UUID) -> list[dict[str, Any]]:
        """Get trace segments for an evidence video.

        Args:
            evidence_id: Evidence video ID

        Returns:
            List of segment dictionaries
        """
        evidence_tracklets = (
            self.session.query(EvidenceTracklet)
            .filter(EvidenceTracklet.evidence_video_id == evidence_id)
            .order_by(EvidenceTracklet.segment_order.asc())
            .all()
        )

        segments = []
        for et in evidence_tracklets:
            tracklet = et.tracklet if et.tracklet_id else None

            time_start = None
            time_end = None
            if et.time_range:
                if et.time_range.get("start"):
                    time_start = datetime.fromisoformat(et.time_range["start"])
                if et.time_range.get("end"):
                    time_end = datetime.fromisoformat(et.time_range["end"])

            duration = None
            if time_start and time_end:
                duration = (time_end - time_start).total_seconds()

            segments.append({
                "segment_order": et.segment_order,
                "tracklet_id": et.tracklet_id,
                "camera_id": et.camera_id,
                "time_start": time_start,
                "time_end": time_end,
                "duration_seconds": duration,
                "thumbnail_url": et.thumbnail_url,
                "video_clip_url": et.video_clip_url or (self._get_video_clip_url(tracklet) if tracklet else None),
                "confidence": et.confidence,
            })

        return segments

    def delete_old_evidence(self, candidate_id: UUID | str) -> int:
        """Delete old evidence videos for a candidate.

        This implements cache overwrite behavior - when a new candidate is selected,
        old evidence is cleared.

        Args:
            candidate_id: Candidate ID

        Returns:
            Number of deleted records
        """
        count = (
            self.session.query(EvidenceVideo)
            .filter(EvidenceVideo.query_candidate_id == str(candidate_id))
            .delete(synchronize_session=False)
        )
        return count

    def _get_video_clip_url(self, tracklet: Tracklet | None) -> str | None:
        """Get video clip URL for a tracklet.

        Args:
            tracklet: Tracklet instance

        Returns:
            Video clip URL or None
        """
        if not tracklet or not tracklet.video:
            return None

        video = tracklet.video
        if not video.storage_path:
            return None

        # Construct clip URL based on tracklet timing
        base_url = settings.storage_base_url.rstrip("/")
        clip_url = f"{base_url}/videos/{video.id}/clips/{tracklet.id}.mp4"
        return clip_url

    def _generate_merged_video_url(self, query_id: UUID | str, candidate_id: UUID | str) -> str:
        """Generate URL for merged trace video.

        Args:
            query_id: Query ID
            candidate_id: Candidate ID

        Returns:
            Merged video URL
        """
        base_url = settings.storage_base_url.rstrip("/")
        return f"{base_url}/traces/{query_id}/{candidate_id}/merged.mp4"

    def get_camera_path_from_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[str]:
        """Extract camera path from trace segments.

        Args:
            segments: List of trace segments

        Returns:
            Ordered list of camera IDs in the path
        """
        camera_path = []
        for seg in segments:
            if seg.get("camera_id") and seg["camera_id"] not in camera_path:
                camera_path.append(seg["camera_id"])
        return camera_path
