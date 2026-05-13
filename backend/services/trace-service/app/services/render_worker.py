"""Background renderer for evidence trace clips.

The synchronous /trace/build was creating evidence_videos + evidence_tracklets
*and* rendering every per-tracklet clip in one request. For candidates with
many tracklets (we have rows with 40+) this easily runs past the upstream
proxy's response timeout, surfacing as "Internal Server Error" on the
frontend even though trace-service ultimately returned 200.

This module renders clips off the request thread. The HTTP handler now stores
evidence rows with empty `video_clip_url`s and enqueues a background job;
that job opens its own DB session, renders the missing clips in parallel,
and writes the resulting URLs back to `evidence_tracklets`. The frontend
polls /trace/status to track progress.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..core.models import EvidenceTracklet, EvidenceVideo, Tracklet
from ..database import SessionLocal
from .clip_render import render_tracklet_clip

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Render N clips in parallel. ffmpeg+OpenCV are both I/O-bound on disk reads
# of the source HEVC and CPU-bound on the libx264 encode; 4 workers fits an
# A100 host comfortably without thrashing the page cache.
_MAX_RENDER_WORKERS = int(os.getenv("TRACE_CLIP_PARALLEL_WORKERS", "4"))

# In-flight evidence IDs, so a slow first-render doesn't get re-enqueued by a
# second client polling /trace/status. Concurrent renders for the same clip
# would also race on the same .mp4.tmp path.
_running: set[int] = set()
_running_lock = threading.Lock()


def _resolve_source_path(session: "Session", tracklet: Tracklet) -> str | None:
    """Mirror of TraceService._resolve_source_video_path but standalone."""
    from pathlib import Path

    video = tracklet.video
    if video is None:
        return None
    if video.storage_path:
        sp = Path(str(video.storage_path))
        if sp.exists() and sp.stat().st_size > 0:
            return str(sp)
    drive_id = getattr(video, "drive_file_id", None)
    if drive_id:
        from .drive_fetch import fetch_video
        local = fetch_video(drive_id)
        if local is not None:
            return str(local)
    return None


def _render_one(
    *,
    evidence_tracklet_id: int,
    tracklet_id: str,
    query_id: str,
    candidate_id: str,
    start_time: float,
    end_time: float,
) -> tuple[int, str | None]:
    """Render a single tracklet's clip in its own DB session.

    Returns (evidence_tracklet_id, clip_url) — clip_url may be None on
    failure, in which case the row's URL is left as an empty string so
    the polling loop knows it's still pending after a retry.
    """
    session = SessionLocal()
    try:
        tracklet = session.query(Tracklet).filter_by(tracklet_id=tracklet_id).one_or_none()
        if tracklet is None:
            logger.warning("[render_worker] tracklet %s missing", tracklet_id)
            return evidence_tracklet_id, None

        source_path = _resolve_source_path(session, tracklet)
        if not source_path:
            logger.warning("[render_worker] no source path for %s", tracklet_id)
            return evidence_tracklet_id, None

        obs_payload = [
            {
                "frame_index": o.frame_index,
                "timestamp_second": o.timestamp_second,
                "bbox": list(o.bbox) if o.bbox else [],
                "confidence": o.confidence,
            }
            for o in (tracklet.observations or [])
        ]

        clip_url = render_tracklet_clip(
            source_video_path=source_path,
            observations=obs_payload,
            start_time=start_time,
            end_time=end_time,
            query_id=query_id,
            candidate_id=candidate_id,
            tracklet_id=tracklet.tracklet_id,
            draw_bbox=bool(obs_payload),
        )
        return evidence_tracklet_id, clip_url
    except Exception as exc:
        logger.warning("[render_worker] render failed for %s: %s", tracklet_id, exc)
        return evidence_tracklet_id, None
    finally:
        session.close()


def render_evidence_clips(evidence_id: int) -> None:
    """Render every pending clip for one evidence row, in parallel.

    Idempotent: if already in progress, returns immediately. Clip rendering
    itself is also idempotent (skip-if-exists in `render_tracklet_clip`), so
    re-enqueuing after a partial failure simply finishes the remaining rows.
    """
    with _running_lock:
        if evidence_id in _running:
            logger.info("[render_worker] evidence %s already rendering, skip", evidence_id)
            return
        _running.add(evidence_id)

    try:
        _render_evidence_clips_impl(evidence_id)
    finally:
        with _running_lock:
            _running.discard(evidence_id)


def _render_evidence_clips_impl(evidence_id: int) -> None:
    session = SessionLocal()
    try:
        evidence = session.get(EvidenceVideo, evidence_id)
        if evidence is None:
            logger.warning("[render_worker] evidence %s not found", evidence_id)
            return

        query_id = str(evidence.query_id)
        candidate_id = str(evidence.query_candidate_id)

        pending = (
            session.query(EvidenceTracklet)
            .filter(
                EvidenceTracklet.evidence_video_id == evidence_id,
                EvidenceTracklet.tracklet_id.isnot(None),
                # NOT NULL column — pending rows carry empty string.
                EvidenceTracklet.video_clip_url == "",
            )
            .all()
        )
        if not pending:
            logger.info("[render_worker] evidence %s: nothing to render", evidence_id)
            return

        # Snapshot the fields we need before fanning out — workers use their
        # own sessions so we don't share ORM objects across threads.
        jobs = []
        for et in pending:
            tracklet = et.tracklet
            if tracklet is None:
                continue
            jobs.append({
                "evidence_tracklet_id": et.id,
                "tracklet_id": tracklet.tracklet_id,
                "start_time": float(tracklet.start_time or 0.0),
                "end_time": float(tracklet.end_time or 0.0),
            })

        logger.info(
            "[render_worker] evidence %s: rendering %d clips with %d workers",
            evidence_id, len(jobs), _MAX_RENDER_WORKERS,
        )

        with ThreadPoolExecutor(max_workers=_MAX_RENDER_WORKERS) as pool:
            futures = [
                pool.submit(
                    _render_one,
                    evidence_tracklet_id=job["evidence_tracklet_id"],
                    tracklet_id=job["tracklet_id"],
                    query_id=query_id,
                    candidate_id=candidate_id,
                    start_time=job["start_time"],
                    end_time=job["end_time"],
                )
                for job in jobs
            ]
            for fut in futures:
                evidence_tracklet_id, clip_url = fut.result()
                if not clip_url:
                    continue
                # Commit each clip URL as it lands so the polling client sees
                # incremental progress instead of one big batch at the end.
                row = session.get(EvidenceTracklet, evidence_tracklet_id)
                if row is not None:
                    row.video_clip_url = clip_url
                    session.commit()
        logger.info("[render_worker] evidence %s: done", evidence_id)
    except Exception:
        logger.exception("[render_worker] evidence %s failed", evidence_id)
        session.rollback()
    finally:
        session.close()
