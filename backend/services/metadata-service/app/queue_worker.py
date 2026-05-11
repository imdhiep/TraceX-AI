"""
Queue Worker — Batch cross-camera processing via metadata-service.

Polls the queue for unprocessed videos from VinUni Storage,
groups them by timestamp (e.g. all 50 cameras at 2026-04-28_11-00),
and sends each timestamp-group as ONE batch to the metadata-service
batch endpoint for cross-camera MCBLT association.

Each batch:
  1. GROUP: Collect up to --batch-size videos with same timestamp
  2. BATCH: POST /api/v1/video/batch/process with all videos
  3. SAVE: Write unified cross-camera tracklets to database
  4. MARK: Set processed_at on all videos in batch
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Match cam_01_2026-04-28_11-00.mp4 → (camera_id, date, hour, minute)
_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


@dataclass
class BatchItem:
    """One video entry inside a batch (same timestamp across cameras)."""
    video_id: str
    camera_id: Optional[str]
    video_path: str
    source_filename: str


@dataclass
class TimestampBatch:
    """A group of videos from different cameras at the same timestamp."""
    timestamp_key: str          # e.g. "2026-04-28_11-00"
    recorded_at: datetime       # parsed datetime
    videos: list[BatchItem]


def _parse_timestamp_from_filename(filename: str) -> Optional[tuple[str, datetime]]:
    """Parse timestamp from filename like cam_01_2026-04-28_11-00.mp4."""
    m = _TIMESTAMP_PATTERN.fullmatch(filename)
    if not m:
        return None
    try:
        date_str = m.group("date")
        hour = int(m.group("hour"))
        minute = int(m.group("minute"))
        second = int(m.group("second") or "0")
        recorded_at = datetime(
            int(date_str[0:4]),
            int(date_str[5:7]),
            int(date_str[8:10]),
            hour, minute, second,
        )
        key = f"{date_str}_{hour:02d}-{minute:02d}"
        return key, recorded_at
    except (ValueError, IndexError):
        return None


def _wait_for_metadata_service(max_wait: int = 120) -> bool:
    url = os.getenv(
        "METADATA_SERVICE_URL",
        "http://metadata-service:8002"
    ).rstrip("/") + "/health"
    deadline = time.time() + max_wait

    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
            if response.is_success:
                logger.info("Metadata service is ready")
                return True
        except httpx.HTTPError as e:
            logger.info("Waiting for metadata service: %s", e)
        time.sleep(5)

    logger.warning("Metadata service not ready after %ds, proceeding anyway", max_wait)
    return True


def _get_metadata_service_url() -> str:
    return os.getenv(
        "METADATA_SERVICE_URL",
        "http://metadata-service:8002"
    ).rstrip("/")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _group_videos_by_timestamp(
    queue_videos: list,
) -> list[TimestampBatch]:
    """
    Group queue videos by timestamp (date + hour + minute).

    Videos from the same timestamp (different cameras) share the same batch.
    Returns list of TimestampBatch sorted by timestamp.
    """
    entries: list[tuple[Optional[str], datetime, BatchItem]] = []

    for video in queue_videos:
        source_fn = video.source_filename or ""
        parsed = _parse_timestamp_from_filename(source_fn)

        if parsed:
            key, recorded_at = parsed
        else:
            # Fallback: use camera_id + video_id as unique key
            key = f"{video.camera_id}_{video.video_id}"
            recorded_at = datetime.now(timezone.utc)

        camera_id = video.camera_id or f"cam_unknown"
        video_path = video.local_video_path or ""

        entries.append((
            key,
            recorded_at,
            BatchItem(
                video_id=video.video_id,
                camera_id=camera_id,
                video_path=video_path,
                source_filename=source_fn,
            ),
        ))

    # Group by timestamp key
    entries.sort(key=lambda x: (x[0] or "", x[1]))
    batches: list[TimestampBatch] = []

    for key, group in groupby(entries, key=lambda x: x[0]):
        group_list = list(group)
        group_list.sort(key=lambda x: x[1])
        first_key, first_recorded = group_list[0]
        batch = TimestampBatch(
            timestamp_key=str(first_key or "unknown"),
            recorded_at=first_recorded,
            videos=[item for _, _, item in group_list],
        )
        batches.append(batch)

    return batches


def _send_batch_to_metadata(batch: TimestampBatch, timeout: int = 600) -> dict:
    """POST one timestamp batch to the metadata-service batch endpoint."""
    import uuid

    payload = {
        "batch_id": f"vinuni_{batch.timestamp_key}_{uuid.uuid4().hex[:8]}",
        "videos": [
            {
                "video_id": v.video_id,
                "camera_id": v.camera_id,
                "video_path": v.video_path,
                "source_filename": v.source_filename,
                "sample_interval": 15,
            }
            for v in batch.videos
        ],
    }

    url = f"{_get_metadata_service_url()}/api/v1/video/batch/process"
    logger.info(
        "[%s] Sending batch: %d cameras, timestamp=%s",
        payload["batch_id"],
        len(batch.videos),
        batch.timestamp_key,
    )

    with httpx.Client(timeout=float(timeout)) as client:
        response = client.post(url, json=payload)

    response.raise_for_status()
    return response.json()


def _save_candidates_from_batch(
    session: Session,
    batch: TimestampBatch,
    result: dict,
) -> int:
    """Save unified cross-camera tracklets from batch response to database.

    Writes to:
      - QueryHistory  : one record per batch (query_id links all candidates)
      - Tracklet      : one row per unified person (all cameras combined)
      - TrackletEmbedding : SigLIP2 1152-dim vectors per tracklet
      - TrackletAction    : VideoMAE action classification per tracklet
    """
    from shared.models import QueryHistory, Tracklet, TrackletEmbedding, TrackletAction

    imported = 0
    tracklets = result.get("tracklets", [])
    batch_id = result.get("batch_id", "")

    if not tracklets:
        return 0

    # Create one QueryHistory row to anchor all candidates from this batch
    query_record = QueryHistory(
        user_id=1,  # system user
        query_text=f"batch:{batch_id}",
        status="completed",
        result_count=len(tracklets),
    )
    session.add(query_record)
    session.flush()  # get query_record.query_id

    for tracklet in tracklets:
        tracklet_id = str(tracklet.get("tracklet_id") or "").strip()
        if not tracklet_id:
            continue

        primary_cam = str(tracklet.get("camera_id") or "")
        primary_vid = str(tracklet.get("video_id") or "")
        contributing_cams = tracklet.get("contributing_cameras", [])
        contributing_vids = tracklet.get("contributing_video_ids", [])
        summary = str(tracklet.get("appearance_summary") or "")
        gender = str(tracklet.get("gender") or "unknown")
        upper_color = str(
            tracklet.get("upper_clothing_color")
            or "unknown"
        )
        lower_color = str(
            tracklet.get("lower_clothing_color")
            or "unknown"
        )
        shoes_color = str(tracklet.get("shoes_color") or "unknown")
        quality_score = float(tracklet.get("quality_score") or 0.0)
        occlusion_score = float(tracklet.get("occlusion_score") or 0.0)
        start_time = float(tracklet.get("start_time") or 0.0)
        end_time = float(tracklet.get("end_time") or 0.0)
        rep_bbox = tracklet.get("representative_bbox", [])

        # Upsert Tracklet
        existing = session.scalar(
            select(Tracklet).where(Tracklet.tracklet_id == tracklet_id)
        )
        if existing:
            existing.camera_id = primary_cam
            existing.quality_score = quality_score
            existing.gender = gender
            existing.upper_clothing_color = upper_color
            existing.lower_clothing_color = lower_color
            existing.shoes_color = shoes_color
            existing.appearance_summary = summary
            existing.contributing_cameras = contributing_cams
            existing.contributing_video_ids = contributing_vids
            existing.batch_id = batch_id
        else:
            row = Tracklet(
                tracklet_id=tracklet_id,
                video_id=primary_vid or "unknown",
                camera_id=primary_cam,
                track_id=str(tracklet.get("track_id") or 0),
                start_time=start_time,
                end_time=end_time,
                quality_score=quality_score,
                occlusion_score=occlusion_score,
                gender=gender,
                age_range="unknown",
                upper_clothing_color=upper_color,
                lower_clothing_color=lower_color,
                shoes_color=shoes_color,
                appearance_summary=summary,
                representative_bbox=rep_bbox,
                contributing_cameras=contributing_cams,
                contributing_video_ids=contributing_vids,
                batch_id=batch_id,
            )
            session.add(row)
            imported += 1

        # Upsert TrackletEmbedding (SigLIP2 1152-dim)
        siglip_vec = tracklet.get("siglip_embedding") or []
        if siglip_vec and len(siglip_vec) > 0:
            emb_existing = session.scalar(
                select(TrackletEmbedding).where(TrackletEmbedding.tracklet_id == tracklet_id)
            )
            if emb_existing:
                emb_existing.siglip_embedding = siglip_vec
                emb_existing.model_version = "siglip2"
            else:
                emb_row = TrackletEmbedding(
                    tracklet_id=tracklet_id,
                    siglip_embedding=siglip_vec,
                    model_version="siglip2",
                )
                session.add(emb_row)

        # Upsert TrackletAction (VideoMAE action)
        action_label = str(tracklet.get("action") or "unknown")
        if action_label and action_label != "unknown":
            act_existing = session.scalar(
                select(TrackletAction).where(TrackletAction.tracklet_id == tracklet_id)
            )
            if act_existing:
                act_existing.action_label = action_label
            else:
                act_row = TrackletAction(
                    tracklet_id=tracklet_id,
                    action_label=action_label,
                    kinetics_label=None,
                    confidence=1.0,
                )
                session.add(act_row)

    session.flush()
    return imported


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

def run_worker(
    session_factory,
    poll_interval: int = 30,
    batch_size: int = 50,
    request_timeout: int = 600,
):
    """
    Main worker loop for batch cross-camera processing.

    1. Poll queue for unprocessed videos
    2. Group by timestamp (all cameras at same time → one batch)
    3. Send each batch to /api/v1/video/batch/process
    4. Save unified tracklets to DB
    5. Mark all videos in batch as processed
    """
    from shared.models import QueueVideoAsset

    _wait_for_metadata_service()

    while True:
        session = session_factory()
        try:
            # Fetch unprocessed videos, ordered by source_filename (≈ timestamp order)
            statement = (
                select(QueueVideoAsset)
                .where(QueueVideoAsset.processed_at.is_(None))
                .order_by(QueueVideoAsset.source_filename.asc())
                .limit(batch_size * 3)   # fetch more to ensure full timestamp batches
            )

            queued_videos = list(session.scalars(statement).all())
            if not queued_videos:
                logger.debug("Queue empty, sleeping %ds", poll_interval)
                time.sleep(poll_interval)
                continue

            logger.info("Found %d unprocessed videos in queue", len(queued_videos))

            # Group by timestamp
            batches = _group_videos_by_timestamp(queued_videos)
            logger.info("Formed %d timestamp batches", len(batches))

            for batch in batches:
                if not batch.videos:
                    continue

                # Check all videos in batch exist
                missing = [v for v in batch.videos if not Path(v.video_path).exists()]
                if missing:
                    logger.warning(
                        "[%s] %d video(s) have missing paths — skipping batch",
                        batch.timestamp_key, len(missing),
                    )
                    continue

                try:
                    result = _send_batch_to_metadata(batch, timeout=request_timeout)

                    # Save cross-camera tracklets
                    n_saved = _save_candidates_from_batch(session, batch, result)

                    # Mark all videos in this batch as processed
                    video_ids = [v.video_id for v in batch.videos]
                    session.query(QueueVideoAsset).filter(
                        QueueVideoAsset.video_id.in_(video_ids)
                    ).update(
                        {QueueVideoAsset.processed_at: datetime.now(timezone.utc)},
                        synchronize_session=False,
                    )
                    session.commit()

                    n_tracklets = result.get("n_tracklets", 0)
                    n_detections = result.get("total_detections", 0)
                    elapsed = result.get("processing_time_s", 0)
                    logger.info(
                        "[%s] Batch done: %d cameras → %d tracklets "
                        "(%d detections) in %.1fs. Saved %d candidates.",
                        batch.timestamp_key,
                        len(batch.videos),
                        n_tracklets,
                        n_detections,
                        elapsed,
                        n_saved,
                    )

                except httpx.HTTPStatusError as e:
                    logger.error(
                        "[%s] HTTP error %d: %s",
                        batch.timestamp_key, e.response.status_code, e.response.text[:500],
                    )
                    session.rollback()
                except Exception as exc:
                    logger.exception("[%s] Batch processing failed: %s", batch.timestamp_key, exc)
                    session.rollback()

        except Exception as e:
            logger.exception("Worker error: %s", e)
            session.rollback()
        finally:
            session.close()

        time.sleep(poll_interval)
