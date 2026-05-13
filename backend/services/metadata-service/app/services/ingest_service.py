"""
ingest_service.py — Drive Temp→Storage move + LOCAL GPU ingest into v3.3 schema.

Flow:
  1. Move videos from Drive Temp/ folder to Storage/ folder
  2. Scan Storage/ for all .mp4 files
  3. For each video NOT yet in `videos` table:
     a. Auto-register camera in `cameras` if new
     b. Insert video metadata → `videos`
     c. Stream bytes from Drive → GPU service (multipart upload, no disk cache)
     d. GPU service writes temp file, processes, deletes it
     e. Save people → `tracklets` + `tracklets_embeddings` + `tracklets_actions`
  4. Return summary

Tables written (v3.3 schema):
  cameras, videos, tracklets, tracklets_embeddings, tracklets_actions
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drive folder IDs (from move.py / .env)
# ---------------------------------------------------------------------------
DRIVE_TEMP_FOLDER_ID = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"
DRIVE_STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Used as metadata path in DB records only — no actual files written here
_STORAGE_PATH_METADATA = Path("/workspace/storage/storage")

# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

DRIVE_TEMP_FOLDER_ID = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"
DRIVE_STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


def _ensure_shared_secret() -> None:
    import sys
    if "/workspace/secrets" not in sys.path:
        sys.path.insert(0, "/workspace/secrets")


def _build_drive_service():
    _ensure_shared_secret()
    import importlib
    mod = importlib.import_module("shared_secret_runtime")
    try:
        return mod.build_google_drive_sa_service()
    except FileNotFoundError:
        return mod.build_google_drive_service()


def _list_drive_mp4s(service, folder_id: str) -> list[dict]:
    results: list[dict] = []
    _walk_folder(service, folder_id, results)
    return results


def _walk_folder(service, folder_id: str, out: list[dict]):
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink, webContentLink)",
            pageSize=200,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == DRIVE_FOLDER_MIME:
                _walk_folder(service, f["id"], out)
            elif f["name"].lower().endswith(".mp4"):
                out.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _list_drive_folders(service, parent_id: str) -> list[dict]:
    resp = service.files().list(
        q=f"'{parent_id}' in parents and trashed=false and mimeType='{DRIVE_FOLDER_MIME}'",
        fields="files(id, name)",
        pageSize=200,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    return resp.get("files", [])


def _find_or_create_folder(service, parent_id: str, name: str) -> str:
    escaped = name.replace("'", "\\'")
    resp = service.files().list(
        q=f"'{parent_id}' in parents and trashed=false and mimeType='{DRIVE_FOLDER_MIME}' and name='{escaped}'",
        fields="files(id)",
        pageSize=5,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    items = resp.get("files", [])
    if items:
        return items[0]["id"]
    created = service.files().create(
        body={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _move_file(service, file_id: str, from_parent: str, to_parent: str) -> None:
    service.files().update(
        fileId=file_id,
        addParents=to_parent,
        removeParents=from_parent,
        fields="id,parents",
        supportsAllDrives=True,
    ).execute()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_camera_id(filename: str) -> str | None:
    m = CAMERA_VIDEO_PATTERN.match(filename)
    return m.group("camera_id").lower() if m else None


def _parse_recorded_at(filename: str) -> datetime | None:
    m = CAMERA_VIDEO_PATTERN.search(filename)
    if not m:
        return None
    try:
        return datetime(
            int(m.group("recorded_date")[:4]),
            int(m.group("recorded_date")[5:7]),
            int(m.group("recorded_date")[8:10]),
            int(m.group("hour")),
            int(m.group("minute")),
            int(m.group("second") or "0"),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# GPU processing (direct call — no HTTP self-loop)
# ---------------------------------------------------------------------------

def _process_video_stream(drive_file_id: str, filename: str, video_id: str, camera_id: str) -> dict:
    """
    Download video from Google Drive then process directly with GPU models.
    Calls _process_video_sync() directly to avoid HTTP self-call deadlock.
    """
    import sys, tempfile
    from pathlib import Path
    from io import BytesIO
    sys.path.insert(0, "/workspace/secrets")
    from shared_secret_runtime import build_google_drive_sa_service
    from googleapiclient.http import MediaIoBaseDownload

    logger.info("[gpu] Downloading %s from Drive (id=%s)", filename, drive_file_id)

    drive_service = build_google_drive_sa_service()
    request = drive_service.files().get_media(fileId=drive_file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request, chunksize=1024 * 1024 * 50)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    video_bytes = buffer.getvalue()
    logger.info("[gpu] Downloaded %s: %d bytes, processing with GPU", filename, len(video_bytes))

    # Write to temp file and call GPU pipeline directly (no HTTP)
    suffix = Path(filename).suffix.lower() or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(video_bytes)

        from ..api.routers.video_process import _process_video_sync
        result = _process_video_sync(
            str(tmp_path), video_id, camera_id,
            sample_interval=15, bev_max_dist=1.5,
        )

        tracklets = [t.model_dump() if hasattr(t, "model_dump") else t for t in (result.tracklets or [])]
        logger.info("[gpu] Got %d tracklets for %s (%.1fs)", len(tracklets), video_id, result.processing_time_s)
        return {
            "tracklets": tracklets,
            "person_count": len(tracklets),
            "total_detections": result.total_detections or 0,
            "processing_time_s": result.processing_time_s or 0,
        }
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# DB helpers (v3.3)
# ---------------------------------------------------------------------------

def _ensure_camera(session: Session, camera_id: str) -> None:
    from shared.models import Camera
    existing = session.scalar(select(Camera).where(Camera.camera_id == camera_id))
    if existing is None:
        session.add(Camera(
            camera_id=camera_id,
            name=f"Camera {camera_id.replace('cam_', 'Camera ').title()}",
            location=None,
            fps=30.0,
            resolution_width=1920,
            resolution_height=1080,
            is_active=True,
        ))
        logger.info("[db] Auto-registered camera: %s", camera_id)


def _video_exists_in_db(session: Session, video_id: str) -> bool:
    """Return True only if video exists AND has been successfully processed."""
    from shared.models import Video
    video = session.scalar(select(Video).where(Video.video_id == video_id))
    if video is None:
        return False
    return bool(video.processed)


def _upsert_video(
    session: Session,
    video_id: str,
    camera_id: str,
    title: str,
    source_filename: str,
    drive_file_id: str,
    recorded_at: datetime | None,
) -> None:
    from shared.models import Video
    existing = session.scalar(select(Video).where(Video.video_id == video_id))
    if existing is not None:
        return

    session.add(Video(
        video_id=video_id,
        camera_id=camera_id,
        title=title,
        storage_path=str(_STORAGE_PATH_METADATA / source_filename),
        storage_backend="google_drive",
        drive_file_id=drive_file_id,
        source_filename=source_filename,
        processed=False,
    ))
    session.flush()


def _save_tracklets_from_gpu_result(
    session: Session,
    gpu_result: dict,
    video_id: str,
    camera_id: str,
) -> int:
    """
    Parse GPU result and save to v3.3 tables.
    Returns number of tracklets saved.
    """
    from shared.models import Tracklet, TrackletEmbedding, TrackletAction, TrackletObservation

    tracklets = gpu_result.get("tracklets", [])
    saved = 0

    for t in tracklets:
        tracklet_id = str(t.get("tracklet_id") or "")
        if not tracklet_id:
            continue

        existing = session.scalar(select(Tracklet).where(Tracklet.tracklet_id == tracklet_id))
        if existing is not None:
            continue

        def _opt_float(val) -> float | None:
            return float(val) if val is not None else None

        def _s(val, max_len: int, fallback: str = "unknown") -> str:
            """str-coerce, fallback on empty, hard-truncate to column max_len."""
            v = str(val).strip() if val is not None else ""
            return (v or fallback)[:max_len]

        tracklet = Tracklet(
            tracklet_id=tracklet_id,
            video_id=video_id,
            camera_id=camera_id,
            track_id=str(t.get("track_id") or "0"),
            start_time=float(t.get("start_time") or 0.0),
            end_time=float(t.get("end_time") or 0.0),
            quality_score=float(t.get("quality_score") or 0.0),

            # Demographic
            gender=_s(t.get("gender"), 32),
            gender_conf=_opt_float(t.get("gender_conf")),
            age_range=_s(t.get("age_range"), 32),
            age_range_conf=_opt_float(t.get("age_range_conf")),

            # Upper
            upper_color=_s(t.get("upper_color"), 64, "unknown") if t.get("upper_color") else None,
            upper_type=_s(t.get("upper_type"), 128, "unknown") if t.get("upper_type") else None,
            upper_desc=t.get("upper_desc"),
            upper_conf=_opt_float(t.get("upper_conf")),
            upper_desc_conf=_opt_float(t.get("upper_desc_conf")),

            # Lower
            lower_color=_s(t.get("lower_color"), 64, "unknown") if t.get("lower_color") else None,
            lower_type=_s(t.get("lower_type"), 128, "unknown") if t.get("lower_type") else None,
            lower_desc=t.get("lower_desc"),
            lower_conf=_opt_float(t.get("lower_conf")),
            lower_desc_conf=_opt_float(t.get("lower_desc_conf")),

            # Shoes
            shoes_color=_s(t.get("shoes_color"), 64, "unknown") if t.get("shoes_color") else None,
            shoes_type=_s(t.get("shoes_type"), 128, "unknown") if t.get("shoes_type") else None,
            shoes_desc=t.get("shoes_desc"),
            shoes_conf=_opt_float(t.get("shoes_conf")),
            shoes_desc_conf=_opt_float(t.get("shoes_desc_conf")),

            # Bag
            bag_presence=_s(t.get("bag_presence"), 16, "unknown") if t.get("bag_presence") else None,
            bag_type=_s(t.get("bag_type"), 64, "unknown") if t.get("bag_type") else None,
            bag_desc=t.get("bag_desc"),
            bag_conf=_opt_float(t.get("bag_conf")),
            bag_desc_conf=_opt_float(t.get("bag_desc_conf")),

            # Hat
            hat_presence=_s(t.get("hat_presence"), 16, "unknown") if t.get("hat_presence") else None,
            hat_color=_s(t.get("hat_color"), 64, "unknown") if t.get("hat_color") else None,
            hat_type=_s(t.get("hat_type"), 128, "unknown") if t.get("hat_type") else None,
            hat_desc=t.get("hat_desc"),
            hat_conf=_opt_float(t.get("hat_conf")),
            hat_desc_conf=_opt_float(t.get("hat_desc_conf")),

            # Mask / hair
            mask_presence=_s(t.get("mask_presence"), 16),
            mask_conf=_opt_float(t.get("mask_conf")),
            hair_style=_s(t.get("hair_style"), 64),
            hair_style_conf=_opt_float(t.get("hair_style_conf")),
            hair_color=_s(t.get("hair_color"), 64),
            hair_color_conf=_opt_float(t.get("hair_color_conf")),

            # Summary
            appearance_summary=str(t.get("appearance_summary") or ""),
            appearance_summary_conf=_opt_float(t.get("appearance_summary_conf")),

            # Spatial / crop
            bev_x=float(t.get("bev_x") or 0.0),
            bev_y=float(t.get("bev_y") or 0.0),
            crop_url=str(t.get("crop_url") or ""),
            representative_bbox=t.get("representative_bbox") or [],
        )
        session.add(tracklet)

        # Embedding (SigLIP2 1152-dim only)
        siglip_vec = t.get("siglip_embedding") or []
        if siglip_vec:
            session.add(TrackletEmbedding(
                tracklet_id=tracklet_id,
                siglip_embedding=siglip_vec,
            ))

        # Action (VideoMAE V2)
        action_label = str(t.get("action") or t.get("action_label") or "")
        if action_label and action_label != "unknown":
            session.add(TrackletAction(
                tracklet_id=tracklet_id,
                action_label=action_label,
                kinetics_label=str(t.get("kinetics_label") or ""),
                confidence=float(t.get("action_confidence") or 0.0),
            ))

        # Per-frame observations (bbox timeline) for trace-service evidence rendering
        obs_list = t.get("observations") or []
        obs_by_frame: dict[int, dict] = {}
        for o in obs_list:
            try:
                bbox = o.get("bbox") or []
                if not bbox or len(bbox) < 4:
                    continue
                frame_index = int(o.get("frame_index") or 0)
                confidence = float(o.get("confidence") or 0.0)
                normalized = {
                    "timestamp_second": float(o.get("timestamp_second") or 0.0),
                    "bbox": [int(v) for v in bbox[:4]],
                    "confidence": confidence or None,
                }
                existing_obs = obs_by_frame.get(frame_index)
                existing_conf = float(existing_obs.get("confidence") or 0.0) if existing_obs else -1.0
                if existing_obs is None or confidence >= existing_conf:
                    obs_by_frame[frame_index] = normalized
            except Exception as exc:
                logger.warning("[ingest] skip bad observation for %s: %s", tracklet_id, exc)

        if len(obs_by_frame) < len(obs_list):
            logger.debug(
                "[ingest] dedup observations for %s: %d -> %d unique frames",
                tracklet_id, len(obs_list), len(obs_by_frame),
            )

        for frame_index, o in sorted(obs_by_frame.items()):
            session.add(TrackletObservation(
                tracklet_id=tracklet_id,
                frame_index=frame_index,
                timestamp_second=o["timestamp_second"],
                bbox=o["bbox"],
                confidence=o["confidence"],
            ))

        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------

def ingest_move_and_process(
    session: Session,
    dry_run: bool = False,
    max_videos: int | None = None,
) -> dict:
    """
    Full pipeline: move Temp→Storage, download, process with LOCAL GPU, save to v3.3 DB.
    """
    from shared.models import Video
    from sqlalchemy import func

    total_existing = int(session.scalar(select(func.count()).select_from(Video)) or 0)
    logger.info("[ingest] Starting. Existing videos in DB: %d", total_existing)

    if dry_run:
        logger.info("[ingest] DRY RUN")

    # --- Step 1: Move Temp → Storage (best-effort, skip if SA has read-only permissions) ---
    drive = _build_drive_service()
    temp_files = _list_drive_mp4s(drive, DRIVE_TEMP_FOLDER_ID)
    logger.info("[drive] Found %d .mp4 files in Temp/", len(temp_files))

    moved_count = 0
    sa_read_only = False
    for f in temp_files:
        filename = f["name"]
        if not _parse_camera_id(filename):
            logger.info("[drive] SKIP (bad name): %s", filename)
            continue

        m = CAMERA_VIDEO_PATTERN.match(filename)
        if not m:
            logger.info("[drive] SKIP (no match): %s", filename)
            continue

        recorded_date = m.group("recorded_date")

        if not dry_run:
            try:
                cam_folder_id = _find_or_create_folder(drive, DRIVE_STORAGE_FOLDER_ID, m.group("camera_id"))
                date_folder_id = _find_or_create_folder(drive, cam_folder_id, recorded_date)
                _move_file(drive, f["id"], DRIVE_TEMP_FOLDER_ID, date_folder_id)
                logger.info("[drive] Moved: %s → Storage/%s/%s/", filename, m.group("camera_id"), recorded_date)
            except Exception as move_err:
                logger.warning("[drive] Move failed (SA read-only): %s — will process from Temp directly.", move_err)
                sa_read_only = True
                break  # stop trying moves

        moved_count += 1

    logger.info("[drive] Moved %d files", moved_count)

    # --- Step 2: Scan Storage/ + Temp/ (if SA is read-only) ---
    storage_files = _list_drive_mp4s(drive, DRIVE_STORAGE_FOLDER_ID)
    logger.info("[drive] Found %d .mp4 files in Storage/", len(storage_files))

    # If SA can't move, also include Temp files for processing
    if sa_read_only and temp_files:
        valid_temp = [f for f in temp_files if _parse_camera_id(f["name"])]
        storage_files = storage_files + valid_temp
        logger.info("[drive] SA read-only: added %d files from Temp/ for processing (total: %d)", len(valid_temp), len(storage_files))

    pending: list[dict] = []
    for f in storage_files:
        video_id = f["name"]
        if not _video_exists_in_db(session, video_id):
            pending.append(f)
        else:
            logger.debug("[ingest] Skip already ingested: %s", video_id)

    # Sort cam_01 → cam_50 theo thứ tự đúng
    def _sort_key(f: dict) -> tuple:
        m = CAMERA_VIDEO_PATTERN.match(f["name"])
        if not m:
            return (999, f["name"])
        cam_num = int(m.group("camera_id").replace("cam_", ""))
        return (cam_num, f["name"])

    pending.sort(key=_sort_key)

    if max_videos:
        pending = pending[:max_videos]

    skipped_count = len(storage_files) - len(pending)
    logger.info("[ingest] Pending: %d videos (skipped %d already in DB)", len(pending), skipped_count)

    if not pending:
        return {
            "status": "completed",
            "moved_count": moved_count,
            "ingested_count": 0,
            "skipped_count": skipped_count,
            "total_videos_in_db": total_existing,
            "errors": [],
            "message": f"Moved {moved_count} videos, 0 pending (all already in DB)",
        }

    # --- Step 3: Process videos — download in parallel, prefetch frame decode, GPU sequential ---
    ingested_count = 0
    tracklet_count = 0
    errors: list[str] = []
    processed_videos = 0
    PARALLEL_VIDEOS = 4

    import queue as _queue
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _Path

    def _download_video(f: dict) -> tuple[dict, bytes | None, str | None]:
        try:
            from io import BytesIO
            drive_dl = _build_drive_service()
            from googleapiclient.http import MediaIoBaseDownload
            req = drive_dl.files().get_media(fileId=f["id"])
            buf = BytesIO()
            dl = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024 * 50)
            done = False
            while not done:
                _, done = dl.next_chunk()
            buf.seek(0)
            return f, buf.getvalue(), None
        except Exception as e:
            return f, None, str(e)

    def _presample_bytes(f: dict, video_bytes: bytes):
        """Write bytes to tempfile and decode frames — runs in background thread while GPU is busy."""
        from ..api.routers.tracking_pipeline import VideoFrameSampler
        filename = f["name"]
        suffix = _Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = _Path(tmp.name)
            tmp.write(video_bytes)
        try:
            sampled = VideoFrameSampler(sample_fps=4).sample(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        return sampled

    def _run_gpu(f: dict, presampled) -> dict:
        """GPU pipeline using pre-decoded frames (skips frame sampling stage)."""
        from ..api.routers.video_process import _process_video_sync
        filename = f["name"]
        camera_id = _parse_camera_id(filename) or "unknown"
        result = _process_video_sync("", filename, camera_id, 15, 1.5, presampled_frames=presampled)
        tracklets = [t.model_dump() if hasattr(t, "model_dump") else t for t in (result.tracklets or [])]
        return {"tracklets": tracklets, "person_count": len(tracklets),
                "total_detections": result.total_detections or 0,
                "processing_time_s": result.processing_time_s or 0}

    for batch_start in range(0, len(pending), PARALLEL_VIDEOS):
        batch = pending[batch_start: batch_start + PARALLEL_VIDEOS]
        batch_num = batch_start // PARALLEL_VIDEOS + 1
        total_batches = (len(pending) + PARALLEL_VIDEOS - 1) // PARALLEL_VIDEOS
        logger.warning("[ingest] Batch %d/%d: downloading %d videos in parallel",
                       batch_num, total_batches, len(batch))

        if dry_run:
            for f in batch:
                logger.info("[dry-run] Would ingest: %s", f["name"])
            continue

        for f in batch:
            try:
                cam_id = _parse_camera_id(f["name"]) or "unknown"
                _ensure_camera(session, cam_id)
                _upsert_video(session, video_id=f["name"], camera_id=cam_id,
                              title=f["name"], source_filename=f["name"],
                              drive_file_id=f["id"],
                              recorded_at=_parse_recorded_at(f["name"]))
            except Exception as exc:
                logger.error("[db] Failed to register video %s: %s", f.get("name"), exc)
        session.commit()

        # Stage A: download 4 videos in parallel
        # Stage B: as each download finishes, immediately start frame decoding in background
        # Stage C: GPU processes each video using pre-decoded frames (no wait for decode)
        download_queue: _queue.Queue = _queue.Queue()
        decode_exec = ThreadPoolExecutor(max_workers=2)

        def _download_and_start_decode(f: dict) -> None:
            f_item, vid_bytes, err = _download_video(f)
            if err or vid_bytes is None:
                download_queue.put((f_item, None, err))
                return
            logger.warning("[ingest] ✓ Downloaded %s (%d MB) → decoding frames...",
                           f_item["name"], len(vid_bytes) // 1024 // 1024)
            decode_fut = decode_exec.submit(_presample_bytes, f_item, vid_bytes)
            download_queue.put((f_item, decode_fut, None))

        with ThreadPoolExecutor(max_workers=PARALLEL_VIDEOS) as dl_pool:
            for f in batch:
                dl_pool.submit(_download_and_start_decode, f)

            for _ in range(len(batch)):
                f_item, decode_fut, err = download_queue.get()
                filename = f_item["name"]
                i = batch_start + next((j for j, b in enumerate(batch) if b["name"] == filename), 0) + 1
                if err:
                    errors.append(f"{filename}: download failed: {err}")
                    logger.error("[ingest] ✗ Download failed %s: %s", filename, err)
                    continue
                try:
                    presampled = decode_fut.result()
                    logger.warning("[ingest] ✓ %s frames ready → GPU", filename)
                    gpu_result = _run_gpu(f_item, presampled)
                    saved = _save_tracklets_from_gpu_result(session, gpu_result, filename,
                                                             _parse_camera_id(filename) or "unknown")
                    tracklet_count += saved
                    video = session.scalar(select(Video).where(Video.video_id == filename))
                    if video:
                        video.processed = True
                    session.commit()
                    processed_videos += 1
                    ingested_count += saved
                    logger.warning("[%d/%d] ✓ %s: %d tracklets in %.1fs",
                                   i, len(pending), filename, saved, gpu_result.get("processing_time_s", 0))
                except Exception as exc:
                    logger.error("[%d/%d] ✗ FAILED %s: %s", i, len(pending), filename, exc)
                    errors.append(f"{filename}: {exc}")
                    try:
                        session.rollback()
                    except Exception:
                        pass

        decode_exec.shutdown(wait=False)

    # (error handling done inline per-video above)

    final_total = int(session.scalar(select(func.count()).select_from(Video)) or 0)

    return {
        "status": "completed" if not errors else "completed_with_errors",
        "moved_count": moved_count,
        "ingested_count": ingested_count,
        "skipped_count": skipped_count,
        "total_videos_in_db": final_total,
        "total_tracklets_saved": tracklet_count,
        "processed_videos": processed_videos,
        "errors": errors[:10],
        "message": (
            f"Moved {moved_count} videos, "
            f"processed {processed_videos} videos with {ingested_count} tracklets, "
            f"skipped {skipped_count} already in DB"
            + (f", {len(errors)} errors" if errors else "")
        ),
    }
