from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

import cv2
import httpx
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import A20_ROOT, PROJECT_ROOT, settings

if TYPE_CHECKING:
    from shared.models import User

logger = logging.getLogger(__name__)

STORAGE_INGEST_SOURCE_MODE = "storage_ingest"
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_STORAGE_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = str(os.getenv(name, "") or "").strip()
    if not raw_value:
        return default
    try:
        return max(int(raw_value), minimum)
    except ValueError:
        return default


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "video"


def _trim_tracking_segments(segments: object, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(segments, list):
        return []
    trimmed: list[dict[str, Any]] = []
    for segment in segments[:limit]:
        if not isinstance(segment, dict):
            continue
        trimmed.append(
            {
                "start_second": segment.get("start_second"),
                "end_second": segment.get("end_second"),
                "action_summary": str(segment.get("action_summary") or "").strip(),
            }
        )
    return trimmed


def _trim_tracklet_frames(frames: object, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(frames, list):
        return []
    trimmed: list[dict[str, Any]] = []
    for frame in frames[:limit]:
        if not isinstance(frame, dict):
            continue
        trimmed.append(
            {
                "frame_idx": frame.get("frame_idx"),
                "timestamp_second": frame.get("timestamp_second"),
                "bbox": frame.get("bbox"),
                "confidence": frame.get("confidence"),
            }
        )
    return trimmed


# ============================================================================
# Storage Video Items
# ============================================================================

@dataclass(frozen=True)
class StorageVideoIdentity:
    camera_id: str
    recorded_date: date
    recorded_at: datetime
    source_filename: str

    @classmethod
    def parse(cls, source_path: Path) -> "StorageVideoIdentity | None":
        match = _STORAGE_VIDEO_PATTERN.fullmatch(source_path.name)
        if not match:
            return None
        try:
            recorded_date = date.fromisoformat(match.group("recorded_date"))
            recorded_at = datetime(
                recorded_date.year,
                recorded_date.month,
                recorded_date.day,
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second") or "0"),
            )
        except ValueError:
            return None
        return cls(
            camera_id=match.group("camera_id").lower(),
            recorded_date=recorded_date,
            recorded_at=recorded_at,
            source_filename=source_path.name,
        )


@dataclass(frozen=True)
class StorageVideoItem:
    source_path: Path | None
    source_drive_file_id: str | None
    relative_path: str
    source_filename: str
    camera_id: str
    recorded_at: datetime
    size_bytes: int
    modified_ns: int
    fingerprint: str

    @property
    def output_basename(self) -> str:
        return self.source_filename

    def marker_payload(self) -> dict:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path) if self.source_path is not None else None
        payload["recorded_at"] = self.recorded_at.isoformat()
        return payload


class StorageIngestRegistry:
    def __init__(self, marker_dir: Path) -> None:
        self.marker_dir = marker_dir

    def is_processed(self, item: StorageVideoItem) -> bool:
        marker_path = self._marker_path(item)
        if not marker_path.exists():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            str(marker.get("relative_path") or "") == item.relative_path
            and int(marker.get("size_bytes") or -1) == item.size_bytes
            and int(marker.get("modified_ns") or -1) == item.modified_ns
        )

    def mark_processed(self, item: StorageVideoItem, result: dict) -> Path:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = self._marker_path(item)
        payload = item.marker_payload()
        payload["processed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        payload["result_video_id"] = str((result.get("video") or {}).get("video_id") or "")
        payload["person_count"] = int(result.get("person_count") or 0)
        marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return marker_path

    def _marker_path(self, item: StorageVideoItem) -> Path:
        return self.marker_dir / f"{item.fingerprint}.json"


# ============================================================================
# Queue Sync Service
# ============================================================================

class QueueSyncService:
    def __init__(self) -> None:
        self.local_root = Path(settings.queue_local_root)
        self.local_queue_dir = self.local_root / "local" / settings.google_drive_queue_folder_name
        self.local_queue_video_dir = self.local_queue_dir / settings.queue_video_folder_name
        self.local_queue_metadata_dir = self.local_queue_dir / settings.google_drive_metadata_folder_name
        self.storage_ingest_root = Path(settings.storage_ingest_root)
        self.storage_processed_dir = self.local_queue_dir / settings.storage_ingest_processed_dir_name
        self._drive_service = None
        self._drive_layout: dict[str, str] | None = None

    @staticmethod
    def _parallel_jobs(count: int, default: int) -> int:
        return max(1, min(count, int(default)))

    def ensure_local_layout(self) -> None:
        self.local_queue_video_dir.mkdir(parents=True, exist_ok=True)
        self.local_queue_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.storage_processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _video_media_type(path_or_name: str | Path) -> str:
        suffix = Path(str(path_or_name)).suffix.lower()
        if suffix == ".mp4":
            return "video/mp4"
        return "application/octet-stream"

    def _build_drive_service(self):
        if not settings.google_drive_enabled:
            raise RuntimeError("Google Drive sync is disabled.")
        if self._drive_service is not None:
            return self._drive_service
        _ensure_shared_secret()
        from shared_secret_runtime import build_google_drive_oauth_service
        self._drive_service = build_google_drive_oauth_service()
        return self._drive_service

    def process_storage_queue(self, session: Session) -> dict:
        self.ensure_local_layout()
        if not settings.storage_ingest_enabled:
            return {
                "processed_videos": 0,
                "imported_source_files": [],
                "evicted_video_ids": [],
            }

        registry = StorageIngestRegistry(self.storage_processed_dir)
        if str(settings.storage_ingest_source_backend or "filesystem").strip().lower() == "google_drive":
            _ensure_shared_secret()
            from shared_secret_runtime import build_google_drive_oauth_service
            drive_service = build_google_drive_oauth_service()
            source_root_id = self._resolve_drive_source_storage_folder_id()
            storage_items = self._scan_drive_pending(drive_service, source_root_id, registry)
        else:
            storage_items = self._scan_storage_pending(registry)

        processed_videos = 0
        imported_source_files: list[str] = []
        evicted_video_ids: list[str] = []

        for item in storage_items:
            try:
                result = self._process_storage_item(session, item)
                evicted_video_ids.extend(result.get("evicted_video_ids", []))
                imported_source_files.append(item.relative_path)
                processed_videos += 1
                registry.mark_processed(item, result)
            except Exception as exc:
                logger.exception("Failed to process storage item: %s", item.source_filename)

        return {
            "processed_videos": processed_videos,
            "imported_source_files": imported_source_files,
            "evicted_video_ids": evicted_video_ids,
        }

    def _scan_storage_pending(self, registry: StorageIngestRegistry) -> list[StorageVideoItem]:
        pending: list[StorageVideoItem] = []
        if not self.storage_ingest_root.exists():
            return pending

        candidates = sorted(self.storage_ingest_root.glob("cam_*/*/*.mp4"))
        for source_path in candidates:
            if not source_path.is_file() or source_path.name.startswith("."):
                continue
            identity = StorageVideoIdentity.parse(source_path)
            if identity is None:
                continue
            try:
                relative_path = source_path.relative_to(self.storage_ingest_root).as_posix()
                stat = source_path.stat()
            except OSError:
                continue

            item = StorageVideoItem(
                source_path=source_path,
                source_drive_file_id=None,
                relative_path=relative_path,
                source_filename=source_path.name,
                camera_id=identity.camera_id,
                recorded_at=identity.recorded_at,
                size_bytes=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
                fingerprint=_fingerprint_storage(relative_path),
            )
            if not self._is_too_new(item) and not registry.is_processed(item):
                pending.append(item)
        return pending

    def _scan_drive_pending(self, drive_service, source_root_id: str, registry: StorageIngestRegistry) -> list[StorageVideoItem]:
        pending: list[StorageVideoItem] = []
        items = self._list_drive_storage_items(drive_service, source_root_id)
        for item in items:
            if not self._is_too_new(item) and not registry.is_processed(item):
                pending.append(item)
        return pending

    def _list_drive_storage_items(self, drive_service, source_root_id: str) -> list[StorageVideoItem]:
        import os
        min_date = (os.environ.get("STORAGE_INGEST_MIN_DATE") or "").strip()
        items: list[StorageVideoItem] = []

        camera_folders = self._list_drive_folders(drive_service, source_root_id)
        for camera_folder in camera_folders:
            if not camera_folder.name.lower().startswith("cam_"):
                continue
            date_folders = self._list_drive_folders(drive_service, camera_folder.id)
            for date_folder in date_folders:
                date_name = date_folder.name
                if min_date and date_name < min_date:
                    continue
                for file_row in self._list_drive_mp4_files(drive_service, date_folder.id):
                    identity = StorageVideoIdentity.parse(Path(str(file_row.get("name") or "")))
                    if identity is None:
                        continue
                    if identity.camera_id != camera_folder.name.lower():
                        continue
                    if date_name != identity.recorded_date.isoformat():
                        continue
                    relative_path = f"{camera_folder.name}/{date_name}/{identity.source_filename}"
                    modified_at = self._parse_drive_time(str(file_row.get("modifiedTime") or ""))
                    items.append(StorageVideoItem(
                        source_path=None,
                        source_drive_file_id=str(file_row["id"]),
                        relative_path=relative_path,
                        source_filename=identity.source_filename,
                        camera_id=identity.camera_id,
                        recorded_at=identity.recorded_at,
                        size_bytes=int(file_row.get("size") or 0),
                        modified_ns=int(modified_at.timestamp() * 1_000_000_000),
                        fingerprint=_fingerprint_drive(relative_path, str(file_row["id"])),
                    ))

        seen: dict[str, StorageVideoItem] = {}
        for item in items:
            existing = seen.get(item.relative_path)
            if existing is None or item.modified_ns > existing.modified_ns:
                seen[item.relative_path] = item
        return sorted(seen.values(), key=lambda item: item.relative_path)

    def _list_drive_folders(self, drive_service, parent_id: str) -> list[Any]:
        query = f"'{parent_id}' in parents and trashed = false and mimeType = '{DRIVE_FOLDER_MIME_TYPE}'"
        response = drive_service.files().list(
            q=query, spaces="drive", fields="files(id, name)",
            pageSize=1000, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        rows = response.get("files") or []
        return [type("F", (), {"id": str(r["id"]), "name": str(r["name"])})() for r in rows]

    def _list_drive_mp4_files(self, drive_service, parent_id: str) -> list[dict]:
        query = f"'{parent_id}' in parents and trashed = false"
        response = drive_service.files().list(
            q=query, spaces="drive", fields="files(id, name, size, modifiedTime, mimeType)",
            pageSize=1000, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        return [r for r in (response.get("files") or []) if str(r.get("name") or "").lower().endswith(".mp4")]

    def _parse_drive_time(self, raw: str) -> datetime:
        value = raw.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)

    @staticmethod
    def _is_too_new(item: StorageVideoItem) -> bool:
        if item.modified_ns <= 0:
            return False
        modified_seconds = item.modified_ns / 1_000_000_000
        return (time.time() - modified_seconds) < settings.storage_ingest_min_file_age_seconds

    def _resolve_drive_source_storage_folder_id(self) -> str:
        configured_id = str(settings.google_drive_source_storage_folder_id or "").strip()
        if configured_id:
            return configured_id
        vinuni_id = str(settings.google_drive_vinuni_folder_id or "").strip()
        root_id = str(settings.google_drive_root_folder_id or "").strip()
        if not vinuni_id and not root_id:
            raise RuntimeError("Set GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID or VINUNI/ROOT folder IDs")
        if not vinuni_id:
            vinuni_id = self._ensure_drive_folder(root_id, settings.google_drive_vinuni_folder_name)
        folder = self._find_drive_child(vinuni_id, settings.google_drive_source_storage_folder_name)
        if folder is None:
            raise RuntimeError(f"Storage folder not found: {settings.google_drive_source_storage_folder_name}")
        return folder.id

    def _ensure_drive_folder(self, parent_id: str, name: str) -> str:
        existing = self._find_drive_child(parent_id, name)
        if existing:
            return existing.id
        service = self._build_drive_service()
        created = service.files().create(
            body={"name": name, "mimeType": DRIVE_FOLDER_MIME_TYPE, "parents": [parent_id]},
            fields="id", supportsAllDrives=True,
        ).execute()
        return str(created["id"])

    def _find_drive_child(self, parent_id: str, name: str) -> Any | None:
        escaped = name.replace("'", "\\'")
        query = f"'{parent_id}' in parents and trashed = false and mimeType = '{DRIVE_FOLDER_MIME_TYPE}' and name = '{escaped}'"
        service = self._build_drive_service()
        response = service.files().list(
            q=query, spaces="drive", fields="files(id, name)",
            pageSize=10, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        rows = response.get("files") or []
        if not rows:
            return None
        r = rows[0]
        return type("F", (), {"id": str(r["id"]), "name": str(r["name"])})()

    def _process_storage_item(self, session: Session, item: StorageVideoItem) -> dict:
        from shared.models import QueueVideoAsset

        video_id = str(item.recorded_at.strftime("%Y%m%d_%H%M%S")) + "_" + item.camera_id
        queue_position = self._get_next_queue_position(session) + 1
        available_link_video = f"/api/v1/queue/videos/{video_id}/file"
        available_link_metadata = f"/api/v1/queue/videos/{video_id}/metadata"

        existing = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
        if existing:
            return {"evicted_video_ids": []}

        row = QueueVideoAsset(
            video_id=video_id,
            camera_id=item.camera_id,
            title=item.source_filename,
            queue_position=queue_position,
            available_link_video=available_link_video,
            available_link_metadata=available_link_metadata,
            storage_backend="local_queue_storage",
            source_filename=item.source_filename,
            source_mode=STORAGE_INGEST_SOURCE_MODE,
            local_video_path=str(item.source_path) if item.source_path else None,
            raw_video_metadata={"source": "storage_ingest"},
        )
        session.add(row)
        session.commit()
        return {"evicted_video_ids": []}

    def _get_next_queue_position(self, session: Session) -> int:
        from shared.models import QueueVideoAsset
        result = session.query(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.desc()).first()
        return result.queue_position if result else 0


def _ensure_shared_secret() -> None:
    import sys
    if A20_ROOT and str(A20_ROOT) not in sys.path:
        sys.path.insert(0, str(A20_ROOT))


def _fingerprint_storage(relative_path: str) -> str:
    import hashlib
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()


def _fingerprint_drive(relative_path: str, file_id: str) -> str:
    import hashlib
    return hashlib.sha1(f"{relative_path}|{file_id}".encode("utf-8")).hexdigest()


def queue_video_to_payload(video) -> dict:
    return {
        "video_id": video.video_id,
        "camera_id": video.camera_id,
        "title": video.title,
        "queue_position": video.queue_position,
        "storage_backend": video.storage_backend,
        "available_link_video": video.available_link_video,
        "available_link_metadata": video.available_link_metadata,
        "source_filename": video.source_filename,
        "source_mode": video.source_mode,
        "created_at": video.created_at,
        "updated_at": video.updated_at,
        "local_video_path": video.local_video_path,
        "local_metadata_path": video.local_metadata_path,
        "raw_video_metadata": video.raw_video_metadata or {},
    }


def list_queue_videos(session: Session) -> list[dict]:
    from shared.models import QueueVideoAsset
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return [queue_video_to_payload(video) for video in session.scalars(statement).all()]


def load_queue_video_metadata(session: Session, video_id: str) -> dict[str, Any]:
    from shared.models import QueueVideoAsset, Tracklet
    row = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    if row is None:
        raise FileNotFoundError(f"Queue video not found: {video_id}")
    metadata_path = Path(str(row.local_metadata_path or "")).expanduser()
    if metadata_path.exists() and metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    people = (
        session.scalars(
            select(Tracklet)
            .where(Tracklet.video_id == video_id)
            .order_by(Tracklet.start_time.asc(), Tracklet.id.asc())
        ).all()
    )
    return {
        "video": row.raw_video_metadata or {},
        "people": [_tracklet_to_candidate_payload(t) for t in people],
        "source": "database_fallback",
    }


def _tracklet_to_candidate_payload(tracklet) -> dict:
    """Convert Tracklet row to legacy candidate payload shape for frontend compatibility."""
    return {
        "candidate_id": tracklet.tracklet_id,
        "camera_id": tracklet.camera_id,
        "video_id": tracklet.video_id,
        "track_id": tracklet.track_id,
        "gender": tracklet.gender,
        "upper_color": tracklet.upper_clothing_color or "unknown",
        "lower_color": tracklet.lower_clothing_color or "unknown",
        "appearance_summary": tracklet.appearance_summary,
        "quality_score": tracklet.quality_score,
        "occlusion_score": tracklet.occlusion_score,
        "start_time": tracklet.start_time,
        "end_time": tracklet.end_time,
        "representative_bbox": tracklet.representative_bbox,
        "contributing_cameras": tracklet.contributing_cameras,
        "contributing_video_ids": tracklet.contributing_video_ids,
        "batch_id": tracklet.batch_id,
    }


def load_queue_video_file_path(session: Session, video_id: str) -> Path:
    from shared.models import QueueVideoAsset
    row = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    if row is None:
        raise FileNotFoundError(f"Queue video not found: {video_id}")
    path = Path(str(row.local_video_path or "")).expanduser()
    if path.exists() and path.is_file():
        return path
    raise FileNotFoundError(f"Queue video file not found: {video_id}")


def sync_local_queue_state(session: Session, *, only_if_empty: bool = False) -> dict[str, int]:
    from shared.models import Tracklet, QueueVideoAsset, User
    from sqlalchemy import func

    existing_candidates = int(session.scalar(select(func.count()).select_from(Tracklet)) or 0)
    existing_queue_videos = int(session.scalar(select(func.count()).select_from(QueueVideoAsset)) or 0)
    if only_if_empty and (existing_candidates > 0 or existing_queue_videos > 0):
        return {
            "imported_count": 0,
            "updated_count": 0,
            "queue_videos_count": existing_queue_videos,
            "file_count": 0,
        }

    local_root = Path(settings.queue_local_root)
    queue_root = local_root / "local" / settings.google_drive_queue_folder_name
    metadata_root = queue_root / settings.google_drive_metadata_folder_name
    video_root = queue_root / settings.queue_video_folder_name
    metadata_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(metadata_root.glob("*.json"))
    imported_count = 0
    updated_count = 0
    queue_videos_count = 0

    for queue_position, metadata_path in enumerate(metadata_files, start=1):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        video_payload = payload.get("video") or {}
        if not isinstance(video_payload, dict):
            video_payload = {}
        people_payload = payload.get("people") or []
        if not isinstance(people_payload, list):
            people_payload = []

        video_id = str(video_payload.get("video_id") or metadata_path.stem).strip()
        if not video_id:
            continue
        local_video_path = Path(str(video_payload.get("compressed_path") or "")).expanduser()
        # Use absolute path directly - no PROJECT_ROOT fallback for VPS deployment
        if not local_video_path.is_absolute():
            local_video_path = Path("/workspace/storage/videos") / local_video_path.name
        if not local_video_path.exists():
            fallback_video_path = video_root / f"{metadata_path.stem}.mp4"
            if fallback_video_path.exists():
                local_video_path = fallback_video_path

        title = str(
            video_payload.get("source_path")
            or video_payload.get("camera_id")
            or metadata_path.stem
        ).strip()
        source_filename = Path(title).name if title else f"{metadata_path.stem}.mp4"
        available_link_video = f"/api/v1/queue/videos/{video_id}/file"
        available_link_metadata = f"/api/v1/queue/videos/{video_id}/metadata"

        _upsert_queue_video_asset(
            session,
            video_id=video_id,
            camera_id=str(video_payload.get("camera_id") or "").strip() or None,
            title=source_filename,
            queue_position=queue_position,
            available_link_video=available_link_video,
            available_link_metadata=available_link_metadata,
            storage_backend="local_queue_storage",
            source_filename=source_filename,
            source_mode=str(video_payload.get("source_mode") or "").strip() or None,
            drive_video_file_id=None,
            drive_metadata_file_id=None,
            local_video_path=str(local_video_path) if local_video_path.exists() else None,
            local_metadata_path=str(metadata_path),
            raw_video_metadata=video_payload,
        )
        counts = _upsert_person_candidates(session, people_payload, str(metadata_path))
        imported_count += int(counts.get("imported_count") or 0)
        updated_count += int(counts.get("updated_count") or 0)
        queue_videos_count += 1

    session.commit()
    return {
        "imported_count": imported_count,
        "updated_count": updated_count,
        "queue_videos_count": queue_videos_count,
        "file_count": len(metadata_files),
    }


def _upsert_queue_video_asset(
    session: Session,
    *,
    video_id: str,
    camera_id: str | None,
    title: str,
    queue_position: int,
    available_link_video: str,
    available_link_metadata: str | None,
    storage_backend: str,
    source_filename: str | None,
    source_mode: str | None,
    drive_video_file_id: str | None,
    drive_metadata_file_id: str | None,
    local_video_path: str | None,
    local_metadata_path: str | None,
    raw_video_metadata: dict,
):
    from shared.models import QueueVideoAsset
    row = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    values = {
        "video_id": video_id,
        "camera_id": camera_id,
        "title": title,
        "queue_position": queue_position,
        "available_link_video": available_link_video,
        "available_link_metadata": available_link_metadata,
        "storage_backend": storage_backend,
        "source_filename": source_filename,
        "source_mode": source_mode,
        "drive_video_file_id": drive_video_file_id,
        "drive_metadata_file_id": drive_metadata_file_id,
        "local_video_path": local_video_path,
        "local_metadata_path": local_metadata_path,
        "raw_video_metadata": raw_video_metadata,
    }
    if row is None:
        row = QueueVideoAsset(**values)
        session.add(row)
    else:
        for key, value in values.items():
            setattr(row, key, value)
    session.flush()
    return row


def _upsert_person_candidates(session: Session, people: list[dict], metadata_path: str | None = None) -> dict[str, int]:
    """Upsert person candidates into Tracklet table (legacy path for local queue sync).

    Maps old PersonCandidate fields to new Tracklet schema:
      - candidate_id     -> tracklet_id
      - camera_id        -> camera_id
      - video_id        -> video_id
      - raw_metadata     -> TrackletEmbedding row (if has siglip_embedding)
    """
    from shared.models import Tracklet, TrackletEmbedding
    imported_count = 0
    updated_count = 0

    for person in people:
        if not isinstance(person, dict):
            continue
        tracklet_id = str(person.get("candidate_id") or "").strip()
        if not tracklet_id:
            continue

        camera_id = str(person.get("camera_id") or "")
        video_id = str(person.get("video_id") or "")
        gender = str(person.get("gender") or "unknown")
        upper_color = str(person.get("upper_clothing_color") or "unknown")
        lower_color = str(person.get("lower_clothing_color") or "unknown")
        appearance_summary = str(person.get("appearance_summary") or "")
        raw = person.get("raw_metadata") or {}

        existing = session.scalar(select(Tracklet).where(Tracklet.tracklet_id == tracklet_id))
        if existing:
            existing.camera_id = camera_id
            existing.gender = gender
            existing.upper_clothing_color = upper_color
            existing.lower_clothing_color = lower_color
            existing.appearance_summary = appearance_summary
            updated_count += 1
        else:
            row = Tracklet(
                tracklet_id=tracklet_id,
                video_id=video_id or "unknown",
                camera_id=camera_id,
                track_id=str(person.get("track_id") or "0"),
                start_time=0.0,
                end_time=0.0,
                quality_score=float(raw.get("quality_score") or 0.0),
                occlusion_score=float(raw.get("occlusion_score") or 0.0),
                gender=gender,
                age_range="unknown",
                upper_clothing_color=upper_color,
                lower_clothing_color=lower_color,
                shoes_color="unknown",
                appearance_summary=appearance_summary,
                representative_bbox=raw.get("representative_bbox") or [],
                contributing_cameras=raw.get("contributing_cameras") or [],
                contributing_video_ids=raw.get("contributing_video_ids") or [],
                batch_id=metadata_path,
            )
            session.add(row)
            imported_count += 1

        # Also upsert SigLIP2 embedding if present
        siglip_vec = raw.get("siglip_embedding") or []
        if siglip_vec and len(siglip_vec) > 0:
            emb_existing = session.scalar(select(TrackletEmbedding).where(TrackletEmbedding.tracklet_id == tracklet_id))
            if emb_existing:
                emb_existing.siglip_embedding = siglip_vec
                emb_existing.model_version = "siglip2"
            else:
                session.add(TrackletEmbedding(
                    tracklet_id=tracklet_id,
                    siglip_embedding=siglip_vec,
                    model_version="siglip2",
                ))

    session.flush()
    return {"imported_count": imported_count, "updated_count": updated_count}


def delete_queue_video_asset(session: Session, video_id: str) -> None:
    from shared.models import QueueVideoAsset, Tracklet

    session.execute(delete(Tracklet).where(Tracklet.video_id == video_id))
    session.execute(delete(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    session.flush()
