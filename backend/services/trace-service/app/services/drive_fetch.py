"""On-demand Google Drive video fetch with a small disk LRU cache.

Source videos are not persisted on disk after ingest (the pipeline streams from
Drive → GPU → DB). To render an evidence clip with a moving bbox, trace-service
needs the source MP4 bytes. This module downloads the file from Drive on first
use, keeps it in `TRACE_CACHE_ROOT/drive/`, and serves the cached copy for
subsequent renders.

The cache is bounded by `DRIVE_CACHE_MAX_BYTES` (default 20 GiB). Eviction is
LRU by file mtime when a download would push the cache over the limit.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(os.getenv("TRACE_CACHE_ROOT", "/workspace/storage/cache")) / "drive"
_MAX_BYTES = int(os.getenv("DRIVE_CACHE_MAX_BYTES", str(20 * 1024 * 1024 * 1024)))
_SECRETS_ROOT = os.getenv("MCPT_SECRETS_ROOT", "/workspace/secrets")

# Per-drive-id locks so concurrent renders for the same video share one download.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(drive_file_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(drive_file_id)
        if lock is None:
            lock = threading.Lock()
            _locks[drive_file_id] = lock
        return lock


def _build_drive_service():
    if _SECRETS_ROOT not in sys.path:
        sys.path.insert(0, _SECRETS_ROOT)
    import importlib
    mod = importlib.import_module("shared_secret_runtime")
    try:
        return mod.build_google_drive_sa_service()
    except FileNotFoundError:
        return mod.build_google_drive_service()


def _evict_to_fit(needed_bytes: int) -> None:
    """Remove oldest cached files until used + needed <= max."""
    if not _CACHE_ROOT.exists():
        return
    files = [p for p in _CACHE_ROOT.iterdir() if p.is_file()]
    used = sum(p.stat().st_size for p in files)
    if used + needed_bytes <= _MAX_BYTES:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    for p in files:
        try:
            sz = p.stat().st_size
            p.unlink()
            used -= sz
            logger.info("[drive_fetch] evicted %s (%d bytes)", p.name, sz)
            if used + needed_bytes <= _MAX_BYTES:
                break
        except OSError:
            continue


def fetch_video(drive_file_id: str, suffix: str = ".mp4") -> Optional[Path]:
    """Return a local path to the video file for `drive_file_id`, downloading
    from Drive if the cache misses. Returns None on failure.

    Concurrent calls for the same id share one download. The returned file's
    mtime is touched so LRU eviction sees it as recently used.
    """
    if not drive_file_id:
        return None

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    target = _CACHE_ROOT / f"{drive_file_id}{suffix}"

    if target.exists() and target.stat().st_size > 0:
        os.utime(target, None)
        return target

    lock = _lock_for(drive_file_id)
    with lock:
        # Recheck under lock — another thread may have just finished.
        if target.exists() and target.stat().st_size > 0:
            os.utime(target, None)
            return target

        try:
            from googleapiclient.http import MediaIoBaseDownload
        except Exception as exc:
            logger.warning("[drive_fetch] googleapiclient unavailable: %s", exc)
            return None

        try:
            svc = _build_drive_service()
        except Exception as exc:
            logger.warning("[drive_fetch] cannot build Drive service: %s", exc)
            return None

        try:
            meta = svc.files().get(
                fileId=drive_file_id,
                fields="id,name,size",
                supportsAllDrives=True,
            ).execute()
            size = int(meta.get("size") or 0)
        except Exception as exc:
            logger.warning("[drive_fetch] metadata fetch failed for %s: %s", drive_file_id, exc)
            size = 0

        if size > 0:
            _evict_to_fit(size)

        tmp = target.with_suffix(target.suffix + ".part")
        try:
            request = svc.files().get_media(fileId=drive_file_id, supportsAllDrives=True)
            with open(tmp, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=64 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            tmp.replace(target)
            logger.info(
                "[drive_fetch] cached %s (%d bytes) -> %s",
                drive_file_id, target.stat().st_size, target,
            )
            return target
        except Exception as exc:
            logger.warning("[drive_fetch] download failed for %s: %s", drive_file_id, exc)
            tmp.unlink(missing_ok=True)
            return None
