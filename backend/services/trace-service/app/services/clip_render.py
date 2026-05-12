"""Evidence clip rendering — extract per-tracklet clip with a moving bbox.

Given a tracklet's source video + its list of per-frame observations from
`tracklet_observations`, write an mp4 clip covering [start_time, end_time]
with a bbox drawn on each frame, interpolated linearly between observed
frame indexes.

Output path: /workspace/storage/traces/{query_id}/{candidate_id}/{tracklet_id}.mp4
Static URL : /static/traces/{query_id}/{candidate_id}/{tracklet_id}.mp4
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

import cv2

logger = logging.getLogger(__name__)

_TRACES_ROOT = Path(os.getenv("TRACES_DIR", "/workspace/storage/traces"))
_STATIC_PREFIX = "/static/traces"

# Bbox visual settings (BGR)
_BBOX_COLOR = (0, 255, 0)
_BBOX_THICKNESS = 3


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "tracklet"


def _build_source_frame_map(
    observations: list[dict],
    fps: float,
) -> dict[int, list[int]]:
    """Return {source_video_frame_index -> [x1,y1,x2,y2]}.

    The tracker stores `frame_index` in sampled-frame space. `timestamp_second`
    is the stable coordinate for the original video, so convert it back to the
    source frame index using the opened video's FPS.
    """
    out: dict[int, list[int]] = {}
    for o in observations:
        ts = float(o.get("timestamp_second") or 0.0)
        fi = max(0, int(round(ts * fps)))
        bbox = o.get("bbox") or []
        if len(bbox) >= 4:
            out[fi] = [int(v) for v in bbox[:4]]
    return out


def _interp_bbox(
    frame_idx: int,
    sorted_obs_frames: list[int],
    obs_map: dict[int, list[int]],
) -> list[int] | None:
    """Linear-interpolate bbox at `frame_idx` from neighbouring observations.

    If frame_idx is before the first obs or after the last, clamp to the nearest.
    """
    if not sorted_obs_frames:
        return None
    if frame_idx <= sorted_obs_frames[0]:
        return obs_map[sorted_obs_frames[0]]
    if frame_idx >= sorted_obs_frames[-1]:
        return obs_map[sorted_obs_frames[-1]]
    # Binary search would be tidier; linear scan is fine for typical N≈200
    for i in range(len(sorted_obs_frames) - 1):
        lo, hi = sorted_obs_frames[i], sorted_obs_frames[i + 1]
        if lo <= frame_idx <= hi:
            t = (frame_idx - lo) / max(1, (hi - lo))
            a, b = obs_map[lo], obs_map[hi]
            return [int(a[k] + t * (b[k] - a[k])) for k in range(4)]
    return None


def render_tracklet_clip(
    *,
    source_video_path: str,
    observations: list[dict],
    start_time: float,
    end_time: float,
    query_id: str,
    candidate_id: str,
    tracklet_id: str,
    draw_bbox: bool = True,
) -> str | None:
    """Render an mp4 clip for [start_time, end_time] of `source_video_path` with
    a bbox drawn on each frame (interpolated from `observations`).

    Returns a `/static/traces/...` URL path on success, or None on failure.

    `observations` shape: [{frame_index, timestamp_second, bbox: [x1,y1,x2,y2], confidence}, ...]
    """
    src = Path(source_video_path)
    if not src.exists():
        logger.warning("[clip_render] source video missing: %s", src)
        return None

    safe_query_id = _slugify(str(query_id))
    safe_candidate_id = _slugify(str(candidate_id))
    out_dir = _TRACES_ROOT / safe_query_id / safe_candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tracklet_id = _slugify(tracklet_id)
    out_path = out_dir / f"{safe_tracklet_id}.mp4"

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        logger.warning("[clip_render] cannot open: %s", src)
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_frame = max(0, int(round(start_time * fps)))
        end_frame = min(total_frames - 1, int(round(end_time * fps))) if total_frames else int(round(end_time * fps))
        if end_frame < start_frame:
            logger.warning("[clip_render] empty range %d→%d", start_frame, end_frame)
            return None

        obs_map = _build_source_frame_map(observations, float(fps))
        sorted_obs_frames = sorted(obs_map)

        # mp4v works in most environments; switch to avc1 if you have encoder support
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.warning("[clip_render] cannot open writer for %s", out_path)
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fi in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if draw_bbox:
                bbox = obs_map.get(fi) or _interp_bbox(fi, sorted_obs_frames, obs_map)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width - 1, x2))
                    y2 = max(0, min(height - 1, y2))
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), _BBOX_COLOR, _BBOX_THICKNESS)

            writer.write(frame)

        writer.release()
    finally:
        cap.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        logger.warning("[clip_render] output not produced: %s", out_path)
        return None

    rel = f"{safe_query_id}/{safe_candidate_id}/{safe_tracklet_id}.mp4"
    return f"{_STATIC_PREFIX}/{rel}"
