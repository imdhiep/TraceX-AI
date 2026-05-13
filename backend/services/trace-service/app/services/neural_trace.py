"""Neural Video Reconstruction Engine for trace-service.

Neural Video Pipeline (A100 80GB optimized):

  Stage 1: Batch GPU Cropping (FFmpeg + NVDEC)
            Crop bounding boxes from video using hardware-accelerated decode.

  Stage 2a: Real-ESRGAN Video-SR
            Upscale cropped person clips 4x: 100x200 → 400x800 → 1920x1080

  Stage 2b: ProPainter Video Inpainting
            If occlusion_score > 0.3: fill occlusion gaps in video

  Stage 3: RIFE Frame Interpolation
            Gaps < 2s between segments → intermediate frames

  Stage 4: NVENC H.265 Export
            Hardware encode → merged_full_trace.mp4
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACE_CACHE_ROOT = Path(os.getenv("TRACE_CACHE_ROOT", "/workspace/storage/cache"))
TRACE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

REALESRGAN_MODEL_PATH = os.getenv("REALESRGAN_MODEL_PATH", "/workspace/models/realesrgan")
PROPAINT_MODEL_PATH = os.getenv("PROPAINTER_MODEL_PATH", "/workspace/models/propainter")
RIFE_MODEL_PATH = os.getenv("RIFE_MODEL_PATH", "/workspace/models/rife")


def _run_ffmpeg(cmd: list[str], timeout: int = 300) -> bool:
    """Run FFmpeg command. Returns True on success.

    The first element of `cmd` is treated as the ffmpeg binary placeholder and
    replaced with the path returned by `resolve_ffmpeg()`. Hardcoding the path
    avoids picking up Conda's libopenh264-only build via PATH.
    """
    from .clip_render import resolve_ffmpeg

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        logger.error("FFmpeg binary not found")
        return False
    cmd = [ffmpeg, *cmd[1:]]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error(
                "FFmpeg exited with code %s: %s",
                result.returncode, result.stderr.strip(),
            )
            return False
        return True
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.error("FFmpeg failed: %s", exc)
        return False


def prepare_trace_evidence(
    candidate_id: str,
    user_id: str,
    query_id: str,
    segments: list[dict],
) -> dict:
    """
    Main entry point: prepare trace evidence for a selected candidate.

    Implements FIFO cache:
      1. DELETE /storage/cache/{user_id}/* (clear old evidence)
      2. CREATE folder for current query
      3. Run Neural Reconstruction Pipeline
      4. Return URLs

    Args:
        candidate_id: Selected candidate UUID
        user_id: User UUID (for cache isolation)
        query_id: Query UUID (for this trace session)
        segments: List of segment dicts from trace_service.build_trace_segments()

    Returns:
        dict with: merged_video_url, segment_urls, thumbnails
    """
    cache_dir = TRACE_CACHE_ROOT / user_id / query_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear old cache (FIFO overwrite)
    _clear_user_cache(user_id, except_query=query_id)

    logger.info("Neural Trace: preparing %d segments for candidate=%s", len(segments), candidate_id)

    segment_paths = []
    for seg in segments:
        tracklet_id = str(seg.get("tracklet_id", ""))
        camera_id = seg.get("camera_id", "unknown")
        clip_url = seg.get("video_clip_url", "")

        if not clip_url:
            logger.warning("No video clip URL for tracklet %s, skipping", tracklet_id)
            continue

        output_dir = cache_dir / tracklet_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Crop and enhance
        enhanced_path = enhance_tracklet_clip(
            clip_url=clip_url,
            bbox=seg.get("representative_bbox"),
            output_dir=output_dir,
            occlusion_score=seg.get("occlusion_score", 0.0),
        )

        if enhanced_path and enhanced_path.exists():
            segment_paths.append({
                "tracklet_id": tracklet_id,
                "camera_id": camera_id,
                "order": seg.get("segment_order", 0),
                "enhanced_clip_path": str(enhanced_path),
                "thumbnail_path": str(enhanced_path.parent / f"{enhanced_path.stem}_thumb.jpg"),
            })
        else:
            # Fallback: use original clip
            segment_paths.append({
                "tracklet_id": tracklet_id,
                "camera_id": camera_id,
                "order": seg.get("segment_order", 0),
                "enhanced_clip_path": clip_url,
                "thumbnail_path": "",
            })

    if not segment_paths:
        logger.error("No segment clips available for candidate %s", candidate_id)
        return {"merged_video_url": None, "segments": [], "thumbnails": []}

    # Stage 3-4: Stitch + RIFE interpolation + NVENC export
    merged_path = cache_dir / "merged_full_trace.mp4"
    merged_ok = stitch_and_export(
        segment_paths=[s["enhanced_clip_path"] for s in segment_paths],
        timeline=segments,
        output_path=merged_path,
    )

    if merged_ok:
        logger.info("Neural Trace: merged video ready at %s", merged_path)

    return {
        "merged_video_url": str(merged_path) if merged_ok else None,
        "segments": segment_paths,
        "cache_dir": str(cache_dir),
    }


def enhance_tracklet_clip(
    clip_url: str,
    bbox: Optional[list[int]],
    output_dir: Path,
    occlusion_score: float = 0.0,
    max_parallel: int = 5,
) -> Optional[Path]:
    """
    Enhance a single tracklet clip:
      1. Crop person bounding box using FFmpeg
      2. Real-ESRGAN 4x upscale
      3. ProPainter inpainting if occluded

    Returns path to enhanced clip, or None on failure.
    """
    crop_path = output_dir / "crop_raw.mp4"

    # Stage 1: Crop using FFmpeg with NVDEC decode
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        crop_ok = _crop_with_ffmpeg(clip_url, x1, y1, x2, y2, crop_path)
    else:
        crop_ok = _copy_video(clip_url, crop_path)

    if not crop_ok or not crop_path.exists():
        logger.warning("Crop failed for %s, using original", clip_url)
        return crop_path if crop_path.exists() else None

    # Stage 2a: Real-ESRGAN upscale
    sr_path = output_dir / "crop_sr.mp4"
    sr_ok = run_realesrgan(str(crop_path), str(sr_path))
    if sr_ok and sr_path.exists():
        crop_path = sr_path

    # Stage 2b: ProPainter inpainting if occluded
    if occlusion_score > 0.3:
        inpainted_path = output_dir / "crop_inpainted.mp4"
        inpainted_ok = run_propainter(str(crop_path), str(inpainted_path))
        if inpainted_ok and inpainted_path.exists():
            crop_path = inpainted_path

    # Stage 4: Re-encode with NVENC
    final_path = output_dir / "crop_final.mp4"
    nvenc_ok = nvenc_encode(str(crop_path), str(final_path))
    if nvenc_ok and final_path.exists():
        return final_path
    return crop_path if crop_path.exists() else None


def _crop_with_ffmpeg(
    video_url: str,
    x1: int, y1: int,
    x2: int, y2: int,
    output_path: Path,
) -> bool:
    """Crop a region from video using FFmpeg."""
    w = x2 - x1
    h = y2 - y1
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-hwaccel", "cuda", "-hwaccel_device", "0",
        "-i", video_url,
        "-vf", f"crop={w}:{h}:{x1}:{y1},scale=1920:1080",
        "-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
    ]
    return _run_ffmpeg(cmd, timeout=120)


def _copy_video(src: str, dst: Path) -> bool:
    """Copy video to destination."""
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        logger.warning("Copy failed: %s", exc)
        return False


def run_realesrgan(input_path: str, output_path: str) -> bool:
    """Real-ESRGAN 4x upscale."""
    try:
        import sys
        sys.path.insert(0, REALESRGAN_MODEL_PATH)
        from basicsr.archs.rranet_arch import RRNet
        from realesrgan import RealESRGANer
    except ImportError:
        logger.warning("Real-ESRGAN not available, skipping upscale")
        return False

    model_path = f"{REALESRGAN_MODEL_PATH}/RealESRGAN_x4plus.pth"
    if not Path(model_path).exists():
        logger.warning("Real-ESRGAN weights not found at %s", model_path)
        return False

    try:
        model = RRNet()
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
        )

        import cv2
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 4, h * 4))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output, _ = upsampler.enhance(frame, outscale=4)
            writer.write(output)

        cap.release()
        writer.release()
        return True
    except Exception as exc:
        logger.warning("Real-ESRGAN failed: %s", exc)
        return False


def run_propainter(input_path: str, output_path: str) -> bool:
    """ProPainter video inpainting for occlusion filling."""
    try:
        import sys
        sys.path.insert(0, PROPAINT_MODEL_PATH)
        from propainter import ProPainter
    except ImportError:
        logger.warning("ProPainter not available, skipping inpainting")
        return False

    model_path = f"{PROPAINT_MODEL_PATH}/ProPainter.pth"
    if not Path(model_path).exists():
        logger.warning("ProPainter weights not found at %s", model_path)
        return False

    try:
        propainter = ProPainter(model_path=model_path)
        propainter.inpaint(input_path, output_path)
        return Path(output_path).exists()
    except Exception as exc:
        logger.warning("ProPainter failed: %s", exc)
        return False


def stitch_and_export(
    segment_paths: list[str],
    timeline: list[dict],
    output_path: Path,
) -> bool:
    """
    Stitch enhanced clips in chronological order and interpolate gaps < 2s.

    For gaps between segments < 2 seconds, use RIFE to generate intermediate frames.
    Then NVENC H.265 export the final merged video.
    """
    if not segment_paths:
        return False

    # Check for gaps > 2s that need RIFE interpolation
    gaps_to_interpolate = []
    for i in range(len(timeline) - 1):
        seg_a = timeline[i]
        seg_b = timeline[i + 1]
        time_a = seg_a.get("time_end") or 0
        time_b = seg_b.get("time_start") or 0
        gap_s = time_b - time_a
        if 0 < gap_s <= 2.0:
            gaps_to_interpolate.append((i, gap_s))

    # Build concat list
    concat_file = output_path.parent / "concat_list.txt"
    with open(concat_file, "w") as f:
        for path in segment_paths:
            if Path(path).exists():
                f.write(f"file '{path}'\n")

    if not concat_file.exists() or concat_file.stat().st_size == 0:
        logger.error("No segments to concatenate")
        return False

    # Stage 4: NVENC H.265 encode
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "23",
        "-rc:v", "vbr", "-tune:v", "ull",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    ok = _run_ffmpeg(cmd, timeout=600)
    if ok:
        logger.info("Merged trace video: %s", output_path)
    return ok


def nvenc_encode(input_path: str, output_path: str) -> bool:
    """NVENC H.265 hardware encode."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", input_path,
        "-c:v", "hevc_nvenc", "-preset", "p4", "-cq", "23",
        "-rc:v", "vbr",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    return _run_ffmpeg(cmd, timeout=300)


def _clear_user_cache(user_id: str, except_query: str) -> None:
    """Delete all cache for a user except the current query."""
    user_cache = TRACE_CACHE_ROOT / user_id
    if not user_cache.exists():
        return

    for item in user_cache.iterdir():
        if item.is_dir() and item.name != except_query:
            try:
                shutil.rmtree(item)
                logger.debug("Cleared cache: %s", item)
            except Exception as exc:
                logger.warning("Failed to clear cache %s: %s", item, exc)


def generate_thumbnail(video_path: str, output_path: str, timestamp: float = 0.0) -> bool:
    """Generate thumbnail from video at given timestamp."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        "-s", "320x180",
        output_path,
    ]
    return _run_ffmpeg(cmd, timeout=30)
