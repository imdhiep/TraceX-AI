"""Video process router — redirects to metadata-service (GPU AI models).

trace-service forwards /api/v1/video/process to metadata-service.
GPU models (RT-DETR, SigLIP 2, VideoMAE V2, Qwen2-VL-7B) live in metadata-service.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video_process"])

_METADATA_SERVICE_URL = os.getenv(
    "METADATA_SERVICE_URL",
    "http://metadata-service:8002"
)


@router.post("/api/v1/video/process")
def process_video(req: dict[str, Any]) -> dict[str, Any]:
    """Forward video processing request to metadata-service.

    The metadata-service handles:
      - RT-DETR person detection
      - SigLIP 2 appearance embeddings
      - Qwen2-VL attribute captioning
      - VideoMAE V2 action classification

    Returns per-video tracklets with appearance metadata, SigLIP embeddings, and action labels.
    """
    url = f"{_METADATA_SERVICE_URL.rstrip('/')}/api/v1/video/process"
    try:
        with httpx.Client(timeout=600) as client:
            response = client.post(url, json=req)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("metadata-service error: %s %s", exc.response.status_code, exc.response.text[:500])
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text[:500])
    except httpx.TimeoutException:
        logger.error("metadata-service timeout for video %s", req.get("video_id"))
        raise HTTPException(status_code=504, detail="Video processing timed out")
    except Exception as exc:
        logger.exception("metadata-service unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=f"metadata-service unavailable: {exc}")
