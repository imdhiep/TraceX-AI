"""Candidates router — serves preview thumbnails for tracklet candidates."""

from __future__ import annotations

import hashlib
import io
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

router = APIRouter(tags=["candidates"])

# Palette for deterministic card colors based on candidate_id hash
_PALETTE = [
    (99, 102, 241),   # indigo
    (59, 130, 246),   # blue
    (16, 185, 129),   # emerald
    (245, 158, 11),   # amber
    (239, 68, 68),    # red
    (168, 85, 247),   # purple
    (20, 184, 166),   # teal
    (249, 115, 22),   # orange
]


def _placeholder_png(candidate_id: str) -> bytes:
    """Generate a 320×180 placeholder PNG with the candidate_id as label."""
    h = int(hashlib.md5(candidate_id.encode()).hexdigest(), 16)
    bg = _PALETTE[h % len(_PALETTE)]
    fg = (255, 255, 255)

    img = Image.new("RGB", (320, 180), color=bg)
    draw = ImageDraw.Draw(img)

    # Draw a simple person silhouette shape
    cx, cy = 160, 75
    draw.ellipse([cx - 22, cy - 30, cx + 22, cy + 10], fill=fg)
    draw.rectangle([cx - 20, cy + 10, cx + 20, cy + 55], fill=fg)
    draw.rectangle([cx - 28, cy + 30, cx - 12, cy + 70], fill=fg)
    draw.rectangle([cx + 12, cy + 30, cx + 28, cy + 70], fill=fg)

    # Label: last 12 chars of candidate_id
    label = candidate_id[-12:] if len(candidate_id) > 12 else candidate_id
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 155), label, fill=fg, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@router.get("/{candidate_id}/preview")
def candidate_preview(candidate_id: str) -> Response:
    """Return a preview thumbnail JPEG for a candidate tracklet.

    Serves the saved crop image from /workspace/storage/crops/.
    Falls back to a deterministic placeholder if no crop is available.
    """
    from pathlib import Path
    from app.database import SessionLocal
    from shared.models import Tracklet

    _CROPS_ROOT = Path("/workspace/storage/crops")

    db = SessionLocal()
    try:
        tracklet = db.query(Tracklet).filter(Tracklet.tracklet_id == candidate_id).first()
        if tracklet and tracklet.crop_url and tracklet.crop_url.strip():
            # crop_url is "/static/crops/{filename}" — resolve to disk path
            filename = tracklet.crop_url.lstrip("/").removeprefix("static/crops/")
            crop_path = _CROPS_ROOT / filename
            if crop_path.exists():
                return Response(
                    content=crop_path.read_bytes(),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"},
                )
            # Local file missing — proxy to LightningAI metadata-service if configured
            lightning_base = os.getenv("LIGHTNING_METADATA_URL", "").rstrip("/")
            if lightning_base:
                try:
                    upstream = f"{lightning_base}/api/v1/candidates/{candidate_id}/preview"
                    r = httpx.get(upstream, timeout=10.0, follow_redirects=True)
                    if r.status_code == 200:
                        return Response(
                            content=r.content,
                            media_type=r.headers.get("content-type", "image/jpeg"),
                            headers={"Cache-Control": "public, max-age=86400"},
                        )
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        db.close()

    # Fallback: colored placeholder with candidate_id label
    png = _placeholder_png(candidate_id)
    return Response(content=png, media_type="image/png", headers={
        "Cache-Control": "public, max-age=3600",
    })
