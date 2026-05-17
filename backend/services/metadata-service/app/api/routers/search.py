"""Search router — forwards to query-service (GPU) for ranking.

Frontend calls POST /api/v1/search → metadata-service → query-service (GPU).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from ...core.dependencies import get_current_user
from shared.models import User


class SearchRequest(BaseModel):
    query: str | None = None
    text: str | None = None  # alias for query (FE may send either)
    top_k: int = 20
    offset: int = 0
    camera_ids: list[str] | None = None
    time_from: str | None = None
    time_to: str | None = None
    query_image_url: str | None = None  # URL of uploaded query image (for history display)
    query_image_path: str | None = None  # local shared-storage path for query-service image tower
    query_id: str | None = None  # passed back by FE during pagination so the qh row is reused

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

_QUERY_SERVICE_URL: str | None = None
_QUERY_IMAGE_ROOT = Path(os.getenv("QUERY_IMAGE_ROOT", "/workspace/storage/query-images"))
_MAX_QUERY_IMAGE_BYTES = int(os.getenv("MAX_QUERY_IMAGE_BYTES", str(5 * 1024 * 1024)))
_IMAGE_EXT_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _get_query_service_url() -> str:
    global _QUERY_SERVICE_URL
    if _QUERY_SERVICE_URL is None:
        import os
        _QUERY_SERVICE_URL = os.getenv(
            "QUERY_SERVICE_URL",
            "http://query-service:8003"
        )
    return _QUERY_SERVICE_URL


def _post_to_query_service(
    path: str,
    payload: dict[str, Any],
    timeout: float = 120.0,
    authorization: str | None = None,
) -> dict[str, Any]:
    url = f"{_get_query_service_url().rstrip('/')}/api/v1{path}"
    max_attempts = 3
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            headers = {"Authorization": authorization} if authorization else None
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < max_attempts:
                import time
                time.sleep(0.5 * attempt)
        except Exception as exc:
            last_exc = exc
            break
        import time
        time.sleep(0.5)

    raise HTTPException(
        status_code=503,
        detail=f"Query service unavailable: {last_exc}",
    )


def _form_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _query_image_extension(filename: str | None, content_type: str | None) -> str:
    if content_type:
        ext = _IMAGE_EXT_BY_TYPE.get(content_type.lower())
        if ext:
            return ext
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


async def _save_query_image(upload: StarletteUploadFile) -> tuple[str, str]:
    content_type = (upload.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh cho query_image.")

    content = await upload.read(_MAX_QUERY_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Ảnh truy vấn đang trống.")
    if len(content) > _MAX_QUERY_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh truy vấn không được vượt quá 5MB.")

    _QUERY_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    ext = _query_image_extension(upload.filename, content_type)
    filename = f"{uuid.uuid4().hex}{ext}"
    path = _QUERY_IMAGE_ROOT / filename
    path.write_bytes(content)
    return f"/static/query-images/{filename}", str(path)


async def _parse_search_request(request: Request) -> SearchRequest:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        camera_ids = [
            str(value).strip()
            for value in form.getlist("camera_ids")
            if str(value).strip()
        ]
        body = SearchRequest(
            query=str(form.get("query") or ""),
            text=str(form.get("text") or "") or None,
            top_k=_form_int(form.get("top_k"), 20),
            offset=_form_int(form.get("offset"), 0),
            camera_ids=camera_ids or None,
            time_from=str(form.get("time_from") or "") or None,
            time_to=str(form.get("time_to") or "") or None,
            query_id=str(form.get("query_id") or "") or None,
        )
        query_image = form.get("query_image")
        if isinstance(query_image, StarletteUploadFile) and query_image.filename:
            image_url, image_path = await _save_query_image(query_image)
            body.query_image_url = image_url
            body.query_image_path = image_path
        return body

    try:
        raw_body = await request.json()
    except Exception:
        raw_body = {}
    return SearchRequest(**(raw_body or {}))


@router.post("")
async def search_candidates(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Search candidates — forwarded to query-service for GPU ranking.
    Accepts JSON body with: query (or text), top_k, offset, camera_ids, time_from, time_to.
    """
    body = await _parse_search_request(request)
    query_text = body.query or body.text or ""
    payload: dict[str, Any] = {
        "query": query_text,
        "top_k": body.top_k,
        "offset": body.offset,
        "user_id": current_user.id,
    }
    if body.camera_ids:
        payload["camera_ids"] = body.camera_ids
    if body.time_from:
        payload["time_from"] = body.time_from
    if body.time_to:
        payload["time_to"] = body.time_to
    if body.query_image_url:
        payload["query_image_url"] = body.query_image_url
    if body.query_image_path:
        payload["query_image_path"] = body.query_image_path
    if body.query_id:
        payload["query_id"] = body.query_id

    try:
        result = _post_to_query_service(
            "/search",
            payload,
            authorization=request.headers.get("authorization"),
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.get("/overview")
def search_overview() -> dict[str, Any]:
    """Get overview statistics — forwarded to query-service."""
    try:
        return _post_to_query_service("/overview", {}, timeout=30.0)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Overview failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Overview failed: {exc}")
