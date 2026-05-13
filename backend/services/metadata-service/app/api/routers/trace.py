"""Trace router — thin proxy that forwards to trace-service.

Frontend always talks to metadata-service. Trace endpoints (candidate detail,
select, build, status, timeline) live on trace-service; this router relays
them so the frontend doesn't need a separate base URL.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Request

from ...core.dependencies import get_current_user
from shared.models import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trace"])

_TRACE_SERVICE_URL: str | None = None


def _get_trace_service_url() -> str:
    global _TRACE_SERVICE_URL
    if _TRACE_SERVICE_URL is None:
        _TRACE_SERVICE_URL = os.getenv("TRACE_SERVICE_URL", "http://trace-service:8004")
    return _TRACE_SERVICE_URL


def _forward(method: str, path: str, *, json: Any = None, timeout: float = 1200.0) -> Any:
    url = f"{_get_trace_service_url().rstrip('/')}/api/v1{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, json=json)
    except httpx.HTTPError as exc:
        logger.exception("trace-service request failed: %s %s — %s", method, path, exc)
        raise HTTPException(status_code=503, detail=f"Trace service unavailable: {exc}")

    if response.status_code >= 400:
        try:
            payload = response.json()
        except Exception:
            payload = {"detail": response.text}
        raise HTTPException(status_code=response.status_code, detail=payload.get("detail", payload))
    if not response.content:
        return None
    return response.json()


@router.post("/candidate-detail")
def candidate_detail(
    body: dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Return full candidate detail (all tracklets + appearance + actions + embedding meta)."""
    return _forward("POST", "/trace/candidate-detail", json=body, timeout=60.0)


@router.post("/select")
def select_candidate(
    body: dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> Any:
    return _forward("POST", "/trace/select", json=body, timeout=60.0)


@router.post("/build")
def build_trace(
    body: dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
) -> Any:
    # Trace build can take a while when source videos must be downloaded from
    # Drive on a cold cache, so the timeout here matches the upstream default.
    return _forward("POST", "/trace/build", json=body, timeout=1800.0)


@router.get("/status/{evidence_id}")
def trace_status(
    evidence_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    return _forward("GET", f"/trace/status/{evidence_id}", timeout=60.0)


@router.get("/timeline/{evidence_id}")
def trace_timeline(
    evidence_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    return _forward("GET", f"/trace/timeline/{evidence_id}", timeout=60.0)
