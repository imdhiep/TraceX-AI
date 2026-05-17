"""FastAPI application for query-service.

Responsibilities:
- Search candidates with hybrid ranking (text + vector)
- SeamlessM4T v2-large (Vietnamese ↔ English translation)
- Forward GPU re-ranking to trace-service (SigLIP 2 text tower)
- Forward shortlist to trace-service for EVA-02 cosine re-ranking
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

# ── Timestamp logging ──────────────────────────────────────────────────────────
# Uvicorn configures uvicorn.* loggers, but the root logger used by app modules
# can be left at WARNING with no handlers. Configure it before importing routers
# so query ranking logs from app.api.routers.candidates appear in docker logs.
_ts_fmt = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
if not logging.root.handlers:
    _ts_handler = logging.StreamHandler(sys.stdout)
    _ts_handler.setFormatter(_ts_fmt)
    logging.root.addHandler(_ts_handler)

_root_level_name = os.getenv("QUERY_SERVICE_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
if _root_level_name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    _root_level_name = "INFO"
logging.root.setLevel(getattr(logging, _root_level_name))
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, text

from .api.routers import candidates
from shared.database import SessionLocal
from shared.models import Camera, QueryCandidate, QueryHistory, User, Video

logger = logging.getLogger(__name__)


class TranslateToVietnameseRequest(BaseModel):
    texts: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Query service starting — model warmup...")

    # SeamlessM4T v2-large translation model
    try:
        from .services.translation import warmup as warmup_translation
        warmup_translation()
        logger.info("SeamlessM4T v2-large ready")
    except Exception as exc:
        logger.warning("SeamlessM4T warmup skipped (non-fatal): %s", exc)

    # Internal search runtime:
    # - RT-DETR crops uploaded person images
    # - PersonViT handles same-person image retrieval
    # - SigLIP handles text/image semantic retrieval
    try:
        from .services.model_warmup import warmup_models
        await warmup_models()
        logger.info("RT-DETR + PersonViT + SigLIP ready")
    except Exception as exc:
        logger.warning("Query model warmup skipped (non-fatal): %s", exc)

    logger.info("Query service warmup complete")
    yield
    logger.info("Query service shutting down...")


app = FastAPI(
    title="Query Service",
    description="Candidate search, ranking, and SeamlessM4T translation for TraceX-AI",
    version="2.0.0",
    lifespan=lifespan,
)

# Keep CORS permissive for operator/debug access, but the normal browser path is
# still frontend → metadata-service → query-service. Frontend auth uses Bearer
# headers rather than cookies, so credentials are not required here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates.router, prefix="/api/v1", tags=["candidates"])

# Expose the shared static assets referenced by candidate payloads for internal
# compatibility and operator/debug access.
_CROPS_DIR = Path("/workspace/storage/crops")
_CROPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/crops", StaticFiles(directory=str(_CROPS_DIR)), name="crops")

_QUERY_IMAGES_DIR = Path("/workspace/storage/query-images")
_QUERY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/query-images", StaticFiles(directory=str(_QUERY_IMAGES_DIR)), name="query-images")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "query-service", "version": "2.0.0"}


@app.post("/api/v1/translate/to-vietnamese")
def translate_to_vietnamese_batch(request: TranslateToVietnameseRequest):
    """Translate UI display strings to Vietnamese using query-service's model."""
    from .services.translation import translate_to_vietnamese_cached

    texts = [str(text or "") for text in request.texts[:1000]]
    return {"items": [translate_to_vietnamese_cached(text) for text in texts]}


# ── Runtime log-level toggle ──────────────────────────────────────────────
# Lets operators flip the candidates router between INFO and DEBUG without
# restarting the container. DEBUG enables the heavy per-tracklet payloads
# (prefilter_top_N, vector_top_N, merged_group_preview, top_detail) that
# are otherwise gated by isEnabledFor(DEBUG) in candidates.py.
#
#   curl -X POST http://<host>:8003/debug/log-level/DEBUG
#   curl -X POST http://<host>:8003/debug/log-level/INFO
#
# Scope is limited to the search router so other loggers (uvicorn, sqlalchemy)
# stay at their current level — DEBUG on those is overwhelming.
_TOGGLEABLE_LOGGERS = ("app.api.routers.candidates",)
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@app.post("/debug/log-level/{level}")
def set_log_level(level: str):
    level_upper = level.upper()
    if level_upper not in _VALID_LOG_LEVELS:
        return {
            "error": f"invalid level {level!r}",
            "valid": sorted(_VALID_LOG_LEVELS),
        }
    numeric = getattr(logging, level_upper)
    applied = []
    for name in _TOGGLEABLE_LOGGERS:
        logging.getLogger(name).setLevel(numeric)
        applied.append(name)
    logger.warning("Log level toggled to %s for: %s", level_upper, applied)
    return {"level": level_upper, "loggers": applied}


@app.get("/debug/log-level")
def get_log_level():
    return {
        name: logging.getLevelName(logging.getLogger(name).getEffectiveLevel())
        for name in _TOGGLEABLE_LOGGERS
    }


@app.post("/api/v1/overview")
@app.get("/api/v1/overview")
def overview():
    """Return system-wide metrics for the settings/overview page."""
    db = SessionLocal()
    try:
        total_users = db.query(func.count(User.id)).scalar() or 0
        total_cameras = db.query(func.count(Camera.camera_id)).scalar() or 0
        total_managed_videos = db.query(func.count(Video.video_id)).filter(Video.processed == True).scalar() or 0
        total_queries = db.query(func.count(QueryHistory.query_id)).scalar() or 0
        total_candidates = db.query(func.count(QueryCandidate.candidate_id)).scalar() or 0
        try:
            total_queue_videos = db.execute(text("SELECT COUNT(*) FROM queue_video_assets")).scalar() or 0
        except Exception:
            total_queue_videos = 0
        try:
            total_candidate_videos = db.execute(text("SELECT COUNT(*) FROM evidence_videos")).scalar() or 0
        except Exception:
            total_candidate_videos = 0
    finally:
        db.close()

    return {
        "metrics": {
            "total_users": total_users,
            "total_cameras": total_cameras,
            "total_managed_videos": total_managed_videos,
            "total_queries": total_queries,
            "total_candidates": total_candidates,
            "total_candidate_videos": total_candidate_videos,
            "total_queue_videos": total_queue_videos,
        }
    }


@app.get("/")
def root():
    return {
        "service": "query-service",
        "version": "2.0.0",
        "description": "Candidate search + SeamlessM4T v2 translation + GPU re-ranking",
        "translation_model": "SeamlessM4T v2-large",
        "ranking_service": "trace-service (SigLIP 2 text tower + EVA-02 cosine)",
    }
