"""FastAPI application for query-service.

Responsibilities:
- Search candidates with hybrid ranking (text + vector)
- SeamlessM4T v2-large (Vietnamese ↔ English translation)
- SigLIP 2-So400m text tower: encode query text → 1152-dim embedding → cosine vs stored siglip_embedding
- Union-find identity merge (SigLIP2 cosine + metadata + temporal/camera guards)
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text

from .api.routers import candidates
from shared.database import SessionLocal
from shared.models import Camera, QueryCandidate, QueryHistory, User, Video

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Query service starting — GPU model warmup...")

    # SigLIP 2 text tower (text→embedding encoding)
    try:
        from .services.model_warmup import warmup_models
        await warmup_models()
        logger.info("SigLIP 2 text tower ready")
    except Exception as exc:
        logger.warning("SigLIP 2 warmup skipped (non-fatal): %s", exc)

    # SeamlessM4T v2-large translation model
    try:
        from .services.translation import warmup as warmup_translation
        warmup_translation()
        logger.info("SeamlessM4T v2-large ready")
    except Exception as exc:
        logger.warning("SeamlessM4T warmup skipped (non-fatal): %s", exc)

    logger.info("Query service warmup complete")
    yield
    logger.info("Query service shutting down...")


app = FastAPI(
    title="Query Service",
    description="Candidate search, ranking, and SeamlessM4T translation for TraceX-AI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates.router, prefix="/api/v1", tags=["candidates"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "query-service", "version": "2.0.0"}


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
        "description": "Candidate search + SeamlessM4T v2 translation + SigLIP 2 text-image search",
        "translation_model": "SeamlessM4T v2-large",
        "ranking_model": "SigLIP 2-So400m text tower (query→1152-dim) vs stored siglip_embedding",
    }
