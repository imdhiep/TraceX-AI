"""FastAPI application for metadata-service.

Handles queue management, authentication, video metadata, and AI video processing.
GPU models (RT-DETR R50, PersonViT-S MSMT17, SigLIP 2, VideoMAE V2, Qwen2.5-VL-7B) are
loaded at startup and used for the /api/v1/video/process endpoint.

Search/ranking is forwarded to query-service (GPU).
Trace building is forwarded to trace-service (Neural Video Reconstruction).
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── Timestamp logging ──────────────────────────────────────────────────────────
# uvicorn's dictConfig only configures uvicorn.* loggers, leaving root with no
# handlers. Add one here (at import time, after uvicorn dictConfig) so all app
# loggers get timestamps.
_ts_fmt = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_ts_handler = logging.StreamHandler(sys.stdout)
_ts_handler.setFormatter(_ts_fmt)
logging.root.addHandler(_ts_handler)
logging.root.setLevel(logging.INFO)
# ──────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Set HF token before any model downloads (non-fatal if missing)
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from .api.routers import auth, search, users, videos, ingest, history
from .api.routers.candidates import router as candidates_router
from .api.routers.finetune import router as finetune_router
from .api.routers.trace import router as trace_router
from .api.routers.video_process import router as video_process_router
from .config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Import here to avoid circular imports
    from .database import Base, SessionLocal, engine
    from .services.model_warmup import warmup_models
    from .services.user_service import ensure_bootstrap_admin

    # 1. Create all tables (idempotent)
    # Import shared Base which has all models registered
    from shared.models import Base as SharedBase
    logger.info("Creating database tables if they don't exist...")
    SharedBase.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # 2. Bootstrap admin user
    if settings.bootstrap_admin_email and settings.bootstrap_admin_password:
        logger.info("Ensuring bootstrap admin user: %s", settings.bootstrap_admin_email)
        session = SessionLocal()
        try:
            ensure_bootstrap_admin(
                session,
                email=settings.bootstrap_admin_email,
                password=settings.bootstrap_admin_password,
                full_name=settings.bootstrap_admin_full_name,
            )
            logger.info("Bootstrap admin user ready.")
        except Exception as exc:
            logger.error("Failed to bootstrap admin user: %s", exc)
        finally:
            session.close()
    else:
        logger.warning("BOOTSTRAP_ADMIN_EMAIL or BOOTSTRAP_ADMIN_PASSWORD not set — skipping admin bootstrap.")

    # 3. Warmup GPU models (RT-DETR, PersonViT-S, SigLIP 2 image encoder, VideoMAE V2, Qwen2.5-VL-7B)
    logger.info("Starting GPU model warmup...")
    await warmup_models()
    logger.info("GPU models ready.")

    yield

    logger.info("Shutting down metadata-service...")


app = FastAPI(
    title="Metadata Service",
    description="Queue management, authentication, and video metadata for TraceX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tracex-ai.smartnovi.tech",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_origin_regex=r"https://.*\.cloudspaces\.litng\.ai",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(candidates_router, prefix="/api/v1/candidates", tags=["candidates"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(video_process_router, prefix="/api/v1/video", tags=["video"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(finetune_router, prefix="/api/v1/finetune", tags=["finetune"])
app.include_router(history.router, prefix="/api/v1/history", tags=["history"])
app.include_router(trace_router, prefix="/api/v1/trace", tags=["trace"])
# /api/v1/admin/users/{user_id}/queries — admin view of per-user query history
app.include_router(users.router, prefix="/api/v1/admin/users", tags=["admin"])

# Serve crop/preview images saved during video processing
_CROPS_DIR = Path("/workspace/storage/crops")
_CROPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/crops", StaticFiles(directory=str(_CROPS_DIR)), name="crops")

# Serve uploaded query images so history can render the original visual prompt.
_QUERY_IMAGES_DIR = Path("/workspace/storage/query-images")
_QUERY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/query-images", StaticFiles(directory=str(_QUERY_IMAGES_DIR)), name="query-images")

# Serve evidence clips written by trace-service (same shared volume).
_TRACES_DIR = Path("/workspace/storage/traces")
_TRACES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/traces", StaticFiles(directory=str(_TRACES_DIR)), name="traces")

@app.get("/health")
async def health_check():
    from .services.model_warmup import get_loaded_models, is_warmup_done
    return {
        "status": "healthy",
        "service": "metadata-service",
        "gpu_warmup_done": is_warmup_done(),
        "loaded_models": get_loaded_models(),
    }


@app.get("/")
def root():
    return {
        "service": "metadata-service",
        "version": "2.0.0",
        "description": "Queue management, authentication, video metadata + SOTA AI processing",
        "gpu_models": ["RT-DETR R50", "PersonViT-S MSMT17", "SigLIP 2 (image encoder)", "VideoMAE V2", "Qwen2.5-VL-7B-Instruct"],
    }


@app.get("/debug/routes")
def debug_routes():
    """List all registered routes for debugging."""
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return {"routes": routes, "total": len(routes)}


@app.get("/debug/config")
def debug_config():
    """Show current runtime config for debugging."""
    from .config import settings
    return {
        "bootstrap_admin_email": settings.bootstrap_admin_email or "(not set)",
        "bootstrap_admin_password_set": bool(settings.bootstrap_admin_password),
        "postgres_host": settings.postgres_host or "(not set)",
        "postgres_db": settings.postgres_db or "(not set)",
    }
