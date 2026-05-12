"""FastAPI application for trace-service — Neural Video Reconstruction Engine.

Responsibilities:
  - Neural Video Reconstruction: Real-ESRGAN SR + ProPainter + RIFE + NVENC
  - Trace building: select candidate, build evidence segments
  - FIFO cache per user: /workspace/storage/cache/{user_id}/{query_id}/
  - Forward video processing to metadata-service (GPU AI models)

GPU models have been moved to metadata-service.
This service focuses on video enhancement + trace logic.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routers import candidates, trace, video_process, ingestion

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # GPU warmup is now done in metadata-service
    # This service focuses on Neural Video Reconstruction
    logger.info("Trace service starting — Neural Video Reconstruction Engine ready")
    yield
    logger.info("Trace service shutting down...")


app = FastAPI(
    title="Trace Service",
    description="Neural Video Reconstruction + Trace building for TraceX-AI",
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

app.include_router(trace.router, prefix="/api/v1/trace", tags=["trace"])
app.include_router(candidates.router, tags=["candidates"])
app.include_router(video_process.router, tags=["video_process"])
app.include_router(ingestion.router, tags=["ingestion"])

_TRACES_DIR = Path("/workspace/storage/traces")
_TRACES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/traces", StaticFiles(directory=str(_TRACES_DIR)), name="traces")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "trace-service", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "service": "trace-service",
        "version": "2.0.0",
        "description": "Neural Video Reconstruction + Trace building",
        "neural_pipeline": ["Real-ESRGAN", "ProPainter", "RIFE", "NVENC"],
    }
