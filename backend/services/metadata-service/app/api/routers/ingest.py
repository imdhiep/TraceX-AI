"""
ingest.py — API endpoints for Drive Temp→Storage move + LightningAI ingest.

POST /api/v1/ingest/move-and-process   — trigger ingest only
POST /api/v1/ingest/full-pipeline      — finetune trên MTMC GT rồi ingest
GET  /api/v1/ingest/stats              — quick counts
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database import get_session
from ...services.ingest_service import ingest_move_and_process

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    action: str = Field(default="move_and_ingest", description="Action to perform")
    dry_run: bool = Field(default=False, description="If true, list files without moving or processing")


class IngestResponse(BaseModel):
    status: str
    moved_count: int
    ingested_count: int
    skipped_count: int
    total_videos_in_db: int
    total_tracklets_saved: int = 0
    processed_videos: int = 0
    errors: list[str]
    message: str
    finetune: dict | None = None


class IngestStatsResponse(BaseModel):
    total_videos_in_db: int
    total_cameras_in_db: int
    total_tracklets_in_db: int
    total_embeddings_in_db: int


@router.post("/move-and-process", response_model=IngestResponse)
def move_and_process(
    body: IngestRequest,
    session: Session = Depends(get_session),
) -> IngestResponse:
    """
    Trigger the full ingest pipeline:

    1. Move videos from Google Drive Temp/ folder to Storage/ folder
    2. Scan Storage/ for .mp4 files not yet in DB
    3. Submit each to LightningAI GPU pipeline
    4. Save results to v3.3 normalized tables (videos, tracklets, embeddings, actions)

    Use `dry_run: true` to preview what would be moved/ingested without side effects.
    """
    try:
        result = ingest_move_and_process(
            session=session,
            dry_run=body.dry_run,
        )
        return IngestResponse(**result)
    except Exception as exc:
        logger.exception("Ingest pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        session.close()


@router.post("/full-pipeline", response_model=IngestResponse)
async def full_pipeline(
    body: IngestRequest,
    session: Session = Depends(get_session),
) -> IngestResponse:
    """
    Full pipeline với MTMC Ground Truth fine-tuning:

    Phase 1 — Fine-tune models trên MTMC GT data:
      - Auto-detect tất cả scenes có sẵn trong /workspace/storage/dataset/MTMC_Tracking_2024/train/
      - Chạy MTMCFineTuningPipeline cho từng scene (num_samples=100 để nhanh)
      - Cải thiện độ chính xác RT-DETR + SigLIP 2 cho môi trường VinUni

    Phase 2 — Ingest 50 video:
      - Move Temp → Storage (Google Drive)
      - Process từng video với GPU models đã fine-tuned
      - Save tracklets/embeddings vào DB
    """
    from pathlib import Path
    from ...services.ground_truth_finetune import auto_detect_scenes, SCENE_CONFIG, MTMCFineTuningPipeline

    DATASET_ROOT = Path("/workspace/storage/dataset/MTMC_Tracking_2024/train")

    # ── Phase 1: Fine-tune trên tất cả scenes MTMC GT ─────────────────────────
    finetune_summary: dict = {"scenes_processed": 0, "scenes_failed": 0, "errors": []}

    if DATASET_ROOT.exists():
        logger.info("[full-pipeline] Phase 1: Auto-detecting MTMC scenes in %s", DATASET_ROOT)
        auto_detect_scenes(DATASET_ROOT)
        scenes = sorted(SCENE_CONFIG.keys())
        logger.info("[full-pipeline] Found %d scenes: %s", len(scenes), scenes)

        for scene_name in scenes:
            try:
                logger.info("[full-pipeline] Fine-tuning %s...", scene_name)
                pipeline = MTMCFineTuningPipeline(
                    scene_name=scene_name,
                    dataset_root=str(DATASET_ROOT),
                    batch_size=8,
                )
                metrics = await pipeline.run(
                    num_samples=100,
                    models_to_finetune=["rtdetr", "dinov2"],
                )
                finetune_summary["scenes_processed"] += 1
                logger.info(
                    "[full-pipeline] ✓ %s: iou=%.3f f1=%.3f reid_acc=%.3f",
                    scene_name,
                    metrics.avg_detection_iou,
                    metrics.detection_f1,
                    metrics.reid_accuracy,
                )
            except Exception as exc:
                finetune_summary["scenes_failed"] += 1
                finetune_summary["errors"].append(f"{scene_name}: {exc}")
                logger.warning("[full-pipeline] Fine-tune failed for %s: %s", scene_name, exc)
    else:
        logger.warning("[full-pipeline] DATASET_ROOT not found: %s — skipping fine-tuning", DATASET_ROOT)
        finetune_summary["errors"].append(f"Dataset not found: {DATASET_ROOT}")

    # ── Phase 2: Ingest videos ─────────────────────────────────────────────────
    logger.info("[full-pipeline] Phase 2: Starting ingest pipeline")
    try:
        result = ingest_move_and_process(session=session, dry_run=body.dry_run)
        result["finetune"] = finetune_summary
        return IngestResponse(**result)
    except Exception as exc:
        logger.exception("Ingest pipeline failed after fine-tuning")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/stats", response_model=IngestStatsResponse)
async def get_stats(
    session: Session = Depends(get_session),
) -> IngestStatsResponse:
    """Return quick counts from the DB."""
    try:
        from shared.models import Video, Camera, Tracklet, TrackletEmbedding
        from sqlalchemy import func

        total_videos = int(session.scalar(func.count(Video.id)) or 0)
        total_cameras = int(session.scalar(func.count(Camera.id)) or 0)
        total_tracklets = int(session.scalar(func.count(Tracklet.id)) or 0)
        total_embeddings = int(session.scalar(func.count(TrackletEmbedding.id)) or 0)

        return IngestStatsResponse(
            total_videos_in_db=total_videos,
            total_cameras_in_db=total_cameras,
            total_tracklets_in_db=total_tracklets,
            total_embeddings_in_db=total_embeddings,
        )
    except Exception as exc:
        logger.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        session.close()
