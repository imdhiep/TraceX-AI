"""GPU model warmup for query-service.

Loads the models needed for a fully independent search runtime:
- PersonViT-S MSMT17 for same-person image retrieval
- RT-DETR R50 for query-image person cropping
- SigLIP So400m text/image towers for semantic retrieval

Models stay resident in VRAM so search requests do not depend on
metadata-service at inference time.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_warmup_done = False
_warmup_error: Optional[str] = None

# Model registry — populated by the warmup helpers below.
_MODELS: dict[str, Any] = {}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("Using GPU: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
        return device
    logger.warning("CUDA not available, using CPU (slow inference)")
    return torch.device("cpu")


def get_model(key: str):
    """Lookup a warmed-up model. Returns None if not loaded."""
    return _MODELS.get(key)


async def warmup_models():
    global _warmup_done, _warmup_error

    if _warmup_done:
        logger.info("Models already warmed up")
        return

    logger.info("=== Starting query-service model warmup ===")
    device = get_device()

    try:
        _warmup_rtdetr(device)
    except Exception as exc:
        logger.warning("RT-DETR warmup skipped; full-image query fallback will be used: %s", exc)

    try:
        _warmup_personvit(device)
    except Exception as exc:
        logger.warning("PersonViT warmup skipped; image search will be unavailable: %s", exc)

    try:
        _warmup_siglip2(device)
    except Exception as e:
        _warmup_error = str(e)
        logger.exception("SigLIP warmup failed: %s", e)
        raise

    _warmup_done = True
    logger.info("=== Model warmup COMPLETE ===")


def _warmup_siglip2(device: torch.device) -> None:
    """Load SigLIP 1 So400m to match the metadata-service ingest pipeline.

    The function is still named `_warmup_siglip2` and the cache keys are
    still `siglip2*` because every downstream caller (candidates.py,
    candidate_query.py) uses those names. The model_id is what matters:
    `google/siglip-so400m-patch14-384` is the same checkpoint that
    metadata-service ends up loading after its own SigLIP 2 fallback, so
    query embeddings live in the SAME 1152-dim space as
    `tracklets_embeddings.siglip_embedding`.

    Switching to genuine SigLIP 2 here would require re-ingesting every
    video to refresh the stored embeddings — not worth it given SigLIP 2's
    marginal gain on text-vs-person-crop retrieval.
    """
    logger.info("Loading SigLIP 1-So400m (text + image)...")
    try:
        from transformers import AutoModel, AutoProcessor

        model_id = "google/siglip-so400m-patch14-384"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device).eval()
        logger.info("  SigLIP loaded OK (%s)", model_id)

        _MODELS["siglip2"] = model
        _MODELS["siglip2_processor"] = processor
        _MODELS["siglip2_model_id"] = model_id
    except Exception as e:
        logger.exception("SigLIP warmup failed: %s", e)
        raise


def _warmup_personvit(device: torch.device) -> None:
    """Load the same PersonViT checkpoint used by metadata-service ingest."""
    logger.info("Loading PersonViT-S MSMT17...")
    try:
        from transformers import AutoModel

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_id = "maennyn/personvit-reid-msmt17-vit-s"
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device).eval()

        dummy = torch.zeros(1, 3, 256, 128, device=device, dtype=torch_dtype)
        with torch.no_grad():
            out = model(dummy)
        feat = out[0] if isinstance(out, tuple) else out.embeddings
        assert feat.shape[-1] == 384, f"Expected 384-dim, got {feat.shape[-1]}"

        _MODELS["personvit"] = model
        _MODELS["personvit_model_id"] = model_id
        logger.info("  PersonViT-S MSMT17 loaded OK")
    except Exception as exc:
        logger.exception("PersonViT warmup failed: %s", exc)
        raise


def _warmup_rtdetr(device: torch.device) -> None:
    """Load RT-DETR for single-person query-image cropping."""
    logger.info("Loading RT-DETR R50 (query-image person detector)...")
    try:
        from pathlib import Path
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        finetuned = Path("/workspace/models/weights/rtdetr_person")
        model_id = str(finetuned) if finetuned.exists() else "PekingU/rtdetr_r50vd"
        source = "fine-tuned" if finetuned.exists() else "base"

        processor = RTDetrImageProcessor.from_pretrained(model_id)
        model = RTDetrForObjectDetection.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device).eval()

        id2label = model.config.id2label
        person_ids = {k for k, v in id2label.items() if "person" in str(v).lower()}
        _MODELS["rtdetr"] = model
        _MODELS["rtdetr_processor"] = processor
        _MODELS["rtdetr_person_ids"] = person_ids or {0, 1}
        _MODELS["rtdetr_model_id"] = model_id
        logger.info("  RT-DETR R50 loaded OK (%s, model: %s)", source, model_id)
    except Exception as exc:
        logger.exception("RT-DETR warmup failed: %s", exc)
        raise


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error
