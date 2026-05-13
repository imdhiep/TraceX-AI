"""GPU model warmup for query-service.

Loads SigLIP 2-So400m (text + image towers) on startup. Models stay resident
in VRAM so text/image queries can be encoded online during search requests.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_warmup_done = False
_warmup_error: Optional[str] = None

# Model registry — populated by _warmup_siglip2.
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
        _warmup_siglip2(device)
        _warmup_done = True
        logger.info("=== Model warmup COMPLETE ===")
    except Exception as e:
        _warmup_error = str(e)
        logger.exception("Model warmup failed: %s", e)
        raise


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


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error
