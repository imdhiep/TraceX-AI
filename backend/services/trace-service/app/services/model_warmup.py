"""GPU model warmup for trace-service.

Loads SigLIP 2 and VideoMAE V2 into GPU memory at startup via FastAPI lifespan.
VRAM budget (trace-service, A100 80GB):
  - SigLIP 2-So400m (image encoder for text-image search): ~3GB fp16
  - VideoMAE V2-Large (action recognition):               ~3GB fp16
  Total trace-service:                                     ~6GB / 80GB
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_warmup_done = False
_warmup_error: Optional[str] = None

_MODELS: dict[str, Any] = {}


def get_model(name: str) -> Optional[Any]:
    return _MODELS.get(name)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / 1024**3
        logger.info("GPU: %s  VRAM: %.1f GB", gpu_name, gpu_mem_gb)
        return device
    logger.warning("CUDA unavailable — CPU fallback")
    return torch.device("cpu")


async def warmup_models() -> None:
    global _warmup_done, _warmup_error

    if _warmup_done:
        logger.info("Models already loaded")
        return

    logger.info("=== trace-service warmup starting ===")
    device = get_device()

    _load_siglip2(device)
    _load_videomae_v2(device)

    _warmup_done = True
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(
            "=== Warmup complete — VRAM allocated: %.1f GB / reserved: %.1f GB ===",
            allocated, reserved,
        )
    else:
        logger.info("=== Warmup complete (CPU) ===")


# ---------------------------------------------------------------------------
# SigLIP 2-So400m — text-image search embeddings (1152-dim)
# ---------------------------------------------------------------------------

def _load_siglip2(device: torch.device) -> None:
    """Load SigLIP 2-So400m image encoder for text-image search embeddings."""
    logger.info("Loading SigLIP 2-So400m...")
    try:
        from transformers import AutoProcessor, AutoModel

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_id = "google/siglip2-so400m-patch14-384"
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        except Exception:
            model_id = "google/siglip-so400m-patch14-384"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        model.eval()

        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        labels = ["person in red shirt", "person in blue jeans"]
        inputs = processor(text=labels, images=dummy_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

        _MODELS["siglip2"] = model
        _MODELS["siglip2_processor"] = processor
        logger.info("  SigLIP 2 loaded OK (model: %s)", model_id)

    except Exception as exc:
        logger.warning("SigLIP 2 load failed: %s", exc)


# ---------------------------------------------------------------------------
# VideoMAE V2-Large — action recognition
# ---------------------------------------------------------------------------

def _load_videomae_v2(device: torch.device) -> None:
    """Load VideoMAE V2 for temporal action recognition."""
    logger.info("Loading VideoMAE V2...")
    try:
        from transformers import AutoProcessor, AutoModelForVideoClassification
        import numpy as np

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_ids = [
            "MCG-NJU/videomae-large-finetuned-kinetics",
            "MCG-NJU/videomae-base-finetuned-kinetics",
        ]
        loaded_model_id = None
        for model_id in model_ids:
            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVideoClassification.from_pretrained(
                    model_id, torch_dtype=dtype,
                ).to(device)
                loaded_model_id = model_id
                break
            except Exception:
                logger.warning("  Could not load %s, trying next...", model_id)
                continue

        if loaded_model_id is None:
            logger.warning("No VideoMAE model available, skipping")
            return

        model.eval()

        dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        inputs = processor(dummy_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logger.info("  VideoMAE V2 logits dim: %d", out.logits.shape[-1])

        _MODELS["videomae"] = model
        _MODELS["videomae_processor"] = processor
        logger.info("  VideoMAE V2 loaded OK (model: %s)", loaded_model_id)

    except Exception as exc:
        logger.warning("VideoMAE V2 load failed: %s", exc)


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error


def get_loaded_models() -> list[str]:
    return [k for k in _MODELS if not k.endswith(("_processor", "_model_id"))]
