"""GPU model warmup for query-service.

Loads SigLIP 2-So400m text tower at startup for text→embedding encoding.
VRAM budget (query-service, A100 80GB):
  - SigLIP 2-So400m text tower: ~3GB fp16
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

    logger.info("=== query-service warmup starting ===")
    device = get_device()

    _load_siglip2(device)

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


def _load_siglip2(device: torch.device) -> None:
    """Load SigLIP 2-So400m for text→embedding encoding (text tower)."""
    logger.info("Loading SigLIP 2-So400m text tower...")
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

        # Warmup: encode a dummy text sample
        dummy_texts = ["person walking", "person running"]
        inputs = processor(text=dummy_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model.get_text_features(**{k: v for k, v in inputs.items()
                                          if k in ["input_ids", "attention_mask"]})

        _MODELS["siglip2"] = model
        _MODELS["siglip2_processor"] = processor
        logger.info("  SigLIP 2 text tower loaded OK (model: %s)", model_id)

    except Exception as exc:
        logger.warning("SigLIP 2 load failed: %s", exc)


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error
