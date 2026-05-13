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


class _Siglip2Processor:
    """Minimal processor that mimics the HF SiglipProcessor interface for
    SigLIP 2 checkpoints. We need this because transformers 4.49 ships only
    `SiglipProcessor`, which rejects the Gemma-based tokenizer that SigLIP 2
    stores. Constructing the pieces directly (GemmaTokenizerFast + the v1
    SiglipImageProcessor) keeps the call sites in candidates.py unchanged
    while still loading the genuine SigLIP 2 weights for `get_text_features`
    / `get_image_features`.
    """

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text=None, images=None, **kwargs):
        out: dict = {}
        if text is not None:
            tok_kwargs = {k: v for k, v in kwargs.items()
                          if k in {"padding", "truncation", "max_length", "return_tensors"}}
            tok_kwargs.setdefault("padding", "max_length")
            tok_kwargs.setdefault("truncation", True)
            tok_kwargs.setdefault("return_tensors", "pt")
            tok_out = self.tokenizer(text, **tok_kwargs)
            out.update(dict(tok_out))
        if images is not None:
            img_kwargs = {k: v for k, v in kwargs.items() if k in {"return_tensors"}}
            img_kwargs.setdefault("return_tensors", "pt")
            img_out = self.image_processor(images=images, **img_kwargs)
            out.update(dict(img_out))
        return out


def _warmup_siglip2(device: torch.device) -> None:
    """Load SigLIP 2-So400m. Same checkpoint as metadata-service's ingest
    pipeline so the query embedding lives in the SAME space as
    `tracklets_embeddings.siglip_embedding`."""
    logger.info("Loading SigLIP 2-So400m (text + image)...")
    try:
        from transformers import (
            AutoModel,
            AutoProcessor,
            GemmaTokenizerFast,
            SiglipImageProcessor,
        )

        model_id = "google/siglip2-so400m-patch14-384"
        try:
            # SigLIP 2 weights load fine through AutoModel on transformers 4.49,
            # but AutoProcessor crashes because it tries to pair the Gemma
            # tokenizer with the legacy SiglipTokenizer wrapper. Construct the
            # processor ourselves: GemmaTokenizerFast + SiglipImageProcessor.
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ).to(device).eval()
            tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
            image_processor = SiglipImageProcessor.from_pretrained(model_id)
            processor = _Siglip2Processor(tokenizer, image_processor)
            logger.info("  SigLIP 2 loaded OK (%s, gemma-tokenizer wrapper)", model_id)
        except Exception as primary_exc:
            # Fallback to the v1 checkpoint if siglip2 isn't available offline
            logger.warning("  siglip2 load failed (%s) — falling back to siglip", primary_exc)
            model_id = "google/siglip-so400m-patch14-384"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ).to(device).eval()
            logger.info("  SigLIP 1 fallback loaded (%s)", model_id)

        _MODELS["siglip2"] = model
        _MODELS["siglip2_processor"] = processor
        _MODELS["siglip2_model_id"] = model_id
    except Exception as e:
        logger.exception("SigLIP 2 warmup failed: %s", e)
        raise


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error
