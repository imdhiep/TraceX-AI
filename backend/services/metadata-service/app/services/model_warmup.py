"""GPU model warmup for metadata-service — SOTA 2026 AI pipeline.

Loads all models into VRAM once at startup via FastAPI lifespan.
VRAM budget (metadata-service, A100 80GB):
  - RT-DETR R50 (person detection):                    ~3GB fp16
  - DINOv2 ViT-L/14 (appearance embedding, 1024-dim):  ~5GB fp16
  - SigLIP 2-So400m (image encoder for text search):   ~3GB fp16
  - VideoMAE V2 (action recognition):                  ~3GB fp16
  - Qwen2.5-VL-7B-Instruct (open-vocabulary metadata): ~14GB fp16
  Runtime overhead (KV cache, activations):            ~3GB
  Total metadata-service:                              ~31GB / 80GB

Pre-download models: python scripts/download_models.py --models qwen25vl siglip2 videomae

Forbidden: YOLO (any version), ByteTrack, Grounding DINO (replaced by RT-DETR).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_warmup_done = False
_warmup_error: Optional[str] = None

_MODELS: dict[str, Any] = {}
DEFAULT_QWEN25VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_model(name: str) -> Optional[Any]:
    """Retrieve a loaded model by name."""
    return _MODELS.get(name)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / 1024**3
        logger.info("GPU: %s  VRAM: %.1f GB", gpu_name, gpu_mem_gb)
        return device
    logger.warning("CUDA unavailable — CPU fallback (very slow for production)")
    return torch.device("cpu")


async def warmup_models() -> None:
    """Load all SOTA 2026 models into GPU memory. Called once at FastAPI lifespan startup."""
    global _warmup_done, _warmup_error

    if _warmup_done:
        logger.info("GPU models already loaded")
        return

    logger.info("=== metadata-service SOTA 2026 warmup starting ===")
    device = get_device()

    if device.type == "cuda":
        # A100 Tensor Cores support TF32 — free ~10 % matmul speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        # cuDNN picks the fastest kernel for each fixed input shape
        torch.backends.cudnn.benchmark        = True
        # Prefer TF32 over FP32 for internal matmul precision
        torch.set_float32_matmul_precision("high")
        logger.info("A100 flags: TF32=on  cuDNN.benchmark=on  matmul_precision=high")

    _load_rtdetr(device)
    _load_dinov2(device)
    _load_siglip2(device)
    _load_videomae_v2(device)
    _load_qwen25vl(device)

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


def _load_dinov2(device: torch.device) -> None:
    """Load DINOv2 ViT-L/14 for 1024-dim appearance embeddings."""
    logger.info("Loading DINOv2 ViT-L/14...")
    try:
        from transformers import AutoImageProcessor, AutoModel
        import numpy as np
        from PIL import Image

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_id = "facebook/dinov2-large"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype)
        model = model.to(device)
        model.eval()

        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        inputs = processor(images=[dummy], return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=torch_dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            feat = model(**inputs).pooler_output  # [1, 1024]
        logger.info("  DINOv2 output dim: %d", feat.shape[-1])
        assert feat.shape[-1] == 1024, f"Expected 1024-dim, got {feat.shape[-1]}"

        _MODELS["dinov2"] = model
        _MODELS["dinov2_processor"] = processor
        logger.info("  DINOv2 ViT-L/14 loaded OK")

    except Exception as exc:
        logger.warning("DINOv2 load failed (non-fatal): %s", exc)



def _load_siglip2(device: torch.device) -> None:
    """Load SigLIP 2-So400m image encoder for text-image search embeddings (1152-dim)."""
    logger.info("Loading SigLIP 2-So400m...")
    try:
        from transformers import AutoProcessor, AutoModel

        siglip_torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_id = "google/siglip2-so400m-patch14-384"
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=siglip_torch_dtype)
        except Exception:
            model_id = "google/siglip-so400m-patch14-384"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=siglip_torch_dtype)

        model = model.to(device)
        model.eval()

        import numpy as np
        from PIL import Image
        siglip_dtype = torch.float16 if device.type == "cuda" else torch.float32
        dummy_img = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        labels = ["person in red shirt", "person in blue jeans"]
        inputs = processor(text=labels, images=dummy_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device=device, dtype=siglip_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

        _MODELS["siglip2"] = model
        _MODELS["siglip2_processor"] = processor
        logger.info("  SigLIP 2 loaded OK (model: %s)", model_id)

    except Exception as exc:
        logger.warning("SigLIP 2 load failed (non-fatal): %s", exc)


def _load_rtdetr(device: torch.device) -> None:
    """Load RT-DETR R50 for fast person detection (primary detector).

    Loads fine-tuned weights from /workspace/models/weights/rtdetr_person/ if available
    (host: ./storage/model-weights/rtdetr_person/), otherwise base PekingU/rtdetr_r50vd.
    Fine-tune with: python scripts/finetune_rtdetr.py
    """
    logger.info("Loading RT-DETR R50 (person detector)...")
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        from pathlib import Path

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        finetuned = Path("/workspace/models/weights/rtdetr_person")
        model_id = str(finetuned) if finetuned.exists() else "PekingU/rtdetr_r50vd"
        source = "fine-tuned" if finetuned.exists() else "base"

        processor = RTDetrImageProcessor.from_pretrained(model_id)
        model = RTDetrForObjectDetection.from_pretrained(model_id, torch_dtype=torch_dtype)
        model = model.to(device)
        model.eval()

        import numpy as np
        from PIL import Image
        dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        inputs = processor(images=[dummy], return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=torch_dtype) if v.is_floating_point() else v.to(device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logger.info("  RT-DETR logits dim: %d", out.logits.shape[-1])

        id2label = model.config.id2label
        person_ids = [k for k, v in id2label.items() if "person" in str(v).lower()]
        logger.info("  Person class IDs: %s", person_ids)

        _MODELS["rtdetr"] = model
        _MODELS["rtdetr_processor"] = processor
        _MODELS["rtdetr_person_ids"] = set(person_ids)
        logger.info("  RT-DETR R50 loaded OK (%s, model: %s)", source, model_id)

    except Exception as exc:
        logger.warning("RT-DETR load failed (non-fatal): %s", exc)


def _load_videomae_v2(device: torch.device) -> None:
    """Load VideoMAE V2 fine-tuned on Kinetics-400 for temporal action recognition."""
    logger.info("Loading VideoMAE V2 (Kinetics-400 fine-tuned)...")
    try:
        from transformers import AutoProcessor, AutoModelForVideoClassification
        import numpy as np

        videomae_torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_ids = [
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "MCG-NJU/videomae-small-finetuned-kinetics",
        ]
        loaded_model_id = None
        for model_id in model_ids:
            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVideoClassification.from_pretrained(model_id, torch_dtype=videomae_torch_dtype)
                loaded_model_id = model_id
                break
            except Exception:
                logger.warning("  Could not load %s, trying next...", model_id)
                continue

        if loaded_model_id is None:
            logger.warning("No VideoMAE Kinetics model available, skipping")
            return

        model = model.to(device)
        model.eval()

        videomae_dtype = torch.float16 if device.type == "cuda" else torch.float32
        dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        inputs = processor(dummy_frames, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=videomae_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logger.info("  VideoMAE V2 logits dim: %d", out.logits.shape[-1])

        _MODELS["videomae"] = model
        _MODELS["videomae_processor"] = processor
        _MODELS["videomae_model_id"] = loaded_model_id
        logger.info("  VideoMAE V2 loaded OK (model: %s)", loaded_model_id)

    except Exception as exc:
        logger.warning("VideoMAE V2 load failed (non-fatal): %s", exc)


def _load_qwen25vl(device: torch.device) -> None:
    """Load Qwen2.5-VL-7B-Instruct for open-vocabulary appearance captioning.

    Loads from /workspace/models/weights/qwen2_5_vl/ if pre-downloaded
    (host: ./storage/model-weights/qwen2_5_vl/), otherwise downloads from HuggingFace.
    QWEN25VL_MODEL_ID overrides both the local path and default HuggingFace repo.
    Pre-download with: python scripts/download_models.py --models qwen25vl
    """
    logger.info("Loading Qwen2.5-VL-7B-Instruct...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from pathlib import Path
        import numpy as np
        from PIL import Image

        local = Path("/workspace/models/weights/qwen2_5_vl")
        env_override = os.getenv("QWEN25VL_MODEL_ID", "").strip()
        local_available = local.exists() and any(local.iterdir())
        if env_override:
            model_id = env_override
            source = "env_override"
        elif local_available:
            model_id = str(local)
            source = "local"
        else:
            model_id = DEFAULT_QWEN25VL_MODEL_ID
            source = "HuggingFace"
        logger.info("  Qwen2.5-VL source: %s (%s)", model_id, source)

        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        dummy = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": dummy},
            {"type": "text", "text": "Describe this person briefly."},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy], return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=16)

        _MODELS["qwen25vl"] = model
        _MODELS["qwen25vl_processor"] = processor
        logger.info("  Qwen2.5-VL-7B-Instruct loaded OK (%s, model: %s)", source, model_id)

    except Exception as exc:
        logger.warning("Qwen2.5-VL-7B-Instruct load failed (non-fatal): %s", exc)


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error


def get_loaded_models() -> list[str]:
    return [k for k in _MODELS if not k.endswith(("_processor", "_transform", "_model_id", "_person_ids", "_classes"))]
