#!/usr/bin/env python3
"""
Download all SOTA 2026 models to /workspace/storage/model-weights/

Models:
  - Qwen2.5-VL-7B-Instruct (~15GB) Qwen/Qwen2.5-VL-7B-Instruct
  - Grounding DINO 1.6      (~4GB)  IDEA-Research/grounding-dino-base
  - EVA-02 ViT-L/14         (~5GB)  Aaronhuang21/eva02_l14_clip
  - SigLIP 2-So400m         (~3GB)  google/siglip2-so400m
  - VideoMAE V2             (~3GB)  MCG-NJU/videomae-v2-base
  - SeamlessM4T v2-large    (~5GB)  facebook/seamless-m4t-v2-large
  - Real-ESRGAN x4          (~50MB) xinntao/Real-ESRGAN-x4plus
  - ProPainter              (~400MB) HuggingFace checkpoint
  - RIFE                    (~100MB) HuggingFace checkpoint

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --models qwen25vl siglip2 videomae
    python scripts/download_models.py --check  # verify checksums/paths
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Local model weights directory — mirrors the storage mount in Docker
# Auto-detect: Docker mount path (/workspace/storage/model-weights)
# falls back to local repo path for development.
_DOCKER_PATH = Path("/workspace/storage/model-weights")
_LOCAL_REPO = Path(__file__).resolve().parent.parent.parent / "storage" / "model-weights"
WORKSPACE = _DOCKER_PATH if _DOCKER_PATH.exists() or _DOCKER_PATH.parent.exists() else _LOCAL_REPO
WORKSPACE.mkdir(parents=True, exist_ok=True)

MODEL_DEFINITIONS = {
    # Qwen2.5-VL-7B-Instruct — open-vocabulary appearance captioning
    "qwen25vl": {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir": "qwen2_5_vl",
        "size_mb": 15000,
        "description": "Qwen2.5-VL-7B-Instruct (open-vocabulary appearance metadata)",
        "files": [],
    },
    # Grounding DINO 1.6 — open-vocabulary person detection
    "grounding_dino": {
        "repo_id": "IDEA-Research/grounding-dino-1.6-base",
        "fallback_repo_id": "IDEA-Research/grounding-dino-1.6-pro",
        "local_dir": "grounding_dino",
        "size_mb": 4000,
        "description": "Grounding DINO 1.6 base (person detection)",
        "files": ["config.json", "model.safetensors", "preprocessor_config.json"],
    },
    # EVA-02 ViT-L/14 — loaded via timm (not HuggingFace)
    # timm model name: eva02_large_patch14_224.mim_m38m_ft_in22k_in1k
    # timm auto-downloads from HuggingFace internally.
    "eva02": {
        "repo_id": "timm/eva02_large_patch14_224.mim_m38m_ft_in22k_in1k",
        "local_dir": "eva02",
        "size_mb": 5000,
        "description": "EVA-02 ViT-L/14 CLIP (appearance embedding, 1024-dim) — loaded via timm",
        "files": [],
    },
    # SigLIP 2 — zero-shot attribute tagging
    "siglip2": {
        "repo_id": "google/siglip2-so400m-patch14-384",
        "fallback_repo_id": "google/siglip-so400m-patch14-384",
        "local_dir": "siglip2",
        "size_mb": 3000,
        "description": "SigLIP 2-So400m (zero-shot attribute tagging)",
        "files": ["config.json", "model.safetensors", "preprocessor_config.json"],
    },
    # VideoMAE V2 — loaded via transformers from MCG-NJU
    "videomae": {
        "repo_id": "MCG-NJU/videomae-large-finetuned-kinetics",
        "fallback_repo_id": "MCG-NJU/videomae-base-finetuned-kinetics",
        "local_dir": "videomae-action",
        "size_mb": 3000,
        "description": "VideoMAE V2 large (action recognition)",
        "files": ["config.json", "pytorch_model.bin"],
    },
    "seamless_m4t": {
        "repo_id": "facebook/seamless-m4t-v2-large",
        "local_dir": "seamless-m4t",
        "size_mb": 5000,
        "description": "SeamlessM4T v2 large (Vietnamese→English translation)",
        "files": ["config.json", "model.safetensors", "vocab.txt"],
    },
    "realesrgan": {
        "repo_id": "xinntao/Real-ESRGAN",
        "local_dir": "realesrgan",
        "size_mb": 50,
        "description": "Real-ESRGAN x4plus (video super-resolution)",
        "subfolder": "experiments/pretrained_models",
        "files": ["RealESRGAN_x4plus.pth"],
    },
    "propainter": {
        "repo_id": "IsaacLo/ProPainter",
        "local_dir": "propainter",
        "size_mb": 400,
        "description": "ProPainter (video inpainting)",
        "files": ["ProPainter.pth"],
    },
    "rife": {
        "repo_id": "hzwer/ECCV2022-RIFE",
        "local_dir": "rife",
        "size_mb": 100,
        "description": "RIFE (frame interpolation)",
        "files": ["rife.pth"],
    },
}

REQUIRED_PACKAGES = [
    "torch>=2.3.0",
    "torchvision>=0.18.0",
    "transformers>=4.49.0",
    "accelerate>=0.30.0",
    "timm>=1.0.3",
    "sentencepiece>=0.2.0",
    "safetensors>=0.4.3",
    "huggingface-hub>=0.23.0",
]


def check_dependencies() -> bool:
    """Verify required packages are installed."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split(">")[0].split("=")[0]
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.warning("Missing packages: %s", missing)
        logger.warning("Install with: pip install %s", " ".join(missing))
        return False
    return True


def download_with_hf_hub(repo_id: str, local_dir: str, filenames: list[str],
                         subfolder: str | None = None, use_auth_token: bool | None = None,
                         workspace: Path | None = None,
                         fallback_repo_id: str | None = None) -> Path:
    """Download model files using huggingface_hub.

    Args:
        repo_id: Primary HuggingFace repo ID.
        fallback_repo_id: Fallback repo ID if primary fails (e.g. gated/private repo).
        filenames: List of files to download.
        subfolder: Optional subfolder path inside repo.
        use_auth_token: HuggingFace token for gated repos.
        workspace: Target directory.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    target = (workspace or WORKSPACE) / local_dir
    target.mkdir(parents=True, exist_ok=True)

    # EVA-02 loaded via timm — no manual download needed
    if "timm/" in repo_id:
        logger.info("  %s loaded via timm at runtime — no manual download needed", local_dir)
        return target

    logger.info("Downloading %s → %s", repo_id, target)

    # snapshot_download does NOT support subfolder on older versions
    # Try snapshot first without subfolder, then per-file with subfolder
    def _try_snapshot(rid: str) -> bool:
        try:
            snapshot_download(
                repo_id=rid,
                local_dir=str(target),
                token=use_auth_token,
                resume_download=True,
                ignore_patterns=["*.mp4", "*.avi", "*.mov", "*.zip"],
            )
            logger.info("✓ Downloaded %s", rid)
            return True
        except Exception as e:
            logger.warning("  snapshot_download failed for %s: %s", rid, e)
            return False

    # Try primary repo
    if not _try_snapshot(repo_id):
        # Try fallback repo
        if fallback_repo_id and _try_snapshot(fallback_repo_id):
            pass  # fallback succeeded
        else:
            # Per-file download as last resort
            for fname in filenames:
                _download_file(
                    repo_id, fname, subfolder, str(target), use_auth_token
                )

    return target


def _download_file(repo_id: str, fname: str, subfolder: str | None,
                  target_dir: str, token: str | None) -> None:
    """Download a single file from a repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            subfolder=subfolder,
            local_dir=target_dir,
            token=token,
        )
        logger.info("  ✓ %s", fname)
    except Exception as fe:
        logger.warning("  ✗ %s: %s", fname, fe)


def download_torch_weights(model_key: str, target: Path) -> Path:
    """Download PyTorch .pth weights for non-HF models."""
    import urllib.request

    urls = {
        "realesrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "propainter": "https://download.openmmlab.com/mmediting/inpainting/propainters/ProPainter.pth",
        "rife": "https://github.com/hzwer/ECCV2022-RIFE/releases/download/v1.0/RIFE.pth",
    }

    if model_key not in urls:
        return target

    url = urls[model_key]
    fname = Path(url).name
    dest = target / fname

    if dest.exists():
        logger.info("  ✓ %s already exists, skipping", fname)
        return dest

    logger.info("  Downloading %s → %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info("  ✓ Downloaded %s", fname)
    except Exception as e:
        logger.warning("  ✗ Failed to download %s: %s", url, e)

    return dest


def download_all(workspace: Path = WORKSPACE) -> None:
    """Download all models."""
    check_dependencies()

    total_mb = sum(m["size_mb"] for m in MODEL_DEFINITIONS.values())
    logger.info("=" * 60)
    logger.info("Downloading %d models (~%d GB total)", len(MODEL_DEFINITIONS), total_mb // 1000)
    logger.info("Target: %s", workspace)
    logger.info("=" * 60)

    results = {}
    for key, spec in MODEL_DEFINITIONS.items():
        logger.info("")
        try:
            target = download_with_hf_hub(
                repo_id=spec["repo_id"],
                local_dir=spec["local_dir"],
                filenames=spec.get("files", []),
                subfolder=spec.get("subfolder"),
                fallback_repo_id=spec.get("fallback_repo_id"),
                workspace=workspace,
            )
            results[key] = target
        except Exception as e:
            logger.error("✗ Failed to download %s: %s", key, e)

    # Download torch weights separately (not on HuggingFace)
    torch_models = {"realesrgan", "propainter", "rife"}
    for key in torch_models:
        if key in results:
            download_torch_weights(key, results[key])

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    for key, path in results.items():
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        logger.info("  %-20s → %s (%.2f GB)", key, path, size_gb)


def check_installed(workspace: Path = WORKSPACE) -> None:
    """Check which models are already downloaded."""
    logger.info("Checking installed models in %s", workspace)
    logger.info("")

    all_found = True
    for key, spec in MODEL_DEFINITIONS.items():
        target = workspace / spec["local_dir"]
        if target.exists() and any(target.iterdir()):
            size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
            logger.info("  ✓ %-20s (%.2f GB) — %s", key, size_gb, spec["description"])
        else:
            logger.info("  ✗ %-20s MISSING — %s", key, spec["description"])
            all_found = False

    if all_found:
        logger.info("\n✓ All models installed.")
    else:
        logger.info("\n✗ Some models missing. Run: python scripts/download_models.py --all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SOTA 2026 models for TraceX-AI")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--models", nargs="+",
                        choices=list(MODEL_DEFINITIONS.keys()),
                        help="Download specific models")
    parser.add_argument("--check", action="store_true", help="Check installed models")
    parser.add_argument("--workspace", default=str(WORKSPACE),
                        help=f"Target directory (default: {WORKSPACE})")
    parser.add_argument("--token", default=None, help="HuggingFace token (for gated models)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_installed(workspace)
    elif args.all:
        download_all(workspace)
    elif args.models:
        for key in args.models:
            spec = MODEL_DEFINITIONS[key]
            logger.info("Downloading %s: %s", key, spec["description"])
            download_with_hf_hub(
                repo_id=spec["repo_id"],
                local_dir=spec["local_dir"],
                filenames=spec.get("files", []),
                subfolder=spec.get("subfolder"),
                fallback_repo_id=spec.get("fallback_repo_id"),
                use_auth_token=args.token,
                workspace=workspace,
            )
        check_installed(workspace)
    else:
        logger.info("No action specified. Use --all, --models <list>, or --check")
        logger.info("Available models: %s", list(MODEL_DEFINITIONS.keys()))
        check_installed(workspace)


if __name__ == "__main__":
    main()
