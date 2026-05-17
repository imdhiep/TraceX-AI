"""Configuration settings for query-service."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Settings for query-service."""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # SeamlessM4T v2-large (Vietnamese ↔ English translation)
    # Replaces deprecated Helsinki-NLP opus-mt-vi-en
    seamless_model: str = os.getenv(
        "SEAMLESS_MODEL",
        "facebook/seamless-m4t-v2-large"
    )

    # Translation model path (local cache)
    translation_model_path: Path = Path(
        os.getenv("TRANSLATION_MODEL_PATH", "/workspace/models/seamless-m4t")
    )

    # Trace service (for GPU re-ranking)
    trace_service_url: str = os.getenv(
        "TRACE_SERVICE_URL",
        "http://trace-service:8004"
    )

    # Storage
    preview_root: Path = Path(
        os.getenv("PREVIEW_ROOT", "/workspace/storage/candidate-previews")
    )

    # Public API
    public_api_base_url: str = os.getenv("PUBLIC_API_BASE_URL", "")

    # Hybrid search: min score threshold for returning results
    min_fusion_score: float = float(os.getenv("MIN_FUSION_SCORE", "0.5"))

    # Max candidates to return
    max_candidates: int = int(os.getenv("MAX_CANDIDATES", "50"))


settings = Settings()
