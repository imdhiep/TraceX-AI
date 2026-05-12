"""Backward-compat shim — the action-keyword vocabulary moved into
`query_metadata_parse` so query parsing has one source of truth.

This module re-exports the previous public names so older imports keep
working. New code should import from `app.services.query_metadata_parse`.
"""

from __future__ import annotations

from app.services.query_metadata_parse import (  # noqa: F401
    detect_query_actions,
    all_known_actions,
    ACTION_BONUS_SCORE,
    ACTION_CONFIDENCE_FLOOR,
)
