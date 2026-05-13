"""Query → structured metadata parser for query-service.

Loads vocabulary from `backend/config/query_metadata_vocab.json` once at import
time and exposes:

  parse_query_metadata(query)            → ParsedQueryMetadata
  ACTION_BONUS_SCORE                     → fusion bonus when action matches
  ACTION_CONFIDENCE_FLOOR                → min VideoMAE confidence to count
  METADATA_BONUS_PER_MATCH               → per-field bonus when query attr
                                           matches tracklet attr
  METADATA_MAX_BONUS                     → cap on summed metadata bonus

Design notes:
- Color ↔ garment binding uses a sliding window: a color hit at token index i
  binds to the nearest garment whose start index is within
  `_garment_color_window` tokens of i (either direction). When no garment is
  close enough, the color falls back to a generic "any upper or lower" slot
  ("red dress" without explicit upper/lower attaches loosely to clothing).
- Gender / action don't depend on garment binding — they're flat keyword hits.
- ASCII single-token keywords use word-boundary regex (so "run" doesn't hit
  "running" unless explicitly listed). Phrases / non-ASCII use substring.
- Tier A scoring: every field match adds METADATA_BONUS_PER_MATCH; we never
  hard-filter on color/garment (VLM ingest can mis-label). Gender stays a
  hard filter at the prefilter stage — kept in candidates.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []

    env_path = os.environ.get("QUERY_METADATA_VOCAB_CONFIG")
    if env_path:
        paths.append(Path(env_path))

    paths.append(Path("/app/backend/config/query_metadata_vocab.json"))

    # Local source tree fallback. In Docker, __file__ is under /app/app/... and
    # this parent depth may not exist, so keep it best-effort.
    try:
        paths.append(Path(__file__).resolve().parents[4] / "config" / "query_metadata_vocab.json")
    except IndexError:
        pass

    return paths

# Bare-minimum fallback so the service still boots without the config file.
_FALLBACK_VOCAB: dict = {
    "_bonus_per_match": 0.04,
    "_max_metadata_bonus": 0.20,
    "_garment_color_window": 4,
    "_action_bonus": 0.05,
    "_action_confidence_floor": 0.3,
    "gender":  {"woman": ["woman", "female"], "man": ["man", "male"]},
    "color":   {"red": ["red"], "white": ["white"], "black": ["black"]},
    "garment": {"upper": ["shirt"], "lower": ["pants"], "shoes": ["shoes"]},
    "actions": {"walking": ["walking"], "running": ["running"]},
}


def _load_vocab() -> dict:
    for path in _candidate_config_paths():
        if not path or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("color"):
                continue
            logger.info("[query_metadata_parse] loaded vocab from %s", path)
            return data
        except Exception as exc:
            logger.warning("[query_metadata_parse] failed to load %s: %s", path, exc)
    logger.warning("[query_metadata_parse] no vocab found, using fallback")
    return _FALLBACK_VOCAB


_VOCAB = _load_vocab()
METADATA_BONUS_PER_MATCH = float(_VOCAB.get("_bonus_per_match", 0.04))
METADATA_MAX_BONUS       = float(_VOCAB.get("_max_metadata_bonus", 0.20))
ACTION_BONUS_SCORE       = float(_VOCAB.get("_action_bonus", 0.05))
ACTION_CONFIDENCE_FLOOR  = float(_VOCAB.get("_action_confidence_floor", 0.3))
_GARMENT_WINDOW          = int(_VOCAB.get("_garment_color_window", 4))


# garment field → set of keywords (lowercased)
def _build_inverse(group: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Returns list of (keyword_lower, canonical_label) for all keywords."""
    out: list[tuple[str, str]] = []
    for label, kws in group.items():
        for kw in kws:
            kw_l = str(kw).lower().strip()
            if kw_l:
                out.append((kw_l, label))
    # Longest keyword first so "đi bộ" matches before "đi"
    out.sort(key=lambda x: -len(x[0]))
    return out


_GENDER_LOOKUP  = _build_inverse(_VOCAB.get("gender", {}))
_COLOR_LOOKUP   = _build_inverse(_VOCAB.get("color", {}))
_GARMENT_LOOKUP = _build_inverse(_VOCAB.get("garment", {}))
_ACTION_LOOKUP  = _build_inverse(_VOCAB.get("actions", {}))


def _is_ascii_word(s: str) -> bool:
    return bool(s) and " " not in s and s.isascii() and s.replace("_", "").replace("-", "").isalnum()


@dataclass
class _Hit:
    """A keyword hit at a position in the (whitespace-tokenized) query."""
    label: str
    start_token: int   # token index of first matched word
    end_token: int     # exclusive — token index after last matched word


def _find_hits(query: str, lookup: list[tuple[str, str]]) -> list[_Hit]:
    """Find all keyword hits in `query`. Token positions use whitespace split.
    For multi-word / non-ASCII keywords we still report token positions by
    locating the substring and mapping char span → token span."""
    q = query.lower()
    tokens = q.split()
    # Precompute char→token map (start char of each token in q)
    token_char_spans: list[tuple[int, int]] = []
    cur = 0
    for tok in tokens:
        idx = q.find(tok, cur)
        token_char_spans.append((idx, idx + len(tok)))
        cur = idx + len(tok)

    def char_span_to_token_span(c0: int, c1: int) -> tuple[int, int]:
        # First token whose end > c0, last token whose start < c1
        t_start = 0
        while t_start < len(tokens) and token_char_spans[t_start][1] <= c0:
            t_start += 1
        t_end = t_start
        while t_end < len(tokens) and token_char_spans[t_end][0] < c1:
            t_end += 1
        return t_start, max(t_end, t_start + 1)

    hits: list[_Hit] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for kw, label in lookup:
        if _is_ascii_word(kw):
            for m in re.finditer(rf"\b{re.escape(kw)}\b", q):
                ts, te = char_span_to_token_span(m.start(), m.end())
                key = (ts, te, label)
                if key not in seen_spans:
                    seen_spans.add(key)
                    hits.append(_Hit(label, ts, te))
        else:
            # Substring scan — phrases and non-ASCII
            start = 0
            while True:
                idx = q.find(kw, start)
                if idx < 0:
                    break
                ts, te = char_span_to_token_span(idx, idx + len(kw))
                key = (ts, te, label)
                if key not in seen_spans:
                    seen_spans.add(key)
                    hits.append(_Hit(label, ts, te))
                start = idx + len(kw)
    return hits


@dataclass
class ParsedQueryMetadata:
    """Structured constraints extracted from a (translated, lowercased) query.

    Each field maps to a list of canonical values. Empty list = no constraint.
    Lists allow multi-value queries ("red or blue shirt") in the future without
    a schema change.
    """
    gender: list[str] = field(default_factory=list)
    upper_color: list[str] = field(default_factory=list)
    lower_color: list[str] = field(default_factory=list)
    shoes_color: list[str] = field(default_factory=list)
    hat_color: list[str] = field(default_factory=list)
    bag_type: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    # Colors that couldn't be bound to a specific garment — they still
    # contribute a smaller bonus by matching upper OR lower.
    unbound_colors: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.gender or self.upper_color or self.lower_color
            or self.shoes_color or self.hat_color or self.bag_type
            or self.actions or self.unbound_colors
        )


# Map garment label → which Tracklet color column it constrains
_GARMENT_TO_FIELD = {
    "upper": "upper_color",
    "lower": "lower_color",
    "shoes": "shoes_color",
    "hat":   "hat_color",
}


def parse_query_metadata(query: str) -> ParsedQueryMetadata:
    """Parse a translated (English-preferred) lowercase query into structured
    constraints. See module docstring for binding rules."""
    out = ParsedQueryMetadata()
    if not query or not query.strip():
        return out

    q = query.lower().strip()
    gender_hits  = _find_hits(q, _GENDER_LOOKUP)
    color_hits   = _find_hits(q, _COLOR_LOOKUP)
    garment_hits = _find_hits(q, _GARMENT_LOOKUP)
    action_hits  = _find_hits(q, _ACTION_LOOKUP)

    # Gender — only commit if exactly one canonical gender appears.
    gender_labels = {h.label for h in gender_hits}
    if len(gender_labels) == 1:
        out.gender = [next(iter(gender_labels))]

    # Actions — flat set, dedup.
    out.actions = sorted({h.label for h in action_hits})

    # Color ↔ garment binding via sliding window.
    used_garment_hits: set[int] = set()
    for color_hit in color_hits:
        best_idx = -1
        best_dist = _GARMENT_WINDOW + 1
        for i, g_hit in enumerate(garment_hits):
            if i in used_garment_hits:
                continue
            # Distance between the two hits in token space.
            if g_hit.end_token <= color_hit.start_token:
                dist = color_hit.start_token - g_hit.end_token
            elif color_hit.end_token <= g_hit.start_token:
                dist = g_hit.start_token - color_hit.end_token
            else:
                dist = 0  # overlapping / adjacent
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist <= _GARMENT_WINDOW:
            g_hit = garment_hits[best_idx]
            used_garment_hits.add(best_idx)
            # "dress" is in both upper and lower vocab — bind to whichever
            # the lookup table returned for this token.
            field_name = _GARMENT_TO_FIELD.get(g_hit.label)
            if field_name == "upper_color":
                if color_hit.label not in out.upper_color:
                    out.upper_color.append(color_hit.label)
            elif field_name == "lower_color":
                if color_hit.label not in out.lower_color:
                    out.lower_color.append(color_hit.label)
            elif field_name == "shoes_color":
                if color_hit.label not in out.shoes_color:
                    out.shoes_color.append(color_hit.label)
            elif field_name == "hat_color":
                if color_hit.label not in out.hat_color:
                    out.hat_color.append(color_hit.label)
        else:
            if color_hit.label not in out.unbound_colors:
                out.unbound_colors.append(color_hit.label)

    # Bag presence: any garment hit with label "bag" → bag is part of query.
    # We don't have a "bag_color" column today; record that bag is requested
    # so future filters can prefer bag_presence='yes'. Stored loosely under
    # bag_type for now (string-compare in scoring).
    if any(g.label == "bag" for g in garment_hits):
        out.bag_type = ["any"]

    return out


# ── Backward-compat shim for the previous action_match module ─────────────
# The first iteration shipped a separate `app.services.action_match` module
# with these symbols; downstream code already imports them. Re-export so
# callers keep working without an explicit refactor.

def detect_query_actions(query: str) -> set[str]:
    """Subset of parse_query_metadata() exposing only action labels.
    Preserved for callers that only need actions."""
    return set(parse_query_metadata(query).actions)


def all_known_actions() -> Iterable[str]:
    return tuple(_VOCAB.get("actions", {}).keys())
