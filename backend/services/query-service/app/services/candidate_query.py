"""Candidate search, ranking, and preview service.

Adapted from metadata-service candidate_service.py
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import cv2
import httpx
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from shared import PersonCandidate, QueueVideoAsset
from shared.config import settings

logger = logging.getLogger(__name__)

PERSON_CANDIDATE_INSERT_CHUNK_SIZE = 50


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "video"


def _coerce_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            result.append(text)
    return result


def _coerce_mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(token) >= 2}


def _semantic_overlap(query_text: str, candidate: dict) -> float:
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokenize(_candidate_search_document(candidate))
    if not candidate_tokens:
        return 0.0
    return float(len(query_tokens & candidate_tokens) / max(len(query_tokens), 1))


def _score_candidate(query_text: str, candidate: dict) -> float:
    score = _semantic_overlap(query_text, candidate)
    attributes = _coerce_string_list(candidate.get("semantic_attributes"))
    if attributes and any(token in " ".join(attributes).lower() for token in _tokenize(query_text)):
        score += 0.1
    timeline = candidate.get("timeline")
    if isinstance(timeline, list) and timeline:
        score += min(len(timeline), 5) * 0.01
    return round(score, 6)


def _candidate_search_document(person: dict) -> str:
    parts: list[str] = []
    for key in ("search_text", "appearance_summary"):
        text = str(person.get(key) or "").strip()
        if text:
            parts.append(text)

    attributes = _coerce_string_list(person.get("semantic_attributes"))
    if attributes:
        parts.append("attributes: " + ", ".join(attributes))

    timeline = person.get("timeline")
    if isinstance(timeline, list):
        actions = [str(item.get("action_summary") or "").strip() for item in timeline if isinstance(item, dict)]
        actions = [item for item in actions if item]
        if actions:
            parts.append("timeline: " + " ".join(actions))

    return " ".join(parts).strip()


def _candidate_bbox(raw_metadata: dict[str, Any]) -> list[int]:
    for key in ("bbox", "representative_bbox"):
        bbox = raw_metadata.get(key)
        if isinstance(bbox, list) and len(bbox) >= 4:
            cleaned: list[int] = []
            for value in bbox[:4]:
                try:
                    cleaned.append(int(float(value)))
                except (TypeError, ValueError):
                    cleaned.append(0)
            return cleaned
    return []


def candidate_to_payload(candidate: PersonCandidate, queue_video: QueueVideoAsset | None = None) -> dict:
    raw_metadata = candidate.raw_metadata or {}
    bbox = _candidate_bbox(raw_metadata)
    storage_path = (
        queue_video.local_video_path
        if queue_video and (queue_video.local_video_path or "").strip()
        else queue_video.available_link_video
        if queue_video
        else None
    )
    return {
        "candidate_id": candidate.candidate_id,
        "camera_id": candidate.camera_id,
        "video_id": candidate.video_id,
        "track_id": candidate.track_id,
        "human_key": candidate.human_key,
        "frame_idx": candidate.frame_idx,
        "bbox": bbox,
        "search_text": candidate.search_text,
        "metadata_path": candidate.metadata_path,
        "attribute_summary": raw_metadata.get("attribute_summary"),
        "appearance_summary": raw_metadata.get("appearance_summary"),
        "attribute_embedding_vector": raw_metadata.get("attribute_embedding_vector") if isinstance(raw_metadata.get("attribute_embedding_vector"), list) else [],
        "appearance_embedding_vector": raw_metadata.get("appearance_embedding_vector") if isinstance(raw_metadata.get("appearance_embedding_vector"), list) else [],
        "semantic_attributes": _coerce_string_list(raw_metadata.get("semantic_attributes")),
        "embedding_vector": raw_metadata.get("embedding_vector") if isinstance(raw_metadata.get("embedding_vector"), list) else [],
        "visibility_scores": _coerce_mapping(raw_metadata.get("visibility_scores")),
        "world_position": raw_metadata.get("world_position") or raw_metadata.get("top_point_projection"),
        "score": raw_metadata.get("score"),
        "timeline": raw_metadata.get("timeline") if isinstance(raw_metadata.get("timeline"), list) else [],
        "matched_segments": raw_metadata.get("matched_segments") or [],
        "action_semantic_embedding": _coerce_mapping(raw_metadata.get("action_semantic_embedding")),
        "tracklet_feature_pipeline": _coerce_mapping(raw_metadata.get("tracklet_feature_pipeline")),
        "available_link_video": queue_video.available_link_video if queue_video else None,
        "available_link_metadata": queue_video.available_link_metadata if queue_video else None,
        "drive_video_file_id": queue_video.drive_video_file_id if queue_video else None,
        "drive_metadata_file_id": queue_video.drive_metadata_file_id if queue_video else None,
        "local_video_path": queue_video.local_video_path if queue_video else None,
        "local_metadata_path": queue_video.local_metadata_path if queue_video else None,
        "storage_path": storage_path,
        "video_title": queue_video.title if queue_video else None,
        "source_filename": queue_video.source_filename if queue_video else None,
        "recorded_start": raw_metadata.get("recorded_start"),
        "preview_image_url": f"/api/v1/candidates/{candidate.candidate_id}/preview",
        "raw_metadata": raw_metadata,
    }


def candidate_to_ranking_payload(candidate: PersonCandidate, queue_video: QueueVideoAsset | None = None) -> dict:
    raw_metadata = candidate.raw_metadata or {}
    bbox = _candidate_bbox(raw_metadata)
    storage_path = (
        queue_video.local_video_path
        if queue_video and (queue_video.local_video_path or "").strip()
        else queue_video.available_link_video
        if queue_video
        else None
    )
    return {
        "candidate_id": candidate.candidate_id,
        "camera_id": candidate.camera_id,
        "video_id": candidate.video_id,
        "track_id": candidate.track_id,
        "human_key": candidate.human_key,
        "frame_idx": candidate.frame_idx,
        "bbox": bbox,
        "search_text": candidate.search_text,
        "metadata_path": candidate.metadata_path,
        "attribute_summary": raw_metadata.get("attribute_summary"),
        "appearance_summary": raw_metadata.get("appearance_summary"),
        "attribute_embedding_vector": raw_metadata.get("attribute_embedding_vector") if isinstance(raw_metadata.get("attribute_embedding_vector"), list) else [],
        "appearance_embedding_vector": raw_metadata.get("appearance_embedding_vector") if isinstance(raw_metadata.get("appearance_embedding_vector"), list) else [],
        "semantic_attributes": _coerce_string_list(raw_metadata.get("semantic_attributes")),
        "visibility_scores": _coerce_mapping(raw_metadata.get("visibility_scores")),
        "world_position": raw_metadata.get("world_position"),
        "score": raw_metadata.get("score"),
        "embedding_vector": raw_metadata.get("embedding_vector") if isinstance(raw_metadata.get("embedding_vector"), list) else [],
        # BEV coordinates (projected 3D world position)
        "bev_x": raw_metadata.get("bev_x", 0.0),
        "bev_y": raw_metadata.get("bev_y", 0.0),
        # Attribute fields from video_process pipeline (post-refactor field names)
        "top_color": raw_metadata.get("upper_color") or raw_metadata.get("top_color", "unknown"),
        "bottom_color": raw_metadata.get("lower_color") or raw_metadata.get("bottom_color", "unknown"),
        "gender": raw_metadata.get("gender", "unknown"),
        "has_bag": raw_metadata.get("bag_presence") or raw_metadata.get("bag", "no_bag"),
        "has_hat": raw_metadata.get("hat_presence") or raw_metadata.get("hat", "no_hat"),
        "timeline": raw_metadata.get("timeline") if isinstance(raw_metadata.get("timeline"), list) else [],
        "matched_segments": raw_metadata.get("matched_segments") or [],
        "action_semantic_embedding": _coerce_mapping(raw_metadata.get("action_semantic_embedding")),
        "tracklet_feature_pipeline": _coerce_mapping(raw_metadata.get("tracklet_feature_pipeline")),
        "available_link_video": queue_video.available_link_video if queue_video else None,
        "available_link_metadata": queue_video.available_link_metadata if queue_video else None,
        "local_video_path": queue_video.local_video_path if queue_video else None,
        "local_metadata_path": queue_video.local_metadata_path if queue_video else None,
        "storage_path": storage_path,
        "video_title": queue_video.title if queue_video else None,
        "source_filename": queue_video.source_filename if queue_video else None,
        "preview_image_url": f"/api/v1/candidates/{candidate.candidate_id}/preview",
        "raw_metadata": raw_metadata,
    }


def _queue_video_map(session: Session, video_ids: list[str]) -> dict[str, QueueVideoAsset]:
    cleaned = [video_id for video_id in video_ids if video_id]
    if not cleaned:
        return {}
    rows = session.scalars(select(QueueVideoAsset).where(QueueVideoAsset.video_id.in_(cleaned))).all()
    return {row.video_id: row for row in rows}


def search_candidates(
    session: Session,
    query: str | None = None,
    limit: int = 20,
    camera_ids: list[str] | None = None,
) -> list[dict]:
    statement = select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())
    cleaned_query = (query or "").strip()
    if cleaned_query:
        tokens = [t for t in cleaned_query.lower().split() if len(t) >= 2]
        for token in tokens:
            pattern = f"%{token}%"
            statement = statement.where(
                or_(
                    PersonCandidate.search_text.ilike(pattern),
                    PersonCandidate.camera_id.ilike(pattern),
                    PersonCandidate.video_id.ilike(pattern),
                    PersonCandidate.human_key.ilike(pattern),
                )
            )
    if camera_ids:
        statement = statement.where(PersonCandidate.camera_id.in_(camera_ids))
    rows = session.scalars(statement.limit(max(1, min(limit, 500)))).all()
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    return [candidate_to_payload(row, queue_map.get(str(row.video_id or ""))) for row in rows]


def get_candidate(session: Session, candidate_id: str) -> dict | None:
    row = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
    if row is None:
        return None
    queue_map = _queue_video_map(session, [str(row.video_id or "")])
    return candidate_to_payload(row, queue_map.get(str(row.video_id or "")))


def get_queue_video(session: Session, video_id: str) -> QueueVideoAsset | None:
    return session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))


def list_queue_videos(session: Session) -> list[dict]:
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return [
        {
            "video_id": video.video_id,
            "camera_id": video.camera_id,
            "title": video.title,
            "queue_position": video.queue_position,
            "storage_backend": video.storage_backend,
            "available_link_video": video.available_link_video,
            "available_link_metadata": video.available_link_metadata,
            "source_filename": video.source_filename,
            "created_at": video.created_at,
        }
        for video in session.scalars(statement).all()
    ]


def _remote_ranking_shortlist_limit(limit: int) -> int:
    bounded_limit = max(1, min(limit, 50))
    return min(max(200, bounded_limit * 40), 400)


def _local_prefilter_ranked_candidates(
    rows: list[PersonCandidate],
    queue_map: dict[str, QueueVideoAsset],
    *,
    query_text: str,
    shortlist_limit: int,
) -> list[dict]:
    scored: list[tuple[float, int, dict[str, Any]]] = []

    for row in rows:
        payload = candidate_to_ranking_payload(row, queue_map.get(str(row.video_id or "")))
        score = _score_candidate(query_text, payload)

        if score <= 0:
            row_search_text = str(row.search_text or "").lower()
            query_tokens = _tokenize(query_text)
            if row_search_text and any(token in row_search_text for token in query_tokens):
                score = 0.005

        scored.append((score, row.id, payload))

    scored.sort(
        key=lambda item: (
            -float(item[0]),
            -int(item[1]),
            str(item[2].get("camera_id") or ""),
            str(item[2].get("track_id") or ""),
            str(item[2].get("candidate_id") or ""),
        )
    )

    positive = [payload for score, _row_id, payload in scored if score > 0][:shortlist_limit]
    if positive:
        return positive
    return [payload for _score, _row_id, payload in scored[:shortlist_limit]]


def get_overview(session: Session) -> dict:
    total_candidates = session.scalar(select(func.count()).select_from(PersonCandidate)) or 0
    total_cameras = session.scalar(select(func.count(func.distinct(PersonCandidate.camera_id))).select_from(PersonCandidate)) or 0
    total_videos = session.scalar(select(func.count(func.distinct(PersonCandidate.video_id))).select_from(PersonCandidate)) or 0
    total_queue_videos = session.scalar(select(func.count()).select_from(QueueVideoAsset)) or 0
    
    return {
        "metrics": {
            "total_candidates": int(total_candidates),
            "total_cameras": int(total_cameras),
            "total_videos": int(total_videos),
            "total_queue_videos": int(total_queue_videos),
        }
    }
