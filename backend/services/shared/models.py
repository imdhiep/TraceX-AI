"""SQLAlchemy models — TraceX-AI v3.3 aligned with Section 20 spec.

Tables (17):
  1.  users
  2.  cameras
  3.  camera_zones
  4.  camera_edges
  5.  camera_settings
  6.  videos
  7.  tracklets
  8.  tracklets_embeddings
  9.  tracklets_actions
  10. query_history
  11. query_candidates
  12. query_candidate_tracklets
  13. query_jobs
  14. spatiotemporal_groups
  15. evidence_videos
  16. evidence_tracklets
  17. verified_objects
  18. verified_objects_tracklets

Plus internal (non-spec) staging table:
  - queue_video_assets (VinUni Storage ingest queue)

Naming follows spec exactly (Section 20.11).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
try:
    from pgvector.sqlalchemy import Vector as PgVector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    PgVector = None
    _PGVECTOR_AVAILABLE = False


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ============================================================================
# Group 1: Users & Authentication
# ============================================================================

class User(Base):
    """User model — staff members within the organization (Section 20.2, row 1)."""
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    email: Mapped[str] = mapped_column(String(255), nullable=False,)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="USER", server_default="USER")
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True, server_default="true")
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    videos: Mapped[list["Video"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    query_history: Mapped[list["QueryHistory"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    verified_objects: Mapped[list["VerifiedObject"]] = relationship(
        back_populates="verified_by", cascade="all, delete-orphan",
    )


# ============================================================================
# Group 2: Camera & Topology
# ============================================================================

class Camera(Base):
    """Camera — global camera registry (Section 20.2, row 2)."""
    __tablename__ = "cameras"
    __table_args__ = (
        UniqueConstraint("camera_id", name="uq_cameras_camera_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False, unique=True,)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    fps: Mapped[float] = mapped_column(Float, nullable=False, default=30.0)
    resolution_width: Mapped[int] = mapped_column(Integer, nullable=False, default=1920)
    resolution_height: Mapped[int] = mapped_column(Integer, nullable=False, default=1080)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # Relationships
    zones: Mapped[list["CameraZone"]] = relationship(back_populates="camera", cascade="all, delete-orphan")
    settings: Mapped[list["CameraSettings"]] = relationship(back_populates="camera", cascade="all, delete-orphan")


class CameraZone(Base):
    """Camera zone — entry/exit region polygons (Section 20.2, row 2)."""
    __tablename__ = "camera_zones"
    __table_args__ = (
        UniqueConstraint("camera_id", "zone_type", name="uq_camera_zones_camera_zone"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    camera_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("cameras.camera_id", ondelete="CASCADE"), nullable=False,
    )
    zone_type: Mapped[str] = mapped_column(String(32), nullable=False)  # "entry" | "exit"
    polygon: Mapped[dict] = mapped_column(JSON, nullable=False)  # [{"x": 0, "y": 0}, ...]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    camera: Mapped[Camera] = relationship(back_populates="zones")


class CameraEdge(Base):
    """Camera edge — temporal transition between cameras (Section 20.2, row 2).

    Represents: person seen at from_camera → to_camera within [min_seconds, max_seconds].
    Used for spatiotemporal matching in Trace (Section 20.6 Luồng 3).
    """
    __tablename__ = "camera_edges"
    __table_args__ = (
        UniqueConstraint("from_camera_id", "to_camera_id", name="uq_camera_edges_pair"),
        Index("ix_camera_edges_from", "from_camera_id"),
        Index("ix_camera_edges_to", "to_camera_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    from_camera_id: Mapped[str] = mapped_column(String(50), nullable=False,)
    to_camera_id: Mapped[str] = mapped_column(String(50), nullable=False,)
    min_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=30.0)
    max_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=120.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )


class CameraSettings(Base):
    """Per-camera settings (Section 20.2, row 2)."""
    __tablename__ = "camera_settings"
    __table_args__ = (
        UniqueConstraint("camera_id", "setting_key", name="uq_camera_settings_camera_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    camera_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("cameras.camera_id", ondelete="CASCADE"), nullable=False,
    )
    setting_key: Mapped[str] = mapped_column(String(128), nullable=False)
    setting_value: Mapped[str] = mapped_column(String(512), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )

    camera: Mapped[Camera] = relationship(back_populates="settings")


# ============================================================================
# Group 3: Video & Tracklet (AI Detection Data)
# ============================================================================

class Video(Base):
    """Video — uploaded video metadata (Section 20.2 row 6, Section 20.3)."""
    __tablename__ = "videos"
    __table_args__ = (
        UniqueConstraint("video_id", name="uq_videos_video_id"),
        Index("ix_videos_camera_id", "camera_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    video_id: Mapped[str] = mapped_column(
        String(255), default=lambda: str(uuid.uuid4()), nullable=False,
    )
    camera_id: Mapped[str | None] = mapped_column(String(50), nullable=True,)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False, default="local_volume")
    drive_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed: Mapped[bool] = mapped_column(nullable=False, default=False)
    recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    owner: Mapped[User | None] = relationship(back_populates="videos")
    tracklets: Mapped[list["Tracklet"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    query_history: Mapped[list["QueryHistory"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class Tracklet(Base):
    """Tracklet — one detected person in a video (Section 20.2 row 7, Section 20.4).

    Each row = one person tracked across frames in a single video.
    Extracted attributes: appearance (color, gender), quality, BEV position.
    """
    __tablename__ = "tracklets"
    __table_args__ = (
        UniqueConstraint("tracklet_id", name="uq_tracklets_tracklet_id"),
        Index("ix_tracklets_video_id", "video_id"),
        Index("ix_tracklets_camera_id", "camera_id"),
        Index("ix_tracklets_bev_xy", "bev_x", "bev_y"),
        Index("ix_tracklets_upper_color",  "upper_color"),
        Index("ix_tracklets_upper_type",   "upper_type"),
        Index("ix_tracklets_lower_color",  "lower_color"),
        Index("ix_tracklets_lower_type",   "lower_type"),
        Index("ix_tracklets_bag_presence", "bag_presence"),
        Index("ix_tracklets_hat_presence", "hat_presence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tracklet_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    video_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False,
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    track_id: Mapped[str] = mapped_column(String(50), nullable=False)

    # Temporal
    start_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    end_time:   Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Quality
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Demographic (Qwen2-VL) ──────────────────────────────────────────
    gender:         Mapped[str]           = mapped_column(String(32), nullable=False, default="unknown")
    gender_conf:    Mapped[float | None]  = mapped_column(Float, nullable=True)
    age_range:      Mapped[str]           = mapped_column(String(32), nullable=False, default="unknown")
    age_range_conf: Mapped[float | None]  = mapped_column(Float, nullable=True)

    # ── Upper clothing (Qwen2-VL) ───────────────────────────────────────
    upper_color:     Mapped[str | None]    = mapped_column(String(64),  nullable=True)
    upper_type:      Mapped[str | None]    = mapped_column(String(128), nullable=True)
    upper_desc:      Mapped[str | None]    = mapped_column(Text,        nullable=True)
    upper_conf:      Mapped[float | None]  = mapped_column(Float,       nullable=True)
    upper_desc_conf: Mapped[float | None]  = mapped_column(Float,       nullable=True)

    # ── Lower clothing ──────────────────────────────────────────────────
    lower_color:     Mapped[str | None]    = mapped_column(String(64),  nullable=True)
    lower_type:      Mapped[str | None]    = mapped_column(String(128), nullable=True)
    lower_desc:      Mapped[str | None]    = mapped_column(Text,        nullable=True)
    lower_conf:      Mapped[float | None]  = mapped_column(Float,       nullable=True)
    lower_desc_conf: Mapped[float | None]  = mapped_column(Float,       nullable=True)

    # ── Shoes ───────────────────────────────────────────────────────────
    shoes_color:     Mapped[str | None]    = mapped_column(String(64),  nullable=True)
    shoes_type:      Mapped[str | None]    = mapped_column(String(128), nullable=True)
    shoes_desc:      Mapped[str | None]    = mapped_column(Text,        nullable=True)
    shoes_conf:      Mapped[float | None]  = mapped_column(Float,       nullable=True)
    shoes_desc_conf: Mapped[float | None]  = mapped_column(Float,       nullable=True)

    # ── Bag (accessory) ─────────────────────────────────────────────────
    bag_presence:    Mapped[str | None]    = mapped_column(String(16),  nullable=True)
    bag_type:        Mapped[str | None]    = mapped_column(String(64),  nullable=True)
    bag_desc:        Mapped[str | None]    = mapped_column(Text,        nullable=True)
    bag_conf:        Mapped[float | None]  = mapped_column(Float,       nullable=True)
    bag_desc_conf:   Mapped[float | None]  = mapped_column(Float,       nullable=True)

    # ── Hat (accessory) ─────────────────────────────────────────────────
    hat_presence:    Mapped[str | None]    = mapped_column(String(16),  nullable=True)
    hat_color:       Mapped[str | None]    = mapped_column(String(64),  nullable=True)
    hat_type:        Mapped[str | None]    = mapped_column(String(128), nullable=True)
    hat_desc:        Mapped[str | None]    = mapped_column(Text,        nullable=True)
    hat_conf:        Mapped[float | None]  = mapped_column(Float,       nullable=True)
    hat_desc_conf:   Mapped[float | None]  = mapped_column(Float,       nullable=True)

    # ── Mask ────────────────────────────────────────────────────────────
    mask_presence:   Mapped[str]           = mapped_column(String(16), nullable=False, default="unknown")
    mask_conf:       Mapped[float | None]  = mapped_column(Float,      nullable=True)

    # ── Hair ────────────────────────────────────────────────────────────
    hair_style:      Mapped[str]           = mapped_column(String(64), nullable=False, default="unknown")
    hair_style_conf: Mapped[float | None]  = mapped_column(Float,      nullable=True)
    hair_color:      Mapped[str]           = mapped_column(String(64), nullable=False, default="unknown")
    hair_color_conf: Mapped[float | None]  = mapped_column(Float,      nullable=True)

    # ── Free-text whole-person summary ──────────────────────────────────
    appearance_summary:      Mapped[str]          = mapped_column(Text,  nullable=False, default="")
    appearance_summary_conf: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Spatial (BEV) ───────────────────────────────────────────────────
    bev_x: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bev_y: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Crop & bbox ─────────────────────────────────────────────────────
    crop_url:            Mapped[str]  = mapped_column(String(2048), nullable=False, default="")
    representative_bbox: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    video: Mapped[Video] = relationship(back_populates="tracklets")
    embedding: Mapped["TrackletEmbedding | None"] = relationship(
        back_populates="tracklet", uselist=False, cascade="all, delete-orphan",
    )
    actions: Mapped[list["TrackletAction"]] = relationship(
        back_populates="tracklet", cascade="all, delete-orphan",
    )
    observations: Mapped[list["TrackletObservation"]] = relationship(
        back_populates="tracklet", cascade="all, delete-orphan",
        order_by="TrackletObservation.frame_index",
    )
    query_candidate_tracklets: Mapped[list["QueryCandidateTracklet"]] = relationship(
        back_populates="tracklet", cascade="all, delete-orphan",
    )
    evidence_tracklets: Mapped[list["EvidenceTracklet"]] = relationship(
        back_populates="tracklet", cascade="all, delete-orphan",
    )
    verified_object_tracklets: Mapped[list["VerifiedObjectTracklet"]] = relationship(
        back_populates="tracklet", cascade="all, delete-orphan",
    )


class TrackletEmbedding(Base):
    """SigLIP 2-So400m embedding per tracklet (1152-dim).

    The single embedding model in use. Same space as text queries → enables
    text-to-image search directly via cosine similarity on `siglip_embedding`.
    Vector is L2-normalized at write time so cosine = dot product.
    """
    __tablename__ = "tracklets_embeddings"
    __table_args__ = (
        UniqueConstraint("tracklet_id", name="uq_tracklets_embeddings_tracklet_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tracklet_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tracklets.tracklet_id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    siglip_embedding: Mapped[list | None] = mapped_column(
        PgVector(1152) if _PGVECTOR_AVAILABLE else JSON, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    tracklet: Mapped[Tracklet] = relationship(back_populates="embedding")


class TrackletAction(Base):
    """VideoMAE V2 action classification per tracklet (Section 20.2 row 9).

    Maps Kinetics-400 labels → TraceX simplified taxonomy.
    """
    __tablename__ = "tracklets_actions"
    __table_args__ = (
        UniqueConstraint("tracklet_id", name="uq_tracklets_actions_tracklet_id"),
        Index("ix_tracklets_actions_action", "action_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    tracklet_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tracklets.tracklet_id", ondelete="CASCADE"),
        nullable=False,
    )
    action_label: Mapped[str] = mapped_column(String(64), nullable=False)  # simplified taxonomy
    kinetics_label: Mapped[str | None] = mapped_column(String(128), nullable=True)  # raw Kinetics-400
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    tracklet: Mapped[Tracklet] = relationship(back_populates="actions")


class TrackletObservation(Base):
    """Per-frame bbox of a tracklet — used to render evidence clips with a
    bbox that follows the subject across frames (instead of one frozen
    `representative_bbox`).

    Saved at ingest time from the merged tracklet's `lt.observations`. One row
    per frame the tracker emitted for this identity.
    """
    __tablename__ = "tracklet_observations"
    __table_args__ = (
        UniqueConstraint("tracklet_id", "frame_index", name="uq_tracklet_observations_tracklet_frame"),
        Index("ix_tracklet_obs_tracklet_id", "tracklet_id"),
        Index("ix_tracklet_obs_tracklet_frame", "tracklet_id", "frame_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tracklet_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tracklets.tracklet_id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_second: Mapped[float] = mapped_column(Float, nullable=False)
    # [x1, y1, x2, y2] in original video pixel coordinates
    bbox: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    tracklet: Mapped[Tracklet] = relationship(back_populates="observations")


# ============================================================================
# Group 4: Query & Results
# ============================================================================

class QueryHistory(Base):
    """Query history — search queries (Section 20.2 row 10, Section 20.5 Luồng 2)."""
    __tablename__ = "query_history"
    __table_args__ = (
        UniqueConstraint("query_id", name="uq_query_history_query_id"),
        Index("ix_query_history_user_id", "user_id"),
        Index("ix_query_history_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    query_id: Mapped[str] = mapped_column(
        String(36), default=lambda: str(uuid.uuid4()), nullable=False,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    video_id: Mapped[str | None] = mapped_column(
        String(255), ForeignKey("videos.video_id", ondelete="SET NULL"), nullable=True,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(64), nullable=False, default="pending",

    )  # pending → searching → candidates_found → completed
    result_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_candidate_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True,
    )
    ai_job_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    query_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    owner: Mapped[User] = relationship(back_populates="query_history")
    video: Mapped[Video | None] = relationship(back_populates="query_history")
    candidates: Mapped[list["QueryCandidate"]] = relationship(
        back_populates="query", cascade="all, delete-orphan",
    )
    jobs: Mapped[list["QueryJob"]] = relationship(
        back_populates="query", cascade="all, delete-orphan",
    )
    spatiotemporal_groups: Mapped[list["SpatiotemporalGroup"]] = relationship(
        back_populates="query", cascade="all, delete-orphan",
    )


class QueryCandidate(Base):
    """Query candidate — result of a search query (Section 20.2 row 11, Section 20.5)."""
    __tablename__ = "query_candidates"
    __table_args__ = (
        UniqueConstraint("candidate_id", name="uq_query_candidates_candidate_id"),
        Index("ix_query_candidates_query_id", "query_id"),
        Index("ix_query_candidates_rank", "query_id", "rank_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    candidate_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    query_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_history.query_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Fusion scoring
    fusion_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    vector_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    text_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    spatiotemporal_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Ranking
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Selection
    is_selected: Mapped[bool] = mapped_column(nullable=False, default=False)
    selected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    candidate_key: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    preview_url: Mapped[str] = mapped_column(String(2048), nullable=False, default="")
    # Appearance summary (denormalized for fast display)
    appearance_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    gender: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    top_color: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    bottom_color: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    # BEV position (representative)
    bev_x: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bev_y: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Primary camera (denormalized for fast filter & display)
    primary_camera_id: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # Relationships
    query: Mapped[QueryHistory] = relationship(back_populates="candidates")
    tracklets: Mapped[list["QueryCandidateTracklet"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan",
    )
    spatiotemporal_groups: Mapped[list["SpatiotemporalGroup"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan",
    )
    evidence_videos: Mapped[list["EvidenceVideo"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan",
    )
    verified_object: Mapped["VerifiedObject | None"] = relationship(
        back_populates="candidate", uselist=False,
    )


class QueryCandidateTracklet(Base):
    """Junction: query candidate → contributing tracklets (Section 20.2 row 12).

    A query candidate is matched to one or more tracklets across cameras/frames.
    """
    __tablename__ = "query_candidate_tracklets"
    __table_args__ = (
        UniqueConstraint("candidate_id", "tracklet_id", name="uq_query_candidate_tracklets_pair"),
        Index("ix_qct_candidate", "candidate_id"),
        Index("ix_qct_tracklet", "tracklet_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    candidate_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("query_candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    )
    tracklet_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tracklets.tracklet_id", ondelete="CASCADE"),
        nullable=False,
    )
    match_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    match_type: Mapped[str] = mapped_column(String(32), nullable=False, default="vector")  # vector | text | spatiotemporal
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    candidate: Mapped[QueryCandidate] = relationship(back_populates="tracklets")
    tracklet: Mapped[Tracklet] = relationship(back_populates="query_candidate_tracklets")


class QueryJob(Base):
    """Query job — async worker queue item (Section 20.2 row 13)."""
    __tablename__ = "query_jobs"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_query_jobs_job_id"),
        Index("ix_query_jobs_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    job_id: Mapped[str] = mapped_column(
        String(36), default=lambda: str(uuid.uuid4()), nullable=False, unique=True,
    )
    query_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_history.query_id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
    )  # pending | running | completed | failed
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    query: Mapped[QueryHistory] = relationship(back_populates="jobs")


class SpatiotemporalGroup(Base):
    """Spatiotemporal group — camera transition cluster (Section 20.2 row 14).

    Groups consecutive tracklets that represent a person moving through camera topology.
    """
    __tablename__ = "spatiotemporal_groups"
    __table_args__ = (
        UniqueConstraint("group_id", name="uq_spatiotemporal_groups_group_id"),
        Index("ix_sg_query_id", "query_id"),
        Index("ix_sg_candidate_id", "candidate_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    group_id: Mapped[str] = mapped_column(
        String(36), default=lambda: str(uuid.uuid4()), nullable=False, unique=True,
    )
    query_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_history.query_id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("query_candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    )
    group_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="camera_transition",
    )  # camera_transition | temporal_cluster
    # Ordered list of tracklet IDs in this group (chronological)
    tracklet_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Camera path through the group
    camera_path: Mapped[list] = mapped_column(JSON, nullable=False, default=list)  # ["cam_01", "cam_03", ...]
    total_duration_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    query: Mapped[QueryHistory] = relationship(back_populates="spatiotemporal_groups")
    candidate: Mapped[QueryCandidate] = relationship(back_populates="spatiotemporal_groups")


# ============================================================================
# Group 5: Evidence & Verification
# ============================================================================

class EvidenceVideo(Base):
    """Evidence video — merged trace result video (Section 20.2 row 15, Section 20.6)."""
    __tablename__ = "evidence_videos"
    __table_args__ = (
        Index("ix_evidence_videos_candidate", "query_candidate_id"),
        Index("ix_evidence_videos_query", "query_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    query_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_history.query_id", ondelete="CASCADE"),
        nullable=False,
    )
    query_candidate_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("query_candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    )
    video_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    total_duration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    segment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    time_window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    time_window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trace_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    candidate: Mapped[QueryCandidate] = relationship(back_populates="evidence_videos")
    tracklets: Mapped[list["EvidenceTracklet"]] = relationship(
        back_populates="evidence_video", cascade="all, delete-orphan",
    )


class EvidenceTracklet(Base):
    """Evidence tracklet — one segment within an evidence video (Section 20.2 row 16, Section 20.6)."""
    __tablename__ = "evidence_tracklets"
    __table_args__ = (
        UniqueConstraint("evidence_video_id", "segment_order", name="uq_evidence_tracklets_order"),
        Index("ix_et_evidence_video", "evidence_video_id"),
        Index("ix_et_tracklet", "tracklet_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    evidence_video_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evidence_videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    tracklet_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tracklets.tracklet_id", ondelete="SET NULL"),
        nullable=True,
    )
    segment_order: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based ordering
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    time_range: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)  # {"start": iso, "end": iso}
    video_clip_url: Mapped[str] = mapped_column(String(2048), nullable=False, default="")
    thumbnail_url: Mapped[str] = mapped_column(String(2048), nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    evidence_video: Mapped[EvidenceVideo] = relationship(back_populates="tracklets")
    tracklet: Mapped[Tracklet | None] = relationship(back_populates="evidence_tracklets")


class VerifiedObject(Base):
    """Verified object — user-confirmed identity (Section 20.2 row 17, Section 20.10 step 5)."""
    __tablename__ = "verified_objects"
    __table_args__ = (
        Index("ix_vo_candidate", "candidate_id"),
        Index("ix_vo_user", "verified_by_user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    candidate_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("query_candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    )
    verified_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    is_correct: Mapped[bool] = mapped_column(nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    candidate: Mapped[QueryCandidate] = relationship(back_populates="verified_object", uselist=False)
    verified_by: Mapped[User | None] = relationship(back_populates="verified_objects")
    tracklets: Mapped[list["VerifiedObjectTracklet"]] = relationship(
        back_populates="verified_object", cascade="all, delete-orphan",
    )


class VerifiedObjectTracklet(Base):
    """Junction: verified object → contributing tracklets (Section 20.2 row 18)."""
    __tablename__ = "verified_objects_tracklets"
    __table_args__ = (
        UniqueConstraint("verified_object_id", "tracklet_id", name="uq_vot_pair"),
        Index("ix_vot_verified", "verified_object_id"),
        Index("ix_vot_tracklet", "tracklet_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    verified_object_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("verified_objects.id", ondelete="CASCADE"),
        nullable=False,
    )
    tracklet_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("tracklets.tracklet_id", ondelete="SET NULL"),
        nullable=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    verified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    verified_object: Mapped[VerifiedObject] = relationship(back_populates="tracklets")
    tracklet: Mapped[Tracklet | None] = relationship(back_populates="verified_object_tracklets")


# ============================================================================
# Internal: Queue Staging (Non-Spec — VinUni Storage Ingest)
# ============================================================================

class QueueVideoAsset(Base):
    """VinUni Storage ingest queue — non-spec staging table.

    Stores video queue metadata during the upload/ingest process.
    Deleted from this table after processing completes (moved to videos table).
    """
    __tablename__ = "queue_video_assets"
    __table_args__ = (
        UniqueConstraint("video_id", name="uq_queue_video_assets_video_id"),
        Index("ix_qva_processed", "processed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,)
    video_id: Mapped[str] = mapped_column(String(255), nullable=False,)
    camera_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    queue_position: Mapped[int] = mapped_column(Integer, nullable=False,)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False, default="google_drive")
    available_link_video: Mapped[str] = mapped_column(String(2048), nullable=False)
    available_link_metadata: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    drive_video_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    drive_metadata_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    local_video_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    local_metadata_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    raw_video_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True,)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# =============================================================================
# Backward-compatibility aliases — match old class names used in existing code.
# The new spec-compliant class is the primary definition; these are thin aliases.
# =============================================================================
# OLD: VideoAsset → NEW: Video (same table "videos")
VideoAsset = Video

# OLD: VideoQuery → NEW: QueryHistory (renamed table "query_history")
# NOTE: QueryHistory uses "query_id" as FK; VideoQuery used "video_id" as FK.
# Existing code using VideoQuery.video foreign key needs migration.
VideoQuery = QueryHistory

# OLD: PersonCandidate → NEW: QueryCandidate (renamed table "query_candidates")
PersonCandidate = QueryCandidate
