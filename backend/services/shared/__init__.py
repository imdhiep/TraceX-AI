"""Shared package — exports models, database, and core utilities."""

from .models import (
    Base,
    # Users
    User,
    # Camera topology
    Camera,
    CameraEdge,
    CameraSettings,
    CameraZone,
    # Video & Tracklet (AI)
    Video,
    Tracklet,
    TrackletAction,
    TrackletEmbedding,
    TrackletObservation,
    # Query & Results
    EvidenceTracklet,
    EvidenceVideo,
    QueryCandidate,
    QueryCandidateTracklet,
    QueryHistory,
    QueryJob,
    SpatiotemporalGroup,
    # Verification
    VerifiedObject,
    VerifiedObjectTracklet,
    # Queue staging (non-spec)
    QueueVideoAsset,
    # Backward-compat aliases (old class names still used in existing code)
    PersonCandidate,  # = QueryCandidate
    VideoAsset,      # = Video
    VideoQuery,      # = QueryHistory
)

__all__ = [
    "Base",
    # Group 1
    "User",
    # Group 2
    "Camera",
    "CameraZone",
    "CameraEdge",
    "CameraSettings",
    # Group 3
    "Video",
    "Tracklet",
    "TrackletEmbedding",
    "TrackletAction",
    "TrackletObservation",
    # Group 4
    "QueryHistory",
    "QueryCandidate",
    "QueryCandidateTracklet",
    "QueryJob",
    "SpatiotemporalGroup",
    # Group 5
    "EvidenceVideo",
    "EvidenceTracklet",
    "VerifiedObject",
    "VerifiedObjectTracklet",
    # Queue
    "QueueVideoAsset",
    # Aliases
    "PersonCandidate",
    "VideoAsset",
    "VideoQuery",
]
