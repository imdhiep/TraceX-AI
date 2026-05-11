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
]
