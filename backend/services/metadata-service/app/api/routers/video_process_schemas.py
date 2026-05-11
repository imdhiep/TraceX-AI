"""Pydantic schemas for metadata-service API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ProcessVideoRequest(BaseModel):
    video_id: str = Field(..., description="Unique video identifier")
    camera_id: Optional[str] = Field(None, description="Camera label, e.g. cam_01")
    video_path: str = Field(..., description="Local filesystem path to the video file")
    drive_file_id: Optional[str] = Field(None, description="Google Drive file ID if remote")
    source_filename: Optional[str] = Field(None, description="Original filename")
    sample_interval: Optional[int] = Field(
        15, description="Frame interval for person detection sampling"
    )


class ProcessVideoStreamRequest(BaseModel):
    """Request body for the streaming upload endpoint.

    Video bytes are sent as a multipart/form-data file field named "video".
    """
    video_id: str = Field(..., description="Unique video identifier")
    camera_id: Optional[str] = Field(None, description="Camera label, e.g. cam_01")
    source_filename: Optional[str] = Field(None, description="Original filename for suffix detection")
    sample_interval: Optional[int] = Field(15, description="Frame interval for person detection")


class BatchVideoEntry(BaseModel):
    """One video entry inside a batch request."""
    video_id: str = Field(..., description="Unique video identifier")
    camera_id: Optional[str] = Field(None, description="Camera label, e.g. cam_01")
    video_path: str = Field(..., description="Local filesystem path to the video file")
    source_filename: Optional[str] = Field(None, description="Original filename")
    sample_interval: int = Field(15, description="Frame interval for person detection")


class BatchProcessRequest(BaseModel):
    """
    Batch processing request.

    All videos in a batch share the same timestamp (e.g. 50 cameras at 11:00).
    The pipeline:
      1. Process each video independently with the per-video tracking pipeline
      2. Aggregate all resulting tracklets into one batch response
      3. Preserve per-tracklet SigLIP2 image embeddings for text-image search

    Usage:
      - queue_worker groups videos by timestamp, sends 1 batch per timestamp
      - Each batch processes up to 50 cameras simultaneously
    """
    videos: list[BatchVideoEntry] = Field(
        ..., min_length=1, max_length=100,
        description="Videos in this batch (same timestamp, up to 100 cameras)"
    )
    batch_id: Optional[str] = Field(
        None,
        description="Optional batch identifier for tracing. "
                    "Defaults to timestamp-based auto-generated ID."
    )


class TrackletResult(BaseModel):
    tracklet_id: str
    video_id: str
    camera_id: str
    track_id: int
    start_time: float = 0.0
    end_time: float = 0.0
    quality_score: float = 0.0
    gender: str = "unknown"
    age_range: str = "unknown"
    upper_clothing_color: str = "unknown"
    lower_clothing_color: str = "unknown"
    shoes_color: str = "unknown"
    appearance_summary: str = ""
    crop_url: str = ""
    representative_bbox: list[int] = [0, 0, 0, 0]
    action: str = "standing"
    action_confidence: float = 0.0
    kinetics_label: str = ""
    occlusion_score: float = 0.0
    hat_color: str = "unknown"
    bag_type: str = "unknown"
    is_wearing_mask: str = "unknown"
    hair_style: str = "unknown"
    hair_color: str = "unknown"
    # Per-attribute confidence scores [0, 1] (None = not yet extracted)
    gender_conf: Optional[float] = None
    upper_clothing_conf: Optional[float] = None
    lower_clothing_conf: Optional[float] = None
    shoes_conf: Optional[float] = None
    accessory_conf: Optional[float] = None
    age_range_conf: Optional[float] = None
    hat_color_conf: Optional[float] = None
    bag_type_conf: Optional[float] = None
    mask_conf: Optional[float] = None
    hair_style_conf: Optional[float] = None
    hair_color_conf: Optional[float] = None
    # Open-vocabulary VLM metadata (Qwen2-VL-7B-Instruct)
    upper_clothing_desc: Optional[str] = None
    upper_clothing_color: Optional[str] = None
    upper_clothing_type: Optional[str] = None
    upper_clothing_conf: Optional[float] = None
    lower_clothing_desc: Optional[str] = None
    lower_clothing_color: Optional[str] = None
    lower_clothing_type: Optional[str] = None
    lower_clothing_conf: Optional[float] = None
    shoes_desc: Optional[str] = None
    shoes_type: Optional[str] = None
    bag_desc: Optional[str] = None
    bag_presence: Optional[str] = None
    bag_conf: Optional[float] = None
    hat_desc: Optional[str] = None
    hat_presence: Optional[str] = None
    hat_type: Optional[str] = None
    hat_conf: Optional[float] = None
    # SigLIP2 image embedding for text-image search (1152-dim)
    siglip_embedding: list[float] = Field(default_factory=list)
    # Cross-camera: which cameras/frames contributed to this tracklet
    contributing_cameras: list[str] = Field(default_factory=list)
    contributing_video_ids: list[str] = Field(default_factory=list)


class ProcessVideoResponse(BaseModel):
    video_id: str
    camera_id: str
    tracklets: list[TrackletResult] = Field(default_factory=list)
    total_detections: int = 0
    processing_time_s: float = 0.0


class BatchProcessResponse(BaseModel):
    """Response for batch cross-camera processing."""
    batch_id: str
    n_videos: int
    n_cameras: int
    total_detections: int
    n_tracklets: int
    tracklets: list[TrackletResult]
    processing_time_s: float
    camera_stats: dict[str, int] = Field(
        default_factory=dict,
        description="Per-camera detection counts, e.g. {'cam_01': 12, 'cam_02': 8}"
    )
