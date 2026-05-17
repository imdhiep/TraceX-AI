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
    bev_max_dist: Optional[float] = Field(
        1.5, description="Maximum BEV distance (metres) for MCBLT association"
    )
    use_mtmc_calibration: bool = Field(
        False, description="Use MTMC calibration file instead of deriving homography"
    )


class ProcessVideoStreamRequest(BaseModel):
    """Request body for the streaming upload endpoint.

    Video bytes are sent as a multipart/form-data file field named "video".
    """
    video_id: str = Field(..., description="Unique video identifier")
    camera_id: Optional[str] = Field(None, description="Camera label, e.g. cam_01")
    source_filename: Optional[str] = Field(None, description="Original filename for suffix detection")
    sample_interval: Optional[int] = Field(15, description="Frame interval for person detection")
    bev_max_dist: Optional[float] = Field(1.5, description="MCBLT max BEV distance in metres")


class BatchVideoEntry(BaseModel):
    """One video entry inside a batch request."""
    video_id: str = Field(..., description="Unique video identifier")
    camera_id: Optional[str] = Field(None, description="Camera label, e.g. cam_01")
    video_path: str = Field(..., description="Local filesystem path to the video file")
    source_filename: Optional[str] = Field(None, description="Original filename")
    sample_interval: int = Field(15, description="Frame interval for person detection")
    bev_max_dist: float = Field(1.5, description="MCBLT max BEV distance in metres")


class BatchProcessRequest(BaseModel):
    """
    Batch cross-camera processing request.

    All videos in a batch share the same timestamp (e.g. 50 cameras at 11:00).
    The pipeline:
      1. Load frames + detect persons in each video independently
      2. Project all detections to BEV
      3. MCBLT Hungarian cross-camera association (ONE call across all cameras)
      4. PersonViT-S Re-ID embeddings + Qwen2.5-VL metadata + VideoMAE V2 actions
         per unified cross-camera tracklet (SigLIP2 image encoder for text-image search)
      5. Return unified tracklets (global tracklet IDs across cameras)

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
    """One tracklet — schema mirrors the refactored DB. All confidence values
    are derived from Qwen2.5-VL token logprobs (geometric mean of P(token | ctx))."""
    tracklet_id: str
    video_id: str
    camera_id: str
    track_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    quality_score: float = 0.0

    # Demographic
    gender: str = "unknown"
    gender_conf: Optional[float] = None
    age_range: str = "unknown"
    age_range_conf: Optional[float] = None

    # Upper / lower / shoes / bag / hat (color + type share one *_conf; desc has its own)
    upper_color: Optional[str] = None
    upper_type: Optional[str] = None
    upper_desc: Optional[str] = None
    upper_conf: Optional[float] = None
    upper_desc_conf: Optional[float] = None

    lower_color: Optional[str] = None
    lower_type: Optional[str] = None
    lower_desc: Optional[str] = None
    lower_conf: Optional[float] = None
    lower_desc_conf: Optional[float] = None

    shoes_color: Optional[str] = None
    shoes_type: Optional[str] = None
    shoes_desc: Optional[str] = None
    shoes_conf: Optional[float] = None
    shoes_desc_conf: Optional[float] = None

    bag_presence: Optional[str] = None
    bag_type: Optional[str] = None
    bag_desc: Optional[str] = None
    bag_conf: Optional[float] = None
    bag_desc_conf: Optional[float] = None

    hat_presence: Optional[str] = None
    hat_color: Optional[str] = None
    hat_type: Optional[str] = None
    hat_desc: Optional[str] = None
    hat_conf: Optional[float] = None
    hat_desc_conf: Optional[float] = None

    # Mask / hair
    mask_presence: str = "unknown"
    mask_conf: Optional[float] = None
    hair_style: str = "unknown"
    hair_style_conf: Optional[float] = None
    hair_color: str = "unknown"
    hair_color_conf: Optional[float] = None

    # Summary
    appearance_summary: str = ""
    appearance_summary_conf: Optional[float] = None

    # Spatial / crop
    crop_url: str = ""
    representative_bbox: list[int] = [0, 0, 0, 0]
    bev_x: float = 0.0
    bev_y: float = 0.0

    # Embeddings (DB)
    siglip_embedding: list[float] = Field(default_factory=list)
    reid_embedding: list[float] = Field(default_factory=list)
    reid_model_version: str = ""

    # Action (VideoMAE)
    action: str = "standing"
    action_confidence: float = 0.0
    kinetics_label: str = ""

    # Per-frame bbox timeline. Each entry: {frame_index, timestamp_second, bbox: [x1,y1,x2,y2], confidence}.
    # Saved into tracklet_observations table so trace-service can render
    # evidence clips with a moving bbox.
    observations: list[dict] = Field(default_factory=list)


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
