"""
Tracking pipeline: BodyPartAdaptiveTracker + post-hoc TrackletFragmentMerger.

BodyPartAdaptiveTracker uses head-dominant or foot-dominant cost depending on
whether the head region is estimated to be visible in each detection bbox.
TrackletFragmentMerger re-joins fragments of the same person after feature
extraction using SigLIP2 / PersonViT cosine similarity.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import av
import cv2
import numpy as np


# ── Vectorized geometry helpers ───────────────────────────────────────────────

def _head_bbox_arr(bboxes: np.ndarray, head_ratio: float, shrink_x: float) -> np.ndarray:
    """Vectorized head bbox for [N, 4] array → [N, 4]."""
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w = x2 - x1
    return np.stack([x1 + w * shrink_x, y1, x2 - w * shrink_x, y1 + (y2 - y1) * head_ratio], axis=1)


def _foot_points_arr(bboxes: np.ndarray) -> np.ndarray:
    """Bottom-center points for [N, 4] array → [N, 2]."""
    return np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2.0, bboxes[:, 3]], axis=1)


def _batch_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU: a [M,4], b [N,4] → [M,N]."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_second: float
    image: np.ndarray


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    timestamp_second: float
    bbox: tuple[int, int, int, int]   # x1 y1 x2 y2
    confidence: float
    laplacian_score: float
    crop_bgr: Optional[np.ndarray] = None
    # Marker for downstream code (embedding pool, ReID rep selection) to optionally
    # down-weight this observation. NOT yet consumed — currently set by the
    # detection stage but no consumer reads it. Wire in carefully: in hospital
    # CCTV, edge-clipped/non-upright crops are common and tracker must still keep
    # the ID through them, so any consumer should down-weight rather than reject.
    is_low_quality_crop: bool = False


@dataclass(frozen=True)
class TrackletObservation:
    frame_index: int
    timestamp_second: float
    bbox: tuple[int, int, int, int]
    confidence: float
    laplacian_score: float
    crop_bgr: Optional[np.ndarray] = None
    is_low_quality_crop: bool = False


@dataclass(frozen=True)
class LocalTracklet:
    video_id: str
    camera_id: Optional[str]
    track_id: str
    observations: tuple[TrackletObservation, ...]


@dataclass(frozen=True)
class TrackletQualityResult:
    accepted: bool
    average_confidence: float
    average_laplacian: float
    frame_count: int
    frame_density: float
    duration_seconds: float
    rejection_reason: Optional[str]


# ── Scalar geometry helpers ───────────────────────────────────────────────────

def _bbox_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(bbox: tuple) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _crop_from_bbox(image: np.ndarray, bbox: tuple, *, padding_ratio: float = 0.08) -> Optional[np.ndarray]:
    """Crop image at bbox, optionally expanded by padding_ratio on each side
    (clamped to image bounds). Padding gives ReID models a bit of context
    (shoulders, hair outline) which improves embedding quality. Set
    padding_ratio=0.0 to disable.
    """
    h, w = image.shape[:2]
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))
    x1 = max(0, bbox[0] - pad_x)
    y1 = max(0, bbox[1] - pad_y)
    x2 = min(w, bbox[2] + pad_x)
    y2 = min(h, bbox[3] + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


def _crop_laplacian_variance(crop_bgr: Optional[np.ndarray]) -> float:
    """Variance-of-Laplacian on grayscale crop. Higher = sharper.
    Returns 0.0 for missing / degenerate crops so the quality scorer rejects them.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(round(cv2.Laplacian(gray, cv2.CV_64F).var(), 6))


def _is_low_quality_crop(
    bbox: tuple,
    frame_h: int,
    frame_w: int,
    *,
    # Balanced for AICity-style demo (mostly walking persons, aspect ~2.0-3.0)
    # while leaving headroom for slight pose variation and bending. Flag only
    # when aspect drops below 0.9 (clearly non-upright → likely half body) or
    # exceeds 4.5 (thin sliver). Hospital CCTV with many seated patients would
    # need a lower min_aspect (~0.7) — make per-camera.
    min_aspect: float = 0.8,
    max_aspect: float = 4.5,
    edge_margin_px: int = 4,
) -> bool:
    """True when the bbox is unreliable for ReID embedding even though it's
    still useful for tracker continuity. Triggers on extreme aspect ratios
    (truly degenerate boxes) and frame-edge clipping (person cropped by FOV)."""
    bw = max(bbox[2] - bbox[0], 1)
    bh = max(bbox[3] - bbox[1], 1)
    aspect = bh / bw
    if aspect < min_aspect or aspect > max_aspect:
        return True
    if bbox[0] < edge_margin_px or bbox[1] < edge_margin_px:
        return True
    if bbox[2] > frame_w - edge_margin_px or bbox[3] > frame_h - edge_margin_px:
        return True
    return False


def _head_bbox(bbox: tuple, head_ratio: float = 0.35, shrink_x: float = 0.08) -> tuple:
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w, h = x2 - x1, y2 - y1
    return (
        int(x1 + w * shrink_x),
        int(y1),
        int(x2 - w * shrink_x),
        int(y1 + h * head_ratio),
    )


def _foot_point(bbox: tuple) -> tuple[float, float]:
    """Bottom-center of bbox — stable anchor when head is occluded."""
    return ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def _head_visible(bbox: tuple, min_aspect: float = 1.1, min_height: int = 30) -> bool:
    """
    Heuristic: head is likely visible when bbox is taller than wide (aspect ≥ min_aspect)
    and the bbox is tall enough to contain a head region.

    Fails gracefully for partial-body detections (torso-only, legs-only) where
    the 35%-top-of-bbox head estimate would be meaningless.
    """
    bw = max(bbox[2] - bbox[0], 1)
    bh = max(bbox[3] - bbox[1], 1)
    return (bh / bw) >= min_aspect and bh >= min_height


# ── VideoFrameSampler ─────────────────────────────────────────────────────────

class VideoFrameSampler:
    """Sample frames at fixed FPS using PyAV (multi-threaded HEVC/H.264 decode).

    PyAV → FFmpeg lets HEVC decode go multi-threaded (thread_type="AUTO"), which
    is the main win over cv2.VideoCapture's single-thread Python loop on H.265
    footage. Frame selection is index-based to avoid float drift on long videos.
    """

    def __init__(self, sample_fps: int = 4):
        self.sample_fps = max(1, sample_fps)

    def sample(self, video_path: str) -> tuple[SampledFrame, ...]:
        try:
            container = av.open(str(video_path))
        except (av.AVError, FileNotFoundError) as exc:
            raise FileNotFoundError(f"Cannot open video: {video_path}") from exc

        try:
            if not container.streams.video:
                raise FileNotFoundError(f"No video stream: {video_path}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"  # multi-threaded HEVC decode

            avg_rate = stream.average_rate or stream.base_rate
            source_fps = float(avg_rate) if avg_rate else float(self.sample_fps)
            if source_fps <= 0:
                source_fps = float(self.sample_fps)

            # Index-based stride avoids cumulative float drift on long videos.
            # stride = how many source frames per sampled frame.
            stride = max(source_fps / self.sample_fps, 1.0)
            next_emit_src = 0.0
            src_idx = 0
            sampled_idx = 0
            frames: list[SampledFrame] = []

            for frame in container.decode(stream):
                if src_idx + 1e-9 >= next_emit_src:
                    # to_ndarray("bgr24") returns a fresh contiguous buffer →
                    # no .copy() needed (cv2 path used .copy() defensively).
                    img = frame.to_ndarray(format="bgr24")
                    ts = src_idx / source_fps
                    frames.append(SampledFrame(
                        frame_index=sampled_idx,
                        timestamp_second=round(ts, 6),
                        image=img,
                    ))
                    sampled_idx += 1
                    next_emit_src = sampled_idx * stride
                src_idx += 1
        finally:
            container.close()

        return tuple(frames)


# ── BodyPartAdaptiveTracker ───────────────────────────────────────────────────

class _PersonKalman:
    """8-D Kalman filter for a single person bbox, SORT/DeepSORT style.

    State: [u, v, s, r, du, dv, ds, dr] where
        u, v  = bbox center
        s     = bbox area (scale)
        r     = aspect ratio (w/h)
        d.    = corresponding velocity (assumed near-constant)
    Measurement: [u, v, s, r].

    Why this and not a hand-rolled linear predictor:
      • Process noise covariance scales with bbox height → a person far from
        the camera has more positional uncertainty per frame, matching reality.
      • Velocity is smoothed across observations rather than being a delta
        between the last two frames (which is very noisy at 4 fps).
      • Mahalanobis distance from predicted state gives a principled "this
        detection is too far given my uncertainty" gate that adapts to how
        well the track has been tracked.

    Implementation kept dependency-free: pure numpy, ~5 lines per step.
    """

    # Standard SORT noise coefficients (Bewley et al. 2016).
    _STD_POS = 1.0 / 20.0
    _STD_VEL = 1.0 / 160.0

    def __init__(self, bbox: tuple):
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        s = w * h
        r = w / h
        # Initial mean and covariance.
        self.mean = np.array([u, v, s, r, 0, 0, 0, 0], dtype=np.float64)
        std = np.array([
            2 * self._STD_POS * h,
            2 * self._STD_POS * h,
            2 * s * 0.05,
            1e-2,
            10 * self._STD_VEL * h,
            10 * self._STD_VEL * h,
            10 * s * 0.05,
            1e-5,
        ])
        self.cov = np.diag(std ** 2)

        # Constant-velocity transition F (dt = 1 frame; gap is handled by
        # iterating predict() the correct number of times).
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        # Measurement matrix H (we observe position only)
        self._H = np.zeros((4, 8))
        for i in range(4):
            self._H[i, i] = 1.0

    def predict(self, n_steps: int = 1) -> np.ndarray:
        """Advance by n_steps frames. Returns predicted measurement [u, v, s, r]."""
        if n_steps <= 0:
            return self._H @ self.mean
        # Compose F^n_steps for n>1 (so noise grows correctly per step).
        for _ in range(n_steps):
            h = max(np.sqrt(max(self.mean[2] / max(self.mean[3], 1e-6), 1.0)), 1.0)
            std_pos = self._STD_POS * h
            std_vel = self._STD_VEL * h
            Q = np.diag(np.array([
                std_pos, std_pos, 1e-2 * abs(self.mean[2]) + 1e-2, 1e-2,
                std_vel, std_vel, 1e-5 * abs(self.mean[2]) + 1e-5, 1e-5,
            ]) ** 2)
            self.mean = self._F @ self.mean
            self.cov = self._F @ self.cov @ self._F.T + Q
        return self._H @ self.mean

    def update(self, bbox: tuple) -> None:
        """Fold a new bbox observation into the state."""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        z = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w * h, w / h])

        std = np.array([self._STD_POS * h, self._STD_POS * h,
                        1e-1 * abs(self.mean[2]) + 1e-1, 1e-1])
        R = np.diag(std ** 2)

        y = z - self._H @ self.mean
        S = self._H @ self.cov @ self._H.T + R
        try:
            K = self.cov @ self._H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # singular → skip update, keep prior
        self.mean = self.mean + K @ y
        self.cov = (np.eye(8) - K @ self._H) @ self.cov

    def predicted_bbox(self, n_steps: int = 0) -> tuple:
        """Predict ahead by n_steps without mutating state. Returns (x1,y1,x2,y2)."""
        if n_steps <= 0:
            u, v, s, r = self.mean[:4]
        else:
            # Project without altering state — clone mean
            m = self.mean.copy()
            for _ in range(n_steps):
                m = self._F @ m
            u, v, s, r = m[:4]
        s = max(float(s), 1.0)
        r = max(float(r), 1e-3)
        w = float(np.sqrt(s * r))
        h = float(s / max(w, 1e-3))
        return (u - w / 2, v - h / 2, u + w / 2, v + h / 2)

    def mahalanobis(self, bbox: tuple, *, position_only: bool = True) -> float:
        """Squared Mahalanobis distance between predicted state and a det bbox.
        position_only=True uses just (u,v) — robust to scale errors that occur
        when detector returns a slightly differently-sized box. Returns a
        single scalar in [0, ∞). 9.21 is the chi-square 99% level for 2 DoF."""
        x1, y1, x2, y2 = bbox
        u_obs = (x1 + x2) / 2.0
        v_obs = (y1 + y2) / 2.0
        if position_only:
            pred_uv = self.mean[:2]
            S = self.cov[:2, :2] + np.eye(2) * 1.0  # add small jitter
            d = np.array([u_obs - pred_uv[0], v_obs - pred_uv[1]])
            try:
                return float(d @ np.linalg.inv(S) @ d)
            except np.linalg.LinAlgError:
                return float("inf")
        z = np.array([u_obs, v_obs, (x2 - x1) * (y2 - y1), (x2 - x1) / max(y2 - y1, 1)])
        innov = z - self._H @ self.mean
        S = self._H @ self.cov @ self._H.T + np.eye(4)
        try:
            return float(innov @ np.linalg.inv(S) @ innov)
        except np.linalg.LinAlgError:
            return float("inf")


class BodyPartAdaptiveTracker:
    """
    ByteTrack-style tracker with adaptive cost based on head visibility.

    When both the tracked person and the incoming detection have visible heads
    (bbox aspect ≥ min_head_aspect and height ≥ min_head_height), the matcher
    uses a head-dominant cost formula:

        cost = 0.35 * head_center + 0.25 * full_bbox_IoU + 0.25 * foot_point + 0.15 * velocity

    When the head is estimated to be occluded (short / partial bbox), the
    matcher switches to a foot-dominant formula:

        cost = 0.40 * foot_point + 0.35 * full_bbox_IoU + 0.25 * velocity

    Spatial gating blocks pairs only when BOTH the head-center distance AND the
    foot-point distance exceed their respective thresholds, so a track is never
    dropped solely because the head left the expected region.
    """

    def __init__(
        self,
        track_thresh: float = 0.40,
        low_thresh: float = 0.10,
        new_track_threshold: float = 0.45,
        # P1.5 — relaxed from 0.65 → 0.75. With G4 post-hoc split active, the
        # live tracker no longer needs to break aggressively at the matching
        # stage; G4 is more accurate at the same job (it sees full track
        # history). 0.75 is the middle ground between legacy 0.80 (too loose)
        # and 0.65 (too aggressive, hurts purity when combined with G4).
        max_match_cost: float = 0.75,
        # Buffer reactivation kept at 0.80 — same-ID rescue is appearance-free
        # but already guarded by IoU floor (min_buffer_iou=0.30).
        max_buffer_match_cost: float = 0.80,
        max_head_center_distance: float = 120.0,
        max_foot_distance: float = 150.0,
        max_predicted_distance: float = 180.0,
        track_buffer: int = 20,
        max_buffer_frames: int = 300,
        head_ratio: float = 0.35,
        shrink_x: float = 0.08,
        min_head_aspect: float = 1.1,
        min_head_height: int = 30,
        min_track_frames: int = 2,
        min_track_density: float = 0.05,
        min_buffer_iou: float = 0.30,
        # Fix A — do not require short-gap IoU. At 3 FPS a fast walker can have
        # IoU=0 between adjacent sampled frames; G2 margin + center-jump guard
        # are the handoff protection, so the IoU floor only fractures tracks.
        min_active_iou_short_gap: float = 0.00,
        short_gap_frames: int = 2,                # what "short gap" means in frames
        max_lowconf_match_cost: float = 0.50,     # low-conf rescue gets stricter cost cap
        max_center_jump_ratio: float = 2.0,       # bbox-center jump > N × bbox_h flags handoff
        # B — bounded phantom extension. When a live track misses a detection,
        # keep a shadow Kalman state for a tiny gap (default: exactly 1 sampled
        # frame) without adding synthetic observations to the tracklet.
        # Bench cam_0002: max=1 improved purity by ~0.0099 and reduced IDsw
        # ~30%; max>=2 drifted away from the true position and hurt purity.
        max_phantom_frames: int = 1,
        # P3 — Kalman filter for live prediction during matching.
        # DISABLED by default: benchmark on camera_0002 showed the live KF
        # actually HURT purity (67.5% → 72.9% but with 9× cost), while G4
        # (post-hoc split below) is what really matters. Kept here as opt-in
        # so we can revisit if a per-camera benchmark says otherwise.
        use_kalman: bool = False,
        kalman_gate_chi2: float = 9.21,
        # G2 — discriminative gate. After Hungarian assigns det i to track j
        # with cost c_ij, also compute the second-best track cost c_ik (k≠j).
        # If c_ik - c_ij < discriminative_margin the match is ambiguous (two
        # tracks compete almost equally for the same detection — classic
        # crossing signature) and we REJECT the match. Det becomes unmatched,
        # tracker may break into a new fragment instead of swapping ID.
        # Sweep on camera_0002 (3 FPS): 0.00 → 0.20 raises purity 92.7 → 97.7
        # at the cost of 33% more fragments (which downstream appearance
        # merger rejoins). Set to 0.0 to disable.
        discriminative_margin: float = 0.20,
        # G4 — post-hoc Kalman split. After each tracklet completes, re-run
        # the Kalman filter over its observations and split it at any frame
        # whose innovation exceeds split_chi2. This catches occlusion handoffs
        # the live tracker missed (the new bbox passed the live gate because
        # uncertainty had grown, but the bbox center actually jumped far).
        # Disabled when use_kalman=False. Set split_chi2 high (e.g. 25) — too
        # low fractures legitimate fast-movement tracks.
        split_post_hoc: bool = True,
        split_chi2: float = 25.0,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_threshold = new_track_threshold
        self.max_match_cost = max_match_cost
        self.max_buffer_match_cost = max_buffer_match_cost
        self.max_head_center_distance = max_head_center_distance
        self.max_foot_distance = max_foot_distance
        self.max_predicted_distance = max_predicted_distance
        self.track_buffer = track_buffer
        self.max_buffer_frames = max_buffer_frames
        self.head_ratio = head_ratio
        self.shrink_x = shrink_x
        self.min_head_aspect = min_head_aspect
        self.min_head_height = min_head_height
        self.min_track_frames = min_track_frames
        self.min_track_density = min_track_density
        self.min_buffer_iou = min_buffer_iou
        self.min_active_iou_short_gap = min_active_iou_short_gap
        self.short_gap_frames = short_gap_frames
        self.max_lowconf_match_cost = max_lowconf_match_cost
        self.max_center_jump_ratio = max_center_jump_ratio
        self.max_phantom_frames = max(0, int(max_phantom_frames))
        self.use_kalman = use_kalman
        self.kalman_gate_chi2 = kalman_gate_chi2
        self.split_post_hoc = split_post_hoc
        self.split_chi2 = split_chi2
        self.discriminative_margin = discriminative_margin
        self._reset()

    def _reset(self) -> None:
        self.active: dict[str, list[TrackletObservation]] = {}
        self.active_last_bbox: dict[str, tuple] = {}
        self.active_last_frame: dict[str, int] = {}
        self.buffer: dict[str, list[TrackletObservation]] = {}
        self.buffer_last_bbox: dict[str, tuple] = {}
        self.buffer_entry_frame: dict[str, int] = {}
        # P3: Kalman filter per active track. Reset on track expiry; tracks
        # moved to buffer drop their KF (re-seeded on reactivation).
        self.kf: dict[str, _PersonKalman] = {}
        # B: Shadow Kalman state for one-frame phantom extension. These do not
        # create observations and do not mutate active_last_bbox.
        self._phantom_kf: dict[str, _PersonKalman] = {}
        self._phantom_streak: dict[str, int] = {}
        self._real_last_frame: dict[str, int] = {}
        self.next_id = 1

    def _hbox(self, bbox: tuple) -> tuple:
        return _head_bbox(bbox, self.head_ratio, self.shrink_x)

    def _build_state(self, obs: list[TrackletObservation], last_bbox: tuple, target_fk: int,
                     tid: Optional[str] = None):
        """
        Returns (hbox, head_center, predicted_center, full_bbox, foot_point, head_visible).
        Predicted_center comes from the Kalman filter if available, else falls
        back to a 2-frame linear velocity.
        """
        state_bbox = last_bbox
        phantom_active = False
        if tid is not None:
            streak = self._phantom_streak.get(tid, 0)
            pkf = self._phantom_kf.get(tid)
            if pkf is not None and 0 < streak <= self.max_phantom_frames:
                steps = max(target_fk - self.active_last_frame.get(tid, target_fk), 0)
                state_bbox = pkf.predicted_bbox(n_steps=steps)
                phantom_active = True

        hbox = self._hbox(state_bbox)
        hc   = _center(hbox)
        fc   = _foot_point(state_bbox)
        hv   = _head_visible(state_bbox, self.min_head_aspect, self.min_head_height)

        if phantom_active:
            return hbox, hc, hc, state_bbox, fc, hv

        # P3 Kalman branch
        if self.use_kalman and tid is not None and tid in self.kf:
            steps = max(target_fk - obs[-1].frame_index, 0) if obs else 0
            steps = min(steps, self.track_buffer * 2)
            pred_bbox = self.kf[tid].predicted_bbox(n_steps=steps)
            # We compare on head-center (cost matrix expectation) — derive from
            # predicted bbox the same way an observed bbox would.
            pred_hbox = self._hbox(pred_bbox)
            pred = _center(pred_hbox)
            return hbox, hc, pred, state_bbox, fc, hv

        if len(obs) < 2:
            return hbox, hc, hc, state_bbox, fc, hv

        prev_hbox = self._hbox(obs[-2].bbox)
        pc = _center(prev_hbox)
        delta = max(obs[-1].frame_index - obs[-2].frame_index, 1)
        vx = (hc[0] - pc[0]) / delta
        vy = (hc[1] - pc[1]) / delta
        steps = min(max(target_fk - obs[-1].frame_index, 0), self.track_buffer * 2)
        pred = (hc[0] + vx * steps, hc[1] + vy * steps)
        return hbox, hc, pred, state_bbox, fc, hv

    def _mark_real_detection(self, tid: str, bbox: tuple, fk: int, *, seed_phantom: bool = False) -> None:
        """Record a real detection and refresh the bounded phantom state."""
        if seed_phantom or tid not in self._phantom_kf:
            self._phantom_kf[tid] = _PersonKalman(bbox)
        else:
            pkf = self._phantom_kf[tid]
            last_state_frame = self.active_last_frame.get(tid, self._real_last_frame.get(tid, fk))
            steps = max(fk - last_state_frame, 0)
            if steps > 0:
                pkf.predict(n_steps=steps)
            pkf.update(bbox)
        self._phantom_streak[tid] = 0
        self._real_last_frame[tid] = fk

    def _phantom_extend_unmatched(self, track_ids: set[str], fk: int) -> None:
        """Advance unmatched active tracks without writing fake observations."""
        if self.max_phantom_frames <= 0:
            return
        for tid in track_ids:
            if tid not in self.active:
                continue
            streak = self._phantom_streak.get(tid, 0) + 1
            self._phantom_streak[tid] = streak
            if streak > self.max_phantom_frames:
                continue
            pkf = self._phantom_kf.get(tid)
            if pkf is None:
                pkf = _PersonKalman(self.active_last_bbox[tid])
                self._phantom_kf[tid] = pkf
            last_state_frame = self.active_last_frame.get(tid, self._real_last_frame.get(tid, fk))
            steps = max(fk - last_state_frame, 0)
            if steps > 0:
                pkf.predict(n_steps=steps)
            self.active_last_frame[tid] = fk

    def _build_cost_matrix(
        self,
        det_bboxes: list[tuple],
        states: dict,
        track_ids: list[str],
        *,
        gap_frames: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Adaptive cost matrix [M, N] with spatial gating applied (gated cells = 1e9).

        P1 hard gates (applied after the soft-cost blend):
          • IoU floor on short-gap matches (gap ≤ short_gap_frames frames):
              if last seen this frame or last frame, require IoU(det, last_bbox) ≥
              min_active_iou_short_gap. Prevents handoff to a neighboring person
              when the original detection momentarily disappears.
          • Center-jump gate: if det center jumps more than max_center_jump_ratio
              × track_height per gap frame, suppress. Catches the classic
              handoff signature (A is gone, B appears far away on the predicted
              line — large jump relative to body size).
        """
        det_arr  = np.array(det_bboxes, dtype=np.float32)                              # [M, 4]

        t_hboxes = np.array([states[tid][0] for tid in track_ids], dtype=np.float32)  # [N, 4]
        t_hcs    = np.array([states[tid][1] for tid in track_ids], dtype=np.float32)  # [N, 2]
        t_preds  = np.array([states[tid][2] for tid in track_ids], dtype=np.float32)  # [N, 2]
        t_fbboxes= np.array([states[tid][3] for tid in track_ids], dtype=np.float32)  # [N, 4]
        t_fcs    = np.array([states[tid][4] for tid in track_ids], dtype=np.float32)  # [N, 2]
        t_hv     = np.array([states[tid][5] for tid in track_ids], dtype=bool)        # [N]

        d_hboxes = _head_bbox_arr(det_arr, self.head_ratio, self.shrink_x)             # [M, 4]
        d_hcs    = (d_hboxes[:, :2] + d_hboxes[:, 2:]) / 2                            # [M, 2]
        d_fcs    = _foot_points_arr(det_arr)                                            # [M, 2]
        d_hv     = np.array(
            [_head_visible(tuple(int(x) for x in b), self.min_head_aspect, self.min_head_height)
             for b in det_bboxes], dtype=bool,
        )                                                                               # [M]

        # Distance matrices [M, N]
        hcd = np.linalg.norm(d_hcs[:, None] - t_hcs[None], axis=-1)
        prd = np.linalg.norm(d_hcs[:, None] - t_preds[None], axis=-1)
        fcd = np.linalg.norm(d_fcs[:, None] - t_fcs[None], axis=-1)

        # Full-bbox IoU [M, N]
        iou_full = _batch_iou_np(det_arr, t_fbboxes)

        # Normalized individual costs
        hcd_n = np.minimum(hcd / max(self.max_head_center_distance, 1e-6), 1.0)
        prd_n = np.minimum(prd / max(self.max_predicted_distance,    1e-6), 1.0)
        fcd_n = np.minimum(fcd / max(self.max_foot_distance,         1e-6), 1.0)
        iou_c = 1.0 - iou_full

        # Adaptive blend: head-dominant when both bbox have visible head
        both_head = d_hv[:, None] & t_hv[None]                                         # [M, N]
        cost_head = 0.35 * hcd_n + 0.25 * iou_c + 0.25 * fcd_n + 0.15 * prd_n
        cost_foot = 0.40 * fcd_n + 0.35 * iou_c + 0.25 * prd_n
        cost = np.where(both_head, cost_head, cost_foot)

        # Spatial gating: suppress only when BOTH primary signals exceed max distance
        too_far = (hcd > self.max_head_center_distance) & (fcd > self.max_foot_distance * 1.2)
        cost = np.where(too_far, 1e9, cost)

        # P3 — Kalman Mahalanobis gate. Tight when the track is well-tracked,
        # lenient on fresh tracks (cov is large). Operates BEFORE the P1 hard
        # gates so a Kalman-impossible match is killed regardless of geometry.
        if self.use_kalman and self.kf:
            for j, tid in enumerate(track_ids):
                kf = self.kf.get(tid)
                if kf is None:
                    continue
                for i, bbox in enumerate(det_bboxes):
                    if cost[i, j] >= 1e8:
                        continue  # already gated
                    if kf.mahalanobis(bbox) > self.kalman_gate_chi2:
                        cost[i, j] = 1e9

        # P1 — hard gates beyond the soft cost blend
        if gap_frames is not None:
            gap_arr = np.asarray(gap_frames, dtype=np.int32)                # [N]
            short_mask = (gap_arr <= self.short_gap_frames)[None, :]        # [1, N]

            # Gate 1: IoU floor for short-gap matches
            if short_mask.any():
                fail_iou = short_mask & (iou_full < self.min_active_iou_short_gap)
                cost = np.where(fail_iou, 1e9, cost)

            # Gate 2: center-jump check (det center vs track last bbox center,
            # normalized by track bbox height × gap frames)
            d_centers = (det_arr[:, :2] + det_arr[:, 2:]) / 2                # [M, 2]
            t_centers = (t_fbboxes[:, :2] + t_fbboxes[:, 2:]) / 2            # [N, 2]
            t_heights = np.maximum(t_fbboxes[:, 3] - t_fbboxes[:, 1], 1.0)   # [N]
            jump = np.linalg.norm(d_centers[:, None] - t_centers[None], axis=-1)  # [M, N]
            allowed = self.max_center_jump_ratio * t_heights[None] * np.maximum(gap_arr[None], 1)
            cost = np.where(jump > allowed, 1e9, cost)

        return cost

    def _match_frame_greedy(
        self,
        det_bboxes: list[tuple],
        states: dict,
        candidates: set,
        max_cost: float,
        *,
        gap_frames_by_tid: Optional[dict[str, int]] = None,
    ) -> list[Optional[tuple[str, float]]]:
        """Adaptive matching. Hungarian when M>=2 and N>=2 (avoids order-dependent
        ID switches when persons cross); greedy otherwise (equivalent and cheaper).

        gap_frames_by_tid: optional per-track frame gap (current_frame - last_seen).
        When provided, _build_cost_matrix applies P1 hard gates (IoU floor on
        short gaps, center-jump guard). When None, only soft cost + the legacy
        spatial too_far gate apply (kept for the buffer reactivation path which
        has its own dedicated IoU guard downstream)."""
        if not candidates or not det_bboxes:
            return [None] * len(det_bboxes)

        track_ids = list(candidates)
        M, N = len(det_bboxes), len(track_ids)
        gap_list = (
            [gap_frames_by_tid.get(tid, 0) for tid in track_ids]
            if gap_frames_by_tid is not None else None
        )
        cost = self._build_cost_matrix(
            det_bboxes, states, track_ids, gap_frames=gap_list,
        )

        results: list[Optional[tuple[str, float]]] = [None] * M
        margin = self.discriminative_margin

        if M >= 2 and N >= 2:
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    cost_assigned = cost[r, c]
                    if cost_assigned > max_cost:
                        continue
                    # G2 — discriminative gate: assigned cost must beat the
                    # second-best track cost for this det by at least `margin`.
                    # When two tracks are almost equally close, the assignment
                    # is a coin flip and likely a crossing handoff.
                    if margin > 0.0 and N >= 2:
                        row_copy = cost[r].copy()
                        row_copy[c] = np.inf
                        second_best = float(row_copy.min())
                        if second_best - cost_assigned < margin:
                            continue
                    results[r] = (track_ids[c], float(cost_assigned))
                return results
            except Exception as exc:
                # Fall through to greedy on any scipy issue — semantics preserved.
                pass

        used: list[int] = []
        for m in range(M):
            row = cost[m].copy()
            if used:
                row[used] = 1e9
            best_j = int(row.argmin())
            best_cost = row[best_j]
            if best_cost > max_cost:
                continue
            if margin > 0.0 and N >= 2:
                row2 = row.copy()
                row2[best_j] = np.inf
                second_best = float(row2.min())
                if second_best - best_cost < margin:
                    continue
            results[m] = (track_ids[best_j], float(best_cost))
            used.append(best_j)
        return results

    def _should_keep(self, obs: list[TrackletObservation]) -> bool:
        if len(obs) < self.min_track_frames:
            return False
        span = max(obs[-1].frame_index - obs[0].frame_index + 1, 1)
        return (len(obs) / span) >= self.min_track_density

    def _finalize_track(self, video_id: str, camera_id: Optional[str],
                        tid: str, obs: list[TrackletObservation],
                        completed: list[LocalTracklet]) -> None:
        """Apply post-hoc split + min-length filter to one finished track and
        append the resulting sub-tracklet(s) to `completed`. Sub-tracklets get
        suffixes (e.g. tid='12_a', '12_b') so downstream code can still treat
        them as distinct raw tracklets."""
        if not obs:
            return
        segments = self._split_post_hoc(obs)
        if len(segments) == 1:
            if self._should_keep(segments[0]):
                completed.append(LocalTracklet(video_id, camera_id, tid, tuple(segments[0])))
            return
        # Multiple segments after splitting — each gets a unique track_id
        # suffix. min_track_frames is applied per-segment so a 2-frame
        # leftover stub is dropped naturally.
        for i, seg in enumerate(segments):
            if not self._should_keep(seg):
                continue
            sub_tid = f"{tid}s{i}"
            completed.append(LocalTracklet(video_id, camera_id, sub_tid, tuple(seg)))

    def _adaptive_split_threshold(self, n: int) -> float:
        """G7 — continuous adaptive threshold by Kalman 'trust'.

        Replaces the original step-function (1.6 / 1.0 / 0.65 in three length
        buckets) with a smooth piecewise-linear interpolation anchored on
        Kalman convergence (a filter property, not a camera property).

        Curve:
          n = 1   → 1.30 × base   (short, lenient: Kalman noisy)
          n = 20  → 1.00 × base   (converged: use base)
          n ≥ 60  → 0.95 × base   (long: chỉ siết nhẹ — multiplier 0.80 trước đây
                                   chặt 1 track dài thành 30+ segments khi
                                   người đi qua góc khuất tạm thời)

        Differences vs v1 (1.6 / 1.0 / 0.65 step):
          • Continuous — no cliff edge at bucket boundaries.
          • Multipliers smaller (1.30 instead of 1.60, 0.80 instead of 0.65)
            so the per-camera bias from the original constants is reduced.
          • Convergence anchors (20, 60 observations) come from Kalman
            theory, not from observed cam_2 track distribution.

        Expected trade-off vs v1: slightly less purity (~1-2 pts), notably
        fewer fragments, more robust when ported across cameras.
        """
        base = self.split_chi2
        if n <= 1:
            return base * 1.30
        if n <= 20:
            # Linear 1.30 → 1.00 as n goes 1 → 20
            t = (n - 1) / 19.0
            return base * (1.30 - 0.30 * t)
        if n <= 60:
            # Linear 1.00 → 0.95 as n goes 20 → 60 (nới từ 0.80)
            t = (n - 20) / 40.0
            return base * (1.00 - 0.05 * t)
        return base * 0.95

    def _split_pass(
        self, obs: list[TrackletObservation],
    ) -> list[list[TrackletObservation]]:
        """One forward Kalman-replay pass: split at any observation whose
        innovation exceeds the adaptive threshold. Returns sub-tracklets."""
        if len(obs) < 3:
            return [obs]
        threshold = self._adaptive_split_threshold(len(obs))
        kf = _PersonKalman(obs[0].bbox)
        cut_points: list[int] = []
        for i in range(1, len(obs)):
            gap = max(obs[i].frame_index - obs[i - 1].frame_index, 1)
            kf.predict(n_steps=gap)
            m = kf.mahalanobis(obs[i].bbox)
            if m > threshold:
                cut_points.append(i)
                kf = _PersonKalman(obs[i].bbox)
            else:
                kf.update(obs[i].bbox)
        if not cut_points:
            return [obs]
        segments: list[list[TrackletObservation]] = []
        prev = 0
        for c in cut_points:
            segments.append(obs[prev:c])
            prev = c
        segments.append(obs[prev:])
        return segments

    def _split_post_hoc(
        self, obs: list[TrackletObservation],
    ) -> list[list[TrackletObservation]]:
        """G4 + G7 + G8 — replay Kalman over the tracklet and split at jumps.

        G8: recursive — after a split, re-run the pass on each sub-tracklet
        in case it still contains further handoffs (3+ different GT persons
        merged into one raw track). Bounded by max_split_passes to avoid
        pathological loops on noisy data.
        """
        if not self.split_post_hoc or len(obs) < 3:
            return [obs]

        max_passes = 4
        segments = [obs]
        for _ in range(max_passes):
            changed = False
            new_segments: list[list[TrackletObservation]] = []
            for seg in segments:
                sub = self._split_pass(seg)
                if len(sub) > 1:
                    changed = True
                new_segments.extend(sub)
            segments = new_segments
            if not changed:
                break
        return segments

    @staticmethod
    def _make_obs(det: FrameDetection, ts: float) -> TrackletObservation:
        return TrackletObservation(
            frame_index=det.frame_index,
            timestamp_second=ts,
            bbox=det.bbox,
            confidence=det.confidence,
            laplacian_score=det.laplacian_score,
            crop_bgr=det.crop_bgr,
            is_low_quality_crop=det.is_low_quality_crop,
        )

    def track(
        self,
        video_id: str,
        camera_id: Optional[str],
        detections_by_frame: dict[int, list[FrameDetection]],
    ) -> tuple[LocalTracklet, ...]:
        self._reset()
        completed: list[LocalTracklet] = []

        for fk in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(fk) or [])
            ts   = dets[0].timestamp_second if dets else 0.0

            # Move stale active → buffer. real_last_frame is the last frame
            # with a true detector observation; active_last_frame may include a
            # bounded phantom bump and must not keep a dead track alive forever.
            stale = [tid for tid in self.active
                     if fk - self._real_last_frame.get(tid, self.active_last_frame.get(tid, fk))
                     > self.track_buffer + self.max_phantom_frames]
            for tid in stale:
                self.buffer[tid] = self.active.pop(tid)
                self.buffer_last_bbox[tid] = self.active_last_bbox.pop(tid)
                self.buffer_entry_frame[tid] = fk
                self.active_last_frame.pop(tid, None)
                self._real_last_frame.pop(tid, None)
                self._phantom_streak.pop(tid, None)
                self._phantom_kf.pop(tid, None)
                # P3: drop Kalman filter when track goes to buffer. Re-seeded
                # on reactivation — the long gap means the old velocity is
                # stale and could mislead matching.
                self.kf.pop(tid, None)

            # Expire old buffer tracks
            expired = [tid for tid in self.buffer
                       if fk - self.buffer_entry_frame.get(tid, fk) > self.max_buffer_frames]
            for tid in expired:
                obs = self.buffer.pop(tid, [])
                self._finalize_track(video_id, camera_id, tid, obs, completed)
                self.buffer_last_bbox.pop(tid, None)
                self.buffer_entry_frame.pop(tid, None)

            matched_this_frame: set[str] = set()
            if not dets:
                self._phantom_extend_unmatched(set(self.active.keys()), fk)
                continue

            high = [d for d in dets if d.confidence >= self.track_thresh]
            low  = [d for d in dets if self.low_thresh <= d.confidence < self.track_thresh]

            active_unmatched = set(self.active.keys())
            # P3: advance Kalman state to current frame before matching, so the
            # cost matrix and Mahalanobis gate see the correctly-projected mean
            # / covariance. The number of predict steps = frame gap since the
            # last update.
            if self.use_kalman:
                for tid in active_unmatched:
                    kf = self.kf.get(tid)
                    if kf is None:
                        continue
                    gap = max(fk - self.active_last_frame.get(tid, fk), 1)
                    kf.predict(n_steps=gap)
            states = {
                tid: self._build_state(self.active[tid], self.active_last_bbox[tid], fk, tid=tid)
                for tid in active_unmatched
            }
            # Per-track frame gap → used by P1 hard gates in _build_cost_matrix
            gap_by_tid = {
                tid: max(fk - self._real_last_frame.get(tid, self.active_last_frame.get(tid, fk)), 0)
                for tid in active_unmatched
            }
            unmatched_high: list[FrameDetection] = []
            if high and active_unmatched:
                matches = self._match_frame_greedy(
                    [d.bbox for d in high], states, active_unmatched, self.max_match_cost,
                    gap_frames_by_tid=gap_by_tid,
                )
                for det, m in zip(high, matches):
                    if m:
                        tid, _ = m
                        self.active[tid].append(self._make_obs(det, ts))
                        self._mark_real_detection(tid, det.bbox, fk)
                        matched_this_frame.add(tid)
                        self.active_last_bbox[tid] = det.bbox
                        self.active_last_frame[tid] = fk
                        active_unmatched.discard(tid)
                        states.pop(tid, None)
                        gap_by_tid.pop(tid, None)
                        if self.use_kalman and tid in self.kf:
                            self.kf[tid].update(det.bbox)
                    else:
                        unmatched_high.append(det)
            else:
                unmatched_high = list(high)

            if low and active_unmatched:
                # P1 — low-confidence rescue must clear a STRICTER cost cap. Low-conf
                # detections in hospital CCTV are often partial-body / blurry / FP,
                # and the cost of letting one merge nearby into a different person's
                # track is permanent (fragment merger cannot split).
                matches = self._match_frame_greedy(
                    [d.bbox for d in low], states, active_unmatched,
                    self.max_lowconf_match_cost,
                    gap_frames_by_tid=gap_by_tid,
                )
                for det, m in zip(low, matches):
                    if m:
                        tid, _ = m
                        self.active[tid].append(self._make_obs(det, ts))
                        self._mark_real_detection(tid, det.bbox, fk)
                        matched_this_frame.add(tid)
                        self.active_last_bbox[tid] = det.bbox
                        self.active_last_frame[tid] = fk
                        active_unmatched.discard(tid)
                        states.pop(tid, None)
                        if self.use_kalman and tid in self.kf:
                            self.kf[tid].update(det.bbox)

            # Buffer reactivation: re-attach high-confidence dets to recently-lost tracks
            # so the same person keeps the original track_id across short occlusions.
            #
            # Conservative: same cost ceiling as active match (max_match_cost), no
            # relaxation of spatial gates, AND require last_bbox IoU ≥ min_buffer_iou
            # to prevent cross-person re-attachment that would poison the SigLIP
            # embedding pool downstream and cause over-merging in TrackletFragmentMerger.
            if unmatched_high and self.buffer:
                buffer_tids = list(self.buffer.keys())
                buffer_states = {
                    tid: self._build_state(self.buffer[tid], self.buffer_last_bbox[tid], fk)
                    for tid in buffer_tids
                }
                # C3 — apply center-jump gate to buffer reactivation too. A det
                # that drifted to a different person's location can pass the
                # min_buffer_iou check (overlap > 0.30) yet still be a wrong
                # match if its center jumped implausibly far given the gap.
                # Gap measured from buffer entry frame (when track went stale).
                buffer_gap_by_tid = {
                    tid: max(fk - self.buffer_entry_frame.get(tid, fk), 1)
                    for tid in buffer_tids
                }
                matches = self._match_frame_greedy(
                    [d.bbox for d in unmatched_high],
                    buffer_states,
                    set(buffer_tids),
                    min(self.max_buffer_match_cost, self.max_match_cost),
                    gap_frames_by_tid=buffer_gap_by_tid,
                )

                still_unmatched: list[FrameDetection] = []
                for det, m in zip(unmatched_high, matches):
                    if not m:
                        still_unmatched.append(det)
                        continue
                    tid, _ = m
                    # Spatial sanity check: last_bbox of the lost track must overlap
                    # the new det. Without this a det that drifted to a different
                    # person's location can be re-attached to the wrong track.
                    if _bbox_iou(det.bbox, self.buffer_last_bbox[tid]) < self.min_buffer_iou:
                        still_unmatched.append(det)
                        continue
                    obs = self.buffer.pop(tid)
                    self.buffer_last_bbox.pop(tid, None)
                    self.buffer_entry_frame.pop(tid, None)
                    obs.append(self._make_obs(det, ts))
                    self.active[tid] = obs
                    self._mark_real_detection(tid, det.bbox, fk, seed_phantom=True)
                    matched_this_frame.add(tid)
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_frame[tid] = fk
                    if self.use_kalman:
                        # Re-seed Kalman from the new det — old velocity/cov is
                        # stale after a buffer-length gap.
                        self.kf[tid] = _PersonKalman(det.bbox)
                unmatched_high = still_unmatched

            for det in unmatched_high:
                if det.confidence >= self.new_track_threshold:
                    tid = str(self.next_id)
                    self.next_id += 1
                    self.active[tid] = [self._make_obs(det, ts)]
                    self._mark_real_detection(tid, det.bbox, fk, seed_phantom=True)
                    matched_this_frame.add(tid)
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_frame[tid] = fk
                    if self.use_kalman:
                        self.kf[tid] = _PersonKalman(det.bbox)

            self._phantom_extend_unmatched(set(self.active.keys()) - matched_this_frame, fk)

        # Finalize all remaining tracks
        for tid, obs in list(self.active.items()) + list(self.buffer.items()):
            self._finalize_track(video_id, camera_id, tid, list(obs), completed)
        self._reset()
        return tuple(completed)



# Backward-compatible alias — existing code importing HeadBoxTracker still works.
HeadBoxTracker = BodyPartAdaptiveTracker


# ── TrackletQualityScorer ─────────────────────────────────────────────────────

class TrackletQualityScorer:
    def __init__(
        self,
        min_confidence: float = 0.30,
        min_frames: int = 3,
        min_density: float = 0.05,
        min_duration_s: float = 0.5,
        min_laplacian: float = 8.0,
    ):
        self.min_confidence = min_confidence
        self.min_frames = min_frames
        self.min_density = min_density
        self.min_duration_s = min_duration_s
        self.min_laplacian = min_laplacian

    def score(self, tracklet: LocalTracklet) -> TrackletQualityResult:
        obs = tracklet.observations
        if not obs:
            return TrackletQualityResult(False, 0.0, 0.0, 0, 0.0, 0.0, "empty")

        confs = [o.confidence for o in obs]
        laps  = [o.laplacian_score for o in obs]
        avg_conf = sum(confs) / len(confs)
        avg_lap  = sum(laps)  / len(laps)
        n    = len(obs)
        span = max(obs[-1].frame_index - obs[0].frame_index + 1, 1)
        density  = n / span
        duration = max(obs[-1].timestamp_second - obs[0].timestamp_second, 0.0) if n >= 2 else 0.0

        if n < self.min_frames:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "insufficient_frames")
        if density < self.min_density:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "sparse_tracklet")
        if duration < self.min_duration_s:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "short_tracklet")
        if avg_conf < self.min_confidence:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "low_confidence")
        if avg_lap < self.min_laplacian:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "blurry_tracklet")
        return TrackletQualityResult(True, avg_conf, avg_lap, n, density, duration, None)


# ── TrackletFragmentMerger ────────────────────────────────────────────────────

def _cosine_sim_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Pairwise cosine similarity for N embeddings → [N, N] float32."""
    n = len(embeddings)
    if n == 0:
        return np.zeros((n, n), dtype=np.float32)
    dim = next((len(e) for e in embeddings if e), 0)
    if dim == 0:
        return np.zeros((n, n), dtype=np.float32)
    arr = np.zeros((n, dim), dtype=np.float32)
    for i, emb in enumerate(embeddings):
        if emb and len(emb) == dim:
            arr[i] = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = arr / norms
    return (normed @ normed.T).astype(np.float32)


class TrackletFragmentMerger:
    """
    Post-hoc fragment merging using SigLIP2 or PersonViT cosine similarity.

    Fragments of the same person caused by occlusion are re-joined when:
      1. They are strictly temporally ordered (tj starts after ti ends — no overlap).
      2. The temporal gap is within max_gap_seconds / max_gap_frames.
      3. Their appearance embeddings have cosine similarity ≥ similarity_threshold.
      4. The foot-point displacement between ti.end and tj.start is plausible
         given the gap duration (max_speed_px_per_s), unless SigLIP similarity
         is high enough (≥ sim_thresh + spatial_bypass_margin) to override.

    Union-Find is guarded at component level so a weak bridge cannot collapse
    many different people into one large merged tracklet.

    Returns the merged LocalTracklet list and a groups list so the caller can
    pool the corresponding feature arrays (embeddings, attributes, actions).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_gap_seconds: float = 480.0,
        max_gap_frames: int = 1920,
        component_similarity_margin: float = 0.03,
        max_speed_px_per_s: float = 800.0,
        spatial_bypass_margin: float = 0.05,
        # 2026-05-17: absolute upper bound on appearance-pass spatial budget.
        # Without this, max_dist = gap_t × speed grows linearly with gap; at
        # gap=30s with speed=300 it reaches 9000 px — wider than any FOV, so
        # the spatial gate becomes a no-op for long gaps. Cap at the camera's
        # plausible cross-frame travel for a single appearance (≈1/5 of a 1080p
        # frame width). Set to 0 to disable the absolute cap and keep purely
        # gap-scaled behavior.
        max_spatial_dist_px: float = 400.0,
        # E — motion-only post-merge pass. After SigLIP union-find, run an
        # extra pass merging tracklet pairs with short gap + plausible
        # trajectory continuation, ignoring appearance. Catches over-fragmented
        # tracks where G8 post-hoc split chopped one person mid-walk.
        #
        # R6: gap_s 3.0 (was 5.0) — beyond 3s extrapolation is too noisy,
        # let SigLIP handle longer gaps.
        # R1: max_extrap_ratio replaces absolute px — error must be ≤
        # this × bbox height of ti's last observation. Scales với độ xa
        # tới camera (person 60px height → 30px budget; 200px → 100px).
        # R2: max_velocity_angle_deg — angle giữa velocity tail của ti và
        # velocity head của tj. Cross-walk handoff sẽ có angle >60°.
        # R3: min_velocity_px_per_s — dưới ngưỡng này coi như "đứng yên",
        # extrapolation không đáng tin, dùng tight foot-point gate thay thế.
        # R4: min_obs_each_side — cả ti và tj phải có ≥ obs này để velocity
        # reliable.
        motion_merge_max_gap_seconds: float = 4.0,
        motion_merge_max_extrap_ratio: float = 0.5,
        motion_merge_max_velocity_angle_deg: float = 60.0,
        motion_merge_min_velocity_px_per_s: float = 20.0,
        motion_merge_min_obs_each_side: int = 3,
        # 2026-05-17: appearance floor for the motion-only pass. Without this,
        # two different people crossing at the same point (cos < 0.85 but
        # dist≈0px) get merged purely on motion continuity. The floor stays
        # BELOW sim_thresh so motion-merge can still rescue same-person
        # fragments whose crops are degraded (backlit/partial body), which
        # was the whole purpose of this pass.
        motion_merge_min_appearance_sim: float = 0.85,
    ):
        self.sim_thresh     = similarity_threshold
        self.max_gap_s      = max_gap_seconds
        self.max_gap_frames = max_gap_frames
        self.component_floor = max(0.0, similarity_threshold - component_similarity_margin)
        self.max_speed_px_per_s = max_speed_px_per_s
        self.spatial_bypass_thresh = min(1.0, similarity_threshold + spatial_bypass_margin)
        self.motion_merge_max_gap_s = motion_merge_max_gap_seconds
        self.motion_merge_max_extrap_ratio = motion_merge_max_extrap_ratio
        self.motion_merge_max_velocity_angle_rad = math.radians(motion_merge_max_velocity_angle_deg)
        self.motion_merge_min_velocity = motion_merge_min_velocity_px_per_s
        self.motion_merge_min_obs = motion_merge_min_obs_each_side
        self.motion_merge_min_appearance_sim = float(motion_merge_min_appearance_sim)
        self.max_spatial_dist_px = float(max_spatial_dist_px)

    def merge(
        self,
        tracklets: list[LocalTracklet],
        embeddings: list[list[float]],
    ) -> tuple[list[LocalTracklet], list[list[int]]]:
        """
        Parameters
        ----------
        tracklets   : accepted tracklets from the tracker (any order)
        embeddings  : one embedding vector per tracklet (same index)

        Returns
        -------
        merged_tracklets : new list, length ≤ len(tracklets)
        groups           : groups[i] = list of original indices merged into
                           merged_tracklets[i]; primary (earliest) is groups[i][0]
        """
        n = len(tracklets)
        if n == 0:
            return [], []

        import logging as _logging
        _log = _logging.getLogger(__name__)

        # Work in start-frame order for the gap-break optimisation
        order = sorted(range(n), key=lambda i: tracklets[i].observations[0].frame_index)
        sim   = _cosine_sim_matrix([embeddings[i] for i in order])  # [n, n]

        # ── Union-Find ────────────────────────────────────────────────────────
        parent = list(range(n))
        members: dict[int, set[int]] = {i: {i} for i in range(n)}

        starts_f = [tracklets[order[i]].observations[0].frame_index for i in range(n)]
        ends_f   = [tracklets[order[i]].observations[-1].frame_index for i in range(n)]
        starts_t = [tracklets[order[i]].observations[0].timestamp_second for i in range(n)]
        ends_t   = [tracklets[order[i]].observations[-1].timestamp_second for i in range(n)]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def components_temporally_compatible(px: int, py: int) -> bool:
            for a in members[px]:
                for b in members[py]:
                    early, late = (a, b) if starts_t[a] <= starts_t[b] else (b, a)
                    # Same-person fragments cannot be visible concurrently in
                    # one camera. Any temporal overlap → different people.
                    if starts_t[late] - ends_t[early] < 0.0:
                        return False
                    if starts_f[late] - ends_f[early] < 0:
                        return False
            return True

        def components_appearance_compatible(px: int, py: int) -> bool:
            left = sorted(members[px])
            right = sorted(members[py])
            cross = sim[np.ix_(left, right)]
            if cross.size == 0:
                return False
            return (
                float(cross.mean()) >= self.sim_thresh
                and float(cross.min()) >= self.component_floor
            )

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if not components_temporally_compatible(px, py):
                return
            if not components_appearance_compatible(px, py):
                return
            # Keep the earliest-start member as the root for stable output order.
            if min(members[py]) < min(members[px]):
                px, py = py, px
            parent[py] = px
            members[px].update(members.pop(py))

        for i in range(n):
            ti        = tracklets[order[i]]
            ti_end_f  = ti.observations[-1].frame_index
            ti_end_t  = ti.observations[-1].timestamp_second
            ti_end_foot = _foot_point(ti.observations[-1].bbox)

            for j in range(i + 1, n):
                tj         = tracklets[order[j]]
                tj_start_f = tj.observations[0].frame_index
                tj_start_t = tj.observations[0].timestamp_second

                gap_f = tj_start_f - ti_end_f
                if gap_f > self.max_gap_frames:
                    break  # sorted → all future j will also exceed frame gap

                gap_t = tj_start_t - ti_end_t
                if gap_t < 0.0:   # any temporal overlap → different people
                    continue
                if gap_t > self.max_gap_s:
                    continue

                if sim[i, j] < self.sim_thresh:
                    continue

                # Spatial transition gate: foot-point displacement between
                # ti's last frame and tj's first frame must be plausible for
                # the elapsed gap. A weak/borderline appearance match that
                # violates this gate is rejected; a very strong match
                # (≥ spatial_bypass_thresh) can override (e.g. camera with
                # large dead zones, fast occluded transit).
                tj_start_foot = _foot_point(tj.observations[0].bbox)
                dist_px = _dist(ti_end_foot, tj_start_foot)
                # Floor at 0.25 s so gap≈0 doesn't collapse the budget to 0.
                max_dist = max(gap_t, 0.25) * self.max_speed_px_per_s
                # Hard cap: prevent the gap-scaled budget from exceeding any
                # plausible cross-frame travel for a single appearance.
                if self.max_spatial_dist_px > 0.0:
                    max_dist = min(max_dist, self.max_spatial_dist_px)
                if dist_px > max_dist and sim[i, j] < self.spatial_bypass_thresh:
                    continue

                # Audit log so we can see WHY each pair was merged. Filter by
                # "[app-merge]" to investigate suspicious group sizes. DEBUG
                # vì với N tracklet hàng trăm cặp có thể qua gate → log spam.
                _log.debug(
                    "[app-merge] %s ↔ %s | cos=%.3f gap=%.2fs dist=%.0fpx max_dist=%.0fpx bypass=%s",
                    tracklets[order[i]].track_id,
                    tracklets[order[j]].track_id,
                    float(sim[i, j]), gap_t, dist_px, max_dist,
                    "Y" if sim[i, j] >= self.spatial_bypass_thresh else "N",
                )
                union(i, j)

        # ── E: motion-only post-pass (hardened) ───────────────────────────────
        # SigLIP threshold 0.85 vẫn từ chối nhiều pair cùng người do crop bị
        # ngược sáng / partial body. Chạy thêm pass merge các cặp có:
        #   • Gap ≤ motion_merge_max_gap_s (3s)                          R6
        #   • Cả ti, tj có ≥ motion_merge_min_obs obs                    R4
        #   • Velocity tail của ti và head của tj cùng hướng (≤60°)      R2
        #   • Foot-point extrapolation error ≤ ratio × bbox_height       R1
        #   • Speed cap toàn cục max_speed_px_per_s
        # Nếu velocity quá thấp (đứng yên), fallback: tight foot-point   R3
        # gate (dist ≤ 0.3 × bbox_height).
        # Mọi merge được log để audit (R7).
        if self.motion_merge_max_gap_s > 0.0:
            n_motion_merges = 0

            def _foot_velocity_tail(obs: tuple) -> tuple[float, float, float]:
                """Velocity tại đuôi tracklet, từ 2 obs cuối. Trả (vx, vy, speed)."""
                if len(obs) < 2:
                    return 0.0, 0.0, 0.0
                a, b = obs[-2], obs[-1]
                dt = b.timestamp_second - a.timestamp_second
                if dt <= 1e-6:
                    return 0.0, 0.0, 0.0
                fa = _foot_point(a.bbox)
                fb = _foot_point(b.bbox)
                vx = (fb[0] - fa[0]) / dt
                vy = (fb[1] - fa[1]) / dt
                return vx, vy, math.hypot(vx, vy)

            def _foot_velocity_head(obs: tuple) -> tuple[float, float, float]:
                """Velocity tại đầu tracklet, từ 2 obs đầu."""
                if len(obs) < 2:
                    return 0.0, 0.0, 0.0
                a, b = obs[0], obs[1]
                dt = b.timestamp_second - a.timestamp_second
                if dt <= 1e-6:
                    return 0.0, 0.0, 0.0
                fa = _foot_point(a.bbox)
                fb = _foot_point(b.bbox)
                vx = (fb[0] - fa[0]) / dt
                vy = (fb[1] - fa[1]) / dt
                return vx, vy, math.hypot(vx, vy)

            def _angle_between(v1: tuple, v2: tuple) -> float:
                """Góc giữa 2 vector (rad), trong [0, π]. Trả 0 nếu vector rỗng."""
                n1 = math.hypot(v1[0], v1[1])
                n2 = math.hypot(v2[0], v2[1])
                if n1 < 1e-6 or n2 < 1e-6:
                    return 0.0
                cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
                return math.acos(max(-1.0, min(1.0, cos_a)))

            for i in range(n):
                ti = tracklets[order[i]]
                # R4: skip tracklets quá ngắn để velocity reliable
                if len(ti.observations) < self.motion_merge_min_obs:
                    continue
                ti_end_t = ends_t[i]
                ti_end_bbox = ti.observations[-1].bbox
                ti_end_foot = _foot_point(ti_end_bbox)
                ti_h = max(ti_end_bbox[3] - ti_end_bbox[1], 1.0)
                vx_i, vy_i, speed_i = _foot_velocity_tail(ti.observations)

                # R1: extrap budget scale theo bbox height
                max_extrap = self.motion_merge_max_extrap_ratio * ti_h

                for j in range(i + 1, n):
                    if find(i) == find(j):
                        continue
                    tj = tracklets[order[j]]
                    # R4: skip stub-stub
                    if len(tj.observations) < self.motion_merge_min_obs:
                        continue

                    tj_start_t = starts_t[j]
                    gap_t = tj_start_t - ti_end_t
                    if gap_t < 0.0 or gap_t > self.motion_merge_max_gap_s:
                        continue

                    tj_start_bbox = tj.observations[0].bbox
                    tj_start_foot = _foot_point(tj_start_bbox)

                    # Speed cap toàn cục
                    raw_dist = _dist(ti_end_foot, tj_start_foot)
                    if raw_dist > max(gap_t, 0.25) * self.max_speed_px_per_s:
                        continue

                    # Appearance floor — block "two different people crossing
                    # at the same point" (cos ≪ sim_thresh but dist≈0).
                    # Still BELOW sim_thresh so SigLIP-borderline same-person
                    # rescues remain possible.
                    if float(sim[i, j]) < self.motion_merge_min_appearance_sim:
                        continue

                    if speed_i < self.motion_merge_min_velocity:
                        # R3: ti đứng yên — không tin extrapolation, chỉ chấp
                        # nhận khi foot-point gần như chồng nhau (0.3× height)
                        if raw_dist > 0.3 * ti_h:
                            continue
                    else:
                        # R2: velocity angle check. So velocity tail ti vs
                        # velocity head tj. Cross-walk handoff sẽ có góc lớn.
                        vx_j, vy_j, speed_j = _foot_velocity_head(tj.observations)
                        if speed_j >= self.motion_merge_min_velocity:
                            ang = _angle_between((vx_i, vy_i), (vx_j, vy_j))
                            if ang > self.motion_merge_max_velocity_angle_rad:
                                continue

                        # R1: extrap error ≤ ratio × bbox_h
                        pred_x = ti_end_foot[0] + vx_i * gap_t
                        pred_y = ti_end_foot[1] + vy_i * gap_t
                        err = _dist((pred_x, pred_y), tj_start_foot)
                        if err > max_extrap:
                            continue

                    # R7: log audit — mức INFO để dễ filter ra.
                    # cos_ij là cosine sim đã tính ở pass appearance — log để
                    # phân biệt motion-merge của cùng người (cos cao nhưng <
                    # 0.89) với gộp nhầm (cos thấp, chỉ bridge bằng motion).
                    _log.info(
                        "[motion-merge] %s ↔ %s | cos=%.3f gap=%.2fs dist=%.0fpx speed_i=%.0fpx/s",
                        tracklets[order[i]].track_id,
                        tracklets[order[j]].track_id,
                        float(sim[i, j]),
                        gap_t, raw_dist, speed_i,
                    )
                    n_motion_merges += 1
                    union(i, j)

            if n_motion_merges > 0:
                _log.info("[motion-merge] total %d pairs merged via motion-only pass", n_motion_merges)

        # ── Build merged tracklets ────────────────────────────────────────────
        root_to_members: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            root_to_members[find(i)].append(i)  # sorted indices (into order[])

        merged_tracklets: list[LocalTracklet] = []
        groups: list[list[int]] = []

        for root in sorted(root_to_members):
            sorted_members = root_to_members[root]
            orig_idxs      = [order[m] for m in sorted_members]  # original indices

            if len(orig_idxs) == 1:
                merged_tracklets.append(tracklets[orig_idxs[0]])
                groups.append(orig_idxs)
                continue

            # Combine observations from all fragments in temporal order
            all_obs: list[TrackletObservation] = []
            for idx in orig_idxs:
                all_obs.extend(tracklets[idx].observations)
            all_obs.sort(key=lambda o: o.frame_index)

            # Primary = earliest start (orig_idxs[0] due to sorted order)
            primary = tracklets[orig_idxs[0]]
            merged  = LocalTracklet(
                video_id=primary.video_id,
                camera_id=primary.camera_id,
                track_id=primary.track_id,   # kept so quality_results lookup still works
                observations=tuple(all_obs),
            )
            merged_tracklets.append(merged)
            groups.append(orig_idxs)

        return merged_tracklets, groups
