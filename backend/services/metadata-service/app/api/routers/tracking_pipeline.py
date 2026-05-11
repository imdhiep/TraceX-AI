"""
Tracking pipeline: BodyPartAdaptiveTracker + post-hoc TrackletFragmentMerger.

BodyPartAdaptiveTracker uses head-dominant or foot-dominant cost depending on
whether the head region is estimated to be visible in each detection bbox.
TrackletFragmentMerger re-joins fragments of the same person after feature
extraction using SigLIP2 cosine similarity.
"""
from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

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
    laplacian_score: float


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    timestamp_second: float
    bbox: tuple[int, int, int, int]   # x1 y1 x2 y2
    confidence: float
    laplacian_score: float
    crop_bgr: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TrackletObservation:
    frame_index: int
    timestamp_second: float
    bbox: tuple[int, int, int, int]
    confidence: float
    laplacian_score: float
    crop_bgr: Optional[np.ndarray] = None


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


def _crop_from_bbox(image: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


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
    """Sample frames at fixed FPS with Laplacian blur scoring."""

    def __init__(self, sample_fps: int = 4):
        self.sample_fps = max(1, sample_fps)

    def sample(self, video_path: str) -> tuple[SampledFrame, ...]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.sample_fps)
        if source_fps <= 0:
            source_fps = float(self.sample_fps)

        # Collect sampled frames first (sequential — codec requires in-order reads)
        raw: list[tuple[int, float, np.ndarray]] = []
        next_emit = 0.0
        src_idx = 0
        sampled_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                ts = src_idx / source_fps
                if ts + 1e-9 >= next_emit:
                    raw.append((sampled_idx, round(ts, 6), frame.copy()))
                    sampled_idx += 1
                    next_emit += 1.0 / self.sample_fps
                src_idx += 1
        finally:
            cap.release()

        # Compute Laplacian in parallel — cv2 releases GIL so threads run truly concurrently
        def _laplacian(item: tuple[int, float, np.ndarray]) -> SampledFrame:
            idx, ts, f = item
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            return SampledFrame(
                frame_index=idx,
                timestamp_second=ts,
                image=f,
                laplacian_score=round(lap, 6),
            )

        workers = min(8, len(raw) or 1)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(_laplacian, raw))

        return tuple(frames)


# ── BodyPartAdaptiveTracker ───────────────────────────────────────────────────

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
        max_match_cost: float = 0.80,
        max_buffer_match_cost: float = 0.90,
        max_head_center_distance: float = 120.0,
        max_foot_distance: float = 150.0,
        max_predicted_distance: float = 180.0,
        track_buffer: int = 20,
        max_buffer_frames: int = 300,
        head_ratio: float = 0.35,
        shrink_x: float = 0.08,
        min_head_aspect: float = 1.1,
        min_head_height: int = 30,
        min_track_frames: int = 3,
        min_track_density: float = 0.05,
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
        self._reset()

    def _reset(self) -> None:
        self.active: dict[str, list[TrackletObservation]] = {}
        self.active_last_bbox: dict[str, tuple] = {}
        self.active_last_frame: dict[str, int] = {}
        self.buffer: dict[str, list[TrackletObservation]] = {}
        self.buffer_last_bbox: dict[str, tuple] = {}
        self.buffer_entry_frame: dict[str, int] = {}
        self.next_id = 1

    def _hbox(self, bbox: tuple) -> tuple:
        return _head_bbox(bbox, self.head_ratio, self.shrink_x)

    def _build_state(self, obs: list[TrackletObservation], last_bbox: tuple, target_fk: int):
        """
        Returns (hbox, head_center, predicted_center, full_bbox, foot_point, head_visible).
        Velocity is computed on head center for consistency with head-mode cost.
        """
        hbox = self._hbox(last_bbox)
        hc   = _center(hbox)
        fc   = _foot_point(last_bbox)
        hv   = _head_visible(last_bbox, self.min_head_aspect, self.min_head_height)

        if len(obs) < 2:
            return hbox, hc, hc, last_bbox, fc, hv

        prev_hbox = self._hbox(obs[-2].bbox)
        pc = _center(prev_hbox)
        delta = max(obs[-1].frame_index - obs[-2].frame_index, 1)
        vx = (hc[0] - pc[0]) / delta
        vy = (hc[1] - pc[1]) / delta
        steps = min(max(target_fk - obs[-1].frame_index, 0), self.track_buffer * 2)
        pred = (hc[0] + vx * steps, hc[1] + vy * steps)
        return hbox, hc, pred, last_bbox, fc, hv

    def _match_frame_greedy(
        self,
        det_bboxes: list[tuple],
        states: dict,
        candidates: set,
        max_cost: float,
    ) -> list[Optional[tuple[str, float]]]:
        """Vectorized adaptive greedy matching."""
        if not candidates or not det_bboxes:
            return [None] * len(det_bboxes)

        track_ids = list(candidates)
        M = len(det_bboxes)

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

        results: list[Optional[tuple[str, float]]] = [None] * M
        used: list[int] = []
        for m in range(M):
            row = cost[m].copy()
            if used:
                row[used] = 1e9
            best_j = int(row.argmin())
            if row[best_j] <= max_cost:
                results[m] = (track_ids[best_j], float(row[best_j]))
                used.append(best_j)
        return results

    def _should_keep(self, obs: list[TrackletObservation]) -> bool:
        if len(obs) < self.min_track_frames:
            return False
        span = max(obs[-1].frame_index - obs[0].frame_index + 1, 1)
        return (len(obs) / span) >= self.min_track_density

    @staticmethod
    def _make_obs(det: FrameDetection, ts: float) -> TrackletObservation:
        return TrackletObservation(
            frame_index=det.frame_index,
            timestamp_second=ts,
            bbox=det.bbox,
            confidence=det.confidence,
            laplacian_score=det.laplacian_score,
            crop_bgr=det.crop_bgr,
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

            # Move stale active → buffer
            stale = [tid for tid in self.active
                     if fk - self.active_last_frame.get(tid, fk) > self.track_buffer]
            for tid in stale:
                self.buffer[tid] = self.active.pop(tid)
                self.buffer_last_bbox[tid] = self.active_last_bbox.pop(tid)
                self.buffer_entry_frame[tid] = fk
                self.active_last_frame.pop(tid, None)

            # Expire old buffer tracks
            expired = [tid for tid in self.buffer
                       if fk - self.buffer_entry_frame.get(tid, fk) > self.max_buffer_frames]
            for tid in expired:
                obs = self.buffer.pop(tid, [])
                if obs and self._should_keep(obs):
                    completed.append(LocalTracklet(video_id, camera_id, tid, tuple(obs)))
                self.buffer_last_bbox.pop(tid, None)
                self.buffer_entry_frame.pop(tid, None)

            if not dets:
                continue

            high = [d for d in dets if d.confidence >= self.track_thresh]
            low  = [d for d in dets if self.low_thresh <= d.confidence < self.track_thresh]

            active_unmatched = set(self.active.keys())
            states = {
                tid: self._build_state(self.active[tid], self.active_last_bbox[tid], fk)
                for tid in active_unmatched
            }

            unmatched_high: list[FrameDetection] = []
            if high and active_unmatched:
                matches = self._match_frame_greedy(
                    [d.bbox for d in high], states, active_unmatched, self.max_match_cost,
                )
                for det, m in zip(high, matches):
                    if m:
                        tid, _ = m
                        self.active[tid].append(self._make_obs(det, ts))
                        self.active_last_bbox[tid] = det.bbox
                        self.active_last_frame[tid] = fk
                        active_unmatched.discard(tid)
                        states.pop(tid, None)
                    else:
                        unmatched_high.append(det)
            else:
                unmatched_high = list(high)

            if low and active_unmatched:
                matches = self._match_frame_greedy(
                    [d.bbox for d in low], states, active_unmatched, self.max_match_cost,
                )
                for det, m in zip(low, matches):
                    if m:
                        tid, _ = m
                        self.active[tid].append(self._make_obs(det, ts))
                        self.active_last_bbox[tid] = det.bbox
                        self.active_last_frame[tid] = fk
                        active_unmatched.discard(tid)
                        states.pop(tid, None)

            for det in unmatched_high:
                if det.confidence >= self.new_track_threshold:
                    tid = str(self.next_id)
                    self.next_id += 1
                    self.active[tid] = [self._make_obs(det, ts)]
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_frame[tid] = fk

        # Finalize all remaining tracks
        for tid, obs in list(self.active.items()) + list(self.buffer.items()):
            if obs and self._should_keep(obs):
                completed.append(LocalTracklet(video_id, camera_id, tid, tuple(obs)))
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
    if n == 0 or not embeddings[0]:
        return np.zeros((n, n), dtype=np.float32)
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = arr / norms
    return (normed @ normed.T).astype(np.float32)


class TrackletFragmentMerger:
    """
    Post-hoc fragment merging using SigLIP2 cosine similarity.

    Fragments of the same person caused by occlusion are re-joined when:
      1. They are temporally ordered (tj starts after ti ends, overlap ≤ 1 s).
      2. The temporal gap is within max_gap_seconds / max_gap_frames.
      3. Their appearance embeddings have cosine similarity ≥ similarity_threshold.

    Union-Find handles transitive chains (A~B and B~C → A merged into C).

    Returns the merged LocalTracklet list and a groups list so the caller can
    pool the corresponding feature arrays (embeddings, attributes, actions).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_gap_seconds: float = 30.0,
        max_gap_frames: int = 120,
    ):
        self.sim_thresh     = similarity_threshold
        self.max_gap_s      = max_gap_seconds
        self.max_gap_frames = max_gap_frames

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

        # Work in start-frame order for the gap-break optimisation
        order = sorted(range(n), key=lambda i: tracklets[i].observations[0].frame_index)
        sim   = _cosine_sim_matrix([embeddings[i] for i in order])  # [n, n]

        # ── Union-Find ────────────────────────────────────────────────────────
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px  # px is the earlier-start member (x < y in sorted order)

        for i in range(n):
            ti        = tracklets[order[i]]
            ti_end_f  = ti.observations[-1].frame_index
            ti_end_t  = ti.observations[-1].timestamp_second

            for j in range(i + 1, n):
                tj         = tracklets[order[j]]
                tj_start_f = tj.observations[0].frame_index
                tj_start_t = tj.observations[0].timestamp_second

                gap_f = tj_start_f - ti_end_f
                if gap_f > self.max_gap_frames:
                    break  # sorted → all future j will also exceed frame gap

                gap_t = tj_start_t - ti_end_t
                if gap_t < -1.0:   # overlap > 1 s → concurrent, different people
                    continue
                if gap_t > self.max_gap_s:
                    continue

                if sim[i, j] >= self.sim_thresh:
                    union(i, j)

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
