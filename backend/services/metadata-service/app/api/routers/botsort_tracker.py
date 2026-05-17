"""BoT-SORT tracker adapter matching the BodyPartAdaptiveTracker contract.

The adapter keeps the public pipeline interface unchanged:
track(video_id, camera_id, detections_by_frame) -> tuple[LocalTracklet, ...].

Design notes:
  - ReID is disabled. Appearance grouping remains downstream in
    TrackletFragmentMerger/SigLIP, same as the legacy tracker.
  - GMC is neutralized with an identity transform because the call site only
    passes detections, and the target CCTV cameras are fixed rigs.
  - BoxMOT internally rescales track_buffer as frame_rate / 30 * track_buffer;
    this adapter accepts track_buffer in sampled-frame units and converts it.
  - A conservative post-association split guard breaks suspicious BoT-SORT
    joins before quality filtering and fragment merging. False splits are much
    cheaper than letting an already-merged tracklet reach SigLIP/VLM.
"""

from __future__ import annotations

from collections import Counter
import logging
import math
import os
from typing import Optional

import numpy as np

from .tracking_pipeline import FrameDetection, LocalTracklet, TrackletObservation


logger = logging.getLogger(__name__)


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def _bbox_center(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _bbox_height(bbox: tuple) -> float:
    _x1, y1, _x2, y2 = (float(v) for v in bbox)
    return max(1.0, y2 - y1)


class BotSortTracker:
    """BoT-SORT (no ReID, no GMC) with legacy pipeline output objects."""

    def __init__(
        self,
        track_thresh: float = 0.40,
        low_thresh: float = 0.10,
        new_track_threshold: float = 0.45,
        match_thresh: float = 0.80,
        track_buffer: int = 20,
        frame_rate: int = 4,
        min_track_frames: int = 2,
        min_track_density: float = 0.05,
        max_center_jump_ratio: float = 1.60,
        max_speed_px_per_s: float = 800.0,
        min_short_gap_iou: float = 0.02,
        short_gap_frames: int = 2,
        **_legacy_ignored,
    ):
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_threshold = float(new_track_threshold)
        self.match_thresh = float(match_thresh)
        self.track_buffer = max(1, int(track_buffer))
        self.frame_rate = max(1, int(frame_rate))
        self.boxmot_track_buffer = max(1, int(round(self.track_buffer * 30.0 / self.frame_rate)))
        self.min_track_frames = int(min_track_frames)
        self.min_track_density = float(min_track_density)
        self.max_center_jump_ratio = float(max_center_jump_ratio)
        self.max_speed_px_per_s = float(max_speed_px_per_s)
        self.min_short_gap_iou = float(min_short_gap_iou)
        self.short_gap_frames = max(1, int(short_gap_frames))

        if _legacy_ignored:
            logger.info(
                "BotSortTracker ignoring legacy kwargs (BoT-SORT has its own gates): %s",
                sorted(_legacy_ignored.keys()),
            )

    def track(
        self,
        video_id: str,
        camera_id: Optional[str],
        detections_by_frame: dict[int, list[FrameDetection]],
    ) -> tuple[LocalTracklet, ...]:
        from boxmot.trackers.botsort.botsort import BotSort

        botsort = BotSort(
            reid_model=None,
            with_reid=False,
            cmc_method="sof",
            frame_rate=self.frame_rate,
            track_high_thresh=self.track_thresh,
            track_low_thresh=self.low_thresh,
            new_track_thresh=self.new_track_threshold,
            track_buffer=self.boxmot_track_buffer,
            match_thresh=self.match_thresh,
        )
        logger.info(
            "[botsort] %s: lost buffer=%d sampled frames (boxmot_arg=%d, fps=%d)",
            video_id,
            self.track_buffer,
            self.boxmot_track_buffer,
            self.frame_rate,
        )

        identity = np.eye(2, 3, dtype=np.float32)
        if hasattr(botsort, "cmc"):
            botsort.cmc.apply = lambda img, dets=None: identity.copy()  # type: ignore[assignment]
        dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)

        by_track: dict[int, list[TrackletObservation]] = {}

        for fk in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(fk) or [])
            if not dets:
                botsort.update(np.empty((0, 6), dtype=np.float32), dummy_img)
                continue

            arr = np.empty((len(dets), 6), dtype=np.float32)
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d.bbox
                arr[i, 0] = float(x1)
                arr[i, 1] = float(y1)
                arr[i, 2] = float(x2)
                arr[i, 3] = float(y2)
                arr[i, 4] = float(d.confidence)
                arr[i, 5] = 0.0

            try:
                results = botsort.update(arr, dummy_img)
            except Exception as exc:
                logger.warning("BoT-SORT update failed at frame %s: %s", fk, exc)
                continue

            if results is None or len(results) == 0:
                continue

            results_np = np.asarray(results)
            for row in results_np:
                if len(row) < 8:
                    continue
                tid = int(row[4])
                det_ind = int(row[7])
                if det_ind < 0 or det_ind >= len(dets):
                    continue
                src = dets[det_ind]
                obs = TrackletObservation(
                    frame_index=src.frame_index,
                    timestamp_second=src.timestamp_second,
                    bbox=src.bbox,
                    confidence=src.confidence,
                    laplacian_score=src.laplacian_score,
                    crop_bgr=src.crop_bgr,
                    is_low_quality_crop=src.is_low_quality_crop,
                )
                by_track.setdefault(tid, []).append(obs)

        completed: list[LocalTracklet] = []
        split_reasons: Counter[str] = Counter()
        dropped_fragments = 0

        for tid, observations in by_track.items():
            observations.sort(key=lambda o: o.frame_index)
            fragments, reasons = self._split_observations(observations)
            split_reasons.update(reasons)

            for frag_idx, fragment in enumerate(fragments, start=1):
                if not self._keep(fragment):
                    dropped_fragments += 1
                    continue
                out_tid = str(tid) if len(fragments) == 1 else f"{tid}_s{frag_idx}"
                completed.append(
                    LocalTracklet(
                        video_id=video_id,
                        camera_id=camera_id,
                        track_id=out_tid,
                        observations=tuple(fragment),
                    )
                )

        if split_reasons:
            logger.warning(
                "[botsort] %s: post-split guard broke %d suspicious joins (%s); kept=%d dropped=%d",
                video_id,
                sum(split_reasons.values()),
                dict(split_reasons),
                len(completed),
                dropped_fragments,
            )

        return tuple(completed)

    def _split_observations(
        self,
        observations: list[TrackletObservation],
    ) -> tuple[list[list[TrackletObservation]], Counter[str]]:
        if not observations:
            return [], Counter()

        fragments: list[list[TrackletObservation]] = [[observations[0]]]
        reasons: Counter[str] = Counter()
        prev = observations[0]
        for obs in observations[1:]:
            reason = self._split_reason(prev, obs)
            if reason:
                reasons[reason] += 1
                fragments.append([obs])
            else:
                fragments[-1].append(obs)
            prev = obs
        return fragments, reasons

    def _split_reason(self, prev: TrackletObservation, curr: TrackletObservation) -> str | None:
        gap = int(curr.frame_index - prev.frame_index)
        if gap <= 0:
            return "non_monotonic_frame"

        pcx, pcy = _bbox_center(prev.bbox)
        ccx, ccy = _bbox_center(curr.bbox)
        dist = math.hypot(ccx - pcx, ccy - pcy)
        height = max(_bbox_height(prev.bbox), _bbox_height(curr.bbox), 1.0)
        jump_ratio = dist / height

        dt = float(curr.timestamp_second - prev.timestamp_second)
        if dt <= 1e-6:
            dt = gap / float(self.frame_rate)
        speed = dist / max(dt, 1e-6)
        iou = _bbox_iou(prev.bbox, curr.bbox)
        gap_scale = float(max(gap, 1))

        if jump_ratio > self.max_center_jump_ratio * gap_scale:
            return "center_jump"
        if self.max_speed_px_per_s > 0 and speed > self.max_speed_px_per_s:
            return "speed"

        if gap <= self.short_gap_frames and iou < self.min_short_gap_iou:
            if jump_ratio > self.max_center_jump_ratio * 0.75 * gap_scale:
                return "short_gap_low_iou_jump"
            if (prev.is_low_quality_crop or curr.is_low_quality_crop) and jump_ratio > self.max_center_jump_ratio * 0.50 * gap_scale:
                return "low_quality_low_iou_jump"

        return None

    def _keep(self, observations: list[TrackletObservation]) -> bool:
        n = len(observations)
        if n < self.min_track_frames:
            return False
        span = max(observations[-1].frame_index - observations[0].frame_index + 1, 1)
        density = n / float(span)
        return density >= self.min_track_density


def is_botsort_backend() -> bool:
    """True when env selects BoT-SORT (the default)."""
    return os.environ.get("TRACER_BACKEND", "botsort").strip().lower() == "botsort"
