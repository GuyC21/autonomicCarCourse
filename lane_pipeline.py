import os
from collections import deque
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class LaneDetector:
    """
    Lane detection pipeline for recorded footage or a CARLA camera feed.

    The detector follows the requested OpenCV pipeline:
      1. Build lane evidence with color thresholds, gradients, and Canny edges.
      2. Restrict the search to a road ROI.
      3. Warp the ROI to a bird's-eye view for stable geometry.
      4. Run a probabilistic Hough transform on Canny edges.
      5. Use Hough segments to seed lane searches and polynomial fits.
      6. Validate/smooth the lane pair and infer a temporarily missing side.
      7. Render the lane overlay and estimate lane curvature.
    """

    def __init__(self, frame_resolution: Tuple[int, int]):
        self.height, self.width = frame_resolution

        # The destination perspective transform maps a 3.7 m lane to this
        # many pixels.  For the project video resolution this remains 355 px,
        # matching the previous calibration while making the value scalable.
        self.expected_lane_width_px = int(round(self.width * 0.555))
        self.nominal_lane_width_px = float(self.expected_lane_width_px)
        self.min_lane_width_px = int(round(self.expected_lane_width_px * 0.62))
        self.max_lane_width_px = int(round(self.expected_lane_width_px * 1.42))

        self.ym_per_pix = 30.0 / self.height
        self.xm_per_pix = 3.7 / max(self.expected_lane_width_px, 1)

        self.M, self.Minv = self._compute_perspective_matrices()
        self.roi_mask = self._build_roi_mask()

        self.n_windows = 12
        self.window_margin = max(32, int(self.width * 0.075))
        self.min_pix_recenter = max(12, int(self.width * 0.025))

        self.hough_threshold = max(12, int(self.width * 0.025))
        self.hough_min_line_length = max(14, int(self.height * 0.045))
        self.hough_max_line_gap = max(18, int(self.height * 0.070))
        self.max_warped_slope = 1.10

        self.smooth_n = 8
        self.left_fit_hist = deque(maxlen=self.smooth_n)
        self.right_fit_hist = deque(maxlen=self.smooth_n)
        self.lane_width_hist = deque(maxlen=self.smooth_n * 2)
        self.curvature_hist = deque(maxlen=self.smooth_n)
        self.missed = 0
        self.last_status = "LOST"
        self.last_left_base = None
        self.last_right_base = None
        self.prev_render_left_x = None
        self.prev_render_right_x = None

    # Perspective transform
    def _compute_perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.height, self.width
        lane_ratio = self.expected_lane_width_px / float(w)
        dst_margin = (1.0 - lane_ratio) / 2.0

        src = np.float32([
            [w * 0.440, h * 0.650],
            [w * 0.535, h * 0.650],
            [w * 0.720, h * 0.990],
            [w * 0.270, h * 0.990],
        ])
        dst = np.float32([
            [w * dst_margin, 0],
            [w * (1.0 - dst_margin), 0],
            [w * (1.0 - dst_margin), h],
            [w * dst_margin, h],
        ])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

    # ROI mask
    def _build_roi_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        h, w = self.height, self.width
        pts = np.array([[
            (int(w * 0.03), int(h * 0.99)),
            (int(w * 0.40), int(h * 0.58)),
            (int(w * 0.60), int(h * 0.58)),
            (int(w * 0.98), int(h * 0.99)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    # Feature extraction
    def _lane_color_mask(self, frame: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h_channel, l_channel, s_channel = cv2.split(hls)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l_channel)

        roi_values = l_equalized[self.roi_mask > 0]
        white_thresh = 175
        if roi_values.size:
            white_thresh = max(white_thresh, int(np.percentile(roi_values, 88)))

        white = ((l_equalized >= white_thresh) & (s_channel <= 145)) | (l_channel >= 220)
        yellow = (
            (h_channel >= 12) & (h_channel <= 40) &
            (s_channel >= 65) & (l_channel >= 70)
        )

        mask = np.zeros_like(l_channel, dtype=np.uint8)
        mask[white | yellow] = 255
        mask = cv2.bitwise_and(mask, self.roi_mask)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _extract_lane_features(self, frame: np.ndarray) -> np.ndarray:
        color_mask = self._lane_color_mask(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)

        roi_values = blurred[self.roi_mask > 0]
        median = float(np.median(roi_values)) if roi_values.size else 0.0
        lower = max(40, int(0.66 * median))
        upper = max(95, int(1.33 * median))
        if upper <= lower:
            upper = lower + 55
        canny = cv2.Canny(blurred, lower, upper)
        canny = cv2.bitwise_and(canny, self.roi_mask)

        sobelx = cv2.Sobel(gray_equalized, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_sobel = np.max(abs_sobelx)
        if max_sobel > 0:
            scaled_sobel = np.uint8(np.clip(255 * abs_sobelx / max_sobel, 0, 255))
        else:
            scaled_sobel = np.zeros_like(gray, dtype=np.uint8)
        _, sobel_binary = cv2.threshold(scaled_sobel, 45, 255, cv2.THRESH_BINARY)
        sobel_binary = cv2.bitwise_and(sobel_binary, self.roi_mask)

        combined = cv2.bitwise_or(color_mask, canny)
        combined = cv2.bitwise_or(combined, sobel_binary)

        kernel = np.ones((3, 3), dtype=np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.bitwise_and(combined, self.roi_mask)

    # Hough transform and lane candidate selection
    def _hough_segments(self, warped_binary: np.ndarray) -> np.ndarray:
        if np.count_nonzero(warped_binary) == 0:
            return np.empty((0, 4), dtype=np.int32)

        blurred = cv2.GaussianBlur(warped_binary, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is None:
            return np.empty((0, 4), dtype=np.int32)
        return lines.reshape(-1, 4)

    def _segment_bottom_x(self, segment: Sequence[int]) -> Optional[float]:
        x1, y1, x2, y2 = [float(v) for v in segment]
        dy = y2 - y1
        dx = x2 - x1
        if abs(dy) < 1.0:
            return None

        slope = dx / dy
        length = float(np.hypot(dx, dy))
        if length < self.hough_min_line_length or abs(slope) > self.max_warped_slope:
            return None

        bottom_x = x1 + slope * ((self.height - 1) - y1)
        if -self.width * 0.20 <= bottom_x <= self.width * 1.20:
            return float(bottom_x)
        return None

    def _filtered_hough_segments(self, warped_binary: np.ndarray) -> np.ndarray:
        segments = self._hough_segments(warped_binary)
        if len(segments) == 0:
            return segments

        kept: List[np.ndarray] = []
        for segment in segments:
            if self._segment_bottom_x(segment) is not None:
                kept.append(segment)
        if not kept:
            return np.empty((0, 4), dtype=np.int32)
        return np.asarray(kept, dtype=np.int32)

    def _current_lane_width(self) -> float:
        if not self.lane_width_hist:
            return self.nominal_lane_width_px
        width = float(np.median(self.lane_width_hist))
        low = self.nominal_lane_width_px * 0.90
        high = self.nominal_lane_width_px * 1.10
        return float(np.clip(width, low, high))

    def _average_fit(self, history: deque) -> Optional[np.ndarray]:
        if not history:
            return None
        fits = np.asarray(history, dtype=np.float64)
        weights = np.linspace(1.0, 2.0, len(fits))
        return np.average(fits, axis=0, weights=weights)

    def _previous_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._average_fit(self.left_fit_hist), self._average_fit(self.right_fit_hist)

    def _histogram(self, warped_binary: np.ndarray) -> np.ndarray:
        lower = warped_binary[int(self.height * 0.55):, :] > 0
        hist = np.sum(lower, axis=0).astype(np.float32)
        kernel_size = max(9, int(self.width * 0.035))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        return np.convolve(hist, kernel, mode="same")

    def _histogram_peaks(self, hist: np.ndarray, low: int, high: int,
                         max_peaks: int = 4) -> List[float]:
        low = max(0, low)
        high = min(self.width, high)
        if high <= low:
            return []

        work = hist.copy()
        peaks: List[float] = []
        global_peak = float(np.max(hist)) if hist.size else 0.0
        min_peak = max(6.0, global_peak * 0.16)
        suppression = max(18, int(self.width * 0.050))

        for _ in range(max_peaks):
            region = work[low:high]
            if region.size == 0:
                break
            local_peak = float(np.max(region))
            if local_peak < min_peak:
                break
            x = int(np.argmax(region)) + low
            peaks.append(float(x))
            x0 = max(0, x - suppression)
            x1 = min(self.width, x + suppression + 1)
            work[x0:x1] = 0
        return peaks

    def _dedupe_candidates(self, values: Sequence[float]) -> List[float]:
        if not values:
            return []
        values = sorted(float(v) for v in values if np.isfinite(v))
        merged: List[List[float]] = []
        merge_dist = max(10.0, self.width * 0.025)
        for value in values:
            if not merged or abs(value - np.mean(merged[-1])) > merge_dist:
                merged.append([value])
            else:
                merged[-1].append(value)
        return [float(np.mean(group)) for group in merged]

    def _choose_base_pair(self, warped_binary: np.ndarray,
                          segments: Optional[np.ndarray] = None) -> Tuple[float, float]:
        hist = self._histogram(warped_binary)
        expected = self._current_lane_width()
        prev_l, prev_r = self._previous_pair()

        if prev_l is not None and prev_r is not None:
            target_center = (
                np.polyval(prev_l, self.height - 1) +
                np.polyval(prev_r, self.height - 1)
            ) / 2.0
            prev_left_base = float(np.polyval(prev_l, self.height - 1))
            prev_right_base = float(np.polyval(prev_r, self.height - 1))
        else:
            target_center = self.width / 2.0
            prev_left_base = target_center - expected / 2.0
            prev_right_base = target_center + expected / 2.0

        segment_bases: List[float] = []
        if segments is not None:
            for segment in segments:
                base = self._segment_bottom_x(segment)
                if base is not None:
                    segment_bases.append(base)

        left_values = self._histogram_peaks(hist, int(self.width * 0.03), int(self.width * 0.58))
        right_values = self._histogram_peaks(hist, int(self.width * 0.42), int(self.width * 0.88))
        left_values += [base for base in segment_bases if base < target_center + expected * 0.20]
        right_values += [
            base for base in segment_bases
            if (base > target_center - expected * 0.20 and base < self.width * 0.88)
        ]

        left_candidates = self._dedupe_candidates(left_values)
        right_candidates = self._dedupe_candidates(right_values)

        if not left_candidates:
            left_candidates = [prev_left_base]
        if not right_candidates:
            right_candidates = [prev_right_base]

        best_pair: Optional[Tuple[float, float]] = None
        best_score = float("inf")
        min_width = max(float(self.min_lane_width_px), expected * 0.62)
        max_width = min(float(self.max_lane_width_px), expected * 1.42)

        for left_base in left_candidates:
            for right_base in right_candidates:
                lane_width = right_base - left_base
                if lane_width <= 0 or lane_width < min_width or lane_width > max_width:
                    continue
                center = (left_base + right_base) / 2.0
                score = abs(lane_width - expected) * 1.25 + abs(center - target_center) * 0.90
                score += abs(left_base - prev_left_base) * 0.20
                score += abs(right_base - prev_right_base) * 0.20
                # Strongly discourage snapping to the road edge on the far right.
                if right_base > self.width * 0.84:
                    over = right_base - self.width * 0.84
                    score += over * 7.0 + 0.10 * over * over
                if left_base > self.width * 0.58:
                    score += (left_base - self.width * 0.58) * 3.0
                li = int(np.clip(round(left_base), 0, self.width - 1))
                ri = int(np.clip(round(right_base), 0, self.width - 1))
                score -= min(20.0, float(hist[li] + hist[ri]) * 0.04)
                if score < best_score:
                    best_score = score
                    best_pair = (float(left_base), float(right_base))

        if best_pair is not None:
            return best_pair

        if left_candidates and not right_candidates:
            left_base = min(left_candidates, key=lambda x: abs(x - (target_center - expected / 2.0)))
            return float(left_base), float(left_base + expected)
        if right_candidates and not left_candidates:
            right_base = min(right_candidates, key=lambda x: abs(x - (target_center + expected / 2.0)))
            return float(right_base - expected), float(right_base)
        return float(prev_left_base), float(prev_right_base)

    def _sample_segments_near_base(self, segments: np.ndarray, base_x: float,
                                   margin: float) -> np.ndarray:
        if len(segments) == 0:
            return np.empty((0, 2), dtype=np.float64)

        sampled: List[np.ndarray] = []
        for segment in segments:
            bottom_x = self._segment_bottom_x(segment)
            if bottom_x is None or abs(bottom_x - base_x) > margin:
                continue
            x1, y1, x2, y2 = [float(v) for v in segment]
            length = float(np.hypot(x2 - x1, y2 - y1))
            n = max(2, int(length / 6.0))
            xs = np.linspace(x1, x2, n)
            ys = np.linspace(y1, y2, n)
            sampled.append(np.column_stack([xs, ys]))

        if not sampled:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(sampled).astype(np.float64)

    # Sliding-window search, seeded by Hough bases when available.
    def _sliding_window_search(self, warped_bin: np.ndarray,
                               left_base: Optional[float] = None,
                               right_base: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        h, w = warped_bin.shape
        nzy, nzx = warped_bin.nonzero()
        if len(nzx) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        if left_base is None or right_base is None:
            left_base, right_base = self._choose_base_pair(warped_bin)

        left_cur = int(np.clip(round(left_base), 0, w - 1))
        right_cur = int(np.clip(round(right_base), 0, w - 1))
        expected = self._current_lane_width()

        win_h = max(1, h // self.n_windows)
        l_inds: List[np.ndarray] = []
        r_inds: List[np.ndarray] = []
        l_drift = 0
        r_drift = 0

        for i in range(self.n_windows):
            y_lo = h - (i + 1) * win_h
            y_hi = h - i * win_h if i > 0 else h
            xl_lo = max(0, left_cur - self.window_margin)
            xl_hi = min(w, left_cur + self.window_margin)
            xr_lo = max(0, right_cur - self.window_margin)
            xr_hi = min(w, right_cur + self.window_margin)

            good_l = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
            good_r = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]

            l_inds.append(good_l)
            r_inds.append(good_r)

            if len(good_l) > self.min_pix_recenter:
                new_l = int(np.mean(nzx[good_l]))
                l_drift = int(np.clip(new_l - left_cur, -self.window_margin, self.window_margin))
                left_cur = new_l
            else:
                left_cur = int(np.clip(left_cur + l_drift, 0, w - 1))

            if len(good_r) > self.min_pix_recenter:
                new_r = int(np.mean(nzx[good_r]))
                r_drift = int(np.clip(new_r - right_cur, -self.window_margin, self.window_margin))
                right_cur = new_r
            else:
                right_cur = int(np.clip(right_cur + r_drift, 0, w - 1))

            current_width = right_cur - left_cur
            if current_width < expected * 0.55 or current_width > expected * 1.55:
                lane_center = (left_cur + right_cur) / 2.0
                left_cur = int(np.clip(lane_center - expected / 2.0, 0, w - 1))
                right_cur = int(np.clip(lane_center + expected / 2.0, 0, w - 1))

        l_all = np.concatenate(l_inds) if l_inds else np.array([], dtype=int)
        r_all = np.concatenate(r_inds) if r_inds else np.array([], dtype=int)

        left_pts = np.column_stack([nzx[l_all], nzy[l_all]]).astype(np.float64) if len(l_all) else np.empty((0, 2))
        right_pts = np.column_stack([nzx[r_all], nzy[r_all]]).astype(np.float64) if len(r_all) else np.empty((0, 2))
        return left_pts, right_pts

    def _hough_lane_search(self, warped_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        segments = self._filtered_hough_segments(warped_bin)
        left_base, right_base = self._choose_base_pair(warped_bin, segments)
        self.last_left_base = float(left_base)
        self.last_right_base = float(right_base)

        left_slide, right_slide = self._sliding_window_search(warped_bin, left_base, right_base)

        hough_margin = max(self.window_margin * 1.35, self._current_lane_width() * 0.18)
        left_hough = self._sample_segments_near_base(segments, left_base, hough_margin)
        right_hough = self._sample_segments_near_base(segments, right_base, hough_margin)

        if len(left_hough):
            left_pts = np.vstack([left_slide, left_hough]) if len(left_slide) else left_hough
        else:
            left_pts = left_slide
        if len(right_hough):
            right_pts = np.vstack([right_slide, right_hough]) if len(right_slide) else right_hough
        else:
            right_pts = right_slide
        return left_pts, right_pts

    # Polynomial fitting and validation
    def _fit_poly(self, pts: np.ndarray) -> Optional[np.ndarray]:
        if pts.ndim != 2 or pts.shape[0] < 35 or pts.shape[1] != 2:
            return None
        if np.ptp(pts[:, 1]) < self.height * 0.22:
            return None

        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        try:
            fit = np.polyfit(y, x, 2)
            residual = np.abs(x - np.polyval(fit, y))
            cutoff = max(14.0, float(np.percentile(residual, 75)) * 2.75)
            keep = residual <= cutoff
            if np.count_nonzero(keep) >= max(35, int(0.55 * len(pts))):
                fit = np.polyfit(y[keep], x[keep], 2)
            return fit
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return None

    def _infer_missing_lane(self, detected_fit: np.ndarray,
                            is_right: bool) -> np.ndarray:
        inferred = detected_fit.copy()
        offset = self._current_lane_width()
        if is_right:
            inferred[2] -= offset
        else:
            inferred[2] += offset
        return inferred

    def _anchor_fit_bottom(self, fit: Optional[np.ndarray], base_x: Optional[float]) -> Optional[np.ndarray]:
        if fit is None or base_x is None or not np.isfinite(base_x):
            return fit
        anchored = fit.copy()
        bottom_error = float(base_x - np.polyval(anchored, self.height - 1))
        anchored[2] += bottom_error
        return anchored

    def _anchor_pair_to_hough_bases(self, left_fit: Optional[np.ndarray],
                                    right_fit: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.last_left_base is None or self.last_right_base is None:
            return left_fit, right_fit
        base_width = self.last_right_base - self.last_left_base
        if not (self.min_lane_width_px <= base_width <= self.max_lane_width_px):
            return left_fit, right_fit
        return (
            self._anchor_fit_bottom(left_fit, self.last_left_base),
            self._anchor_fit_bottom(right_fit, self.last_right_base),
        )

    def _straight_pair_from_hough_bases(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.last_left_base is None or self.last_right_base is None:
            return None, None
        base_width = self.last_right_base - self.last_left_base
        if not (self.min_lane_width_px <= base_width <= self.max_lane_width_px):
            return None, None
        return (
            np.array([0.0, 0.0, self.last_left_base], dtype=np.float64),
            np.array([0.0, 0.0, self.last_right_base], dtype=np.float64),
        )

    def _valid_single_fit(self, fit: Optional[np.ndarray], side: str) -> bool:
        if fit is None or not np.all(np.isfinite(fit)):
            return False

        ys = np.array([self.height * 0.35, self.height * 0.60, self.height - 1], dtype=np.float64)
        xs = np.polyval(fit, ys)
        if not np.all(np.isfinite(xs)):
            return False
        if np.any(xs < -self.width * 0.12) or np.any(xs > self.width * 1.12):
            return False

        bottom_x = float(xs[-1])
        if side == "left" and not (self.width * 0.02 < bottom_x < self.width * 0.58):
            return False
        if side == "right" and not (self.width * 0.38 < bottom_x < self.width * 0.88):
            return False

        derivatives = np.abs(2.0 * fit[0] * ys + fit[1])
        if np.max(derivatives) > self.max_warped_slope * 1.35:
            return False
        if np.ptp(xs) > self.width * 0.58:
            return False
        return True

    def _valid_lane_pair(self, left_fit: np.ndarray, right_fit: np.ndarray) -> bool:
        ys = np.array([self.height * 0.35, self.height * 0.60, self.height - 1], dtype=np.float64)
        widths = np.polyval(right_fit, ys) - np.polyval(left_fit, ys)
        if not np.all(np.isfinite(widths)):
            return False

        expected = self._current_lane_width()
        if np.any(widths < max(self.min_lane_width_px, expected * 0.62)):
            return False
        if np.any(widths > min(self.max_lane_width_px, expected * 1.42)):
            return False

        bottom_center = (
            np.polyval(left_fit, self.height - 1) +
            np.polyval(right_fit, self.height - 1)
        ) / 2.0
        return bool(self.width * 0.12 < bottom_center < self.width * 0.78)

    def _fit_deviation(self, fit: np.ndarray, reference: np.ndarray) -> float:
        ys = np.linspace(self.height * 0.35, self.height - 1, 5)
        return float(np.mean(np.abs(np.polyval(fit, ys) - np.polyval(reference, ys))))

    def _update_lane_width(self, left_fit: np.ndarray, right_fit: np.ndarray) -> None:
        ys = np.array([self.height * 0.45, self.height * 0.70, self.height - 1], dtype=np.float64)
        widths = np.polyval(right_fit, ys) - np.polyval(left_fit, ys)
        low = self.nominal_lane_width_px * 0.88
        high = self.nominal_lane_width_px * 1.12
        valid = widths[(widths > low) & (widths < high)]
        if valid.size:
            self.lane_width_hist.append(float(np.median(valid)))
            self.xm_per_pix = 3.7 / max(self._current_lane_width(), 1.0)

    def _validate_and_smooth(self, left_fit: Optional[np.ndarray],
                             right_fit: Optional[np.ndarray],
                             left_conf: int = 0,
                             right_conf: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        prev_l, prev_r = self._previous_pair()

        valid_l = self._valid_single_fit(left_fit, "left")
        valid_r = self._valid_single_fit(right_fit, "right")

        if valid_l and valid_r and not self._valid_lane_pair(left_fit, right_fit):
            anchored_l, anchored_r = self._anchor_pair_to_hough_bases(left_fit, right_fit)
            if (
                anchored_l is not left_fit or anchored_r is not right_fit
            ) and self._valid_single_fit(anchored_l, "left") and self._valid_single_fit(anchored_r, "right") and self._valid_lane_pair(anchored_l, anchored_r):
                left_fit, right_fit = anchored_l, anchored_r
                valid_l, valid_r = True, True
            elif prev_l is None and prev_r is None:
                straight_l, straight_r = self._straight_pair_from_hough_bases()
                if (
                    self._valid_single_fit(straight_l, "left") and
                    self._valid_single_fit(straight_r, "right") and
                    self._valid_lane_pair(straight_l, straight_r)
                ):
                    left_fit, right_fit = straight_l, straight_r
                    valid_l, valid_r = True, True

        if valid_l and valid_r and not self._valid_lane_pair(left_fit, right_fit):
            if prev_l is not None and prev_r is not None:
                l_dev = self._fit_deviation(left_fit, prev_l)
                r_dev = self._fit_deviation(right_fit, prev_r)
                if l_dev <= r_dev:
                    right_fit = self._infer_missing_lane(left_fit, is_right=False)
                    valid_r = self._valid_single_fit(right_fit, "right")
                else:
                    left_fit = self._infer_missing_lane(right_fit, is_right=True)
                    valid_l = self._valid_single_fit(left_fit, "left")
            elif right_conf >= left_conf:
                left_fit = self._infer_missing_lane(right_fit, is_right=True)
                valid_l = self._valid_single_fit(left_fit, "left")
            else:
                right_fit = self._infer_missing_lane(left_fit, is_right=False)
                valid_r = self._valid_single_fit(right_fit, "right")

        if valid_r and not valid_l:
            left_fit = self._infer_missing_lane(right_fit, is_right=True)
            valid_l = self._valid_single_fit(left_fit, "left")
        elif valid_l and not valid_r:
            right_fit = self._infer_missing_lane(left_fit, is_right=False)
            valid_r = self._valid_single_fit(right_fit, "right")

        if valid_l and valid_r:
            center_bottom = (
                float(np.polyval(left_fit, self.height - 1)) +
                float(np.polyval(right_fit, self.height - 1))
            ) / 2.0
            # In left-lane driving, a truck often hides the right line.
            # Prefer a stable left-line fit and infer the missing/noisy right side.
            if center_bottom < self.width * 0.45 and right_conf < left_conf * 0.75:
                inferred_r = self._infer_missing_lane(left_fit, is_right=False)
                if self._valid_single_fit(inferred_r, "right"):
                    right_fit = inferred_r
            # Mirror case for right-lane driving.
            elif center_bottom > self.width * 0.55 and left_conf < right_conf * 0.75:
                inferred_l = self._infer_missing_lane(right_fit, is_right=True)
                if self._valid_single_fit(inferred_l, "left"):
                    left_fit = inferred_l

        pair_ok = bool(valid_l and valid_r and self._valid_lane_pair(left_fit, right_fit))

        if pair_ok and prev_l is not None and prev_r is not None:
            max_dev = max(55.0, self._current_lane_width() * 0.22)
            l_dev = self._fit_deviation(left_fit, prev_l)
            r_dev = self._fit_deviation(right_fit, prev_r)
            if l_dev > max_dev and r_dev > max_dev:
                prev_lb = float(np.polyval(prev_l, self.height - 1))
                prev_rb = float(np.polyval(prev_r, self.height - 1))
                new_lb = float(np.polyval(left_fit, self.height - 1))
                new_rb = float(np.polyval(right_fit, self.height - 1))
                shift_l = new_lb - prev_lb
                shift_r = new_rb - prev_rb
                same_direction = np.sign(shift_l) == np.sign(shift_r)
                coherent_shift = abs(shift_l - shift_r) < max(22.0, self._current_lane_width() * 0.08)
                width_ok = self._valid_lane_pair(left_fit, right_fit)
                # Allow coherent lateral motion of both lines (lane-change behavior).
                if same_direction and coherent_shift and width_ok:
                    pair_ok = True
                else:
                    pair_ok = False
            elif l_dev > max_dev:
                left_fit = self._infer_missing_lane(right_fit, is_right=True)
                pair_ok = self._valid_lane_pair(left_fit, right_fit)
            elif r_dev > max_dev:
                right_fit = self._infer_missing_lane(left_fit, is_right=False)
                pair_ok = self._valid_lane_pair(left_fit, right_fit)

        if pair_ok:
            self.left_fit_hist.append(left_fit)
            self.right_fit_hist.append(right_fit)
            self._update_lane_width(left_fit, right_fit)
            self.missed = 0
            self.last_status = "TRACKING"
        else:
            self.missed += 1
            self.last_status = "DEAD_RECKONING" if self.left_fit_hist and self.right_fit_hist else "LOST"
            if self.missed > self.smooth_n * 2:
                self.left_fit_hist.clear()
                self.right_fit_hist.clear()
                self.lane_width_hist.clear()
                self.curvature_hist.clear()
                self.last_status = "LOST"

        return self._previous_pair()

    # Curvature and rendering
    def _curvature(self, lf: np.ndarray, rf: np.ndarray, y_eval: float) -> float:
        ys = np.linspace(self.height * 0.35, y_eval, 80)
        lx = np.polyval(lf, ys)
        rx = np.polyval(rf, ys)

        lcr = np.polyfit(ys * self.ym_per_pix, lx * self.xm_per_pix, 2)
        rcr = np.polyfit(ys * self.ym_per_pix, rx * self.xm_per_pix, 2)

        ye = y_eval * self.ym_per_pix
        radii = []
        for coeffs in (lcr, rcr):
            denom = abs(2.0 * coeffs[0])
            if denom <= 1e-6:
                radii.append(10_000.0)
            else:
                radii.append(((1 + (2 * coeffs[0] * ye + coeffs[1]) ** 2) ** 1.5) / denom)
        return float(np.mean(radii))

    def _smooth_curvature(self, curvature: float) -> float:
        if not np.isfinite(curvature):
            return 0.0

        curvature = float(np.clip(curvature, 50.0, 10_000.0))
        if self.curvature_hist:
            previous = float(np.median(self.curvature_hist))
            lower = max(50.0, previous * 0.45)
            upper = min(10_000.0, previous * 2.20)
            curvature = float(np.clip(curvature, lower, upper))

        self.curvature_hist.append(curvature)
        return float(np.median(self.curvature_hist))

    def _stabilize_render_points(self, lf: np.ndarray, rf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        py = np.linspace(0, self.height - 1, self.height)
        lx = np.polyval(lf, py)
        rx = np.polyval(rf, py)

        if self.last_left_base is not None and self.last_right_base is not None:
            near = py >= self.height * 0.78
            blend = np.zeros_like(py, dtype=np.float64)
            blend[near] = (py[near] - self.height * 0.78) / max(self.height * 0.22, 1.0)
            blend = np.clip(blend, 0.0, 1.0) ** 1.7

            lx = lx * (1.0 - blend) + self.last_left_base * blend
            rx = rx * (1.0 - blend) + self.last_right_base * blend

        min_width = self.expected_lane_width_px * 0.70
        max_width = self.expected_lane_width_px * 1.35
        lane_width = rx - lx
        bad_width = (lane_width < min_width) | (lane_width > max_width)
        if np.any(bad_width):
            center = (lx + rx) / 2.0
            target_width = np.clip(lane_width, min_width, max_width)
            lx[bad_width] = center[bad_width] - target_width[bad_width] / 2.0
            rx[bad_width] = center[bad_width] + target_width[bad_width] / 2.0

        lx = np.clip(lx, 0, self.width - 1)
        rx = np.clip(rx, 0, self.width - 1)
        return py, lx, rx

    def _temporal_render_filter(self, lx: np.ndarray, rx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prev_render_left_x is None or self.prev_render_right_x is None:
            self.prev_render_left_x = lx.copy()
            self.prev_render_right_x = rx.copy()
            return lx, rx

        prev_l = self.prev_render_left_x
        prev_r = self.prev_render_right_x
        if prev_l.shape != lx.shape or prev_r.shape != rx.shape:
            self.prev_render_left_x = lx.copy()
            self.prev_render_right_x = rx.copy()
            return lx, rx

        jump = float(np.median(np.abs(lx - prev_l) + np.abs(rx - prev_r)) * 0.5)
        if jump > 35.0:
            alpha = 0.25
        elif jump > 20.0:
            alpha = 0.45
        else:
            alpha = 0.70

        filtered_l = alpha * lx + (1.0 - alpha) * prev_l
        filtered_r = alpha * rx + (1.0 - alpha) * prev_r
        self.prev_render_left_x = filtered_l.copy()
        self.prev_render_right_x = filtered_r.copy()
        return filtered_l, filtered_r

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        features = self._extract_lane_features(frame)
        warped = cv2.warpPerspective(features, self.M, (self.width, self.height))

        left_pts, right_pts = self._hough_lane_search(warped)
        left_raw = self._fit_poly(left_pts)
        right_raw = self._fit_poly(right_pts)

        lf, rf = self._validate_and_smooth(left_raw, right_raw, len(left_pts), len(right_pts))
        if self.last_status != "TRACKING":
            # Never render a stale polygon during low-confidence tracking.
            lf, rf = None, None

        overlay = np.zeros_like(frame)
        curvature = 0.0

        if lf is not None and rf is not None:
            py, lx, rx = self._stabilize_render_points(lf, rf)
            lx, rx = self._temporal_render_filter(lx, rx)

            pts_l = np.array([np.column_stack([lx, py])])
            pts_r = np.array([np.flipud(np.column_stack([rx, py]))])
            lane_poly = np.hstack((pts_l, pts_r))

            cv2.fillPoly(overlay, np.int32(lane_poly), (0, 255, 0))
            cv2.polylines(overlay, [np.int32(np.column_stack([lx, py]))], False, (0, 0, 255), 4)
            cv2.polylines(overlay, [np.int32(np.column_stack([rx, py]))], False, (255, 0, 0), 4)
            curvature = self._smooth_curvature(self._curvature(lf, rf, self.height - 1))

        unwarped = cv2.warpPerspective(overlay, self.Minv, (self.width, self.height))
        out = cv2.addWeighted(frame, 1.0, unwarped, 0.4, 0)

        if lf is None or rf is None:
            self.prev_render_left_x = None
            self.prev_render_right_x = None
            text, color = ("TEMPORARY OCCLUSION", (0, 165, 255)) if self.last_status == "DEAD_RECKONING" else ("LANE LOST", (0, 0, 255))
        elif curvature > 5000:
            text, color = "Curve Radius: ~Straight", (0, 255, 0)
        else:
            text, color = f"Curve Radius: {curvature:.0f}m", (0, 255, 0)

        cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return out


def run_simulation(video_path: str, output_path: Optional[str] = None,
                   display: bool = True) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open stream at {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    pipeline = LaneDetector((h, w))

    out_path = output_path or (os.path.splitext(video_path)[0] + "_output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Failed to open output writer at {out_path}")

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = pipeline.process_frame(frame)
        writer.write(processed)
        n += 1

        if display:
            cv2.imshow("AV Lane Detection Pipeline", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Processed {n} frames. Output saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    run_simulation("project_video.mp4")
