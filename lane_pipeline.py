import cv2
import numpy as np
from typing import Tuple, Optional
from collections import deque
import os


class LaneDetector:
    """
    Lane detection pipeline using sliding-window histogram search on a
    bird's-eye view with lane-width inference for the left boundary.

    The yellow lane paint in this footage is completely sun-bleached and
    undetectable by colour or gradient methods.  The pipeline therefore:
      - Detects the RIGHT lane (solid white line, easily visible)
      - Detects the LEFT lane (dashed centre markings when visible)
      - Infers missing left-lane data from the right lane using a known
        lane width constraint

    Pipeline stages:
      1. Multi-channel binary extraction (Sobel-X + white colour + Canny)
      2. ROI mask (crop sky/grass/hood)
      3. Perspective warp to bird's-eye view
      4. Sliding-window histogram search for lane pixel clusters
      5. Polynomial fitting + lane-width inference
      6. Validation, temporal smoothing, curvature estimation
      7. Overlay rendering back to camera view
    """

    def __init__(self, frame_resolution: Tuple[int, int]):
        self.height, self.width = frame_resolution

        # Metres-per-pixel calibration
        self.ym_per_pix = 30.0 / self.height
        self.xm_per_pix = 3.7 / (self.width * 0.55)

        # Expected lane width in warped pixels.
        # Measured via perspectiveTransform of actual lane line positions:
        # yellow line bottom (170,340) -> warped x=144
        # white line bottom  (473,340) -> warped x=499
        # Width = 499 - 144 = 355 pixels
        self.expected_lane_width_px = 355

        self.M, self.Minv = self._compute_perspective_matrices()
        self.roi_mask = self._build_roi_mask()

        # Sliding window parameters
        self.n_windows = 14
        self.window_margin = 50
        self.min_pix_recenter = 25

        # Temporal smoothing
        self.smooth_n = 8
        self.left_fit_hist = deque(maxlen=self.smooth_n)
        self.right_fit_hist = deque(maxlen=self.smooth_n)
        self.missed = 0

    # ──────────────────────── Perspective Transform ────────────────────────
    def _compute_perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.height, self.width
        src = np.float32([
            [w * 0.44, h * 0.65],
            [w * 0.57, h * 0.65],
            [w * 0.76, h * 0.95],
            [w * 0.24, h * 0.95],
        ])
        dst = np.float32([
            [w * 0.20, 0],
            [w * 0.80, 0],
            [w * 0.80, h],
            [w * 0.20, h],
        ])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

    # ──────────────────────── ROI Mask ─────────────────────────────────────
    def _build_roi_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        h, w = self.height, self.width
        pts = np.array([[
            (int(w * 0.05), int(h * 0.98)),
            (int(w * 0.42), int(h * 0.62)),
            (int(w * 0.58), int(h * 0.62)),
            (int(w * 0.95), int(h * 0.98)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    # ──────────────────────── Feature Extraction ───────────────────────────
    def _extract_lane_features(self, frame: np.ndarray) -> np.ndarray:
        # 1. HLS thresholds
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        # L-channel with CLAHE (good for bright white lines)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)
        _, l_binary = cv2.threshold(l_clahe, 190, 255, cv2.THRESH_BINARY)

        # S-channel (great for yellow lines)
        _, s_binary = cv2.threshold(s_channel, 170, 255, cv2.THRESH_BINARY)

        # 2. Sobel X on L-channel
        sobelx = cv2.Sobel(l_clahe, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_sobel = np.max(abs_sobelx)
        if max_sobel > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
        else:
            scaled_sobel = np.zeros_like(abs_sobelx, dtype=np.uint8)
        _, sobel_binary = cv2.threshold(scaled_sobel, 50, 255, cv2.THRESH_BINARY)

        # 3. Canny edges
        blurred = cv2.GaussianBlur(l_clahe, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Combine everything
        combined = np.zeros_like(l_channel)
        combined[(l_binary == 255) | (s_binary == 255) | (sobel_binary == 255) | (edges == 255)] = 255

        # Apply ROI mask
        return cv2.bitwise_and(combined, self.roi_mask)

    # ──────────────────────── Sliding Window Search ────────────────────────
    def _sliding_window_search(self, warped_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = warped_bin.shape
        histogram = np.sum(warped_bin[h // 2:, :], axis=0)
        
        # Restrict base search regions to ignore adjacent lanes or far-edge shadows.
        # Ego lane boundaries at the bottom of the image are highly predictable.
        left_min, left_max = int(self.width * 0.10), int(self.width * 0.45)
        right_min, right_max = int(self.width * 0.55), int(self.width * 0.85)

        if np.max(histogram[left_min:left_max]) > 0:
            l_base = np.argmax(histogram[left_min:left_max]) + left_min
        else:
            l_base = int(self.width * 0.25)

        if np.max(histogram[right_min:right_max]) > 0:
            r_base = np.argmax(histogram[right_min:right_max]) + right_min
        else:
            r_base = int(self.width * 0.75)

        if histogram[l_base] < 300:
            l_base = int(w * 0.25)
        if histogram[r_base] < 300:
            r_base = int(w * 0.75)

        win_h = h // self.n_windows
        nzy, nzx = warped_bin.nonzero()

        l_cur, r_cur = l_base, r_base
        l_inds, r_inds = [], []
        
        l_drift, r_drift = 0, 0

        for i in range(self.n_windows):
            y_lo = h - (i + 1) * win_h
            y_hi = h - i * win_h
            xl_lo = max(0, l_cur - self.window_margin)
            xl_hi = min(w, l_cur + self.window_margin)
            xr_lo = max(0, r_cur - self.window_margin)
            xr_hi = min(w, r_cur + self.window_margin)

            good_l = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
            good_r = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]

            l_inds.append(good_l)
            r_inds.append(good_r)

            if len(good_l) > self.min_pix_recenter:
                new_l = int(np.mean(nzx[good_l]))
                l_drift = new_l - l_cur
                l_cur = new_l
            else:
                l_cur += l_drift

            if len(good_r) > self.min_pix_recenter:
                new_r = int(np.mean(nzx[good_r]))
                r_drift = new_r - r_cur
                r_cur = new_r
            else:
                r_cur += r_drift

        l_inds = np.concatenate(l_inds) if l_inds else np.array([], dtype=int)
        r_inds = np.concatenate(r_inds) if r_inds else np.array([], dtype=int)

        l_pts = np.column_stack([nzx[l_inds], nzy[l_inds]]).astype(np.float64) if len(l_inds) > 0 else np.empty((0, 2))
        r_pts = np.column_stack([nzx[r_inds], nzy[r_inds]]).astype(np.float64) if len(r_inds) > 0 else np.empty((0, 2))
        return l_pts, r_pts

    # ──────────────────────── Polynomial Fitting ───────────────────────────
    def _fit_poly(self, pts: np.ndarray) -> Optional[np.ndarray]:
        if pts.ndim != 2 or pts.shape[0] < 40 or pts.shape[1] != 2:
            return None
        if np.ptp(pts[:, 1]) < self.height * 0.25:
            return None
        try:
            return np.polyfit(pts[:, 1], pts[:, 0], 2)
        except (np.linalg.LinAlgError, ValueError):
            return None

    # ──────────────────────── Lane Width Inference ─────────────────────────
    def _infer_missing_lane(self, detected_fit: np.ndarray,
                            is_right: bool) -> np.ndarray:
        """
        Generate the polynomial for the missing lane by offsetting the
        detected lane by the expected lane width.
        """
        offset = self.expected_lane_width_px
        inferred = detected_fit.copy()
        if is_right:
            # Detected is right → infer left by subtracting offset
            inferred[2] -= offset
        else:
            # Detected is left → infer right by adding offset
            inferred[2] += offset
        return inferred

    # ──────────────────────── History Validation ───────────────────────────
    def _validate_and_smooth(self, left_fit: Optional[np.ndarray],
                             right_fit: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        valid_l = False
        valid_r = False

        if left_fit is not None:
            lb = np.polyval(left_fit, self.height)
            if self.width * 0.05 < lb < self.width * 0.50:
                valid_l = True
                
        if right_fit is not None:
            rb = np.polyval(right_fit, self.height)
            if self.width * 0.50 < rb < self.width * 0.95:
                valid_r = True

        if valid_l and valid_r:
            w_bot = np.polyval(right_fit, self.height) - np.polyval(left_fit, self.height)
            w_top = np.polyval(right_fit, 0) - np.polyval(left_fit, 0)
            if not (200 < w_bot < 550 and 200 < w_top < 550):
                valid_l = False  # Trust the right lane more since it's higher contrast white

        if valid_r and not valid_l:
            left_fit = self._infer_missing_lane(right_fit, is_right=True)
            valid_l = True
        elif valid_l and not valid_r:
            right_fit = self._infer_missing_lane(left_fit, is_right=False)
            valid_r = True

        if valid_l and valid_r:
            self.left_fit_hist.append(left_fit)
            self.right_fit_hist.append(right_fit)
            self.missed = 0
        else:
            self.missed += 1

        if self.missed > self.smooth_n * 2:
            self.left_fit_hist.clear()
            self.right_fit_hist.clear()

        lf = np.mean(self.left_fit_hist, axis=0) if self.left_fit_hist else None
        rf = np.mean(self.right_fit_hist, axis=0) if self.right_fit_hist else None
        return lf, rf

    # ──────────────────────── Curvature ────────────────────────────────────
    def _curvature(self, lf: np.ndarray, rf: np.ndarray, y_eval: float) -> float:
        ys = np.linspace(0, y_eval, 50)
        lx = np.polyval(lf, ys)
        rx = np.polyval(rf, ys)

        lcr = np.polyfit(ys * self.ym_per_pix, lx * self.xm_per_pix, 2)
        rcr = np.polyfit(ys * self.ym_per_pix, rx * self.xm_per_pix, 2)

        ye = y_eval * self.ym_per_pix
        rads = []
        for c in [lcr, rcr]:
            d = np.abs(2.0 * c[0])
            rads.append(((1 + (2*c[0]*ye + c[1])**2)**1.5) / d if d > 1e-6 else 10_000.0)
        return float(np.mean(rads))

    # ──────────────────────── Process Frame ────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        feat = self._extract_lane_features(frame)
        warped = cv2.warpPerspective(feat, self.M, (self.width, self.height))

        l_pts, r_pts = self._sliding_window_search(warped)
        l_raw = self._fit_poly(l_pts)
        r_raw = self._fit_poly(r_pts)

        lf, rf = self._validate_and_smooth(l_raw, r_raw)

        overlay = np.zeros_like(frame)
        curv = 0.0

        if lf is not None and rf is not None:
            py = np.linspace(0, self.height - 1, self.height)
            lx = np.clip(np.polyval(lf, py), 0, self.width - 1)
            rx = np.clip(np.polyval(rf, py), 0, self.width - 1)

            pts_l = np.array([np.column_stack([lx, py])])
            pts_r = np.array([np.flipud(np.column_stack([rx, py]))])
            pts = np.hstack((pts_l, pts_r))

            cv2.fillPoly(overlay, np.int32(pts), (0, 255, 0))
            cv2.polylines(overlay, [np.int32(np.column_stack([lx, py]))], False, (0, 0, 255), 4)
            cv2.polylines(overlay, [np.int32(np.column_stack([rx, py]))], False, (255, 0, 0), 4)
            curv = self._curvature(lf, rf, self.height)

        uw = cv2.warpPerspective(overlay, self.Minv, (self.width, self.height))
        out = cv2.addWeighted(frame, 1.0, uw, 0.4, 0)

        if lf is None or rf is None:
            txt, clr = "LANE LOST - DEAD RECKONING", (0, 0, 255)
        elif curv > 5000:
            txt, clr = "Curve Radius: ~Straight", (0, 255, 0)
        else:
            txt, clr = f"Curve Radius: {curv:.0f}m", (0, 255, 0)

        cv2.putText(out, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2, cv2.LINE_AA)
        return out


def run_simulation(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open stream at {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    pipeline = LaneDetector((h, w))

    out_path = os.path.splitext(video_path)[0] + "_output.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = pipeline.process_frame(frame)
        writer.write(processed)
        n += 1
        cv2.imshow("AV Lane Detection Pipeline", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {n} frames. Output saved to: {out_path}")


if __name__ == "__main__":
    run_simulation("project_video.mp4")