import cv2
import numpy as np
from typing import Tuple, Optional


class LaneDetector:
    """
    A robust lane detection pipeline utilizing spatial transformation,
    probabilistic Hough features, and polynomial curvature estimation.
    """
    def __init__(self, frame_resolution: Tuple[int, int]):
        self.height, self.width = frame_resolution
        
        # Metric conversions for curvature estimation (Standard US Highway specs)
        self.ym_per_pix = 30.0 / 720.0   # Meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700.0    # Meters per pixel in x dimension

        # Precompute static perspective transformation matrices (IPM)
        self.M, self.Minv = self._compute_perspective_matrices()

    def _compute_perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        # Calibration trapezoid assumes center-mounted dashcam geometry.
        # In a production AV stack, these points are derived from the camera intrinsic/extrinsic matrix.
        src = np.float32([
            [self.width * 0.45, self.height * 0.65],
            [self.width * 0.55, self.height * 0.65],
            [self.width * 0.90, self.height],
            [self.width * 0.10, self.height]
        ])
        
        dst = np.float32([
            [self.width * 0.20, 0],
            [self.width * 0.80, 0],
            [self.width * 0.80, self.height],
            [self.width * 0.20, self.height]
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def _isolate_lane_colors(self, frame: np.ndarray) -> np.ndarray:
        # HLS color space is significantly more resilient to varying illumination (shadows, glare) than RGB.
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        white_lower = np.array([0, 200, 0], dtype=np.uint8)
        white_upper = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, white_lower, white_upper)

        yellow_lower = np.array([10, 0, 100], dtype=np.uint8)
        yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

        return cv2.bitwise_or(white_mask, yellow_mask)

    def _extract_edges(self, binary_mask: np.ndarray) -> np.ndarray:
        # Low-pass filter to attenuate high-frequency noise before edge gradient calculation
        blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def _extract_hough_features(self, warped_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Extract linear features from the bird's-eye view using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            warped_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=20,
            minLineLength=20,
            maxLineGap=300
        )

        left_pts, right_pts = [], []
        if lines is None:
            return np.array([]), np.array([])

        midpoint = self.width // 2

        # Cluster segments geometrically based on their x-coordinates relative to the camera centerline
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < midpoint and x2 < midpoint:
                left_pts.extend([[x1, y1], [x2, y2]])
            elif x1 > midpoint and x2 > midpoint:
                right_pts.extend([[x1, y1], [x2, y2]])

        return np.array(left_pts), np.array(right_pts)

    def _fit_polynomial(self, points: np.ndarray) -> Optional[np.ndarray]:
        # Requires at least 3 points to formulate a 2nd order polynomial (x = Ay^2 + By + C)
        if len(points) < 3:
            return None
        return np.polyfit(points[:, 1], points[:, 0], 2)

    def _compute_curvature(self, left_fit: np.ndarray, right_fit: np.ndarray, y_eval: float) -> float:
        # Reproject pixel space polynomial into metric space to derive physical turn radius
        y_vals = np.array([0, y_eval/2, y_eval])
        
        left_x = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
        right_x = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]

        left_fit_cr = np.polyfit(y_vals * self.ym_per_pix, left_x * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(y_vals * self.ym_per_pix, right_x * self.xm_per_pix, 2)

        # Curvature radius formula: R = (1 + (dx/dy)^2)^1.5 / |d^2x/dy^2|
        left_rad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
        right_rad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])

        return float(np.mean([left_rad, right_rad]))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 1. Perception
        color_mask = self._isolate_lane_colors(frame)
        edges = self._extract_edges(color_mask)

        # 2. Spatial Transform (IPM)
        warped_edges = cv2.warpPerspective(edges, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

        # 3. Feature Extraction & Modeling
        left_pts, right_pts = self._extract_hough_features(warped_edges)
        left_fit = self._fit_polynomial(left_pts)
        right_fit = self._fit_polynomial(right_pts)

        # 4. Rendering & Telemetry Output
        overlay = np.zeros_like(frame)
        curvature = 0.0

        if left_fit is not None and right_fit is not None:
            plot_y = np.linspace(0, self.height - 1, self.height)
            left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
            right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))
            curvature = self._compute_curvature(left_fit, right_fit, self.height)

        # Unwarp the active polygon back to camera space and composite
        unwarped_overlay = cv2.warpPerspective(overlay, self.Minv, (self.width, self.height))
        output_frame = cv2.addWeighted(frame, 1.0, unwarped_overlay, 0.4, 0)

        cv2.putText(output_frame, f"Curve Radius: {curvature:.1f}m", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return output_frame


def run_simulation(video_path: str):
    """Execution loop for processing recorded footage."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open stream at {video_path}")

    # Initialize the pipeline state once
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline = LaneDetector((height, width))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = pipeline.process_frame(frame)
        
        cv2.imshow("AV Lane Detection Pipeline", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Point this to a standard dashcam mp4 for testing
    run_simulation("project_video.mp4")