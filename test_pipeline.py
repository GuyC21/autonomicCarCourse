"""
Unit tests for the lane_pipeline module.
Covers every public and key private method of LaneDetector.
"""
import unittest
import numpy as np
import cv2
from lane_pipeline import LaneDetector


class TestLaneDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.h, cls.w = 360, 640
        cls.det = LaneDetector((cls.h, cls.w))

    def setUp(self):
        self.det = LaneDetector((self.h, self.w))

    # ── Construction ──
    def test_init_resolution(self):
        self.assertEqual(self.det.height, 360)
        self.assertEqual(self.det.width, 640)

    def test_perspective_matrix_shapes(self):
        self.assertEqual(self.det.M.shape, (3, 3))
        self.assertEqual(self.det.Minv.shape, (3, 3))

    def test_perspective_inverse_identity(self):
        product = self.det.M @ self.det.Minv
        product /= product[2, 2]
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=5)

    def test_roi_mask_shape(self):
        self.assertEqual(self.det.roi_mask.shape, (360, 640))

    def test_roi_mask_not_all_zero(self):
        self.assertGreater(np.sum(self.det.roi_mask), 0)

    def test_roi_mask_sky_is_zero(self):
        top_strip = self.det.roi_mask[:int(self.h * 0.50), :]
        self.assertEqual(np.sum(top_strip), 0)

    def test_expected_lane_width(self):
        self.assertEqual(self.det.expected_lane_width_px, 355)

    # ── Feature Extraction ──
    def test_extract_features_shape(self):
        frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
        feat = self.det._extract_lane_features(frame)
        self.assertEqual(feat.shape, (self.h, self.w))

    def test_extract_features_binary(self):
        frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
        feat = self.det._extract_lane_features(frame)
        unique = set(np.unique(feat))
        self.assertTrue(unique.issubset({0, 255}))

    def test_extract_features_black_frame(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        feat = self.det._extract_lane_features(frame)
        self.assertEqual(np.sum(feat), 0)

    def test_extract_features_white_frame(self):
        frame = np.full((self.h, self.w, 3), 255, dtype=np.uint8)
        feat = self.det._extract_lane_features(frame)
        self.assertGreater(np.sum(feat), 0)

    # ── Sliding Window ──
    def test_sliding_window_empty(self):
        blank = np.zeros((self.h, self.w), dtype=np.uint8)
        l_pts, r_pts = self.det._sliding_window_search(blank)
        self.assertEqual(l_pts.shape[0], 0)
        self.assertEqual(r_pts.shape[0], 0)

    def test_sliding_window_right_line(self):
        binary = np.zeros((self.h, self.w), dtype=np.uint8)
        # Draw a vertical white line at x=480 (right side)
        binary[:, 478:482] = 255
        l_pts, r_pts = self.det._sliding_window_search(binary)
        self.assertGreater(r_pts.shape[0], 0)
        # Mean x should be near 480
        self.assertAlmostEqual(np.mean(r_pts[:, 0]), 480, delta=10)

    def test_sliding_window_left_line(self):
        binary = np.zeros((self.h, self.w), dtype=np.uint8)
        binary[:, 158:162] = 255
        l_pts, r_pts = self.det._sliding_window_search(binary)
        self.assertGreater(l_pts.shape[0], 0)

    # Hough Transform
    def test_hough_segments_detect_vertical_lanes(self):
        binary = np.zeros((self.h, self.w), dtype=np.uint8)
        binary[:, 158:162] = 255
        binary[:, 478:482] = 255
        segments = self.det._hough_segments(binary)
        self.assertGreater(segments.shape[0], 0)
        self.assertEqual(segments.shape[1], 4)

    def test_hough_lane_search_separates_lane_points(self):
        binary = np.zeros((self.h, self.w), dtype=np.uint8)
        binary[:, 158:162] = 255
        binary[:, 478:482] = 255
        l_pts, r_pts = self.det._hough_lane_search(binary)
        self.assertGreater(l_pts.shape[0], 0)
        self.assertGreater(r_pts.shape[0], 0)
        self.assertAlmostEqual(np.mean(l_pts[:, 0]), 160, delta=18)
        self.assertAlmostEqual(np.mean(r_pts[:, 0]), 480, delta=18)

    # ── Polynomial Fitting ──
    def test_fit_poly_insufficient_points(self):
        pts = np.array([[100, 100], [101, 101]], dtype=np.float64)
        self.assertIsNone(self.det._fit_poly(pts))

    def test_fit_poly_low_span(self):
        # 50 points but all at the same y — low vertical span
        pts = np.column_stack([np.linspace(100, 200, 50), np.full(50, 100)])
        self.assertIsNone(self.det._fit_poly(pts))

    def test_fit_poly_valid(self):
        ys = np.linspace(50, 350, 200)
        xs = 0.001 * ys**2 - 0.5 * ys + 200
        pts = np.column_stack([xs, ys])
        fit = self.det._fit_poly(pts)
        self.assertIsNotNone(fit)
        self.assertEqual(len(fit), 3)

    def test_fit_poly_straight_line(self):
        ys = np.linspace(0, 359, 200)
        xs = np.full_like(ys, 400.0)
        pts = np.column_stack([xs, ys])
        fit = self.det._fit_poly(pts)
        self.assertIsNotNone(fit)
        # First coefficient (quadratic) should be ~0
        self.assertAlmostEqual(fit[0], 0, places=3)

    # ── Lane Inference ──
    def test_infer_left_from_right(self):
        right_fit = np.array([0.0001, -0.1, 500.0])
        left_fit = self.det._infer_missing_lane(right_fit, is_right=True)
        self.assertAlmostEqual(left_fit[2], 500.0 - 355)
        self.assertEqual(left_fit[0], right_fit[0])
        self.assertEqual(left_fit[1], right_fit[1])

    def test_infer_right_from_left(self):
        left_fit = np.array([0.0001, -0.1, 150.0])
        right_fit = self.det._infer_missing_lane(left_fit, is_right=False)
        self.assertAlmostEqual(right_fit[2], 150.0 + 355)

    def test_infer_does_not_mutate_original(self):
        original = np.array([0.001, -0.5, 300.0])
        original_copy = original.copy()
        self.det._infer_missing_lane(original, is_right=True)
        np.testing.assert_array_equal(original, original_copy)

    # ── Validation & Smoothing ──
    def test_validate_right_valid(self):
        det = LaneDetector((self.h, self.w))
        right_fit = np.array([0.0, 0.0, 480.0])  # bot=480
        lf, rf = det._validate_and_smooth(None, right_fit)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(lf)  # inferred from right

    def test_validate_right_invalid(self):
        det = LaneDetector((self.h, self.w))
        right_fit = np.array([0.0, 0.0, 50.0])  # bot=50, too far left
        lf, rf = det._validate_and_smooth(None, right_fit)
        self.assertIsNone(rf)
        self.assertIsNone(lf)

    def test_smoothing_accumulates(self):
        det = LaneDetector((self.h, self.w))
        for i in range(5):
            right_fit = np.array([0.0, 0.0, 480.0 + i])
            det._validate_and_smooth(None, right_fit)
        self.assertEqual(len(det.right_fit_hist), 5)
        self.assertEqual(len(det.left_fit_hist), 5)

    def test_invalid_pair_keeps_stable_side(self):
        det = LaneDetector((self.h, self.w))
        left_fit = np.array([0.0, 0.0, 150.0])
        right_fit = np.array([0.0, 0.0, 505.0])
        det._validate_and_smooth(left_fit, right_fit, left_conf=100, right_conf=100)

        bad_right = np.array([0.0, 0.0, 280.0])
        lf, rf = det._validate_and_smooth(left_fit, bad_right, left_conf=100, right_conf=100)

        self.assertIsNotNone(lf)
        self.assertIsNotNone(rf)
        self.assertAlmostEqual(np.polyval(lf, self.h - 1), 150, delta=3)
        self.assertAlmostEqual(np.polyval(rf, self.h - 1), 505, delta=3)

    def test_missed_frame_counter(self):
        det = LaneDetector((self.h, self.w))
        det._validate_and_smooth(None, None)
        self.assertEqual(det.missed, 1)
        det._validate_and_smooth(None, None)
        self.assertEqual(det.missed, 2)
        # Good frame resets
        det._validate_and_smooth(None, np.array([0.0, 0.0, 480.0]))
        self.assertEqual(det.missed, 0)

    # ── Curvature ──
    def test_curvature_straight(self):
        lf = np.array([0.0, 0.0, 150.0])
        rf = np.array([0.0, 0.0, 500.0])
        curv = self.det._curvature(lf, rf, 360)
        self.assertGreater(curv, 5000)

    def test_curvature_curved(self):
        lf = np.array([0.001, -0.5, 200.0])
        rf = np.array([0.001, -0.5, 500.0])
        curv = self.det._curvature(lf, rf, 360)
        self.assertGreater(curv, 0)
        self.assertLess(curv, 10000)

    # ── Full Pipeline ──
    def test_process_frame_output_shape(self):
        frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
        result = self.det.process_frame(frame)
        self.assertEqual(result.shape, (self.h, self.w, 3))

    def test_process_frame_black_input(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        result = self.det.process_frame(frame)
        self.assertEqual(result.shape, (self.h, self.w, 3))

    def test_process_frame_does_not_crash_on_varied_input(self):
        for _ in range(5):
            frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
            result = self.det.process_frame(frame)
            self.assertEqual(result.shape, (self.h, self.w, 3))


if __name__ == "__main__":
    unittest.main()
