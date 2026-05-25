# Autonomic Car Course

OpenCV/Python lane detection pipeline for recorded footage or a camera stream.

Pipeline:
- Canny edge detection over a road ROI
- Probabilistic Hough transform for lane segment candidates
- Optional HLS color thresholding and inverse perspective mapping
- Polynomial lane fitting with temporal validation/smoothing
- Lane overlay and curvature radius estimation

Run:

```powershell
uv run --with numpy --with opencv-python lane_pipeline.py
```

Tests:

```powershell
uv run --with numpy --with opencv-python -m unittest test_pipeline.py
```
