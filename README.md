# Autonomous Vehicle Lane Detection

OpenCV/Python lane detection pipeline for recorded driving footage or a camera stream.

The project implements a classical computer vision pipeline that detects lane markings,
estimates the drivable lane area, overlays the detected lane on the original video, and
reports an estimated curve radius.

## Project Overview

This project was developed as part of an autonomous vehicle software development course.
The goal is to demonstrate a perception subsystem that receives camera frames and extracts
lane geometry using image processing methods rather than deep learning.

The pipeline is designed for highway-style driving footage with visible lane markings. It
supports both straight and curved lanes and includes temporal smoothing to reduce frame-to-frame
jitter.

## Pipeline

1. Read frames from a video file or camera stream.
2. Convert each frame to color spaces useful for lane detection.
3. Apply HLS color thresholding to detect white and yellow lane markings.
4. Apply grayscale preprocessing, CLAHE contrast enhancement, Gaussian blur, Sobel gradients,
   and Canny edge detection.
5. Restrict detection to a road region of interest.
6. Warp the road area into a bird's-eye view using a perspective transform.
7. Detect lane segment candidates using the probabilistic Hough transform.
8. Use Hough segments and sliding windows to collect lane pixels.
9. Fit second-order polynomials to the left and right lane boundaries.
10. Validate, smooth, and recover temporarily missing lane sides.
11. Warp the detected lane overlay back onto the original frame.
12. Estimate and display the curve radius.

## Repository Files

- `lane_pipeline.py` - main OpenCV lane detection pipeline and video processing entry point.
- `test_pipeline.py` - unit tests for the core lane detection functions.
- `requirements.txt` - Python dependencies required to run the project.
- `Project_Theory_Description.docx` - written project theory and explanation.
- `Project_Presentation.pptx` - project presentation slides.

## Installation

Create and activate a Python environment, then install the dependencies:

```powershell
pip install -r requirements.txt
```

The project requires:

- Python 3.9+
- NumPy
- OpenCV

## Running the Pipeline

Place the input video in the project folder and name it:

```text
project_video.mp4
```

Then run:

```powershell
python lane_pipeline.py
```

The processed output video will be written next to the input video as:

```text
project_video_output.mp4
```

You can also run with `uv`:

```powershell
uv run --with numpy --with opencv-python lane_pipeline.py
```

## Running Tests

Run the unit tests with:

```powershell
python -B -m unittest test_pipeline.py
```

Or with `uv`:

```powershell
uv run --with numpy --with opencv-python -m unittest test_pipeline.py
```

The test suite validates the main processing components, including ROI construction,
feature extraction, Hough segment detection, sliding-window search, polynomial fitting,
lane validation, smoothing, curvature estimation, and full-frame processing.

## Output

For each processed frame, the pipeline displays:

- A green overlay for the detected drivable lane area.
- Colored boundary lines for the left and right lane estimates.
- A text label with the estimated curve radius.
- A warning state when the lane is temporarily lost or occluded.
