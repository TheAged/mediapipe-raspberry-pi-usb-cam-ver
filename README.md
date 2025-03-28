# Real-time Fall Detection using MediaPipe Pose

This project implements a basic real-time fall detection system using Google's MediaPipe Pose library and OpenCV in Python. It analyzes human pose landmarks extracted from a video stream (webcam or file) to identify potential fall events based on kinematic and positional cues.

## Description

The script captures video frames, processes them with MediaPipe Pose to obtain 33 3D human body landmarks, and then applies a set of rules to determine if a fall has occurred. The detection logic considers:

1.  **Vertical Velocity:** Tracks the downward speed of the person's hip center. A rapid increase suggests a fall.
2.  **Body Orientation:** Calculates the angle of the torso (shoulder center to hip center) relative to the vertical axis. A large angle sustained over time indicates a horizontal posture, common after a fall.
3.  **Relative Height:** Monitors the vertical position (Y-coordinate) of the hip center relative to the frame's height. A position near the bottom edge can indicate lying or sitting on the floor.
4.  **State Persistence:** Requires potential fall indicators (like high velocity or horizontal/low posture) to persist for a defined duration (`FALL_CONFIRM_DURATION`) before confirming a fall, reducing false positives from quick movements.

The script provides visual feedback by drawing the detected pose skeleton and displaying the current detection status ("Normal", "Potential Fall...", "FALL DETECTED!") on the output video window.

## Features

* Real-time pose estimation via MediaPipe Pose.
* Fall detection based on vertical velocity, body angle, and relative height.
* Temporal filtering (state persistence) to improve robustness.
* Visual output with skeleton overlay and status messages.
* Basic FPS counter.
* Keyboard controls: 'q' to quit, 'r' to reset fall status.
* Configurable detection thresholds.

## Requirements

* Python 3.7+
* OpenCV (`opencv-python`)
* MediaPipe (`mediapipe`)
* NumPy (`numpy`)

## Installation

1.  **Clone the repository or download the script.**
2.  **(Optional but recommended) Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install opencv-python mediapipe numpy
    ```

## Usage

1.  **Connect a webcam** or ensure you have a video file path ready.
2.  **Run the script from your terminal:**
    ```bash
    python3 your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual filename, e.g., `falldect_mediapipe.py`)*
3.  The script will open a window displaying the video feed with pose landmarks and the detection status.
4.  **Interact:**
    * Press **'q'** to quit the application.
    * Press **'r'** to manually reset the "FALL DETECTED!" status back to "Normal" during testing.

*Note: By default, the script uses webcam index `0`. To use a different webcam or a video file, modify the `cv2.VideoCapture()` line within the script.*

## Configuration and Tuning (VERY IMPORTANT)

The accuracy of the fall detection heavily depends on the threshold values set within the script. These **must be tuned** based on your specific setup and environment:

* `Y_VELOCITY_THRESHOLD`: Controls how fast the downward hip movement must be to trigger a potential fall. Increase if too sensitive, decrease if falls are missed.
* `ANGLE_THRESHOLD`: Determines the angle (from vertical) at which the body is considered "horizontal". Increase if tilting is falsely flagged, decrease if horizontal falls are missed.
* `HEIGHT_THRESHOLD_FACTOR`: Defines how low the hips must be (relative to frame height, 0.0=top, 1.0=bottom) to be considered "low". Adjust based on camera view and typical user positions.
* `FALL_CONFIRM_DURATION`: The time (in seconds) a potential fall state must persist before being confirmed. Increase to avoid flagging temporary actions (like picking something up), decrease for faster confirmation.
* `model_complexity` (in `mp_pose.Pose`): Set to `0` (lite), `1` (full), or `2` (heavy). Lower values run faster but may be less accurate, especially on low-power devices like Raspberry Pi.

**Experimentation with these values in your target environment is crucial for reliable performance.**

## Limitations

* **Threshold Dependency:** Performance is highly sensitive to the chosen thresholds.
* **Environmental Factors:** Accuracy can be affected by camera angle, distance, lighting conditions, and background clutter.
* **Occlusion:** If key body parts (especially hips, shoulders) are hidden, detection may fail.
* **Ambiguous Actions:** May confuse actual falls with similar intentional actions like lying down quickly, certain exercises (e.g., burpees), or sitting down abruptly.
* **Performance:** Real-time processing can be computationally intensive, especially on resource-constrained devices. Adjust `model_complexity` if FPS is too low.
* **Multi-Person:** The current script assumes a single person in the frame and does not handle multiple individuals.

## License

[Specify Your License Here - e.g., MIT License, Apache 2.0, etc. If unsure, you can research common open-source licenses.]
