# Realtime Object Segmentation with YOLO and OpenCV in Low Light Conditions

![YOLO Segmentation Demo](YOLO_v11_Night_Vision_Segmentation.gif)

---

## Overview

This project demonstrates real-time object segmentation using the YOLO model and OpenCV. YOLO (You Only Look Once) is a deep-learning-based model designed for fast and efficient object detection, classification, and segmentation. In this implementation, segmentation masks are generated in real-time to identify and isolate objects, even in low-light conditions.

---

## Key Concepts

### 1. YOLO (You Only Look Once)

YOLO is a deep-learning model used for tasks such as object detection, tracking, and segmentation. It divides an image into smaller grids and predicts the presence of objects, their bounding boxes, and segmentation masks for isolating objects from the background. Its speed and efficiency make it suitable for real-time applications.

### 2. Segmentation

Segmentation involves dividing an image into multiple parts or segments to isolate areas of interest. YOLO with segmentation generates masks that highlight the regions occupied by detected objects.

---

## Requirements

### Prerequisites

1. **Code Editor:** Visual Studio, VS Code, or any preferred editor.
2. **Python:** Ensure Python is installed on your system.
3. **Libraries:**
   - `opencv-python`
   - `ultralytics`
4. **YOLO Model File:** Pretrained segmentation model file, e.g., `yolov11-seg.pt`.

### Installation

Run the following commands to install the required libraries:

```bash
pip install opencv-python
pip install ultralytics
```

---

## Step-by-Step Implementation

### 1. Import Libraries

```python
import time
import cv2 as cv
from ultralytics import YOLO
```

### 2. Load the YOLO Model

Load the pre-trained YOLO segmentation model:

```python
model = YOLO("yolov11-seg.pt")
```
Replace `yolov11-seg.pt` with the path to your specific YOLO model file.

### 3. Capture Video Input

#### Option 1: Webcam

To use the default webcam:

```python
cap = cv.VideoCapture(0)
```

#### Option 2: Video File

To use a video file:

```python
path = 'your_video_path.mp4'
cap = cv.VideoCapture(path)
```

### 4. Configure Video Properties

```python
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps_video = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
```

- **`fourcc`:** Specifies the codec for video writing.
- **`fps_video`:** Frames per second of the video.
- **`width` and `height`:** Frame dimensions.

### 5. Save Output Video

Set the output path and initialize the video writer:

```python
output_path = '/path_to_output_video.mp4'
out = cv.VideoWriter(output_path, fourcc, fps_video, (width, height))
```

### 6. FPS Calculation Helper Function

```python
def fps(start, end):
    return int(1 // (end - start))
```

This function calculates the FPS based on the time taken to process each frame.

### 7. Main Processing Loop

The main loop performs the following steps:

1. **Check Video Input:** First the program checks whether the camera is opened.
2. **Read and Process Frames:** Reads the camera input by frames. If no camera is opened or no frames are read (ret is false), the loop ends.
3. **Generate Segmentation Masks and Overlay FPS:** Applies the YOLO segmentation model to the image or the frames, measures the time spent doing the segmentation, generates the segmentation mask and puts the FPS as text on the output.
4. **Save and Display Frames:** Saves the output (as mp4 in our case). Displays the frame with segmenttions in a window.
5. **Exit on Keypress:** The program exits the loop and stops the program when `q` key is pressed (`cv.waitKey(1)` => for 1ms).
6. **Cleanup:** Finally, video capture stops, finalized and output is saved as mp4 file and all OpenCV windows close.

---

## Running the Project

1. Save the script as `YOLO_v11_seg.py`.
2. Open a terminal and navigate to the script's directory.
3. (Optional) Activate your Python virtual environment.
4. Run the script:

```bash
python YOLO_v11_seg.py
```

5. Press `q` to save the output and exit.

---

## Notes

- Adjust the model path and video source based on your setup.
- Ensure sufficient lighting for better segmentation, even in low-light conditions.

---

## References

- [YOLO by Ultralytics](https://ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

