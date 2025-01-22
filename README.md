# Realtime Object Segmentation with YOLO and OpenCV in Low Light Condition

## Key Concepts

### 1. YOLO (You Only Look Once)

YOLO is a deep-learning based model used for object **detection**, **tracking**, **classification** and other similar tasks on images or videos. As the name suggests YOLO looks at the image (or video frame) only once performs detection making it fast and efficient. This is possible by dividing the image into smaller grids and then predicting the presence of objects and their bounding boxes within each grid cell. It can also predict the segments that belongs to the objects thus separating the sementation mask of one object from another object or the background.

### 2. Segmentation

Segmentation is dividing an image into multiple parts or segments often to isolate the areas of interest. YOLO with segmentation produces segmentation masks on the detected objects that highlight the regions covered by the objects.

## Requirements

It is important to have your system set up with the following requirements to execute this project.

### Prerequisites:

1. **Visual Studio** or any code editor that suits you
2. **Python** installed on your system
3. **Libraries:**
    * `opencv-python`
    * `ultralytics`
4. YOLO segmentation model file: `yolov11-seg.pt`

### Installation:

Run the following commands on your terminal to install the required libraries.

```bash
pip install opencv-python
pip install ultralytics
```

### OpenCV:

**OpenCV** (Open Source Computer Vision) is a tool for working with images and videos. `opencv-python` is the python library for this tool.

### Ultralytics:

**Ultralytics** is the company that created '**YOLO**' that we are using for this project. By installing the python library `ultralytics` we can use any YOLO model that fits our requirements.

## Import Libraries

We need to import the installed `OpenCV` and `Ultralytics` libraries first, along with another built-in library `time` for measuring the FPS.

```python
import time
import cv2 as cv
from ultralytics import YOLO
```

## Loading The YOLO Model

Here, the YOLO segmentation model is loaded. We initialize it with a pre-trained weight file `yolov8n-seg.pt`. Replace "yolo11n-seg.pt" with your specific model path if necessary.

```python
model = YOLO("yolo11n-seg.pt")
```

## Capturing The Video

### Webcam

```python
cap = cv.VideoCapture(0)
```

This line opens the default webcam of your pc. **(0)** indicates the default webcam. If you want to capture video with any other webcam you may want to change the index (0) with (1) or (2) or so on.

### Video file

```python
path = 'your_video_path.mp4'
cap = cv.VideoCapture(path)
```

If you have a video file and want to perform image segmentation on it use the followings. This is optional since this project is focused on real-time segmentation from webcam.


## Setting Video Properties

```python
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps_video = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
```

- **`fourcc`:** Specifies the codec for video writing.
- **`fps_video`:** Retrieves **frames per second** of the video.
- **`width` and `height`:** **Width** and **Height** of the video frame or the image.

## Save The Output Video

Set the pathe and file name that you want to save the processed video

```python
output_path = '/path_to_output_video.mp4'
out = cv.VideoWriter(output_path, fourcc, fps_video, (width, height))
```

## Helper Function For FPS Calculation

```python
def fps(start, end):
    return int(1//(end-start))
```

This function calculates FPS by measuring the time taken to process each frame.

## Main Loop

```python
try:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print('No camera detected, aborting')
            break
```

- `cap.isOpened()` checks whether the camera is opened.
- `cap.read()` reads the camera input by frames. If no camera is opened or no frames are read (ret is false), the loop ends.

```python
        start = time.perf_counter()
        results = model(image)
        end = time.perf_counter()
```

- `model(image)` applies the YOLO segmentation model to the image or the frames
- `start` and `end` measures the time spent doing the segmentation

```python
        segments = results[0].plot()
        cv.putText(segments, f'FPS: {fps(start, end)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

- `results[0].plot()` generates the segmentation mask
- `cv.putText()` put the FPS as text on the output

```python
        out.write(segments)
        cv.imshow('Image Segmentation', segments)
```

- `out.write()` saves the output (as mp4 in our case)
- `cv.imshow()` displays the frame with segmenttions in a window

```python
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            print('Exit sequence initiated')
            break
```

- Exits the loop and stops the program when `q` key is pressed (`cv.waitKey(1)` => for 1ms)

## Cleanup

### Stops video capture, finalizes and saves the output video file and closes all OpenCV windows.

```python
finally:
    cap.release()
    out.release()
    cv.destroyAllWindows()
```

## Running The Code

- Save the code to the directory you want with a suitable name (e.g.YOLO_v11_seg.py)
- Open the terminal and change directory to the folder you saved the code
- If you want to run the code in a virtual environment then activate the virtual environment first.
- Run the code:
```bash
python YOLO_v11_seg.py
```
- Press `q` to save the output and exit the program.
