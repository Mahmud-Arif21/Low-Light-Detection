import time
import cv2 as cv
from ultralytics import YOLO

# Load the YOLO model for image segmentation
model = YOLO("yolo11n-seg.pt")

# Capture video from the default camera (0)
cap = cv.VideoCapture(0)

# Uncomment the following two lines to use a pre-recorded video instead of the live camera
# path = '~/real_time_detection_with_yolov11/test/video/night_vision.mp4'
# cap = cv.VideoCapture(path)

# Get video properties such as frame width, height, and FPS
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps_video = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the output file path for the processed video
output_path = '~/real_time_detection_with_yolov11/result/video/realtime_test.mp4'
# Uncomment the following line if you are using the night vision video sample as input
# output_path = '~/real_time_detection_with_yolov11/result/video/night_vision.mp4'
out = cv.VideoWriter(output_path, fourcc, fps_video, (width, height))

# Function to calculate the FPS
def fps(start, end):
    return int(1 // (end - start))

try:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print('No camera detected, aborting')
            break

        start = time.perf_counter()

        # Perform image segmentation using the YOLO model
        results = model(image)
        end = time.perf_counter()
        segments = results[0].plot()

        cv.putText(segments, f'FPS: {fps(start, end)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(segments)
        cv.imshow('Image Segmentation', segments)
        
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            print('Exit sequence initiated')
            break

finally:
    # Release video capture and writer resources
    cap.release()
    out.release()
    cv.destroyAllWindows()