'''
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('yolov8m.pt')

# Function to perform object detection on a frame and draw bounding boxes
def detect_objects(frame, model):
    results = model(frame)  # Perform detection
    detections = results[0].boxes  # Extract the bounding boxes

    # Draw bounding boxes on the frame
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        conf = box.conf[0]  # Get confidence score
        cls = int(box.cls[0])  # Get class ID

        label = model.names[cls]  # Get class label
        confidence = conf.item()

        # Draw the bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

# Start the webcam video stream and perform real-time object detection
cap = cv2.VideoCapture(0)  # Open the default camera

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform YOLOv8 detection on the frame
    frame_with_detections = detect_objects(frame, model)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame_with_detections)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


'''

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('yolov8m.pt')

# Function to perform object detection on a frame and draw bounding boxes
def detect_objects(frame, model):
    results = model(frame)  # Perform detection
    detections = results[0].boxes  # Extract the bounding boxes

    if detections is not None:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0]  # Get confidence score
            cls = int(box.cls[0])  # Get class ID

            label = model.names[cls]  # Get class label
            confidence = conf.item()

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

# Start the webcam video stream and perform real-time object detection
cap = cv2.VideoCapture(0)  # Open the default camera

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize frame for better processing speed
    frame = cv2.resize(frame, (640, 480))

    # Perform YOLOv8 detection on the frame
    frame_with_detections = detect_objects(frame, model)

    # Convert the frame to RGB format for Matplotlib
    frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)

    # Display the frame with bounding boxes using Matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide axes
    plt.show(block=False)  # Show the image without blocking
    plt.pause(0.001)  # Pause to allow the display to update

    # Check for exit condition
    if plt.waitforbuttonpress(0.1):  # Wait for 0.1 seconds for any key press
        break

# Release the capture and close any Matplotlib figures
cap.release()
plt.close()  # Close the Matplotlib window





