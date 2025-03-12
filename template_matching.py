import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Initialize variables
template = None
keypoints_template = None
descriptors_template = None
tracker = None
roi = None

# Function to select ROI
def select_roi(event, x, y, flags, param):
    global roi
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x, y, 0, 0)
    elif event == cv2.EVENT_LBUTTONUP:
        roi = (roi[0], roi[1], x - roi[0], y - roi[1])
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        cv2.imshow('Webcam Feed', frame)

# Set mouse callback function
cv2.namedWindow('Webcam Feed')
cv2.setMouseCallback('Webcam Feed', select_roi)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t') and roi is not None:  # 't' key to initialize tracking
        # Crop the template from the frame
        template = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        # Convert the template to grayscale
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # Detect keypoints and compute descriptors for the template
        keypoints_template, descriptors_template = orb.detectAndCompute(gray_template, None)
        # Initialize the tracker
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, tuple(roi))
        print("Tracking initialized.")
    elif key == ord('q'):  # 'q' key to quit
        break

    # If the tracker has been initialized, update the tracking
    if tracker is not None:
        success, box = tracker.update(frame)
        if success:
            # Draw the bounding box
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

