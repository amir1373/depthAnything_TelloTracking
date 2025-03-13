import cv2
import os

# Initialize the webcam (0 is typically the default camera)
SERVER_URL = 'http://127.0.0.1:5000/video_feed'
cap = cv2.VideoCapture(SERVER_URL)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
image_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    # If 'l' is pressed, save the current frame in the 'calib' folder
    key = cv2.waitKey(1) & 0xFF
    if key == ord('l'):
        image_count += 1
        filename = os.path.join("calib", f"calibration_image_{image_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    # Press 'q' to exit the webcam feed


# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()