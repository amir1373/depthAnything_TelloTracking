import cv2
import os
import time
from djitellopy import Tello

# Create the 'calib' folder if it doesn't exist
os.makedirs("calib", exist_ok=True)

# Initialize and connect to the Tello drone
# tello = Tello()
# tello.connect()
# print(f"Battery: {tello.get_battery()}%")

# Turn off any existing video streams
# tello.streamoff()

# Start the video stream
# tello.streamon()
# print("Stream started")

# Allow some time for the video stream to initialize
time.sleep(2)
SERVER_URL = 'http://127.0.0.1:5000/video_feed'
# Create a VideoCapture object using the Tello's video stream URL
cap = cv2.VideoCapture(SERVER_URL)
print("cap is made")

# Check if the VideoCapture object has been initialized correctly
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    # tello.streamoff()
    # tello.end()
    exit(1)
print("frame not none")
# Create a named window for display
cv2.namedWindow("Tello Feed", cv2.WINDOW_NORMAL)
print("cv2 window created")
# Counter for saved images
image_count = 0

try:
    while True:
        # Capture frame-by-frame
        print("in the loop")
        ret, frame = cap.read()
        print("after cap read")
        # If the frame was not retrieved properly, print a warning and continue
        if not ret or frame is None:
            print("Warning: Unable to retrieve video frame.")
            continue

        # Display the frame
        cv2.imshow("Tello Feed", frame)

        # Wait for key press for 1 ms
        key = cv2.waitKey(1) & 0xFF

        # If 'l' is pressed, save the current frame in the 'calib' folder
        if key == ord('l'):
            image_count += 1
            filename = os.path.join("calib", f"calibration_image_{image_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

        # Press 'q' to quit the loop and end the program
        elif key == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Cleanup: release the VideoCapture object, turn off the video stream, and close the window
    cap.release()
    # tello.streamoff()
    cv2.destroyAllWindows()
    # tello.end()
