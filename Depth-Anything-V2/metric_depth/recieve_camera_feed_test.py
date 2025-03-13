import cv2

# URL of the video stream
stream_url = 'http://127.0.0.1:5000/video_feed'  # Replace with your server's URL

# Open a connection to the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Read a frame from the stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        break

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
