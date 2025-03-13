import cv2
import signal
import sys
import threading
from flask import Flask, Response, request
from djitellopy import Tello

app = Flask(__name__)

# Initialize Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Function to generate video frames from the Tello's stream
def generate():
    while True:
        frame = tello.get_frame_read().frame
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle drone control commands via POST requests
@app.route('/command', methods=['POST'])
def command():
    data = request.json
    action = data.get('action')
    
    try:
        if action == 'takeoff':
            tello.takeoff()
        elif action == 'land':
            tello.land()
        elif action == 'move':
            direction = data.get('direction')
            distance = data.get('distance', 10)
            if direction == 'forward':
                tello.move_forward(distance)
            elif direction == 'backward':
                tello.move_back(distance)
            elif direction == 'left':
                tello.move_left(distance)
            elif direction == 'right':
                tello.move_right(distance)
            elif direction == 'up':
                tello.move_up(distance)
            elif direction == 'down':
                tello.move_down(distance)
        elif action == 'rotate':
            direction = data.get('direction')
            angle = data.get('angle', 90)
            if direction == 'clockwise':
                tello.rotate_clockwise(angle)
            elif direction == 'counter_clockwise':
                tello.rotate_counter_clockwise(angle)
        elif action == 'flip':
            direction = data.get('direction')
            if direction == 'forward':
                tello.flip_forward()
            elif direction == 'backward':
                tello.flip_back()
            elif direction == 'left':
                tello.flip_left()
            elif direction == 'right':
                tello.flip_right()
        else:
            return 'Invalid action', 400
    except Exception as e:
        print("Error executing command:", e)
        return f"Error executing command: {e}", 500

    return 'Command executed', 200

# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    print("Shutting down gracefully...")
    tello.streamoff()
    sys.exit(0)

# Register shutdown signals for graceful cleanup
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == '__main__':
    # Run Flask with threaded=True and disable reloader to avoid extra threads during shutdown
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
