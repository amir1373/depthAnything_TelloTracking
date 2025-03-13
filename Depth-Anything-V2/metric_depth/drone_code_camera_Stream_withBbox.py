import cv2
import torch
import numpy as np
import pygame
import threading
import time
import socket
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------
# Global variables and flags
# -------------------------------
running = True

# Shared images for display (updated by threads)
global_depth_surf = None
global_feed_surf = None
global_pc_surf = None
latest_frame = None  # Latest raw frame from drone

# Global bounding box list (each bbox is a tuple of (center_x, center_y, width, height))
bbox_data = []

# -------------------------------
# Load Depth Estimation Model
# -------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitl'
dataset = 'hypersim'
max_depth = 20
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

# Server URL to fetch video frames (assuming Flask server is running locally)
SERVER_URL = 'http://127.0.0.1:5000/video_feed'

# -------------------------------
# UDP Receiver for Bounding Box Data
# -------------------------------
def receive_bbox_data():
    global bbox_data
    udp_ip = "127.0.0.1"  # Localhost (adjust if needed)
    udp_port = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    print(f"Listening for bounding box data on {udp_ip}:{udp_port}")
    
    while running:
        try:
            message, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
            bbox_str = message.decode('utf-8')
            # Expecting data in the format "center_x,center_y,width,height"
            parts = bbox_str.split(',')
            if len(parts) != 4:
                print(f"Invalid bbox data from {addr}: {bbox_str}")
                continue
            center_x, center_y, width, height = map(int, parts)
            bbox_data.append((center_x, center_y, width, height))
            print(f"Received Bounding Box: Center=({center_x}, {center_y}), Width={width}, Height={height}")
        except Exception as e:
            print("Error receiving bbox data:", e)
        time.sleep(0.01)
    sock.close()

# -------------------------------
# Video Capture Function
# -------------------------------
def capture_frame():
    global latest_frame, global_feed_surf
    cap = cv2.VideoCapture(SERVER_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame.")
            break
        latest_frame = frame
        # Resize for display (400x300)
        feed_img_resized = cv2.resize(frame, (400, 300))
        global_feed_surf = pygame.surfarray.make_surface(feed_img_resized.swapaxes(0, 1))
        # Uncomment sleep if you want to limit capture FPS:
        # time.sleep(1 / 60)
    cap.release()

# -------------------------------
# Depth Processing and Bounding Box Drawing
# -------------------------------
def process_video():
    global global_depth_surf, bbox_data
    while running:
        if latest_frame is None:
            continue

        # Get original dimensions of the frame
        orig_h, orig_w = latest_frame.shape[:2]
        # Compute scaling factors for resizing to 400x300
        scale_x = 400 / orig_w
        scale_y = 300 / orig_h

        # Convert frame from BGR to RGB for the depth model
        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            depth_map = model.infer_image(frame_rgb)
        if depth_map is None or depth_map.size == 0:
            continue

        # Process depth map: normalize and apply a colormap
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        # Resize depth image for display
        depth_resized = cv2.resize(depth_colored, (400, 300))

        # Draw bounding boxes on the resized depth image and print center info
        for bbox in bbox_data:
            center_x, center_y, width, height = bbox
            # Scale bounding box coordinates for display
            new_center_x = int(center_x * scale_x)
            new_center_y = int(center_y * scale_y)
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            top_left = (new_center_x - new_width // 2, new_center_y - new_height // 2)
            bottom_right = (new_center_x + new_width // 2, new_center_y + new_height // 2)
            cv2.rectangle(depth_resized, top_left, bottom_right, (0, 0, 255), 2)
            
            # Get depth value at the original bbox center (ensure indices are within bounds)
            if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                depth_value = depth_map[center_y, center_x]
            else:
                depth_value = None
            print(f"BBox Center (Original): x={center_x}, y={center_y}, z={depth_value}")
        
        if bbox_data:
            print(f"Drawing {len(bbox_data)} bounding boxes.")
        # Clear bbox_data so that boxes are not redrawn repeatedly
        bbox_data = []

        # Update the global depth surface for rendering in Pygame
        global_depth_surf = pygame.surfarray.make_surface(depth_resized.swapaxes(0, 1))
        # Uncomment sleep if you want to adjust processing rate:
        # time.sleep(0.02)

# -------------------------------
# Pygame Initialization and Main Loop
# -------------------------------
pygame.init()
screen = pygame.display.set_mode((900, 720))
pygame.display.set_caption('Drone Depth Perception')
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# Start threads for video capture, depth processing, and UDP bbox reception
cam_thread = threading.Thread(target=capture_frame, daemon=True)
process_thread = threading.Thread(target=process_video, daemon=True)
udp_thread = threading.Thread(target=receive_bbox_data, daemon=True)

cam_thread.start()
process_thread.start()
udp_thread.start()

# Main loop: Render the surfaces
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))
    
    if global_depth_surf is not None:
        screen.blit(global_depth_surf, (50, 20))
    if global_feed_surf is not None:
        screen.blit(global_feed_surf, (470, 20))
    if global_pc_surf is not None:
        screen.blit(global_pc_surf, (240, 340))
    
    # Uncomment to display battery info if available:
    # battery_text = f"Battery: {global_battery}%" if global_battery is not None else "Battery: N/A"
    # screen.blit(font.render(battery_text, True, (0, 255, 0)), (10, 10))
    
    pygame.display.update()
    clock.tick(30)

running = False
cam_thread.join()
process_thread.join()
udp_thread.join()
pygame.quit()
