import cv2
import torch
import numpy as np
import pygame
import threading
import time
import requests
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------
# Global variables and flags
# -------------------------------
running = True
send_rc_control = False

# Speed constant for RC commands
S = 60

# Shared RC control velocities (in range -100 to 100)
for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

# Shared images for display (updated by threads)
global_depth_surf = None
global_feed_surf = None
global_pc_surf = None
global_battery = None
latest_frame = None  # Latest raw frame from drone

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

# Server URL to fetch video frames
SERVER_URL = 'http://127.0.0.1:5000/video_feed'  # Assuming Flask server is running locally

def capture_frame():
    global latest_frame, running, global_feed_surf
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
        feed_img_resized = cv2.resize(frame, (400, 300))
        global_feed_surf = pygame.surfarray.make_surface(feed_img_resized.swapaxes(0, 1))
        time.sleep(1 / 60)  # Capture at ~60 FPS
    cap.release()

def create_pointcloud_view(valid_points, valid_colors, view_size=(240, 180), sample_max=2000):
    pc_w, pc_h = view_size
    pc_img = np.zeros((pc_h, pc_w, 3), dtype=np.uint8)
    
    N = valid_points.shape[0]
    sample_count = min(N, sample_max)
    indices = np.linspace(0, N - 1, sample_count).astype(np.int32)
    sub_points = valid_points[indices]
    sub_colors = valid_colors[indices]
    
    x_vals = sub_points[:, 0]
    y_vals = sub_points[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    eps = 1e-6
    for pt, col in zip(sub_points, sub_colors):
        u = int((pt[0] - x_min) / (x_max - x_min + eps) * (pc_w - 1))
        v = int((pt[1] - y_min) / (y_max - y_min + eps) * (pc_h - 1))
        v = pc_h - 1 - v
        color = (np.clip(col * 255, 0, 255)).astype(np.uint8).tolist()
        cv2.circle(pc_img, (u, v), 1, color, -1)
    return pc_img

def process_video():
    global global_depth_surf, global_pc_surf, global_battery, running
    while running:
        if latest_frame is None:
            continue
        
        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            depth_map = model.infer_image(frame_rgb)
        if depth_map is None or depth_map.size == 0:
            continue

        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        depth_surf = pygame.surfarray.make_surface(cv2.resize(depth_colored, (400, 300)).swapaxes(0, 1))
        global_depth_surf = depth_surf
        
        h, w = depth_map.shape
        fx, fy = 500, 500
        cx, cy = w // 2, h // 2
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        X = (xx - cx) * depth_map / fx
        Y = -(yy - cy) * depth_map / fy
        Z = depth_map
        points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        valid_mask = Z.flatten() > 0
        valid_points = points[valid_mask]
        valid_colors = frame_rgb.reshape(-1, 3)[valid_mask] / 255.0
        
        if valid_points.shape[0] > 0:
            pc_img = create_pointcloud_view(valid_points, valid_colors, view_size=(400, 300), sample_max=2000)
        else:
            pc_img = np.zeros((400, 300, 3), dtype=np.uint8)
        global_pc_surf = pygame.surfarray.make_surface(pc_img.swapaxes(0, 1))
        
        # Assuming battery information is still available from the drone (if connected)
        global_battery = 100  # Set this to battery status if you can retrieve it from the drone
        time.sleep(1/30)

pygame.init()
screen = pygame.display.set_mode((900, 720))
pygame.display.set_caption('Drone Depth Perception')
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

cam_thread = threading.Thread(target=capture_frame, daemon=True)
process_thread = threading.Thread(target=process_video, daemon=True)
cam_thread.start()
process_thread.start()



while running:
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         running = False

    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_UP:
    #             for_back_velocity = S
    #         elif event.key == pygame.K_DOWN:
    #             for_back_velocity = -S
    #         elif event.key == pygame.K_LEFT:
    #             left_right_velocity = -S
    #         elif event.key == pygame.K_RIGHT:
    #             left_right_velocity = S
    #         elif event.key == pygame.K_w:
    #             up_down_velocity = S
    #         elif event.key == pygame.K_s:
    #             up_down_velocity = -S
    #         elif event.key == pygame.K_a:
    #             yaw_velocity = -S
    #         elif event.key == pygame.K_d:
    #             yaw_velocity = S
    #         elif event.key == pygame.K_t:
    #             tello.takeoff()
    #             send_rc_control = True
    #         elif event.key == pygame.K_l:
    #             tello.land()
    #             send_rc_control = False

        # if event.type == pygame.KEYUP:
        #     if event.key in [pygame.K_UP, pygame.K_DOWN]:
        #         for_back_velocity = 0
        #     if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
        #         left_right_velocity = 0
        #     if event.key in [pygame.K_w, pygame.K_s]:
        #         up_down_velocity = 0
        #     if event.key in [pygame.K_a, pygame.K_d]:
        #         yaw_velocity = 0

    screen.fill((0, 0, 0))
    if global_depth_surf is not None:
        screen.blit(global_depth_surf, (50, 20))
    if global_feed_surf is not None:
        screen.blit(global_feed_surf, (470, 20))
    if global_pc_surf is not None:
        screen.blit(global_pc_surf, (240, 340))
    
    # battery_text = f"Battery: {global_battery}%" if global_battery is not None else "Battery: N/A"
    # screen.blit(font.render(battery_text, True, (0, 255, 0)), (10, 10))
    pygame.display.update()
    clock.tick(30)

running = False
cam_thread.join()
process_thread.join()
# cmd_thread.join()
pygame.quit()
