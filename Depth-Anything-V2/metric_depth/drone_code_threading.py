import cv2
import torch
import numpy as np
import pygame
import threading
import time
from djitellopy import Tello
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

# Shared images for display (updated by camera thread)
global_depth_surf = None
global_feed_surf = None
global_pc_surf = None
global_battery = None

# -------------------------------
# Helper Function: Create Point Cloud 2D View
# -------------------------------
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
        v = pc_h - 1 - v  # Flip vertically so higher Y is on top
        color = (np.clip(col * 255, 0, 255)).astype(np.uint8).tolist()
        cv2.circle(pc_img, (u, v), 1, color, -1)
    return pc_img

# -------------------------------
# Initialize Tello Drone
# -------------------------------
tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

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
max_depth = 20 # 20 for indoor model, 80 for outdoor model
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model_path = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE).eval()

# -------------------------------
# Pygame Setup
# -------------------------------
pygame.init()
screen = pygame.display.set_mode((900, 720))
pygame.display.set_caption('Drone Depth Perception')
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# -------------------------------
# Camera Stream Thread (processing and updating display surfaces)
# -------------------------------
def camera_loop():
    global global_depth_surf, global_feed_surf, global_pc_surf, global_battery, running
    while running:
        frame = frame_read.frame
        if frame is None:
            continue

        # Rotate the frame 90° counter-clockwise
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Depth inference
        with torch.no_grad():
            depth_map = model.infer_image(frame_rgb)
        if depth_map is None or depth_map.size == 0:
            continue

        # Process depth map: normalize and apply colormap, then rotate counter-clockwise
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        # depth_colored = cv2.rotate(depth_colored, cv2.ROTATE_90_CLOCKWISE)
        
        # Compute 3D Point Cloud from Depth Map
        h, w = depth_map.shape
        fx, fy = 500, 500
        cx, cy = w // 2, h // 2
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        X = (xx - cx) * depth_map / fx
        Y = -(yy - cy) * depth_map / fy  # Flip Y axis
        Z = depth_map
        points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        valid_mask = Z.flatten() > 0
        valid_points = points[valid_mask]
        valid_colors = frame_rgb.reshape(-1, 3)[valid_mask] / 255.0
        
        if valid_points.shape[0] > 0:
            pc_img = create_pointcloud_view(valid_points, valid_colors, view_size=(400, 300), sample_max=2000)
            # pc_img = cv2.rotate(pc_img, cv2.ROTATE_90_CLOCKWISE)
        else:
            pc_img = np.zeros((400, 300, 3), dtype=np.uint8)
        
        # Create Pygame surfaces:
        # Depth image: larger view (400x300)
        depth_img_resized = cv2.resize(depth_colored, (400, 300))
        depth_surf = pygame.surfarray.make_surface(depth_img_resized.swapaxes(0, 1))
        
        # Drone feed: smaller view (400x300)
        feed_img_resized = cv2.resize(frame, (400, 300))
        feed_surf = pygame.surfarray.make_surface(feed_img_resized.swapaxes(0, 1))
        
        # Point cloud view: already 400x300
        pc_surf = pygame.surfarray.make_surface(pc_img.swapaxes(0, 1))
        
        global_depth_surf = depth_surf
        global_feed_surf = feed_surf
        global_pc_surf = pc_surf
        
        
        global_battery = tello.get_battery()
        
        # time.sleep(1/30)  # Aim for ~30 FPS

# -------------------------------
# Command Sending Thread (continuous RC control)
# -------------------------------
def command_sender():
    global running, send_rc_control, left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    while running:
        if send_rc_control:
            tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
        time.sleep(0.025)  # Send commands at 20 Hz

cam_thread = threading.Thread(target=camera_loop, daemon=True)
cmd_thread = threading.Thread(target=command_sender, daemon=True)
cam_thread.start()
cmd_thread.start()

# -------------------------------
# Main Pygame Loop (Event handling & display)
# -------------------------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                for_back_velocity = S
            elif event.key == pygame.K_DOWN:
                for_back_velocity = -S
            elif event.key == pygame.K_LEFT:
                left_right_velocity = -S
            elif event.key == pygame.K_RIGHT:
                left_right_velocity = S
            elif event.key == pygame.K_w:
                up_down_velocity = S
            elif event.key == pygame.K_s:
                up_down_velocity = -S
            elif event.key == pygame.K_a:
                yaw_velocity = -S
            elif event.key == pygame.K_d:
                yaw_velocity = S
            elif event.key == pygame.K_t:
                tello.takeoff()
                send_rc_control = True
            elif event.key == pygame.K_l:
                tello.land()
                send_rc_control = False

        if event.type == pygame.KEYUP:
            if event.key in [pygame.K_UP, pygame.K_DOWN]:
                for_back_velocity = 0
            if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                left_right_velocity = 0
            if event.key in [pygame.K_w, pygame.K_s]:
                up_down_velocity = 0
            if event.key in [pygame.K_a, pygame.K_d]:
                yaw_velocity = 0

    screen.fill((0, 0, 0))
    if global_depth_surf is not None:
        screen.blit(global_depth_surf, (50, 20))
    if global_feed_surf is not None:
        screen.blit(global_feed_surf, (470, 20))
    if global_pc_surf is not None:
        screen.blit(global_pc_surf, (240, 340))
    
    battery_text = f"Battery: {global_battery}%" if global_battery is not None else "Battery: N/A"
    battery_surf = font.render(battery_text, True, (0, 255, 0))
    screen.blit(battery_surf, (10, 10))
    
    pygame.display.update()
    clock.tick(30)

running = False
cam_thread.join()
cmd_thread.join()
tello.streamoff()
pygame.quit()
