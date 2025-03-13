import cv2
import torch
import numpy as np
import pygame
import time
from djitellopy import Tello
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------
# Drone and Control Configuration
# -------------------------------
S = 60  # Speed for RC commands
for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
send_rc_control = False

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
model = DepthAnythingV2(**model_configs[encoder])
model_path = f'depth_anything_v2_metric_{dataset}_{encoder}.pth'
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE).eval()

# -------------------------------
# Pygame Setup
# -------------------------------
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption('Drone Depth Perception')
clock = pygame.time.Clock()

# -------------------------------
# Helper Function: Create Point Cloud View Image
# -------------------------------
def create_pointcloud_view(valid_points, valid_colors, view_size=(240, 180), sample_max=2000):
    """
    Create a 2D projection (bird’s-eye view) of the 3D point cloud.
    valid_points: (N,3) array with X, Y, Z coordinates.
    valid_colors: (N,3) array with colors in [0,1].
    view_size: (width, height) of output image.
    sample_max: Maximum number of points to draw.
    """
    pc_w, pc_h = view_size
    # Create a blank image (black background)
    pc_img = np.zeros((pc_h, pc_w, 3), dtype=np.uint8)
    
    # Subsample points for speed
    N = valid_points.shape[0]
    sample_count = min(N, sample_max)
    indices = np.linspace(0, N - 1, sample_count).astype(np.int32)
    sub_points = valid_points[indices]
    sub_colors = valid_colors[indices]
    
    # Use X and Y coordinates for projection.
    # Compute bounds to scale points to the view size.
    x_vals = sub_points[:, 0]
    y_vals = sub_points[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    # Add a small epsilon to avoid division by zero
    eps = 1e-6
    # Map each point to pixel coordinates in the output image.
    for pt, col in zip(sub_points, sub_colors):
        u = int((pt[0] - x_min) / (x_max - x_min + eps) * (pc_w - 1))
        v = int((pt[1] - y_min) / (y_max - y_min + eps) * (pc_h - 1))
        # Optionally flip vertically so that higher Y is on top:
        v = pc_h - 1 - v
        color = (np.clip(col * 255, 0, 255)).astype(np.uint8).tolist()
        # Draw a small circle for the point
        cv2.circle(pc_img, (u, v), 1, color, -1)
    return pc_img

# -------------------------------
# Main Loop
# -------------------------------
running = True
while running:
    # Handle Pygame events, including drone control keys
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

    # Send RC commands if enabled
    if send_rc_control:
        tello.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

    # -------------------------------
    # Get Drone Frame and Process
    # -------------------------------
    frame = frame_read.frame
    if frame is None:
        continue

    # Rotate the frame 90° clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run depth inference
    with torch.no_grad():
        depth_map = model.infer_image(frame_rgb)

    if depth_map is None or depth_map.size == 0:
        continue

    # Process depth map: normalize and apply colormap
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    # Rotate depth image as well
    depth_colored = cv2.rotate(depth_colored, cv2.ROTATE_90_CLOCKWISE)

    # -------------------------------
    # Compute 3D Point Cloud from Depth Map
    # -------------------------------
    h, w = depth_map.shape
    # Simple intrinsics (adjust as needed)
    fx, fy = 500, 500
    cx, cy = w // 2, h // 2
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    X = (xx - cx) * depth_map / fx
    Y = -(yy - cy) * depth_map / fy  # Flip Y axis for correct orientation
    Z = depth_map
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    valid_mask = Z.flatten() > 0
    valid_points = points[valid_mask]
    valid_colors = frame_rgb.reshape(-1, 3)[valid_mask] / 255.0

    # Create point cloud view image
    if valid_points.shape[0] > 0:
        pc_img = create_pointcloud_view(valid_points, valid_colors, view_size=(240, 180), sample_max=2000)
        # Rotate the point cloud view 90° clockwise
        pc_img = cv2.rotate(pc_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        pc_img = np.zeros((240, 180, 3), dtype=np.uint8)

    # -------------------------------
    # Prepare Pygame Surfaces for Display
    # -------------------------------
    # Depth image: larger view (e.g., 800x600)
    depth_img_resized = cv2.resize(depth_colored, (800, 600))
    depth_surf = pygame.surfarray.make_surface(depth_img_resized.swapaxes(0, 1))
    # Drone feed: smaller view (e.g., 240x180)
    feed_img_resized = cv2.resize(frame, (240, 180))
    feed_surf = pygame.surfarray.make_surface(feed_img_resized.swapaxes(0, 1))
    # Point cloud view: already 240x180 from our function
    pc_surf = pygame.surfarray.make_surface(pc_img.swapaxes(0, 1))

    # -------------------------------
    # Blit Views onto Pygame Window
    # -------------------------------
    screen.fill((0, 0, 0))
    # Place the depth image on the left (large view)
    screen.blit(depth_surf, (0, 0))
    # Place the drone feed on the top-right
    screen.blit(feed_surf, (810, 20))
    # Place the point cloud view below the drone feed on the right
    screen.blit(pc_surf, (810, 220))
    pygame.display.update()

    clock.tick(30)

# Clean up on exit
pygame.quit()
tello.streamoff()
