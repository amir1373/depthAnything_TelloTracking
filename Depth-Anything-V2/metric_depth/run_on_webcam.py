import cv2
import torch
import numpy as np
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
import time

DEBUG_COLOR = True  # Set to True to assign random colors for debugging.

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'      # Options: 'vits', 'vitb', 'vitl', 'vitg'
dataset = 'hypersim'  # 'hypersim' for indoor, 'vkitti' for outdoor

# Initialize model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

# Define Camera Intrinsics (adjust based on your webcam)
fx, fy = 500, 500    # Focal lengths in pixels
cx, cy = 320, 240    # Optical center (for a 640x480 image)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Unable to read from webcam.")
    cap.release()
    exit()
print("Frame shape:", frame.shape)

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window("Real-Time Point Cloud")
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Set render options: point size and background color.
render_option = vis.get_render_option()
render_option.point_size = 2.0
render_option.background_color = np.asarray([0, 0, 0])

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Convert from BGR (OpenCV default) to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference to obtain a depth map (expected shape: 480x640)
    with torch.no_grad():
        depth_map = model.infer_image(frame_rgb)

    if depth_map is None or depth_map.size == 0:
        print("Warning: Received empty depth map!")
        continue

    # Create meshgrid for pixel coordinates and compute 3D points using the pinhole camera model.
    h, w = depth_map.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    X = (xx - cx) * depth_map / fx
    Y = -(yy - cy) * depth_map / fy   # Flip Y axis to correct inversion.
    Z = depth_map

    # Form the point cloud in (N, 3) format.
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # Filter out invalid points (depth <= 0)
    valid_mask = Z.flatten() > 0
    valid_points = points[valid_mask]

    # Use random colors for debugging if flag is set; otherwise use image colors.
    if DEBUG_COLOR:
        valid_colors = np.random.rand(valid_points.shape[0], 3)
    else:
        valid_colors = frame_rgb.reshape(-1, 3)[valid_mask] / 255.0

    if valid_points.shape[0] == 0:
        continue

    # Update the Open3D point cloud geometry.
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

    # Force the visualizer to update by clearing and re-adding the geometry.
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Adjust the view control so that the point cloud is in view.
    center = valid_points.mean(axis=0)
    extent = valid_points.max(axis=0) - valid_points.min(axis=0)
    max_extent = np.max(extent)
    cam_pos = center + np.array([0, 0, 1.5 * max_extent])
    view_ctl = vis.get_view_control()
    view_ctl.set_lookat(center)
    front = (center - cam_pos) / np.linalg.norm(center - cam_pos)
    view_ctl.set_front(front.tolist())
    view_ctl.set_up([0, 1, 0])

    # Display the webcam feed and depth map for reference.
    cv2.imshow("Webcam", frame)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imshow("Depth Map", depth_map_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
