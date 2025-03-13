import cv2
import numpy as np
import glob

# Termination criteria for corner sub-pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on square size (20 mm)
square_size = 20  # mm
obj_points = np.zeros((12 * 9, 3), np.float32)
obj_points[:, :2] = np.indices((12, 9)).T.reshape(-1, 2) * square_size

# Arrays to store object points and image points
obj_points_list = []
img_points_list = []

# Load calibration images
images = glob.glob('calib/*.jpg')

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (12, 9), None)
    
    if ret:
        obj_points_list.append(obj_points)
        img_points_list.append(corners)
        
        # Refine corner positions
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (12, 9), corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, img_points_list, gray.shape[::-1], None, None)

# Save calibration results
np.savez('calibration.npz', mtx=mtx, dist=dist)
