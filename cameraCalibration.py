import numpy as np
import cv2
import glob

# Termination criteria for subpixel corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real world space)
# Assuming chessboard dimensions (9x6 corners)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load all calibration images
images = glob.glob('calibration_images/*.jpg')  # Change path as needed

for image in images:
    # Read the image
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        
        # Refine corner locations to subpixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)  # Pause for a while to see the corners

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the results
print("Camera Matrix:\n", mtx)
print("\nDistortion Coefficients:\n", dist)

# Undistort an image (optional)
img = cv2.imread('calibration_images/example_image.jpg')
h, w = img.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# Crop the image based on ROI (optional)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Show the result
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
