# Depth Anything Tello Tracking

Computer vision experiments for camera calibration, template matching, and depth-assisted tracking around a DJI Tello-style workflow.

## Contents

- `Depth-Anything-V2/` - depth estimation code or imported model workspace.
- `cameraCalibration.py` - camera calibration utilities.
- `generatePattern.py` - checkerboard/pattern generation helper.
- `template_matching.py` - template matching experiment code.
- `Checkerboard-Custom.png` and `calibration_1.pdf` - calibration assets.

## Typical Workflow

1. Generate or print the calibration pattern.
2. Capture calibration frames from the target camera.
3. Run the calibration script to estimate camera parameters.
4. Use the tracking/template matching scripts with calibrated camera inputs.

## Notes

This repository appears to be an experimental robotics/computer-vision workspace. Hardware-specific paths, camera settings, and model weights may need to be adjusted before running on a new machine.