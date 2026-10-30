# Stereo Disparity Estimation

## Overview

This project estimates disparity from a rectified stereo image pair using OpenCV. Despite the broader directory name, the available source implements stereo block matching only; it does not contain visual odometry or a complete SLAM system.

• Implemented stereo disparity estimation from rectified grayscale image pairs using OpenCV StereoBM with explicit disparity-range and block-size settings.

• Normalized and saved the resulting disparity map for relative depth inspection while keeping the scope limited to block matching rather than claiming a SLAM pipeline.

## Approach

`task1.py` loads left and right grayscale images, configures `StereoBM` with explicit disparity-range and block-size parameters, computes the disparity map, normalizes it for display, and saves the resulting image. The workflow is useful for examining how block matching converts horizontal correspondence into relative depth structure. Camera calibration, metric-depth conversion, input stereo images, and generated disparity outputs are not included, so geometric accuracy is not claimed.

## Reproduce with AI

```text
Inspect task1.py and identify its required OpenCV version, image names, rectification assumptions, StereoBM parameters, and output behavior. Prepare a legally usable rectified stereo pair with compatible dimensions, reproduce the existing disparity computation, and report the generated map and visible limitations. Fix only OpenCV/API, display, or relative-path compatibility issues; do not add SLAM, change the matching algorithm, tune parameters, or claim metric depth without calibration. Clearly state if suitable stereo inputs or calibration data are missing.
```
