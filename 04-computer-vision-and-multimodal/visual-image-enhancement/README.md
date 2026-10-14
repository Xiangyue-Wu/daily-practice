# Image Contrast Enhancement and Dehazing

## Overview

This project implements two practical image-enhancement operations: local contrast improvement and single-image haze removal. Each operation is kept in a focused script and writes processed images for direct visual comparison.

## Approach

The contrast workflow uses OpenCV CLAHE to enhance local luminance while limiting excessive amplification. The dehazing workflow follows the dark-channel-prior pipeline: it computes the dark channel, estimates atmospheric light and transmission, constrains transmission for numerical stability, reconstructs the scene radiance, and produces comparison figures. Both scripts expect source images in local input folders and save outputs to result directories. No test images, generated images, or objective quality scores are included in this repository.

## Reproduce with AI

```text
Inspect the contrast and dehazing scripts, identify their expected image folders, dependencies, and callable entry paths, and prepare legally usable test images with both low contrast and visible haze. Reproduce the CLAHE and dark-channel-prior outputs and report the actual generated files and any observable limitations without inventing quality gains. Fix only OpenCV/API or relative-path compatibility issues; do not change algorithm steps, window sizes, CLAHE settings, atmospheric-light logic, or transmission parameters. State clearly if suitable inputs are missing.
```
