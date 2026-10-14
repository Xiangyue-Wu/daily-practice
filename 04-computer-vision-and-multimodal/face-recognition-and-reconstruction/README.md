# Face Recognition and Reconstruction with PCA

## Overview

This project applies principal component analysis to face images for compact representation, recognition, and reconstruction. It provides data preparation and two executable exercises that expose both the recognition workflow and the visual effect of changing the retained subspace.

## Approach

The code reads aligned grayscale face samples, reshapes them into vectors, centers the data, and derives an eigenface basis. Projected feature vectors support identity comparison, while inverse projection reconstructs faces from a selected number of principal components. The exercises include visual interfaces or plots for inspecting source images, eigenfaces, nearest matches, and reconstruction quality. Example identity labels were anonymized during source preparation. The required face dataset and generated images are not included, so no recognition rate or reconstruction score is claimed here.

## Reproduce with AI

```text
Inspect data.py and both exercise scripts, identify the expected face-image directory structure, executable entry points, PCA dimensions, and dependencies. Prepare a legally usable aligned face dataset matching the loader, reproduce recognition and reconstruction, and report observed matches and visual outputs. Fix only necessary GUI, image-format, or relative-path compatibility issues without changing PCA logic, distance rules, component counts, or parameters. Clearly explain any missing dataset or interface limitation.
```
