# License Plate Digit Recognition

## Overview

This project recognizes numeric characters in clean and simulated license-plate images. It combines image preprocessing, character localization and segmentation, handcrafted features, template-derived training samples, and classifier-based digit prediction.

• Built plate preprocessing and character segmentation with contrast enhancement, denoising, adaptive thresholding, morphology, contour analysis, and candidate filtering.

• Combined augmented digit templates, HOG and topology-related features, auxiliary K-means representations, and SVM/random-forest predictions for recognition.

## Approach

The retained scripts cover single-character tests, clean-plate processing, and a more extensive real-image pipeline. OpenCV operations include grayscale conversion, contrast enhancement, denoising, adaptive thresholding, morphology, contour analysis, and candidate filtering. The real-image implementation augments digit templates, extracts HOG and topology-related features, uses K-means as an auxiliary representation, and combines SVM and random-forest predictions. It expects `data/char_tmpl` and plate images under a data directory. Those inputs and generated diagnostics are not distributed, so recognition performance is not asserted.

## Reproduce with AI

```text
Inspect the three scripts and identify the single-digit, clean-plate, and simulated-plate entry paths and their expected template/image folders. Configure compatible OpenCV and scikit-learn dependencies, legally prepare matching digit templates and plate samples, and reproduce segmentation, feature extraction, training, and recognition. Report actual predictions and failure cases. Fix only environment, image-format, or relative-path issues; do not change thresholds, features, classifiers, augmentation, or hyperparameters. List any unavailable images explicitly.
```
