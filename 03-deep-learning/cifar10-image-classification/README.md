# CIFAR-10 Image Classification

## Overview

This project implements a compact convolutional neural network for ten-class image classification. It separates model training from checkpoint-based evaluation and visualization, making the core workflow easy to inspect without bundling data or learned weights.

## Approach

`final.py` loads CIFAR-10 images arranged in `train/` and `test/` folders, applies normalization and augmentation, trains `ImprovedCIFAR10CNN`, tracks loss and accuracy, and saves the best state dictionary. `use.py` reconstructs the same architecture, loads a checkpoint, computes overall and per-class accuracy, and produces a confusion matrix and class-accuracy chart. The dataset directory, `best_model.pth`, and generated visualizations are intentionally absent, so this source snapshot does not claim a verified accuracy.

## Reproduce with AI

```text
Read final.py and use.py, verify that their model definitions and expected folder layout agree, and identify all Python and PyTorch dependencies. Prepare a legally available CIFAR-10 copy in the ImageFolder structure expected by the code, reproduce training and checkpoint evaluation, and report observed overall and per-class metrics. Resolve only environment, device, checkpoint-loading, or relative-path compatibility issues; do not change the CNN, augmentations, optimizer, scheduler, epochs, or other hyperparameters. Clearly identify missing data or weights.
```
