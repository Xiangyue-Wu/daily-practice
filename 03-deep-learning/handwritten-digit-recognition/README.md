# Handwritten Digit Recognition

## Overview

This combined project compares several approaches to recognizing handwritten digits. It preserves foundational implementations of nearest-neighbor classification, a backpropagation neural network, and a compact convolutional network, plus a separate PyTorch CNN notebook from another implementation track.

• Compared pixel-distance matching, a feed-forward backpropagation network, and a compact convolutional recognizer as progressively learned digit-classification approaches.

• Preserved a separate PyTorch MNIST CNN notebook so its framework, data-loading assumptions, and training workflow remain independently reproducible.

## Approach

The `foundations/` directory contains three standalone Python programs that expose progressively more learned representations: pixel-distance matching, a feed-forward BP network, and convolutional recognition. The `pytorch-cnn/` directory contains an MNIST-oriented notebook using PyTorch. Keeping both branches makes their different assumptions and data-loading conventions visible without presenting them as one unified benchmark. Input images, datasets, trained weights, and executed notebook outputs are not included. Numeric accuracies stated in earlier task material are not repeated here because they were not independently verified against this cleaned source snapshot.

## Reproduce with AI

```text
Inspect both subdirectories, identify the entry point and expected input format for the distance classifier, BP network, foundational CNN, and PyTorch notebook, and configure the minimal compatible environments. Obtain MNIST or the exact legally usable digit inputs required by each implementation and reproduce the existing comparisons. Do not merge implementations or change architectures, preprocessing, or hyperparameters. Fix only compatibility and path problems, report actual results separately, and clearly identify any missing samples, datasets, or checkpoints.
```
