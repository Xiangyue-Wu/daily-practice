# Skeleton-Based Hand Action Recognition

## Overview

This project classifies hand actions from skeleton sequences in the FPHA data format. It compares recurrent, temporal-convolutional, and attention-based sequence models while retaining dedicated preprocessing and repeated-run evaluation utilities.

## Approach

The code loads skeleton and label arrays, normalizes joint coordinates, pads or crops sequences, and constructs PyTorch datasets. `lstm_baseline.py` implements a single-layer LSTM baseline. `Res-TCN.py` builds a residual temporal convolutional network and records training, validation, confusion-matrix, and stability diagnostics. `base.py` and `transformer.py` implement Transformer encoders; the latter adds wrist normalization, temporal cropping, spatial jitter, and mixup. `data_processing.py` supports preparation of the source arrays. FPHA data, learned checkpoints, TensorBoard logs, and result images are not included.

## Reproduce with AI

```text
Inspect the preprocessing, LSTM, Res-TCN, and Transformer files and determine the exact FPHA skeleton/label inputs and runnable entry points. Configure a compatible PyTorch environment, legally obtain the required FPHA data, reproduce each existing model workflow, and report measured accuracy and stability separately. Repair only necessary file-format, device, or path compatibility issues without changing normalization, sequence length, augmentation, architectures, or hyperparameters. Clearly document any missing arrays, checkpoints, or ambiguous preprocessing step.
```
