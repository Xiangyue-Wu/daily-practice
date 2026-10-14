# Semi-Supervised Radar Image Classification

## Overview

This project investigates limited-label classification on MSTAR radar imagery with ResNet-18. It retains a sequence of supervised baselines, reduced-label comparisons, a FixMatch-style semi-supervised implementation, and scripts for evaluating recovery from scarce labels.

## Approach

Scripts `2.1.py` and `2.2.py` are independent supervised comparisons. For the final limited-label sequence, `2.3.py` produces the 10% labeled ResNet-18 baseline, `2.4.py` trains the FixMatch-style model with weak/strong augmentation, confidence-filtered pseudo-labels, supervised/unsupervised losses, and an exponential-moving-average model, and `2.5.py` compares their checkpoints and parses the FixMatch log. All expect class folders under `data/mstar-train` and `data/mstar-test`. MSTAR data, pretrained downloads, checkpoints, and logs are not included.

## Reproduce with AI

```text
Inspect scripts 2.1.py through 2.5.py and map the supervised baseline, limited-label experiment, FixMatch-style training, and final comparison stages. Configure compatible PyTorch and vision dependencies, legally obtain MSTAR-format data, and reproduce only the workflows supported by the files. Preserve splits, augmentations, confidence rules, ResNet-18 architecture, schedules, and hyperparameters. Fix necessary device or path issues only, report measured results, and explicitly list missing data, pretrained weights, checkpoints, or logs.
```
