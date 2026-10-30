# Neural Networks for MNIST and CIFAR-10

## Overview

This collection explores convolutional classification and transfer-style fine-tuning across handwritten digits and natural images. The source includes a compact MNIST CNN, an AlexNet-based CIFAR-10 path, and scripts that reload checkpoints while selectively training later layers.

• Implemented a compact MNIST CNN and an AlexNet-based CIFAR-10 workflow with augmentation, training curves, confusion analysis, and prediction visualization.

• Explored checkpoint reuse through selective parameter freezing and later-layer fine-tuning for CIFAR-10 and MNIST adaptation.

## Approach

`CNN.py` trains a small CNN on an ImageFolder-formatted MNIST PNG dataset. `clas_cifar10.py` adapts AlexNet for CIFAR-10 and includes augmentation, training curves, confusion analysis, and prediction visualization. `cifarckpt.py` and `fine.py` load an existing checkpoint and freeze most parameters for continued training, while `rewrite.py` adapts an AlexNet checkpoint to MNIST and controls which classifier or convolutional layers remain trainable. The repository does not include datasets, checkpoints, or generated figures, and the scripts represent related experiments rather than one automatic pipeline.

## Reproduce with AI

```text
Inspect all five scripts and map the standalone MNIST CNN, CIFAR-10 AlexNet, checkpoint fine-tuning, and MNIST adaptation workflows. Determine each expected dataset layout and checkpoint format, configure compatible PyTorch/torchvision versions, and legally prepare the required data. Reproduce only stages whose resources are available and report their observed metrics. Fix device, API, or relative-path compatibility issues only; do not redesign models, change trainable layers, or alter hyperparameters. List missing checkpoints and data explicitly.
```
