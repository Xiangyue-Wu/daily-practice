# Evolutionary Channel Pruning for ResNet-32

## Overview

This project explores channel pruning of a CIFAR-10 ResNet-32 using genetic algorithms and particle swarm optimization. It also contains baseline training and a comparison utility for accuracy, parameter count, estimated computation, and inference time.

## Approach

`rest32/1.py` trains the CIFAR ResNet-32 baseline. `ga/2ga-14.py` and `pso/3pso-14.py` search stage-channel configurations, transfer compatible weights, fine-tune candidates, and save pruned checkpoints; the PSO script additionally expects its prior-round checkpoint. Only after those artifacts exist can `compare/compare.py` evaluate accuracy, parameters, computation, and runtime. The required CIFAR-10 data, baseline and pruned weights, prior PSO state, and result logs are not included. Numeric values embedded in comments are therefore not treated as verified results.

## Reproduce with AI

```text
Inspect the baseline, GA, PSO, and comparison scripts and identify every required CIFAR-10 layout, checkpoint, prior-search artifact, and output. Configure compatible PyTorch/torchvision versions, legally prepare the data, and reproduce the workflow only in its intended order. Report measured accuracy, parameters, FLOPs estimate, compression, and runtime. Fix only device, serialization, API, or path compatibility issues; do not change ResNet-32, pruning representation, fitness functions, search settings, fine-tuning, or hyperparameters. Clearly list resources that prevent a stage from running.
```
