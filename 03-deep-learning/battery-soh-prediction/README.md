# Battery State-of-Health Prediction

## Overview

This project builds a multi-stage workflow for estimating lithium-ion battery state of health from cycling measurements. It covers raw-data preparation, feature extraction, dataset partitioning, correlation analysis, baseline sequence modeling, and cross-battery transfer experiments.

• Built a numbered preprocessing pipeline that converts MATLAB cycling records into engineered features, dataset splits, and correlation diagnostics for battery SOH modeling.

• Implemented an LSTM baseline and a CNN–BiLSTM–attention transfer workflow with pretraining, selective freezing, fine-tuning, scheduling, and cross-battery comparison.

## Approach

The executable order is `3.1.1_pre.py`, `3.1.2_extract_features.py`, `3.1.3_split.py`, and `3.1.4_pcc.py`; these stages turn MATLAB records into feature arrays under `outputs/prepared_data`. `3.2.py` then trains the LSTM baseline, while `3.3.py` independently runs the CNN–BiLSTM–attention and transfer workflow from the same prepared directory. The latter includes pretraining, selective freezing, fine-tuning, scheduling, metrics, and comparisons across NASA battery identifiers. Raw data, prepared arrays, weights, and generated reports are not included.

## Reproduce with AI

```text
Inspect the numbered scripts and reconstruct their intended battery-SOH pipeline order, required columns, intermediate files, and dependencies. Locate a legally obtainable compatible battery-aging dataset, then reproduce preprocessing, feature extraction, splitting, correlation analysis, LSTM training, and the CNN-BiLSTM-attention transfer workflow. Report actual metrics for each battery. Fix only compatibility or relative-path issues without altering features, model structures, transfer strategy, or hyperparameters, and list every missing dataset, checkpoint, or intermediate artifact.
```
