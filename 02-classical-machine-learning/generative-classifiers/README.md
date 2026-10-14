# Generative Classifiers on Mixed-Type Data

## Overview

This project demonstrates three generative approaches on a small watermelon-quality dataset containing categorical and continuous attributes. It predicts the class of a held-out sample while exposing how different probability assumptions affect classification.

## Approach

`NBCfinal.py` implements a Naive Bayes classifier for mixed features. `GDAfinal.py` provides one- and two-dimensional Gaussian discriminant analysis using sugar content and density, including evaluation and decision visualizations. `GE1.py` encodes non-numeric attributes and implements Gaussian-mixture estimation with expectation maximization, repeated initialization, cluster-to-class mapping, and a PCA-based plot. The dataset is embedded directly in the scripts, so no external file is required. Printed scores and generated images should be treated as run outputs, not as guaranteed repository results.

## Reproduce with AI

```text
Inspect all three classifier scripts in this repository and explain the role of Naive Bayes, Gaussian discriminant analysis, and EM-based Gaussian mixtures. Configure the minimal Python environment, run the existing embedded-data workflows, and report the predicted held-out class, evaluation values, and generated plots exactly as observed. Fix only compatibility or output-path issues; do not alter the embedded samples, probability assumptions, model logic, or parameters. Flag any numerical instability or ambiguous cluster labeling.
```
