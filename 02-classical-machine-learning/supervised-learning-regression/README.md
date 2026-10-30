# Wine Quality Regression and Classification

## Overview

This project uses physicochemical wine measurements for both score prediction and binary quality classification. It combines exploratory analysis, feature scaling, classical supervised models, evaluation, and parameter search in a notebook-oriented workflow.

• Modeled numeric wine-quality scores with linear regression and evaluated predictions using MSE, RMSE, MAE, and R².

• Converted quality into a binary task and compared logistic regression, decision trees, k-nearest neighbors, and random forests with ROC, confusion-matrix, and grid-search analysis.

## Approach

The implementation trains linear regression for the numeric quality score and evaluates it with MSE, RMSE, MAE, and R². A derived good/bad label is modeled with logistic regression, a decision tree, and k-nearest neighbors, using classification metrics, ROC analysis, confusion matrices, and grid search. An additional random-forest path is present for comparison. The code expects a Wine Quality-compatible table and includes visual analysis, but neither the dataset nor executed notebook outputs are committed.

## Reproduce with AI

```text
Inspect the notebook and companion Python script, identify the expected Wine Quality input and the implemented regression and classification branches, and configure their Python dependencies. Obtain a legally accessible compatible dataset, reproduce the current preprocessing, model evaluation, and grid searches, and report only measured results. Fix necessary environment or path issues without changing label thresholds, splits, algorithms, search grids, or hyperparameters. State clearly if the input schema or any optional comparison cannot be reproduced.
```
