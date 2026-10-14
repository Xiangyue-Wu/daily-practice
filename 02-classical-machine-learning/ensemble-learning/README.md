# Bagging and Boosting with Decision Stumps

## Overview

This project explores how ensembles improve on shallow decision trees for two tabular classification settings. Separate scripts cover Iris and mushroom data so bagging-style aggregation and AdaBoost-style reweighting can be examined alongside single-layer baselines.

## Approach

The implementations construct decision stumps, bootstrap or resample training observations, combine multiple weak predictions, and track classification behavior. The Iris workflow uses an available scientific-Python dataset, while the mushroom workflow expects an external categorical CSV. The retained files represent baseline and ensemble variants rather than one universal entry point. They are intended for method comparison and visualization; no accuracy claim is made here because the required data and generated results are not included.

## Reproduce with AI

```text
Inspect the four scripts and map each one to the Iris or mushroom baseline, bagging, or AdaBoost workflow. Set up the needed Python packages, use the built-in Iris data and a legally obtainable mushroom dataset with the expected schema, and reproduce each existing comparison. Report actual metrics and visual outputs. Correct only necessary compatibility, encoding, or path issues, and do not change sampling logic, weak learners, ensemble sizes, or other hyperparameters. Identify any variant that cannot run because an input is missing.
```
