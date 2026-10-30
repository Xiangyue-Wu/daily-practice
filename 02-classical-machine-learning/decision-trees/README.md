# Decision Tree Implementations

## Overview

This project studies decision-tree classification for mushroom edibility. It retains two implementations so the tree-building process can be compared with a library-based workflow rather than hidden behind a single estimator.

• Implemented categorical decision-tree construction around information gain, gain ratio, and Gini impurity to study the split logic behind ID3, C4.5, and CART.

• Preserved manual and library-backed classifiers for side-by-side evaluation on mushroom edibility data with the same categorical feature setting.

## Approach

The source covers categorical data loading and tree construction using common split criteria associated with ID3, C4.5, and CART: information gain, gain ratio, and Gini impurity. One script builds the logic directly, while the other uses established machine-learning utilities for comparison and evaluation. The workflow expects a mushroom CSV file with a class column and categorical attributes. That dataset and any generated charts are not included, so reported performance must come from a fresh, traceable run with a compatible source.

## Reproduce with AI

```text
Read both decision-tree scripts and identify the manual and library-backed paths, their expected mushroom-data schema, split criteria, and evaluation flow. Prepare a suitable Python environment and a legally available compatible mushroom dataset, then reproduce the existing comparisons and report the measured results. Fix only dependency, encoding, or relative-path problems without changing tree logic, cross-validation choices, or hyperparameters. Clearly state if the required dataset or a claimed criterion is not actually usable from the current code.
```
