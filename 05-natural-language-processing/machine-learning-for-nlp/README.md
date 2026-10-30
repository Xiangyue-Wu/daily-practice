# Chinese Sentiment Classification

## Overview

This project compares TF-IDF logistic regression, TextCNN, and bidirectional LSTM on binary Chinese hotel-review sentiment classification. It includes a standard training workflow and a three-seed stability study for the two neural models.

• Compared a 5,000-feature TF-IDF logistic-regression baseline with TextCNN and bidirectional LSTM models on cleaned Chinese hotel reviews.

• Evaluated accuracy, precision, recall, F1, ROC-AUC, confusion matrices, learning curves, and three-seed stability; the verified logistic-regression run reached 0.8609 accuracy and 0.9434 ROC-AUC.

## Approach

The code cleans and tokenizes ChnSentiCorp reviews with jieba, uses a 5,000-feature TF-IDF representation for logistic regression, and builds fixed-length token sequences for PyTorch models. TextCNN applies convolution windows of 3, 4, and 5; Bi-LSTM uses bidirectional context. Evaluation covers accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and training curves. In the verified single run, logistic regression achieved `0.8609` accuracy and `0.9434` ROC-AUC. Across seeds 42, 123, and 2026, TextCNN averaged `0.8347` accuracy, while Bi-LSTM averaged `0.8219`. Dataset redistribution rights still require independent confirmation.

## Reproduce with AI

```text
Inspect the README and four source modules, identify the single-run and repeated-seed entry points, and configure compatible scikit-learn and PyTorch dependencies. Before acquiring ChnSentiCorp, verify lawful access and redistribution terms. Reproduce preprocessing, logistic regression, TextCNN, Bi-LSTM, and the three-seed comparison, then report actual metrics and variability. Fix only environment, device, download, or path compatibility issues; do not change splits, architectures, seeds, epochs, or hyperparameters. State any data or licensing limitation clearly.
```
