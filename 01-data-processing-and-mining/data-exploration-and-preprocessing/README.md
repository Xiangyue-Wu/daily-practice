# Retail Data Exploration and Preprocessing

## Overview

This project explores and prepares online retail transaction data for later analysis. It focuses on data quality, distribution analysis, customer–item representations, similarity measurement, scaling, and categorical encoding rather than fitting a final predictive model.

## Approach

The main Python program loads a sample of `online_retail_II.csv`, reviews the table, handles missing records and abnormal values, normalizes dates and descriptions, and creates univariate and multivariate visualizations. It builds a customer–item matrix and compares Euclidean distance, cosine similarity, and Pearson correlation for active customers. Min–max and Z-score scaling are also demonstrated, along with optional categorical encodings. The cleaned table is written to an output directory. Source code is included, but the retail dataset and generated artifacts are not.

## Reproduce with AI

```text
Inspect this repository and its retail preprocessing entry point. Determine the required columns and dependencies, obtain a legally accessible Online Retail II-compatible dataset, and reproduce the existing overview, cleaning, plots, customer similarity, scaling, encoding, and export workflow. Keep the current sampling and transformation logic intact, fixing only compatibility or path issues. Report the actual produced tables and figures, and state clearly if the expected data schema or any required resource is unavailable.
```
