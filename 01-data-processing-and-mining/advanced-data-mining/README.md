# Household Power Data Mining

## Overview

This project applies a compact data-mining workflow to household electricity measurements. It combines time-series inspection, anomaly detection, supervised feature construction, and short-horizon power prediction. The implementation expects the UCI household power-consumption text file under `data/` and can work on a bounded subset for practical experimentation.

• Built an end-to-end household-power analysis workflow covering timestamp cleanup, aggregation, decomposition, autocorrelation, stationarity checks, and seasonal inspection.

• Combined Z-score and Isolation Forest anomaly detection with lag and calendar features for standardized MLP forecasting evaluated by RMSE, MAE, and R².

## Approach

The Python workflow parses timestamps, handles missing measurements, aggregates the series, and examines autocorrelation, stationarity, trend, and seasonality. It marks unusual observations with both a Z-score rule and Isolation Forest, then converts lagged power values and calendar fields into regression features. Standardized features feed an `MLPRegressor`, with RMSE, MAE, and R² used for comparison. The repository contains source only; the dataset and generated figures are intentionally not included.

## Reproduce with AI

```text
Inspect this repository and its household-power time-series workflow. Identify the executable Python entry point and dependencies, then locate a legally obtainable copy of the expected UCI dataset and place it at the path required by the code. Reproduce the cleaning, decomposition, anomaly-detection, and MLP forecasting stages and report the actual metrics. Fix only environment or path compatibility issues; do not change the algorithms, feature definitions, or hyperparameters. Clearly report any unavailable data or outputs.
```
