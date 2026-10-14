# Genetic Algorithms for Optimization

## Overview

This project demonstrates genetic algorithms across three problem types: bounded continuous optimization, traveling-salesperson routing, and binary knapsack selection. Python implementations cover the first two tasks, while the knapsack workflow is written in Julia.

## Approach

The continuous optimizer uses tournament selection, arithmetic crossover, Gaussian mutation, bounds handling, and population-history visualization. The TSP implementation represents tours as permutations and applies tournament selection, order crossover, mutation, repeated runs, and route/convergence plots. The Julia program evolves binary item-selection vectors with fitness penalties, selection, crossover, mutation, and elitism under a capacity constraint. All task data are embedded in the source, so external datasets are not required. Results vary with random initialization and were not rerun for this README.

## Reproduce with AI

```text
Inspect the two Python scripts and the Julia script, identify their objective functions, representations, and runtime dependencies, and reproduce the continuous, TSP, and knapsack workflows. Record random seeds if the current code exposes them and report actual best solutions plus convergence behavior without treating one stochastic run as universal. Fix only language-version, plotting, or compatibility issues; do not change objective data, population sizes, selection, crossover, mutation, elitism, generation counts, or bounds. State any unavailable Julia or Python runtime limitation.
```
