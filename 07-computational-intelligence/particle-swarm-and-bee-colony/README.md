# Particle Swarm and Artificial Bee Colony Optimization

## Overview

This project compares particle swarm optimization and artificial bee colony methods on continuous and discrete problems. It includes two-dimensional function minimization, 0–1 knapsack search, and a permutation-based traveling-salesperson variant.

• Compared dynamic-inertia particle swarm optimization with employed, onlooker, and scout phases from artificial bee colony search on continuous functions.

• Adapted PSO and ABC to an embedded 0–1 knapsack instance and implemented swap-sequence PSO with 2-opt local search for traveling-salesperson routing.

## Approach

The continuous scripts implement PSO with dynamic inertia and ABC with employed, onlooker, and scout phases, producing convergence and search-surface plots. Binary PSO and a discrete ABC solve the same embedded knapsack instance using capacity-aware fitness. The TSP script adapts particle movement to swap sequences and combines it with 2-opt local search. Inputs are embedded in the programs, while stochastic plots and console results are not included. The files are related comparisons but do not form one automatic execution pipeline.

## Reproduce with AI

```text
Inspect all five scripts and map the continuous PSO/ABC, binary knapsack PSO/ABC, and TSP PSO-plus-2-opt workflows. Configure the required numerical and plotting packages, run each implementation with its existing settings, and report actual best positions, values, selections, routes, feasibility, and convergence behavior. Fix only compatibility or plotting issues; do not change objectives, embedded instances, swarm or colony rules, local search, population sizes, iteration counts, or hyperparameters. Note stochastic variation between runs.
```
