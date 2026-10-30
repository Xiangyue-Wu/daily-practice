# Cliff Walking with Dynamic Programming and Reinforcement Learning

## Overview

This project solves the classic cliff-walking gridworld and compares model-based planning with tabular reinforcement learning. It visualizes value functions, learned policies, paths, and reward histories without presenting the task as a large-scale control system.

• Modeled Cliff Walking as a finite gridworld and used dynamic programming to obtain a value function, policy, and 13-step optimal path after 14 iterations.

• Compared epsilon-greedy SARSA and Q-learning on the same transitions and rewards to examine convergence, learned routes, and risk around the cliff.

## Approach

`DP.py` defines the grid transitions, step cost, cliff penalty, and goal, then performs dynamic-programming updates to obtain an optimal value function and policy. `Sarsa_QL.py` implements epsilon-greedy SARSA and Q-learning on the same environment and compares their state values, policies, and episode returns. The environment constants and learning settings are embedded in the scripts, while generated plots are not committed. Because stochastic training was not rerun for this source snapshot, no convergence or return claim is made here.

## Reproduce with AI

```text
Inspect both scripts and verify that dynamic programming, SARSA, and Q-learning use the same cliff-walking dynamics, rewards, start, and goal. Configure the minimal Python environment, run the existing planning and learning workflows, and report the resulting policies, paths, reward curves, and any seed-dependent behavior. Fix only plotting or compatibility issues; do not change the grid, reward values, exploration rule, update equations, learning rates, episode counts, or other hyperparameters. Clearly report any execution limitation.
```
