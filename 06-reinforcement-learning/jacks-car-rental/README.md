# Jack's Car Rental with Dynamic Programming

## Overview

This project models overnight vehicle transfers between two rental locations as a finite Markov decision process. It uses dynamic programming to estimate long-term returns and determine how many cars should be moved for each pair of inventory levels.

## Approach

The environment represents rental requests and returns with Poisson probabilities, transfer costs, rental income, capacity constraints, and discounted future value. The retained variants implement policy iteration and value iteration, including a modified reward case with parking-capacity penalties. Each script produces heatmaps for the state-value surface and transfer policy as iterations proceed. The implementations are self-contained apart from scientific-Python plotting dependencies; generated figures are not included, and convergence was not rerun during repository preparation.

## Reproduce with AI

```text
Inspect ptry6.py, task2.py, and vtry5.py and identify the policy-iteration, value-iteration, and modified-reward variants. Configure the required numerical and plotting environment, reproduce each existing dynamic-programming workflow, and report convergence behavior plus the resulting policy and value heatmaps. Fix only compatibility or output-path issues without changing Poisson assumptions, capacities, rewards, costs, discounting, stopping tolerance, or update logic. Explain any differences caused by the parking-penalty variant.
```
