# Retail Association Mining and Customer Clustering

## Overview

This project analyzes online retail transactions from two complementary perspectives: product co-occurrence and customer segmentation. It turns invoice records into shopping baskets for association-rule mining and derives RFM features for clustering customers.

## Approach

The notebook and Python implementation clean positive, customer-linked transactions, encode baskets, and use Apriori to generate frequent itemsets and rules with support, confidence, and lift. For segmentation, they compute recency, frequency, and monetary value, apply transformations and standardization, and evaluate K-means solutions with inertia and silhouette scores. PCA supports two-dimensional visualization, while hierarchical clustering and a dendrogram provide an alternative grouping view. The code includes a PyTorch-based K-means path as well as standard scientific-Python components. Required retail data and generated outputs are not distributed in this folder.

## Reproduce with AI

```text
Review the repository and identify the complete association-mining and clustering workflow. Set up its Python dependencies, acquire a legally usable Online Retail II-compatible dataset, verify the expected schema, and reproduce the Apriori rules, RFM construction, K-means selection, PCA visualization, and hierarchical comparison. Preserve all current thresholds and algorithms, changing only necessary environment or path details. Report actual rule and clustering diagnostics and explicitly note missing data or unsupported steps.
```
