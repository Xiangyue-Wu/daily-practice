# Chinese NLP Case Studies

## Overview

This project collects several Chinese public-comment analysis workflows. The source covers text cleaning, jieba-based frequency analysis, word clouds, sentiment scoring, TF-IDF logistic regression, and an ALBERT sentiment-classification path.

• Built local Chinese comment workflows for cleanup, jieba frequency analysis, word clouds, SnowNLP scoring, and visual exploration.

• Added TF-IDF logistic-regression and ALBERT classification paths, while keeping optional authorized collection isolated behind an environment-provided session value.

## Approach

For local data, `Demo03/data.py` cleans a comment table and `Demo03/fenci.py` applies jieba, word frequencies, SnowNLP, and visualizations. `src3/try1.py` is the labeled TF-IDF logistic-regression path; `src2/albert_chinese_small_sentiment.py` is the Transformers alternative. The optional collector reads an authorized session value from `WEIBO_COOKIE` and stops when it is absent. No dataset, model files, or outputs are included, and collection is unnecessary for local analysis. Any acquisition must be independently authorized.

## Reproduce with AI

```text
Inspect the case-study subdirectories, identify each cleaning, visualization, logistic-regression, SnowNLP, and ALBERT workflow, and determine the expected comment-table columns. Use only legally obtained local data and verify pretrained-model terms before downloading. Configure the required NLP dependencies, reproduce the available workflows, and report observed metrics or visual outputs. Fix only compatibility, encoding, model-loading, or relative-path issues; do not change labeling logic, models, or parameters. Flag missing data, credentials, models, and any network-dependent step.
```
