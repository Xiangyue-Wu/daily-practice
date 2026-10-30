# Advanced Text Processing Utilities

## Overview

This project provides small, reproducible demonstrations of three text-processing tasks: Chinese short-text similarity, Chinese keyword extraction, and dictionary-based spelling correction. The modules are coordinated by `src/main.py` and have pinned environment descriptions.

• Compared Chinese short texts with jieba tokenization, TF-IDF vectors, and cosine similarity, and contrasted TF-IDF with TextRank keyword extraction.

• Implemented Levenshtein distance from scratch for dictionary-based correction; verified runs recovered all four intended words, with the closest topical pair scoring 0.171635 cosine similarity.

## Approach

Text similarity uses jieba tokenization, TF-IDF vectors, and cosine similarity. Keyword extraction compares jieba's TF-IDF and TextRank methods before and after task-specific stop-word filtering. Spelling correction uses a manually implemented Levenshtein dynamic program and ranks candidates from a compact dictionary. A verified local run found the closest topical text pair at cosine similarity `0.171635`; all four misspellings selected the intended word with edit distance one, except `intelegence`, which required distance two. The small inputs and generated CSV/figures remain outside this source-only folder.

## Reproduce with AI

```text
Inspect the README, environment files, src/main.py, and the three task modules. Configure the declared lightweight Python environment, determine whether the original small inputs are present or can be reconstructed only from authorized project resources, and reproduce similarity, TF-IDF/TextRank keywords, and Levenshtein correction. Keep tokenization, stop-word behavior, TopK values, vocabulary, and algorithms unchanged. Fix only compatibility or path issues, report actual CSV/figure results, and identify any missing input file explicitly.
```
