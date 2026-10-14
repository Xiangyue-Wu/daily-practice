# Movie Data Cleaning and Visualization

## Overview

This project cleans and explores a movie metadata table stored as `movies.csv`. The scripts normalize irregular fields, separate structured information from combined text columns, fill selected missing values, and produce visual summaries of ratings, genres, runtime, and gross revenue.

## Approach

The implementation uses pandas and NumPy for tabular cleanup and Matplotlib/Seaborn for exploratory plots. Separate scripts preserve different stages of the workflow: basic normalization, director and actor parsing, rated/unrated record separation, missing-rating estimation, and visual analysis. One path also constructs TF-IDF features from people-related text and fits a random-forest regressor with randomized parameter search. The project expects the actual input file to be named `movies.csv`; neither the dataset nor generated CSV and image outputs are included here.

## Reproduce with AI

```text
Read the repository and determine which movie-cleaning script best represents the complete workflow and which alternatives are partial variants. Set up the required Python environment, inspect the expected movies.csv schema, and help obtain or prepare a legally usable matching input. Reproduce the existing cleaning, visualization, and optional rating-model steps, reporting only observed outputs. Repair only necessary dependency or relative-path issues without changing the cleaning rules, model logic, or parameters, and identify any missing columns or data.
```
