# Chinese Text Collection and Basic Processing

## Overview

This project collects and processes Chinese text for exploratory analysis. The retained scripts demonstrate web-content acquisition, text cleanup, tokenization, frequency analysis, and word-cloud or chart generation as separate stages rather than as a packaged crawler.

## Approach

`src/ls.py` and `src/try7CLOUDE.py` are alternative complete entry points for the same sequence: fetch listed news pages, extract and clean text, segment it with jieba, compute frequencies, and draw a word cloud. The root-level script is a separate local demonstration of similarity, keywords, and edit distance and does not fetch pages. Cached responses, stop-word resources, and generated images are not included. Any web acquisition must respect current terms, robots policy, rate limits, and applicable law; authorized local text can replace collection.

## Reproduce with AI

```text
Inspect all scripts, identify which one is the most complete local-text workflow and which code attempts network collection, and list the expected dependencies and inputs. Prefer authorized local Chinese text; use network access only after verifying that collection is permitted. Reproduce cleaning, jieba segmentation, frequency analysis, and visualization as implemented. Fix only compatibility, encoding, or relative-path issues without changing token filters or analysis logic, report actual outputs, and clearly identify unavailable pages or resources.
```
