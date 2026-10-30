# Intelligent Sentiment Analysis Web Application

## Overview

This project provides a Flask-based interface for analyzing Chinese text sentiment. It supports direct text requests and tabular uploads, returns model scores through JSON endpoints, and generates word-cloud summaries for batches of text.

• Built Flask endpoints for single-text sentiment analysis and authorized CSV/XLSX batch processing with Transformers, SnowNLP, JSON responses, and word-cloud summaries.

• Developed a browser interface with model cards, confidence displays, batch summaries, and a mock-data route that demonstrates the workflow without contacting the submitted URL.

## Approach

The backend combines a Transformers sentiment pipeline with SnowNLP and exposes routes for the page, single-text analysis, CSV/XLSX upload, and a mock-data demonstration. The mock route accepts a URL-shaped input but does not contact that URL. The frontend presents model cards, confidence bars, batch summaries, and visual content. Importing the backend attempts to load the named pretrained model, which may require a large download. Model files, generated assets, and deployment configuration are not included, so production readiness is not claimed.

## Reproduce with AI

```text
Inspect app.py and templates/index.html, map every Flask route to its frontend request, and identify the expected Transformers model, SnowNLP, spreadsheet, plotting, and font dependencies. Configure a local environment, verify model licensing before obtaining weights, and reproduce text and authorized local-file analysis. Report actual endpoint behavior and model outputs. Fix only dependency, model-loading, template, or relative-path compatibility issues; do not change model choices, scoring logic, routes, or UI behavior. Do not contact external sites without explicit authorization, and list missing models or assets.
```
