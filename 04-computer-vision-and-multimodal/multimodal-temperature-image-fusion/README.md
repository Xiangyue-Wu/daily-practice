# Temperature and Image Fusion for Fire Detection

## Overview

This project explores fire detection from two complementary modalities: temperature sensor readings and scene images. Its scripts cover data cleanup and filename alignment, visual feature extraction, and learned fusion of sensor and image information.

• Prepared paired temperature and image records by validating filenames, removing unusable samples, and producing a cleaned multimodal index.

• Implemented alternative fire-detection paths using CLIP visual features and a PyTorch FiLM model conditioned by temperature sequences.

## Approach

`src/wash.py` is the preparation entry point: it reads `data/mn.xlsx`, validates paired images under `data/dataset`, and writes `data/cleaned_dataset.csv`. After that, `src/clip.py` runs the CLIP-feature classifier, while `src/f2.py` is an alternative PyTorch fusion model in which temperature sequences generate FiLM conditioning for visual features. The filename utility is optional and should be used only before pairing data. Required spreadsheet records, image folders, CLIP downloads, weights, and output figures are not included, so no detection accuracy is claimed.

## Reproduce with AI

```text
Read all four scripts and map the data-cleaning, renaming, CLIP feature, and multimodal fusion stages. Determine the required spreadsheet columns, image layout, pretrained components, and entry points. Prepare legally usable paired temperature/image data and reproduce the existing fusion variants, reporting measured classification results. Fix only dependency, device, filename, or relative-path compatibility issues; do not alter fusion logic, model structure, preprocessing, or hyperparameters. Clearly identify missing paired data, pretrained weights, or ambiguous script order.
```
