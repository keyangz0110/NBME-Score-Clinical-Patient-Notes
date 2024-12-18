# NBME - Score Clinical Patient Notes

Team Members:

- Keyang Zhang
- Wenhan Jia
- Shaodong Hu

## Overview

This notebook implements an ensemble model for the NBME - Score Clinical Patient Notes competition on Kaggle. The goal is to identify specific clinical concepts in patient notes.

## Model Architecture

The solution uses an ensemble of DeBERTa models:

- `DeBERTa-V3-Large`
- `DeBERTa-Large`
- `DeBERTa-Large-MNLI`
- `DeBERTa-XLarge`

## Key Components

### Custom Model

The core model architecture is defined in the CustomModel class:

- Uses pretrained DeBERTa models
- Adds dropout and linear layers
- Implements custom weight initialization

### Data Processing

Several utility functions handle data preprocessing:

- `process_feature_text()`: Cleans and standardizes feature text
- `clean_spaces()`: Normalizes whitespace
- `prepare_input_fast()`: Tokenizes and prepares model inputs
- `get_char_probs()`: Converts token predictions to character probabilities

### Evaluation Metrics

The model uses micro F1 score for evaluation.

### Inference Pipeline

The inference process includes:

- Loading pretrained models
- Making predictions
- Ensembling predictions
- Post-processing predictions

### Model Ensemble Weights

The final ensemble uses the following weights:

- `DeBERTa-V3-Large`: 0.55
- `DeBERTa-XLarge`: 0.20
- `DeBERTa-Large-MNLI`: 0.15
- `DeBERTa-Large`: 0.10

## Notes

- Uses gradient checkpointing and mixed precision training
- Implements custom data loading for efficiency
- Includes extensive data preprocessing and cleaning
- Employs model ensembling for improved performance

## Link

- [Kaggle Competition Overview](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/overview)
