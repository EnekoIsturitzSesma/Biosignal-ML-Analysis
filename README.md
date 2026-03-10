# EEGNet-Motor-Imagery-BCI

Development and evaluation of EEGNet-based architectures for motor imagery classification in Brain–Computer Interfaces (BCI).

## Project Overview

This repository implements and evaluates EEG-based motor imagery classification models using state-of-the-art deep learning (EEGNet) and classical pipelines (CSP + traditional classifier).

- Main dataset: BCIC IV 2a (GDF)
- Models: EEGNet (PyTorch) and CSP + classifier (LDA/SVM)
- Goal: compare performance and establish a reproducible workflow

## Repository Structure

- `datasets/BCICIV_2a_gdf/`: GDF training and test files (A01..A09)
- `src/load_data_BCICIV.py`: data loading and dataset assembly
- `src/preprocess.py`: filtering, epoching, and transformations
- `src/train_CSP.py`: CSP pipeline + classifier training/evaluation
- `src/train_EEGNet.py`: EEGNet training and evaluation
- `models/EEGNet.py`: EEGNet architecture definition
- `notebooks/`: exploratory analysis, training experiments, visualization
- `figures/`: generated figures

## Requirements

Use a Python virtual environment (`venv`) to isolate dependencies and ensure reproducibility. This project uses `pyproject.toml` and `uv.lock` for dependency management.

```bash
source .venv/bin/activate
uv sync
```

Key dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

Main dependencies include:
- numpy
- scipy
- mne
- torch
- torchvision
- scikit-learn
- matplotlib

## Notebooks

- `notebooks/EEGNet.ipynb`: EDA and EEGNet training
- `notebooks/CSP_Classifier.ipynb`: CSP + classifier pipeline
- `notebooks/visualization.ipynb`: signal visualizations and confusion matrices

## Pipeline Details

### Data loading
- `src/load_data_BCICIV.py` reads `.gdf` files, extracts events, and builds motor imagery epochs.

### Preprocessing
- Bandpass filtering (7-30 Hz)
- Normalization
- Laplacian filter

### Models
- `models/EEGNet.py`: 2D convolutional block, depthwise conv, separable conv, batch norm, dropout
- Loss: CrossEntropyLoss
- Optimizer: Adam

## Metrics & Evaluation

- Accuracy
- Per subject cross-validation (Leave One Subject Out)

## Contact

- Author: Eneko Isturitz Sesma

