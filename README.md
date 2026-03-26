# Biosignal-ML-Analysis

> Machine learning methods for biosignal analysis applied to neurological diseases.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE-Python-6a0dad)](https://mne.tools/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Module 1 — EEG Motor Imagery (BCI)](#-module-1--eeg-motor-imagery-bci)
- [Module 2 — Gait Event Detection](#-module-2--gait-event-detection)
- [Models](#-models)
- [Experiment Tracking with MLflow](#-experiment-tracking-with-mlflow)
- [Installation](#-installation)
- [Running with Docker](#-running-with-docker)
- [Results](#-results)
- [Contact](#-contact)

---

## Overview

This repository implements and evaluates machine learning and deep learning pipelines for two complementary biosignal analysis tasks applied to neurological research:

| Module | Signal | Task | Models |
|--------|--------|------|--------|
| **EEG — BCI** | Electroencephalography | 4-class motor imagery classification | EEGNet, CSP + LDA/SVM |
| **Gait** | Inertial / kinematic signals | Gait event detection (heel strike, toe-off) | LSTM |

Both modules share a common infrastructure: reproducible environment via `uv` and Docker, and experiment tracking via MLflow.

---

## Repository Structure

```
Biosignal-ML-Analysis/
│
├── notebooks/
│   ├── EEGNet.ipynb            # EEG: EEGNet training and evaluation
│   ├── CSP_Classifier.ipynb    # EEG: CSP + classical classifier pipeline
│   ├── GAIT.ipynb              # Gait event detection: LSTM and CNN experiments
│   ├── visualization.ipynb     # Signal visualizations and result plots 
│
├── src/
│   ├── load_data_BCICIV.py     # EEG data loader (GDF → epochs)
│   ├── load_data_gait.py       # Gait signal loader and segmentation
│   ├── plot_data_gait.py       # Gait signal visualization utilities
│   ├── preprocess.py           # Shared preprocessing (filtering, normalization, epoching)
│   ├── train_CSP.py            # CSP pipeline: feature extraction + classifier training
│   ├── train_EEGNet.py         # EEGNet training loop + evaluation
│   └── train_LSTMGait.py       # LSTM training loop for gait event detection
│
├── models/
│   ├── EEGNet.py               # EEGNet architecture (PyTorch)
│   ├── CNN.py                  # ShallowConvNet and DeepConvNet architectures
│   └── LSTMGait.py             # LSTM architecture for gait event detection (PyTorch)
│
├── results/
│   ├── results_raw_gait_lstm.csv           # LSTM results on raw gait signals
│   └── results_preprocessed_gait_lstm.csv  # LSTM results on preprocessed gait signals
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

---

## Module 1 — EEG Motor Imagery (BCI)

### Objective

Classify four motor imagery tasks (left hand, right hand, feet, tongue) from 22-channel EEG recordings, towards brain-computer interface (BCI) applications.

### Dataset

**BCI Competition IV — Dataset 2a**
- https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
- 9 subjects, 22 EEG channels + 3 EOG channels
- 4 classes: left hand, right hand, feet, tongue
- 288 trials per subject (72 per class)`

### Preprocessing (`src/preprocess.py`, `src/load_data_BCICIV.py`)

1. Read `.gdf` files and extract motor imagery events
2. **Bandpass filter** — 7–30 Hz (mu + beta rhythms)
3. **Laplacian spatial filter** — local surface Laplacian for noise reduction
5. **Z-score normalization** per channel

### Approaches

#### A) EEGNet — Deep Learning (`models/EEGNet.py`, `src/train_EEGNet.py`)

A compact convolutional neural network designed specifically for EEG signals

#### B) CSP + Classifier — Classical Pipeline (`src/train_CSP.py`)

1. **Common Spatial Patterns (CSP)** — learns spatial filters that maximize variance ratio between classes
3. **Classifier** — LDA or SVM trained on extracted features

### Evaluation

- Strategy: **Leave-One-Subject-Out (LOSO)** cross-validation
- Metrics: Accuracy

### Notebooks

- `notebooks/EEGNet.ipynb` — EDA, full training run, per-subject results
- `notebooks/CSP_Classifier.ipynb` — CSP pipeline, feature visualization, classifier comparison

---

## Module 2 — Gait Event Detection

### Objective

Detect gait events (e.g., heel strike, toe-off) from inertial or kinematic biosignals. This task is relevant to the monitoring and analysis of neurological conditions that affect locomotion, such as Parkinson's disease.

### Dataset

https://springernature.figshare.com/ndownloader/files/53704514

### Preprocessing

The module evaluates two experimental conditions to assess the impact of preprocessing on detection performance:

- **Raw** — signals fed directly to the model after basic loading and windowing
- **Preprocessed** — includes filtering, normalization, and signal conditioning

This comparison is reflected in the saved results CSV files.

### Models

#### LSTM (`models/LSTMGait.py`, `src/train_LSTMGait.py`)

A recurrent architecture that exploits the sequential and temporal nature of gait signals:

The LSTM is trained and evaluated independently on raw and preprocessed signals to quantify the contribution of preprocessing to detection performance.

### Evaluation

- Metrics: F1-score
- Key comparison: raw vs. preprocessed signals (see `results/`)

### Notebook

`notebooks/GAIT.ipynb` — data loading, signal visualization, LSTM and CNN training, raw vs. preprocessed performance comparison

---

## Models

| File | Architecture | Task |
|------|-------------|------|
| `models/EEGNet.py` | 2D CNN (temporal + depthwise + separable convolutions) | EEG motor imagery |
| `models/CNN.py` | 2 different CNN architectures | EEG motor imagery |
| `models/LSTMGait.py` | LSTM | Gait event detection |

All models are implemented in **PyTorch**.

---

## Experiment Tracking with MLflow

All training runs are logged with [MLflow](https://mlflow.org/), tracking hyperparameters, metrics, and model artifacts across experiments.

```bash
# Launch the MLflow UI from the project root
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db

# Open in browser
open http://localhost:5000
```

---

## Installation

Requires **Python ≥ 3.11** and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/EnekoIsturitzSesma/Biosignal-ML-Analysis.git
cd Biosignal-ML-Analysis

# Create virtual environment and install locked dependencies
uv sync

# Activate
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Launch Jupyter
jupyter notebook
```

### Dependencies (`pyproject.toml`)

| Package | Role |
|---------|------|
| `numpy`, `scipy` | Numerical computing and signal processing |
| `pandas` | Data loading and results management |
| `matplotlib` | Visualization |
| `mne` | EEG / GDF file reading, filtering, epoching |
| `torch` | Deep learning models (EEGNet, LSTM, CNN) |
| `scikit-learn` | CSP, LDA, SVM, evaluation metrics |
| `mlflow` | Experiment tracking |
| `tqdm` | Progress bars |

---

## Running with Docker

A pre-configured Docker environment exposes Jupyter Notebook on port **8888**.

```bash
# Build and launch
docker compose up --build

# Access Jupyter in your browser
open http://localhost:8888
```

---

## Results

LSTM gait event detection results are saved as CSV files for reproducibility and post-hoc analysis:

| File | Description |
|------|-------------|
| `results/results_raw_gait_lstm.csv` | Per-subject metrics for LSTM trained on raw signals |
| `results/results_preprocessed_gait_lstm.csv` | Per-subject metrics for LSTM trained on preprocessed signals |

---

## Contact

**Author:** Eneko Isturitz Sesma  
GitHub: [@EnekoIsturitzSesma](https://github.com/EnekoIsturitzSesma)

---
