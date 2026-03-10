# EEGNet-Motor-Imagery-BCI

Development and evaluation of EEGNet-based architectures for motor imagery classification in Brain–Computer Interfaces (BCI).

## 📌 Project Overview

This repository implements and evaluates EEG-based motor imagery classification models using state-of-the-art deep learning (EEGNet) and classical pipelines (CSP + traditional classifier).

- Main dataset: BCIC IV 2a (GDF)
- Models: EEGNet (PyTorch) and CSP + classifier (e.g. LDA/SVM)
- Goal: compare performance and establish a reproducible workflow

## 🗂️ Repository Structure

- `datasets/BCICIV_2a_gdf/`: GDF training and test files (A01..A09)
- `src/load_data_BCICIV.py`: data loading and dataset assembly
- `src/preprocess.py`: filtering, epoching, and transformations
- `src/train_CSP.py`: CSP pipeline + classifier training/evaluation
- `src/train_EEGNet.py`: EEGNet training and evaluation
- `models/EEGNet.py`: EEGNet architecture definition
- `notebooks/`: exploratory analysis, training experiments, visualization
- `figures/`: generated figures (optional)

## 🛠️ Requirements

Use a Python virtual environment (`venv`) to isolate dependencies and ensure reproducibility. This project uses `pyproject.toml` and `uv.lock` for dependency management.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install uv
uv install
```

Key dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

Typical dependencies include:
- numpy
- scipy
- mne
- torch
- torchvision
- scikit-learn
- matplotlib
- seaborn

## 🚀 Quick Start

1. Download BCIC IV 2a data into `datasets/BCICIV_2a_gdf/`.
2. Run preprocessing (optional custom configuration):
    ```bash
    python src/preprocess.py
    ```
3. Train EEGNet:
    ```bash
    python src/train_EEGNet.py
    ```
4. Train CSP + classifier:
    ```bash
    python src/train_CSP.py
    ```

## 📊 Notebooks

- `notebooks/EEGNet.ipynb`: EDA and EEGNet training
- `notebooks/CSP_Classifier.ipynb`: CSP + classifier pipeline
- `notebooks/visualization.ipynb`: signal visualizations and confusion matrices

## 🧾 Pipeline Details

### Data loading
- `src/load_data_BCICIV.py` reads `.gdf` files, extracts events, and builds motor imagery epochs.

### Preprocessing
- Bandpass filtering (4-40 Hz)
- Normalization and optional resampling
- Epoch creation and artifact rejection pipeline

### Models
- `models/EEGNet.py`: 2D convolutional block, depthwise conv, separable conv, batch norm, dropout
- Loss: CrossEntropyLoss
- Optimizer: Adam

## ✅ Metrics & Evaluation

- Accuracy
- Confusion matrix
- Per-class performance (left hand, right hand, feet, tongue)
- Training and validation curves (loss and accuracy)

## 🧪 Suggested Extensions

- Subject-wise cross-validation
- Online/rolling window classification
- Adversarial regularization / stronger dropout
- Transfer learning across subjects

## 📁 Expected Outcomes

Performance depends on hyperparameter configuration. Similar studies on BCIC IV 2a typically report 75-85% accuracy.

## 🛡️ Best Practices

- Use version control (Git)
- Avoid committing raw data (GDF files) if large; update `.gitignore` accordingly
- Save model checkpoints in `models/` or `checkpoints/`

## 💬 Contact

- Author: (add your name here)
- [Optional] Link to paper, blog, or presentation.

