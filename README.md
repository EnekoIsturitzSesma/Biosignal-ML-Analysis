# Biosignal ML Analysis

## Overview
This repository focuses on analyzing biosignals through various methodologies, with an emphasis on Machine Learning. The toolset encompasses a range of functionalities catering to different types of signal analysis, notably EEG and LSTM gait analysis.

## Functionalities

### EEG Analysis
- **Description:** This module facilitates the analysis of electroencephalogram (EEG) data to extract meaningful information regarding brain activity.
- **Key Features:**
  - Signal pre-processing (filtering, normalization)
  - Feature extraction (time-domain, frequency-domain)
  - Visualization tools (spectrograms, time series)
- **Usage:** Run `eeg_analysis.py` with your dataset to obtain analysis results.

### LSTM Gait Analysis
- **Description**: Implements Long Short-Term Memory (LSTM) networks for analyzing gait patterns.
- **Key Features**: 
  - Data collection from motion sensors
  - LSTM network training and evaluation
  - Visualization of gait patterns and predictions
- **Usage**: Utilize `lstm_gait_analysis.ipynb` for a step-by-step analysis framework.

## Notebooks and Source Modules
- **Notebooks:**
  - `eeg_notebook.ipynb`: Interactive analysis of EEG data.
  - `gait_analysis_notebook.ipynb`: Step-by-step LSTM implementation.

- **Source Modules:**
  - `eeg_analysis.py`: Contains functions for EEG signal processing.
  - `lstm_gait_analysis.py`: Contains functions for LSTM training and evaluation.

## Installation
Follow the instructions in the repository to set up the necessary environment, including libraries and dependencies to run the analyses.

## Contributions
For contributions and feature requests, please open an issue or submit a pull request. Your feedback is much appreciated!