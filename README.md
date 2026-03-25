# Biosignal ML Analysis

## Overview
This repository is dedicated to the analysis of biosignal data, implementing various functionalities related to machine learning and signal processing for applications such as EEG analysis and gait analysis.

## Functionalities

### CSP Classifier
- **Description:** Implements the Common Spatial Patterns (CSP) algorithm for feature extraction from EEG signals.
- **Usage:** Provides methods to apply CSP on EEG datasets, enhancing the ability to classify mental states.

### EEGNet
- **Description:** A hybrid deep learning architecture designed specifically for decoding EEG signals.
- **Features:** 
  - Efficient in handling temporal and spatial dimensions of EEG data.
  - Capable of working with small datasets.

### GAIT Analysis
- **Description:** Tools and methodologies for analyzing gait patterns using biosignal data.
- **Applications:** Useful in rehabilitation and sports science for assessing movement patterns.

## Source Modules
- `csp_classifier.py`: Implements the CSP algorithm.
- `eegnet.py`: Contains the EEGNet architecture.
- `gait_analysis.py`: Offers functionalities for gait analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/EnekoIsturitzSesma/Biosignal-ML-Analysis.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Refer to the respective module documentation for detailed usage instructions. 
