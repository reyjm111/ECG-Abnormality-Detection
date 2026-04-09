# EKG Abnormality Detection

A machine learning project for detecting abnormal heartbeats from electrocardiogram EKG waveforms. This repository focuses on building an end-to-end workflow for organizing raw files, converting signals for analysis, preprocessing heartbeat segments, training a 1D convolutional neural network, and evaluating classification performance.

## Overview

This project was built to explore how deep learning can be applied to physiological time-series data for binary heartbeat classification. The pipeline covers:

- file organization for ECG records
- conversion of source waveform files into analysis-ready format
- preprocessing and heartbeat segmentation
- aggregation of samples across records
- training a 1D CNN for abnormality detection
- evaluation of model performance across folds

## Project Goals

The main goals of this project are to:

- build a reproducible ECG abnormality detection pipeline
- classify heartbeats as normal vs abnormal
- evaluate performance with appropriate validation splits
- demonstrate practical machine learning skills on biomedical waveform data

## Repository Structure

ekg-abnormality/
├── misc/
│   ├── edf_converter.py
│   └── file_organization.py
├── model/
│   ├── cnn_model.py
│   ├── collect_files.py
│   ├── file_aggregation.py
│   ├── main.ipynb
│   ├── obtain_metrics.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt