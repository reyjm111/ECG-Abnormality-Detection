# ECG Abnormality Detection

A deep learning project for detecting abnormal heartbeats from electrocardiogram (ECG/EKG) waveforms. This repository implements an end-to-end pipeline for organizing raw waveform files, preprocessing cardiac signals, segmenting heartbeat windows, training a 1D convolutional neural network (CNN), and evaluating binary heartbeat classification performance on physiological time-series data.

## Overview

This project explores how deep learning can be applied to biomedical waveform data for ECG abnormality detection. The workflow covers raw data organization, signal preprocessing, segmentation of heartbeat-centered windows, dataset aggregation across records, model development, and evaluation using validation strategies designed to reduce data leakage across patients.

The project was built to strengthen practical skills in:

- biomedical signal processing
- time-series preprocessing
- deep learning for waveform classification
- class imbalance handling
- leakage-aware validation
- performance evaluation on physiological data

## Project Goals

The main goals of this project are to:

- build a reproducible ECG abnormality detection pipeline
- classify heartbeats as normal versus abnormal
- apply preprocessing and segmentation techniques to ECG waveform data
- reduce data leakage through group-aware validation
- evaluate model performance on imbalanced physiological time-series data
- demonstrate practical machine learning and biomedical signal processing skills

## Dataset

This project uses the **MIT-BIH Arrhythmia Database**, a widely used public dataset for ECG analysis and arrhythmia detection.

The dataset contains annotated ECG recordings with beat-level labels that support supervised classification. For this project, heartbeat segments were extracted and grouped into a **binary classification task**:

- **Normal**
- **Abnormal**

## Pipeline

The overall workflow is shown below:

1. organize raw ECG waveform files
2. convert source files into analysis-ready format
3. preprocess ECG signals
4. segment heartbeat-centered waveform windows
5. aggregate samples across records
6. split data using group-aware validation
7. train a 1D CNN for classification
8. evaluate performance across folds

## Methods

### 1. Data Organization

Raw ECG records were organized into a consistent directory structure to support preprocessing and model training. This step ensured that waveform files, annotations, and derived outputs could be processed reproducibly across records.

### 2. Preprocessing

The preprocessing workflow included:

- filtering ECG waveforms
- scaling/normalization
- preparing waveform segments for model input
- extracting heartbeat-centered windows

These steps were used to improve signal consistency and prepare the data for deep learning.

### 3. Segmentation

Heartbeat-level segments were created from the ECG recordings so that each training example represented a local waveform pattern centered around an annotated beat. These segments served as the input samples for binary heartbeat classification.

### 4. Modeling

A **1D convolutional neural network (CNN)** was developed for ECG abnormality detection. The model was designed to learn discriminative temporal patterns directly from ECG waveform segments without requiring manual feature engineering.

### 5. Validation Strategy

To reduce patient-level data leakage, the project used **group-aware train/test splitting** and **10-fold stratified group cross-validation**. This ensured that waveform segments from the same subject or record were not improperly shared across training and validation sets.

### 6. Class Imbalance Handling

Because abnormality detection often involves imbalanced class distributions, targeted sampling strategies were applied to improve minority-class representation during training.

## Model Architecture

The core classifier is a **1D CNN** trained on segmented ECG waveforms. Convolutional layers were used to learn local temporal patterns associated with normal and abnormal beats, while downstream layers mapped these learned representations to binary class predictions.

This architecture was selected because CNNs are well suited for physiological waveform classification, where local shape, amplitude changes, and temporal morphology can be informative for abnormality detection.

## Results

The model achieved strong performance on binary ECG abnormality classification using group-aware cross-validation.

**Mean cross-validation performance:**

- **AUROC:** 0.952
- **Accuracy:** 0.873
- **Recall:** 0.822
- **F1-score:** 0.760

These results suggest that the model was effective at distinguishing normal and abnormal heartbeat segments while maintaining robustness under leakage-aware validation.
