# ECG Data Augmentation Reproduction (MIT-BIH Arrhythmia Dataset)

This repository contains the full implementation for the reproduction study  
of **“ECG Data Augmentation: Towards Improved Arrhythmia Classification.”**  
The project re-implements the baseline 1D CNN classifier, evaluates several  
ECG-specific augmentation methods, and performs ablation studies using the  
MIT-BIH Arrhythmia dataset.

This work was completed as the final project for **CS 441: Applied Machine Learning**  
at the University of Illinois Urbana-Champaign.

---

## Project Overview

The goal of this reproduction study is to verify the claims of the original ECG
augmentation paper by:

- Reproducing the baseline classification model (1D CNN)
- Implementing all augmentation methods from the paper:
  - jitter  
  - scaling  
  - jitter + scaling  
  - magnitude warp  
  - time warp  
- Evaluating their impact on arrhythmia classification performance
- Conducting an additional ablation on augmentation strength

---

## Repository Structure

---
│
├── data/                      # MIT-BIH CSV files (not included)
│
├── dataset.py                 # Custom PyTorch dataset loader
├── augmentations.py           # All ECG augmentation functions
├── model.py                   # Baseline 1D CNN implementation
├── train.py             # Training / evaluation loops
│
├── 01_dataset_exploration.ipynb   # Data inspection + visualization
├── 02_baseline_training.ipynb      # Baseline CNN reproduction
├── 03_augmentation_experiments.ipynb # Augmentation experiments + ablations
│
├── results/
│   ├── figures/
│   │   ├── loss_curves.png
│   │   ├── aug_comparison.png
│   │   └── jitter_ablation.png
│   └── metrics/*.json
│
└── report/
├── main.tex               # Final AAAI-style reproduction report
└── references.bib

## Dataset (MIT-BIH Arrhythmia)

This project uses the Kaggle MIT-BIH Arrhythmia dataset:

https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Download:
mitbih_train.csv,
mitbih_test.csv

