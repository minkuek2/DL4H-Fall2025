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

## Dataset (MIT-BIH Arrhythmia)

This project uses the Kaggle MIT-BIH Arrhythmia dataset:

https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Download:
mitbih_train.csv,
mitbih_test.csv

