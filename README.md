# 🌀 Tropical Cyclone Formation Prediction

An end-to-end machine learning pipeline for tropical cyclone formation classification.

## Contents

| File | Description |
|------|-------------|
| `cyclone_prediction_pipeline.ipynb` | Full ML pipeline notebook (EDA → data cleaning → model training → evaluation) |
| `cyclone_dataset-1.csv` | Raw cyclone dataset (~240 KB) |

## Pipeline Overview

1. **Load & Explore** — initial data inspection and class balance check  
2. **Data Leakage Diagnosis** — identifies and removes the leaky feature causing 100% accuracy  
3. **Data Cleaning** — noise injection for realistic model training  
4. **Model Training** — compares Random Forest, Logistic Regression, and SVM  
5. **Evaluation** — accuracy, confusion matrix, ROC/AUC, and feature importance  
6. **Conclusions** — model recommendation

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

Open `cyclone_prediction_pipeline.ipynb` in Jupyter and run all cells top to bottom.

> **Note:** Trained model files (`.pkl`, `.h5`, etc.) are **not** committed to this repo.  
> Use [Git LFS](https://git-lfs.github.com/) or [DVC](https://dvc.org/) for large model artifacts.
