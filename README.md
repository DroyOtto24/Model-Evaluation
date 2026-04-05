# Credit Card Fraud Detection – Model Evaluation

A comprehensive evaluation of two machine learning models for credit card fraud detection using synthetic transaction data. This project focuses on handling imbalanced datasets and comparing model performance using key classification metrics and visualizations.

## Overview

Fraud detection is a classic imbalanced classification problem. This notebook:
- Generates a synthetic dataset with realistic patterns for legitimate and fraudulent transactions.
- Trains **Logistic Regression** and **Random Forest** classifiers.
- Evaluates models using accuracy, precision, recall, F1-score, and confusion matrices.
- Provides clear visual comparisons to help choose the best model for fraud detection.

## Dataset

The synthetic dataset contains 5,000 transactions with a 5% fraud rate (250 fraudulent, 4750 legitimate). Features include:

| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount (normal distribution, higher mean for fraud) |
| `hour`   | Hour of day (fraudulent transactions more common late night) |
| `v1, v2, v3` | Anonymized numerical features with distinct distributions for fraud vs. legitimate |
| `Class`  | Target: 0 = legitimate, 1 = fraudulent |

The data is split into 80% training / 20% testing, stratified to preserve class imbalance.

## Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model with L2 regularization, `max_iter=1000` |
| **Random Forest** | Ensemble of 100 decision trees, handles non‑linear patterns well |

Both models are trained on scaled features (`StandardScaler`).

## Evaluation Metrics

Because the dataset is imbalanced, we focus on:

- **Accuracy** – overall correctness (can be misleading for imbalanced data)
- **Precision** – how many predicted frauds are actually fraud
- **Recall** – how many actual frauds are correctly caught
- **F1‑Score** – harmonic mean of precision and recall

## Results

From the notebook output (example run):

| Metric               | Logistic Regression | Random Forest |
|----------------------|---------------------|---------------|
| Accuracy             | 0.9930              | 0.9920        |
| Precision            | 1.0000              | 0.9565        |
| Recall               | 0.8600              | 0.8800        |
| F1‑Score             | 0.9247              | 0.9167        |

- **Logistic Regression** achieves perfect precision (no false positives) but slightly lower recall.
- **Random Forest** has slightly higher recall (catches more fraud) but lower precision.

Both models perform very well due to the distinct patterns engineered into the synthetic data.

## Visualizations

The notebook generates two key visual outputs:

1. **Confusion Matrices** – side‑by‑side heatmaps for both models, showing true vs. predicted labels.
2. **Metrics Comparison Bar Chart** – a grouped bar chart comparing accuracy, precision, recall, and F1‑score.

Both plots are saved as PNG files (`confusion_matrices.png`, `metrics_comparison.png`).

## How to Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
