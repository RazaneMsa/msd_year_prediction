# Music Decade Prediction from Audio Features

This project explores the use of machine learning models to predict the release decade of a song based solely on audio features.  
The work is based on the Million Song Dataset (MSD) and was developed as part of an academic project in machine learning and advanced programming.

# Dataset

The Dataset used in this project is the Year Prediction MSD dataset from the UCI Machine Learning Repository.
Due to its large size, the dataset is not included in this repository.
**Features**: High-dimensional audio descriptors extracted from songs
**Target**: Song release decade (treated as a multi-class classification problem)

The dataset is highly imbalanced across decades, which motivates the use of balanced evaluation metrics.

Link of the dataset : https://archive.ics.uci.edu/dataset/203/yearpredictionmsd 


# Project Overview

The objective of this project is to evaluate how well different machine learning models can capture long-term musical trends and temporal information from audio features.

Several classification models are trained and compared, ranging from linear methods to more advanced ensemble and gradient-boosting approaches.  
Special attention is given to class imbalance across decades and to appropriate evaluation metrics.

# Models Implemented

The following models are implemented and compared:

- Linear Discriminant Analysis (LDA)
- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- XGBoost (standard and capped variants)
- Ordinal Ridge Regression (using ordinal encoding of decades)

Each model is evaluated against both true labels and shuffled labels to estimate chance-level performance.

# Evaluation Metrics

To account for class imbalance, the following metrics are used:

- **Accuracy**
- **Balanced Accuracy**
- **Macro F1-score**

# Code Structure

The main notebook includes:

- Data preprocessing and feature scaling
- Model training using pipelines where appropriate
- Evaluation on a held-out test set
- Comparison with chance-level baselines (label shuffling)
- Hyperparameter optimization using Optuna (for Random Forest and XGBoost)
- Visualization of results and confusion matrices
- Feature importance analysis for tree-based models

# Note on Execution Time

Some parts of the notebook, especially hyperparameter optimization and advanced models are computationally expensive and can take several hours to run.
For this reason, some intermediate results are saved as '.pkl' files.
These files allow the results to be reproduced without re-running computationally expensive steps.

# Author
**Razane Moossa**  
