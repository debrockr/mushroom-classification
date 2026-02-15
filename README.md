# Multi-Model Classification Comparison

## a. Problem Statement
This project implements and compares six different classification machine learning models on the Mushroom Classification dataset. Models are trained locally, and results are displayed through a Streamlit web application.

## b. Dataset Description
**Dataset**: Mushroom Classification (UCI Machine Learning Repository)
- **Features**: 22 categorical features
- **Instances**: 8,124
- **Target**: Binary (edible/poisonous)
- **Characteristics**: All features are categorical, requiring encoding

## c. Models Used and Evaluation Metrics

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.9523 | 0.9876 | 0.9524 | 0.9523 | 0.9523 | 0.9047 |
| **Decision Tree** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **kNN** | 0.9985 | 0.9998 | 0.9985 | 0.9985 | 0.9985 | 0.9970 |
| **Naive Bayes** | 0.9185 | 0.9785 | 0.9215 | 0.9185 | 0.9180 | 0.8375 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-----------------------------------|
| **Logistic Regression** | Performs well with 95.23% accuracy. Being a linear model, it handles the encoded categorical features effectively but may miss complex non-linear patterns. Good balance between performance and interpretability. |
| **Decision Tree** | Achieves perfect classification on this dataset. The tree structure can capture all decision rules in the mushroom dataset perfectly. However, this might indicate potential overfitting despite the perfect scores. |
| **kNN** | Near-perfect performance (99.85% accuracy). The dataset has clear separability between classes, making distance-based classification very effective. Slightly lower than tree-based models due to the high-dimensional feature space. |
| **Naive Bayes** | Lowest performance among all models (91.85% accuracy). The assumption of feature independence is violated in this dataset as many mushroom characteristics are correlated. Still provides respectable performance due to dataset simplicity. |
| **Random Forest** | Perfect classification achieved. Ensemble of trees captures all patterns without overfitting due to averaging. Most robust model for this type of structured data with clear decision boundaries. |
| **XGBoost** | Perfect classification with efficient gradient boosting. Handles the categorical features exceptionally well after encoding. Provides the fastest training among perfect-scoring models with built-in regularization. |

## Project Structure
assignment2/
├── train_models.py # Local training script
├── app.py # Streamlit demo app
├── requirements.txt # Dependencies
├── README.md # This file
├── data/ # Dataset directory
│ ├── mushroom.csv # Original dataset
│ └── test_data.csv # Saved test data for demo
├── models/ # Saved trained models
│ ├── logistic_regression.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ ├── scaler.pkl
│ └── metadata.pkl
└── results/ # Saved results
├── metrics_comparison.csv
├── confusion_matrices.pkl
├── classification_reports.pkl
└── predictions.pkl