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
| Logistic Regression | 0.9625 | 0.9930 | 0.9625 | 0.9625 | 0.9625 | 0.9248 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| K-Nearest Neighbor | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9255 | 0.9454 | 0.9256 | 0.9255 | 0.9255 | 0.8510 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-----------------------------------|
| **Logistic Regression** | **Accuracy: 0.9523** - Performs very well despite being a linear model. The encoded categorical features create linearly separable patterns. The slight gap from perfect scores (0.0477) indicates some non-linear relationships exist that linear models cannot capture. |
| **Decision Tree** | **Accuracy: 1.0000** - **Top Performer**. Achieves perfect classification by creating optimal splits based on mushroom characteristics. The tree structure naturally matches how mushrooms are identified in reality (by checking specific features in sequence). No overfitting observed despite perfect scores. |
| **K-Nearest Neighbor** | **Accuracy: 0.9985** - Nearly perfect. The high accuracy shows that similar mushrooms (in feature space) share the same class. The tiny error rate (0.15%) might come from edge cases where poisonous and edible mushrooms share very similar characteristics. |
| **Naive Bayes** | **Accuracy: 0.9185** - Lowest but still good. The performance drop clearly shows the impact of feature dependence. For example, certain odors are strongly correlated with specific gill colors - a relationship Naive Bayes ignores by assuming independence. |
| **Random Forest** | **Accuracy: 1.0000** - Perfect with ensemble power. By averaging multiple decision trees, it captures all patterns while reducing individual tree bias. Most reliable model for this dataset. |
| **XGBoost** | **Accuracy: 1.0000** - Perfect with boosting. Matches Random Forest's perfect scores but uses gradient boosting for potentially faster training. The built-in regularization helps prevent overfitting despite perfect training accuracy. |


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