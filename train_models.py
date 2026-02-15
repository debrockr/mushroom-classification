# train_models_fixed.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "data/mushrooms.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(filepath):
    """Load and preprocess the mushroom dataset"""
    print("Loading dataset...")
    
    # Read the CSV
    df = pd.read_csv(filepath)
    
    print(f"\n📊 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Target is first column ('class')
    target_col = df.columns[0]
    print(f"\n🎯 Target column: '{target_col}'")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check for missing values (represented as '?')
    print(f"\n🔍 Checking for missing values:")
    missing_counts = X.isin(['?']).sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"Found missing values in columns: {cols_with_missing.index.tolist()}")
        # Replace '?' with NaN
        X = X.replace('?', np.nan)
    else:
        print("No missing values found")
    
    print(f"\n📈 Target distribution:")
    print(y.value_counts())
    
    # Convert all columns to categorical and encode
    print(f"\n🔄 Encoding features...")
    
    # First, handle missing values by encoding '?' as a separate category
    X_encoded = pd.DataFrame()
    label_encoders = {}
    
    for col in X.columns:
        # Fill NaN with 'missing' string
        X[col] = X[col].fillna('missing')
        
        # Encode the column
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  - {col}: {len(le.classes_)} unique values")
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"\n🏷️ Target encoded - Classes: {target_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Data split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save metadata
    metadata = {
        'feature_names': X.columns.tolist(),
        'target_name': target_col,
        'target_classes': target_encoder.classes_.tolist(),
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'n_features': X.shape[1],
        'n_samples': len(df),
        'columns_with_missing': cols_with_missing.index.tolist() if len(cols_with_missing) > 0 else []
    }
    
    # Save test data for Streamlit
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df[target_col] = y_test
    test_df.to_csv("data/test_data.csv", index=False)
    print(f"\n✅ Test data saved to data/test_data.csv")
    
    return (X_train, X_test, y_train, y_test, 
            X_train_scaled, X_test_scaled, scaler, 
            metadata)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, 
                            X_train_scaled, X_test_scaled, use_scaled=False):
    """Train and evaluate a single model"""
    print(f"\n🔄 Training {model_name}...")
    
    # Use appropriate data
    if use_scaled:
        train_data = X_train_scaled
        test_data = X_test_scaled
    else:
        train_data = X_train
        test_data = X_test
    
    # Train the model
    model.fit(train_data, y_train)
    
    # Save the model
    with open(f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Make predictions
    y_pred = model.predict(test_data)
    
    # Get prediction probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(test_data)
        # For binary classification
        if len(np.unique(y_train)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    else:
        auc = None
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': auc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    print(f"✓ {model_name} - Accuracy: {metrics['Accuracy']:.4f}")
    
    return metrics, y_pred, y_test

def main():
    """Main training function"""
    print("=" * 60)
    print("MUSHROOM CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    (X_train, X_test, y_train, y_test, 
     X_train_scaled, X_test_scaled, scaler, 
     metadata) = load_and_preprocess_data(DATASET_PATH)
    
    # Save metadata
    with open(f"{MODELS_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    # Save scaler
    with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Define models with their scaling requirements
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'use_scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, max_depth=10),
            'use_scaled': False
        },
        'K-Nearest Neighbor': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'use_scaled': True
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'use_scaled': False
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            'use_scaled': False
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'use_scaled': False
        }
    }
    
    # Train all models
    results = []
    predictions = {}
    
    for model_name, config in models_config.items():
        metrics, y_pred, y_test_result = train_and_evaluate_model(
            config['model'], 
            model_name,
            X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            use_scaled=config['use_scaled']
        )
        
        results.append(metrics)
        predictions[model_name] = {'y_pred': y_pred, 'y_true': y_test_result}
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/metrics_comparison.csv", index=False)
    
    with open(f"{RESULTS_DIR}/predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
    
    # Save confusion matrices and classification reports
    confusion_matrices = {}
    classification_reports = {}
    
    for model_name, config in models_config.items():
        # Use appropriate data for predictions
        if config['use_scaled']:
            test_data = X_test_scaled
        else:
            test_data = X_test
            
        y_pred = config['model'].predict(test_data)
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
        classification_reports[model_name] = classification_report(y_test, y_pred, output_dict=True)
    
    with open(f"{RESULTS_DIR}/confusion_matrices.pkl", "wb") as f:
        pickle.dump(confusion_matrices, f)
    
    with open(f"{RESULTS_DIR}/classification_reports.pkl", "wb") as f:
        pickle.dump(classification_reports, f)
    
    # Display formatted results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - RESULTS")
    print("=" * 60)
    
    print("\n📊 Performance Comparison Table:")
    print("-" * 90)
    print(f"{'ML Model Name':<20} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        auc_val = f"{row['AUC']:.4f}" if row['AUC'] is not None else "N/A"
        print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {auc_val:<10} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1']:<10.4f} {row['MCC']:<10.4f}")
    
    print("-" * 90)
    
    # Find best model
    best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_acc = results_df['Accuracy'].max()
    print(f"\n🏆 Best Model: {best_model} with Accuracy: {best_acc:.4f}")
    
    print("\n✅ All models trained and saved successfully!")
    print(f"📁 Models saved in: {MODELS_DIR}/")
    print(f"📊 Results saved in: {RESULTS_DIR}/")
    print("\n📝 Next steps:")
    print("1. Review the results above")
    print("2. Run Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    main()