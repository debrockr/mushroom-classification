"""
Streamlit Demo App for Mushroom Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification - Model Comparison",
    page_icon="🍄",
    layout="wide"
)

# Title and description
st.title("🍄 Mushroom Classification - Model Comparison")
st.markdown("""
This application demonstrates 6 pre-trained classification models on the Mushroom dataset.
All models were trained locally and the results are displayed here.
""")

# Check if results exist
RESULTS_DIR = "results"
MODELS_DIR = "models"

if not os.path.exists(RESULTS_DIR) or not os.path.exists(MODELS_DIR):
    st.error("""
    ⚠️ Pre-trained models not found!
    
    Please run the training script locally first:
    ```
    python train_models.py
    ```
    """)
    st.stop()

# Load pre-computed results
@st.cache_data
def load_results():
    """Load all pre-computed results"""
    metrics_df = pd.read_csv(f"{RESULTS_DIR}/metrics_comparison.csv")
    
    with open(f"{RESULTS_DIR}/confusion_matrices.pkl", "rb") as f:
        confusion_matrices = pickle.load(f)
    
    with open(f"{RESULTS_DIR}/classification_reports.pkl", "rb") as f:
        classification_reports = pickle.load(f)
    
    with open(f"{RESULTS_DIR}/predictions.pkl", "rb") as f:
        predictions = pickle.load(f)
    
    with open(f"{MODELS_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return metrics_df, confusion_matrices, classification_reports, predictions, metadata

# Load metadata
with open(f"{MODELS_DIR}/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Sidebar
with st.sidebar:
    st.header("📁 Dataset Info")
    st.markdown(f"""
    - **Dataset**: Mushroom Classification
    - **Samples**: {metadata['n_samples']}
    - **Features**: {metadata['n_features']}
    - **Target classes**: {metadata['target_classes']}
    - **Test samples**: {int(metadata['n_samples'] * 0.2)}
    """)
    
    if 'columns_with_missing' in metadata and metadata['columns_with_missing']:
        st.warning(f"Columns with missing values: {metadata['columns_with_missing']}")
    
    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("""
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbor
    - Naive Bayes
    - Random Forest
    - XGBoost
    """)

# Main content
try:
    # Load all results
    metrics_df, confusion_matrices, classification_reports, predictions, metadata = load_results()
    
    # Display metrics table
    st.header("📊 Model Performance Comparison")
    
    # Format metrics for display
    display_df = metrics_df.copy()
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    st.table(display_df.set_index('Model'))
    
    # Model selection for detailed view
    st.header("🔍 Detailed Model Analysis")
    selected_model = st.selectbox("Select model to analyze", metrics_df['Model'].tolist())
    
    if selected_model:
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrices[selected_model]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Classification Report
            st.subheader("Classification Report")
            report = classification_reports[selected_model]
            report_df = pd.DataFrame(report).transpose()
            
            # Format for display
            if 'accuracy' in report_df.index:
                report_df = report_df.drop('accuracy')
            
            st.dataframe(report_df.style.format("{:.3f}"))
        
        # Model metrics visualization
        st.subheader(f"📈 {selected_model} - Performance Metrics")
        
        # Get metrics for selected model
        model_metrics = metrics_df[metrics_df['Model'] == selected_model].iloc[0]
        
        # Create bar chart of metrics
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        values = [float(model_metrics[m]) for m in metrics_to_plot if pd.notna(model_metrics[m])]
        labels = [m for m in metrics_to_plot if pd.notna(model_metrics[m])]
        
        if model_metrics['AUC'] and pd.notna(model_metrics['AUC']):
            metrics_to_plot.insert(1, 'AUC')
            values.insert(1, float(model_metrics['AUC']))
            labels.insert(1, 'AUC')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color='skyblue')
        ax.set_ylabel('Score')
        ax.set_title(f'{selected_model} - Performance Metrics')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Model comparison visualization
    st.header("📈 Models Comparison")
    
    comparison_metric = st.selectbox(
        "Select metric to compare",
        ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    )
    
    if comparison_metric:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter out models with null values
        valid_data = metrics_df[metrics_df[comparison_metric].notna()]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5']
        bars = ax.bar(valid_data['Model'], valid_data[comparison_metric], color=colors[:len(valid_data)])
        ax.set_xlabel('Model')
        ax.set_ylabel(comparison_metric)
        ax.set_title(f'{comparison_metric} Comparison Across Models')
        ax.set_xticklabels(valid_data['Model'], rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, valid_data[comparison_metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Best model recommendation
    st.header("🏆 Model Recommendation")
    
    # Find best model based on average of all metrics
    metrics_for_ranking = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']].fillna(0)
    if 'AUC' in metrics_df.columns and metrics_df['AUC'].notna().any():
        metrics_for_ranking['AUC'] = metrics_df['AUC'].fillna(0)
    
    # Calculate average score
    avg_scores = metrics_for_ranking.mean(axis=1)
    best_idx = avg_scores.argmax()
    best_model = metrics_df.iloc[best_idx]['Model']
    best_score = avg_scores.iloc[best_idx]
    
    st.success(f"""
    ### 🥇 Recommended Model: **{best_model}**
    
    Based on average performance across all metrics, **{best_model}** achieves the highest 
    combined score of **{best_score:.3f}**.
    
    **Key strengths:**
    - Accuracy: {metrics_df.iloc[best_idx]['Accuracy']:.3f}
    - F1 Score: {metrics_df.iloc[best_idx]['F1']:.3f}
    - MCC: {metrics_df.iloc[best_idx]['MCC']:.3f}
    """)
    
    # Download results
    st.header("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download metrics as CSV
        csv = metrics_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="model_metrics.csv">📊 Download Metrics CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        # Download full report
        if st.button("Generate Full Report"):
            report_text = "# Mushroom Classification - Model Performance Report\n\n"
            report_text += f"## Dataset Info\n- Samples: {metadata['n_samples']}\n"
            report_text += f"- Features: {metadata['n_features']}\n"
            report_text += f"- Target classes: {metadata['target_classes']}\n\n"
            report_text += "## Performance Metrics\n"
            report_text += metrics_df.to_string()
            
            b64 = base64.b64encode(report_text.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="full_report.txt">📄 Download Full Report</a>'
            st.markdown(href, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading results: {str(e)}")
    st.info("Please ensure you've run train_models.py successfully first.")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This demo uses pre-trained models on the Mushroom dataset.
To retrain models, run `train_models.py` locally.
""")