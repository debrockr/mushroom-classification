# debug_dataset.py
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('data/mushrooms.csv')  # Change to your filename

print("=" * 50)
print("DATASET DEBUG INFORMATION")
print("=" * 50)

# Basic info
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Instances: {df.shape[0]}")

# Check for missing values
print(f"\n🔍 Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Data types
print(f"\n📋 Data Types:")
print(df.dtypes.value_counts())

# Target variable (last column)
target_col = df.columns[-1]
print(f"\n🎯 Target Column: '{target_col}'")
print(f"Target unique values: {df[target_col].unique()}")
print(f"Target value counts:")
print(df[target_col].value_counts())

# Check first few rows
print(f"\n👀 First 5 rows:")
print(df.head())

# Check for any issues
print(f"\n⚠️ Potential Issues:")
issues = []

# Check if target is numeric
if df[target_col].dtype == 'object':
    issues.append("❌ Target column is not numeric - needs encoding")
else:
    issues.append("✅ Target column is numeric")

# Check class balance
class_counts = df[target_col].value_counts()
if len(class_counts) > 2:
    issues.append(f"ℹ️ Multi-class classification detected ({len(class_counts)} classes)")
if class_counts.min() / class_counts.max() < 0.2:
    issues.append("⚠️ Class imbalance detected")

# Check feature types
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    issues.append(f"⚠️ {len(categorical_cols)} categorical columns need encoding")

# Print issues
for issue in issues:
    print(issue)

print("\n" + "=" * 50)