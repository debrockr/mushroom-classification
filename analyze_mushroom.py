# analyze_mushroom.py
import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('data/mushrooms.csv')

print("=" * 60)
print("MUSHROOM DATASET ANALYSIS")
print("=" * 60)

# Basic info
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")

# Check for missing values
print(f"\n🔍 Missing Values Analysis:")
missing_counts = df.isin(['?']).sum()
columns_with_missing = missing_counts[missing_counts > 0]
if len(columns_with_missing) > 0:
    print(f"Columns with missing values:")
    for col, count in columns_with_missing.items():
        print(f"  - {col}: {count} missing values ({count/len(df)*100:.1f}%)")
else:
    print("No missing values found!")

# Check data types
print(f"\n📋 Data Types:")
print(df.dtypes.value_counts())

# Target variable analysis
target_col = df.columns[0]  # First column is 'class'
print(f"\n🎯 Target Column: '{target_col}'")
print(f"Target values: {df[target_col].unique()}")
print(f"Class distribution:")
print(df[target_col].value_counts())
print(f"Class balance: {df[target_col].value_counts().values[0]/df[target_col].value_counts().values[1]:.2f}")

# Check for inconsistent rows
print(f"\n⚠️ Checking for inconsistent rows:")
row_lengths = df.apply(lambda x: len(x), axis=1)
if row_lengths.nunique() > 1:
    print(f"WARNING: Inconsistent row lengths found!")
    print(f"Row length distribution:")
    print(row_lengths.value_counts())
else:
    print(f"All rows have {row_lengths.iloc[0]} columns")

# Show first few rows
print(f"\n👀 First 5 rows:")
print(df.head())

# Show column names
print(f"\n📑 Column Names:")
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

print("\n" + "=" * 60)