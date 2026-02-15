# =========================================================
# APPLIED DATA SCIENCE
# Assignment 2 – Normalization & Standardization
# Dataset: Hotel Booking Demand (500 Records)
# =========================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("hotel_500_records.csv")

print("Dataset Loaded Successfully")
print("=" * 80)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

print("Column Names:", df.columns.tolist())
print("=" * 80)

# ---------------------------------------------------------
# 4. Basic Dataset Information
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())
print("=" * 80)

print("Dataset Shape:", df.shape)
print("=" * 80)

print("Missing Values:")
print(df.isnull().sum())
print("=" * 80)

# ---------------------------------------------------------
# 5. Remove Duplicates
# ---------------------------------------------------------
df.drop_duplicates(inplace=True)
print("Duplicates Removed")
print("=" * 80)

# ---------------------------------------------------------
# 6. Handle Missing Values
# ---------------------------------------------------------
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled")
print("=" * 80)

# ---------------------------------------------------------
# 7. Select Numerical Columns for Scaling
# ---------------------------------------------------------
numeric_data = df.select_dtypes(include=np.number)

# ---------------------------------------------------------
# 8. Normalization (Min-Max Scaling)
# ---------------------------------------------------------
minmax = MinMaxScaler()
normalized = minmax.fit_transform(numeric_data)

normalized_df = pd.DataFrame(normalized, columns=numeric_data.columns)

print("Normalized Data (First 5 Rows):")
print(normalized_df.head())
print("=" * 80)

# ---------------------------------------------------------
# 9. Standardization (Z-Score Scaling)
# ---------------------------------------------------------
standard = StandardScaler()
standardized = standard.fit_transform(numeric_data)

standardized_df = pd.DataFrame(standardized, columns=numeric_data.columns)

print("Standardized Data (First 5 Rows):")
print(standardized_df.head())

print("Data Handling Completed Successfully")
