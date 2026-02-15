# =========================================================
# APPLIED DATA SCIENCE
# Assignment 1 – Python for Data Handling
# Dataset: Hotel Booking Demand (Kaggle)
# =========================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 2. Load Dataset (500 Records Already Selected)
# ---------------------------------------------------------
df = pd.read_csv("hotel_500_records.csv")

print("Dataset Loaded Successfully")
print("="*100)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

print("Columns after cleaning:")
print(df.columns.tolist())
print("="*100)

# ---------------------------------------------------------
# 4. Dataset Exploration
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()
print("="*100)

# ---------------------------------------------------------
# 5. Check Missing Values and Zero Values
# ---------------------------------------------------------
print("Missing Values:")
print(df.isnull().sum())

numerical_cols = df.select_dtypes(include=np.number).columns

print("\nZero Values in Numerical Columns:")
print((df[numerical_cols] == 0).sum())
print("="*100)

# ---------------------------------------------------------
# 6. Remove Duplicate Records
# ---------------------------------------------------------
print("Duplicate Records:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates Removed")
print("="*100)

# ---------------------------------------------------------
# 7. Handle Missing Values
# ---------------------------------------------------------
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled")
print("="*100)

# ---------------------------------------------------------
# 8. Feature Engineering – Total Guests
# ---------------------------------------------------------
df['total_guests'] = df['adults'] + df['children']
print("Total Guests Column Created")
print("="*100)

# ---------------------------------------------------------
# 9. Statistical Measures
# ---------------------------------------------------------
print("ADR Statistics")
print("Mean:", df['adr'].mean())
print("Median:", df['adr'].median())
print("Mode:", df['adr'].mode()[0])
print("Skewness:", df['adr'].skew())
print("="*100)

# ---------------------------------------------------------
# 10. Basic Visualization
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['adr'], kde=True)
plt.title("Distribution of ADR")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='hotel', data=df)
plt.title("Hotel Type Distribution")
plt.show()

print("Preprocessing Completed Successfully")
