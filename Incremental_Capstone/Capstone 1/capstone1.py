# ================================================================
# Capstone Project - Session 1
# Data Import, Cleaning, Inspection & Export
# Dataset: NSMES1988.csv
# ================================================================

import pandas as pd
import numpy as np

# 1. IMPORT THE DATA -----------------------------------------------------------

file_path = "NSMES1988.csv"   # Path from user-uploaded file
df = pd.read_csv(file_path)

print("\n======= DATA IMPORTED SUCCESSFULLY =======\n")
print(df.head())


# 2. INSPECT BASIC INFORMATION -------------------------------------------------

print("\n======= DATAFRAME INFO =======")
print(df.info())

print("\n======= SHAPE OF DATA =======")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n======= SUMMARY STATISTICS =======")
print(df.describe(include='all'))


# 3. CHECK FOR MISSING VALUES --------------------------------------------------

print("\n======= MISSING VALUE ANALYSIS =======")
missing = df.isnull().sum()
print(missing)

total_missing = missing.sum()
print(f"\nTotal Missing Values in Dataset: {total_missing}")


# 4. EXAMINE AGE & INCOME COLUMNS ---------------------------------------------

print("\n======= AGE COLUMN SUMMARY =======")
print(df["age"].describe())

print("\n======= INCOME COLUMN SUMMARY =======")
print(df["income"].describe())


# 5. EXPORT TO JSON ------------------------------------------------------------

json_path = "NSMES1988.json"
df.to_json(json_path, orient='records', indent=4)

print(f"\nJSON exported successfully to: {json_path}")


# 6. MEMORY USAGE BEFORE OPTIMIZATION -----------------------------------------

print("\n======= MEMORY USAGE BEFORE OPTIMIZATION =======")
print(df.memory_usage(deep=True))

print("\nTotal Memory Used:", df.memory_usage(deep=True).sum(), "bytes")


# 7. OPTIMIZE DATA TYPES -------------------------------------------------------

df_opt = df.copy()

# Convert int columns to smaller types
int_cols = df_opt.select_dtypes(include=['int64']).columns
for col in int_cols:
    df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')

# Convert float columns to float32
float_cols = df_opt.select_dtypes(include=['float64']).columns
for col in float_cols:
    df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')

# Convert categorical columns
cat_cols = ["health", "adl", "region", "married", "gender", "insurance", "employed", "medicaid"]
for col in cat_cols:
    if col in df_opt.columns:
        df_opt[col] = df_opt[col].astype('category')

print("\n======= MEMORY USAGE AFTER OPTIMIZATION =======")
print(df_opt.memory_usage(deep=True))
print("\nTotal Memory Used After Optimization:",
      df_opt.memory_usage(deep=True).sum(), "bytes")


# 8. EXPORT OPTIMIZED DATAFRAME ------------------------------------------------

new_csv_path = "NSMES1988new.csv"
df_opt.to_csv(new_csv_path, index=False)

print(f"\nOptimized CSV exported to: {new_csv_path}")

# 9. FINAL COMMENTS ------------------------------------------------------------

print("\n======= RECOMMENDATIONS BEFORE ANALYSIS =======")
print("""
1. Convert categorical variables to 'category' dtype (done).
2. Downcast numeric variables to reduce memory usage (done).
3. Check for outliers in income and hospital visits.
4. Normalize skewed columns before modeling.
5. Create derived variables if needed (e.g., total visits = visits + nvisits).
6. Validate categorical encodings (gender, region, insurance).
7. Handle missing values via imputation or removal if significant.
""")

