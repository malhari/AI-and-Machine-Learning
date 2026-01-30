# -----------------------------------------------------------
#  Capstone Session 3 - Data Analysis with Pandas
#  Works for NSMES1988.csv or NSMES1988updated.csv
# -----------------------------------------------------------

import pandas as pd
import numpy as np

# -----------------------------------------------------------
# 1. Import CSV file
# -----------------------------------------------------------

file_path = "NSMES1988.csv"   # change if needed
df = pd.read_csv(file_path)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

# -----------------------------------------------------------
# 2. Identify different data types
# -----------------------------------------------------------

print("\n=== DATA TYPES ===")
print(df.dtypes)

# Separate numeric and categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# -----------------------------------------------------------
# 3. Convert specific columns to categorical
# (as per dataset description on page 5–6)
# -----------------------------------------------------------

categorical_expected = [
    'health', 'adl', 'region', 'married', 'gender',
    'insurance', 'employed', 'medicaid'
]

for col in categorical_expected:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("\n=== UPDATED CATEGORICAL COLUMNS ===")
print(df.select_dtypes(include=['category']).columns.tolist())

# -----------------------------------------------------------
# 4. Perform detailed pivoting analysis
# Example: Average visits by Health & Region
# -----------------------------------------------------------

if "visits" in df.columns:
    pivot_health_region = df.pivot_table(
        values="visits",
        index="health",
        columns="region",
        aggfunc="mean"
    )
    print("\n=== PIVOT: Average Visits by Health & Region ===")
    print(pivot_health_region)

# -----------------------------------------------------------
# 5. AGE & GENDER DISTRIBUTION
# -----------------------------------------------------------

if "age" in df.columns and "gender" in df.columns:
    age_gender_dist = df.groupby(["age", "gender"]).size().unstack(fill_value=0)
    print("\n=== AGE & GENDER DISTRIBUTION ===")
    print(age_gender_dist)

# -----------------------------------------------------------
# 6. HEALTH STATUS BY GENDER
# -----------------------------------------------------------

if "health" in df.columns and "gender" in df.columns:
    health_gender_dist = df.groupby(["health", "gender"]).size().unstack(fill_value=0)
    print("\n=== HEALTH STATUS BY GENDER ===")
    print(health_gender_dist)

# -----------------------------------------------------------
# 7. INCOME DISTRIBUTION BY GENDER
# -----------------------------------------------------------

if "income" in df.columns and "gender" in df.columns:
    income_gender_dist = df.groupby("gender")["income"].describe()
    print("\n=== INCOME DISTRIBUTION BY GENDER ===")
    print(income_gender_dist)

# -----------------------------------------------------------
# 8. REGIONAL INCOME DISTRIBUTION
# -----------------------------------------------------------

if "income" in df.columns and "region" in df.columns:
    regional_income_dist = df.groupby("region")["income"].describe()
    print("\n=== REGIONAL INCOME DISTRIBUTION ===")
    print(regional_income_dist)

# -----------------------------------------------------------
# 9. AGE-WISE INCOME ANALYSIS
# -----------------------------------------------------------

if "income" in df.columns and "age" in df.columns:
    age_income_dist = df.groupby("age")["income"].mean()
    print("\n=== AGE-WISE INCOME ANALYSIS (Mean Income by Age) ===")
    print(age_income_dist)

# -----------------------------------------------------------
# 10. CLEANED SUMMARY REPORT
# -----------------------------------------------------------

print("\n=== FINAL SUMMARY REPORT ===")
print("Rows:", len(df))
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
print("\nAnalysis Completed Successfully!")
