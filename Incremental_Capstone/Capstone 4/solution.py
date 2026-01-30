# -----------------------------------------------------------
# Capstone Session 4 - Data Visualization with Python
# Dataset: NSMES1988updated.csv (or NSMES1988.csv)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Plotting library choice
# -------------------------------------------------------------------
# We are using:
# - matplotlib: core plotting library in Python, gives fine control
# - seaborn: built on top of matplotlib, easier high-level statistical plots
#            with nice defaults and built-in support for DataFrames.

sns.set_theme(style="whitegrid")  # nice default theme

# -------------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------------

# Change this to "NSMES1988updated.csv" if that is your file name
FILE_PATH = "NSMES1988.csv"

df = pd.read_csv(FILE_PATH)

print("=== DATA LOADED ===")
print(df.head())
print("\nShape:", df.shape)

# -------------------------------------------------------------------
# 3. Basic preprocessing / type setup (similar to Session 3 work)
# -------------------------------------------------------------------

# Convert known categorical variables
categorical_cols = [
    "health", "adl", "region", "married", "gender",
    "insurance", "employed", "medicaid"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Identify numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", df.select_dtypes(include='category').columns.tolist())

# -------------------------------------------------------------------
# 4. Visualize categorical data from Week 3:
#    Health and Region
# -------------------------------------------------------------------

# 4.1 Count of records by Health
if "health" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="health")
    plt.title("Count of Individuals by Health Status")
    plt.xlabel("Health Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 4.2 Count of records by Region
if "region" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="region")
    plt.title("Count of Individuals by Region")
    plt.xlabel("Region")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 4.3 Example: Average visits by Health and Region (from Session 3 pivot)
if all(col in df.columns for col in ["visits", "health", "region"]):
    pivot_visits = (
        df.groupby(["health", "region"])["visits"]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=pivot_visits,
        x="health",
        y="visits",
        hue="region"
    )
    plt.title("Average Physician Visits by Health & Region")
    plt.xlabel("Health Status")
    plt.ylabel("Average Number of Visits")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 5. Visualize analyses from Session 3 (Week 3/4 analysis)
#    Age & Gender distribution, Health by Gender, Income analysis,
#    Regional income, Age-wise income, etc.
# -------------------------------------------------------------------

# 5.1 Age and Gender Distribution
if "age" in df.columns and "gender" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="age", hue="gender")
    plt.title("Age and Gender Distribution")
    plt.xlabel("Age (years / 10 in dataset description)")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

# 5.2 Health Status by Gender
if "health" in df.columns and "gender" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="health", hue="gender")
    plt.title("Health Status Distribution by Gender")
    plt.xlabel("Health Status")
    plt.ylabel("Count")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

# 5.3 Income Distribution by Gender (boxplot)
if "income" in df.columns and "gender" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="gender", y="income")
    plt.title("Income Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Family Income (in $10,000 units)")
    plt.tight_layout()
    plt.show()

# 5.4 Regional Income Distribution (boxplot)
if "income" in df.columns and "region" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="region", y="income")
    plt.title("Income Distribution Across Regions")
    plt.xlabel("Region")
    plt.ylabel("Family Income (in $10,000 units)")
    plt.tight_layout()
    plt.show()

# 5.5 Age-wise Income Analysis (line plot of mean income vs age)
if "income" in df.columns and "age" in df.columns:
    age_income = (
        df.groupby("age")["income"]
          .mean()
          .reset_index()
          .sort_values("age")
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=age_income, x="age", y="income", marker="o")
    plt.title("Average Income by Age")
    plt.xlabel("Age (years / 10 in dataset description)")
    plt.ylabel("Average Income (in $10,000 units)")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 6. Correlation Analysis (Week 4 correlation plots)
# -------------------------------------------------------------------

if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()

    # Heatmap of correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        square=True
    )
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    plt.show()

    # Example pairplot for a subset of numeric columns
    subset_cols = [c for c in numeric_cols if c in ["visits", "nvisits", "hospital", "income", "age"]]
    if len(subset_cols) > 1:
        sns.pairplot(df[subset_cols].dropna())
        plt.suptitle("Pairplot of Selected Numeric Variables", y=1.02)
        plt.show()

# -------------------------------------------------------------------
# 7. Optional: Save summary of plots/observations
# -------------------------------------------------------------------

# You can add manual observations after reviewing the plots.
# Example (placeholder):
print("\nVisualization completed. Review the generated plots and write observations for your report:")
print("- How health status and region relate to number of visits.")
print("- How income varies across gender, region, and age.")
print("- Which numeric variables are strongly correlated in the heatmap.")
