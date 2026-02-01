# ---------------------------------------------------------------
# Capstone Session 6
# Credit Card Customer Classification
# Dataset: CC GENERAL.csv
#
# Requirements:
#  - Load dataset
#  - Exploratory analysis (EDA)
#  - Handle missing values
#  - Feature engineering
#  - Train/Test split
#  - Standard scaling
#  - Classification Models: Logistic Regression, Decision Tree, Random Forest
#  - Compare performance
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Set style
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

df = pd.read_csv("CC GENERAL.csv")

print("\n" + "="*60)
print("DATA LOADING")
print("="*60)

print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- SHAPE ---")
print(f"Shape: {df.shape}")

print("\n--- NA CHECK ---")
print(df.isna().sum())

print("\n--- MISSING VALUES PERCENTAGE ---")
missing_pct = (df.isna().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])

# ---------------------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------------------

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Store customer ID for reference (but drop for modeling)
cust_id = df["CUST_ID"].copy()

# Drop CUST_ID as it's not a feature
df_clean = df.drop(columns=["CUST_ID"])

# Handle missing values
print(f"\nRows before handling missing values: {len(df_clean)}")

# Fill missing values with median for numeric columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df_clean[col].isna().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"Filled {col} missing values with median: {median_val:.2f}")

df_clean = df_clean.dropna()  # Drop any remaining rows with missing values
print(f"Rows after handling missing values: {len(df_clean)}")

# ---------------------------------------------------------------
# CREATE TARGET VARIABLE
# ---------------------------------------------------------------

print("\n" + "="*60)
print("TARGET VARIABLE CREATION")
print("="*60)

# Create a binary target: "High Value Customer" (1) vs "Low Value Customer" (0)
# High Value = High purchases AND good payment behavior
# Using median as threshold for high purchases
purchase_median = df_clean["PURCHASES"].median()
prc_full_payment_median = df_clean["PRC_FULL_PAYMENT"].median()

print(f"Median PURCHASES: {purchase_median:.2f}")
print(f"Median PRC_FULL_PAYMENT: {prc_full_payment_median:.4f}")

# High Value Customer: Above median purchases OR above median full payment rate
df_clean["HIGH_VALUE_CUSTOMER"] = (
    (df_clean["PURCHASES"] > purchase_median) | 
    (df_clean["PRC_FULL_PAYMENT"] > prc_full_payment_median)
).astype(int)

target = "HIGH_VALUE_CUSTOMER"

# Check target distribution
print(f"\n--- TARGET DISTRIBUTION ---")
print(df_clean[target].value_counts())
print(f"\nTarget distribution (%):")
print(df_clean[target].value_counts(normalize=True) * 100)

# ---------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Distribution of target
plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x=target)
plt.title("Distribution of High Value Customers")
plt.xlabel("High Value Customer (1=Yes, 0=No)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()

# Balance distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["BALANCE"], kde=True, bins=50)
plt.title("Balance Distribution")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("balance_distribution.png")
plt.close()

# Purchases distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["PURCHASES"], kde=True, bins=50)
plt.title("Purchases Distribution")
plt.xlabel("Purchases")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("purchases_distribution.png")
plt.close()

# Credit Limit distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["CREDIT_LIMIT"], kde=True, bins=50)
plt.title("Credit Limit Distribution")
plt.xlabel("Credit Limit")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("credit_limit_distribution.png")
plt.close()

# Balance vs Purchases by target
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="BALANCE", y="PURCHASES", hue=target, alpha=0.6)
plt.title("Balance vs Purchases by Customer Type")
plt.xlabel("Balance")
plt.ylabel("Purchases")
plt.legend(title="High Value Customer")
plt.tight_layout()
plt.savefig("balance_vs_purchases.png")
plt.close()

# Boxplot: Credit Limit by target
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x=target, y="CREDIT_LIMIT")
plt.title("Credit Limit by Customer Type")
plt.xlabel("High Value Customer (1=Yes, 0=No)")
plt.ylabel("Credit Limit")
plt.tight_layout()
plt.savefig("credit_limit_by_target.png")
plt.close()

# Boxplot: Purchases by target
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x=target, y="PURCHASES")
plt.title("Purchases by Customer Type")
plt.xlabel("High Value Customer (1=Yes, 0=No)")
plt.ylabel("Purchases")
plt.tight_layout()
plt.savefig("purchases_by_target.png")
plt.close()

# Boxplot: Payment Full Percentage by target
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x=target, y="PRC_FULL_PAYMENT")
plt.title("Full Payment Percentage by Customer Type")
plt.xlabel("High Value Customer (1=Yes, 0=No)")
plt.ylabel("Full Payment Percentage")
plt.tight_layout()
plt.savefig("payment_by_target.png")
plt.close()

# Correlation heatmap
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)

if len(numeric_cols) > 1:
    plt.figure(figsize=(14, 10))
    corr_matrix = df_clean[numeric_cols + [target]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

# ---------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------

print("\n" + "="*60)
print("FEATURE PREPARATION")
print("="*60)

# Select features (exclude target)
X = df_clean.drop(columns=[target])
y = df_clean[target]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {X.columns.tolist()}")

# ---------------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n--- TRAIN/TEST SPLIT ---")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# ---------------------------------------------------------------
# SCALING
# ---------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nFeatures scaled using StandardScaler")

# ---------------------------------------------------------------
# MODELING
# ---------------------------------------------------------------

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled features for Logistic Regression, original for tree-based models
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "model": model
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

# ---------------------------------------------------------------
# RESULTS SUMMARY
# ---------------------------------------------------------------

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame({
    name: {
        "Accuracy": r["Accuracy"],
        "Precision": r["Precision"],
        "Recall": r["Recall"],
        "F1-Score": r["F1-Score"],
        "ROC-AUC": r["ROC-AUC"]
    }
    for name, r in results.items()
}).T

print(results_df.round(4))

# Find best model by F1-score
best_model = max(results.items(), key=lambda x: x[1]["F1-Score"])
print(f"\nBest Model (by F1-Score): {best_model[0]}")
print(f"  Accuracy: {best_model[1]['Accuracy']:.4f}")
print(f"  Precision: {best_model[1]['Precision']:.4f}")
print(f"  Recall: {best_model[1]['Recall']:.4f}")
print(f"  F1-Score: {best_model[1]['F1-Score']:.4f}")
print(f"  ROC-AUC: {best_model[1]['ROC-AUC']:.4f}")

# ---------------------------------------------------------------
# DETAILED CLASSIFICATION REPORTS
# ---------------------------------------------------------------

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

for name, r in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, r["y_pred"], 
                                target_names=["Low Value", "High Value"]))

# ---------------------------------------------------------------
# CONFUSION MATRICES
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"{name}\nConfusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
    axes[idx].set_xticklabels(["Low Value", "High Value"])
    axes[idx].set_yticklabels(["Low Value", "High Value"])

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.close()

# ---------------------------------------------------------------
# ROC CURVES
# ---------------------------------------------------------------

plt.figure(figsize=(8, 6))

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_pred_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['ROC-AUC']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()

# ---------------------------------------------------------------
# FEATURE IMPORTANCE (for tree-based models)
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Decision Tree Feature Importance
if "Decision Tree" in results:
    dt_model = results["Decision Tree"]["model"]
    feature_importance = pd.Series(
        dt_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(15)
    
    axes[0].barh(range(len(feature_importance)), feature_importance.values)
    axes[0].set_yticks(range(len(feature_importance)))
    axes[0].set_yticklabels(feature_importance.index)
    axes[0].set_xlabel("Importance")
    axes[0].set_title("Decision Tree - Top 15 Feature Importance")
    axes[0].invert_yaxis()

# Random Forest Feature Importance
if "Random Forest" in results:
    rf_model = results["Random Forest"]["model"]
    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(15)
    
    axes[1].barh(range(len(feature_importance)), feature_importance.values)
    axes[1].set_yticks(range(len(feature_importance)))
    axes[1].set_yticklabels(feature_importance.index)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Random Forest - Top 15 Feature Importance")
    axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ---------------------------------------------------------------
# SUMMARY STATISTICS BY CUSTOMER TYPE
# ---------------------------------------------------------------

print("\n" + "="*60)
print("SUMMARY STATISTICS BY CUSTOMER TYPE")
print("="*60)

summary_stats = df_clean.groupby(target).agg({
    "BALANCE": ["mean", "median"],
    "PURCHASES": ["mean", "median"],
    "CREDIT_LIMIT": ["mean", "median"],
    "PAYMENTS": ["mean", "median"],
    "PRC_FULL_PAYMENT": ["mean", "median"]
}).round(2)

print(summary_stats)

print("\n" + "="*60)
print("SCRIPT COMPLETED")
print("="*60)
print("\nGenerated visualizations:")
print("  - target_distribution.png")
print("  - balance_distribution.png")
print("  - purchases_distribution.png")
print("  - credit_limit_distribution.png")
print("  - balance_vs_purchases.png")
print("  - credit_limit_by_target.png")
print("  - purchases_by_target.png")
print("  - payment_by_target.png")
print("  - correlation_heatmap.png")
print("  - confusion_matrices.png")
print("  - roc_curves.png")
print("  - feature_importance.png")