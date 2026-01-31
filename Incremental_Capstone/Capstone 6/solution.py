# ---------------------------------------------------------------
# Capstone Session 6
# Adult Census Income Classification
#
# Requirements:
#  - Load dataset
#  - Exploratory analysis (EDA)
#  - Handle missing values
#  - Encodings
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

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

df = pd.read_csv("adultcensusincome.csv")

print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- SHAPE ---")
print(f"Shape: {df.shape}")

print("\n--- NA CHECK ---")
print(df.isna().sum())

print("\n--- MISSING VALUES (?) ---")
# Check for "?" which represents missing values in this dataset
for col in df.columns:
    missing_count = (df[col] == "?").sum()
    if missing_count > 0:
        print(f"{col}: {missing_count} missing values")

# ---------------------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------------------

# Replace "?" with NaN for proper handling
df = df.replace("?", np.nan)

# Drop rows with missing values
print(f"\nRows before dropping NA: {len(df)}")
df = df.dropna()
print(f"Rows after dropping NA: {len(df)}")

# Target variable
target = "income"

# Check target distribution
print(f"\n--- TARGET DISTRIBUTION ---")
print(df[target].value_counts())
print(f"\nTarget distribution (%):")
print(df[target].value_counts(normalize=True) * 100)

# ---------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------

# Distribution of target
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=target)
plt.title("Distribution of Income")
plt.xlabel("Income")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("income_distribution.png")
plt.close()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], kde=True, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()

# Income by education
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="education", hue=target)
plt.title("Income Distribution by Education")
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Income")
plt.tight_layout()
plt.savefig("income_by_education.png")
plt.close()

# Income by sex
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sex", hue=target)
plt.title("Income Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Income")
plt.tight_layout()
plt.savefig("income_by_sex.png")
plt.close()

# Income by workclass
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="workclass", hue=target)
plt.title("Income Distribution by Workclass")
plt.xlabel("Workclass")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Income")
plt.tight_layout()
plt.savefig("income_by_workclass.png")
plt.close()

# Correlation heatmap for numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

# Boxplot: Hours per week by income
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x=target, y="hours.per.week")
plt.title("Hours per Week by Income")
plt.xlabel("Income")
plt.ylabel("Hours per Week")
plt.tight_layout()
plt.savefig("hours_by_income.png")
plt.close()

# ---------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------

# Create a copy for encoding
df_encoded = df.copy()

# Encode target variable (binary: 0 for <=50K, 1 for >50K)
df_encoded[target] = df_encoded[target].map({"<=50K": 0, ">50K": 1})

# One-hot encode categorical features
# Note: education.num is already numeric, so we can drop education if needed
# or keep both. We'll keep education.num and drop education to avoid redundancy
categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
if target in categorical_cols:
    categorical_cols.remove(target)

print(f"\n--- CATEGORICAL COLUMNS TO ENCODE ---")
print(categorical_cols)

# One-hot encoding with drop_first=True to avoid multicollinearity
df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

print(f"\n--- SHAPE AFTER ENCODING ---")
print(f"Shape: {df_encoded.shape}")

# ---------------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------------

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

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

# ---------------------------------------------------------------
# MODELING
# ---------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

results = {}

print("\n--- TRAINING MODELS ---")
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
        "y_pred_proba": y_pred_proba
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

# Find best model by F1-score (good balance of precision and recall)
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
                                target_names=["<=50K", ">50K"]))

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
    axes[idx].set_xticklabels(["<=50K", ">50K"])
    axes[idx].set_yticklabels(["<=50K", ">50K"])

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
    dt_model = models["Decision Tree"]
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
    rf_model = models["Random Forest"]
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

print("\n" + "="*60)
print("SCRIPT COMPLETED")
print("="*60)
print("\nGenerated visualizations:")
print("  - income_distribution.png")
print("  - age_distribution.png")
print("  - income_by_education.png")
print("  - income_by_sex.png")
print("  - income_by_workclass.png")
print("  - correlation_heatmap.png")
print("  - hours_by_income.png")
print("  - confusion_matrices.png")
print("  - roc_curves.png")
print("  - feature_importance.png")