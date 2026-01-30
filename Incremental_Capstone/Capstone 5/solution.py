# ---------------------------------------------------------------
# Capstone Session 5
# Florida Bike Rentals Analysis + ML Regression Models
#
# Requirements from PDF:
#  - Load dataset
#  - Exploratory analysis (EDA)
#  - Date features
#  - Encodings
#  - Train/Test split
#  - Standard scaling
#  - Models: Linear, Lasso, Ridge
#  - Compare performance
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

df = pd.read_csv("FloridaBikeRentals.csv")

print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- NA CHECK ---")
print(df.isna().sum())

# There should be no missing but handling if present
df = df.dropna()

# ---------------------------------------------------------------
# FEATURE ENGINEERING (From PDF instructions)
# ---------------------------------------------------------------
# Convert Date
df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.dayofweek
df["Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

# Target variable (as per dataset)
target = "Rented Bike Count"

# ---------------------------------------------------------------
# EXPLORATORY ANALYSIS (Charts optional for script)
# ---------------------------------------------------------------

# Distribution of target
plt.figure(figsize=(6,4))
sns.histplot(df[target], kde=True)
plt.title("Distribution of Rented Bike Count")
plt.savefig("dist_rented_bike_count.png")

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")

# Boxplot categorical vs target (examples in PDF)
plt.figure(figsize=(6,4))
sns.boxplot(x="Weekend", y=target, data=df)
plt.title("Weekend vs Rented Bikes")
plt.savefig("weekend_boxplot.png")

plt.figure(figsize=(6,4))
sns.boxplot(x="Holiday", y=target, data=df)
plt.title("Holiday vs Rented Bikes")
plt.savefig("holiday_boxplot.png")

# ---------------------------------------------------------------
# ENCODING CATEGORICAL FEATURES
# ---------------------------------------------------------------

df_encoded = pd.get_dummies(df, drop_first=True)

# ---------------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------------

columns_to_drop = [target]
if "Date" in df_encoded.columns:
    columns_to_drop.append("Date")

X = df_encoded.drop(columns=columns_to_drop)
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# ---------------------------------------------------------------
# SCALING
# ---------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# MODELING
# ---------------------------------------------------------------

models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "Ridge": Ridge(alpha=1.0)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {"R2": r2, "RMSE": rmse}

# ---------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------

print("\n--- MODEL RESULTS ---")
for m, r in results.items():
    print(f"{m}: R2={r['R2']:.4f}, RMSE={r['RMSE']:.2f}")

best = max(results.items(), key=lambda x: x[1]["R2"])
print(f"\nBest Model: {best[0]} (by R2)\n")

print("\n--- SCRIPT COMPLETED ---\n")
