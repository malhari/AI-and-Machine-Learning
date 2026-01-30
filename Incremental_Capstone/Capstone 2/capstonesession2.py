# python
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def load_csv_auto(path: str):
    # try reading with header=None to match user's pasted data
    df = pd.read_csv(path, header=None)
    # give simple column names if none
    df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def basic_summary(df: pd.DataFrame, n=5):
    print("shape:", df.shape)
    print("\nhead:")
    print(df.head(n))
    print("\ndtypes and missing count:")
    print(df.dtypes)
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nunique value counts (sample):")
    for c in df.columns[:min(10, len(df.columns))]:
        print(c, "->", df[c].nunique())

def detect_column_types(df: pd.DataFrame, cat_threshold=30):
    # treat object dtype or low-unique as categorical
    categorical = []
    numeric = []
    for c in df.columns:
        if df[c].dtype == 'object' or df[c].nunique() <= cat_threshold:
            categorical.append(c)
        else:
            try:
                pd.to_numeric(df[c])
                numeric.append(c)
            except Exception:
                categorical.append(c)
    # remove overlap: if numeric strings were detected, adjust
    categorical = [c for c in categorical if c not in numeric]
    return numeric, categorical

def prepare_target(y_series: pd.Series):
    # convert yes/no or boolean-like to 0/1 if needed
    if y_series.dtype == object:
        vals = y_series.str.lower().map({'yes':1, 'no':0})
        if vals.notna().all():
            return vals.astype(int)
    if y_series.dtype == bool:
        return y_series.astype(int)
    # if numeric already, return as-is
    return pd.to_numeric(y_series, errors='coerce').astype(int)

def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(srategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))
    ])
    return model

def run_training(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = prepare_target(df[target_col])
    # basic type detection but exclude target from features
    numeric_cols, categorical_cols = detect_column_types(X)
    print("numeric_cols:", numeric_cols)
    print("categorical_cols:", categorical_cols)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    # quick cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print("CV accuracy:", cv_scores.mean(), cv_scores)
    # small grid search for n_estimators / max_depth
    param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 10]
    }
    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=0)
    gs.fit(X_train, y_train)
    print("best params:", gs.best_params_, "best score:", gs.best_score_)
    # evaluate on test
    y_pred = gs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("test accuracy:", acc)
    print("\nclassification report:\n", classification_report(y_test, y_pred))
    # try ROC AUC if possible (binary)
    try:
        y_prob = gs.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        print("test ROC AUC:", auc)
    except Exception:
        pass
    # save pipeline
    joblib.dump(gs.best_estimator_, "pipeline.joblib")
    print("saved pipeline to pipeline.joblib")
    return gs.best_estimator_

def main():
    parser = argparse.ArgumentParser(description="Generic capstone analysis pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV (no header assumed)")
    parser.add_argument("--target", "-t", default=None, help="Target column name or index (default: last column)")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    df = load_csv_auto(str(path))
    basic_summary(df)
    # determine default target: last column
    if args.target is None:
        target_col = df.columns[-1]
    else:
        if args.target.isdigit():
            target_col = df.columns[int(args.target)]
        else:
            target_col = args.target
            if target_col not in df.columns:
                raise ValueError(f"target column not found in dataframe columns:{target_col}")
    print("\nUsing target column:", target_col)
    run_training(df, target_col)

if __name__ == "__main__":
    main()
