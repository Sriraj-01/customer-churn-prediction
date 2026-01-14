"""
train_model.py
Train a baseline model and save model+preprocessor.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from data_preprocessing import basic_cleaning, build_preprocessor
from feature_engineering import add_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Telco_Customer_Churn.csv")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

def prepare_data(df: pd.DataFrame):
    df = basic_cleaning(df)
    df = add_features(df)
    # Ensure target exists
    if 'Churn' not in df.columns:
        raise ValueError("No 'Churn' column found in dataframe.")
    X = df.drop(columns=['Churn'])
    y = df['Churn'].apply(lambda x: 1 if str(x).lower() in ['yes','y','true','1'] else 0)
    return X, y

def train(df_path: str):
    df = pd.read_csv(df_path)
    X, y = prepare_data(df)
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    # Fit preprocessor and transform
    X_trans = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42, stratify=y)
    # Baseline model
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    # Evaluate
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:,1]
    print("Best params:", grid.best_params_)
    print("Classification report:\n", classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    # Save preprocessor and model together
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'preprocessor': preprocessor, 'model': best, 'num_cols': num_cols, 'cat_cols': cat_cols}, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
    return MODEL_PATH

if __name__ == "__main__":
    train(DATA_PATH)
