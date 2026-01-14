"""
data_preprocessing.py
Functions to load and preprocess the Telco churn dataset.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    return pd.read_csv(path)

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: trim whitespace, convert TotalCharges to numeric, drop customerID."""
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # Convert TotalCharges if present
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # strip whitespace from object columns
    obj_cols = df.select_dtypes(['object']).columns
    for c in obj_cols:
        df[c] = df[c].str.strip()
    return df

def build_preprocessor(df: pd.DataFrame, num_cols=None, cat_cols=None):
    """Return a ColumnTransformer with scaler for numeric and one-hot for categorical."""
    if num_cols is None:
        num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        # remove target if present
        num_cols = [c for c in num_cols if c != 'Churn']
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c != 'Churn']
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])
    return preprocessor, num_cols, cat_cols

def fit_and_save_preprocessor(df: pd.DataFrame, path: str):
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    # Fit on df (drop target if exists)
    X = df.drop(columns=['Churn']) if 'Churn' in df.columns else df.copy()
    preprocessor.fit(X)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'preprocessor': preprocessor, 'num_cols': num_cols, 'cat_cols': cat_cols}, path)
    print(f"Preprocessor saved to {path}")
    return preprocessor, num_cols, cat_cols

def load_preprocessor(path: str):
    obj = joblib.load(path)
    return obj['preprocessor'], obj['num_cols'], obj['cat_cols']

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # quick test
    data_path = os.path.join(BASE_DIR,"..","data","raw","Telco_Customer_Churn.csv")
    model_path = os.path.join(BASE_DIR,"..","models","preprocessor.pkl")
    df = load_data(data_path)
    df = basic_cleaning(df)
    fit_and_save_preprocessor(df, model_path)