"""
predict.py
Utility to load a saved model and run one-off predictions (dict -> label/prob).
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict
import os
try:
    from src.data_preprocessing import basic_cleaning
    from src.feature_engineering import add_features
except ImportError:
    from data_preprocessing import basic_cleaning
    from feature_engineering import add_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")

def load_artifacts(path=MODEL_PATH):
    obj = joblib.load(path)
    return obj['preprocessor'], obj['model'], obj.get('num_cols', None), obj.get('cat_cols', None)

def predict_from_dict(input_dict: Dict):
    preprocessor, model, num_cols, cat_cols = load_artifacts()
    # Convert to DataFrame (single row)
    
    df = pd.DataFrame([input_dict])
    df = basic_cleaning(df)
    df = add_features(df)
    X_t = preprocessor.transform(df)
    prob = model.predict_proba(X_t)[0,1]
    pred = int(prob >= 0.5)
    return {'prediction': pred, 'probability': float(prob)}

if __name__ == "__main__":
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.65
    }

    print(predict_from_dict(sample))

