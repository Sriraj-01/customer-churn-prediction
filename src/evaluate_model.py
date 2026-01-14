"""
evaluate_model.py
Load model and run evaluation on a hold-out test set or new CSV.
"""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
from data_preprocessing import basic_cleaning
from feature_engineering import add_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Telco_Customer_Churn.csv")

def load_model(path=MODEL_PATH):
    obj = joblib.load(path)
    return obj['preprocessor'], obj['model']

def evaluate(csv_path: str):
    preprocessor, model = load_model()
    df = pd.read_csv(csv_path)
    df = basic_cleaning(df)
    df = add_features(df)
    if 'Churn' not in df.columns:
        raise ValueError("CSV must contain 'Churn' column for evaluation")
    X = df.drop(columns=['Churn'])
    y = df['Churn'].apply(lambda x: 1 if str(x).lower() in ['yes','y','true','1'] else 0)
    X_t = preprocessor.transform(X)
    preds = model.predict(X_t)
    probs = model.predict_proba(X_t)[:,1]
    print("Classification report:\n", classification_report(y, preds))
    print("ROC AUC:", roc_auc_score(y, probs))
    print("Confusion matrix:\n", confusion_matrix(y, preds))

if __name__ == "__main__":
    evaluate(DATA_PATH)
