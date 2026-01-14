"""
feature_engineering.py
Create or transform features for modeling.
"""
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple features (example)."""
    df = df.copy()
    # Example: flag high monthly charges
    if 'MonthlyCharges' in df.columns:
        df['high_monthly'] = (df['MonthlyCharges'] > 70).astype(int)
    # Example: tenure groups
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 100], labels=['0-12','13-24','25-48','48+'])
    return df

if __name__ == "__main__":
    import os
    import pandas as pd
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR,"..","data","raw","Telco_Customer_Churn.csv")
    df = pd.read_csv(data_path)
    df = add_features(df)
    print(df.columns)
