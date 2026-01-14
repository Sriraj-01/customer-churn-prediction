import sys
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
sys.path.append(PROJECT_ROOT)

from src.predict import predict_from_dict
from flask import Flask, request, render_template, jsonify

APP = Flask(__name__, template_folder='templates', static_folder='static')
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")

def load_model():
    obj = joblib.load(MODEL_PATH)
    return obj['preprocessor'], obj['model']

PREPROCESSOR, MODEL = load_model()

@APP.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@APP.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    # Get correct feature list from preprocessor
    for k, v in data.items():
        try:
            data[k] = float(v)
        except:
            pass

    result = predict_from_dict(data)
    
    pred = "Yes" if result['prediction'] == 1 else "No"
    prob = result["probability"]

    return render_template("index.html", prediction = pred, probability = round(prob,3), percent=round(prob*100,1))


@APP.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json()
    result = predict_from_dict(payload)
    return jsonify(result)

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=True)
