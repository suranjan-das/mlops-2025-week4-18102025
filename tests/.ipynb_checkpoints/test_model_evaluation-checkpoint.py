# tests/test_model_evaluation.py
import os
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

DATA_PATH = os.environ.get("IRIS_CSV_PATH", "data/iris.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.9"))

def test_model_exists():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found"

def test_model_performance():
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = load(MODEL_PATH)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= MIN_ACCURACY, f"Model accuracy {acc:.4f} < {MIN_ACCURACY}"
