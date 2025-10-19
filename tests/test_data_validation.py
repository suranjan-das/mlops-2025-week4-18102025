# tests/test_data_validation.py
import pandas as pd
import os

DATA_PATH = os.environ.get("IRIS_CSV_PATH", "data/iris.csv")

def test_data_exists():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found"

def test_columns_and_types():
    df = pd.read_csv(DATA_PATH)
    # Standard iris columns: sepal length, sepal width, petal length, petal width, species
    assert df.shape[1] >= 5
    # no nulls
    assert not df.isnull().values.any(), "Data contains nulls"
    # last column is target
    assert df.columns[-1].lower() in ("species", "target", "label"), f"Unexpected target column: {df.columns[-1]}"
