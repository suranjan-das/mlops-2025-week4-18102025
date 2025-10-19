# scripts/evaluate.py
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from joblib import load

# Paths (adjust if your files are elsewhere)
DATA_PATH = os.environ.get("IRIS_CSV_PATH", "data/iris.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
REPORT_PATH = os.environ.get("REPORT_PATH", "report.md")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.9"))

def load_data(path):
    df = pd.read_csv(path)
    # assume last column is target (standard iris)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y, df

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model file not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)

    X, y, df = load_data(DATA_PATH)
    model = load(MODEL_PATH)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    cls_report = classification_report(y, preds, output_dict=False)

    md = []
    md.append("# Evaluation Report")
    md.append("")
    md.append(f"**Accuracy:** {acc:.4f}")
    md.append("")
    md.append("## Classification Report")
    md.append("")
    md.append("```\n" + cls_report + "\n```")
    md.append("")

    md.append(f"**Threshold:** {ACCURACY_THRESHOLD}")
    md.append("")
    if acc >= ACCURACY_THRESHOLD:
        md.append("✅ **Sanity check passed**")
        exit_code = 0
    else:
        md.append("❌ **Sanity check failed**")
        exit_code = 3

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(md))

    print(f"Evaluation accuracy: {acc:.4f}")
    print(f"Report written to {REPORT_PATH}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
