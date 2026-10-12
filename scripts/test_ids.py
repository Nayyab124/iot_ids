from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ciciot_xgboost_binary.json"
PREPROCESS_PATH = ROOT / "models" / "ciciot_preprocessing.joblib"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
DATA_PATH = (
    ROOT / "data" / "raw" / "CSV" / "Benign_Final"
    / "BenignTraffic.pcap.csv"
)

for path in (MODEL_PATH, PREPROCESS_PATH, CONFIG_PATH, DATA_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

threshold = float(
    config.get("detection", {}).get("confidence_threshold", 0.80)
)
if not 0.0 <= threshold <= 1.0:
    raise ValueError("Detection confidence threshold must be between 0 and 1.")

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

prep = joblib.load(PREPROCESS_PATH)
feature_columns = prep["feature_columns"]
medians = pd.Series(prep["medians"]).reindex(feature_columns)

df = pd.read_csv(DATA_PATH, nrows=10, low_memory=False)
X = df.reindex(columns=feature_columns)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(medians).fillna(0)

probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("Saved model loaded successfully.")
print(f"Rows evaluated: {len(X)}")
print(f"Configured alert threshold: {threshold:.2f}")
print("\nPredictions:")

for i, (pred, prob) in enumerate(
    zip(predictions, probabilities), start=1
):
    label = "ATTACK" if pred == 1 else "BENIGN"
    print(f"Row {i}: {label} (attack probability: {prob:.4f})")

print("\nSmoke test completed.")
print("Firewall blocking remains disabled.")
