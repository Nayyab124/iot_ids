
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
MODEL = ROOT / "models" / "ciciot_xgboost_binary.json"
PREP = ROOT / "models" / "ciciot_preprocessing.joblib"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
import yaml
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
THRESHOLD = float(config.get("detection", {}).get("confidence_threshold", 0.80))

model = xgb.XGBClassifier()
model.load_model(str(MODEL))

prep = joblib.load(PREP)
columns = prep["feature_columns"]
medians = pd.Series(prep["medians"])

# These are candidate evaluation files from attack categories.
# They are only genuinely unseen if their capture files were not
# included in training. Verify that before interpreting the scores.
candidates = [
    ("BrowserHijacking", "BrowserHijacking.pcap.csv", 1),
    ("CommandInjection", "CommandInjection.pcap.csv", 1),
    ("Backdoor_Malware", "Backdoor_Malware.pcap.csv", 1),
]

print(f"Decision threshold: {THRESHOLD:.2f}")
results = []

print(f"Decision threshold: {THRESHOLD:.2f}")

for folder, filename, label in candidates:
    path = DATA / folder / filename

    if not path.exists():
        print("Missing:", path)
        continue

    df = pd.read_csv(path, nrows=5000, low_memory=False)
    if df.empty:
        print("Empty:", path)
        continue

    X = df.reindex(columns=columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians).fillna(0)

    probabilities = model.predict_proba(X)[:, 1]
    pred = (probabilities >= THRESHOLD).astype(int)

    detected = int((pred == 1).sum())
    total = len(pred)
    rate = detected / total

    results.append((filename, total, detected))

    print(f"\\n{filename}")
    print(f"Rows evaluated: {total}")
    print(f"Predicted as attack: {detected}")
    print(f"Predicted as benign: {total - detected}")
    print(f"Attack detection rate: {rate:.2%}")

if results:
    total_rows = sum(rows for _, rows, _ in results)
    total_detected = sum(detected for _, _, detected in results)
    overall_rate = total_detected / total_rows

    print("\\nATTACK-CATEGORY DETECTION SUMMARY")
    print(f"Total attack rows evaluated: {total_rows}")
    print(f"Rows detected as attack: {total_detected}")
    print(f"Rows predicted as benign: {total_rows - total_detected}")
    print(f"Overall attack detection rate: {overall_rate:.2%}")
    print(
        "Caution: these categories may overlap with training data. "
        "Do not call them independently unseen without verifying provenance."
    )
else:
    print("No evaluation files were loaded.")
