from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
MODEL_PATH = ROOT / "models" / "ciciot_xgboost_binary.json"
PREP_PATH = ROOT / "models" / "ciciot_preprocessing.joblib"
REPORT_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
import yaml
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
THRESHOLD = float(config.get("detection", {}).get("confidence_threshold", 0.80))
LIMIT_PER_CAPTURE = 20000

captures = [
    ("BenignTraffic1.pcap.csv", DATA / "Benign_Final" / "BenignTraffic1.pcap.csv", 0),
    ("BenignTraffic2.pcap.csv", DATA / "Benign_Final" / "BenignTraffic2.pcap.csv", 0),
    ("BenignTraffic3.pcap.csv", DATA / "Benign_Final" / "BenignTraffic3.pcap.csv", 0),
    ("DDoS-HTTP_Flood-.pcap.csv", DATA / "DDoS-HTTP_Flood" / "DDoS-HTTP_Flood-.pcap.csv", 1),
]

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))
prep = joblib.load(PREP_PATH)
columns = prep["feature_columns"]
medians = pd.Series(prep["medians"])

all_true = []
all_pred = []
details = []

for name, path, label in captures:
    if not path.is_file():
        print(f"SKIP missing capture: {path}")
        continue

    df = pd.read_csv(path, nrows=LIMIT_PER_CAPTURE, low_memory=False)
    if df.empty:
        print(f"SKIP empty capture: {name}")
        continue

    X = df.reindex(columns=columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)

    probabilities = model.predict_proba(X)[:, 1]
    pred = (probabilities >= THRESHOLD).astype(int)
    truth = np.full(len(pred), label, dtype=int)

    all_true.extend(truth.tolist())
    all_pred.extend(pred.tolist())

    attack_count = int((pred == 1).sum())
    benign_count = int((pred == 0).sum())
    detected_rate = attack_count / len(pred) if label == 1 else None
    false_positive_rate = attack_count / len(pred) if label == 0 else None

    details.append(
        f"{name}: rows={len(pred)}, actual={'Attack' if label else 'Benign'}, "
        f"predicted_attack={attack_count}, predicted_benign={benign_count}, "
        f"attack_detection_rate={detected_rate if detected_rate is not None else 'N/A'}, "
        f"benign_false_positive_rate={false_positive_rate if false_positive_rate is not None else 'N/A'}"
    )

if not all_true or len(set(all_true)) < 2:
    raise RuntimeError("Evaluation needs at least one valid benign and one attack capture.")

report = classification_report(
    all_true, all_pred,
    labels=[0, 1],
    target_names=["Benign", "Attack"],
    zero_division=0,
)
matrix = confusion_matrix(all_true, all_pred, labels=[0, 1])
accuracy = accuracy_score(all_true, all_pred)

output = [
    "CAPTURE-LEVEL EVALUATION (candidate independent captures)",
    f"Rows evaluated per capture: up to {LIMIT_PER_CAPTURE}",
    f"Decision threshold: {THRESHOLD:.2f}",
    "Note: verify capture provenance and duplicate traffic before claiming independence.",
    "",
    *details,
    "",
    f"Accuracy: {accuracy:.4f}",
    "",
    "Classification report:",
    report,
    "Confusion matrix (rows=true, columns=predicted):",
    str(matrix),
]

text = "\n".join(output)
print("\n" + text)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
out_path = REPORT_DIR / "capture_level_evaluation.txt"
out_path.write_text(text, encoding="utf-8")
print("\nSaved report:", out_path)
