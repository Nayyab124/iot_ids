from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
MODEL_PATH = ROOT / "models" / "ciciot_xgboost_binary.json"
PREP_PATH = ROOT / "models" / "ciciot_preprocessing.joblib"
REPORT_DIR = ROOT / "reports"
LIMIT_PER_CAPTURE = 20000

CAPTURES = [
    ("BenignTraffic1", DATA / "Benign_Final" / "BenignTraffic1.pcap.csv", 0),
    ("BenignTraffic2", DATA / "Benign_Final" / "BenignTraffic2.pcap.csv", 0),
    ("BenignTraffic3", DATA / "Benign_Final" / "BenignTraffic3.pcap.csv", 0),
    ("DDoS_HTTP_Flood", DATA / "DDoS-HTTP_Flood" / "DDoS-HTTP_Flood-.pcap.csv", 1),
]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))
prep = joblib.load(PREP_PATH)
columns = prep["feature_columns"]
medians = pd.Series(prep["medians"])

y_parts, p_parts, group_parts = [], [], []

for name, path, label in CAPTURES:
    if not path.is_file():
        print(f"SKIP missing file: {path}")
        continue

    df = pd.read_csv(path, nrows=LIMIT_PER_CAPTURE, low_memory=False)
    if df.empty:
        print(f"SKIP empty file: {path}")
        continue

    X = df.reindex(columns=columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)

    probabilities = model.predict_proba(X)[:, 1]
    y_parts.extend([label] * len(df))
    p_parts.extend(probabilities.tolist())
    group_parts.extend([name] * len(df))
    print(f"Loaded {name}: {len(df):,} rows")

if not y_parts or len(set(y_parts)) < 2:
    raise RuntimeError("Need both benign and attack capture data to compare thresholds.")

y = np.asarray(y_parts, dtype=int)
probs = np.asarray(p_parts, dtype=float)
groups = np.asarray(group_parts)
results = []

for threshold in THRESHOLDS:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision_attack": precision_score(y, pred, zero_division=0),
        "recall_attack": recall_score(y, pred, zero_division=0),
        "f1_attack": f1_score(y, pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })

summary = pd.DataFrame(results)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = REPORT_DIR / "threshold_analysis.csv"
txt_path = REPORT_DIR / "threshold_analysis.txt"
summary.to_csv(csv_path, index=False)

lines = [
    "IDS THRESHOLD ANALYSIS",
    "Candidate capture-level evaluation; capture independence still requires verification.",
    f"Total rows: {len(y):,}",
    f"Benign rows: {int((y == 0).sum()):,}",
    f"Attack rows: {int((y == 1).sum()):,}",
    "",
    "OVERALL RESULTS (rates shown as percentages)",
]
for row in results:
    lines.append(
        f"Threshold={row['threshold']:.2f} | "
        f"Accuracy={row['accuracy']*100:.2f}% | "
        f"Attack precision={row['precision_attack']*100:.2f}% | "
        f"Attack recall={row['recall_attack']*100:.2f}% | "
        f"F1={row['f1_attack']*100:.2f}% | "
        f"FPR={row['false_positive_rate']*100:.2f}% | "
        f"TN={row['true_negative']} FP={row['false_positive']} "
        f"FN={row['false_negative']} TP={row['true_positive']}"
    )

lines.extend(["", "PER-CAPTURE PREDICTED ATTACK RATE"])
for name in dict.fromkeys(groups):
    mask = groups == name
    actual_label = int(y[mask][0])
    lines.append(f"{name} (actual {'Attack' if actual_label else 'Benign'}, {int(mask.sum())} rows)")
    for threshold in THRESHOLDS:
        rate = float((probs[mask] >= threshold).mean() * 100)
        lines.append(f"  threshold {threshold:.2f}: predicted attack {rate:.2f}%")

txt_path.write_text("\n".join(lines), encoding="utf-8")
print("\n" + "\n".join(lines))
print(f"\nCSV saved: {csv_path}")
print(f"Text report saved: {txt_path}")
print("\nNo model or IPS settings were changed.")
