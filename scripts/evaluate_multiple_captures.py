from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
REPORTS = ROOT / "reports"
LIMIT = 5000

EXCLUDE = {
    "Benign_Final/BenignTraffic.pcap.csv",
    "Backdoor_Malware/Backdoor_Malware.pcap.csv",
    "BrowserHijacking/BrowserHijacking.pcap.csv",
    "CommandInjection/CommandInjection.pcap.csv",
    "DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation.pcap.csv",
    "Benign_Final/BenignTraffic1.pcap.csv",
    "Benign_Final/BenignTraffic2.pcap.csv",
    "Benign_Final/BenignTraffic3.pcap.csv",
    "DDoS-HTTP_Flood/DDoS-HTTP_Flood-.pcap.csv",
}

model = xgb.XGBClassifier()
model.load_model(str(ROOT / "models" / "ciciot_xgboost_binary.json"))
prep = joblib.load(ROOT / "models" / "ciciot_preprocessing.joblib")
features = prep["feature_columns"]
medians = pd.Series(prep["medians"])

results = []
files = sorted(DATA.rglob("*.csv"))

for path in files:
    rel = path.relative_to(DATA).as_posix()
    if rel in EXCLUDE:
        continue

    # Folder-level labels: Benign_Final is benign; other category folders are attacks.
    category = path.relative_to(DATA).parts[0]
    actual = 0 if category == "Benign_Final" else 1

    try:
        df = pd.read_csv(path, nrows=LIMIT, low_memory=False)
        if df.empty:
            continue

        X = df.reindex(columns=features)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)

        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= 0.80).astype(int)

        tp = int(((pred == 1) & (actual == 1)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())

        results.append({
            "capture": rel,
            "category": category,
            "actual_class": "Benign" if actual == 0 else "Attack",
            "rows": len(df),
            "threshold": 0.80,
            "accuracy": accuracy_score(np.full(len(df), actual), pred),
            "attack_precision": precision_score(
                np.full(len(df), actual), pred, zero_division=0
            ),
            "attack_recall": recall_score(
                np.full(len(df), actual), pred, zero_division=0
            ),
            "attack_f1": f1_score(
                np.full(len(df), actual), pred, zero_division=0
            ),
            "false_positive_rate": fp / len(df) if actual == 0 else np.nan,
            "predicted_attack": int(pred.sum()),
            "predicted_benign": int((pred == 0).sum()),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })
        print(f"Evaluated {rel}: {len(df):,} rows")
    except Exception as exc:
        print(f"SKIPPED {rel}: {type(exc).__name__}: {exc}")

if not results:
    raise RuntimeError("No eligible captures were successfully evaluated.")

out = pd.DataFrame(results)
REPORTS.mkdir(parents=True, exist_ok=True)
csv_path = REPORTS / "multi_capture_evaluation.csv"
out.to_csv(csv_path, index=False)

print("\nPER-CAPTURE RESULTS (threshold 0.80)")
print(out[[
    "capture", "actual_class", "rows", "accuracy",
    "attack_recall", "false_positive_rate",
    "predicted_attack", "predicted_benign"
]].to_string(index=False))

print("\nCATEGORY SUMMARY")
summary = out.groupby(["category", "actual_class"], dropna=False).agg(
    captures=("capture", "count"),
    rows=("rows", "sum"),
    mean_accuracy=("accuracy", "mean"),
    mean_attack_recall=("attack_recall", "mean"),
    mean_false_positive_rate=("false_positive_rate", "mean"),
).reset_index()
print(summary.to_string(index=False))

print(f"\nSaved results: {csv_path}")
print("Reminder: these are candidate holdout captures; capture independence is not proven.")
print("No model or IPS settings were changed.")
