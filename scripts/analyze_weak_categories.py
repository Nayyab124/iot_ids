from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
OUT = ROOT / "reports"
LIMIT = 5000

TARGETS = [
    ("Benign_Final/BenignTraffic1.pcap.csv", 0),
    ("Benign_Final/BenignTraffic2.pcap.csv", 0),
    ("Benign_Final/BenignTraffic3.pcap.csv", 0),
    ("DictionaryBruteForce/DictionaryBruteForce.pcap.csv", 1),
    ("DNS_Spoofing/DNS_Spoofing.pcap.csv", 1),
    ("MITM-ArpSpoofing/MITM-ArpSpoofing.pcap.csv", 1),
    ("MITM-ArpSpoofing/MITM-ArpSpoofing1.pcap.csv", 1),
    ("Recon-HostDiscovery/Recon-HostDiscovery.pcap.csv", 1),
    ("Recon-OSScan/Recon-OSScan.pcap.csv", 1),
    ("Recon-PingSweep/Recon-PingSweep.pcap.csv", 1),
    ("Recon-PortScan/Recon-PortScan.pcap.csv", 1),
    ("SqlInjection/SqlInjection.pcap.csv", 1),
    ("Uploading_Attack/Uploading_Attack.pcap.csv", 1),
    ("VulnerabilityScan/VulnerabilityScan.pcap.csv", 1),
    ("XSS/XSS.pcap.csv", 1),
]

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

model = xgb.XGBClassifier()
model.load_model(str(ROOT / "models" / "ciciot_xgboost_binary.json"))
prep = joblib.load(ROOT / "models" / "ciciot_preprocessing.joblib")
columns = prep["feature_columns"]
medians = pd.Series(prep["medians"])

rows = []

for relative, actual in TARGETS:
    path = DATA / relative

    if not path.is_file():
        print(f"MISSING: {relative}")
        continue

    df = pd.read_csv(path, nrows=LIMIT, low_memory=False)

    if df.empty:
        print(f"EMPTY: {relative}")
        continue

    X = df.reindex(columns=columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    probabilities = model.predict_proba(X)[:, 1]

    for threshold in THRESHOLDS:
        rate = float((probabilities >= threshold).mean())
        rows.append({
            "capture": relative,
            "actual_class": "Benign" if actual == 0 else "Attack",
            "threshold": threshold,
            "metric": "false_positive_rate" if actual == 0 else "attack_detection_rate",
            "rate": rate,
            "rows": len(df),
        })

    print(f"Processed {relative}: {len(df):,} rows")

if not rows:
    raise RuntimeError("No evaluation files were processed. Check the CSV paths.")

result = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "weak_category_thresholds.csv"
result.to_csv(csv_path, index=False)

print("\nDETECTION RATE BY ATTACK CAPTURE (%)")
attacks = result[result["actual_class"] == "Attack"]
print((attacks.pivot(index="capture", columns="threshold", values="rate") * 100).round(2).to_string())

print("\nFALSE-POSITIVE RATE BY BENIGN CAPTURE (%)")
benign = result[result["actual_class"] == "Benign"]
if benign.empty:
    print("No benign captures were processed.")
else:
    print((benign.pivot(index="capture", columns="threshold", values="rate") * 100).round(2).to_string())

print(f"\nSaved: {csv_path}")
print("Evaluation only: model, configured threshold, and IPS settings were not changed.")