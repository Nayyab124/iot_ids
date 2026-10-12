from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_LOG = ROOT / "logs" / "live_multiclass_features.csv"
MODEL_PATH = ROOT / "models" / "iot_ids_multiclass.joblib"

FILES = {
    "Benign training sample": ROOT / "data/raw/CSV/Benign_Final/BenignTraffic.pcap.csv",
    "DDoS training sample": ROOT / "data/raw/CSV/DoS-SYN_Flood/DoS-SYN_Flood.pcap.csv",
}


def main():
    bundle = joblib.load(MODEL_PATH)
    features = list(bundle["features"])
    live = pd.read_csv(FEATURE_LOG)

    live.columns = live.columns.str.strip()
    print("Expected model features:", len(features))
    print("Live rows:", len(live))
    print("Live feature columns present:",
          sum(f in live.columns for f in features), "/", len(features))

    missing_live = [f for f in features if f not in live.columns]
    if missing_live:
        print("Missing live features:", missing_live)

    print("\nLIVE FEATURE MEDIANS")
    print(live[features].apply(pd.to_numeric, errors="coerce").median().round(4).to_string())

    for title, path in FILES.items():
        print(f"\n{'=' * 65}\n{title}\nFile exists: {path.exists()}")

        if not path.exists():
            continue

        df = pd.read_csv(path, nrows=3000, low_memory=False)
        df.columns = df.columns.str.strip()
        missing = [f for f in features if f not in df.columns]
        print("Sample rows read:", len(df))
        print("Training feature columns present:",
              len(features) - len(missing), "/", len(features))
        if missing:
            print("Missing training columns:", missing)
            continue

        numeric = df[features].apply(pd.to_numeric, errors="coerce")
        print("\nTraining feature medians:")
        print(numeric.median().round(4).to_string())

        print("\nSelected training ranges:")
        selected = [f for f in [
            "Header_Length", "Rate", "Tot sum", "Min", "Max",
            "AVG", "Std", "Tot size", "IAT", "Number", "Variance"
        ] if f in numeric.columns]
        print(numeric[selected].quantile([0.1, 0.5, 0.9]).round(4).to_string())

    print("\nNote: training medians are row-level samples; live values are")
    print("packet-window aggregates. Differences identify a mismatch to")
    print("investigate, not proof of model accuracy or attack activity.")


if __name__ == "__main__":
    main()
