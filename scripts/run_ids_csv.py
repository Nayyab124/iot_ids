from pathlib import Path
from datetime import datetime, timezone
import argparse

import joblib
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ciciot_xgboost_binary.json"
PREP_PATH = ROOT / "models" / "ciciot_preprocessing.joblib"
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Run the IoT IDS on a CICIoT2023 feature CSV."
    )
    parser.add_argument("--input", required=True, help="Input feature CSV")
    parser.add_argument(
        "--output",
        default=str(ROOT / "logs" / "ids_predictions.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum input rows to evaluate; use 0 for all rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    threshold = float(
        config.get("detection", {}).get("confidence_threshold", 0.80)
    )

    prep = joblib.load(PREP_PATH)
    feature_columns = prep["feature_columns"]
    medians = pd.Series(prep["medians"]).reindex(feature_columns)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    nrows = args.limit if args.limit > 0 else None
    df = pd.read_csv(input_path, nrows=nrows, low_memory=False)

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    # Match the model's exact training features and their order.
    X = df.reindex(columns=feature_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians).fillna(0)

    attack_probability = model.predict_proba(X)[:, 1]
    is_attack = attack_probability >= threshold

    result = df.copy()
    result["attack_probability"] = attack_probability
    result["prediction"] = np.where(is_attack, "ATTACK", "BENIGN")
    result["alert"] = is_attack
    result["confidence_threshold"] = threshold
    result["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("CSV IDS evaluation completed.")
    print(f"Input: {input_path}")
    print(f"Rows evaluated: {len(result)}")
    print(f"Alert threshold: {threshold:.2f}")
    print(f"Predicted attacks: {int(is_attack.sum())}")
    print(f"Predicted benign: {int((~is_attack).sum())}")
    print(f"Alerts written: {int(is_attack.sum())}")
    print(f"Results saved to: {output_path}")
    print("Firewall blocking: DISABLED")


if __name__ == "__main__":
    main()
