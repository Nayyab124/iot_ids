from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iot_ids_multiclass.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
features = bundle["features"]
encoder = bundle["label_encoder"]

def predict_dataframe(df):
    """
    Predict attack family from a dataframe containing CICIoT2023 features.
    """
    X = df.copy()

    # Remove label if it exists.
    for col in list(X.columns):
        if str(col).strip().lower() in {
            "label", "labels", "class", "attack",
            "attack_type", "attack type"
        }:
            X.drop(columns=col, inplace=True)

    # Guarantee the same feature order used during training.
    for col in features:
        if col not in X.columns:
            X[col] = np.nan

    X = X[features]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X = imputer.transform(X)
    X = scaler.transform(X)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    labels = encoder.inverse_transform(predictions)
    confidence = probabilities.max(axis=1)

    return pd.DataFrame({
        "prediction": labels,
        "confidence": confidence
    })

def predict_csv(csv_path):
    df = pd.read_csv(csv_path)
    return predict_dataframe(df)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/predict_multiclass.py <csv_file>")
        raise SystemExit(1)

    result = predict_csv(sys.argv[1])
    print(result.head(20).to_string(index=False))
