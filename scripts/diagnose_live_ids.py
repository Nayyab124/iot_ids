
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iot_ids_multiclass.joblib"
LOG_PATH = ROOT / "logs" / "live_multiclass_predictions.csv"


def inspect_object(obj, prefix="bundle"):
    print(f"\n--- {prefix} ---")
    print("Type:", type(obj).__name__)

    if isinstance(obj, dict):
        print("Keys:", list(obj.keys()))
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in {
                "feature_names", "features", "feature_columns",
                "classes", "class_names", "model", "pipeline",
                "scaler", "imputer", "label_encoder"
            }:
                print(f"\n{key}:")
                if hasattr(value, "feature_names_in_"):
                    print("  feature_names_in_:",
                          list(value.feature_names_in_))
                elif hasattr(value, "classes_"):
                    print("  classes_:", list(value.classes_))
                elif isinstance(value, (list, tuple)):
                    print(" ", value[:60])
                elif isinstance(value, str):
                    print(" ", value)
                else:
                    print(" ", type(value).__name__)

            if hasattr(value, "feature_names_in_"):
                print(f"{key}.feature_names_in_:",
                      list(value.feature_names_in_))
            if hasattr(value, "classes_"):
                print(f"{key}.classes_:",
                      list(value.classes_))
            if hasattr(value, "named_steps"):
                print(f"{key}.named_steps:",
                      list(value.named_steps.keys()))
                for step_name, step in value.named_steps.items():
                    if hasattr(step, "feature_names_in_"):
                        print(
                            f"  {step_name}.feature_names_in_:",
                            list(step.feature_names_in_)
                        )


def main():
    if not MODEL_PATH.exists():
        print("ERROR: Model not found:", MODEL_PATH)
        return

    bundle = joblib.load(MODEL_PATH)
    inspect_object(bundle)

    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        print("\n--- LIVE LOG SUMMARY ---")
        print("Rows:", len(df))
        print("Columns:", list(df.columns))

        if "prediction" in df.columns:
            print("\nPrediction counts:")
            print(df["prediction"].value_counts(dropna=False))

        if "confidence" in df.columns:
            print("\nConfidence summary:")
            print(pd.to_numeric(
                df["confidence"], errors="coerce"
            ).describe().to_string())

        if "packets" in df.columns:
            print("\nPackets per window:")
            print(pd.to_numeric(
                df["packets"], errors="coerce"
            ).describe().to_string())

        if "status" in df.columns:
            print("\nStatus counts:")
            print(df["status"].value_counts(dropna=False))
    else:
        print("\nNo live multiclass CSV exists yet:", LOG_PATH)


if __name__ == "__main__":
    main()
