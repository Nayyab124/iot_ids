from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iot_ids_multiclass.joblib"
BENIGN_PATH = (
    ROOT / "data" / "raw" / "CSV" / "Benign_Final"
    / "BenignTraffic.pcap.csv"
)

spec = spec_from_file_location(
    "predict_multiclass",
    ROOT / "scripts" / "predict_multiclass.py",
)
predictor = module_from_spec(spec)
spec.loader.exec_module(predictor)


def test_multiclass_model_has_expected_classes_and_features():
    assert MODEL_PATH.is_file()

    bundle = predictor.bundle
    assert len(bundle["features"]) == 39
    assert set(bundle["classes"]) == {
        "Benign",
        "Brute Force",
        "DDoS",
        "IoT Malware",
        "Network attacks",
    }


def test_prediction_returns_valid_labels_and_confidences():
    features = predictor.features

    # Deliberately provide only one feature. The inference
    # pipeline should fill the other missing features safely.
    sample = pd.DataFrame({
        features[0]: [0.0, 1.0],
        "Label": ["Benign", "DDoS"],
    })

    result = predictor.predict_dataframe(sample)

    assert list(result.columns) == ["prediction", "confidence"]
    assert len(result) == 2
    assert result["prediction"].isin(predictor.bundle["classes"]).all()
    assert np.isfinite(result["confidence"]).all()
    assert result["confidence"].between(0.0, 1.0).all()


def test_benign_csv_rows_can_be_processed():
    assert BENIGN_PATH.is_file()

    sample = pd.read_csv(BENIGN_PATH, nrows=5, low_memory=False)
    assert not sample.empty

    result = predictor.predict_dataframe(sample)

    assert len(result) == len(sample)
    assert result["prediction"].notna().all()
    assert result["confidence"].between(0.0, 1.0).all()
