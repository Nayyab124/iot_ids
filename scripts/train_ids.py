from pathlib import Path
from datetime import datetime
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ciciot_xgboost_binary.json"
PREP_PATH = MODEL_DIR / "ciciot_preprocessing.joblib"
REPORT_PATH = REPORT_DIR / "training_report.txt"

CHUNK_SIZE = 50_000
CHUNK_SAMPLE = 500
ATTACK_CAP = 5_000
BENIGN_CAP = 20_000
RANDOM_STATE = 42
DEPLOYMENT_THRESHOLD = 0.80

rng = np.random.default_rng(RANDOM_STATE)

# Back up the existing artifacts before training.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for path in (MODEL_PATH, PREP_PATH, REPORT_PATH):
    if path.exists():
        backup = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")

if not DATA.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA}")

folders = sorted(p for p in DATA.iterdir() if p.is_dir())
if not folders:
    raise RuntimeError(f"No category folders found in {DATA}")

frames = []
category_counts = {}
expected_columns = None

print("\n========== LOADING ALL CATEGORIES ==========")

for folder in folders:
    files = sorted(folder.glob("*.csv"))
    if not files:
        print(f"SKIP {folder.name}: no CSV files")
        continue

    is_benign = "benign" in folder.name.lower()
    label = 0 if is_benign else 1
    cap = BENIGN_CAP if is_benign else ATTACK_CAP
    reservoir = pd.DataFrame()
    incompatible = 0

    for file in files:
        try:
            for chunk in pd.read_csv(
                file, chunksize=CHUNK_SIZE, low_memory=False
            ):
                if expected_columns is None:
                    expected_columns = list(chunk.columns)

                if list(chunk.columns) != expected_columns:
                    incompatible += 1
                    continue

                if chunk.empty:
                    continue

                take = min(CHUNK_SAMPLE, len(chunk))
                sample = chunk.sample(
                    n=take,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )

                # Keep a bounded sample while processing the entire folder.
                reservoir = pd.concat(
                    [reservoir, sample], ignore_index=True
                )
                if len(reservoir) > cap:
                    reservoir = reservoir.sample(
                        n=cap,
                        random_state=RANDOM_STATE,
                    ).reset_index(drop=True)

        except Exception as exc:
            print(f"WARNING reading {file.name}: {exc}")

    if reservoir.empty:
        print(f"SKIP {folder.name}: no compatible rows")
        continue

    reservoir = reservoir.copy()
    reservoir["_target"] = label
    frames.append(reservoir)
    category_counts[folder.name] = {
        "rows": len(reservoir),
        "label": "Benign" if label == 0 else "Attack",
        "files": len(files),
    }

    print(
        f"{folder.name}: sampled {len(reservoir):,} rows "
        f"from {len(files)} CSV file(s)"
        + (f"; incompatible chunks skipped: {incompatible}" if incompatible else "")
    )

if not frames:
    raise RuntimeError("No compatible dataset rows were loaded.")

df = pd.concat(frames, ignore_index=True)
del frames

y = df.pop("_target").astype("int8")
X = df.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove unusable and constant columns.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.empty:
    raise RuntimeError("No usable numeric features remain.")
if y.nunique() != 2:
    raise RuntimeError("Both benign and attack classes are required.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

medians = X_train.median().fillna(0)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

benign_count = int((y_train == 0).sum())
attack_count = int((y_train == 1).sum())
scale_pos_weight = benign_count / attack_count

print("\n========== DATA SUMMARY ==========")
print("Category folders loaded:", len(category_counts))
print("Total sampled rows:", len(X))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", X_train.shape[1])
print("Training benign rows:", benign_count)
print("Training attack rows:", attack_count)
print("Positive-class weight:", round(scale_pos_weight, 4))

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=4,
)

print("\n========== TRAINING XGBOOST ==========")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
lines = [
    "IoT IDS TRAINING REPORT",
    "Dataset: CICIoT2023 CSV category folders",
    "Validation: stratified row-level random split; not independent-capture validation",
    f"Random seed: {RANDOM_STATE}",
    f"Sampled rows: {len(X)}",
    f"Features: {X_train.shape[1]}",
    f"Training rows: {len(X_train)}",
    f"Testing rows: {len(X_test)}",
    "",
    "CATEGORY SAMPLE COUNTS:",
]

for name, info in category_counts.items():
    lines.append(
        f"{name}: rows={info['rows']}, class={info['label']}, "
        f"csv_files={info['files']}"
    )

for threshold in (0.50, DEPLOYMENT_THRESHOLD):
    pred = (probabilities >= threshold).astype("int8")
    matrix = confusion_matrix(y_test, pred, labels=[0, 1])
    report = classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
    )

    section = [
        "",
        f"========== THRESHOLD {threshold:.2f} ==========",
        f"Accuracy: {accuracy_score(y_test, pred):.4f}",
        f"Attack precision: {precision_score(y_test, pred, zero_division=0):.4f}",
        f"Attack recall: {recall_score(y_test, pred, zero_division=0):.4f}",
        f"Attack F1: {f1_score(y_test, pred, zero_division=0):.4f}",
        "Classification report:",
        report,
        "Confusion matrix (rows=true [Benign, Attack], columns=predicted):",
        str(matrix),
        f"Benign false positives: {int(matrix[0, 1])}",
        f"Missed attacks: {int(matrix[1, 0])}",
    ]
    lines.extend(section)
    print("\n" + "\n".join(section))

# Save artifacts only after training and evaluation succeed.
model.save_model(str(MODEL_PATH))
joblib.dump(
    {
        "feature_columns": list(X_train.columns),
        "medians": medians.to_dict(),
        "threshold": DEPLOYMENT_THRESHOLD,
        "label_mapping": {0: "Benign", 1: "Attack"},
    },
    PREP_PATH,
)

REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n========== SAVED ==========")
print("Model:", MODEL_PATH)
print("Preprocessing:", PREP_PATH)
print("Report:", REPORT_PATH)
print("\nTraining finished. Firewall blocking remains unchanged.")
