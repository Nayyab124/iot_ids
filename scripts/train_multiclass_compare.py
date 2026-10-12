from pathlib import Path
import json, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "CSV"
OUT = ROOT / "models" / "iot_ids_multiclass.joblib"
REPORT = ROOT / "reports" / "multiclass"
SEED = 42
MAX_PER_FOLDER = 1500
BENIGN_CAP = 6000
FAMILY_CAP = 20000

def get_family(folder):
    s = folder.lower().replace("_", " ").replace("-", " ")
    if "benign" in s or "normal" in s:
        return "Benign"
    if any(x in s for x in ["ddos", "distributed denial"]):
        return "DDoS"
    if any(x in s for x in ["malware", "mirai", "backdoor", "botnet"]):
        return "IoT Malware"
    if any(x in s for x in ["brute", "dictionary", "password", "credential", "patator"]):
        return "Brute Force"
    return "Network attacks"

def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset folder missing: {DATA}")
    folders = [p for p in sorted(DATA.iterdir()) if p.is_dir()]
    if not folders:
        raise SystemExit("No dataset category folders found.")

    rng = np.random.default_rng(SEED)
    samples = []
    print("Loading category folders (bounded sample for faster training)...")
    for folder in folders:
        files = sorted(folder.glob("*.csv"))
        if not files:
            continue
        cap = BENIGN_CAP if get_family(folder.name) == "Benign" else MAX_PER_FOLDER
        collected = []
        count = 0
        for file in files:
            if count >= cap:
                break
            try:
                # Read only a bounded sample per file to limit RAM use.
                remaining = cap - count
                chunk = pd.read_csv(file, nrows=min(remaining * 3, 12000),
                                    low_memory=False)
                if len(chunk) > remaining:
                    chunk = chunk.sample(n=remaining, random_state=SEED)
                if not chunk.empty:
                    collected.append(chunk)
                    count += len(chunk)
            except Exception as exc:
                print(f"Skipping {file.name}: {exc}")
        if collected:
            df = pd.concat(collected, ignore_index=True)
            df["__family__"] = get_family(folder.name)
            samples.append(df)
            print(f"{folder.name}: {len(df):,} rows -> {get_family(folder.name)}")

    if not samples:
        raise SystemExit("No usable CSV samples found.")

    df = pd.concat(samples, ignore_index=True, sort=False)
    y_text = df.pop("__family__")
    # Remove source labels to prevent target leakage.
    for c in list(df.columns):
        if str(c).strip().lower() in {
            "label", "labels", "class", "attack", "attack_type", "attack type"
        }:
            df.drop(columns=c, inplace=True)
    X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # Limit each final family so SMOTE and model training remain manageable.
    balanced_parts = []
    for family, group in X.assign(__family__=y_text.values).groupby("__family__"):
        cap = BENIGN_CAP if family == "Benign" else FAMILY_CAP
        if len(group) > cap:
            group = group.sample(n=cap, random_state=SEED)
        balanced_parts.append(group)
    data = pd.concat(balanced_parts, ignore_index=True)
    y_text = data.pop("__family__")
    X = data
    counts = y_text.value_counts()
    valid = counts[counts >= 10].index
    keep = y_text.isin(valid)
    X, y_text = X.loc[keep], y_text.loc[keep]

    if y_text.nunique() < 2:
        raise SystemExit("Need at least two families with 10 or more rows.")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    train_counts = pd.Series(y_train).value_counts()
    min_count = int(train_counts.min())
    if min_count < 2:
        raise SystemExit("Insufficient training samples for SMOTE.")
    X_fit, y_fit = SMOTE(
        random_state=SEED, k_neighbors=min(5, min_count - 1)
    ).fit_resample(X_train, y_train)
    print(f"\nTraining rows: {len(y_train):,}; after SMOTE: {len(y_fit):,}")
    print("Families:", list(encoder.classes_))

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, random_state=SEED, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=SEED, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.1,
            objective="multi:softprob", num_class=len(encoder.classes_),
            eval_metric="mlogloss", tree_method="hist",
            n_jobs=4, random_state=SEED
        ),
    }

    REPORT.mkdir(parents=True, exist_ok=True)
    results = []
    reports = {}
    best_model, best_name, best_f1 = None, None, -1

    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_fit, y_fit)
            pred = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="weighted", zero_division=0
            )
            results.append({
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
            })
            reports[name] = classification_report(
                y_test, pred, labels=np.arange(len(encoder.classes_)),
                target_names=encoder.classes_, zero_division=0
            )
            cm = confusion_matrix(
                y_test, pred, labels=np.arange(len(encoder.classes_))
            )
            pd.DataFrame(
                cm, index=encoder.classes_, columns=encoder.classes_
            ).to_csv(REPORT / (name.lower().replace(" ", "_") + "_confusion.csv"))
            print(f"Accuracy: {accuracy_score(y_test, pred):.4f}; weighted F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_name, best_model = f1, name, model
        except Exception as exc:
            print(f"FAILED {name}: {exc}")

    if best_model is None:
        raise SystemExit("All model training attempts failed.")

    joblib.dump({
        "model": best_model,
        "imputer": imputer,
        "scaler": scaler,
        "features": list(X.columns),
        "label_encoder": encoder,
        "classes": list(encoder.classes_),
        "best_model": best_name,
    }, OUT)

    results_df = pd.DataFrame(results).sort_values(
        "f1_weighted", ascending=False
    )
    results_df.to_csv(REPORT / "model_comparison.csv", index=False)
    (REPORT / "classification_reports.txt").write_text(
        "\n\n".join(f"{k}\n{v}" for k, v in reports.items()),
        encoding="utf-8"
    )
    (REPORT / "summary.json").write_text(json.dumps({
        "rows_used": int(len(X)),
        "features": int(len(X.columns)),
        "classes": list(encoder.classes_),
        "best_model": best_name,
        "smote_training_only": True,
        "split": "stratified 80/20 row-level split",
        "warning": "Validate on separate captures before claiming real-world performance."
    }, indent=2), encoding="utf-8")

    print("\\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))
    print(f"\\nBest model: {best_name}")
    print(f"Saved: {OUT}")
    print(f"Reports: {REPORT}")
    print("Existing binary IDS model was not modified.")

if __name__ == "__main__":
    main()
