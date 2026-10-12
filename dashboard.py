
from pathlib import Path
from io import BytesIO
from datetime import datetime
import csv

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask, request, render_template_string,
    jsonify, send_file
)

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iot_ids_multiclass.joblib"
LOG_FILE = ROOT / "logs" / "live_multiclass_predictions.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
features = bundle["features"]
encoder = bundle["label_encoder"]

CLASS_NAMES = [
    "Benign",
    "Brute Force",
    "DDoS",
    "IoT Malware",
    "Network attacks",
]

last_upload_results = None


def predict_dataframe(df):
    """Predict the five-class label for each uploaded CSV row."""
    X = df.copy()

    label_columns = {
        "label", "labels", "class", "attack",
        "attack_type", "attack type",
    }

    for column in list(X.columns):
        if str(column).strip().lower() in label_columns:
            X.drop(columns=column, inplace=True)

    for feature in features:
        if feature not in X.columns:
            X[feature] = np.nan

    X = X[features]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X = imputer.transform(X)
    X = scaler.transform(X)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    labels = encoder.inverse_transform(predictions)

    return pd.DataFrame({
        "prediction": labels,
        "confidence": probabilities.max(axis=1),
    })


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>IoT Intrusion Detection System</title>
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 24px;
        background: #f4f6f9;
        color: #202938;
    }
    h1 { margin-bottom: 8px; }
    h2 { margin-top: 28px; }
    .subtitle { color: #586579; }
    .notice {
        background: #fff8e6;
        border-left: 4px solid #d99a00;
        padding: 12px;
        margin: 18px 0;
    }
    .cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 15px;
        margin: 22px 0;
    }
    .card, .panel {
        background: white;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,.07);
    }
    .value {
        font-size: 27px;
        font-weight: bold;
        margin-top: 10px;
    }
    .class-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
    }
    .class-card {
        background: white;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #e0e5ec;
    }
    .form-box {
        background: white;
        padding: 20px;
        margin: 22px 0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,.07);
    }
    button, .download {
        display: inline-block;
        padding: 10px 16px;
        background: #2458a6;
        color: white;
        border: 0;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        margin-top: 10px;
    }
    .table-wrap {
        overflow-x: auto;
        background: white;
        border-radius: 8px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
    }
    th, td {
        padding: 10px;
        border-bottom: 1px solid #e1e5eb;
        text-align: left;
        white-space: nowrap;
    }
    th { background: #e9eef5; }
    .attack { color: #b42318; font-weight: bold; }
    .benign { color: #18794e; font-weight: bold; }
    .muted { color: #64748b; }
    .error {
        background: #feecec;
        padding: 12px;
        color: #9d2020;
        border-radius: 6px;
    }
    @media (max-width: 600px) {
        body { padding: 12px; }
    }
</style>
</head>
<body>

<h1>IoT Intrusion Detection System</h1>
<p class="subtitle">
    Experimental live multiclass detection and machine-learning evaluation
</p>

<div class="notice">
    <strong>Experimental status:</strong>
    Live predictions use approximate packet-window features.
    A non-Benign prediction is not proof of a real attack.
    Confidence is a model probability, not measured accuracy.
</div>

<div class="cards">
    <div class="card">
        <div>Total Windows</div>
        <div class="value">{{ total }}</div>
    </div>
    <div class="card">
        <div>Non-Benign Predictions</div>
        <div class="value">{{ attacks }}</div>
    </div>
    <div class="card">
        <div>Experimental Alert Rate</div>
        <div class="value">{{ alert_rate }}%</div>
    </div>
    <div class="card">
        <div>Last Log Update</div>
        <div style="margin-top:10px">{{ last_update }}</div>
    </div>
</div>

<h2>Live Class Distribution</h2>
<div class="class-grid">
{% for name in class_names %}
    <div class="class-card">
        <div>{{ name }}</div>
        <div class="value">{{ class_counts[name] }}</div>
    </div>
{% endfor %}
</div>

<div class="form-box">
    <h2>Multiclass CSV Prediction</h2>
    <p>
        Upload a traffic feature CSV to classify rows as Benign,
        Brute Force, DDoS, IoT Malware, or Network attacks.
    </p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <button type="submit">Predict CSV</button>
    </form>
    <p class="muted">
        Upload data should contain numeric traffic features compatible
        with the model's 39 input features.
    </p>
</div>

{% if upload_columns %}
<h2>Uploaded CSV Prediction Results</h2>
<div class="panel">
    <p>Rows predicted: <strong>{{ upload_total }}</strong></p>
    <a class="download" href="/download?source=upload">
        Download Prediction Results
    </a>
</div>
<div class="table-wrap">
<table>
    <tr>
    {% for col in upload_columns %}
        <th>{{ col }}</th>
    {% endfor %}
    </tr>
    {% for row in upload_rows %}
    <tr>
        {% for value in row %}
            <td>{{ value }}</td>
        {% endfor %}
    </tr>
    {% endfor %}
</table>
</div>
{% endif %}

{% if error %}
<div class="error">{{ error }}</div>
{% endif %}

<h2>Recent Live Multiclass Predictions</h2>
<p class="muted">
    The page refreshes every 15 seconds while open.
    Live log: logs/live_multiclass_predictions.csv
</p>

<div class="table-wrap">
<table>
    <tr>
    {% for col in columns %}
        <th>{{ col }}</th>
    {% endfor %}
    </tr>
    {% for row in rows %}
    <tr>
        {% for value in row %}
            {% if loop.index0 == prediction_index %}
                <td class="{% if value == 'Benign' %}benign{% else %}attack{% endif %}">
                    {{ value }}
                </td>
            {% else %}
                <td>{{ value }}</td>
            {% endif %}
        {% endfor %}
    </tr>
    {% endfor %}
</table>
</div>

</body>
</html>
"""


def load_live_log():
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(LOG_FILE)
        if "prediction" not in df.columns:
            return pd.DataFrame()
        return df
    except (pd.errors.ParserError, UnicodeDecodeError, OSError):
        return pd.DataFrame()


def make_dashboard_context(error=None):
    df = load_live_log()

    class_counts = {name: 0 for name in CLASS_NAMES}

    if not df.empty:
        labels = df["prediction"].astype(str)
        total = len(df)
        attacks = int((labels.str.strip().str.lower() != "benign").sum())

        for name in CLASS_NAMES:
            class_counts[name] = int(
                (labels.str.strip().str.lower() == name.lower()).sum()
            )

        alert_rate = round(100 * attacks / total, 2) if total else 0.0

        if "timestamp" in df.columns:
            last_update = str(df["timestamp"].iloc[-1])
        else:
            last_update = "Available"
    else:
        total = 0
        attacks = 0
        alert_rate = 0.0
        last_update = "No live data yet"

    # Show only the latest 20 live rows and only useful columns.
    preferred = [
        "timestamp", "prediction", "confidence",
        "packets", "window_duration_seconds", "status",
    ]
    columns = [column for column in preferred if column in df.columns]

    if df.empty:
        recent = pd.DataFrame(columns=columns)
    else:
        recent = df[columns].tail(20).iloc[::-1].copy()

        if "confidence" in recent.columns:
            recent["confidence"] = recent["confidence"].apply(
                lambda value: (
                    f"{float(value):.1%}"
                    if pd.notna(value) else ""
                )
            )

    prediction_index = (
        columns.index("prediction") if "prediction" in columns else -1
    )

    upload_df = last_upload_results
    if upload_df is not None:
        upload_columns = list(upload_df.columns)
        upload_rows = upload_df.head(50).fillna("").astype(str).values.tolist()
        upload_total = len(upload_df)
    else:
        upload_columns = []
        upload_rows = []
        upload_total = 0

    return {
        "total": total,
        "attacks": attacks,
        "alert_rate": f"{alert_rate:.2f}",
        "last_update": last_update,
        "class_names": CLASS_NAMES,
        "class_counts": class_counts,
        "columns": columns,
        "rows": recent.fillna("").astype(str).values.tolist(),
        "prediction_index": prediction_index,
        "upload_columns": upload_columns,
        "upload_rows": upload_rows,
        "upload_total": upload_total,
        "error": error,
    }


@app.route("/")
def dashboard():
    return render_template_string(**{}) if False else render_template_string(
        HTML, **make_dashboard_context()
    )


@app.route("/predict", methods=["POST"])
def predict_upload():
    global last_upload_results

    uploaded = request.files.get("file")

    if not uploaded or not uploaded.filename:
        return render_template_string(
            HTML, **make_dashboard_context("Please select a CSV file.")
        ), 400

    if not uploaded.filename.lower().endswith(".csv"):
        return render_template_string(
            HTML, **make_dashboard_context("Please upload a .csv file.")
        ), 400

    try:
        df = pd.read_csv(uploaded)
        if df.empty:
            raise ValueError("The uploaded CSV contains no data rows.")

        result = predict_dataframe(df)
        last_upload_results = pd.concat(
            [df.reset_index(drop=True), result], axis=1
        )

        return render_template_string(
            HTML, **make_dashboard_context()
        )
    except Exception as exc:
        return render_template_string(
            HTML,
            **make_dashboard_context(f"CSV prediction failed: {exc}")
        ), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    uploaded = request.files.get("file")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            result = predict_dataframe(df)
            return jsonify({
                "rows": len(result),
                "results": result.to_dict(orient="records"),
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return jsonify({
        "error": "Upload a CSV using the multipart field named 'file'."
    }), 400


@app.route("/download")
def download():
    source = request.args.get("source", "upload")

    if source == "upload" and last_upload_results is not None:
        df = last_upload_results
        filename = "multiclass_predictions.csv"
    else:
        df = load_live_log()
        filename = "live_multiclass_predictions.csv"

    if df.empty:
        return jsonify({"error": "No results are available to download."}), 404

    buffer = BytesIO(df.to_csv(index=False).encode("utf-8"))
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    print("Starting experimental multiclass IoT IDS dashboard")
    print(f"Live log: {LOG_FILE}")
    app.run(host="127.0.0.1", port=5000, debug=False)
