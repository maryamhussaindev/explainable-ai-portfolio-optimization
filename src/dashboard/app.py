from pathlib import Path

import pandas as pd
import yaml
from flask import Flask, render_template_string, request, send_file

BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = Flask(__name__)

FIG_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

RF_SHAP_IMAGE = FIG_DIR / "random_forest_shap_summary.png"
XGB_SHAP_IMAGE = FIG_DIR / "xgboost_shap_summary.png"
TABPFN_IMPORTANCE_IMAGE = FIG_DIR / "tabpfn_feature_importance.png"
EXPLANATIONS_FILE = TABLES_DIR / "prediction_explanations.csv"

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Explainable AI - Model Explanations</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; color: #222; }
        .header { background: #1f2d3d; color: #fff; padding: 18px 30px; }
        .header h1 { margin: 0; font-size: 22px; }
        .container { padding: 24px 30px; max-width: 1200px; margin: 0 auto; }
        .card {
            background: #fff; border: 1px solid #e3e6ea; border-radius: 8px;
            padding: 18px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
        }
        .card h2 { margin-top: 0; font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 8px; }
        .plots { display: flex; flex-wrap: wrap; gap: 18px; }
        .plot { flex: 1 1 45%; min-width: 320px; }
        .plot img { width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }
        .plot.wide { flex: 1 1 100%; }
        form { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; }
        .field label { display: block; font-size: 13px; margin-bottom: 4px; color: #555; }
        .field select {
            padding: 8px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px;
        }
        .explain-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        .explain-table th, .explain-table td {
            border: 1px solid #e3e6ea; padding: 10px 12px; text-align: left; font-size: 14px;
        }
        .explain-table th { background: #f5f7fa; }
        .model-tag { font-weight: 600; }
        .m-rf { color: #3565b9; } .m-xgb { color: #c2410c; } .m-tab { color: #1c8a4f; }
        .empty { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <div class="header"><h1>AI Model Explanations for Stock Predictions</h1></div>
    <div class="container">

        <div class="card">
            <h2>Forest &amp; Gradient Boosting Explanations (SHAP)</h2>
            <div class="plots">
                <div class="plot">
                    <h3>Random Forest SHAP Summary</h3>
                    <img src="/image/random_forest_shap_summary" alt="Random Forest SHAP summary">
                </div>
                <div class="plot">
                    <h3>XGBoost SHAP Summary</h3>
                    <img src="/image/xgboost_shap_summary" alt="XGBoost SHAP summary">
                </div>
            </div>
        </div>

        <div class="card">
            <h2>TabPFN Feature Importance (Permutation)</h2>
            <div class="plot wide">
                <img src="/image/tabpfn_feature_importance" alt="TabPFN permutation importance">
            </div>
        </div>

        <div class="card">
            <h2>Human-Readable Explanation</h2>
            <form method="get" action="/">
                <div class="field">
                    <label for="ticker">Stock</label>
                    <select name="ticker" id="ticker">
                        {% for t in tickers %}
                        <option value="{{ t }}" {% if t == selected_ticker %}selected{% endif %}>{{ t }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label for="date">Date</label>
                    <select name="date" id="date">
                        {% for d in dates %}
                        <option value="{{ d }}" {% if d == selected_date %}selected{% endif %}>{{ d }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <label for="model">Model</label>
                    <select name="model" id="model">
                        <option value="All" {% if selected_model == 'All' %}selected{% endif %}>All Models</option>
                        {% for m in models %}
                        <option value="{{ m }}" {% if m == selected_model %}selected{% endif %}>{{ m }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="field">
                    <button type="submit"
                        style="background:#1f2d3d;color:#fff;border:none;padding:9px 18px;border-radius:6px;font-size:14px;cursor:pointer;">
                        Show Explanation
                    </button>
                </div>
            </form>

            {% if explanations %}
            <table class="explain-table">
                <thead>
                    <tr><th>Date</th><th>Stock</th><th>Model</th><th>Explanation</th></tr>
                </thead>
                <tbody>
                    {% for ex in explanations %}
                    <tr>
                        <td>{{ ex['Date'] }}</td>
                        <td>{{ ex['Ticker'] }}</td>
                        <td class="model-tag {{ ex['ModelClass'] }}">{{ ex['Model'] }}</td>
                        <td>{{ ex['Explanation'] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p class="empty">No explanation available for the selected stock/date/model.</p>
            {% endif %}
        </div>

    </div>
</body>
</html>
"""


MODEL_CLASSES = {
    "RandomForest": "m-rf",
    "XGBoost": "m-xgb",
    "TabPFN": "m-tab",
}


def _load_explanations() -> pd.DataFrame:
    return pd.read_csv(EXPLANATIONS_FILE)


def _server_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f).get("dashboard", {})


@app.route("/")
def explainability_page():
    if not EXPLANATIONS_FILE.exists():
        return "Explanation report not found. Run src.explainability.explanation_report first.", 404

    explanations = _load_explanations()

    tickers = sorted(explanations["Ticker"].unique().tolist())
    dates = sorted(explanations["Date"].unique().tolist())
    models = explanations["Model"].unique().tolist()

    selected_ticker = request.args.get("ticker", tickers[0] if tickers else "")
    selected_date = request.args.get("date", dates[-1] if dates else "")
    selected_model = request.args.get("model", "All")

    if selected_ticker not in tickers:
        selected_ticker = tickers[0] if tickers else ""
    if selected_date not in dates:
        selected_date = dates[-1] if dates else ""

    filtered = explanations[
        (explanations["Ticker"] == selected_ticker)
        & (explanations["Date"] == selected_date)
    ]
    if selected_model != "All":
        filtered = filtered[filtered["Model"] == selected_model]

    rows = []
    for _, r in filtered.iterrows():
        rows.append({
            "Date": r["Date"],
            "Ticker": r["Ticker"],
            "Model": r["Model"],
            "ModelClass": MODEL_CLASSES.get(r["Model"], ""),
            "Explanation": r["Explanation"],
        })

    return render_template_string(
        PAGE_TEMPLATE,
        tickers=tickers,
        dates=dates,
        models=models,
        selected_ticker=selected_ticker,
        selected_date=selected_date,
        selected_model=selected_model,
        explanations=rows,
    )


@app.route("/image/<name>")
def image(name):
    paths = {
        "random_forest_shap_summary": RF_SHAP_IMAGE,
        "xgboost_shap_summary": XGB_SHAP_IMAGE,
        "tabpfn_feature_importance": TABPFN_IMPORTANCE_IMAGE,
    }
    path = paths.get(name)
    if path is None or not path.exists():
        return "Image not found.", 404
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    cfg = _server_config()
    app.run(
        host=cfg.get("host", "127.0.0.1"),
        port=cfg.get("port", 8050),
        debug=cfg.get("debug", True),
    )
