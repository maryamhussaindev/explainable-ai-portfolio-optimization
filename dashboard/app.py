from pathlib import Path

import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, send_file, url_for

BASE_DIR = Path(__file__).resolve().parent.parent

FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

STATIC_IMAGES = {
    "window_performance": FIGURES_DIR / "window_performance.png",
    "portfolio_growth": FIGURES_DIR / "portfolio_growth.png",
    "cumulative_returns": FIGURES_DIR / "cumulative_returns.png",
    "random_forest_shap_summary": FIGURES_DIR / "random_forest_shap_summary.png",
    "xgboost_shap_summary": FIGURES_DIR / "xgboost_shap_summary.png",
    "tabpfn_feature_importance": FIGURES_DIR / "tabpfn_feature_importance.png",
}

PREDICTION_FILES = {
    "RandomForest": EXPERIMENTS_DIR / "predictions" / "random_forest_predictions.csv",
    "XGBoost": EXPERIMENTS_DIR / "predictions" / "xgboost_predictions.csv",
    "TabPFN": EXPERIMENTS_DIR / "predictions" / "tabpfn_predictions.csv",
}

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

FEATURE_KEYWORDS = [
    ("the macd histogram", "MACD_Histogram"),
    ("the macd signal line", "MACD_Signal"),
    ("the 20-day ema", "EMA_20"),
    ("the daily return", "Daily_Return"),
    ("momentum", "Momentum_10"),
    ("rolling volatility", "Rolling_Volatility_20"),
    ("volatility", "Rolling_Volatility_20"),
    ("macd", "MACD"),
    ("rsi", "RSI_14"),
]

SIGNAL_COLORS = {
    "BUY": "#198754",
    "HOLD": "#ffc107",
    "SELL": "#dc3545",
}

MISSING_MESSAGE = "Result not available. Run the corresponding pipeline module."

app = Flask(__name__)


def read_csv_or_none(csv_path):
    try:
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception:
        return None
    return None


def image_exists(name):
    path = STATIC_IMAGES.get(name)
    return path is not None and path.exists()


def best_strategy_from_metrics(metrics_df):
    if metrics_df is None or metrics_df.empty:
        return None
    sharpe_col = [c for c in metrics_df.columns if "sharpe" in c.lower()]
    if not sharpe_col:
        return metrics_df.iloc[0]
    return metrics_df.sort_values(sharpe_col[0], ascending=False).iloc[0]


def allocation_for_strategy(strategy_name):
    weights = read_csv_or_none(EXPERIMENTS_DIR / "portfolio_weights.csv")
    if weights is None or weights.empty:
        return pd.DataFrame({"Ticker": [], "Weight": []})
    if "Ticker" not in weights.columns or "Weight" not in weights.columns:
        return pd.DataFrame({"Ticker": [], "Weight": []})
    if "Strategy" in weights.columns:
        weights = weights[weights["Strategy"] == strategy_name]
    alloc = weights.groupby("Ticker", as_index=False)["Weight"].mean()
    alloc = alloc.sort_values("Weight", ascending=False)
    return alloc


def latest_window_predictions(pred_df):
    df = pred_df.copy()
    df["_suffix"] = df["window_id"].astype(str).str.split("_").str[-1].astype(int)
    latest = df["_suffix"].max()
    return df[df["_suffix"] == latest].copy()


def signal_for_return(pred):
    if pred is None or pd.isna(pred):
        return "N/A"
    if pred > 0.01:
        return "BUY"
    if pred < -0.01:
        return "SELL"
    return "HOLD"


def prediction_rows(pred_df):
    rows = []
    for rec in pred_df.to_dict("records"):
        signal = signal_for_return(rec.get("Predicted_Return"))
        rows.append({
            "Date": rec.get("Date"),
            "Ticker": rec.get("Ticker"),
            "Actual_Return": rec.get("Actual_Return"),
            "Predicted_Return": rec.get("Predicted_Return"),
            "Signal": signal,
            "SignalColor": SIGNAL_COLORS.get(signal, "#6c757d"),
        })
    return rows


def top_features_for_stock(expl_df, ticker, n=3):
    if expl_df is None or expl_df.empty or "Ticker" not in expl_df.columns:
        return []
    texts = expl_df.loc[expl_df["Ticker"] == ticker, "Explanation"].fillna("")
    counts = {}
    for text in texts:
        t = str(text).lower()
        for phrase, feat in FEATURE_KEYWORDS:
            if phrase in t:
                counts[feat] = counts.get(feat, 0) + 1
                t = t.replace(phrase, "")
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [feat for feat, _ in ranked[:n]]


def load_prediction_file(model_name):
    path = PREDICTION_FILES.get(model_name)
    if path is None:
        return None
    return read_csv_or_none(path)


@app.route("/")
def overview():
    comparison = read_csv_or_none(TABLES_DIR / "model_comparison.csv")

    best_model = None
    if comparison is not None and not comparison.empty:
        model_col = comparison.columns[0]
        rmse_col = None
        for c in comparison.columns:
            if c.lower() == "rmse":
                rmse_col = c
                break
        display_cols = [c for c in comparison.columns if c.lower() in ("model", "mae", "rmse", "r2", "directional_accuracy")]
        if rmse_col:
            best_model = comparison.loc[comparison[rmse_col].idxmin()].to_dict()
    else:
        display_cols = []

    return render_template(
        "index.html",
        comparison=comparison,
        display_cols=display_cols,
        best_model=best_model,
        window_img=image_exists("window_performance"),
        missing_message=MISSING_MESSAGE,
    )


@app.route("/portfolio")
def portfolio():
    metrics = read_csv_or_none(TABLES_DIR / "portfolio_metrics.csv")
    best = best_strategy_from_metrics(metrics)
    allocation = pd.DataFrame({"Ticker": [], "Weight": []})
    pie_html = None

    if best is not None:
        strategy_name = best[metrics.columns[0]]
        allocation = allocation_for_strategy(strategy_name)
        if not allocation.empty:
            fig = px.pie(
                allocation,
                names="Ticker",
                values="Weight",
                title=f"Average Allocation - {strategy_name}",
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            pie_html = fig.to_html(
                full_html=False, include_plotlyjs="cdn"
            )

    return render_template(
        "portfolio.html",
        metrics=metrics,
        best=best,
        allocation=allocation,
        pie_html=pie_html,
        growth_img=image_exists("portfolio_growth"),
        cumulative_img=image_exists("cumulative_returns"),
        missing_message=MISSING_MESSAGE,
    )


@app.route("/explainability")
def explainability():
    tabpfn_importance = read_csv_or_none(TABLES_DIR / "tabpfn_permutation_importance.csv")
    explanations = read_csv_or_none(TABLES_DIR / "prediction_explanations.csv")

    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
    selected = request.args.get("ticker", "AAPL")
    if selected not in tickers:
        selected = tickers[0]

    filtered = None
    if explanations is not None and not explanations.empty:
        if "Ticker" in explanations.columns:
            filtered = explanations[explanations["Ticker"] == selected]
        else:
            filtered = explanations

    return render_template(
        "explainability.html",
        tickers=tickers,
        selected=selected,
        rf_img=image_exists("random_forest_shap_summary"),
        xgb_img=image_exists("xgboost_shap_summary"),
        tabpfn_img=image_exists("tabpfn_feature_importance"),
        tabpfn_importance=tabpfn_importance,
        explanations=filtered,
        missing_message=MISSING_MESSAGE,
    )


@app.route("/predictions")
def predictions():
    model_names = list(PREDICTION_FILES.keys())

    selected_model = request.args.get("model", "RandomForest")
    if selected_model not in model_names:
        selected_model = model_names[0]

    pred_df = load_prediction_file(selected_model)
    explanations = read_csv_or_none(TABLES_DIR / "prediction_explanations.csv")

    latest = None
    rows = []
    bar_chart_html = None
    recommendation = None
    latest_date = None

    if pred_df is not None and not pred_df.empty:
        latest = latest_window_predictions(pred_df)
        if not latest.empty:
            pred_date = max(latest["Date"].astype(str))
            snapshot = latest[latest["Date"].astype(str) == pred_date]

            rows = prediction_rows(latest.sort_values(["Date", "Ticker"]))

            best_idx = snapshot["Predicted_Return"].idxmax()
            best = snapshot.loc[best_idx]
            best_ticker = best["Ticker"]
            recommendation = {
                "Ticker": best_ticker,
                "Predicted_Return": float(best["Predicted_Return"]),
                "Signal": signal_for_return(best["Predicted_Return"]),
                "Date": pred_date,
                "SignalColor": SIGNAL_COLORS.get(
                    signal_for_return(best["Predicted_Return"]), "#6c757d"
                ),
                "TopFeatures": top_features_for_stock(explanations, best_ticker),
            }

            chart_df = snapshot.copy()
            chart_df["Signal"] = chart_df["Predicted_Return"].apply(signal_for_return)
            chart_df = chart_df.sort_values("Predicted_Return", ascending=True)
            fig = px.bar(
                chart_df,
                x="Predicted_Return",
                y="Ticker",
                orientation="h",
                color="Signal",
                color_discrete_map=SIGNAL_COLORS,
                title=f"Predicted 5-Day Returns - {selected_model} ({pred_date})",
                labels={"Predicted_Return": "Predicted 5-Day Return"},
            )
            fig.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            bar_chart_html = fig.to_html(
                full_html=False, include_plotlyjs="cdn"
            )

    selected_ticker = request.args.get("ticker", "AAPL")
    if selected_ticker not in TICKERS:
        selected_ticker = TICKERS[0]

    stock_detail = None
    if latest is not None and not latest.empty:
        stock_rows = latest[latest["Ticker"] == selected_ticker]
        if not stock_rows.empty:
            stock_detail = {
                "Ticker": selected_ticker,
                "Rows": prediction_rows(stock_rows.sort_values("Date")),
                "TopFeatures": top_features_for_stock(explanations, selected_ticker),
            }

    return render_template(
        "predictions.html",
        model_names=model_names,
        selected_model=selected_model,
        tickers=TICKERS,
        selected_ticker=selected_ticker,
        rows=rows,
        bar_chart_html=bar_chart_html,
        recommendation=recommendation,
        stock_detail=stock_detail,
        latest_date=latest_date,
        has_data=latest is not None,
        missing_message=MISSING_MESSAGE,
    )


@app.route("/image/<name>")
def image(name):
    path = STATIC_IMAGES.get(name)
    if path is None or not path.exists():
        return MISSING_MESSAGE, 404
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)