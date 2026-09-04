from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


PREDICTION_FILES = {
    "RandomForest": "random_forest_predictions.csv",
    "XGBoost": "xgboost_predictions.csv",
    "TabPFN": "tabpfn_predictions.csv",
}

PREDICTED_COLUMN = "Predicted_Return"
WINDOW_COLUMN = "window_id"
DAILY_RETURN_COLUMN = "Daily_Return"

STRATEGY_ORDER = [
    "EqualWeight",
    "Markowitz",
    "RandomForest",
    "XGBoost",
    "TabPFN",
]

STRATEGY_COLORS = {
    "EqualWeight": "#9e9e9e",
    "Markowitz": "#7b1fa2",
    "RandomForest": "#4285f4",
    "XGBoost": "#ea4335",
    "TabPFN": "#34a853",
}

PERIODS_PER_YEAR = 252 // 5


def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_daily: float,
    max_weight: float,
) -> np.ndarray:
    n = len(expected_returns)

    def neg_sharpe(w):
        port_ret = float(w @ expected_returns)
        port_vol = float(np.sqrt(w @ cov_matrix @ w))
        if port_vol < 1e-12:
            return 0.0
        return -(port_ret - risk_free_daily) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def run_backtest(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    experiments_path = Path(config.get("experiments", {}).get("path", "experiments"))
    results_path = Path(config.get("results", {}).get("path", "results"))
    processed_path = Path(config["data"]["processed_path"])

    preds_dir = experiments_path / "predictions"
    tables_path = results_path / "tables"
    figures_path = results_path / "figures"
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    features_file = processed_path / "features.csv"
    if not features_file.exists():
        print(f"Features file not found: {features_file}")
        return
    features = pd.read_csv(features_file)
    features["Date"] = pd.to_datetime(features["Date"]).dt.tz_localize(None)

    daily_returns = features.pivot_table(
        index="Date", columns="Ticker", values=DAILY_RETURN_COLUMN
    )
    tickers = daily_returns.columns.tolist()
    cov_matrix = daily_returns.cov().loc[tickers, tickers].values

    risk_free_annual = config.get("portfolio", {}).get("risk_free_rate", 0.02)
    risk_free_daily = risk_free_annual / 252
    max_weight = config.get("portfolio", {}).get("max_weight", 0.25)

    preds = {}
    for model_name, filename in PREDICTION_FILES.items():
        file_path = preds_dir / filename
        if not file_path.exists():
            print(f"Prediction file not found for {model_name}: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df["_suffix"] = df[WINDOW_COLUMN].str.split("_").str[-1]
        preds[model_name] = df

    if not preds:
        print("No prediction files found.")
        return

    reference = next(iter(preds.values()))
    suffixes = sorted(reference["_suffix"].unique(), key=lambda x: int(x))

    periods = []
    for suffix in suffixes:
        expected = {}
        dates = None
        for model_name, df in preds.items():
            period = df[df["_suffix"] == suffix]
            if dates is None:
                dates = sorted(period["Date"].unique())
            exp_series = period.groupby("Ticker")[PREDICTED_COLUMN].mean()
            expected[model_name] = exp_series.reindex(tickers).fillna(0)

        daily = daily_returns.loc[dates, tickers].fillna(0)
        periods.append({
            "suffix": suffix,
            "dates": dates,
            "expected": expected,
            "daily": daily,
        })

    if not periods:
        print("No rebalance periods found.")
        return

    weights_records = []
    strategy_returns = {name: [] for name in STRATEGY_ORDER}

    n_days_total = 0
    for period in periods:
        n = len(tickers)
        weights = {}
        weights["EqualWeight"] = {t: 1.0 / n for t in tickers}

        blended = pd.DataFrame(period["expected"]).T.mean().reindex(tickers).fillna(0)
        for strategy in ["Markowitz", "RandomForest", "XGBoost", "TabPFN"]:
            if strategy == "Markowitz":
                exp = blended.values
            else:
                exp = period["expected"][strategy].values
            opt_w = optimize_portfolio(exp, cov_matrix, risk_free_daily, max_weight)
            weights[strategy] = {t: float(w) for t, w in zip(tickers, opt_w)}

        for strategy in STRATEGY_ORDER:
            w_vec = np.array([weights[strategy][t] for t in tickers])
            window_daily = period["daily"].values @ w_vec
            strategy_returns[strategy].extend(window_daily)
            for t in tickers:
                weights_records.append({
                    "window_id": strategy,
                    "suffix": period["suffix"],
                    "Ticker": t,
                    "Strategy": strategy,
                    "Weight": weights[strategy][t],
                })

        n_days_total += len(period["dates"])

    weights_df = pd.DataFrame(weights_records)
    weights_file = experiments_path / "portfolio_weights.csv"
    weights_df.to_csv(weights_file, index=False)
    print(f"Saved portfolio weights to {weights_file}")

    rows = []
    equity_data = {}
    for strategy in STRATEGY_ORDER:
        returns = np.asarray(strategy_returns[strategy], dtype=float)
        equity = np.cumprod(1 + returns)
        cumulative_return = float(equity[-1] - 1)
        ann_vol = np.std(returns, ddof=1) * np.sqrt(252)
        sharpe = (cumulative_return / (len(returns) / 252) - risk_free_annual) / ann_vol if ann_vol > 0 else 0.0
        max_dd = compute_max_drawdown(equity)
        rows.append({
            "Strategy": strategy,
            "Cumulative Return": round(cumulative_return, 6),
            "Sharpe Ratio": round(sharpe, 6),
            "Annual Volatility": round(ann_vol, 6),
            "Max Drawdown": round(max_dd, 6),
        })
        equity_data[strategy] = equity

    metrics_df = pd.DataFrame(rows)
    metrics_file = tables_path / "portfolio_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved portfolio metrics to {metrics_file}")
    print(metrics_df.to_string(index=False))

    time_idx = np.arange(n_days_total)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for strategy in STRATEGY_ORDER:
        ax.plot(time_idx, equity_data[strategy], label=strategy, color=STRATEGY_COLORS[strategy])
    ax.set_title("Cumulative Returns by Strategy")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1 = figures_path / "cumulative_returns.png"
    plt.savefig(fig1, dpi=150)
    plt.close(fig)
    print(f"Saved cumulative returns chart to {fig1}")

    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    for strategy in STRATEGY_ORDER:
        ax2.plot(time_idx, 1 + equity_data[strategy], label=strategy, color=STRATEGY_COLORS[strategy])
    ax2.set_title("Portfolio Growth ($1 Invested)")
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2_path = figures_path / "portfolio_growth.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"Saved portfolio growth chart to {fig2_path}")


if __name__ == "__main__":
    run_backtest()
