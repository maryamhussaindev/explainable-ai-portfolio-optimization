from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def optimize_portfolio(
    expected_returns,
    cov_matrix,
    risk_free_daily: float,
    max_weight: float = 0.20,
) -> np.ndarray:
    n = len(expected_returns)

    def neg_sharpe(w):
        port_ret = w @ expected_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
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


def run_markowitz(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])
    experiments_path = Path(config.get("experiments", {}).get("path", "experiments"))
    results_path = Path(config.get("results", {}).get("path", "results"))

    experiments_path.mkdir(parents=True, exist_ok=True)

    test_file = processed_path / "test.csv"
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    test = pd.read_csv(test_file)

    returns_pivot = test.pivot_table(
        index="Date", columns="Ticker", values="Daily_Return"
    )
    cov_matrix = returns_pivot.cov()

    risk_free_annual = config.get("portfolio", {}).get("risk_free_rate", 0.02)
    risk_free_daily = risk_free_annual / 252
    max_weight = 0.20

    model_files = {
        "RandomForest": experiments_path / "rf_predictions.csv",
        "XGBoost": experiments_path / "xgb_predictions.csv",
        "TabPFN": experiments_path / "tabpfn_predictions.csv",
    }

    all_weights = {}
    for model_name, pred_file in model_files.items():
        if not pred_file.exists():
            print(f"Predictions not found for {model_name}: {pred_file}")
            continue

        preds = pd.read_csv(pred_file)
        expected_returns = preds.groupby("Ticker")["Predicted_Return"].mean()

        tickers = cov_matrix.index.tolist()
        expected_returns = expected_returns.reindex(tickers).fillna(0)
        cov_aligned = cov_matrix.loc[tickers, tickers].values

        weights = optimize_portfolio(
            expected_returns.values, cov_aligned, risk_free_daily, max_weight
        )
        all_weights[model_name] = pd.Series(weights, index=tickers)

        print(f"\n{model_name} optimal weights:")
        for ticker, w in zip(tickers, weights):
            print(f"  {ticker}: {w:.4f}")

    if not all_weights:
        print("No models evaluated.")
        return

    weights_df = pd.DataFrame(all_weights)
    weights_file = experiments_path / "portfolio_weights.csv"
    weights_df.to_csv(weights_file)
    print(f"\nSaved portfolio weights to {weights_file}")


if __name__ == "__main__":
    run_markowitz()
