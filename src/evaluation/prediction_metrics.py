from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_FILES = {
    "RandomForest": "random_forest_predictions.csv",
    "XGBoost": "xgboost_predictions.csv",
    "TabPFN": "tabpfn_predictions.csv",
}

ACTUAL_COLUMN = "Actual_Return"
PREDICTED_COLUMN = "Predicted_Return"
WINDOW_COLUMN = "window_id"


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def directional_accuracy(actual, predicted) -> float:
    if len(actual) == 0:
        return float("nan")
    matches = (np.sign(actual) == np.sign(predicted)).astype(float)
    return float(matches.mean())


def compute_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        nonzero = actual != 0
        if nonzero.sum() == 0:
            mape = float("nan")
        else:
            mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
            if not np.isfinite(mape):
                mape = float("nan")

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "MAPE": mape,
        "Directional_Accuracy": directional_accuracy(actual, predicted),
    }


def evaluate_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    experiments_path = Path(config.get("experiments", {}).get(
        "path", "experiments"
    ))
    results_path = Path(config.get("results", {}).get(
        "path", "results"
    ))

    preds_dir = experiments_path / "predictions"
    tables_path = results_path / "tables"
    figures_path = results_path / "figures"
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    window_rows = []

    for model_name, filename in MODEL_FILES.items():
        file_path = preds_dir / filename
        if not file_path.exists():
            print(f"Prediction file not found for {model_name}: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            print(f"No predictions found for {model_name}")
            continue

        overall = compute_metrics(df[ACTUAL_COLUMN], df[PREDICTED_COLUMN])
        overall_rows.append({"Model": model_name, **overall})

        for window_id, window_df in df.groupby(WINDOW_COLUMN):
            metrics = compute_metrics(
                window_df[ACTUAL_COLUMN], window_df[PREDICTED_COLUMN]
            )
            window_rows.append({"Model": model_name, "window_id": window_id, **metrics})

    if not overall_rows:
        print("No prediction files found to evaluate.")
        return

    comparison = pd.DataFrame(overall_rows)
    tables_file = tables_path / "model_comparison.csv"
    comparison.to_csv(tables_file, index=False)
    print(f"Saved comparison table to {tables_file}")
    print(comparison.to_string(index=False))

    window_df = pd.DataFrame(window_rows)
    window_file = tables_path / "window_metrics.csv"
    window_df.to_csv(window_file, index=False)
    print(f"Saved window metrics to {window_file}")

    compare_cols = ["Model", "MAE", "RMSE", "R2", "MAPE", "Directional_Accuracy"]
    bars_metrics = ["MAE", "RMSE", "MAPE"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(comparison))
    width = 0.25
    colors = ["#4285f4", "#ea4335", "#34a853"]

    for i, metric in enumerate(bars_metrics):
        values = comparison[metric].values.astype(float)
        axes[0].bar(x + i * width - width, values, width=width, label=metric, color=colors[i])
        for xi, val in zip(x, values):
            label = f"{val:.4f}" if np.isfinite(val) else "n/a"
            axes[0].text(
                xi + i * width - width, val, label, ha="center", va="bottom", fontsize=7
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison["Model"].tolist())
    axes[0].set_title("Model Comparison (Error Metrics)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].margins(y=0.15)

    for i, model_name in enumerate(comparison["Model"].tolist()):
        vals = comparison.loc[
            comparison["Model"] == model_name,
            ["R2", "Directional_Accuracy"],
        ].values[0]
        labeled_vals = [
            f"{float(v):.4f}" if np.isfinite(float(v)) else "n/a" for v in vals
        ]
        axes[1].bar(
            np.arange(2) + i * 0.25,
            vals,
            width=0.25,
            label=model_name,
            color=colors[i % len(colors)],
        )
        for xi, val, lv in zip(np.arange(2) + i * 0.25, vals, labeled_vals):
            axes[1].text(xi, val, lv, ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(["R2", "Directional Accuracy"])
    axes[1].set_title("Model Comparison (Accuracy)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].margins(y=0.15)

    plt.tight_layout()
    figures_file = figures_path / "model_comparison.png"
    plt.savefig(figures_file, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {figures_file}")

    fig2, axes2 = plt.subplots(len(comparison), 1, figsize=(9, 12), sharex=True)
    if len(comparison) == 1:
        axes2 = [axes2]

    for ax, (i, model_name) in zip(axes2, enumerate(comparison["Model"].tolist())):
        model_windows = window_df[window_df["Model"] == model_name].copy()
        model_windows = model_windows.sort_values("window_id")
        x_vals = range(len(model_windows))
        ax.plot(x_vals, model_windows["MAE"].astype(float), label="MAE", color="#4285f4")
        ax.plot(x_vals, model_windows["RMSE"].astype(float), label="RMSE", color="#ea4335")
        ax.set_ylabel("Error")
        ax.set_title(f"{model_name} - Per Window Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes2[-1].set_xlabel("Window")
    plt.tight_layout()
    window_fig_file = figures_path / "window_performance.png"
    plt.savefig(window_fig_file, dpi=150)
    plt.close(fig2)
    print(f"Saved window performance chart to {window_fig_file}")


if __name__ == "__main__":
    evaluate_all()
