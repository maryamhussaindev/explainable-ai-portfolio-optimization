from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


MODEL_FILES = {
    "RandomForest": ("experiments", "rf_predictions.csv"),
    "XGBoost": ("experiments", "xgb_predictions.csv"),
    "TabPFN": ("experiments", "tabpfn_predictions.csv"),
}

ACTUAL_COLUMN = "Actual_Return"
PREDICTED_COLUMN = "Predicted_Return"


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_model_metrics(df: pd.DataFrame) -> dict:
    actual = df[ACTUAL_COLUMN]
    predicted = df[PREDICTED_COLUMN]
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
    }


def evaluate_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    experiments_path = Path(config.get("experiments", {}).get(
        "path", "experiments"
    ))
    results_path = Path(config.get("results", {}).get(
        "path", "results"
    ))

    tables_path = results_path / "tables"
    figures_path = results_path / "figures"
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, (folder, filename) in MODEL_FILES.items():
        file_path = Path(folder) / filename
        if not file_path.exists():
            print(f"Prediction file not found for {model_name}: {file_path}")
            continue
        df = pd.read_csv(file_path)
        metrics = compute_model_metrics(df)
        rows.append({
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
        })

    if not rows:
        print("No prediction files found to evaluate.")
        return

    comparison = pd.DataFrame(rows)
    tables_file = tables_path / "model_comparison.csv"
    comparison.to_csv(tables_file, index=False)
    print(f"Saved comparison table to {tables_file}")
    print(comparison.to_string(index=False))

    melted = comparison.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Value",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.35
    x = range(len(comparison))

    for i, metric in enumerate(["MAE", "RMSE"]):
        values = comparison[metric].values
        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width=bar_width,
            label=metric,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["Model"].tolist())
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison: MAE and RMSE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(y=0.1)
    plt.tight_layout()

    figures_file = figures_path / "model_comparison.png"
    plt.savefig(figures_file, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {figures_file}")


if __name__ == "__main__":
    evaluate_all()
