from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from tabpfn import TabPFNRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


FEATURE_COLUMNS = [
    "Daily_Return",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Momentum_10",
    "Rolling_Volatility_20",
]
TARGET_COLUMN = "Target_Return"


def run_tabpfn_importance(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])
    results_path = Path(config.get("results", {}).get("path", "results"))

    tables_dir = results_path / "tables"
    figures_dir = results_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed_path / "train.csv"
    test_file = processed_path / "test.csv"
    if not train_file.exists() or not test_file.exists():
        print("train.csv or test.csv not found. Run split first.")
        return

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train = train.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    test = test.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X_train = train[FEATURE_COLUMNS].values
    y_train = train[TARGET_COLUMN].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test[TARGET_COLUMN].values

    model = TabPFNRegressor(random_state=42)
    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    csv_path = tables_dir / "tabpfn_permutation_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved permutation importance to {csv_path}")

    sorted_df = importance_df.sort_values("Importance_Mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        sorted_df["Feature"],
        sorted_df["Importance_Mean"],
        xerr=sorted_df["Importance_Std"],
        color="#4a90d9",
        edgecolor="white",
        height=0.6,
    )
    ax.set_xlabel("Mean Accuracy Decrease")
    ax.set_title("TabPFN Feature Importance (Permutation)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fig_path = figures_dir / "tabpfn_feature_importance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {fig_path}")


if __name__ == "__main__":
    run_tabpfn_importance()
