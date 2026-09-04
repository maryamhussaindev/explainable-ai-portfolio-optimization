from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

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
TARGET_COLUMN = "target_return_5d"
WINDOW_COLUMN = "window_id"

N_REPEATS = 5


def resolve_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "auto"


def neg_mse_scorer(model, X, y):
    preds = model.predict(X)
    return -float(np.mean((y - preds) ** 2))


def run_tabpfn_importance(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    splits_path = Path(config.get("splits", {}).get("path", "data/splits"))
    results_path = Path(config.get("results", {}).get("path", "results"))

    tables_dir = results_path / "tables"
    figures_dir = results_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_file = splits_path / "train.csv"
    test_file = splits_path / "test.csv"
    if not train_file.exists() or not test_file.exists():
        print(f"train.csv or test.csv not found in {splits_path}. Run split first.")
        return

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train = train.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    test = test.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    window_ids = sorted(
        set(test[WINDOW_COLUMN].unique()).intersection(
            set(train[WINDOW_COLUMN].unique())
        )
    )

    from tabpfn import TabPFNRegressor

    device = resolve_device()
    print(f"Using device: {device}")

    mean_importances = []
    std_importances = []
    for window_id in window_ids:
        train_w = train[train[WINDOW_COLUMN] == window_id]
        test_w = test[test[WINDOW_COLUMN] == window_id]
        if len(train_w) == 0 or len(test_w) == 0:
            continue

        X_train = train_w[FEATURE_COLUMNS].values
        y_train = train_w[TARGET_COLUMN].values
        X_test = test_w[FEATURE_COLUMNS].values
        y_test = test_w[TARGET_COLUMN].values

        model = TabPFNRegressor(
            random_state=42,
            device=device,
        )
        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=N_REPEATS,
            random_state=42,
            scoring=neg_mse_scorer,
            n_jobs=1,
        )
        mean_importances.append(result.importances_mean)
        std_importances.append(result.importances_std)

    if not mean_importances:
        print("No prediction windows found.")
        return

    importance_mean = np.mean(mean_importances, axis=0)
    importance_std = np.mean(std_importances, axis=0)

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance_Mean": importance_mean,
        "Importance_Std": importance_std,
    }).sort_values("Importance_Mean", ascending=False)

    imp_file = tables_dir / "tabpfn_permutation_importance.csv"
    importance_df.to_csv(imp_file, index=False)
    print(f"Saved permutation importance to {imp_file}")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(9, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance_Mean"],
        xerr=importance_df["Importance_Std"],
        color="#34a853",
        alpha=0.85,
    )
    plt.gca().invert_yaxis()
    plt.title("TabPFN Feature Importance (Permutation, Rolling Test Set)")
    plt.xlabel("Mean Importance (Permutation - Negative MSE)")
    plt.tight_layout()
    fig_path = figures_dir / "tabpfn_feature_importance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance chart to {fig_path}")


if __name__ == "__main__":
    run_tabpfn_importance()
