from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


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


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_shap_values(shap_values, feature_df, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(shap_values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values_df = pd.DataFrame(values, columns=feature_df.columns)
    values_df.to_csv(out_dir / f"{name}_shap_values.csv", index=False)
    print(f"Saved SHAP values to {out_dir / f'{name}_shap_values.csv'}")


def explain_tree_model(model, X_test_df, figures_dir: Path, shap_dir: Path, name: str) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    plt.figure()
    shap.summary_plot(shap_values, X_test_df, show=False)
    summary_file = figures_dir / f"{name}_shap_summary.png"
    plt.savefig(summary_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {summary_file}")

    plt.figure()
    shap_summary_importance = np.abs(np.asarray(shap_values)).mean(axis=0)
    shap.summary_plot(
        shap_values,
        X_test_df,
        plot_type="bar",
        show=False,
    )
    importance_file = figures_dir / f"{name}_shap_feature_importance.png"
    plt.savefig(importance_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {importance_file}")

    save_shap_values(shap_values, X_test_df, shap_dir, name)

    return shap_values, explainer


def explain_one_sample(shap_values, explainer, X_test_df, figures_dir: Path, model_name: str) -> None:
    sample_idx = 0
    sample_shap = shap_values[sample_idx]
    base_value = float(np.ravel(shap_values.base_values)[sample_idx]) if hasattr(
        shap_values, "base_values"
    ) and shap_values.base_values is not None else float(np.ravel(explainer.expected_value)[0])

    sample_explanation = pd.DataFrame({
        "Feature": X_test_df.columns,
        "SHAP_Value": np.asarray(sample_shap).flatten(),
    }).sort_values("SHAP_Value", key=abs, ascending=False)

    waterfall_path = figures_dir / f"{model_name}_shap_local_waterfall.png"

    try:
        shap.waterfall_plot(
            shap.Explanation(
                values=np.asarray(sample_shap).flatten(),
                base_values=base_value,
                data=X_test_df.iloc[sample_idx].values.flatten(),
                feature_names=X_test_df.columns.tolist(),
            ),
            max_display=10,
            show=False,
        )
        plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved local explanation to {waterfall_path} ({model_name})")
    except Exception as e:
        print(f"Could not create waterfall plot for {model_name}: {e}")


def run_shap_analysis(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])
    results_path = Path(config.get("results", {}).get("path", "results"))
    experiments_path = Path(config.get("experiments", {}).get(
        "path", "experiments"
    ))

    figures_dir = results_path / "figures"
    tables_dir = results_path / "tables"
    shap_dir = experiments_path / "shap"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed_path / "train.csv"
    test_file = processed_path / "test.csv"
    if not train_file.exists() or not test_file.exists():
        print("train.csv or test.csv not found. Run split first.")
        return

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train = train.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    test = test.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test_df = test[FEATURE_COLUMNS].reset_index(drop=True)

    results = {}

    rf_model = train_random_forest(X_train, y_train)
    results["RandomForest"] = explain_tree_model(
        rf_model, X_test_df, figures_dir, shap_dir, "random_forest"
    )

    xgb_model = train_xgboost(X_train, y_train)
    results["XGBoost"] = explain_tree_model(
        xgb_model, X_test_df, figures_dir, shap_dir, "xgboost"
    )

    for name, (shap_values, explainer) in results.items():
        if shap_values is not None:
            explain_one_sample(shap_values, explainer, X_test_df, figures_dir, name.lower())


if __name__ == "__main__":
    run_shap_analysis()
