from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor


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


def run_xgboost(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])
    experiments_path = Path(config.get("experiments", {}).get(
        "path", "experiments"
    ))
    results_path = Path(config.get("results", {}).get(
        "path", "results"
    ))

    experiments_path.mkdir(parents=True, exist_ok=True)
    tables_path = results_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)

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
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    preds_df = pd.DataFrame({
        "Date": test["Date"].values,
        "Ticker": test["Ticker"].values,
        "Actual_Return": y_test.values,
        "Predicted_Return": predictions,
    })
    preds_file = experiments_path / "xgb_predictions.csv"
    preds_df.to_csv(preds_file, index=False)
    print(f"Saved predictions to {preds_file}")

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    metrics_df = pd.DataFrame({
        "Model": ["XGBoost"],
        "MAE": [mae],
        "RMSE": [rmse],
    })
    metrics_file = tables_path / "xgboost_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics to {metrics_file}")
    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")


if __name__ == "__main__":
    run_xgboost()
