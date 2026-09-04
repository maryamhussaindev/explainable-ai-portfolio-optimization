from pathlib import Path

import pandas as pd
import yaml
from tabpfn import TabPFNRegressor


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


def resolve_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "auto"


def run_tabpfn(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    splits_path = Path(config.get("splits", {}).get("path", "data/splits"))
    experiments_path = Path(config.get("experiments", {}).get(
        "path", "experiments"
    ))

    preds_dir = experiments_path / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

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
        set(test[WINDOW_COLUMN].unique()).intersection(set(train[WINDOW_COLUMN].unique()))
    )

    device = resolve_device()
    print(f"Using device: {device}")

    all_preds = []
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
        predictions = model.predict(X_test)

        all_preds.append(pd.DataFrame({
            "Date": test_w["Date"].values,
            "Ticker": test_w["Ticker"].values,
            "window_id": test_w[WINDOW_COLUMN].values,
            "Actual_Return": y_test,
            "Predicted_Return": predictions,
        }))

    if not all_preds:
        print("No prediction windows found.")
        return

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_file = preds_dir / "tabpfn_predictions.csv"
    preds_df.to_csv(preds_file, index=False)
    print(f"Saved predictions for {len(window_ids)} windows to {preds_file}")


if __name__ == "__main__":
    run_tabpfn()
