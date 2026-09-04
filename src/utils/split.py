from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


TRAIN_WINDOW = 252
PRED_WINDOW = 5
STRIDE = 5


def _rolling_windows_for_ticker(
    ticker_group: pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    pred_window: int = PRED_WINDOW,
    stride: int = STRIDE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_group = (
        ticker_group.sort_values("Date").reset_index(drop=True)
    )
    n = len(ticker_group)

    train_frames = []
    test_frames = []

    start = train_window
    while start + pred_window <= n:
        window_id = f"{ticker_group['Ticker'].iloc[0]}_{start}"

        train_rows = ticker_group.iloc[start - train_window:start].copy()
        train_rows["window_id"] = window_id

        test_rows = ticker_group.iloc[start:start + pred_window].copy()
        test_rows["window_id"] = window_id

        train_frames.append(train_rows)
        test_frames.append(test_rows)

        start += stride

    if not train_frames:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    return pd.concat(train_frames, ignore_index=True), pd.concat(
        test_frames, ignore_index=True
    )


def rolling_split(
    df: pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    pred_window: int = PRED_WINDOW,
    stride: int = STRIDE,
) -> dict:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    all_train = []
    all_test = []
    for _, ticker_group in df.groupby("Ticker", sort=False):
        train_rows, test_rows = _rolling_windows_for_ticker(
            ticker_group, train_window, pred_window, stride
        )
        all_train.append(train_rows)
        all_test.append(test_rows)

    return {
        "train": pd.concat(all_train, ignore_index=True),
        "test": pd.concat(all_test, ignore_index=True),
    }


def split_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])

    splits_path = Path(config.get("splits", {}).get("path", "data/splits"))
    splits_path.mkdir(parents=True, exist_ok=True)

    input_file = processed_path / "features.csv"
    if not input_file.exists():
        print(f"Features file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    splits = rolling_split(df)

    for name, split_df in splits.items():
        output_file = splits_path / f"{name}.csv"
        split_df.to_csv(output_file, index=False)
        print(
            f"Saved {name} split ({len(split_df)} rows) to {output_file}"
        )


if __name__ == "__main__":
    split_all()
