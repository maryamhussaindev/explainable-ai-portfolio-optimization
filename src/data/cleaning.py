from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date", "Ticker"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def clean_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])

    processed_path.mkdir(parents=True, exist_ok=True)

    raw_file = raw_path / "all_stocks_raw.csv"
    if not raw_file.exists():
        print(f"Raw file not found: {raw_file}")
        return

    df = pd.read_csv(raw_file)
    cleaned = clean_stock_data(df)

    output_file = processed_path / "all_stocks_cleaned.csv"
    cleaned.to_csv(output_file, index=False)
    print(f"Saved cleaned data to {output_file}")


if __name__ == "__main__":
    clean_all()
