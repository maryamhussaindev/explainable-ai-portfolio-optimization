from pathlib import Path

import pandas as pd
import yaml

from src.features.technical import add_technical_features


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = []
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        group = add_technical_features(group)
        group["target_return_5d"] = (group["Close"].shift(-5) / group["Close"]) - 1
        group = group.dropna(subset=["target_return_5d"])
        featured.append(group)
    return pd.concat(featured, ignore_index=True)


def build_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])

    input_file = processed_path / "all_stocks_cleaned.csv"
    if not input_file.exists():
        print(f"Cleaned file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    features = build_features(df)

    output_file = processed_path / "features.csv"
    features.to_csv(output_file, index=False)
    print(f"Saved features to {output_file}")


if __name__ == "__main__":
    build_all()
