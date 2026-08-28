from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> dict:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    train = df[df["Date"] <= train_end]
    validation = df[(df["Date"] > train_end) & (df["Date"] <= val_end)]
    test = df[df["Date"] > val_end]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def split_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    processed_path = Path(config["data"]["processed_path"])

    split_config = config.get("split", {})
    train_cutoff = split_config.get("train_end", "2022-12-31")
    val_cutoff = split_config.get("validation_end", "2023-12-31")

    input_file = processed_path / "features.csv"
    if not input_file.exists():
        print(f"Features file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    splits = temporal_split(df, train_cutoff, val_cutoff)

    for name, split_df in splits.items():
        output_file = processed_path / f"{name}.csv"
        split_df.to_csv(output_file, index=False)
        print(
            f"Saved {name} split ({len(split_df)} rows) to {output_file}"
        )


if __name__ == "__main__":
    split_all()
