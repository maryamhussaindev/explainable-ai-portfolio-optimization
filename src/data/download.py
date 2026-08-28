import os
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=False)
    df.insert(0, "Ticker", ticker)
    df.index.name = "Date"
    return df.reset_index()


def download_all(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    tickers = config["data"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_path = Path(config["data"]["raw_path"])

    raw_path.mkdir(parents=True, exist_ok=True)

    all_data = []
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            df = download_stock_data(ticker, start_date, end_date)
            all_data.append(df)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_file = raw_path / "all_stocks_raw.csv"
        combined.to_csv(output_file, index=False)
        print(f"Saved raw data to {output_file}")


if __name__ == "__main__":
    download_all()
