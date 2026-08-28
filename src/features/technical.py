import pandas as pd
import numpy as np


def daily_return(close: pd.Series) -> pd.Series:
    return close.pct_change()


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Histogram": histogram,
    })


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return close / close.shift(period) - 1


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window=window).std()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Daily_Return"] = daily_return(df["Close"])
    df["EMA_20"] = ema(df["Close"], span=20)
    df["RSI_14"] = rsi(df["Close"], period=14)
    macd_df = macd(df["Close"])
    df["MACD"] = macd_df["MACD"]
    df["MACD_Signal"] = macd_df["MACD_Signal"]
    df["MACD_Histogram"] = macd_df["MACD_Histogram"]
    df["Momentum_10"] = momentum(df["Close"], period=10)
    df["Rolling_Volatility_20"] = rolling_volatility(df["Close"], window=20)
    return df
