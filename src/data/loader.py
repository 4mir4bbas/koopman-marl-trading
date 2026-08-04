from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def download_ohlcv(
    symbol: str = "BTC-USD",
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV market data from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker symbol.
    start:
        Inclusive starting date in YYYY-MM-DD format.
    end:
        Exclusive ending date in YYYY-MM-DD format.
        If None, data is downloaded up to the latest available date.
    interval:
        Candle interval, for example "1d".

    Returns
    -------
    pd.DataFrame
        Validated OHLCV data indexed by datetime.
    """
    data = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            f"No data was downloaded for symbol={symbol!r}, "
            f"start={start!r}, end={end!r}, interval={interval!r}."
        )

    # Newer yfinance versions may return MultiIndex columns even
    # when downloading a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(column).strip().lower() for column in data.columns]

    # Adjusted Close is deliberately excluded from the first baseline.
    available_columns = [
        column for column in REQUIRED_COLUMNS if column in data.columns
    ]
    data = data.loc[:, available_columns].copy()

    validate_ohlcv(data)

    data.index = pd.to_datetime(data.index, utc=True)
    data.index.name = "timestamp"

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="first")]

    return data.astype("float64")


def validate_ohlcv(data: pd.DataFrame) -> None:
    """
    Validate the basic structural and financial consistency of OHLCV data.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required OHLCV columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("OHLCV dataset is empty.")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        null_counts = data[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(
            "OHLCV data contains missing values:\n"
            f"{null_counts[null_counts > 0]}"
        )

    price_columns = ["open", "high", "low", "close"]

    if (data[price_columns] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")

    if (data["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    invalid_high = data["high"] < data[
        ["open", "low", "close"]
    ].max(axis=1)

    if invalid_high.any():
        raise ValueError(
            f"Found {int(invalid_high.sum())} rows with invalid high prices."
        )

    invalid_low = data["low"] > data[
        ["open", "high", "close"]
    ].min(axis=1)

    if invalid_low.any():
        raise ValueError(
            f"Found {int(invalid_low.sum())} rows with invalid low prices."
        )


def save_ohlcv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save OHLCV data as a CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(path, index=True)

    return path


def load_ohlcv(input_path: str | Path) -> pd.DataFrame:
    """
    Load and validate OHLCV data from a CSV file.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    data = pd.read_csv(
        path,
        index_col="timestamp",
        parse_dates=["timestamp"],
    )

    data.columns = [column.strip().lower() for column in data.columns]
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    validate_ohlcv(data)

    return data.astype("float64")
