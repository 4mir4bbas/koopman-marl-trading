from __future__ import annotations

import pandas as pd
import yfinance as yf


SYMBOL = "EURUSD=X"
START = "2015-01-01"
END = "2024-11-06"


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    return data[
        ["open", "high", "low", "close", "volume"]
    ].copy()


def inspect(
    data: pd.DataFrame,
    label: str,
) -> None:
    reference_high = data[
        ["open", "low", "close"]
    ].max(axis=1)

    reference_low = data[
        ["open", "high", "close"]
    ].min(axis=1)

    invalid_high = (
        data["high"] < reference_high
    )

    invalid_low = (
        data["low"] > reference_low
    )

    print()
    print(label)
    print("=" * len(label))

    print(
        f"Rows:         {len(data)}"
    )
    print(
        f"Invalid high: {invalid_high.sum()}"
    )
    print(
        f"Invalid low:  {invalid_low.sum()}"
    )

    if invalid_high.any():
        bad = data.loc[
            invalid_high,
            ["open", "high", "low", "close"]
        ].copy()

        bad["expected_min_high"] = (
            reference_high.loc[
                invalid_high
            ]
        )

        bad["high_gap"] = (
            bad["expected_min_high"]
            - bad["high"]
        )

        bad["high_gap_bps"] = (
            bad["high_gap"]
            / bad["expected_min_high"]
            * 10_000.0
        )

        print()
        print("Invalid high examples")
        print("---------------------")
        print(
            bad.head(15).to_string()
        )

        print()
        print(
            "Maximum high violation:"
        )
        print(
            bad["high_gap"].max()
        )

        print(
            "Maximum violation (bps):"
        )
        print(
            bad["high_gap_bps"].max()
        )


def main() -> None:
    raw = yf.download(
        tickers=SYMBOL,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        repair=False,
        progress=False,
    )

    repaired = yf.download(
        tickers=SYMBOL,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        repair=True,
        progress=False,
    )

    inspect(
        prepare(raw),
        "EUR/USD raw",
    )

    inspect(
        prepare(repaired),
        "EUR/USD repaired",
    )


if __name__ == "__main__":
    main()