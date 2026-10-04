from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.loader import (
    REQUIRED_COLUMNS,
    save_ohlcv,
    validate_ohlcv,
)


MARKETS = {
    "btc_usd": {
        "symbol": "BTC-USD",
        "allow_bound_repair": False,
    },
    "eur_usd": {
        "symbol": "EURUSD=X",
        "allow_bound_repair": True,
    },
    "gold": {
        "symbol": "GC=F",
        "allow_bound_repair": False,
    },
    "sp500": {
        "symbol": "^GSPC",
        "allow_bound_repair": False,
    },
}

START_DATE = "2015-01-01"
END_DATE = "2024-11-06"


def download_raw(
    symbol: str,
) -> pd.DataFrame:
    data = yf.download(
        tickers=symbol,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        repair=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            f"No data downloaded for {symbol}."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        data.columns = (
            data.columns.get_level_values(0)
        )

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    data = data.loc[
        :,
        [
            column
            for column in REQUIRED_COLUMNS
            if column in data.columns
        ],
    ].copy()

    data.index = pd.to_datetime(
        data.index,
        utc=True,
    )

    data.index.name = "timestamp"

    data = data.sort_index()

    data = data[
        ~data.index.duplicated(
            keep="first"
        )
    ]

    return data.astype("float64")


def inspect_ohlc_bounds(
    data: pd.DataFrame,
) -> dict[str, float | int]:
    required_high = data[
        [
            "open",
            "low",
            "close",
        ]
    ].max(axis=1)

    required_low = data[
        [
            "open",
            "high",
            "close",
        ]
    ].min(axis=1)

    invalid_high = (
        data["high"]
        < required_high
    )

    invalid_low = (
        data["low"]
        > required_low
    )

    high_gap_bps = (
        (
            required_high - data["high"]
        )
        / required_high
        * 10_000.0
    )

    low_gap_bps = (
        (
            data["low"] - required_low
        )
        / required_low
        * 10_000.0
    )

    return {
        "invalid_high_count": int(
            invalid_high.sum()
        ),
        "invalid_low_count": int(
            invalid_low.sum()
        ),
        "max_high_violation_bps": float(
            high_gap_bps.loc[
                invalid_high
            ].max()
            if invalid_high.any()
            else 0.0
        ),
        "max_low_violation_bps": float(
            low_gap_bps.loc[
                invalid_low
            ].max()
            if invalid_low.any()
            else 0.0
        ),
    }


def repair_ohlc_bounds(
    data: pd.DataFrame,
) -> pd.DataFrame:
    repaired = data.copy()

    row_high = repaired[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ].max(axis=1)

    row_low = repaired[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ].min(axis=1)

    repaired["high"] = row_high
    repaired["low"] = row_low

    return repaired


def main() -> None:
    output_directory = Path(
        "data/market_study"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_rows = []

    print("Market suitability data")
    print("=======================")

    for market, specification in (
        MARKETS.items()
    ):
        symbol = specification["symbol"]

        print()
        print(
            f"Downloading "
            f"{market} ({symbol})..."
        )

        data = download_raw(symbol)

        quality = inspect_ohlc_bounds(
            data
        )

        repaired = False

        has_invalid_bounds = (
            quality[
                "invalid_high_count"
            ] > 0
            or quality[
                "invalid_low_count"
            ] > 0
        )

        if has_invalid_bounds:
            if not specification[
                "allow_bound_repair"
            ]:
                raise ValueError(
                    f"{market} contains invalid "
                    "OHLC bounds and repair is "
                    "not authorized."
                )

            data = repair_ohlc_bounds(
                data
            )

            repaired = True

        validate_ohlcv(data)

        path = save_ohlcv(
            data,
            output_directory
            / f"{market}_1d.csv",
        )

        quality_rows.append(
            {
                "market": market,
                "symbol": symbol,
                "rows": len(data),
                **quality,
                "bounds_repaired": repaired,
                "research_use": (
                    "market_suitability_only"
                    if repaired
                    else "validated"
                ),
            }
        )

        print(
            f"Rows:  {len(data)}"
        )
        print(
            f"Start: "
            f"{data.index.min().date()}"
        )
        print(
            f"End:   "
            f"{data.index.max().date()}"
        )
        print(
            "Invalid high rows: "
            f"{quality['invalid_high_count']}"
        )
        print(
            "Invalid low rows:  "
            f"{quality['invalid_low_count']}"
        )
        print(
            f"Bounds repaired: {repaired}"
        )
        print(
            f"Saved: {path}"
        )

    pd.DataFrame(
        quality_rows
    ).to_csv(
        output_directory
        / "data_quality_report.csv",
        index=False,
    )


if __name__ == "__main__":
    main()