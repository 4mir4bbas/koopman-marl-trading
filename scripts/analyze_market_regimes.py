from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.market_regimes import (
    create_half_year_windows,
    summarize_regime_windows,
)
from src.data.loader import load_ohlcv


MARKETS = {
    "btc_usd": {
        "path": Path(
            "data/market_study/"
            "btc_usd_1d.csv"
        ),
        "periods_per_year": 365,
    },
    "eur_usd": {
        "path": Path(
            "data/market_study/"
            "eur_usd_1d.csv"
        ),
        "periods_per_year": 252,
    },
    "gold": {
        "path": Path(
            "data/market_study/"
            "gold_1d.csv"
        ),
        "periods_per_year": 252,
    },
    "sp500": {
        "path": Path(
            "data/market_study/"
            "sp500_1d.csv"
        ),
        "periods_per_year": 252,
    },
}

ANALYSIS_START = "2015-01-01"

# Use only complete six-month periods.
# 2024-H2 is deliberately excluded.
ANALYSIS_END = "2024-06-30"


def main() -> None:
    window_frames = []

    for (
        market,
        specification,
    ) in MARKETS.items():
        data = load_ohlcv(
            specification["path"]
        )

        windows = (
            create_half_year_windows(
                data=data,
                market=market,
                periods_per_year=(
                    specification[
                        "periods_per_year"
                    ]
                ),
                analysis_start=(
                    ANALYSIS_START
                ),
                analysis_end=(
                    ANALYSIS_END
                ),
            )
        )

        window_frames.append(
            windows
        )

    all_windows = pd.concat(
        window_frames,
        ignore_index=True,
    )

    summary = (
        summarize_regime_windows(
            all_windows
        )
    )

    output_directory = Path(
        "results/market_suitability"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_windows.to_csv(
        output_directory
        / "regime_windows.csv",
        index=False,
    )

    summary.to_csv(
        output_directory
        / "regime_summary.csv",
        index=False,
    )

    print(
        "Market regime diversity"
    )
    print(
        "======================="
    )

    display_columns = [
        "market",
        "window_count",
        "positive_window_rate",
        "negative_window_rate",
        "sign_balance",
        "directional_bias",
        "sign_switch_count",
        "sign_switch_rate",
        "volatility_mean",
        "volatility_std",
    ]

    print()
    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Trend-score distribution"
    )
    print(
        "========================"
    )

    trend_columns = [
        "market",
        "trend_score_q10",
        "trend_score_q25",
        "trend_score_median",
        "trend_score_q75",
        "trend_score_q90",
        "trend_score_std",
    ]

    print(
        summary[
            trend_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Half-year returns"
    )
    print(
        "================="
    )

    pivot = (
        all_windows.pivot(
            index="window",
            columns="market",
            values="total_return",
        )
    )

    print(
        pivot.to_string(
            float_format=lambda value: (
                f"{value: .2%}"
            )
        )
    )


if __name__ == "__main__":
    main()