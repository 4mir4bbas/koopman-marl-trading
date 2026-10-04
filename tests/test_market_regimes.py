from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.market_regimes import (
    create_half_year_windows,
    summarize_regime_windows,
)


def create_market() -> pd.DataFrame:
    index = pd.date_range(
        start="2015-01-01",
        end="2016-12-31",
        freq="D",
        tz="UTC",
    )

    close = np.linspace(
        100.0,
        200.0,
        len(index),
    )

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(
                len(index),
                1_000.0,
            ),
        },
        index=index,
    )


def test_generates_complete_half_year_windows():
    windows = create_half_year_windows(
        data=create_market(),
        market="test",
        periods_per_year=365,
        analysis_start="2015-01-01",
        analysis_end="2016-12-31",
    )

    assert windows[
        "window"
    ].tolist() == [
        "2015_H1",
        "2015_H2",
        "2016_H1",
        "2016_H2",
    ]


def test_excludes_incomplete_half_year():
    windows = create_half_year_windows(
        data=create_market(),
        market="test",
        periods_per_year=365,
        analysis_start="2015-01-01",
        analysis_end="2016-06-30",
    )

    assert windows[
        "window"
    ].tolist() == [
        "2015_H1",
        "2015_H2",
        "2016_H1",
    ]


def test_trend_score_preserves_direction():
    index = pd.date_range(
        start="2015-01-01",
        end="2015-12-31",
        freq="D",
        tz="UTC",
    )

    first_half_count = (
        index.month <= 6
    ).sum()

    second_half_count = (
        index.month >= 7
    ).sum()

    first_half = np.linspace(
        100.0,
        150.0,
        first_half_count,
    )

    second_half = np.linspace(
        150.0,
        90.0,
        second_half_count,
    )

    close = np.concatenate(
        [
            first_half,
            second_half,
        ]
    )

    data = pd.DataFrame(
        {
            "close": close,
        },
        index=index,
    )

    windows = create_half_year_windows(
        data=data,
        market="test",
        periods_per_year=365,
        analysis_start="2015-01-01",
        analysis_end="2015-12-31",
    )

    assert (
        windows.loc[
            windows["window"]
            == "2015_H1",
            "trend_score",
        ].iloc[0]
        > 0.0
    )

    assert (
        windows.loc[
            windows["window"]
            == "2015_H2",
            "trend_score",
        ].iloc[0]
        < 0.0
    )


def test_summary_detects_sign_switches():
    windows = pd.DataFrame(
        {
            "market": [
                "test",
                "test",
                "test",
                "test",
            ],
            "year": [
                2015,
                2015,
                2016,
                2016,
            ],
            "half": [
                "H1",
                "H2",
                "H1",
                "H2",
            ],
            "total_return": [
                0.10,
                -0.10,
                0.10,
                -0.10,
            ],
            "log_return": [
                0.10,
                -0.10,
                0.10,
                -0.10,
            ],
            "annualized_volatility": [
                0.20,
                0.20,
                0.20,
                0.20,
            ],
            "trend_score": [
                0.50,
                -0.50,
                0.50,
                -0.50,
            ],
        }
    )

    summary = summarize_regime_windows(
        windows
    ).iloc[0]

    assert (
        summary[
            "positive_window_rate"
        ]
        == 0.5
    )

    assert (
        summary[
            "negative_window_rate"
        ]
        == 0.5
    )

    assert (
        summary[
            "sign_balance"
        ]
        == 1.0
    )

    assert (
        summary[
            "directional_bias"
        ]
        == 0.0
    )

    assert (
        summary[
            "sign_switch_count"
        ]
        == 3
    )

    assert (
        summary[
            "sign_switch_rate"
        ]
        == 1.0
    )


def test_rejects_unsorted_market_data():
    data = create_market().iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError,
        match="sorted chronologically",
    ):
        create_half_year_windows(
            data=data,
            market="test",
            periods_per_year=365,
            analysis_start="2015-01-01",
            analysis_end="2016-12-31",
        )