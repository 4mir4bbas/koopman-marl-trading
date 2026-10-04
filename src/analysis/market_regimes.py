from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_market_data(
    data: pd.DataFrame,
) -> None:
    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "data must use a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Data index must be sorted chronologically."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "Data index contains duplicate timestamps."
        )

    if "close" not in data.columns:
        raise ValueError(
            "Data must contain a close column."
        )

    if (data["close"] <= 0.0).any():
        raise ValueError(
            "Close prices must be positive."
        )


def create_half_year_windows(
    data: pd.DataFrame,
    *,
    market: str,
    periods_per_year: int,
    analysis_start: str | pd.Timestamp,
    analysis_end: str | pd.Timestamp,
) -> pd.DataFrame:
    _validate_market_data(data)

    start = pd.Timestamp(
        analysis_start,
        tz="UTC",
    )

    end = pd.Timestamp(
        analysis_end,
        tz="UTC",
    )

    if start >= end:
        raise ValueError(
            "analysis_start must precede "
            "analysis_end."
        )

    rows: list[dict[str, object]] = []

    for year in range(
        start.year,
        end.year + 1,
    ):
        specifications = [
            (
                "H1",
                pd.Timestamp(
                    year=year,
                    month=1,
                    day=1,
                    tz="UTC",
                ),
                pd.Timestamp(
                    year=year,
                    month=6,
                    day=30,
                    tz="UTC",
                ),
            ),
            (
                "H2",
                pd.Timestamp(
                    year=year,
                    month=7,
                    day=1,
                    tz="UTC",
                ),
                pd.Timestamp(
                    year=year,
                    month=12,
                    day=31,
                    tz="UTC",
                ),
            ),
        ]

        for half, (
            window_start,
            window_end,
        ) in [
            (
                half,
                (window_start, window_end),
            )
            for (
                half,
                window_start,
                window_end,
            ) in specifications
        ]:
            # Only complete calendar half-years
            # are used in the comparison.
            if (
                window_start < start
                or window_end > end
            ):
                continue

            window = data.loc[
                (data.index >= window_start)
                & (data.index <= window_end)
            ].copy()

            if len(window) < 30:
                raise ValueError(
                    f"{market} {year}_{half} "
                    "contains fewer than 30 "
                    "observations."
                )

            close = window[
                "close"
            ].astype(float)

            daily_log_returns = np.log(
                close / close.shift(1)
            ).dropna()

            log_return = float(
                np.log(
                    close.iloc[-1]
                    / close.iloc[0]
                )
            )

            total_return = float(
                close.iloc[-1]
                / close.iloc[0]
                - 1.0
            )

            daily_volatility = float(
                daily_log_returns.std(
                    ddof=1
                )
            )

            annualized_volatility = (
                daily_volatility
                * np.sqrt(
                    periods_per_year
                )
            )

            # Volatility over the actual window,
            # allowing a duration-consistent
            # trend-strength score.
            window_volatility = (
                daily_volatility
                * np.sqrt(
                    len(daily_log_returns)
                )
            )

            if window_volatility > 0.0:
                trend_score = (
                    log_return
                    / window_volatility
                )
            else:
                trend_score = np.nan

            rows.append(
                {
                    "market": market,
                    "window": (
                        f"{year}_{half}"
                    ),
                    "year": year,
                    "half": half,
                    "calendar_start": (
                        window_start
                    ),
                    "calendar_end": (
                        window_end
                    ),
                    "observation_start": (
                        window.index.min()
                    ),
                    "observation_end": (
                        window.index.max()
                    ),
                    "observation_count": (
                        len(window)
                    ),
                    "start_close": float(
                        close.iloc[0]
                    ),
                    "end_close": float(
                        close.iloc[-1]
                    ),
                    "total_return": (
                        total_return
                    ),
                    "log_return": (
                        log_return
                    ),
                    "annualized_volatility": (
                        annualized_volatility
                    ),
                    "window_volatility": (
                        window_volatility
                    ),
                    "trend_score": (
                        trend_score
                    ),
                }
            )

    if not rows:
        raise ValueError(
            "No complete half-year windows "
            "were generated."
        )

    return pd.DataFrame(rows)


def summarize_regime_windows(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for market, group in windows.groupby(
        "market",
        sort=False,
    ):
        group = group.sort_values(
            [
                "year",
                "half",
            ]
        )

        positive_rate = float(
            (
                group["log_return"] > 0.0
            ).mean()
        )

        negative_rate = float(
            (
                group["log_return"] < 0.0
            ).mean()
        )

        directional_count = (
            positive_rate
            + negative_rate
        )

        if directional_count > 0.0:
            sign_balance = (
                1.0
                - abs(
                    positive_rate
                    - negative_rate
                )
                / directional_count
            )
        else:
            sign_balance = np.nan

        mean_absolute_return = float(
            group[
                "log_return"
            ].abs().mean()
        )

        if mean_absolute_return > 0.0:
            directional_bias = float(
                abs(
                    group[
                        "log_return"
                    ].mean()
                )
                / mean_absolute_return
            )
        else:
            directional_bias = 0.0

        signs = np.sign(
            group[
                "log_return"
            ].to_numpy()
        )

        signs = signs[
            signs != 0
        ]

        if len(signs) > 1:
            sign_switch_count = int(
                np.sum(
                    signs[1:]
                    != signs[:-1]
                )
            )

            sign_switch_rate = float(
                sign_switch_count
                / (len(signs) - 1)
            )
        else:
            sign_switch_count = 0
            sign_switch_rate = np.nan

        trend_scores = group[
            "trend_score"
        ].dropna()

        volatility = group[
            "annualized_volatility"
        ]

        rows.append(
            {
                "market": market,
                "window_count": len(group),
                "return_mean": (
                    group[
                        "total_return"
                    ].mean()
                ),
                "return_median": (
                    group[
                        "total_return"
                    ].median()
                ),
                "return_std": (
                    group[
                        "total_return"
                    ].std(
                        ddof=1
                    )
                ),
                "return_min": (
                    group[
                        "total_return"
                    ].min()
                ),
                "return_max": (
                    group[
                        "total_return"
                    ].max()
                ),
                "positive_window_rate": (
                    positive_rate
                ),
                "negative_window_rate": (
                    negative_rate
                ),
                # Higher is better for a
                # directionally balanced market.
                "sign_balance": (
                    sign_balance
                ),
                # 0 = little persistent drift.
                # 1 = highly one-sided drift.
                "directional_bias": (
                    directional_bias
                ),
                "sign_switch_count": (
                    sign_switch_count
                ),
                "sign_switch_rate": (
                    sign_switch_rate
                ),
                "volatility_mean": (
                    volatility.mean()
                ),
                "volatility_std": (
                    volatility.std(
                        ddof=1
                    )
                ),
                "volatility_min": (
                    volatility.min()
                ),
                "volatility_max": (
                    volatility.max()
                ),
                "trend_score_mean": (
                    trend_scores.mean()
                ),
                "trend_score_std": (
                    trend_scores.std(
                        ddof=1
                    )
                ),
                "trend_score_q10": (
                    trend_scores.quantile(
                        0.10
                    )
                ),
                "trend_score_q25": (
                    trend_scores.quantile(
                        0.25
                    )
                ),
                "trend_score_median": (
                    trend_scores.median()
                ),
                "trend_score_q75": (
                    trend_scores.quantile(
                        0.75
                    )
                ),
                "trend_score_q90": (
                    trend_scores.quantile(
                        0.90
                    )
                ),
            }
        )

    return pd.DataFrame(rows)