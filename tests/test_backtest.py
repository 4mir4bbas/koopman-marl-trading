from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    create_evaluation_environment,
    run_episode,
)
from src.evaluation.baselines import (
    buy_and_hold_policy,
    cash_policy,
)


def create_market(
    length: int = 40,
) -> pd.DataFrame:
    close = np.linspace(
        100.0,
        140.0,
        length,
    )

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(
                length,
                1_000.0,
            ),
        },
        index=pd.date_range(
            start="2025-01-01",
            periods=length,
            freq="D",
            tz="UTC",
        ),
    )


def test_backtest_scores_only_requested_episode():
    data = create_market()

    start_index = 29
    episode_length = 5

    env = create_evaluation_environment(
        data,
        window_size=30,
        start_index=start_index,
        episode_length=episode_length,
        transaction_cost=0.0,
    )

    result = run_episode(
        env=env,
        policy=cash_policy,
    )

    assert (
        result.portfolio_values.index[0]
        == data.index[start_index]
    )

    assert (
        result.portfolio_values.index[-1]
        == data.index[
            start_index + episode_length
        ]
    )

    assert (
        len(result.portfolio_values)
        == episode_length + 1
    )


def test_buy_and_hold_gains_on_increasing_market():
    data = create_market()

    env = create_evaluation_environment(
        data,
        window_size=30,
        start_index=29,
        episode_length=10,
        transaction_cost=0.0,
    )

    result = run_episode(
        env=env,
        policy=buy_and_hold_policy,
    )

    assert result.metrics.total_return > 0.0
    assert result.metrics.trade_count == 1