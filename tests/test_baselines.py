from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.environments.trading_env import TradingEnv
from src.evaluation.baselines import (
    create_momentum_policy,
    create_moving_average_crossover_policy,
)


def create_market(
    closes: np.ndarray,
) -> pd.DataFrame:
    closes = np.asarray(
        closes,
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(
                len(closes),
                1_000.0,
            ),
        },
        index=pd.date_range(
            start="2025-01-01",
            periods=len(closes),
            freq="D",
            tz="UTC",
        ),
    )


def test_moving_average_buys_on_uptrend():
    data = create_market(
        np.linspace(
            100.0,
            150.0,
            50,
        )
    )

    env = TradingEnv(
        data=data,
        window_size=30,
        fixed_start_index=29,
        transaction_cost=0.0,
    )

    observation, info = env.reset()

    policy = (
        create_moving_average_crossover_policy(
            short_window=10,
            long_window=30,
        )
    )

    action = policy(
        env,
        observation,
        info,
        0,
    )

    assert action == env.BUY


def test_moving_average_sells_on_downtrend():
    data = create_market(
        np.linspace(
            150.0,
            100.0,
            50,
        )
    )

    env = TradingEnv(
        data=data,
        window_size=30,
        fixed_start_index=29,
        transaction_cost=0.0,
    )

    observation, info = env.reset()

    # Force a long position first.
    (
        observation,
        _,
        _,
        _,
        info,
    ) = env.step(env.BUY)

    policy = (
        create_moving_average_crossover_policy(
            short_window=10,
            long_window=30,
        )
    )

    action = policy(
        env,
        observation,
        info,
        1,
    )

    assert action == env.SELL


def test_momentum_buys_when_price_is_higher():
    data = create_market(
        np.linspace(
            100.0,
            150.0,
            50,
        )
    )

    env = TradingEnv(
        data=data,
        window_size=30,
        fixed_start_index=29,
        transaction_cost=0.0,
    )

    observation, info = env.reset()

    policy = create_momentum_policy(
        lookback=20,
    )

    action = policy(
        env,
        observation,
        info,
        0,
    )

    assert action == env.BUY


def test_baselines_cannot_use_more_history_than_agent():
    data = create_market(
        np.linspace(
            100.0,
            150.0,
            50,
        )
    )

    env = TradingEnv(
        data=data,
        window_size=30,
        fixed_start_index=29,
    )

    observation, info = env.reset()

    moving_average_policy = (
        create_moving_average_crossover_policy(
            short_window=10,
            long_window=50,
        )
    )

    with pytest.raises(
        ValueError,
        match="observation window",
    ):
        moving_average_policy(
            env,
            observation,
            info,
            0,
        )

    momentum_policy = create_momentum_policy(
        lookback=30,
    )

    with pytest.raises(
        ValueError,
        match="observation window",
    ):
        momentum_policy(
            env,
            observation,
            info,
            0,
        )