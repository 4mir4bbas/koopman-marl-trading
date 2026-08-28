
from __future__ import annotations

import numpy as np
import pandas as pd

from src.environments.trading_env import TradingEnv


def create_increasing_market() -> pd.DataFrame:
    close = np.arange(
        100.0,
        150.0,
    )

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full_like(
                close,
                1_000.0,
            ),
        },
        index=pd.date_range(
            start="2025-01-01",
            periods=len(close),
            freq="D",
            tz="UTC",
        ),
    )


def test_buy_hold_sell_accounting():
    data = create_increasing_market()

    env = TradingEnv(
        data=data,
        window_size=5,
        initial_balance=10_000.0,
        transaction_cost=0.0,
    )

    _, initial_info = env.reset(seed=42)

    initial_step = int(
        initial_info["current_step"]
    )

    (
        _,
        buy_reward,
        _,
        _,
        buy_info,
    ) = env.step(TradingEnv.BUY)

    assert (
        buy_info["execution_step"]
        == initial_step + 1
    )

    expected_execution_price = float(
        data.iloc[initial_step + 1]["open"]
    )

    assert np.isclose(
        buy_info["execution_price"],
        expected_execution_price,
    )

    assert buy_info["position"] == 1
    assert buy_info["btc_holdings"] > 0.0

    assert np.isclose(
        buy_reward,
        0.0,
    )

    assert np.isclose(
        buy_info["portfolio_value"],
        10_000.0,
    )

    (
        _,
        hold_reward,
        _,
        _,
        hold_info,
    ) = env.step(TradingEnv.HOLD)

    assert hold_info["position"] == 1

    assert (
        hold_info["portfolio_value"]
        > buy_info["portfolio_value"]
    )

    assert hold_reward > 0.0

    (
        _,
        sell_reward,
        _,
        _,
        sell_info,
    ) = env.step(TradingEnv.SELL)

    assert sell_info["position"] == 0

    assert np.isclose(
        sell_info["btc_holdings"],
        0.0,
    )

    assert (
        sell_info["cash_balance"]
        > 10_000.0
    )

    assert sell_reward > 0.0


def test_action_executes_at_next_open():
    data = pd.DataFrame(
        {
            "open": [
                100.0,
                100.0,
                100.0,
                100.0,
                200.0,
                200.0,
                200.0,
            ],
            "high": [
                101.0,
                101.0,
                101.0,
                101.0,
                201.0,
                201.0,
                201.0,
            ],
            "low": [
                99.0,
                99.0,
                99.0,
                99.0,
                199.0,
                199.0,
                199.0,
            ],
            "close": [
                100.0,
                100.0,
                100.0,
                100.0,
                200.0,
                200.0,
                200.0,
            ],
            "volume": [1_000.0] * 7,
        },
        index=pd.date_range(
            "2025-01-01",
            periods=7,
            freq="D",
            tz="UTC",
        ),
    )

    env = TradingEnv(
        data=data,
        window_size=3,
        fixed_start_index=3,
        initial_balance=10_000.0,
        transaction_cost=0.0,
    )

    _, info = env.reset()

    assert info["current_step"] == 3
    assert info["current_price"] == 100.0

    (
        _,
        reward,
        _,
        _,
        info,
    ) = env.step(TradingEnv.BUY)

    # Decision occurs after Close(t)=100,
    # so the agent must buy at Open(t+1)=200.
    assert np.isclose(
        info["execution_price"],
        200.0,
    )

    # The agent must not capture the overnight
    # jump from 100 to 200.
    assert np.isclose(
        info["portfolio_value"],
        10_000.0,
    )

    assert np.isclose(
        reward,
        0.0,
    )