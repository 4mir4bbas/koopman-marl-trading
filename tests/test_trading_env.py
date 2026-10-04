
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

def create_flat_market(
    length: int = 20,
) -> pd.DataFrame:
    close = np.full(
        length,
        100.0,
        dtype=np.float64,
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

def test_target_position_action_space_has_two_actions():
    env = TradingEnv(
        data=create_flat_market(),
        window_size=5,
        action_mode="target_position",
    )

    assert env.action_space.n == 2
    assert env.CASH == 0
    assert env.LONG == 1


def test_target_position_actions_change_only_when_needed():
    env = TradingEnv(
        data=create_flat_market(),
        window_size=5,
        fixed_start_index=4,
        initial_balance=10_000.0,
        transaction_cost=0.001,
        action_mode="target_position",
    )

    _, info = env.reset()

    # Already cash -> target cash -> no trade.
    _, _, _, _, info = env.step(
        env.CASH
    )

    assert info["position"] == 0
    assert info["trade_count"] == 0
    assert np.isclose(
        info["total_transaction_cost"],
        0.0,
    )

    # Cash -> target long -> buy.
    _, _, _, _, info = env.step(
        env.LONG
    )

    assert info["position"] == 1
    assert info["trade_count"] == 1
    assert np.isclose(
        info["total_transaction_cost"],
        10.0,
    )

    # Already long -> target long -> no trade.
    _, _, _, _, info = env.step(
        env.LONG
    )

    assert info["position"] == 1
    assert info["trade_count"] == 1
    assert np.isclose(
        info["total_transaction_cost"],
        10.0,
    )

    # Long -> target cash -> sell.
    _, _, _, _, info = env.step(
        env.CASH
    )

    assert info["position"] == 0
    assert info["trade_count"] == 2

    # Buy cost = 10.00
    # Sell cost = 9.99
    assert np.isclose(
        info["total_transaction_cost"],
        19.99,
    )

    assert np.isclose(
        info["cash_balance"],
        9_980.01,
    )

def test_target_position_executes_at_next_open():
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
        action_mode="target_position",
    )

    _, info = env.reset()

    assert info["current_price"] == 100.0

    (
        _,
        reward,
        _,
        _,
        info,
    ) = env.step(env.LONG)

    assert np.isclose(
        info["execution_price"],
        200.0,
    )

    # The agent must not capture the
    # Close(t)=100 -> Open(t+1)=200 gap.
    assert np.isclose(
        info["portfolio_value"],
        10_000.0,
    )

    assert np.isclose(
        reward,
        0.0,
    )

def test_rejects_unknown_action_mode():
    try:
        TradingEnv(
            data=create_flat_market(),
            window_size=5,
            action_mode="invalid",
        )
    except ValueError as error:
        assert "action_mode" in str(error)
    else:
        raise AssertionError(
            "Expected invalid action mode "
            "to raise ValueError."
        )