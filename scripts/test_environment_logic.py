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


def main() -> None:
    data = create_increasing_market()

    env = TradingEnv(
        data=data,
        window_size=5,
        initial_balance=10_000.0,
        transaction_cost=0.0,
    )

    _, initial_info = env.reset(
        seed=42
    )

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
        data.iloc[
            initial_step + 1
        ]["open"]
    )

    assert np.isclose(
        buy_info["execution_price"],
        expected_execution_price,
    )

    assert buy_info["position"] == 1
    assert buy_info["btc_holdings"] > 0.0

    # Open and Close of the execution candle are equal
    # in this synthetic market, so the first buy step
    # produces no market return when transaction costs
    # are zero.
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

    # The position is still held overnight before the sell
    # executes at the next open. Because the synthetic market
    # rises from the previous close to the next open, the agent
    # correctly earns that overnight return before exiting.
    assert sell_reward > 0.0

    print(
        "Causal execution accounting "
        "test passed."
    )

    print(
        f"Buy executed at: "
        f"{buy_info['execution_price']:.2f}"
    )

    print(
        f"Buy reward:  "
        f"{buy_reward:.8f}"
    )

    print(
        f"Hold reward: "
        f"{hold_reward:.8f}"
    )

    print(
        f"Sell reward: "
        f"{sell_reward:.8f}"
    )

    print(
        "Final portfolio value: "
        f"{sell_info['portfolio_value']:.2f}"
    )


if __name__ == "__main__":
    main()