from __future__ import annotations

import numpy as np
import pandas as pd

from src.environments.trading_env import TradingEnv


def create_increasing_market() -> pd.DataFrame:
    close = np.arange(100.0, 150.0)

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full_like(close, 1_000.0),
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

    _, initial_info = env.reset(seed=42)

    _, buy_reward, _, _, buy_info = env.step(TradingEnv.BUY)
    _, hold_reward, _, _, hold_info = env.step(TradingEnv.HOLD)
    _, sell_reward, _, _, sell_info = env.step(TradingEnv.SELL)

    assert initial_info["portfolio_value"] == 10_000.0
    assert buy_info["position"] == 1
    assert buy_info["btc_holdings"] > 0.0
    assert buy_info["portfolio_value"] > 10_000.0

    assert hold_info["position"] == 1
    assert hold_info["portfolio_value"] > buy_info["portfolio_value"]
    assert hold_reward > 0.0

    assert sell_info["position"] == 0
    assert sell_info["btc_holdings"] == 0.0
    assert sell_info["cash_balance"] > 10_000.0

    print("Environment accounting test passed.")
    print(f"Buy reward:  {buy_reward:.8f}")
    print(f"Hold reward: {hold_reward:.8f}")
    print(f"Sell reward: {sell_reward:.8f}")
    print(
        f"Final portfolio value: "
        f"{sell_info['portfolio_value']:.2f}"
    )


if __name__ == "__main__":
    main()