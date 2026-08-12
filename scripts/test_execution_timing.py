from __future__ import annotations

import numpy as np
import pandas as pd

from src.environments.trading_env import TradingEnv


def main() -> None:
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
            "volume": [
                1_000.0,
            ] * 7,
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
    ) = env.step(
        TradingEnv.BUY
    )

    # The market gaps from 100 to 200 overnight.
    # A causal agent deciding after the 100 close
    # must buy at the next open of 200.
    assert np.isclose(
        info["execution_price"],
        200.0,
    )

    assert np.isclose(
        info["portfolio_value"],
        10_000.0,
    )

    assert np.isclose(
        reward,
        0.0,
    )

    print(
        "Execution timing test passed."
    )

    print(
        "The agent correctly did NOT "
        "capture the overnight gap."
    )


if __name__ == "__main__":
    main()