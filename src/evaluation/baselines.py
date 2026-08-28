from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.environments.trading_env import TradingEnv


Policy = Callable[
    [TradingEnv, np.ndarray, dict[str, object], int],
    int,
]


def cash_policy(
    env: TradingEnv,
    observation: np.ndarray,
    info: dict[str, object],
    step_number: int,
) -> int:
    del observation, info, step_number
    return env.HOLD


def buy_and_hold_policy(
    env: TradingEnv,
    observation: np.ndarray,
    info: dict[str, object],
    step_number: int,
) -> int:
    del observation, info

    if step_number == 0:
        return env.BUY

    return env.HOLD


def create_random_policy(
    seed: int,
) -> Policy:
    rng = np.random.default_rng(seed)

    def random_policy(
        env: TradingEnv,
        observation: np.ndarray,
        info: dict[str, object],
        step_number: int,
    ) -> int:
        del observation, info, step_number

        return int(
            rng.integers(
                low=0,
                high=env.action_space.n,
            )
        )

    return random_policy


def create_moving_average_crossover_policy(
    short_window: int = 10,
    long_window: int = 30,
) -> Policy:
    if short_window < 1:
        raise ValueError(
            "short_window must be positive."
        )

    if long_window <= short_window:
        raise ValueError(
            "long_window must be greater than "
            "short_window."
        )

    def moving_average_policy(
        env: TradingEnv,
        observation: np.ndarray,
        info: dict[str, object],
        step_number: int,
    ) -> int:
        del observation, step_number

        if long_window > env.window_size:
            raise ValueError(
                "long_window cannot exceed the "
                "environment observation window."
            )

        current_step = int(
            info["current_step"]
        )

        start_step = (
            current_step - long_window + 1
        )

        closes = env.data["close"].iloc[
            start_step:current_step + 1
        ]

        short_ma = float(
            closes.iloc[-short_window:].mean()
        )

        long_ma = float(
            closes.mean()
        )

        position = int(info["position"])

        if short_ma > long_ma:
            if position == 0:
                return env.BUY

            return env.HOLD

        if short_ma < long_ma:
            if position == 1:
                return env.SELL

            return env.HOLD

        return env.HOLD

    return moving_average_policy


def create_momentum_policy(
    lookback: int = 20,
) -> Policy:
    if lookback < 1:
        raise ValueError(
            "lookback must be positive."
        )

    def momentum_policy(
        env: TradingEnv,
        observation: np.ndarray,
        info: dict[str, object],
        step_number: int,
    ) -> int:
        del observation, step_number

        if lookback >= env.window_size:
            raise ValueError(
                "lookback must be smaller than the "
                "environment observation window."
            )

        current_step = int(
            info["current_step"]
        )

        current_close = float(
            env.data.iloc[current_step]["close"]
        )

        past_close = float(
            env.data.iloc[
                current_step - lookback
            ]["close"]
        )

        position = int(info["position"])

        if current_close > past_close:
            if position == 0:
                return env.BUY

            return env.HOLD

        if current_close < past_close:
            if position == 1:
                return env.SELL

            return env.HOLD

        return env.HOLD

    return momentum_policy