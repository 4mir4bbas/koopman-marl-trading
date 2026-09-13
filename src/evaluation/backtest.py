from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.environments.trading_env import TradingEnv
from src.evaluation.baselines import Policy
from src.evaluation.metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)


@dataclass(frozen=True)
class EpisodeResult:
    portfolio_values: pd.Series
    metrics: PerformanceMetrics


def create_evaluation_environment(
    data: pd.DataFrame,
    *,
    window_size: int = 30,
    start_index: int | None = None,
    episode_length: int | None = None,
    initial_balance: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> TradingEnv:
    if start_index is None:
        start_index = window_size - 1

    return TradingEnv(
        data=data,
        window_size=window_size,
        episode_length=episode_length,
        random_start=False,
        fixed_start_index=start_index,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
    )


def run_episode(
    env: TradingEnv,
    policy: Policy,
    seed: int = 42,
) -> EpisodeResult:
    observation, info = env.reset(seed=seed)

    timestamps = [
        pd.Timestamp(info["timestamp"])
    ]

    portfolio_values = [
        float(info["portfolio_value"])
    ]

    terminated = False
    truncated = False
    step_number = 0

    while not terminated and not truncated:
        action = policy(
            env,
            observation,
            info,
            step_number,
        )

        (
            observation,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        timestamps.append(
            pd.Timestamp(info["timestamp"])
        )

        portfolio_values.append(
            float(info["portfolio_value"])
        )

        step_number += 1

    equity_curve = pd.Series(
        portfolio_values,
        index=pd.DatetimeIndex(timestamps),
        name="portfolio_value",
        dtype=np.float64,
    )

    metrics = calculate_performance_metrics(
        portfolio_values=equity_curve,
        trade_count=int(info["trade_count"]),
        total_transaction_cost=float(
            info["total_transaction_cost"]
        ),
        periods_per_year=365,
    )

    return EpisodeResult(
        portfolio_values=equity_curve,
        metrics=metrics,
    )