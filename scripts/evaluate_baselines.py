from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_ohlcv
from src.data.split import chronological_split
from src.environments.trading_env import TradingEnv
from src.evaluation.metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)


Policy = Callable[
    [TradingEnv, np.ndarray, dict[str, object], int],
    int,
]


@dataclass(frozen=True)
class EpisodeResult:
    portfolio_values: pd.Series
    metrics: PerformanceMetrics


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


def run_episode(
    env: TradingEnv,
    policy: Policy,
    seed: int = 42,
) -> EpisodeResult:
    observation, info = env.reset(seed=seed)

    timestamps = [pd.Timestamp(info["timestamp"])]
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

        timestamps.append(pd.Timestamp(info["timestamp"]))
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


def create_environment(
    data: pd.DataFrame,
) -> TradingEnv:
    return TradingEnv(
        data=data,
        window_size=30,
        initial_balance=10_000.0,
        transaction_cost=0.001,
    )


def summarize_random_runs(
    results: list[EpisodeResult],
) -> dict[str, float]:
    total_returns = np.array(
        [
            result.metrics.total_return
            for result in results
        ],
        dtype=np.float64,
    )

    sharpe_ratios = np.array(
        [
            result.metrics.sharpe_ratio
            for result in results
        ],
        dtype=np.float64,
    )

    drawdowns = np.array(
        [
            result.metrics.maximum_drawdown
            for result in results
        ],
        dtype=np.float64,
    )

    return {
        "random_total_return_mean": float(
            np.nanmean(total_returns)
        ),
        "random_total_return_std": float(
            np.nanstd(total_returns, ddof=1)
        ),
        "random_sharpe_mean": float(
            np.nanmean(sharpe_ratios)
        ),
        "random_max_drawdown_mean": float(
            np.nanmean(drawdowns)
        ),
    }


def print_metrics(
    name: str,
    metrics: PerformanceMetrics,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Initial value:       {metrics.initial_value:,.2f}")
    print(f"Final value:         {metrics.final_value:,.2f}")
    print(f"Total return:        {metrics.total_return:.2%}")
    print(
        f"Annualized return:   "
        f"{metrics.annualized_return:.2%}"
    )
    print(
        f"Annualized vol.:      "
        f"{metrics.annualized_volatility:.2%}"
    )
    print(f"Sharpe ratio:        {metrics.sharpe_ratio:.4f}")
    print(f"Sortino ratio:       {metrics.sortino_ratio:.4f}")
    print(
        f"Maximum drawdown:    "
        f"{metrics.maximum_drawdown:.2%}"
    )
    print(f"Calmar ratio:        {metrics.calmar_ratio:.4f}")
    print(f"Trade count:         {metrics.trade_count}")
    print(
        f"Transaction costs:   "
        f"{metrics.total_transaction_cost:,.2f}"
    )


def save_equity_curves(
    output_path: Path,
    curves: dict[str, pd.Series],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.concat(
        curves,
        axis=1,
    )

    dataframe.to_csv(
        output_path,
        index=True,
    )


def main() -> None:
    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    splits = chronological_split(
        data=data,
        train_ratio=0.70,
        validation_ratio=0.15,
        minimum_split_size=32,
    )

    print("Chronological split")
    print("-------------------")

    for name, split_data in (
        ("Train", splits.train),
        ("Validation", splits.validation),
        ("Test", splits.test),
    ):
        print(
            f"{name:<11} "
            f"{len(split_data):>5} rows | "
            f"{split_data.index.min()} -> "
            f"{split_data.index.max()}"
        )

    # At this stage, baseline development is performed on the
    # validation set. The test set remains untouched.
    evaluation_data = splits.validation

    cash_result = run_episode(
        env=create_environment(evaluation_data),
        policy=cash_policy,
    )

    buy_hold_result = run_episode(
        env=create_environment(evaluation_data),
        policy=buy_and_hold_policy,
    )

    random_results: list[EpisodeResult] = []

    for seed in range(30):
        result = run_episode(
            env=create_environment(evaluation_data),
            policy=create_random_policy(seed),
            seed=seed,
        )
        random_results.append(result)

    print_metrics(
        "Cash baseline",
        cash_result.metrics,
    )

    print_metrics(
        "Buy-and-hold baseline",
        buy_hold_result.metrics,
    )

    random_summary = summarize_random_runs(
        random_results
    )

    print("\nRandom baseline: 30 runs")
    print("------------------------")

    for metric_name, metric_value in random_summary.items():
        print(f"{metric_name}: {metric_value:.6f}")

    representative_random_result = random_results[0]

    save_equity_curves(
        output_path=Path(
            "results/baselines/"
            "validation_equity_curves.csv"
        ),
        curves={
            "cash": cash_result.portfolio_values,
            "buy_and_hold": (
                buy_hold_result.portfolio_values
            ),
            "random_seed_0": (
                representative_random_result.portfolio_values
            ),
        },
    )

    print(
        "\nSaved equity curves to "
        "results/baselines/"
        "validation_equity_curves.csv"
    )


if __name__ == "__main__":
    main()