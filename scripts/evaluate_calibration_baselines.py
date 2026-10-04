from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.calibration import (
    create_calibration_split,
)
from src.data.loader import load_ohlcv
from src.evaluation.backtest import (
    create_evaluation_environment,
    run_episode,
)
from src.evaluation.baselines import (
    buy_and_hold_policy,
    cash_policy,
    create_momentum_policy,
    create_moving_average_crossover_policy,
    create_random_policy,
)


WINDOW_SIZE = 30
INITIAL_BALANCE = 10_000.0
TRANSACTION_COST = 0.001
RANDOM_SEEDS = 30


def create_env(split):
    return create_evaluation_environment(
        data=split.validation_environment_data,
        window_size=WINDOW_SIZE,
        start_index=split.validation_start_index,
        episode_length=split.validation_episode_length,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
    )


def main() -> None:
    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    split = create_calibration_split(
        data,
        window_size=WINDOW_SIZE,
        train_start="2015-01-01",
        train_end="2016-12-31",
        validation_start="2017-01-01",
        validation_end="2017-12-31",
    )

    policies = [
        (
            "cash",
            cash_policy,
        ),
        (
            "buy_and_hold",
            buy_and_hold_policy,
        ),
        (
            "ma_10_30",
            create_moving_average_crossover_policy(
                short_window=10,
                long_window=30,
            ),
        ),
        (
            "momentum_20",
            create_momentum_policy(
                lookback=20,
            ),
        ),
    ]

    rows = []

    print("2017 calibration baselines")
    print("==========================")

    for name, policy in policies:
        result = run_episode(
            env=create_env(split),
            policy=policy,
        )

        metrics = result.metrics

        rows.append(
            {
                "strategy": name,
                **metrics.as_dict(),
            }
        )

        print(
            f"{name:<15} | "
            f"return {metrics.total_return:>8.2%} | "
            f"Sharpe {metrics.sharpe_ratio:>7.3f} | "
            f"MDD {metrics.maximum_drawdown:>8.2%} | "
            f"trades {metrics.trade_count:>3}"
        )

    random_results = []

    for seed in range(RANDOM_SEEDS):
        result = run_episode(
            env=create_env(split),
            policy=create_random_policy(seed),
            seed=seed,
        )

        random_results.append(
            result.metrics.as_dict()
        )

    random_frame = pd.DataFrame(
        random_results
    )

    random_row = {
        "strategy": "random",
    }

    for column in random_frame.columns:
        random_row[column] = float(
            random_frame[column].mean()
        )

    rows.append(random_row)

    print(
        f"{'random':<15} | "
        f"return "
        f"{random_frame['total_return'].mean():>8.2%} "
        f"± "
        f"{random_frame['total_return'].std(ddof=1):.2%}"
    )

    output_directory = Path(
        "results/baselines/calibration_2017"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        output_directory / "metrics.csv",
        index=False,
    )

    random_frame.to_csv(
        output_directory
        / "random_runs.csv",
        index=False,
    )


if __name__ == "__main__":
    main()