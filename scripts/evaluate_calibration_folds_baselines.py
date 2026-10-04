from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.calibration import (
    CalibrationSplit,
    generate_pre_2018_calibration_folds,
)
from src.data.loader import load_ohlcv
from src.evaluation.backtest import (
    create_evaluation_environment,
    run_episode,
)
from src.evaluation.baselines import (
    Policy,
    buy_and_hold_policy,
    cash_policy,
    create_momentum_policy,
    create_moving_average_crossover_policy,
    create_random_policy,
)


WINDOW_SIZE = 30
INITIAL_BALANCE = 10_000.0
TRANSACTION_COST = 0.001
RANDOM_SEED_COUNT = 30


def create_fold_environment(
    fold: CalibrationSplit,
):
    return create_evaluation_environment(
        data=fold.validation_environment_data,
        window_size=WINDOW_SIZE,
        start_index=fold.validation_start_index,
        episode_length=(
            fold.validation_episode_length
        ),
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        # Baseline policies use BUY/SELL/HOLD semantics.
        action_mode="orders",
    )


def result_to_row(
    *,
    fold: CalibrationSplit,
    strategy: str,
    result,
    seed_count: int = 1,
) -> dict[str, object]:
    return {
        "fold": fold.name,
        "validation_start": (
            fold.validation.index.min()
        ),
        "validation_end": (
            fold.validation.index.max()
        ),
        "strategy": strategy,
        "seed_count": seed_count,
        **result.metrics.as_dict(),
    }


def summarize_across_folds(
    per_fold: pd.DataFrame,
) -> pd.DataFrame:
    buy_hold = (
        per_fold.loc[
            per_fold["strategy"]
            == "buy_and_hold",
            [
                "fold",
                "total_return",
            ],
        ]
        .set_index("fold")[
            "total_return"
        ]
    )

    rows: list[dict[str, object]] = []

    for strategy, group in per_fold.groupby(
        "strategy",
        sort=False,
    ):
        returns = group["total_return"]

        benchmark = group[
            "fold"
        ].map(buy_hold)

        if strategy == "buy_and_hold":
            beat_buy_hold_rate = np.nan
        else:
            beat_buy_hold_rate = float(
                (
                    returns.to_numpy()
                    > benchmark.to_numpy()
                ).mean()
            )

        rows.append(
            {
                "strategy": strategy,
                "fold_count": len(group),
                "total_return_mean": (
                    returns.mean()
                ),
                "total_return_median": (
                    returns.median()
                ),
                "total_return_std": (
                    returns.std(ddof=1)
                ),
                "sharpe_mean": (
                    group[
                        "sharpe_ratio"
                    ].mean()
                ),
                "maximum_drawdown_mean": (
                    group[
                        "maximum_drawdown"
                    ].mean()
                ),
                "maximum_drawdown_worst": (
                    group[
                        "maximum_drawdown"
                    ].min()
                ),
                "trade_count_mean": (
                    group[
                        "trade_count"
                    ].mean()
                ),
                "transaction_cost_mean": (
                    group[
                        "total_transaction_cost"
                    ].mean()
                ),
                "positive_fold_rate": float(
                    (returns > 0.0).mean()
                ),
                "beat_buy_hold_return_rate": (
                    beat_buy_hold_rate
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    folds = generate_pre_2018_calibration_folds(
        data,
        window_size=WINDOW_SIZE,
    )

    policies: list[
        tuple[str, Policy]
    ] = [
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

    per_fold_rows: list[
        dict[str, object]
    ] = []

    random_raw_rows: list[
        dict[str, object]
    ] = []

    print(
        "Pre-2018 calibration baselines"
    )
    print(
        "=============================="
    )

    for fold in folds:
        print()
        print(
            f"{fold.name}: "
            f"{fold.validation.index.min().date()} "
            f"-> "
            f"{fold.validation.index.max().date()}"
        )

        for strategy, policy in policies:
            result = run_episode(
                env=create_fold_environment(
                    fold
                ),
                policy=policy,
            )

            per_fold_rows.append(
                result_to_row(
                    fold=fold,
                    strategy=strategy,
                    result=result,
                )
            )

            metrics = result.metrics

            print(
                f"{strategy:<15} | "
                f"return "
                f"{metrics.total_return:>8.2%} | "
                f"Sharpe "
                f"{metrics.sharpe_ratio:>7.3f} | "
                f"MDD "
                f"{metrics.maximum_drawdown:>8.2%} | "
                f"trades "
                f"{metrics.trade_count:>3}"
            )

        random_results = []

        for seed in range(
            RANDOM_SEED_COUNT
        ):
            result = run_episode(
                env=create_fold_environment(
                    fold
                ),
                policy=create_random_policy(
                    seed
                ),
                seed=seed,
            )

            random_results.append(
                result
            )

            random_raw_rows.append(
                {
                    "fold": fold.name,
                    "seed": seed,
                    **result.metrics.as_dict(),
                }
            )

        random_frame = pd.DataFrame(
            [
                result.metrics.as_dict()
                for result
                in random_results
            ]
        )

        random_row = {
            "fold": fold.name,
            "validation_start": (
                fold.validation.index.min()
            ),
            "validation_end": (
                fold.validation.index.max()
            ),
            "strategy": "random",
            "seed_count": (
                RANDOM_SEED_COUNT
            ),
        }

        for column in random_frame.columns:
            random_row[column] = float(
                random_frame[
                    column
                ].mean()
            )

        per_fold_rows.append(
            random_row
        )

        print(
            f"{'random':<15} | "
            f"return "
            f"{random_frame['total_return'].mean():>8.2%} "
            f"± "
            f"{random_frame['total_return'].std(ddof=1):.2%}"
        )

    per_fold = pd.DataFrame(
        per_fold_rows
    )

    summary = summarize_across_folds(
        per_fold
    )

    output_directory = Path(
        "results/baselines/"
        "calibration_pre_2018"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_fold.to_csv(
        output_directory
        / "per_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_directory
        / "summary.csv",
        index=False,
    )

    pd.DataFrame(
        random_raw_rows
    ).to_csv(
        output_directory
        / "random_runs.csv",
        index=False,
    )

    print()
    print("Across-fold summary")
    print("===================")

    display_columns = [
        "strategy",
        "total_return_mean",
        "total_return_median",
        "sharpe_mean",
        "maximum_drawdown_worst",
        "positive_fold_rate",
        "beat_buy_hold_return_rate",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()