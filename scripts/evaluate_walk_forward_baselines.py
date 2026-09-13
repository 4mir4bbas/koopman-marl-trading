from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_ohlcv
from src.data.walk_forward import (
    WalkForwardFold,
    generate_expanding_annual_folds,
)
from src.evaluation.backtest import (
    EpisodeResult,
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
    fold: WalkForwardFold,
):
    return create_evaluation_environment(
        data=fold.evaluation_environment_data,
        window_size=WINDOW_SIZE,
        start_index=fold.evaluation_start_index,
        episode_length=(
            fold.evaluation_episode_length
        ),
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
    )


def result_to_row(
    *,
    fold: WalkForwardFold,
    strategy: str,
    result: EpisodeResult,
    seed_count: int = 1,
) -> dict[str, object]:
    return {
        "fold": fold.name,
        "evaluation_start": (
            fold.evaluation.index.min()
        ),
        "evaluation_end": (
            fold.evaluation.index.max()
        ),
        "strategy": strategy,
        "seed_count": seed_count,
        **result.metrics.as_dict(),
    }


def random_mean_row(
    *,
    fold: WalkForwardFold,
    results: list[EpisodeResult],
) -> dict[str, object]:
    metrics = pd.DataFrame(
        [
            result.metrics.as_dict()
            for result in results
        ]
    )

    row: dict[str, object] = {
        "fold": fold.name,
        "evaluation_start": (
            fold.evaluation.index.min()
        ),
        "evaluation_end": (
            fold.evaluation.index.max()
        ),
        "strategy": "random",
        "seed_count": len(results),
    }

    for column in metrics.columns:
        row[column] = float(
            metrics[column].mean()
        )

    return row


def summarize_across_folds(
    per_fold: pd.DataFrame,
) -> pd.DataFrame:
    buy_hold_returns = (
        per_fold.loc[
            per_fold["strategy"]
            == "buy_and_hold",
            ["fold", "total_return"],
        ]
        .set_index("fold")["total_return"]
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for strategy, group in per_fold.groupby(
        "strategy",
        sort=False,
    ):
        returns = group["total_return"]

        comparison_returns = group[
            "fold"
        ].map(buy_hold_returns)

        if strategy == "buy_and_hold":
            beat_buy_hold_rate = np.nan
        else:
            beat_buy_hold_rate = float(
                (
                    returns.to_numpy()
                    > comparison_returns.to_numpy()
                ).mean()
            )

        summary_rows.append(
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
                "total_return_worst": (
                    returns.min()
                ),
                "total_return_best": (
                    returns.max()
                ),
                "annualized_return_mean": (
                    group[
                        "annualized_return"
                    ].mean()
                ),
                "sharpe_mean": (
                    group[
                        "sharpe_ratio"
                    ].mean()
                ),
                "sharpe_median": (
                    group[
                        "sharpe_ratio"
                    ].median()
                ),
                "sortino_mean": (
                    group[
                        "sortino_ratio"
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

    return pd.DataFrame(summary_rows)


def print_fold_result(
    fold_name: str,
    strategy: str,
    result: EpisodeResult,
) -> None:
    metrics = result.metrics

    print(
        f"{fold_name} | "
        f"{strategy:<15} | "
        f"return {metrics.total_return:>8.2%} | "
        f"Sharpe {metrics.sharpe_ratio:>7.3f} | "
        f"MDD {metrics.maximum_drawdown:>8.2%} | "
        f"trades {metrics.trade_count:>3}"
    )


def main() -> None:
    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    folds = generate_expanding_annual_folds(
        data=data,
        first_evaluation_year=2018,
        final_evaluation_end="2024-11-05",
        window_size=WINDOW_SIZE,
        training_start="2015-01-01",
    )

    deterministic_policies: list[
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
        "Walk-forward baseline evaluation"
    )
    print(
        "================================"
    )

    for fold in folds:
        print()
        print(
            f"Fold {fold.name}: "
            f"{fold.evaluation.index.min().date()} "
            f"-> "
            f"{fold.evaluation.index.max().date()}"
        )

        for (
            strategy_name,
            policy,
        ) in deterministic_policies:
            result = run_episode(
                env=create_fold_environment(
                    fold
                ),
                policy=policy,
            )

            per_fold_rows.append(
                result_to_row(
                    fold=fold,
                    strategy=strategy_name,
                    result=result,
                )
            )

            print_fold_result(
                fold.name,
                strategy_name,
                result,
            )

        random_results: list[
            EpisodeResult
        ] = []

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

            random_results.append(result)

            random_raw_rows.append(
                {
                    "fold": fold.name,
                    "seed": seed,
                    **result.metrics.as_dict(),
                }
            )

        per_fold_rows.append(
            random_mean_row(
                fold=fold,
                results=random_results,
            )
        )

        random_returns = np.array(
            [
                result.metrics.total_return
                for result in random_results
            ]
        )

        print(
            f"{fold.name} | "
            f"{'random':<15} | "
            f"return "
            f"{random_returns.mean():>8.2%} "
            f"± {random_returns.std(ddof=1):.2%}"
        )

    per_fold = pd.DataFrame(
        per_fold_rows
    )

    summary = summarize_across_folds(
        per_fold
    )

    output_directory = Path(
        "results/baselines/walk_forward"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_fold_path = (
        output_directory
        / "per_fold_metrics.csv"
    )

    summary_path = (
        output_directory
        / "summary.csv"
    )

    random_path = (
        output_directory
        / "random_runs.csv"
    )

    per_fold.to_csv(
        per_fold_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(
        random_raw_rows
    ).to_csv(
        random_path,
        index=False,
    )

    print()
    print("Across-fold summary")
    print("===================")

    columns = [
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
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved per-fold metrics to "
        f"{per_fold_path}"
    )

    print(
        f"Saved summary to "
        f"{summary_path}"
    )

    print(
        f"Saved random runs to "
        f"{random_path}"
    )


if __name__ == "__main__":
    main()