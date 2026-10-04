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

MARKETS = {
    "btc_usd": {
        "path": Path(
            "data/market_study/btc_usd_1d.csv"
        ),
        "periods_per_year": 365,
    },
    "eur_usd": {
        "path": Path(
            "data/market_study/eur_usd_1d.csv"
        ),
        "periods_per_year": 252,
    },
    "gold": {
        "path": Path(
            "data/market_study/gold_1d.csv"
        ),
        "periods_per_year": 252,
    },
    "sp500": {
        "path": Path(
            "data/market_study/sp500_1d.csv"
        ),
        "periods_per_year": 252,
    },
}


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
        action_mode="orders",
    )


def result_to_row(
    *,
    market: str,
    fold: WalkForwardFold,
    strategy: str,
    result: EpisodeResult,
    seed_count: int = 1,
) -> dict[str, object]:
    return {
        "market": market,
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


def evaluate_market(
    market: str,
    data: pd.DataFrame,
    periods_per_year: int,
):
    folds = generate_expanding_annual_folds(
        data=data,
        first_evaluation_year=2016,
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

    rows = []
    random_rows = []

    print()
    print(market)
    print("=" * len(market))

    for fold in folds:
        print()
        print(
            f"{fold.name}: "
            f"{fold.evaluation.index.min().date()} "
            f"-> "
            f"{fold.evaluation.index.max().date()}"
        )

        for (
            strategy,
            policy,
        ) in deterministic_policies:
            result = run_episode(
                env=create_fold_environment(
                    fold
                ),
                policy=policy,
                periods_per_year=periods_per_year,
            )

            rows.append(
                result_to_row(
                    market=market,
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
                f"{metrics.maximum_drawdown:>8.2%}"
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
                periods_per_year=periods_per_year,
            )

            random_results.append(result)

            random_rows.append(
                {
                    "market": market,
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

        random_mean = {
            "market": market,
            "fold": fold.name,
            "evaluation_start": (
                fold.evaluation.index.min()
            ),
            "evaluation_end": (
                fold.evaluation.index.max()
            ),
            "strategy": "random",
            "seed_count": (
                RANDOM_SEED_COUNT
            ),
        }

        for column in random_frame.columns:
            random_mean[column] = float(
                random_frame[column].mean()
            )

        rows.append(random_mean)

        print(
            f"{'random':<15} | "
            f"return "
            f"{random_frame['total_return'].mean():>8.2%} "
            f"± "
            f"{random_frame['total_return'].std(ddof=1):.2%}"
        )

    return rows, random_rows


def create_market_summary(
    per_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        market,
        strategy,
    ), group in per_fold.groupby(
        [
            "market",
            "strategy",
        ],
        sort=False,
    ):
        returns = group[
            "total_return"
        ]

        rows.append(
            {
                "market": market,
                "strategy": strategy,
                "fold_count": len(group),
                "return_mean": (
                    returns.mean()
                ),
                "return_median": (
                    returns.median()
                ),
                "return_std": (
                    returns.std(ddof=1)
                ),
                "return_min": (
                    returns.min()
                ),
                "return_max": (
                    returns.max()
                ),
                "positive_fold_rate": float(
                    (returns > 0).mean()
                ),
                "negative_fold_rate": float(
                    (returns < 0).mean()
                ),
                "sharpe_mean": (
                    group[
                        "sharpe_ratio"
                    ].mean()
                ),
                "max_drawdown_worst": (
                    group[
                        "maximum_drawdown"
                    ].min()
                ),
                "trade_count_mean": (
                    group[
                        "trade_count"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    all_random_rows = []

    for market, specification in (
        MARKETS.items()
    ):
        data = load_ohlcv(
            specification["path"]
        )

        rows, random_rows = (
            evaluate_market(
                market,
                data,
                periods_per_year=(
                    specification[
                        "periods_per_year"
                    ]
                ),
            )
        )

        all_rows.extend(rows)
        all_random_rows.extend(
            random_rows
        )

    per_fold = pd.DataFrame(
        all_rows
    )

    summary = create_market_summary(
        per_fold
    )

    output_directory = Path(
        "results/market_suitability"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_fold.to_csv(
        output_directory
        / "baseline_per_fold.csv",
        index=False,
    )

    summary.to_csv(
        output_directory
        / "baseline_summary.csv",
        index=False,
    )

    pd.DataFrame(
        all_random_rows
    ).to_csv(
        output_directory
        / "random_runs.csv",
        index=False,
    )

    print()
    print(
        "Market baseline summary"
    )
    print(
        "======================="
    )

    buy_hold = summary.loc[
        summary["strategy"]
        == "buy_and_hold",
        [
            "market",
            "return_mean",
            "return_median",
            "return_std",
            "positive_fold_rate",
            "negative_fold_rate",
            "sharpe_mean",
            "max_drawdown_worst",
        ],
    ]

    print(
        buy_hold.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()