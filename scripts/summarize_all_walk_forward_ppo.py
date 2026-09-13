from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT_NAME = "ppo_walk_forward_v1"

FOLDS = [
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
]

SEEDS = [
    42,
    123,
    2026,
]


def load_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for fold in FOLDS:
        for seed in SEEDS:
            result_directory = (
                Path("results")
                / EXPERIMENT_NAME
                / f"fold_{fold}"
                / f"seed_{seed}"
            )

            metrics_path = (
                result_directory
                / "metrics.json"
            )

            metadata_path = (
                result_directory
                / "run_metadata.json"
            )

            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"Missing {metrics_path}"
                )

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Missing {metadata_path}"
                )

            with metrics_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metrics = json.load(file)

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

            if metadata[
                "outer_fold_used_during_training"
            ]:
                raise ValueError(
                    f"Fold {fold}, seed {seed} "
                    "was marked as using outer "
                    "evaluation data."
                )

            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def create_fold_summary(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    return (
        per_seed
        .groupby(
            "fold",
            as_index=False,
        )
        .agg(
            total_return_mean=(
                "total_return",
                "mean",
            ),
            total_return_std=(
                "total_return",
                "std",
            ),
            annualized_return_mean=(
                "annualized_return",
                "mean",
            ),
            sharpe_mean=(
                "sharpe_ratio",
                "mean",
            ),
            sharpe_std=(
                "sharpe_ratio",
                "std",
            ),
            sortino_mean=(
                "sortino_ratio",
                "mean",
            ),
            maximum_drawdown_mean=(
                "maximum_drawdown",
                "mean",
            ),
            trade_count_mean=(
                "trade_count",
                "mean",
            ),
            transaction_cost_mean=(
                "total_transaction_cost",
                "mean",
            ),
        )
    )


def add_baseline_comparison(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline_path = Path(
        "results/baselines/walk_forward/"
        "per_fold_metrics.csv"
    )

    baselines = pd.read_csv(
        baseline_path
    )

    selected = baselines.loc[
        baselines["strategy"].isin(
            [
                "buy_and_hold",
                "momentum_20",
            ]
        ),
        [
            "fold",
            "strategy",
            "total_return",
            "sharpe_ratio",
        ],
    ].copy()

    selected["fold"] = (
        selected["fold"].astype(int)
    )

    pivot = selected.pivot(
        index="fold",
        columns="strategy",
        values=[
            "total_return",
            "sharpe_ratio",
        ],
    )

    pivot.columns = [
        f"{metric}_{strategy}"
        for metric, strategy
        in pivot.columns
    ]

    pivot = pivot.reset_index()

    result = fold_summary.merge(
        pivot,
        on="fold",
        how="left",
        validate="one_to_one",
    )

    result["return_excess_vs_buy_hold"] = (
        result["total_return_mean"]
        - result[
            "total_return_buy_and_hold"
        ]
    )

    result["return_excess_vs_momentum"] = (
        result["total_return_mean"]
        - result[
            "total_return_momentum_20"
        ]
    )

    result["beats_buy_hold"] = (
        result["return_excess_vs_buy_hold"]
        > 0.0
    )

    result["beats_momentum"] = (
        result["return_excess_vs_momentum"]
        > 0.0
    )

    return result


def main() -> None:
    per_seed = load_results()

    fold_summary = create_fold_summary(
        per_seed
    )

    comparison = add_baseline_comparison(
        fold_summary
    )

    output_directory = (
        Path("results")
        / EXPERIMENT_NAME
        / "summary"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed.to_csv(
        output_directory
        / "all_per_seed_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        output_directory
        / "per_fold_summary.csv",
        index=False,
    )

    print("PPO walk-forward summary")
    print("========================")

    display_columns = [
        "fold",
        "total_return_mean",
        "total_return_std",
        "sharpe_mean",
        "maximum_drawdown_mean",
        "trade_count_mean",
        "return_excess_vs_buy_hold",
        "return_excess_vs_momentum",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Across-fold PPO statistics")
    print("==========================")

    print(
        "Mean fold return: "
        f"{comparison['total_return_mean'].mean():.2%}"
    )

    print(
        "Median fold return: "
        f"{comparison['total_return_mean'].median():.2%}"
    )

    print(
        "Mean fold Sharpe: "
        f"{comparison['sharpe_mean'].mean():.4f}"
    )

    print(
        "Mean maximum drawdown: "
        f"{comparison['maximum_drawdown_mean'].mean():.2%}"
    )

    print(
        "Beats Buy & Hold: "
        f"{comparison['beats_buy_hold'].sum()}"
        f"/{len(comparison)} folds"
    )

    print(
        "Beats Momentum 20: "
        f"{comparison['beats_momentum'].sum()}"
        f"/{len(comparison)} folds"
    )


if __name__ == "__main__":
    main()
    