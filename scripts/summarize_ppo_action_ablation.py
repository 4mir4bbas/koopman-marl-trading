from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

EXPERIMENT_NAME = (
    "ppo_baseline_v2_calibration"
)

FOLDS = [
    "C1",
    "C2",
    "C3",
]

SEEDS = [
    42,
    123,
    2026,
]

ACTION_MODES = [
    "orders",
    "target_position",
]

TIMESTEPS = 100_000


def load_results() -> pd.DataFrame:
    rows = []

    for fold in FOLDS:
        for action_mode in ACTION_MODES:
            for seed in SEEDS:
                run_name = (
                    f"fold_{fold}"
                    f"_action_{action_mode}"
                    f"_steps_{TIMESTEPS}"
                    f"_seed_{seed}"
                )

                result_directory = (
                    PROJECT_ROOT
                    / "results"
                    / EXPERIMENT_NAME
                    / run_name
                )

                metrics_path = (
                    result_directory
                    / "metrics.json"
                )

                if not metrics_path.exists():
                    raise FileNotFoundError(
                        metrics_path
                    )

                with metrics_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    metrics = json.load(
                        file
                    )

                rows.append(
                    {
                        "fold": fold,
                        "action_mode": (
                            action_mode
                        ),
                        "seed": seed,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def create_fold_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return (
        data
        .groupby(
            [
                "fold",
                "action_mode",
            ],
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
            sharpe_mean=(
                "sharpe_ratio",
                "mean",
            ),
            sharpe_std=(
                "sharpe_ratio",
                "std",
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


def create_action_summary(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        action_mode,
        group,
    ) in fold_summary.groupby(
        "action_mode",
        sort=False,
    ):
        rows.append(
            {
                "action_mode": (
                    action_mode
                ),
                "fold_count": len(
                    group
                ),
                "return_mean": (
                    group[
                        "total_return_mean"
                    ].mean()
                ),
                "sharpe_mean": (
                    group[
                        "sharpe_mean"
                    ].mean()
                ),
                "seed_return_std_mean": (
                    group[
                        "total_return_std"
                    ].mean()
                ),
                "seed_sharpe_std_mean": (
                    group[
                        "sharpe_std"
                    ].mean()
                ),
                "drawdown_mean": (
                    group[
                        "maximum_drawdown_mean"
                    ].mean()
                ),
                "trade_count_mean": (
                    group[
                        "trade_count_mean"
                    ].mean()
                ),
                "transaction_cost_mean": (
                    group[
                        "transaction_cost_mean"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    per_seed = load_results()

    fold_summary = create_fold_summary(
        per_seed
    )

    action_summary = (
        create_action_summary(
            fold_summary
        )
    )

    output_directory = (
        PROJECT_ROOT
        / "results"
        / EXPERIMENT_NAME
        / "action_ablation_summary"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed.to_csv(
        output_directory
        / "per_seed.csv",
        index=False,
    )

    fold_summary.to_csv(
        output_directory
        / "per_fold.csv",
        index=False,
    )

    action_summary.to_csv(
        output_directory
        / "summary.csv",
        index=False,
    )

    print(
        "PPO action-space ablation"
    )
    print(
        "========================="
    )

    print()
    print("Per-fold results")
    print("----------------")

    print(
        fold_summary.to_string(
            index=False
        )
    )

    print()
    print("Across-fold results")
    print("-------------------")

    print(
        action_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Sharpe fold wins"
    )
    print(
        "----------------"
    )

    for fold in FOLDS:
        subset = (
            fold_summary.loc[
                fold_summary["fold"]
                == fold
            ]
        )

        winner = subset.loc[
            subset[
                "sharpe_mean"
            ].idxmax()
        ]

        print(
            f"{fold}: "
            f"{winner['action_mode']} "
            f"("
            f"{winner['sharpe_mean']:.4f}"
            f")"
        )


if __name__ == "__main__":
    main()