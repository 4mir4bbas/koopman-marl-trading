from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPERIMENT_NAME = "ppo_walk_forward_v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize PPO walk-forward results "
            "for one evaluation fold."
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
    )

    return parser.parse_args()


def load_seed_metrics(
    *,
    fold: int,
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for seed in seeds:
        result_directory = (
            Path("results")
            / EXPERIMENT_NAME
            / f"fold_{fold}"
            / f"seed_{seed}"
        )

        metrics_path = (
            result_directory / "metrics.json"
        )

        metadata_path = (
            result_directory
            / "run_metadata.json"
        )

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics for "
                f"fold {fold}, seed {seed}: "
                f"{metrics_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata for "
                f"fold {fold}, seed {seed}: "
                f"{metadata_path}"
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

        if str(metadata["fold"]) != str(fold):
            raise ValueError(
                "Metadata fold does not match "
                "the requested fold."
            )

        if metadata[
            "outer_fold_used_during_training"
        ]:
            raise ValueError(
                "Outer evaluation fold was marked "
                "as used during training."
            )

        rows.append(
            {
                "fold": fold,
                "seed": seed,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "calmar_ratio",
        "trade_count",
        "total_transaction_cost",
    ]

    rows: list[dict[str, object]] = []

    for metric in metric_columns:
        values = per_seed[metric]

        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )

    return pd.DataFrame(rows)


def load_fold_baselines(
    fold: int,
) -> pd.DataFrame | None:
    path = Path(
        "results/baselines/walk_forward/"
        "per_fold_metrics.csv"
    )

    if not path.exists():
        return None

    baselines = pd.read_csv(path)

    return baselines.loc[
        baselines["fold"].astype(str)
        == str(fold)
    ].copy()


def main() -> None:
    args = parse_arguments()

    per_seed = load_seed_metrics(
        fold=args.fold,
        seeds=args.seeds,
    )

    summary = summarize_metrics(
        per_seed
    )

    output_directory = (
        Path("results")
        / EXPERIMENT_NAME
        / f"fold_{args.fold}"
        / "summary"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed_path = (
        output_directory
        / "per_seed_metrics.csv"
    )

    summary_path = (
        output_directory
        / "summary.csv"
    )

    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"PPO seed summary — Fold {args.fold}"
    )
    print(
        "================================"
    )

    display_columns = [
        "seed",
        "total_return",
        "sharpe_ratio",
        "maximum_drawdown",
        "trade_count",
        "total_transaction_cost",
    ]

    print(
        per_seed[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    for metric in [
        "total_return",
        "sharpe_ratio",
        "maximum_drawdown",
        "trade_count",
        "total_transaction_cost",
    ]:
        row = summary.loc[
            summary["metric"] == metric
        ].iloc[0]

        print(
            f"{metric:<24} "
            f"mean={row['mean']:.6f} "
            f"std={row['std']:.6f}"
        )

    baselines = load_fold_baselines(
        args.fold
    )

    if baselines is not None:
        print()
        print("Financial baselines")
        print("===================")

        columns = [
            "strategy",
            "total_return",
            "sharpe_ratio",
            "maximum_drawdown",
        ]

        print(
            baselines[
                columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Saved per-seed metrics to "
        f"{per_seed_path}"
    )

    print(
        f"Saved summary to "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()