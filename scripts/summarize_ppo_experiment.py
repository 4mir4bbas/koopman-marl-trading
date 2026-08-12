from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_NAME = "ppo_baseline_v1"

SEEDS = (
    42,
    123,
    2026,
)


def load_metrics(
    seed: int,
) -> dict[str, float]:
    path = (
        Path("results")
        / EXPERIMENT_NAME
        / f"seed_{seed}"
        / "validation"
        / "metrics.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing metrics: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    rows = []

    for seed in SEEDS:
        metrics = load_metrics(seed)

        rows.append(
            {
                "seed": seed,
                **metrics,
            }
        )

    dataframe = pd.DataFrame(rows)

    summary_metrics = [
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

    summary_rows = []

    for metric in summary_metrics:
        values = dataframe[
            metric
        ].to_numpy(
            dtype=np.float64
        )

        summary_rows.append(
            {
                "metric": metric,
                "mean": float(
                    np.nanmean(values)
                ),
                "std": float(
                    np.nanstd(
                        values,
                        ddof=1,
                    )
                ),
                "min": float(
                    np.nanmin(values)
                ),
                "max": float(
                    np.nanmax(values)
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    output_dir = (
        Path("results")
        / EXPERIMENT_NAME
        / "summary"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_dir
        / "per_seed_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "summary_metrics.csv",
        index=False,
    )

    print()
    print("Per-seed PPO results")
    print("--------------------")
    print(
        dataframe[
            [
                "seed",
                "total_return",
                "sharpe_ratio",
                "maximum_drawdown",
                "trade_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Aggregate PPO results")
    print("---------------------")
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()