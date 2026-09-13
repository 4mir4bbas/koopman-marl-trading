from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]



DEFAULT_FOLDS = [
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
]

DEFAULT_SEEDS = [
    42,
    123,
    2026,
]

EXPERIMENT_NAME = "ppo_walk_forward_v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PPO walk-forward training "
            "for multiple folds and seeds."
        )
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=DEFAULT_FOLDS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run experiments even when "
            "metrics.json already exists."
        ),
    )

    return parser.parse_args()


def metrics_path(
    fold: int,
    seed: int,
) -> Path:
    return (
        Path("results")
        / EXPERIMENT_NAME
        / f"fold_{fold}"
        / f"seed_{seed}"
        / "metrics.json"
    )


def main() -> None:
    args = parse_arguments()

    combinations = [
        (fold, seed)
        for fold in args.folds
        for seed in args.seeds
    ]

    total = len(combinations)

    print("PPO walk-forward batch")
    print("======================")
    print(f"Total requested runs: {total}")
    print()

    completed = 0
    skipped = 0

    for index, (fold, seed) in enumerate(
        combinations,
        start=1,
    ):
        output_path = metrics_path(
            fold,
            seed,
        )

        print()
        print(
            f"[{index}/{total}] "
            f"Fold {fold}, Seed {seed}"
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            print(
                "Existing result found — skipping."
            )
            skipped += 1
            continue

        command = [
            sys.executable,
            "-m",
            "scripts.train_walk_forward_ppo",
            "--fold",
            str(fold),
            "--seed",
            str(seed),
        ]

        subprocess.run(
            command,
            check=True,
            cwd=PROJECT_ROOT,
        )

        completed += 1

    print()
    print("Batch complete")
    print("==============")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Requested: {total}")


if __name__ == "__main__":
    main()