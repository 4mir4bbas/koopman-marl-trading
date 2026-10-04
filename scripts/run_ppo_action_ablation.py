from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

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

EXPERIMENT_NAME = (
    "ppo_baseline_v2_calibration"
)


def metrics_path(
    fold: str,
    action_mode: str,
    seed: int,
) -> Path:
    run_name = (
        f"fold_{fold}"
        f"_action_{action_mode}"
        f"_steps_{TIMESTEPS}"
        f"_seed_{seed}"
    )

    return (
        PROJECT_ROOT
        / "results"
        / EXPERIMENT_NAME
        / run_name
        / "metrics.json"
    )


def main() -> None:
    combinations = [
        (
            fold,
            action_mode,
            seed,
        )
        for fold in FOLDS
        for action_mode in ACTION_MODES
        for seed in SEEDS
    ]

    print("PPO action-space ablation")
    print("=========================")
    print(
        f"Requested runs: "
        f"{len(combinations)}"
    )

    completed = 0
    skipped = 0

    for index, (
        fold,
        action_mode,
        seed,
    ) in enumerate(
        combinations,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(combinations)}] "
            f"{fold} | "
            f"{action_mode} | "
            f"seed {seed}"
        )

        if metrics_path(
            fold,
            action_mode,
            seed,
        ).exists():
            print(
                "Existing result found — "
                "skipping."
            )
            skipped += 1
            continue

        command = [
            sys.executable,
            "-m",
            "scripts.train_ppo_calibration",
            "--fold",
            fold,
            "--seed",
            str(seed),
            "--timesteps",
            str(TIMESTEPS),
            "--action-mode",
            action_mode,
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
    print(
        f"Completed: {completed}"
    )
    print(
        f"Skipped:   {skipped}"
    )


if __name__ == "__main__":
    main()