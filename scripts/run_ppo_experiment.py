from __future__ import annotations

from dataclasses import replace

from src.config.ppo_config import PPOConfig
from scripts.train_ppo import (
    train_single_seed,
)


SEEDS = (
    42,
    123,
    2026,
)


def main() -> None:
    base_config = PPOConfig()

    for seed in SEEDS:
        print()
        print("=" * 60)
        print(
            f"Training PPO | seed={seed}"
        )
        print("=" * 60)

        config = replace(
            base_config,
            seed=seed,
        )

        train_single_seed(
            config=config
        )


if __name__ == "__main__":
    main()