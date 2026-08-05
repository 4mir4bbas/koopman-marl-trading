from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from src.agents.ppo.factory import (
    create_evaluation_environment,
    create_ppo_model,
    create_training_environment,
)
from src.config.ppo_config import PPOConfig
from src.data.loader import load_ohlcv
from src.data.split import chronological_split


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_directories(
    config: PPOConfig,
) -> None:
    config.model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.evaluation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_config(
    config: PPOConfig,
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config.as_serializable_dict(),
            file,
            indent=4,
        )


def main() -> None:
    config = PPOConfig()
    config.validate()

    set_global_seed(config.seed)
    prepare_directories(config)

    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    splits = chronological_split(
        data=data,
        train_ratio=0.70,
        validation_ratio=0.15,
        minimum_split_size=config.window_size + 2,
    )

    print("PPO training experiment")
    print("-----------------------")
    print(
        f"Training rows:   {len(splits.train):,}"
    )
    print(
        f"Validation rows: {len(splits.validation):,}"
    )
    print(
        f"Test rows held out: {len(splits.test):,}"
    )
    print(f"Seed: {config.seed}")
    print(
        f"Total timesteps: "
        f"{config.total_timesteps:,}"
    )
    print(
        "Training episode length: "
        f"{config.train_episode_length}"
    )

    print(
        "Training start mode: random"
    )

    print(
        "Validation start mode: fixed"
    )

    training_env = create_training_environment(
        data=splits.train,
        config=config,
    )

    validation_env = create_evaluation_environment(
        data=splits.validation,
        config=config,
    )

    model = create_ppo_model(
        environment=training_env,
        config=config,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(
            config.model_directory / "checkpoints"
        ),
        name_prefix=f"ppo_seed_{config.seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    evaluation_callback = EvalCallback(
        eval_env=validation_env,
        best_model_save_path=str(
            config.model_directory / "best"
        ),
        log_path=str(
            config.evaluation_directory
        ),
        eval_freq=config.eval_frequency,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList(
        [
            checkpoint_callback,
            evaluation_callback,
        ]
    )

    save_config(
        config=config,
        output_path=(
            config.evaluation_directory
            / f"config_seed_{config.seed}.json"
        ),
    )

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"ppo_seed_{config.seed}",
        )

        final_model_path = (
            config.model_directory
            / f"ppo_final_seed_{config.seed}"
        )

        model.save(final_model_path)

        print()
        print(f"Final model saved to: {final_model_path}")
        print(
            "Best validation model saved under: "
            f"{config.model_directory / 'best'}"
        )

    finally:
        training_env.close()
        validation_env.close()


if __name__ == "__main__":
    main()