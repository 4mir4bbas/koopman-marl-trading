from __future__ import annotations

import json
import random

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
from src.experiments.paths import (
    ExperimentPaths,
    get_experiment_paths,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_directories(
    paths: ExperimentPaths,
) -> None:
    paths.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths.validation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_config(
    config: PPOConfig,
    paths: ExperimentPaths,
) -> None:
    output_path = (
        paths.result_dir / "config.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config.as_serializable_dict(),
            file,
            indent=4,
        )


def train_single_seed(
    config: PPOConfig,
) -> None:
    config.validate()

    set_global_seed(config.seed)

    paths = get_experiment_paths(
        experiment_name=config.experiment_name,
        seed=config.seed,
    )

    prepare_directories(paths)
    save_config(config, paths)

    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    splits = chronological_split(
        data=data,
        train_ratio=0.70,
        validation_ratio=0.15,
        minimum_split_size=(
            config.window_size + 2
        ),
    )

    training_env = (
        create_training_environment(
            data=splits.train,
            config=config,
        )
    )

    validation_env = (
        create_evaluation_environment(
            data=splits.validation,
            config=config,
        )
    )

    model = create_ppo_model(
        environment=training_env,
        config=config,
        tensorboard_log=str(
            paths.log_dir
        ),
    )

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=25_000,
            save_path=str(
                paths.model_dir
                / "checkpoints"
            ),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    )

    evaluation_callback = EvalCallback(
        eval_env=validation_env,
        best_model_save_path=str(
            paths.model_dir / "best"
        ),
        log_path=str(
            paths.evaluation_dir
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

    try:
        model.learn(
            total_timesteps=(
                config.total_timesteps
            ),
            callback=callbacks,
            progress_bar=True,
            tb_log_name=(
                f"seed_{config.seed}"
            ),
        )

        model.save(
            paths.model_dir
            / "final_model"
        )

    finally:
        training_env.close()
        validation_env.close()


def main() -> None:
    config = PPOConfig()

    train_single_seed(config)


if __name__ == "__main__":
    main()