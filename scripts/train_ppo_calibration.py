from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.agents.ppo.factory import (
    create_ppo_model,
    create_training_environment,
)
from src.config.ppo_config import PPOConfig
from src.data.calibration import (
    generate_pre_2018_calibration_folds,
)
from src.data.loader import load_ohlcv
from src.evaluation.backtest import (
    create_evaluation_environment,
    run_episode,
)


EXPERIMENT_NAME = "ppo_baseline_v2_calibration"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--action-mode",
        type=str,
        choices=[
            "orders",
            "target_position",
        ],
        default="target_position",
    )

    parser.add_argument(
        "--fold",
        type=str,
        choices=[
            "C1",
            "C2",
            "C3",
        ],
        required=True,
    )

    return parser.parse_args()


def set_global_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_arguments()

    config = PPOConfig(
        seed=args.seed,
        total_timesteps=args.timesteps,
        experiment_name=EXPERIMENT_NAME,
        action_mode=args.action_mode,
    )

    config.validate()
    set_global_seed(config.seed)

    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    folds = generate_pre_2018_calibration_folds(
        data,
        window_size=config.window_size,
    )

    matches = [
        fold
        for fold in folds
        if fold.name == args.fold
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Calibration fold {args.fold} "
            "was not found."
        )

    split = matches[0]

    run_name = (
        f"fold_{split.name}"
        f"_action_{config.action_mode}"
        f"_steps_{config.total_timesteps}"
        f"_seed_{config.seed}"
    )

    model_directory = (
        Path("models")
        / EXPERIMENT_NAME
        / run_name
    )

    result_directory = (
        Path("results")
        / EXPERIMENT_NAME
        / run_name
    )

    log_directory = (
        result_directory
        / "logs"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "protocol": "pre_2018_calibration_v1",
        "training_start": str(
            split.train.index.min()
        ),
        "training_end": str(
            split.train.index.max()
        ),
        "validation_start": str(
            split.validation.index.min()
        ),
        "validation_end": str(
            split.validation.index.max()
        ),
        "outer_walk_forward_used": False,
        "final_test_used": False,
        "ppo_config": (
            config.as_serializable_dict()
        ),
        "calibration_fold": split.name,
    }

    with (
        result_directory
        / "run_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("PPO-v2 calibration")
    print("==================")
    print(
        f"Seed:       {config.seed}"
    )
    print(
        f"Timesteps:  {config.total_timesteps:,}"
    )
    print(
        f"Action:     {config.action_mode}"
    )
    print(
        "Train:      2015-01-01 -> 2016-12-31"
    )
    print(
        "Validation: 2017-01-01 -> 2017-12-31"
    )
    print(
        f"Fold:       {split.name}"
    )

    training_env = (
        create_training_environment(
            data=split.train,
            config=config,
        )
    )

    try:
        model = create_ppo_model(
            environment=training_env,
            config=config,
            tensorboard_log=str(
                log_directory
            ),
        )

        model.learn(
            total_timesteps=(
                config.total_timesteps
            ),
            progress_bar=True,
            tb_log_name=run_name,
        )

        model.save(
            model_directory
            / "final_model"
        )

    finally:
        training_env.close()

    evaluation_env = (
        create_evaluation_environment(
            data=(
                split.validation_environment_data
            ),
            window_size=config.window_size,
            start_index=(
                split.validation_start_index
            ),
            episode_length=(
                split.validation_episode_length
            ),
            initial_balance=(
                config.initial_balance
            ),
            transaction_cost=(
                config.transaction_cost
            ),
            action_mode=(
                config.action_mode
            ),
        )
    )

    def ppo_policy(
        env,
        observation,
        info,
        step_number,
    ) -> int:
        del env, info, step_number

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        return int(
            np.asarray(action).item()
        )

    try:
        result = run_episode(
            env=evaluation_env,
            policy=ppo_policy,
            seed=config.seed,
        )
    finally:
        evaluation_env.close()

    result.portfolio_values.to_csv(
        result_directory
        / "equity_curve.csv"
    )

    with (
        result_directory
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.metrics.as_dict(),
            file,
            indent=4,
        )

    metrics = result.metrics

    print()
    print("Calibration result")
    print("==================")
    print(
        f"Return:      {metrics.total_return:.2%}"
    )
    print(
        f"Sharpe:      {metrics.sharpe_ratio:.4f}"
    )
    print(
        f"Sortino:     {metrics.sortino_ratio:.4f}"
    )
    print(
        f"Max DD:      {metrics.maximum_drawdown:.2%}"
    )
    print(
        f"Trades:      {metrics.trade_count}"
    )
    print(
        f"Costs:       "
        f"{metrics.total_transaction_cost:,.2f}"
    )


if __name__ == "__main__":
    main()