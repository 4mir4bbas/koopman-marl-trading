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
from src.data.loader import load_ohlcv
from src.data.walk_forward import (
    WalkForwardFold,
    generate_expanding_annual_folds,
)
from src.evaluation.backtest import (
    create_evaluation_environment,
    run_episode,
)


EXPERIMENT_NAME = "ppo_walk_forward_v1"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO on one walk-forward fold "
            "and evaluate the final model once."
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Evaluation year, e.g. 2018.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="PPO random seed.",
    )

    return parser.parse_args()


def select_fold(
    folds: list[WalkForwardFold],
    fold_year: int,
) -> WalkForwardFold:
    fold_name = str(fold_year)

    matches = [
        fold
        for fold in folds
        if fold.name == fold_name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Walk-forward fold {fold_year} "
            "was not found."
        )

    return matches[0]


def get_run_directories(
    *,
    fold: WalkForwardFold,
    seed: int,
) -> tuple[Path, Path, Path]:
    model_directory = (
        Path("models")
        / EXPERIMENT_NAME
        / f"fold_{fold.name}"
        / f"seed_{seed}"
    )

    result_directory = (
        Path("results")
        / EXPERIMENT_NAME
        / f"fold_{fold.name}"
        / f"seed_{seed}"
    )

    log_directory = (
        result_directory / "logs"
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

    return (
        model_directory,
        result_directory,
        log_directory,
    )


def save_run_metadata(
    *,
    config: PPOConfig,
    fold: WalkForwardFold,
    result_directory: Path,
) -> None:
    metadata = {
        "protocol": "expanding_walk_forward_v1",
        "fold": fold.name,
        "training_start": str(
            fold.train.index.min()
        ),
        "training_end": str(
            fold.train.index.max()
        ),
        "training_rows": len(
            fold.train
        ),
        "evaluation_start": str(
            fold.evaluation.index.min()
        ),
        "evaluation_end": str(
            fold.evaluation.index.max()
        ),
        "evaluation_rows": len(
            fold.evaluation
        ),
        "context_rows": len(
            fold.context
        ),
        "checkpoint_selection": None,
        "outer_fold_used_during_training": False,
        "ppo_config": (
            config.as_serializable_dict()
        ),
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


def print_result(
    *,
    fold: WalkForwardFold,
    seed: int,
    metrics,
) -> None:
    print()
    print("PPO walk-forward result")
    print("=======================")

    print(
        f"Fold:              {fold.name}"
    )
    print(
        f"Seed:              {seed}"
    )
    print(
        f"Training period:   "
        f"{fold.train.index.min().date()} "
        f"-> "
        f"{fold.train.index.max().date()}"
    )
    print(
        f"Evaluation period: "
        f"{fold.evaluation.index.min().date()} "
        f"-> "
        f"{fold.evaluation.index.max().date()}"
    )
    print(
        f"Total return:      "
        f"{metrics.total_return:.2%}"
    )
    print(
        f"Annualized return: "
        f"{metrics.annualized_return:.2%}"
    )
    print(
        f"Sharpe ratio:      "
        f"{metrics.sharpe_ratio:.4f}"
    )
    print(
        f"Sortino ratio:     "
        f"{metrics.sortino_ratio:.4f}"
    )
    print(
        f"Maximum drawdown:  "
        f"{metrics.maximum_drawdown:.2%}"
    )
    print(
        f"Calmar ratio:      "
        f"{metrics.calmar_ratio:.4f}"
    )
    print(
        f"Trade count:       "
        f"{metrics.trade_count}"
    )
    print(
        f"Transaction costs: "
        f"{metrics.total_transaction_cost:,.2f}"
    )


def main() -> None:
    args = parse_arguments()

    config = PPOConfig(
        seed=args.seed,
        experiment_name=EXPERIMENT_NAME,
    )

    config.validate()
    set_global_seed(config.seed)

    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    folds = generate_expanding_annual_folds(
        data=data,
        first_evaluation_year=2018,
        final_evaluation_end="2024-11-05",
        window_size=config.window_size,
        training_start="2015-01-01",
    )

    fold = select_fold(
        folds,
        args.fold,
    )

    (
        model_directory,
        result_directory,
        log_directory,
    ) = get_run_directories(
        fold=fold,
        seed=config.seed,
    )

    save_run_metadata(
        config=config,
        fold=fold,
        result_directory=result_directory,
    )

    print("PPO walk-forward training")
    print("=========================")
    print(
        f"Fold:            {fold.name}"
    )
    print(
        f"Seed:            {config.seed}"
    )
    print(
        f"Training rows:   {len(fold.train)}"
    )
    print(
        f"Training period: "
        f"{fold.train.index.min().date()} "
        f"-> "
        f"{fold.train.index.max().date()}"
    )
    print(
        f"Evaluation data is NOT used "
        f"during training."
    )

    training_env = (
        create_training_environment(
            data=fold.train,
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
            tb_log_name=(
                f"fold_{fold.name}"
                f"_seed_{config.seed}"
            ),
        )

        final_model_path = (
            model_directory
            / "final_model"
        )

        model.save(
            final_model_path
        )

    finally:
        training_env.close()

    evaluation_env = (
        create_evaluation_environment(
            data=(
                fold.evaluation_environment_data
            ),
            window_size=config.window_size,
            start_index=(
                fold.evaluation_start_index
            ),
            episode_length=(
                fold.evaluation_episode_length
            ),
            initial_balance=(
                config.initial_balance
            ),
            transaction_cost=(
                config.transaction_cost
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

        predicted_action, _ = (
            model.predict(
                observation,
                deterministic=True,
            )
        )

        return int(
            np.asarray(
                predicted_action
            ).item()
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
        / "equity_curve.csv",
        index=True,
    )

    with (
        result_directory / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.metrics.as_dict(),
            file,
            indent=4,
        )

    print_result(
        fold=fold,
        seed=config.seed,
        metrics=result.metrics,
    )

    print()
    print(
        "Final model saved to: "
        f"{model_directory.resolve()}"
    )

    print(
        "Evaluation results saved to: "
        f"{result_directory.resolve()}"
    )


if __name__ == "__main__":
    main()