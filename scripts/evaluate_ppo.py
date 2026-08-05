from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.agents.ppo.factory import (
    create_evaluation_environment,
)
from src.config.ppo_config import PPOConfig
from src.data.loader import load_ohlcv
from src.data.split import chronological_split
from src.evaluation.metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)


def run_model_episode(
    model: PPO,
    data: pd.DataFrame,
    config: PPOConfig,
) -> tuple[pd.Series, PerformanceMetrics]:
    env = create_evaluation_environment(
        data=data,
        config=config,
    )

    observation, info = env.reset(
        seed=config.seed
    )

    timestamps = [
        pd.Timestamp(info["timestamp"])
    ]
    portfolio_values = [
        float(info["portfolio_value"])
    ]
    actions: list[int] = []

    terminated = False
    truncated = False

    try:
        while not terminated and not truncated:
            predicted_action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = int(
                np.asarray(predicted_action).item()
            )

            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            actions.append(action)
            timestamps.append(
                pd.Timestamp(info["timestamp"])
            )
            portfolio_values.append(
                float(info["portfolio_value"])
            )

    finally:
        env.close()

    equity_curve = pd.Series(
        portfolio_values,
        index=pd.DatetimeIndex(timestamps),
        name="ppo_portfolio_value",
        dtype=np.float64,
    )

    metrics = calculate_performance_metrics(
        portfolio_values=equity_curve,
        trade_count=int(info["trade_count"]),
        total_transaction_cost=float(
            info["total_transaction_cost"]
        ),
        periods_per_year=365,
    )

    unique_actions, action_counts = np.unique(
        actions,
        return_counts=True,
    )

    print("Action distribution")
    print("-------------------")

    for action, count in zip(
        unique_actions,
        action_counts,
        strict=True,
    ):
        print(
            f"Action {int(action)}: {int(count):,}"
        )

    return equity_curve, metrics


def print_metrics(
    metrics: PerformanceMetrics,
) -> None:
    print("\nPPO validation performance")
    print("--------------------------")
    print(
        f"Initial value:       "
        f"{metrics.initial_value:,.2f}"
    )
    print(
        f"Final value:         "
        f"{metrics.final_value:,.2f}"
    )
    print(
        f"Total return:        "
        f"{metrics.total_return:.2%}"
    )
    print(
        f"Annualized return:   "
        f"{metrics.annualized_return:.2%}"
    )
    print(
        f"Annualized vol.:     "
        f"{metrics.annualized_volatility:.2%}"
    )
    print(
        f"Sharpe ratio:        "
        f"{metrics.sharpe_ratio:.4f}"
    )
    print(
        f"Sortino ratio:       "
        f"{metrics.sortino_ratio:.4f}"
    )
    print(
        f"Maximum drawdown:    "
        f"{metrics.maximum_drawdown:.2%}"
    )
    print(
        f"Calmar ratio:        "
        f"{metrics.calmar_ratio:.4f}"
    )
    print(
        f"Trade count:         "
        f"{metrics.trade_count}"
    )
    print(
        f"Transaction costs:   "
        f"{metrics.total_transaction_cost:,.2f}"
    )


def main() -> None:
    config = PPOConfig()

    best_model_path = Path(
        "models/ppo/best/best_model.zip"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            "Best PPO model was not found. "
            "Run scripts.train_ppo first."
        )

    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    splits = chronological_split(
        data=data,
        train_ratio=0.70,
        validation_ratio=0.15,
        minimum_split_size=config.window_size + 2,
    )

    # Intentionally evaluate only on validation.
    # Test remains untouched.
    evaluation_data = splits.validation

    evaluation_env = create_evaluation_environment(
    data=evaluation_data,
    config=config,
)

    try:
        model = PPO.load(
            best_model_path,
            env=evaluation_env,
            device="cpu",
        )
    finally:
        evaluation_env.close()

    equity_curve, metrics = run_model_episode(
        model=model,
        data=evaluation_data,
        config=config,
    )

    print_metrics(metrics)

    output_directory = Path(
        "results/ppo/validation"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    equity_curve.to_csv(
        output_directory
        / "ppo_equity_curve.csv",
        index=True,
    )

    with (
        output_directory / "ppo_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics.as_dict(),
            file,
            indent=4,
        )

    print(
        "\nValidation results saved to: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()