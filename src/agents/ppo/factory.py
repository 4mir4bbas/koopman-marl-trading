from __future__ import annotations

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.config.ppo_config import PPOConfig
from src.environments.trading_env import TradingEnv


def create_training_environment(
    data: pd.DataFrame,
    config: PPOConfig,
) -> Monitor:
    env = TradingEnv(
        data=data,
        window_size=config.window_size,
        episode_length=config.train_episode_length,
        random_start=True,
        fixed_start_index=None,
        initial_balance=config.initial_balance,
        transaction_cost=config.transaction_cost,
    )

    return Monitor(env)


def create_evaluation_environment(
    data: pd.DataFrame,
    config: PPOConfig,
) -> Monitor:
    env = TradingEnv(
        data=data,
        window_size=config.window_size,
        episode_length=(
            config.validation_episode_length
        ),
        random_start=False,
        fixed_start_index=config.window_size,
        initial_balance=config.initial_balance,
        transaction_cost=config.transaction_cost,
    )

    return Monitor(env)


def create_ppo_model(
    environment: Monitor,
    config: PPOConfig,
) -> PPO:
    config.validate()

    model = PPO(
        policy="MlpPolicy",
        env=environment,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=config.policy_kwargs(),
        seed=config.seed,
        verbose=config.verbose,
        tensorboard_log=str(
            config.log_directory
        ),
        device="cpu",
    )

    return model