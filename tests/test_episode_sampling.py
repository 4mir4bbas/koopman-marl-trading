from __future__ import annotations

import numpy as np
import pandas as pd

from src.environments.trading_env import TradingEnv


def create_market(
    length: int = 900,
) -> pd.DataFrame:
    close = np.linspace(
        100.0,
        200.0,
        length,
    )

    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(
                length,
                1_000.0,
            ),
        },
        index=pd.date_range(
            start="2020-01-01",
            periods=length,
            freq="D",
            tz="UTC",
        ),
    )


def run_to_end(
    env: TradingEnv,
    seed: int,
) -> tuple[int, int, int]:
    _, initial_info = env.reset(seed=seed)

    start_step = int(
        initial_info["episode_start_step"]
    )

    expected_end_step = int(
        initial_info["episode_end_step"]
    )

    terminated = False
    truncated = False
    step_count = 0
    final_info = initial_info

    while not terminated and not truncated:
        (
            _,
            _,
            terminated,
            truncated,
            final_info,
        ) = env.step(TradingEnv.HOLD)

        step_count += 1

    final_step = int(
        final_info["current_step"]
    )

    assert truncated
    assert not terminated
    assert final_step == expected_end_step

    return (
        start_step,
        final_step,
        step_count,
    )


def test_random_episode_sampling_is_reproducible():
    data = create_market()

    episode_length = 365

    env = TradingEnv(
        data=data,
        window_size=30,
        episode_length=episode_length,
        random_start=True,
        initial_balance=10_000.0,
        transaction_cost=0.001,
    )

    first_run = run_to_end(
        env,
        seed=42,
    )

    repeated_run = run_to_end(
        env,
        seed=42,
    )

    different_seed_run = run_to_end(
        env,
        seed=43,
    )

    assert first_run == repeated_run

    assert first_run[2] == episode_length
    assert repeated_run[2] == episode_length
    assert different_seed_run[2] == episode_length

    assert first_run[0] >= 29
    assert first_run[1] < len(data)


def test_full_split_episode_is_deterministic():
    data = create_market()

    env = TradingEnv(
        data=data,
        window_size=30,
        episode_length=None,
        random_start=False,
        fixed_start_index=29,
        initial_balance=10_000.0,
        transaction_cost=0.001,
    )

    first_run = run_to_end(
        env,
        seed=42,
    )

    second_run = run_to_end(
        env,
        seed=999,
    )

    assert first_run == second_run

    expected_steps = (
        len(data)
        - 1
        - 29
    )

    assert first_run[2] == expected_steps


def test_manual_start_index_overrides_random_sampling():
    data = create_market()

    env = TradingEnv(
        data=data,
        window_size=30,
        episode_length=365,
        random_start=True,
    )

    _, info = env.reset(
        seed=42,
        options={
            "start_index": 500,
        },
    )

    assert (
        info["episode_start_step"]
        == 500
    )

    assert (
        info["episode_end_step"]
        == 865
    )