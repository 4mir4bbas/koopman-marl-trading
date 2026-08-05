from __future__ import annotations

from src.data.loader import load_ohlcv
from src.data.split import chronological_split
from src.environments.trading_env import TradingEnv


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


def main() -> None:
    data = load_ohlcv(
        "data/raw/btc_usd_1d.csv"
    )

    splits = chronological_split(data)

    episode_length = 365

    training_env = TradingEnv(
        data=splits.train,
        window_size=30,
        episode_length=episode_length,
        random_start=True,
        initial_balance=10_000.0,
        transaction_cost=0.001,
    )

    first_run = run_to_end(
        training_env,
        seed=42,
    )

    repeated_run = run_to_end(
        training_env,
        seed=42,
    )

    different_seed_run = run_to_end(
        training_env,
        seed=43,
    )

    assert first_run == repeated_run

    assert first_run[2] == episode_length
    assert repeated_run[2] == episode_length
    assert different_seed_run[2] == episode_length

    assert first_run[0] >= 30

    assert first_run[1] < len(
        splits.train
    )

    print(
        "Random episode sampling test passed."
    )

    print(
        "Seed 42 start/end: "
        f"{first_run[0]} -> {first_run[1]}"
    )

    print(
        "Repeated seed 42 start/end: "
        f"{repeated_run[0]} -> "
        f"{repeated_run[1]}"
    )

    print(
        "Seed 43 start/end: "
        f"{different_seed_run[0]} -> "
        f"{different_seed_run[1]}"
    )

    validation_env = TradingEnv(
        data=splits.validation,
        window_size=30,
        episode_length=None,
        random_start=False,
        fixed_start_index=30,
        initial_balance=10_000.0,
        transaction_cost=0.001,
    )

    first_validation = run_to_end(
        validation_env,
        seed=42,
    )

    second_validation = run_to_end(
        validation_env,
        seed=999,
    )

    assert (
        first_validation
        == second_validation
    )

    expected_validation_steps = (
        len(splits.validation) - 1 - 30
    )

    assert (
        first_validation[2]
        == expected_validation_steps
    )

    print(
        "Deterministic validation episode "
        "test passed."
    )

    print(
        "Validation start/end: "
        f"{first_validation[0]} -> "
        f"{first_validation[1]}"
    )

    manual_env = TradingEnv(
        data=splits.train,
        window_size=30,
        episode_length=365,
        random_start=True,
    )

    _, manual_info = manual_env.reset(
        seed=42,
        options={"start_index": 500},
    )

    assert (
        manual_info["episode_start_step"]
        == 500
    )

    assert (
        manual_info["episode_end_step"]
        == 865
    )

    print(
        "Manual episode start test passed."
    )


if __name__ == "__main__":
    main()