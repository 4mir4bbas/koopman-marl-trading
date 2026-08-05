from __future__ import annotations

from stable_baselines3.common.env_checker import check_env

from src.data.loader import load_ohlcv
from src.environments.trading_env import TradingEnv


def main() -> None:
    data = load_ohlcv("data/raw/btc_usd_1d.csv")

    env = TradingEnv(
        data=data,
        window_size=30,
        initial_balance=10_000.0,
        transaction_cost=0.001,
    )

    check_env(env, warn=True)

    observation, info = env.reset(seed=42)

    print("Environment check passed.")
    print(f"Observation shape: {observation.shape}")
    print(f"Observation dtype: {observation.dtype}")
    print(f"Initial portfolio value: {info['portfolio_value']:.2f}")

    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    while not terminated and not truncated and step_count < 100:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        step_count += 1

    print(f"Random-policy steps: {step_count}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Final portfolio value: {info['portfolio_value']:.2f}")
    print(f"Executed trades: {info['trade_count']}")
    print(
        "Total transaction costs: "
        f"{info['total_transaction_cost']:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()