from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 42

    total_timesteps: int = 100_000
    learning_rate: float = 3e-4

    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    hidden_size: int = 128

    window_size: int = 30
    train_episode_length: int = 365
    validation_episode_length: int | None = None

    initial_balance: float = 10_000.0
    transaction_cost: float = 0.001

    eval_frequency: int = 10_000
    verbose: int = 1

    model_directory: Path = Path("models/ppo")
    log_directory: Path = Path("results/ppo/logs")
    evaluation_directory: Path = Path(
        "results/ppo/evaluation"
    )

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError(
                "total_timesteps must be positive."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.n_steps <= 1:
            raise ValueError("n_steps must exceed 1.")

        if self.batch_size <= 1:
            raise ValueError("batch_size must exceed 1.")

        rollout_size = self.n_steps

        if rollout_size % self.batch_size != 0:
            raise ValueError(
                "n_steps must be divisible by batch_size "
                "when using one environment."
            )

        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(
                "gamma must be in the interval (0, 1]."
            )

        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must be in the interval [0, 1]."
            )

        if self.eval_frequency <= 0:
            raise ValueError(
                "eval_frequency must be positive."
            )

        if self.train_episode_length < 1:
            raise ValueError(
                "train_episode_length must be positive."
            )

        if (
            self.validation_episode_length is not None
            and self.validation_episode_length < 1
        ):
            raise ValueError(
                "validation_episode_length must be "
                "positive or None."
            )

    def policy_kwargs(self) -> dict[str, Any]:
        return {
            "net_arch": {
                "pi": [
                    self.hidden_size,
                    self.hidden_size,
                ],
                "vf": [
                    self.hidden_size,
                    self.hidden_size,
                ],
            }
        }

    def as_serializable_dict(self) -> dict[str, object]:
        values = asdict(self)

        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)

        return values