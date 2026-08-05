from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env[np.ndarray, int]):
    """
    Long-only single-asset trading environment.

    Actions
    -------
    0:
        Hold the current position.
    1:
        Buy Bitcoin using all available cash.
    2:
        Sell all Bitcoin and move to cash.

    Episode modes
    -------------
    Random fixed-length episode:
        Used during training to sample different historical periods.

    Fixed episode:
        Used during deterministic validation and testing.

    Full-split episode:
        Used when episode_length is None.
    """

    metadata = {"render_modes": ["human"]}

    HOLD = 0
    BUY = 1
    SELL = 2

    MARKET_COLUMNS = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 30,
        episode_length: int | None = None,
        random_start: bool = False,
        fixed_start_index: int | None = None,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._validate_parameters(
            data=data,
            window_size=window_size,
            episode_length=episode_length,
            random_start=random_start,
            fixed_start_index=fixed_start_index,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            render_mode=render_mode,
        )

        self.data = self._prepare_data(data)
        self.window_size = int(window_size)
        self.episode_length = episode_length
        self.random_start = random_start
        self.fixed_start_index = fixed_start_index
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)

        observation_size = (
            window_size * len(self.MARKET_COLUMNS) + 3
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.current_step = self.window_size
        self.episode_start_step = self.window_size
        self.episode_end_step = len(self.data) - 1
        self.episode_step_count = 0

        self.cash_balance = self.initial_balance
        self.btc_holdings = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        self.total_transaction_cost = 0.0
        self.trade_count = 0

    @classmethod
    def _validate_parameters(
        cls,
        data: pd.DataFrame,
        window_size: int,
        episode_length: int | None,
        random_start: bool,
        fixed_start_index: int | None,
        initial_balance: float,
        transaction_cost: float,
        render_mode: str | None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        missing_columns = [
            column
            for column in cls.MARKET_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required market columns: "
                f"{missing_columns}"
            )

        if window_size < 2:
            raise ValueError(
                "window_size must be at least 2."
            )

        if len(data) <= window_size + 1:
            raise ValueError(
                "Dataset must contain more than "
                "window_size + 1 rows."
            )

        maximum_episode_length = (
            len(data) - window_size - 1
        )

        if episode_length is not None:
            if episode_length < 1:
                raise ValueError(
                    "episode_length must be positive."
                )

            if episode_length > maximum_episode_length:
                raise ValueError(
                    "episode_length exceeds the number of "
                    "available transitions. "
                    f"Maximum allowed value is "
                    f"{maximum_episode_length}."
                )

        if random_start and episode_length is None:
            raise ValueError(
                "random_start=True requires a finite "
                "episode_length."
            )

        if random_start and fixed_start_index is not None:
            raise ValueError(
                "fixed_start_index cannot be used when "
                "random_start=True."
            )

        if fixed_start_index is not None:
            if fixed_start_index < window_size:
                raise ValueError(
                    "fixed_start_index must be at least "
                    "window_size."
                )

            if fixed_start_index >= len(data) - 1:
                raise ValueError(
                    "fixed_start_index must leave at least "
                    "one future transition."
                )

            required_length = (
                episode_length
                if episode_length is not None
                else len(data) - 1 - fixed_start_index
            )

            if (
                fixed_start_index + required_length
                > len(data) - 1
            ):
                raise ValueError(
                    "The fixed episode would cross the "
                    "dataset boundary."
                )

        if initial_balance <= 0.0:
            raise ValueError(
                "initial_balance must be positive."
            )

        if not 0.0 <= transaction_cost < 1.0:
            raise ValueError(
                "transaction_cost must be in [0, 1)."
            )

        supported_render_modes = cls.metadata[
            "render_modes"
        ]

        if (
            render_mode is not None
            and render_mode not in supported_render_modes
        ):
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported values: "
                f"{supported_render_modes}"
            )

    @classmethod
    def _prepare_data(
        cls,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        market_data = data.loc[
            :,
            cls.MARKET_COLUMNS,
        ].copy()

        market_data = market_data.astype(np.float64)
        market_data = market_data.sort_index()

        if market_data.index.has_duplicates:
            raise ValueError(
                "Market data contains duplicate timestamps."
            )

        if market_data.isna().any().any():
            raise ValueError(
                "Market data contains missing values."
            )

        if not np.isfinite(
            market_data.to_numpy()
        ).all():
            raise ValueError(
                "Market data contains non-finite values."
            )

        price_columns = [
            "open",
            "high",
            "low",
            "close",
        ]

        if (
            market_data[price_columns] <= 0.0
        ).any().any():
            raise ValueError(
                "All OHLC prices must be positive."
            )

        if (market_data["volume"] < 0.0).any():
            raise ValueError(
                "Volume cannot be negative."
            )

        return market_data

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        start_index = self._select_start_index(
            options=options
        )

        self.current_step = start_index
        self.episode_start_step = start_index
        self.episode_end_step = (
            self._calculate_episode_end(start_index)
        )
        self.episode_step_count = 0

        self.cash_balance = self.initial_balance
        self.btc_holdings = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = (
            self.initial_balance
        )
        self.total_transaction_cost = 0.0
        self.trade_count = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _select_start_index(
        self,
        options: dict[str, Any] | None,
    ) -> int:
        if options is not None:
            requested_start = options.get(
                "start_index"
            )

            if requested_start is not None:
                start_index = int(requested_start)
                self._validate_start_index(start_index)
                return start_index

        if self.random_start:
            maximum_start = self._maximum_start_index()

            return int(
                self.np_random.integers(
                    low=self.window_size,
                    high=maximum_start + 1,
                )
            )

        if self.fixed_start_index is not None:
            return self.fixed_start_index

        return self.window_size

    def _maximum_start_index(self) -> int:
        if self.episode_length is None:
            return self.window_size

        return (
            len(self.data)
            - self.episode_length
            - 1
        )

    def _validate_start_index(
        self,
        start_index: int,
    ) -> None:
        if start_index < self.window_size:
            raise ValueError(
                "start_index must be at least "
                f"{self.window_size}."
            )

        if start_index > self._maximum_start_index():
            raise ValueError(
                "start_index would cause the episode "
                "to cross the dataset boundary."
            )

    def _calculate_episode_end(
        self,
        start_index: int,
    ) -> int:
        if self.episode_length is None:
            return len(self.data) - 1

        return start_index + self.episode_length

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        execution_price = self._get_current_price()

        self.previous_portfolio_value = (
            self._calculate_portfolio_value(
                execution_price
            )
        )

        self._execute_action(
            action=action,
            execution_price=execution_price,
        )

        self.current_step += 1
        self.episode_step_count += 1

        valuation_price = self._get_current_price()

        self.portfolio_value = (
            self._calculate_portfolio_value(
                valuation_price
            )
        )

        reward = self._calculate_reward()

        terminated = self.portfolio_value <= 1e-8

        truncated = (
            self.current_step >= self.episode_end_step
            or self.current_step >= len(self.data) - 1
        )

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def _execute_action(
        self,
        action: int,
        execution_price: float,
    ) -> None:
        if action == self.BUY:
            self._buy(execution_price)
        elif action == self.SELL:
            self._sell(execution_price)

    def _buy(
        self,
        execution_price: float,
    ) -> None:
        if self.cash_balance <= 0.0:
            return

        transaction_fee = (
            self.cash_balance
            * self.transaction_cost
        )

        investable_cash = (
            self.cash_balance
            - transaction_fee
        )

        if investable_cash <= 0.0:
            return

        purchased_btc = (
            investable_cash / execution_price
        )

        self.btc_holdings += purchased_btc
        self.cash_balance = 0.0
        self.total_transaction_cost += (
            transaction_fee
        )
        self.trade_count += 1

    def _sell(
        self,
        execution_price: float,
    ) -> None:
        if self.btc_holdings <= 0.0:
            return

        gross_proceeds = (
            self.btc_holdings
            * execution_price
        )

        transaction_fee = (
            gross_proceeds
            * self.transaction_cost
        )

        net_proceeds = (
            gross_proceeds
            - transaction_fee
        )

        self.cash_balance += net_proceeds
        self.btc_holdings = 0.0
        self.total_transaction_cost += (
            transaction_fee
        )
        self.trade_count += 1

    def _get_observation(self) -> np.ndarray:
        start = (
            self.current_step
            - self.window_size
        )
        end = self.current_step

        window = self.data.iloc[start:end]

        if len(window) != self.window_size:
            raise RuntimeError(
                "Expected observation window of "
                f"{self.window_size}, received "
                f"{len(window)}."
            )

        normalized_market_data = (
            self._normalize_window(window)
        )

        market_features = (
            normalized_market_data.reshape(-1)
        )

        current_price = self._get_current_price()

        current_portfolio_value = (
            self._calculate_portfolio_value(
                current_price
            )
        )

        denominator = max(
            current_portfolio_value,
            1e-12,
        )

        cash_ratio = (
            self.cash_balance / denominator
        )

        bitcoin_value = (
            self.btc_holdings * current_price
        )

        bitcoin_ratio = (
            bitcoin_value / denominator
        )

        position = float(
            self.btc_holdings > 0.0
        )

        portfolio_features = np.array(
            [
                cash_ratio,
                bitcoin_ratio,
                position,
            ],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                market_features,
                portfolio_features,
            ]
        ).astype(np.float32)

        if (
            observation.shape
            != self.observation_space.shape
        ):
            raise RuntimeError(
                "Unexpected observation shape: "
                f"{observation.shape}; expected "
                f"{self.observation_space.shape}."
            )

        if not np.isfinite(observation).all():
            raise RuntimeError(
                "Observation contains non-finite values."
            )

        return observation

    @staticmethod
    def _normalize_window(
        window: pd.DataFrame,
    ) -> np.ndarray:
        values = window.loc[
            :,
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        ].to_numpy(dtype=np.float64)

        reference_close = values[-1, 3]

        normalized_prices = (
            values[:, :4]
            / max(reference_close, 1e-12)
        ) - 1.0

        log_volume = np.log1p(values[:, 4])

        volume_mean = log_volume.mean()
        volume_std = log_volume.std()

        if volume_std < 1e-12:
            normalized_volume = np.zeros_like(
                log_volume
            )
        else:
            normalized_volume = (
                log_volume - volume_mean
            ) / volume_std

        normalized = np.column_stack(
            [
                normalized_prices,
                normalized_volume,
            ]
        )

        return normalized.astype(np.float32)

    def _calculate_reward(self) -> float:
        previous_value = max(
            self.previous_portfolio_value,
            1e-12,
        )

        current_value = max(
            self.portfolio_value,
            1e-12,
        )

        reward = np.log(
            current_value / previous_value
        )

        return float(reward)

    def _calculate_portfolio_value(
        self,
        price: float,
    ) -> float:
        bitcoin_value = (
            self.btc_holdings * price
        )

        return float(
            self.cash_balance + bitcoin_value
        )

    def _get_current_price(self) -> float:
        return float(
            self.data.iloc[
                self.current_step
            ]["close"]
        )

    def _get_info(self) -> dict[str, Any]:
        current_price = self._get_current_price()

        return {
            "timestamp": (
                self.data.index[
                    self.current_step
                ]
            ),
            "current_step": self.current_step,
            "episode_start_step": (
                self.episode_start_step
            ),
            "episode_end_step": (
                self.episode_end_step
            ),
            "episode_step_count": (
                self.episode_step_count
            ),
            "current_price": current_price,
            "cash_balance": float(
                self.cash_balance
            ),
            "btc_holdings": float(
                self.btc_holdings
            ),
            "portfolio_value": float(
                self._calculate_portfolio_value(
                    current_price
                )
            ),
            "position": int(
                self.btc_holdings > 0.0
            ),
            "trade_count": self.trade_count,
            "total_transaction_cost": float(
                self.total_transaction_cost
            ),
        }

    def render(self) -> None:
        info = self._get_info()

        print(
            f"Episode step: "
            f"{info['episode_step_count']} | "
            f"Data step: {self.current_step} | "
            f"Timestamp: {info['timestamp']} | "
            f"Price: {info['current_price']:.2f} | "
            f"Position: {info['position']} | "
            f"Portfolio: "
            f"{info['portfolio_value']:.2f} | "
            f"Trades: {info['trade_count']}"
        )

    def close(self) -> None:
        pass