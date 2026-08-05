from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    initial_value: float
    final_value: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    calmar_ratio: float
    trade_count: int
    total_transaction_cost: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "maximum_drawdown": self.maximum_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "trade_count": self.trade_count,
            "total_transaction_cost": (
                self.total_transaction_cost
            ),
        }


def calculate_performance_metrics(
    portfolio_values: pd.Series,
    trade_count: int,
    total_transaction_cost: float,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate common trading-performance metrics.

    Parameters
    ----------
    portfolio_values:
        Chronologically ordered portfolio equity values.
    trade_count:
        Number of executed trades.
    total_transaction_cost:
        Total fees paid over the episode.
    periods_per_year:
        Number of observations per year. Daily crypto data uses 365.
    risk_free_rate:
        Annual risk-free rate represented as a decimal.
    """
    equity = pd.Series(
        portfolio_values,
        dtype=np.float64,
    ).dropna()

    if len(equity) < 2:
        raise ValueError(
            "At least two portfolio values are required."
        )

    if not np.isfinite(equity.to_numpy()).all():
        raise ValueError(
            "Portfolio values contain non-finite numbers."
        )

    if (equity <= 0.0).any():
        raise ValueError(
            "Portfolio values must be strictly positive."
        )

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    initial_value = float(equity.iloc[0])
    final_value = float(equity.iloc[-1])

    total_return = final_value / initial_value - 1.0

    period_count = len(equity) - 1
    years = period_count / periods_per_year

    if years > 0.0:
        annualized_return = (
            final_value / initial_value
        ) ** (1.0 / years) - 1.0
    else:
        annualized_return = np.nan

    simple_returns = equity.pct_change().dropna()

    annualized_volatility = float(
        simple_returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    period_risk_free_rate = (
        (1.0 + risk_free_rate) ** (1.0 / periods_per_year)
        - 1.0
    )

    excess_returns = simple_returns - period_risk_free_rate
    return_std = excess_returns.std(ddof=1)

    if return_std > 1e-12:
        sharpe_ratio = float(
            excess_returns.mean()
            / return_std
            * np.sqrt(periods_per_year)
        )
    else:
        sharpe_ratio = np.nan

    downside_returns = excess_returns[
        excess_returns < 0.0
    ]
    downside_deviation = downside_returns.std(ddof=1)

    if (
        len(downside_returns) >= 2
        and downside_deviation > 1e-12
    ):
        sortino_ratio = float(
            excess_returns.mean()
            / downside_deviation
            * np.sqrt(periods_per_year)
        )
    else:
        sortino_ratio = np.nan

    running_maximum = equity.cummax()
    drawdowns = equity / running_maximum - 1.0
    maximum_drawdown = float(drawdowns.min())

    drawdown_magnitude = abs(maximum_drawdown)

    if drawdown_magnitude > 1e-12:
        calmar_ratio = float(
            annualized_return / drawdown_magnitude
        )
    else:
        calmar_ratio = np.nan

    return PerformanceMetrics(
        initial_value=initial_value,
        final_value=final_value,
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        maximum_drawdown=maximum_drawdown,
        calmar_ratio=calmar_ratio,
        trade_count=int(trade_count),
        total_transaction_cost=float(
            total_transaction_cost
        ),
    )