import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_performance_metrics


def test_known_equity_curve_metrics():
    equity = pd.Series(
        [100.0, 110.0, 99.0, 108.9]
    )

    metrics = calculate_performance_metrics(
        portfolio_values=equity,
        trade_count=3,
        total_transaction_cost=12.5,
        periods_per_year=3,
        risk_free_rate=0.0,
    )

    assert metrics.initial_value == pytest.approx(100.0)
    assert metrics.final_value == pytest.approx(108.9)

    assert metrics.total_return == pytest.approx(0.089)
    assert metrics.annualized_return == pytest.approx(0.089)

    assert metrics.annualized_volatility == pytest.approx(0.2)
    assert metrics.sharpe_ratio == pytest.approx(0.5)
    assert metrics.sortino_ratio == pytest.approx(1.0)

    assert metrics.maximum_drawdown == pytest.approx(-0.1)
    assert metrics.calmar_ratio == pytest.approx(0.89)

    assert metrics.trade_count == 3
    assert metrics.total_transaction_cost == pytest.approx(12.5)


def test_monotonically_increasing_equity_has_no_drawdown():
    equity = pd.Series(
        [100.0, 101.0, 102.0, 103.0]
    )

    metrics = calculate_performance_metrics(
        portfolio_values=equity,
        trade_count=0,
        total_transaction_cost=0.0,
        periods_per_year=365,
    )

    assert metrics.maximum_drawdown == pytest.approx(0.0)
    assert np.isnan(metrics.sortino_ratio)


def test_rejects_non_positive_equity():
    equity = pd.Series(
        [100.0, 90.0, 0.0]
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        calculate_performance_metrics(
            portfolio_values=equity,
            trade_count=0,
            total_transaction_cost=0.0,
        )


def test_rejects_too_short_equity_curve():
    equity = pd.Series([100.0])

    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        calculate_performance_metrics(
            portfolio_values=equity,
            trade_count=0,
            total_transaction_cost=0.0,
        )