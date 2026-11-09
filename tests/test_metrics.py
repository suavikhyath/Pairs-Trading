
import numpy as np
import pandas as pd
import pytest

from src.statarb.backtesting.metrics import (
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_win_rate,
    calculate_profit_factor
)


def test_cagr():
    equity = pd.Series(
        [100, 121],
        index=pd.to_datetime(["2025-01-01", "2027-01-01"]),
    )

    result = calculate_cagr(equity)

    expected = (121 / 100) ** (365.25 / 730) - 1

    assert result == pytest.approx(expected)


def test_cagr_rejects_nonpositive_equity():
    equity = pd.Series(
        [100, 0],
        index=pd.to_datetime(["2025-01-01", "2026-01-01"]),
    )

    with pytest.raises(ValueError):
        calculate_cagr(equity)


def test_volatility():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])

    result = calculate_volatility(returns)

    assert result == pytest.approx(
        returns.std(ddof=1) * np.sqrt(252)
    )


def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, 0.015, 0.025])

    result = calculate_sharpe_ratio(
        returns,
        risk_free_rate=0,
    )

    expected = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    )

    assert result == pytest.approx(expected)


def test_sharpe_rejects_zero_volatility():
    returns = pd.Series([0.01, 0.01, 0.01])

    with pytest.raises(ValueError):
        calculate_sharpe_ratio(returns)


def test_volatility_rejects_nan():
    returns = pd.Series([0.01, np.nan, 0.02])

    with pytest.raises(ValueError):
        calculate_volatility(returns)



def test_maximum_drawdown():
    equity = pd.Series(
        [100, 120, 90, 110, 80],
        index=pd.date_range("2025-01-01", periods=5),
    )

    result = calculate_max_drawdown(equity)

    assert result == pytest.approx(-1 / 3)


def test_sortino_ratio():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])

    result = calculate_sortino_ratio(returns)

    downside = np.minimum(returns, 0)
    expected = (
        returns.mean()
        / np.sqrt(np.mean(downside ** 2))
        * np.sqrt(252)
    )

    assert result == pytest.approx(expected)


def test_win_rate():
    pnls = pd.Series([100, -50, 200, -25, 75])

    assert calculate_win_rate(pnls) == pytest.approx(0.6)


def test_profit_factor():
    pnls = pd.Series([100, -50, 200, -25, 75])

    result = calculate_profit_factor(pnls)

    assert result == pytest.approx(375 / 75)


def test_win_rate_rejects_empty_trades():
    with pytest.raises(ValueError):
        calculate_win_rate(pd.Series(dtype=float))


def test_profit_factor_rejects_zero_losses():
    with pytest.raises(ValueError):
        calculate_profit_factor(pd.Series([100, 200, 300]))


def test_calculate_performance_metrics():
    from src.statarb.backtesting.metrics import (
        calculate_performance_metrics,
    )

    dates = pd.date_range("2025-01-01", periods=5)

    equity_curve = pd.DataFrame(
        {
            "equity": [100000, 102000, 101000, 105000, 103000]
        },
        index=dates,
    )

    result = calculate_performance_metrics(equity_curve)

    assert "cagr" in result
    assert "annualized_volatility" in result
    assert "sharpe_ratio" in result
    assert "sortino_ratio" in result
    assert "max_drawdown" in result

    assert result["max_drawdown"] == pytest.approx(
        103000 / 105000 - 1
    )
