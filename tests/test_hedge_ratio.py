import numpy as np
import pandas as pd
import pytest

from src.statarb.research.hedge_ratio import estimate_hedge_ratio


def test_recovers_known_linear_relationship():
    dates = pd.date_range("2024-01-01", periods=50)
    b = pd.Series(np.arange(50, dtype=float) + 100, index=dates)
    a = 5 + 2.5 * b

    result = estimate_hedge_ratio(a, b)

    assert result["alpha"] == pytest.approx(5.0)
    assert result["hedge_ratio"] == pytest.approx(2.5)
    assert result["r_squared"] == pytest.approx(1.0)


def test_insufficient_observations_rejected():
    dates = pd.date_range("2024-01-01", periods=20)
    a = pd.Series(np.arange(20) + 100.0, index=dates)
    b = pd.Series(np.arange(20) + 50.0, index=dates)

    with pytest.raises(ValueError, match="30 overlapping"):
        estimate_hedge_ratio(a, b)


def test_duplicate_dates_rejected():
    dates = pd.to_datetime(["2024-01-01"] * 2)
    a = pd.Series([100.0, 101.0], index=dates)
    b = pd.Series([50.0, 51.0], index=dates)

    with pytest.raises(ValueError, match="Duplicate dates"):
        estimate_hedge_ratio(a, b)


def test_constant_series_rejected():
    dates = pd.date_range("2024-01-01", periods=40)
    a = pd.Series(np.ones(40) * 100, index=dates)
    b = pd.Series(np.arange(40, dtype=float) + 50, index=dates)

    with pytest.raises(ValueError, match="must not be constant"):
        estimate_hedge_ratio(a, b)


def test_nonpositive_prices_rejected():
    dates = pd.date_range("2024-01-01", periods=40)
    a = pd.Series(np.arange(40, dtype=float), index=dates)
    b = pd.Series(np.arange(40, dtype=float) + 50, index=dates)

    with pytest.raises(ValueError, match="must be positive"):
        estimate_hedge_ratio(a, b)
        