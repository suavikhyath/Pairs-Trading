import numpy as np
import pandas as pd
import pytest

from src.statarb.research.cointegration import check_cointegration


def make_cointegrated_prices(n=300, seed=42):
    rng = np.random.default_rng(seed)

    common_walk = np.cumsum(rng.normal(0, 1, n))
    stationary_noise = np.zeros(n)

    innovations = rng.normal(0, 0.5, n)
    for i in range(1, n):
        stationary_noise[i] = (
            0.4 * stationary_noise[i - 1] + innovations[i]
        )

    price_a = 100 + common_walk
    price_b = 50 + 1.5 * common_walk + stationary_noise

    dates = pd.date_range("2023-01-01", periods=n)

    return (
        pd.Series(price_a, index=dates),
        pd.Series(price_b, index=dates),
    )


def test_cointegrated_series_detected():
    price_a, price_b = make_cointegrated_prices()

    result = check_cointegration(price_a, price_b)

    assert result["p_value"] < 0.05
    assert result["is_cointegrated"] is True
    assert result["observations"] == 300


def test_empty_series_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        check_cointegration(pd.Series(dtype=float), pd.Series(dtype=float))


def test_invalid_significance_level_rejected():
    prices = pd.Series(np.arange(1, 41, dtype=float))

    with pytest.raises(ValueError, match="between 0 and 1"):
        check_cointegration(prices, prices, significance_level=1.0)


def test_duplicate_dates_rejected():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
    prices_a = pd.Series([100.0, 101.0], index=dates)
    prices_b = pd.Series([50.0, 51.0], index=dates)

    with pytest.raises(ValueError, match="Duplicate dates"):
        check_cointegration(prices_a, prices_b)


def test_insufficient_observations_rejected():
    dates = pd.date_range("2024-01-01", periods=20)
    prices_a = pd.Series(np.arange(100, 120), index=dates)
    prices_b = pd.Series(np.arange(50, 70), index=dates)

    with pytest.raises(ValueError, match="30 overlapping"):
        check_cointegration(prices_a, prices_b)


def test_constant_series_rejected():
    dates = pd.date_range("2024-01-01", periods=40)
    prices_a = pd.Series(np.ones(40) * 100, index=dates)
    prices_b = pd.Series(np.arange(50, 90), index=dates)

    with pytest.raises(ValueError, match="must not be constant"):
        check_cointegration(prices_a, prices_b)


def test_nonpositive_prices_rejected():
    dates = pd.date_range("2024-01-01", periods=40)
    prices_a = pd.Series(np.arange(0, 40, dtype=float), index=dates)
    prices_b = pd.Series(np.arange(50, 90, dtype=float), index=dates)

    with pytest.raises(ValueError, match="must be positive"):
        check_cointegration(prices_a, prices_b)
        