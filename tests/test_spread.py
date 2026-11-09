import numpy as np
import pandas as pd
import pytest

from src.statarb.research.spread import calculate_spread


def test_calculates_spread_correctly():
    dates = pd.date_range("2024-01-01", periods=3)

    price_a = pd.Series([110.0, 120.0, 130.0], index=dates)
    price_b = pd.Series([50.0, 55.0, 60.0], index=dates)

    result = calculate_spread(
        price_a,
        price_b,
        alpha=10.0,
        hedge_ratio=2.0,
    )

    expected = pd.Series(
        [0.0, 0.0, 0.0],
        index=dates,
        name="spread",
    )

    pd.testing.assert_series_equal(result, expected)


def test_missing_observations_are_aligned():
    dates = pd.date_range("2024-01-01", periods=3)

    price_a = pd.Series([110.0, np.nan, 130.0], index=dates)
    price_b = pd.Series([50.0, 55.0, 60.0], index=dates)

    result = calculate_spread(price_a, price_b, 10.0, 2.0)

    assert len(result) == 2
    assert result.isna().sum() == 0


def test_insufficient_observations_rejected():
    dates = pd.date_range("2024-01-01", periods=1)

    price_a = pd.Series([110.0], index=dates)
    price_b = pd.Series([50.0], index=dates)

    with pytest.raises(ValueError, match="At least two"):
        calculate_spread(price_a, price_b, 10.0, 2.0)


def test_duplicate_dates_rejected():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])

    price_a = pd.Series([110.0, 120.0], index=dates)
    price_b = pd.Series([50.0, 55.0], index=dates)

    with pytest.raises(ValueError, match="Duplicate dates"):
        calculate_spread(price_a, price_b, 10.0, 2.0)


def test_nonpositive_prices_rejected():
    dates = pd.date_range("2024-01-01", periods=2)

    price_a = pd.Series([110.0, 120.0], index=dates)
    price_b = pd.Series([0.0, 55.0], index=dates)

    with pytest.raises(ValueError, match="must be positive"):
        calculate_spread(price_a, price_b, 10.0, 2.0)


def test_invalid_parameters_rejected():
    dates = pd.date_range("2024-01-01", periods=2)

    price_a = pd.Series([110.0, 120.0], index=dates)
    price_b = pd.Series([50.0, 55.0], index=dates)

    with pytest.raises(ValueError, match="must be valid"):
        calculate_spread(price_a, price_b, float("nan"), 2.0)
        