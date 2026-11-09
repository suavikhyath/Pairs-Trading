import numpy as np
import pandas as pd
import pytest

from src.statarb.research.stationarity import check_stationarity


def test_stationary_series_detected():
    rng = np.random.default_rng(42)
    values = np.zeros(300)

    for i in range(1, 300):
        values[i] = 0.5 * values[i - 1] + rng.normal()

    spread = pd.Series(values)

    result = check_stationarity(spread)

    assert result["p_value"] < 0.05
    assert result["is_stationary"] is True


def test_empty_spread_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        check_stationarity(pd.Series(dtype=float))


def test_missing_values_rejected():
    spread = pd.Series([1.0, 2.0, np.nan] + list(range(30)))

    with pytest.raises(ValueError, match="missing"):
        check_stationarity(spread)


def test_insufficient_observations_rejected():
    spread = pd.Series(np.arange(20, dtype=float))

    with pytest.raises(ValueError, match="30 observations"):
        check_stationarity(spread)


def test_constant_spread_rejected():
    spread = pd.Series(np.ones(50))

    with pytest.raises(ValueError, match="constant"):
        check_stationarity(spread)


def test_invalid_significance_level_rejected():
    spread = pd.Series(np.random.default_rng(42).normal(size=100))

    with pytest.raises(ValueError, match="between 0 and 1"):
        check_stationarity(spread, significance_level=1.0)
