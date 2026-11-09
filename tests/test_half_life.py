import numpy as np
import pandas as pd
import pytest


from src.statarb.research.half_life import estimate_half_life


def test_estimates_half_life_of_mean_reverting_series():
    rng = np.random.default_rng(42)
    spread = np.zeros(1000)

    for i in range(1, 1000):
        spread[i] = 0.8 * spread[i - 1] + rng.normal()

    result = estimate_half_life(pd.Series(spread))

    expected = -np.log(2) / (-0.2)

    assert result == pytest.approx(expected, abs=0.5)


def test_non_mean_reverting_series_rejected():
    rng = np.random.default_rng(42)
    spread = np.cumsum(rng.normal(size=300))

    with pytest.raises(ValueError, match="mean-reverting"):
        estimate_half_life(pd.Series(spread))


def test_empty_spread_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        estimate_half_life(pd.Series(dtype=float))


def test_missing_values_rejected():
    spread = pd.Series([1.0, 2.0, np.nan] + list(range(30)))

    with pytest.raises(ValueError, match="missing"):
        estimate_half_life(spread)


def test_insufficient_observations_rejected():
    spread = pd.Series(np.arange(20, dtype=float))

    with pytest.raises(ValueError, match="30 observations"):
        estimate_half_life(spread)


def test_constant_spread_rejected():
    spread = pd.Series(np.ones(50))

    with pytest.raises(ValueError, match="constant"):
        estimate_half_life(spread)
