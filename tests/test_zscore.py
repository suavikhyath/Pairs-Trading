import numpy as np
import pandas as pd
import pytest

from src.statarb.research.zscore import calculate_zscore


def test_calculates_rolling_zscore():
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = calculate_zscore(spread, window=3)

    assert result.iloc[:3].isna().all()

    expected_mean = np.mean([1.0, 2.0, 3.0])
    expected_std = np.std([1.0, 2.0, 3.0], ddof=1)

    expected_zscore = (4.0 - expected_mean) / expected_std

    assert result.iloc[3] == pytest.approx(expected_zscore)


def test_current_observation_not_included_in_baseline():
    spread = pd.Series([1.0, 2.0, 3.0, 100.0])

    result = calculate_zscore(spread, window=3)

    expected = (100.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=1)

    assert result.iloc[3] == pytest.approx(expected)


def test_constant_spread_returns_nan():
    spread = pd.Series([5.0] * 10)

    result = calculate_zscore(spread, window=3)

    assert result.isna().all()


def test_empty_spread_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_zscore(pd.Series(dtype=float))


def test_missing_values_rejected():
    spread = pd.Series([1.0, 2.0, np.nan, 4.0])

    with pytest.raises(ValueError, match="missing"):
        calculate_zscore(spread)


def test_invalid_window_rejected():
    spread = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="at least 2"):
        calculate_zscore(spread, window=1)


def test_insufficient_observations():
    spread = pd.Series([1.0, 2.0, 3.0])

    result = calculate_zscore(spread, window=5)

    assert result.isna().all()