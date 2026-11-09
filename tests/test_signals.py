import numpy as np
import pandas as pd
import pytest

from src.statarb.strategy.signals import generate_positions


def test_short_spread_entry_and_exit():
    zscore = pd.Series([0.5, 2.1, 2.5, 1.0, 0.0])

    result = generate_positions(zscore)

    assert result.tolist() == [0, -1, -1, -1, 0]


def test_long_spread_entry_and_exit():
    zscore = pd.Series([0.0, -2.2, -2.5, -1.0, 0.0])

    result = generate_positions(zscore)

    assert result.tolist() == [0, 1, 1, 1, 0]


def test_position_does_not_repeat_entries():
    zscore = pd.Series([2.1, 2.5, 3.0, 2.2])

    result = generate_positions(zscore)

    assert result.tolist() == [-1, -1, -1, -1]


def test_opposite_signal_exits_before_reversing():
    zscore = pd.Series([-2.5, 2.5, 2.6])

    result = generate_positions(zscore)

    assert result.tolist() == [1, 0, -1]


def test_initial_missing_values_are_allowed():
    zscore = pd.Series([np.nan, np.nan, -2.5, -1.0, 0.0])

    result = generate_positions(zscore)

    assert result.tolist() == [0, 0, 1, 1, 0]


def test_empty_zscore_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_positions(pd.Series(dtype=float))


def test_invalid_thresholds_rejected():
    zscore = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="positive"):
        generate_positions(zscore, entry_threshold=0)

    with pytest.raises(ValueError, match="smaller"):
        generate_positions(
            zscore,
            entry_threshold=2.0,
            exit_threshold=2.0,
        )


def test_duplicate_dates_rejected():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
    zscore = pd.Series([1.0, 2.0], index=dates)

    with pytest.raises(ValueError, match="Duplicate dates"):
        generate_positions(zscore)
