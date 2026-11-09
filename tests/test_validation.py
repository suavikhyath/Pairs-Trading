
import numpy as np
import pandas as pd
import pytest

from src.statarb.backtesting.validation import (
    validate_price_data,
    get_common_trading_calendar,
    align_prices_to_common_calendar,
)


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2026-01-05", periods=5)

    return pd.DataFrame(
        {
            "AAPL": [100, 102, 104, np.nan, 108],
            "MSFT": [200, np.nan, 204, 206, 208],
        },
        index=dates,
    )


def test_validate_price_data(sample_prices):
    result = validate_price_data(sample_prices)

    assert result["rows"] == 5
    assert result["tickers"] == 2
    assert result["missing_data_present"] is True
    assert result["missing_counts"]["AAPL"] == 1
    assert result["missing_counts"]["MSFT"] == 1
    assert result["common_trading_dates"] == 3


def test_common_trading_calendar(sample_prices):
    result = get_common_trading_calendar(sample_prices)

    expected = sample_prices.index[[0, 2, 4]]

    assert result.equals(expected)


def test_align_prices_to_common_calendar(sample_prices):
    result = align_prices_to_common_calendar(sample_prices)

    assert len(result) == 3
    assert result.isna().sum().sum() == 0


def test_missing_values_are_not_forward_filled(sample_prices):
    result = align_prices_to_common_calendar(sample_prices)

    assert sample_prices.iloc[1]["MSFT"] != result.iloc[1]["MSFT"]


def test_duplicate_dates_rejected(sample_prices):
    sample_prices.index = [
        sample_prices.index[0],
        sample_prices.index[1],
        sample_prices.index[1],
        sample_prices.index[3],
        sample_prices.index[4],
    ]

    with pytest.raises(ValueError, match="unique"):
        validate_price_data(sample_prices)


def test_unsorted_dates_rejected(sample_prices):
    sample_prices = sample_prices.iloc[::-1]

    with pytest.raises(ValueError, match="chronologically"):
        validate_price_data(sample_prices)


def test_nonpositive_prices_rejected(sample_prices):
    sample_prices.iloc[0, 0] = 0

    with pytest.raises(ValueError, match="positive"):
        validate_price_data(sample_prices)


def test_infinite_prices_rejected(sample_prices):
    sample_prices.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="infinite"):
        validate_price_data(sample_prices)


def test_all_missing_ticker_rejected(sample_prices):
    sample_prices["GOOGL"] = np.nan

    with pytest.raises(ValueError, match="Insufficient observations"):
        validate_price_data(sample_prices)


def test_no_common_dates_rejected():
    dates = pd.date_range("2026-01-01", periods=3)

    prices = pd.DataFrame(
        {
            "AAPL": [100, np.nan, np.nan],
            "MSFT": [np.nan, 200, np.nan],
        },
        index=dates,
    )

    with pytest.raises(ValueError, match="No dates"):
        get_common_trading_calendar(prices)


def test_minimum_observations_validation(sample_prices):
    with pytest.raises(ValueError, match="positive integer"):
        validate_price_data(sample_prices, min_observations=0)
