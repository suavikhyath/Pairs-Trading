import pandas as pd
import pytest

from src.statarb.research.alignment import align_prices


def test_aligns_dates_and_drops_missing_values():
    dates_a = pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-01-03"
    ])
    dates_b = pd.to_datetime([
        "2024-01-01", "2024-01-03", "2024-01-04"
    ])

    price_a = pd.Series([10.0, 11.0, 12.0], index=dates_a)
    price_b = pd.Series([20.0, 22.0, 23.0], index=dates_b)

    result = align_prices(price_a, price_b, min_observations=2)

    assert len(result) == 2
    assert list(result.columns) == ["A", "B"]
    assert result.index.tolist() == list(pd.to_datetime([
        "2024-01-01", "2024-01-03"
    ]))


def test_rejects_duplicate_dates():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
    price_a = pd.Series([10.0, 11.0], index=dates)
    price_b = pd.Series([20.0, 21.0], index=dates)

    with pytest.raises(ValueError, match="Duplicate dates"):
        align_prices(price_a, price_b, min_observations=1)


def test_rejects_nonpositive_prices():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    price_a = pd.Series([10.0, 0.0], index=dates)
    price_b = pd.Series([20.0, 21.0], index=dates)

    with pytest.raises(ValueError, match="positive"):
        align_prices(price_a, price_b, min_observations=1)


def test_rejects_insufficient_observations():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    price_a = pd.Series([10.0, 11.0], index=dates)
    price_b = pd.Series([20.0, 21.0], index=dates)

    with pytest.raises(ValueError, match="Insufficient"):
        align_prices(price_a, price_b, min_observations=3)

