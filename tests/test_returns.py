import pandas as pd
import pytest

from src.statarb.research.returns import calculate_returns


def test_calculates_percentage_returns():
    prices = pd.DataFrame(
        {"KO": [100.0, 110.0, 99.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    returns = calculate_returns(prices)

    assert pd.isna(returns.iloc[0, 0])
    assert returns.iloc[1, 0] == pytest.approx(0.10)
    assert returns.iloc[2, 0] == pytest.approx(-0.10)


def test_missing_price_does_not_get_filled():
    prices = pd.DataFrame(
        {"KO": [100.0, float("nan"), 120.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    returns = calculate_returns(prices)

    assert pd.isna(returns.iloc[1, 0])
    assert pd.isna(returns.iloc[2, 0])


def test_empty_prices_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_returns(pd.DataFrame())


def test_nonpositive_prices_rejected():
    prices = pd.DataFrame({"KO": [100.0, 0.0]})

    with pytest.raises(ValueError, match="positive"):
        calculate_returns(prices)
