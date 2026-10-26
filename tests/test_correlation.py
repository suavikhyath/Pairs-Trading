import pandas as pd
import pytest

from src.statarb.research.correlation import calculate_correlation


def test_returns_correlation_matrix():
    returns = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],
        "B": [2.0, 4.0, 6.0],
        "C": [3.0, 2.0, 1.0],
    })

    result = calculate_correlation(returns)

    assert result.loc["A", "B"] == pytest.approx(1.0)
    assert result.loc["A", "C"] == pytest.approx(-1.0)
    assert result.loc["A", "A"] == pytest.approx(1.0)


def test_empty_returns_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_correlation(pd.DataFrame())


def test_duplicate_tickers_rejected():
    returns = pd.DataFrame([[0.1, 0.2]], columns=["A", "A"])

    with pytest.raises(ValueError, match="Duplicate ticker"):
        calculate_correlation(returns)


def test_all_missing_ticker_rejected():
    returns = pd.DataFrame({
        "A": [0.1, 0.2],
        "B": [float("nan"), float("nan")],
    })

    with pytest.raises(ValueError, match="no usable returns"):
        calculate_correlation(returns)


def test_insufficient_observations():
    returns = pd.DataFrame({"A": [0.1]})

    with pytest.raises(ValueError, match="At least two"):
        calculate_correlation(returns)

