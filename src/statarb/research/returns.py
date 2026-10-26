
import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily percentage returns from prices."""
    if prices.empty:
        raise ValueError("Price DataFrame cannot be empty.")

    if prices.isna().all().any():
        raise ValueError("One or more tickers have no usable prices.")

    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    prices = prices.sort_index()
    return prices.pct_change(fill_method=None)
