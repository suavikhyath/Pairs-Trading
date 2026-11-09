
import numpy as np
import pandas as pd


def validate_price_data(
    prices: pd.DataFrame,
    min_observations: int = 2,
) -> dict:
    """
    Validate historical price data before backtesting.

    Checks:
    - Input is a non-empty DataFrame.
    - Index is a DatetimeIndex.
    - Dates are unique and chronologically sorted.
    - Ticker columns are unique.
    - Price columns contain numeric values.
    - No infinite values are present.
    - All observed prices are positive.
    - At least one common trading date exists.
    - Every ticker has sufficient observations.

    Returns a dictionary containing data-quality statistics.
    """

    # Validate input
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("Price data cannot be empty.")

    if (
        not isinstance(min_observations, int)
        or isinstance(min_observations, bool)
        or min_observations < 1
    ):
        raise ValueError(
            "min_observations must be a positive integer."
        )

    # Validate index
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price data must have a DatetimeIndex.")

    if prices.index.has_duplicates:
        raise ValueError("Dates must be unique.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "Price data must be sorted chronologically."
        )

    # Validate ticker columns
    if prices.columns.has_duplicates:
        raise ValueError("Ticker columns must be unique.")

    if len(prices.columns) == 0:
        raise ValueError(
            "Price data must contain at least one ticker."
        )

    # Validate numeric columns
    non_numeric = [
        ticker
        for ticker in prices.columns
        if not pd.api.types.is_numeric_dtype(prices[ticker])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric price columns found: {non_numeric}"
        )

    # Reject infinite values
    if np.isinf(prices.to_numpy(dtype=float)).any():
        raise ValueError("Price data contains infinite values.")

    # Reject zero or negative observed prices
    observed_prices = prices.stack()

    if (observed_prices <= 0).any():
        raise ValueError(
            "All observed prices must be positive."
        )

    # Check minimum observations per ticker
    observation_counts = prices.count()

    # An entirely missing ticker should be rejected first.
    all_missing = observation_counts[observation_counts == 0]

    if not all_missing.empty:
        raise ValueError(
            f"Insufficient observations for: "
            f"{list(all_missing.index)}"
        )

    # Check common trading dates
    common_dates = prices.dropna(how="any").index

    if common_dates.empty:
        raise ValueError(
            "No dates available with valid prices for every ticker."
        )

    # Check minimum observations after common-date validation
    insufficient = observation_counts[
        observation_counts < min_observations
    ]

    if not insufficient.empty:
        raise ValueError(
            f"Insufficient observations for: "
            f"{list(insufficient.index)}"
        )

    # Calculate missing-data statistics
    missing_counts = prices.isna().sum()
    missing_fractions = prices.isna().mean()

    dates_with_missing_prices = int(
        prices.isna().any(axis=1).sum()
    )

    return {
        "rows": len(prices),
        "tickers": len(prices.columns),
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "observation_counts": observation_counts.to_dict(),
        "missing_counts": missing_counts.to_dict(),
        "missing_fractions": missing_fractions.to_dict(),
        "dates_with_missing_prices": dates_with_missing_prices,
        "common_trading_dates": len(common_dates),
        "missing_data_present": bool(
            prices.isna().any().any()
        ),
    }


def get_common_trading_calendar(
    prices: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Return dates where every ticker has a valid price.

    Missing values are not forward-filled or interpolated.
    """

    validate_price_data(prices)

    common_dates = prices.dropna(how="any").index

    if common_dates.empty:
        raise ValueError(
            "No dates available with valid prices for every ticker."
        )

    return common_dates


def align_prices_to_common_calendar(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align price data to dates shared by all tickers.

    Returns a copy without modifying the original DataFrame.
    """

    common_dates = get_common_trading_calendar(prices)

    return prices.loc[common_dates].copy()
