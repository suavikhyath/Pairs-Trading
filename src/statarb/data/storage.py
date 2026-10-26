from pathlib import Path

import pandas as pd


def save_prices(prices: pd.DataFrame, path: str | Path) -> None:
    """Save a price DataFrame to CSV, preserving its date index."""
    if prices.empty:
        raise ValueError("Cannot save an empty price DataFrame.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    if prices.isna().all().any():
        raise ValueError("One or more tickers have no usable prices.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True) # don't raise an error if the directory already exists

    prices.to_csv(destination, index_label="Date")


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load prices from CSV, parsing the Date column as the index."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"Price file not found: {source}")

    prices = pd.read_csv(source, index_col="Date", parse_dates=True)
    prices.index.name = "Date"

    if prices.empty:
        raise ValueError("Loaded price file is empty.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    if prices.isna().all().any():
        raise ValueError("One or more tickers have no usable prices.")

    return prices.sort_index()
