from datetime import date

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Download adjusted daily closing prices for the requested tickers."""
    if not tickers:
        raise ValueError("At least one ticker is required.")

    if start >= end:
        raise ValueError("'start' must be earlier than 'end'.")

    normalized_tickers = [ticker.strip().upper() for ticker in tickers]

    if any(not ticker for ticker in normalized_tickers):
        raise ValueError("Ticker symbols cannot be empty.")

    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError("Duplicate ticker symbols are not allowed.")

    data = yf.download(
        tickers=normalized_tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column", # determines the upper layer, here, we've set it to the OHLCV
        # multi_level_index=False, 
    )

    if data.empty:
        raise ValueError("No market data. Check tickers and dates.")
    

    if "Close" not in data.columns.get_level_values(0) and "Close" not in data.columns:
        raise ValueError("Downloaded data does not contain closing prices.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = normalized_tickers
    
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices.columns.name = None

    # Keep only requested tickers that Yahoo actually returned.
    # prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()

    if prices.index.has_duplicates:
        raise ValueError("Duplicate trading dates found.")

    if prices.empty or prices.shape[1] == 0:
        raise ValueError("No usable price series returned.")


    # print("DEBUG — prices:")
    # print(prices)
    # print("DEBUG — all missing:", prices.isna().all())

    if prices.isna().all().any():
        raise ValueError("One or more tickers have no usable prices.")
    
    
    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError("Non-positive prices found.")
    
    
    return prices
