from datetime import date

from src.statarb.data.provider import download_prices

prices = download_prices(
    tickers=["KO", "PEP"],
    start=date(2024, 1, 1),
    end=date(2025, 1, 1),
)

print(prices.head())
print("\nShape:", prices.shape)
print("\nMissing values:\n", prices.isna().sum())
