from datetime import date
from pathlib import Path

import pandas as pd

from src.statarb.data.provider import download_prices
from src.statarb.data.storage import save_prices
from src.statarb.data.metadata import create_metadata, save_metadata


def download_and_store(
    tickers: list[str],
    start: date,
    end: date,
    output_path: str | Path,
) -> pd.DataFrame:
    """Download prices, save the CSV, and write its metadata."""
    prices = download_prices(tickers, start, end)

    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json") # with_suffix actually changes .csv to .json

    metadata = create_metadata(
        tickers=tickers,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )

    save_prices(prices, output_path)
    save_metadata(metadata, metadata_path)

    return prices
