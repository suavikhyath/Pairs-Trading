from datetime import date

import pandas as pd

from src.statarb.data import pipeline
from src.statarb.data.storage import load_prices
from src.statarb.data.metadata import load_metadata


def test_pipeline_saves_prices_and_metadata(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {"KO": [60.0, 61.0], "PEP": [170.0, 171.0]},
        index=dates,
    )
    prices.index.name = "Date"

    monkeypatch.setattr(
        pipeline,
        "download_prices",
        lambda tickers, start, end: prices,
    )

    csv_path = tmp_path / "prices.csv"
    json_path = tmp_path / "prices.json"

    result = pipeline.download_and_store(
        ["KO", "PEP"],
        date(2024, 1, 1),
        date(2024, 2, 1),
        csv_path,
    )

    assert csv_path.exists()
    assert json_path.exists()

    loaded_prices = load_prices(csv_path)
    loaded_metadata = load_metadata(json_path)

    pd.testing.assert_frame_equal(loaded_prices, prices)
    pd.testing.assert_frame_equal(result, prices)

    assert loaded_metadata["source"] == "yfinance"
    assert loaded_metadata["tickers"] == ["KO", "PEP"]
    assert loaded_metadata["start_date"] == "2024-01-01"
    assert loaded_metadata["end_date"] == "2024-02-01"
