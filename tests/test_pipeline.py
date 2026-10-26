from datetime import date

import pandas as pd

from src.statarb.data import pipeline


def test_download_and_store_orchestrates_modules(monkeypatch, tmp_path):
    prices = pd.DataFrame(
        {"KO": [60.0, 61.0], "PEP": [170.0, 171.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    calls = {"download": 0, "prices": 0, "metadata": 0}

    def fake_download(tickers, start, end):
        calls["download"] += 1
        return prices

    def fake_save_prices(data, path):
        calls["prices"] += 1
        assert path == tmp_path / "prices.csv"
        pd.testing.assert_frame_equal(data, prices)

    def fake_create_metadata(tickers, start_date, end_date):
        return {
            "source": "yfinance",
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "downloaded_at": "test",
            "price_adjustment": "auto_adjust=True",
        }

    def fake_save_metadata(metadata, path):
        calls["metadata"] += 1
        assert path == tmp_path / "prices.json"

    monkeypatch.setattr(pipeline, "download_prices", fake_download)
    monkeypatch.setattr(pipeline, "save_prices", fake_save_prices)
    monkeypatch.setattr(pipeline, "create_metadata", fake_create_metadata)
    monkeypatch.setattr(pipeline, "save_metadata", fake_save_metadata)

    result = pipeline.download_and_store(
        ["KO", "PEP"],
        date(2024, 1, 1),
        date(2024, 2, 1),
        tmp_path / "prices.csv",
    )

    assert calls == {"download": 1, "prices": 1, "metadata": 1}
    pd.testing.assert_frame_equal(result, prices)
