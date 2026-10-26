import json

import pytest

from src.statarb.data.metadata import (
    create_metadata,
    save_metadata,
    load_metadata,
)


def test_create_metadata():
    metadata = create_metadata(
        ["ko", "pep"], "2024-01-01", "2025-01-01"
    )

    assert metadata["tickers"] == ["KO", "PEP"]
    assert metadata["source"] == "yfinance"
    assert metadata["price_adjustment"] == "auto_adjust=True"
    assert metadata["downloaded_at"]


def test_invalid_date_range_rejected():
    with pytest.raises(ValueError, match="start_date"):
        create_metadata(["KO"], "2025-01-01", "2024-01-01")


def test_save_and_load_metadata(tmp_path):
    metadata = create_metadata(
        ["KO"], "2024-01-01", "2025-01-01"
    )
    path = tmp_path / "prices.json"

    save_metadata(metadata, path)

    assert load_metadata(path) == metadata


def test_missing_fields_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"source": "yfinance"}))

    with pytest.raises(ValueError, match="missing required fields"):
        load_metadata(path)


def test_malformed_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{invalid json")

    with pytest.raises(ValueError, match="Invalid metadata JSON"):
        load_metadata(path)
