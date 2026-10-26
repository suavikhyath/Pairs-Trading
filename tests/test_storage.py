
import pandas as pd
import pytest

from src.statarb.data.storage import save_prices, load_prices


def test_save_and_load_prices(tmp_path):
    dates = pd.to_datetime([
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ])

    original = pd.DataFrame(
        {"KO": [60.0, 61.0, 62.0], "PEP": [170.0, 171.0, 172.0]},
        index=dates,
    )
    original.index.name = "Date"

    path = tmp_path / "prices.csv"

    save_prices(original, path)
    loaded = load_prices(path)

    pd.testing.assert_frame_equal(loaded, original)


def test_save_rejects_empty_dataframe(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_prices(pd.DataFrame(), tmp_path / "prices.csv")


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prices(tmp_path / "missing.csv")


def test_load_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "Date,KO\n2024-01-02,60.0\n2024-01-02,61.0\n"
    )

    with pytest.raises(ValueError, match="Duplicate dates"):
        load_prices(path)
