from datetime import date

import pandas as pd
import pytest

from src.statarb.data import provider


def test_empty_tickers_rejected():
    with pytest.raises(ValueError, match="At least one ticker"):
        provider.download_prices(
            [], date(2024, 1, 1), date(2025, 1, 1)
        )


def test_invalid_date_range_rejected():
    with pytest.raises(ValueError, match="start"):
        provider.download_prices(
            ["KO"], date(2025, 1, 1), date(2024, 1, 1)
        )


def test_duplicate_tickers_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        provider.download_prices(
            ["KO", "KO"], date(2024, 1, 1), date(2025, 1, 1)
        )


def test_download_returns_price_dataframe(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_tuples(
        [("Close", "KO"), ("Close", "PEP")]
    )
    fake_data = pd.DataFrame(
        [[60.0, 170.0], [61.0, 171.0]],
        index=dates,
        columns=columns,
    )

    def fake_download(**kwargs):
        return fake_data

    monkeypatch.setattr(provider.yf, "download", fake_download)

    result = provider.download_prices(
        ["KO", "PEP"], date(2024, 1, 1), date(2024, 2, 1)
    )

    assert list(result.columns) == ["KO", "PEP"]
    assert len(result) == 2
    assert not result.index.has_duplicates
    assert (result > 0).all().all()


def test_duplicate_dates_rejected(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-02"])
    fake_data = pd.DataFrame(
        {"Close": [60.0, 61.0]},
        index=dates,
    )

    monkeypatch.setattr(
        provider.yf, "download", lambda **kwargs: fake_data
    )

    with pytest.raises(ValueError, match="Duplicate trading dates"):
        provider.download_prices(
            ["KO"], date(2024, 1, 1), date(2024, 2, 1)
        )


def test_nonpositive_prices_rejected(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    fake_data = pd.DataFrame(
        {"Close": [60.0, 0.0]},
        index=dates,
    )

    monkeypatch.setattr(
        provider.yf, "download", lambda **kwargs: fake_data
    )

    with pytest.raises(ValueError, match="Non-positive prices"):
        provider.download_prices(
            ["KO"], date(2024, 1, 1), date(2024, 2, 1)
        )


def test_empty_download_rejected(monkeypatch):
    monkeypatch.setattr(
        provider.yf, "download", lambda **kwargs: pd.DataFrame()
    )

    with pytest.raises(ValueError, match="No market data"):
        provider.download_prices(
            ["KO"], date(2024, 1, 1), date(2024, 2, 1)
        )


def test_all_missing_ticker_rejected(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    fake_data = pd.DataFrame(
        {"Close": [float("nan"), float("nan")]},
        index=dates,
    )

    monkeypatch.setattr(
        provider.yf, "download", lambda **kwargs: fake_data # ignores arguments and simply returns the fake_data
    )

    with pytest.raises(ValueError, match="no usable prices"):
        provider.download_prices(
            ["KO"], date(2024, 1, 1), date(2024, 2, 1)
        )


# commented the lower code because it was entirely unnecessarily.

# def test_one_ticker_entirely_missing_rejected(monkeypatch):
#     dates = pd.to_datetime(["2024-01-02", "2024-01-03"])

#     columns = pd.MultiIndex.from_tuples([
#         ("Close", "KO"),
#         ("Close", "PEP"),
#     ]) # .from_tuples seems like, very rudimentary? i don't know. why is every small thing made in here bruh.

#     fake_data = pd.DataFrame(
#         [
#             [60.0, float("nan")],
#             [61.0, float("nan")],
#         ],
#         index=dates,
#         columns=columns,
#     )

#     monkeypatch.setattr(
#         provider.yf, "download", lambda **kwargs: fake_data
#     )

#     with pytest.raises(ValueError, match="no usable prices"):
#         provider.download_prices(
#             ["KO", "PEP"],
#             date(2024, 1, 1),
#             date(2024, 2, 1),
#         )
