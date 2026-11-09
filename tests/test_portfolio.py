
import pytest

from src.statarb.strategy.portfolio import build_portfolio


@pytest.fixture
def sample_pairs():
    return [
        {
            "pair_id": "AAPL_MSFT",
            "position": 1,
            "price_a": 100,
            "price_b": 50,
            "hedge_ratio": 2,
        },
        {
            "pair_id": "GOOGL_META",
            "position": -1,
            "price_a": 200,
            "price_b": 100,
            "hedge_ratio": 1,
        },
    ]


def test_equal_allocation_across_pairs(sample_pairs):
    result = build_portfolio(
        pairs=sample_pairs,
        capital=100000,
        max_gross_exposure=0.50,
    )

    assert result["gross_exposure"] == pytest.approx(50000)

    for pair in result["pairs"].values():
        assert pair["gross_exposure"] == pytest.approx(25000)


def test_portfolio_exposure_limit(sample_pairs):
    result = build_portfolio(
        pairs=sample_pairs,
        capital=100000,
        max_gross_exposure=0.30,
    )

    assert result["gross_exposure"] <= 30000


def test_long_and_short_exposure(sample_pairs):
    result = build_portfolio(
        pairs=sample_pairs,
        capital=100000,
        max_gross_exposure=0.50,
    )

    assert result["long_exposure"] > 0
    assert result["short_exposure"] > 0


def test_net_exposure_calculation(sample_pairs):
    result = build_portfolio(
        pairs=sample_pairs,
        capital=100000,
        max_gross_exposure=0.50,
    )

    assert result["net_exposure"] == pytest.approx(
        result["long_exposure"] - result["short_exposure"]
    )


def test_zero_position_pairs_are_ignored(sample_pairs):
    sample_pairs[0]["position"] = 0

    result = build_portfolio(
        pairs=sample_pairs,
        capital=100000,
        max_gross_exposure=0.50,
    )

    assert len(result["pairs"]) == 1
    assert "GOOGL_META" in result["pairs"]
    assert result["gross_exposure"] == pytest.approx(50000)


def test_empty_portfolio():
    result = build_portfolio(
        pairs=[],
        capital=100000,
    )

    assert result["gross_exposure"] == 0
    assert result["net_exposure"] == 0
    assert result["pairs"] == {}


def test_duplicate_pair_ids_rejected(sample_pairs):
    sample_pairs[1]["pair_id"] = "AAPL_MSFT"

    with pytest.raises(ValueError, match="Duplicate"):
        build_portfolio(sample_pairs, capital=100000)


def test_invalid_exposure_limit(sample_pairs):
    with pytest.raises(ValueError):
        build_portfolio(
            sample_pairs,
            capital=100000,
            max_gross_exposure=1.5,
        )


def test_invalid_capital(sample_pairs):
    with pytest.raises(ValueError):
        build_portfolio(sample_pairs, capital=-1000)


def test_missing_pair_fields():
    with pytest.raises(ValueError, match="missing required"):
        build_portfolio(
            pairs=[{"pair_id": "AAPL_MSFT", "position": 1}],
            capital=100000,
        )
