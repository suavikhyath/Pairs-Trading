import pytest

from src.statarb.strategy.position_sizing import calculate_position_size


def test_long_spread_position():
    result = calculate_position_size(
        position=1,
        price_a=100,
        price_b=50,
        hedge_ratio=2,
        capital=100000,
        allocation=0.10,
    )

    assert result["shares_a"] == pytest.approx(50)
    assert result["shares_b"] == pytest.approx(-100)
    assert result["gross_exposure"] == pytest.approx(10000)


def test_short_spread_position():
    result = calculate_position_size(
        position=-1,
        price_a=100,
        price_b=50,
        hedge_ratio=2,
        capital=100000,
        allocation=0.10,
    )

    assert result["shares_a"] == pytest.approx(-50)
    assert result["shares_b"] == pytest.approx(100)
    assert result["gross_exposure"] == pytest.approx(10000)


def test_zero_position():
    result = calculate_position_size(
        position=0,
        price_a=100,
        price_b=50,
        hedge_ratio=2,
        capital=100000,
    )

    assert result["shares_a"] == 0
    assert result["shares_b"] == 0
    assert result["gross_exposure"] == 0


def test_maximum_exposure_limit():
    result = calculate_position_size(
        position=1,
        price_a=100,
        price_b=50,
        hedge_ratio=2,
        capital=100000,
        allocation=0.8,
        max_gross_exposure=0.5,
    )

    assert result["gross_exposure"] == pytest.approx(50000)


def test_negative_hedge_ratio():
    result = calculate_position_size(
        position=1,
        price_a=100,
        price_b=50,
        hedge_ratio=-1,
        capital=100000,
        allocation=0.10,
    )

    assert result["shares_a"] > 0
    assert result["shares_b"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": 2},
        {"price_a": -100},
        {"price_b": 0},
        {"hedge_ratio": float("nan")},
        {"capital": -1000},
        {"allocation": 1.5},
        {"max_gross_exposure": 2},
    ],
)
def test_invalid_inputs_rejected(kwargs):
    arguments = {
        "position": 1,
        "price_a": 100,
        "price_b": 50,
        "hedge_ratio": 2,
        "capital": 100000,
    }

    arguments.update(kwargs)

    with pytest.raises(ValueError):
        calculate_position_size(**arguments)