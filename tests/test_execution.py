
import pytest

from src.statarb.strategy.execution import execute_trades


def test_new_long_position():
    result = execute_trades(
        target_positions={"AAPL": 10},
        current_positions={},
        market_prices={"AAPL": 100},
    )

    assert result["trades"]["AAPL"]["quantity"] == 10
    assert result["total_traded_notional"] == 1000


def test_new_short_position():
    result = execute_trades(
        target_positions={"AAPL": -10},
        current_positions={},
        market_prices={"AAPL": 100},
    )

    assert result["trades"]["AAPL"]["quantity"] == -10


def test_position_reduction():
    result = execute_trades(
        target_positions={"AAPL": 5},
        current_positions={"AAPL": 10},
        market_prices={"AAPL": 100},
    )

    assert result["trades"]["AAPL"]["quantity"] == -5


def test_no_trade_when_positions_match():
    result = execute_trades(
        target_positions={"AAPL": 10},
        current_positions={"AAPL": 10},
        market_prices={"AAPL": 100},
    )

    assert result["trades"] == {}
    assert result["total_cost"] == 0


def test_transaction_cost_calculation():
    result = execute_trades(
        target_positions={"AAPL": 10},
        current_positions={},
        market_prices={"AAPL": 100},
        transaction_cost=0.001,
        slippage=0.0005,
    )

    assert result["total_transaction_cost"] == pytest.approx(1)
    assert result["total_slippage_cost"] == pytest.approx(0.5)
    assert result["total_cost"] == pytest.approx(1.5)


def test_buy_slippage():
    result = execute_trades(
        target_positions={"AAPL": 10},
        current_positions={},
        market_prices={"AAPL": 100},
        slippage=0.01,
    )

    assert result["trades"]["AAPL"]["execution_price"] == pytest.approx(101)


def test_sell_slippage():
    result = execute_trades(
        target_positions={"AAPL": -10},
        current_positions={},
        market_prices={"AAPL": 100},
        slippage=0.01,
    )

    assert result["trades"]["AAPL"]["execution_price"] == pytest.approx(99)


def test_missing_market_price():
    with pytest.raises(ValueError, match="Missing market price"):
        execute_trades(
            target_positions={"AAPL": 10},
            current_positions={},
            market_prices={},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transaction_cost": -0.1},
        {"transaction_cost": 1.5},
        {"slippage": -0.1},
        {"slippage": float("nan")},
    ],
)
def test_invalid_cost_parameters(kwargs):
    arguments = {
        "target_positions": {"AAPL": 10},
        "current_positions": {},
        "market_prices": {"AAPL": 100},
    }

    arguments.update(kwargs)

    with pytest.raises(ValueError):
        execute_trades(**arguments)
