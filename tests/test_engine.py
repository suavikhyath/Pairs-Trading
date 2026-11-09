
import pandas as pd
import pytest

from src.statarb.backtesting.engine import run_backtest


@pytest.fixture
def sample_data():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")

    prices = pd.DataFrame(
        {
            "AAPL": [100, 110, 120, 130],
            "MSFT": [50, 50, 50, 50],
        },
        index=dates,
    )

    targets = pd.DataFrame(
        {
            "AAPL": [10, 10, 0, 0],
            "MSFT": [-20, -20, 0, 0],
        },
        index=dates,
    )

    return prices, targets


def test_signals_execute_on_next_bar(sample_data):
    prices, targets = sample_data

    result = run_backtest(
        prices,
        targets,
        transaction_cost=0,
        slippage=0,
    )

    trades = result["trades"]

    # First signal is generated on January 1.
    # It should execute on January 2.
    first_trade_date = trades["date"].min()

    assert first_trade_date == prices.index[1]


def test_initial_capital_preserved_without_positions():
    dates = pd.date_range("2026-01-01", periods=3)

    prices = pd.DataFrame(
        {"AAPL": [100, 110, 120]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [0, 0, 0]},
        index=dates,
    )

    result = run_backtest(prices, targets)

    assert result["final_equity"] == pytest.approx(100000)
    assert result["total_return"] == pytest.approx(0)


def test_buy_and_sell_trades():
    dates = pd.date_range("2026-01-01", periods=3)

    prices = pd.DataFrame(
        {"AAPL": [100, 110, 120]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [10, 0, 0]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        transaction_cost=0,
        slippage=0,
    )

    trades = result["trades"]

    assert len(trades) == 2
    assert trades.iloc[0]["quantity"] == 10
    assert trades.iloc[1]["quantity"] == -10


def test_transaction_costs_reduce_equity():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [10, 10]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        transaction_cost=0.01,
        slippage=0,
    )

    # Buy 10 shares at $100.
    # Commission = $10.
    assert result["final_equity"] == pytest.approx(99990)


def test_slippage_reduces_equity():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [10, 10]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        transaction_cost=0,
        slippage=0.01,
    )

    # Buy 10 shares at $101, mark at $100.
    assert result["final_equity"] == pytest.approx(99990)


def test_invalid_price_rejected():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, -10]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [10, 10]},
        index=dates,
    )

    with pytest.raises(ValueError):
        run_backtest(prices, targets)


def test_mismatched_dates_rejected():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 110]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [10]},
        index=dates[:1],
    )

    with pytest.raises(ValueError, match="dates"):
        run_backtest(prices, targets)


def test_mismatched_tickers_rejected():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 110]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"MSFT": [10, 10]},
        index=dates,
    )

    with pytest.raises(ValueError, match="tickers"):
        run_backtest(prices, targets)

def test_cash_decreases_after_buy():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame({"AAPL": [100, 100]}, index=dates)
    targets = pd.DataFrame({"AAPL": [10, 10]}, index=dates)

    result = run_backtest(
        prices, targets,
        transaction_cost=0,
        slippage=0,
    )

    assert result["equity_curve"].iloc[1]["cash"] == pytest.approx(99000)

def test_long_to_short_reversal():
    dates = pd.date_range("2026-01-01", periods=3)

    prices = pd.DataFrame({"AAPL": [100, 100, 100]}, index=dates)
    targets = pd.DataFrame({"AAPL": [10, -10, -10]}, index=dates)

    result = run_backtest(
        prices, targets,
        transaction_cost=0,
        slippage=0,
    )

    trades = result["trades"]

    assert list(trades["quantity"]) == [10, -20]

def test_equity_equals_cash_plus_market_value(sample_data):
    prices, targets = sample_data

    result = run_backtest(
        prices, targets,
        transaction_cost=0,
        slippage=0,
    )

    curve = result["equity_curve"]

    assert (
        curve["equity"]
        == curve["cash"] + curve["market_value"]
    ).all()


def test_exposure_limit_scales_positions():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [2000, 2000]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        initial_capital=100000,
        transaction_cost=0,
        slippage=0,
        max_gross_exposure=1.0,
    )

    curve = result["equity_curve"]

    assert curve.iloc[1]["gross_exposure"] == pytest.approx(100000)
    assert curve.iloc[1]["leverage"] == pytest.approx(1.0)
    assert curve.iloc[1]["exposure_scale"] == pytest.approx(0.5)


def test_positions_not_scaled_when_within_limit():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        initial_capital=100000,
        transaction_cost=0,
        slippage=0,
        max_gross_exposure=1.0,
    )

    assert (
        result["equity_curve"].iloc[1]["exposure_scale"]
        == pytest.approx(1.0)
    )


def test_leverage_limit_above_one():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [2000, 2000]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        initial_capital=100000,
        transaction_cost=0,
        slippage=0,
        max_gross_exposure=1.5,
    )

    assert result["equity_curve"].iloc[1]["leverage"] == pytest.approx(1.5)


def test_invalid_max_gross_exposure():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [10, 10]},
        index=dates,
    )

    with pytest.raises(ValueError):
        run_backtest(
            prices,
            targets,
            max_gross_exposure=0,
        )


def test_short_availability_limits_position():
    dates = pd.date_range("2026-01-01", periods=2)

    prices = pd.DataFrame(
        {"AAPL": [100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [-100, -100]},
        index=dates,
    )

    availability = pd.DataFrame(
        {"AAPL": [50, 50]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        borrow_availability=availability,
        transaction_cost=0,
        slippage=0,
    )

    assert result["equity_curve"].iloc[1]["market_value"] == -5000


def test_borrow_fee_is_charged():
    dates = pd.date_range("2026-01-01", periods=3)

    prices = pd.DataFrame(
        {"AAPL": [100, 100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [-10, -10, -10]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        annual_borrow_rate=0.252,
        transaction_cost=0,
        slippage=0,
    )

    # Short position begins on day 2.
    # Daily fee = 1000 * 0.252 / 252 = 1.
    assert result["total_borrow_fees"] == pytest.approx(2)


def test_forced_buy_in_when_availability_falls():
    dates = pd.date_range("2026-01-01", periods=3)

    prices = pd.DataFrame(
        {"AAPL": [100, 100, 100]},
        index=dates,
    )

    targets = pd.DataFrame(
        {"AAPL": [-100, -100, -100]},
        index=dates,
    )

    availability = pd.DataFrame(
        {"AAPL": [100, 100, 40]},
        index=dates,
    )

    result = run_backtest(
        prices,
        targets,
        borrow_availability=availability,
        transaction_cost=0,
        slippage=0,
    )

    trades = result["trades"]

    # Initially short 100; subsequently buy back 60.
    assert list(trades["quantity"]) == [-100, 60]
