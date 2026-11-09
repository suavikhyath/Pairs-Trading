
import numpy as np
import pandas as pd

from src.statarb.backtesting.metrics import (
    calculate_performance_metrics,
)


def run_backtest(
    prices: pd.DataFrame,
    target_positions: pd.DataFrame,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    max_gross_exposure: float = 1.0,
    borrow_availability: pd.DataFrame | None = None,
    annual_borrow_rate: float = 0.03,
) -> dict:
    """Run a chronological portfolio backtest."""

    # --------------------------------------------------
    # 1. Input validation
    # --------------------------------------------------

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty DataFrame.")

    if not isinstance(target_positions, pd.DataFrame):
        raise ValueError("Target positions must be a DataFrame.")

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be positive and finite.")

    for name, value in {
        "transaction_cost": transaction_cost,
        "slippage": slippage,
    }.items():
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    if (
        not np.isfinite(max_gross_exposure)
        or max_gross_exposure <= 0
    ):
        raise ValueError(
            "Maximum gross exposure must be positive and finite."
        )

    if (
        not np.isfinite(annual_borrow_rate)
        or annual_borrow_rate < 0
    ):
        raise ValueError(
            "Annual borrow rate must be non-negative and finite."
        )

    if prices.index.has_duplicates:
        raise ValueError("Price dates must be unique.")

    if target_positions.index.has_duplicates:
        raise ValueError("Target position dates must be unique.")

    if prices.columns.has_duplicates:
        raise ValueError("Ticker columns must be unique.")

    if target_positions.columns.has_duplicates:
        raise ValueError("Target position columns must be unique.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must have a DatetimeIndex.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be sorted chronologically.")

    if not target_positions.index.equals(prices.index):
        raise ValueError(
            "Target position dates must exactly match price dates."
        )

    if set(target_positions.columns) != set(prices.columns):
        raise ValueError(
            "Target position tickers must match price tickers."
        )

    target_positions = target_positions.reindex(
        columns=prices.columns
    )

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError(
            "Prices cannot contain missing or infinite values."
        )

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if not np.isfinite(target_positions.to_numpy()).all():
        raise ValueError(
            "Target positions cannot contain missing or infinite values."
        )

    if borrow_availability is not None:
        if not isinstance(borrow_availability, pd.DataFrame):
            raise ValueError(
                "Borrow availability must be a DataFrame."
            )

        if not borrow_availability.index.equals(prices.index):
            raise ValueError(
                "Borrow availability dates must match prices."
            )

        if set(borrow_availability.columns) != set(prices.columns):
            raise ValueError(
                "Borrow availability tickers must match prices."
            )

        borrow_availability = borrow_availability.reindex(
            columns=prices.columns
        )

        if not np.isfinite(borrow_availability.to_numpy()).all():
            raise ValueError(
                "Borrow availability must contain finite values."
            )

        if (borrow_availability < 0).any().any():
            raise ValueError(
                "Borrow availability cannot be negative."
            )

    # --------------------------------------------------
    # 2. Initialize simulation
    # --------------------------------------------------

    # Signals from day t execute on day t+1.
    executable_positions = (
        target_positions.shift(1).fillna(0)
    )

    cash = float(initial_capital)

    holdings = pd.Series(
        0.0,
        index=prices.columns,
        dtype=float,
    )

    equity_records = []
    trade_records = []

    total_borrow_fees = 0.0

    # --------------------------------------------------
    # 3. Daily simulation
    # --------------------------------------------------

    for date in prices.index:

        market_prices = prices.loc[date]

        targets = executable_positions.loc[date].astype(
            float
        ).copy()

        # ----------------------------------------------
        # Short availability
        # ----------------------------------------------

        if borrow_availability is not None:
            available = borrow_availability.loc[date]

            for ticker in targets.index:
                if targets[ticker] < 0:
                    targets[ticker] = max(
                        targets[ticker],
                        -float(available[ticker]),
                    )

        # ----------------------------------------------
        # Calculate pre-trade equity
        # ----------------------------------------------

        pre_trade_equity = float(
            cash + (holdings * market_prices).sum()
        )

        if pre_trade_equity <= 0:
            raise ValueError(
                "Portfolio equity must remain positive."
            )

        # ----------------------------------------------
        # Enforce gross exposure limit
        # ----------------------------------------------

        desired_gross_exposure = float(
            (targets.abs() * market_prices).sum()
        )

        exposure_limit = (
            pre_trade_equity * max_gross_exposure
        )

        exposure_scale = 1.0

        if desired_gross_exposure > exposure_limit:
            exposure_scale = (
                exposure_limit / desired_gross_exposure
            )
            targets *= exposure_scale

        # ----------------------------------------------
        # Execute required trades
        # ----------------------------------------------

        trade_quantities = targets - holdings

        for ticker, quantity in trade_quantities.items():

            if quantity == 0:
                continue

            market_price = float(market_prices[ticker])

            execution_price = market_price * (
                1 + slippage if quantity > 0
                else 1 - slippage
            )

            notional = abs(quantity * market_price)

            commission = notional * transaction_cost

            # Update cash using actual execution price.
            cash -= quantity * execution_price
            cash -= commission

            slippage_cost = abs(
                quantity * (execution_price - market_price)
            )

            trade_records.append({
                "date": date,
                "ticker": ticker,
                "quantity": float(quantity),
                "market_price": market_price,
                "execution_price": float(execution_price),
                "traded_notional": float(notional),
                "transaction_cost": float(commission),
                "slippage_cost": float(slippage_cost),
                "total_cost": float(
                    commission + slippage_cost
                ),
            })

        # ----------------------------------------------
        # IMPORTANT: Update holdings ONCE per day
        # ----------------------------------------------

        holdings = targets.astype(float).copy()

        # ----------------------------------------------
        # Calculate daily borrow fees
        # ----------------------------------------------

        short_market_value = float(
            (
                holdings.clip(upper=0).abs()
                * market_prices
            ).sum()
        )

        daily_borrow_fee = (
            short_market_value * annual_borrow_rate / 252
        )

        cash -= daily_borrow_fee
        total_borrow_fees += daily_borrow_fee

        # ----------------------------------------------
        # Calculate portfolio equity and exposure
        # ----------------------------------------------

        market_value = float(
            (holdings * market_prices).sum()
        )

        equity = cash + market_value

        gross_exposure = float(
            (holdings.abs() * market_prices).sum()
        )

        net_exposure = market_value

        leverage = (
            gross_exposure / equity
            if equity > 0
            else np.nan
        )

        # ----------------------------------------------
        # IMPORTANT: Record equity ONCE per day
        # ----------------------------------------------

        equity_records.append({
            "date": date,
            "cash": float(cash),
            "market_value": market_value,
            "equity": equity,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "exposure_limit": float(exposure_limit),
            "exposure_scale": float(exposure_scale),
            "leverage": leverage,
            "borrow_fee": float(daily_borrow_fee),
        })

    # --------------------------------------------------
    # 4. Construct results
    # --------------------------------------------------

    equity_curve = (
        pd.DataFrame(equity_records)
        .set_index("date")
    )

    equity_curve["daily_return"] = (
        equity_curve["equity"].pct_change().fillna(0)
    )

    trade_columns = [
        "date",
        "ticker",
        "quantity",
        "market_price",
        "execution_price",
        "traded_notional",
        "transaction_cost",
        "slippage_cost",
        "total_cost",
    ]

    trades = pd.DataFrame(
        trade_records,
        columns=trade_columns,
    )

    # --------------------------------------------------
    # 5. Performance metrics
    # --------------------------------------------------

    performance_metrics = calculate_performance_metrics(
        equity_curve
    )

    # --------------------------------------------------
    # 6. Return results
    # --------------------------------------------------

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "final_equity": float(
            equity_curve["equity"].iloc[-1]
        ),
        "total_return": float(
            equity_curve["equity"].iloc[-1]
            / initial_capital - 1
        ),
        "total_transaction_cost": float(
            trades["transaction_cost"].sum()
        ),
        "total_slippage_cost": float(
            trades["slippage_cost"].sum()
        ),
        "total_borrow_fees": float(total_borrow_fees),
        "performance_metrics": performance_metrics,
    }
