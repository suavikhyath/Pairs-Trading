
import numpy as np


def execute_trades(
    target_positions: dict,
    current_positions: dict,
    market_prices: dict,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """
    Generate trades required to reach target positions.

    Positions are share quantities.
    Costs and slippage are fractions of traded notional.

    Positive quantity = buy.
    Negative quantity = sell.
    """

    for name, value in {
        "transaction_cost": transaction_cost,
        "slippage": slippage,
    }.items():
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    if not isinstance(target_positions, dict):
        raise ValueError("Target positions must be a dictionary.")

    if not isinstance(current_positions, dict):
        raise ValueError("Current positions must be a dictionary.")

    if not isinstance(market_prices, dict):
        raise ValueError("Market prices must be a dictionary.")

    tickers = set(target_positions) | set(current_positions)

    trades = {}
    total_traded_notional = 0.0
    total_transaction_cost = 0.0
    total_slippage_cost = 0.0

    for ticker in tickers:
        target = target_positions.get(ticker, 0.0)
        current = current_positions.get(ticker, 0.0)

        if (
            not np.isfinite(target)
            or not np.isfinite(current)
        ):
            raise ValueError("Position quantities must be finite.")

        if ticker not in market_prices:
            raise ValueError(f"Missing market price for {ticker}.")

        price = market_prices[ticker]

        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid market price for {ticker}.")

        quantity = target - current
        notional = abs(quantity * price)

        if quantity == 0:
            continue

        transaction_cost_amount = notional * transaction_cost
        slippage_cost_amount = notional * slippage

        execution_price = price * (
            1 + slippage if quantity > 0 else 1 - slippage
        )

        trades[ticker] = {
            "quantity": float(quantity),
            "market_price": float(price),
            "execution_price": float(execution_price),
            "traded_notional": float(notional),
            "transaction_cost": float(transaction_cost_amount),
            "slippage_cost": float(slippage_cost_amount),
            "total_cost": float(
                transaction_cost_amount + slippage_cost_amount
            ),
        }

        total_traded_notional += notional
        total_transaction_cost += transaction_cost_amount
        total_slippage_cost += slippage_cost_amount

    return {
        "trades": trades,
        "total_traded_notional": float(total_traded_notional),
        "total_transaction_cost": float(total_transaction_cost),
        "total_slippage_cost": float(total_slippage_cost),
        "total_cost": float(
            total_transaction_cost + total_slippage_cost
        ),
    }
