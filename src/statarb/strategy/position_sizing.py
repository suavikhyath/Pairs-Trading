import numpy as np


def calculate_position_size(
    position: int,
    price_a: float,
    price_b: float,
    hedge_ratio: float,
    capital: float,
    allocation: float = 0.10,
    max_gross_exposure: float = 1.0,
) -> dict:
    """
    Calculate stock quantities for a pairs-trading position.

    position:
        +1 = Long spread
        -1 = Short spread
         0 = No position

    Returns fractional share quantities and exposure metrics.
    """

    if position not in (-1, 0, 1):
        raise ValueError("Position must be -1, 0, or 1.")

    if not np.isfinite(price_a) or price_a <= 0:
        raise ValueError("Price A must be positive and finite.")

    if not np.isfinite(price_b) or price_b <= 0:
        raise ValueError("Price B must be positive and finite.")

    if not np.isfinite(hedge_ratio):
        raise ValueError("Hedge ratio must be finite.")

    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Capital must be positive and finite.")

    if not np.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError("Allocation must be between 0 and 1.")

    if (
        not np.isfinite(max_gross_exposure)
        or not 0 < max_gross_exposure <= 1
    ):
        raise ValueError("Maximum gross exposure must be between 0 and 1.")

    if position == 0:
        return {
            "shares_a": 0.0,
            "shares_b": 0.0,
            "notional_a": 0.0,
            "notional_b": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        }

    target_exposure = min(
        capital * allocation,
        capital * max_gross_exposure,
    )

    denominator = price_a + abs(hedge_ratio) * price_b

    shares_a = position * target_exposure / denominator
    shares_b = -position * hedge_ratio * target_exposure / denominator

    notional_a = shares_a * price_a
    notional_b = shares_b * price_b

    gross_exposure = abs(notional_a) + abs(notional_b)
    net_exposure = notional_a + notional_b

    return {
        "shares_a": float(shares_a),
        "shares_b": float(shares_b),
        "notional_a": float(notional_a),
        "notional_b": float(notional_b),
        "gross_exposure": float(gross_exposure),
        "net_exposure": float(net_exposure),
    }