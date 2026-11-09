
import numpy as np

from src.statarb.strategy.position_sizing import calculate_position_size


def build_portfolio(
    pairs: list[dict],
    capital: float,
    max_gross_exposure: float = 0.50,
) -> dict:
    """
    Build a portfolio from multiple pairs.

    Each active pair receives an equal share of the
    portfolio's maximum permitted gross exposure.

    Each pair must contain:
        pair_id, position, price_a, price_b, hedge_ratio
    """

    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Capital must be positive and finite.")

    if (
        not np.isfinite(max_gross_exposure)
        or not 0 < max_gross_exposure <= 1
    ):
        raise ValueError(
            "Maximum gross exposure must be between 0 and 1."
        )

    if not isinstance(pairs, list):
        raise ValueError("Pairs must be provided as a list.")

    required_fields = {
        "pair_id",
        "position",
        "price_a",
        "price_b",
        "hedge_ratio",
    }

    pair_ids = []

    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Each pair must be a dictionary.")

        if not required_fields.issubset(pair):
            raise ValueError("Pair is missing required fields.")

        pair_id = pair["pair_id"]

        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("Pair ID must be a non-empty string.")

        if pair_id in pair_ids:
            raise ValueError(f"Duplicate pair ID: {pair_id}")

        pair_ids.append(pair_id)

        if pair["position"] not in (-1, 0, 1):
            raise ValueError("Position must be -1, 0, or 1.")

    active_pairs = [
        pair for pair in pairs if pair["position"] != 0
    ]

    if not active_pairs:
        return {
            "pairs": {},
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "capital": float(capital),
            "exposure_limit": float(
                capital * max_gross_exposure
            ),
        }

    # Equal allocation across active pairs.
    allocation_per_pair = max_gross_exposure / len(active_pairs)

    portfolio_pairs = {}

    total_long = 0.0
    total_short = 0.0

    for pair in active_pairs:
        result = calculate_position_size(
            position=pair["position"],
            price_a=pair["price_a"],
            price_b=pair["price_b"],
            hedge_ratio=pair["hedge_ratio"],
            capital=capital,
            allocation=allocation_per_pair,
            max_gross_exposure=allocation_per_pair,
        )

        portfolio_pairs[pair["pair_id"]] = result

        notionals = [
            result["notional_a"],
            result["notional_b"],
        ]

        total_long += sum(max(value, 0) for value in notionals)
        total_short += sum(abs(min(value, 0)) for value in notionals)

    gross_exposure = total_long + total_short
    net_exposure = total_long - total_short

    return {
        "pairs": portfolio_pairs,
        "long_exposure": float(total_long),
        "short_exposure": float(total_short),
        "gross_exposure": float(gross_exposure),
        "net_exposure": float(net_exposure),
        "capital": float(capital),
        "exposure_limit": float(
            capital * max_gross_exposure
        ),
    }
