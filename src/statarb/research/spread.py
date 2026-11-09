import pandas as pd


def calculate_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    alpha: float,
    hedge_ratio: float,
) -> pd.Series:
    """
    Calculate the spread:

    Spread = Price A - alpha - hedge_ratio * Price B
    """

    if price_a.index.has_duplicates or price_b.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    if pd.isna(alpha) or pd.isna(hedge_ratio):
        raise ValueError("Alpha and hedge ratio must be valid numbers.")

    aligned = pd.concat(
        [price_a.rename("A"), price_b.rename("B")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        raise ValueError("At least two overlapping observations are required.")

    if (aligned <= 0).any().any():
        raise ValueError("Prices must be positive.")

    spread = aligned["A"] - alpha - hedge_ratio * aligned["B"]
    spread.name = "spread"

    return spread
