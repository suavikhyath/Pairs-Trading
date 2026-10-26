import pandas as pd


def align_prices(
    price_a: pd.Series,
    price_b: pd.Series,
    min_observations: int = 100,
) -> pd.DataFrame:
    """Align two price series and retain dates with valid prices."""
    if min_observations < 1:
        raise ValueError("min_observations must be positive.")

    if price_a.index.has_duplicates or price_b.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    aligned = pd.concat(
        [price_a.rename("A"), price_b.rename("B")],
        axis=1,
        join="inner",
    ).dropna() # join = inner does not care about the values, it cares about joining cohering to the index values.

    if (aligned <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if len(aligned) < min_observations:
        raise ValueError("Insufficient overlapping observations.")

    return aligned.sort_index()

