import numpy as np
import pandas as pd


def adjust_pvalues(p_values: pd.Series) -> pd.Series:
    """Adjust p-values using the Benjamini-Hochberg procedure."""

    if p_values.empty:
        raise ValueError("P-values cannot be empty.")

    if p_values.isna().any():
        raise ValueError("P-values cannot contain missing values.")

    if ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("P-values must be between 0 and 1.")

    values = p_values.to_numpy(dtype=float)
    n = len(values)

    order = np.argsort(values)
    sorted_p = values[order]
    ranks = np.arange(1, n + 1)

    adjusted_sorted = sorted_p * n / ranks # testing the raw p-value against its 
    # rank-specific threshold is equivalent to testing the preliminary adjusted p-value against \(q\)

    # Enforce monotonicity, then cap adjusted p-values at 1.
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return pd.Series(
        adjusted,
        index=p_values.index,
        name="adjusted_p_value",
    )
