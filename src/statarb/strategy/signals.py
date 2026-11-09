
import numpy as np
import pandas as pd


def generate_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
) -> pd.Series:
    """
    Generate stateful pairs-trading positions.

    +1 = Long spread (long A, short B)
    -1 = Short spread (short A, long B)
     0 = No position
    """

    if zscore.empty:
        raise ValueError("Z-score series cannot be empty.")

    if not pd.api.types.is_numeric_dtype(zscore):
        raise ValueError("Z-scores must be numeric.")

    if zscore.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    if not np.isfinite(zscore.dropna()).all():
        raise ValueError("Z-scores must contain finite values.")

    if not np.isfinite(entry_threshold) or entry_threshold <= 0:
        raise ValueError("Entry threshold must be positive and finite.")

    if not np.isfinite(exit_threshold) or exit_threshold < 0:
        raise ValueError("Exit threshold must be non-negative and finite.")

    if exit_threshold >= entry_threshold:
        raise ValueError("Exit threshold must be smaller than entry threshold.")

    positions = []
    current_position = 0

    for value in zscore:
        # No trading decision when the z-score is unavailable.
        if pd.isna(value):
            positions.append(current_position)
            continue

        if current_position == 0:
            if value <= -entry_threshold:
                current_position = 1

            elif value >= entry_threshold:
                current_position = -1

        elif current_position == 1:
            if value >= -exit_threshold:
                current_position = 0

        elif current_position == -1:
            if value <= exit_threshold:
                current_position = 0

        positions.append(current_position)

    return pd.Series(
        positions,
        index=zscore.index,
        name="position",
        dtype=int,
    )
