import pandas as pd


def calculate_zscore(
    spread: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate rolling z-scores using prior observations."""

    if spread.empty:
        raise ValueError("Spread cannot be empty.")

    if window < 2:
        raise ValueError("Window must be at least 2.")

    if spread.isna().any():
        raise ValueError("Spread cannot contain missing values.")

    if not pd.api.types.is_numeric_dtype(spread):
        raise ValueError("Spread must contain numeric values.")

    historical_mean = spread.rolling(
        window=window,
        min_periods=window,
    ).mean().shift(1)

    historical_std = spread.rolling(
        window=window,
        min_periods=window,
    ).std().shift(1)

    historical_std = historical_std.replace(0, float("nan"))

    zscore = (spread - historical_mean) / historical_std
    zscore.name = "zscore"

    return zscore