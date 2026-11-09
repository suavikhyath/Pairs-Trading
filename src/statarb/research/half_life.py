import numpy as np
import pandas as pd
import statsmodels.api as sm


from src.statarb.research.stationarity import check_stationarity


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life using:

    ΔS(t) = alpha + beta * S(t-1) + error

    Half-life = -ln(2) / beta
    """

    if spread.empty:
        raise ValueError("Spread cannot be empty.")

    if spread.isna().any():
        raise ValueError("Spread cannot contain missing values.")

    if not pd.api.types.is_numeric_dtype(spread):
        raise ValueError("Spread must contain numeric values.")

    if len(spread) < 30:
        raise ValueError("At least 30 observations are required.")

    if spread.nunique() < 2:
        raise ValueError("Spread must not be constant.")

    stationarity_result = check_stationarity(spread)

    if not stationarity_result["is_stationary"]:
        raise ValueError("Spread is not sufficiently stationary or mean-reverting.")

    lagged_spread = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()

    X = sm.add_constant(lagged_spread)
    model = sm.OLS(delta_spread, X).fit()

    beta = float(model.params.iloc[1])

    if beta >= 0:
        raise ValueError("Spread does not exhibit mean-reverting dynamics.")

    half_life = -np.log(2) / beta

    if not np.isfinite(half_life):
        raise ValueError("Unable to calculate a finite half-life.")

    return float(half_life)