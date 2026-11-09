import pandas as pd
from statsmodels.tsa.stattools import adfuller


def check_stationarity(
    spread: pd.Series,
    significance_level: float = 0.05,
) -> dict:
    """Test spread stationarity using the Augmented Dickey-Fuller test."""

    if spread.empty:
        raise ValueError("Spread cannot be empty.")

    if not 0 < significance_level < 1:
        raise ValueError("Significance level must be between 0 and 1.")

    if spread.isna().any():
        raise ValueError("Spread cannot contain missing values.")

    if len(spread) < 30:
        raise ValueError("At least 30 observations are required.")

    if spread.nunique() < 2:
        raise ValueError("Spread must not be constant.")

    result = adfuller(
    spread,
    regression="c",
    autolag="AIC",
    result_object=True,
    )

    statistic = result.statistic
    p_value = result.pvalue
    
    observations = result.nobs
    critical_values = result.critical_values

    return {
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "critical_value_1pct": float(critical_values["1%"]),
        "critical_value_5pct": float(critical_values["5%"]),
        "critical_value_10pct": float(critical_values["10%"]),
        "observations": int(observations),
        "is_stationary": bool(p_value < significance_level),
    }
