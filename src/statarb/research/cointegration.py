import pandas as pd
from statsmodels.tsa.stattools import coint


def check_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    significance_level: float = 0.05,
) -> dict:
    """Test two price series for cointegration using Engle-Granger."""

    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be between 0 and 1.")

    if price_a.empty or price_b.empty:
        raise ValueError("Price series cannot be empty.")

    if price_a.index.has_duplicates or price_b.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    aligned = pd.concat(
        [price_a.rename("A"), price_b.rename("B")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 30:
        raise ValueError("At least 30 overlapping observations are required.")

    if (aligned <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if aligned.nunique().min() < 2:
        raise ValueError("Price series must not be constant.")

    statistic, p_value, critical_values = coint(
        aligned["A"],
        aligned["B"],
        trend="c",
        autolag="aic",
    )

    return {
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "critical_value_1pct": float(critical_values[0]),
        "critical_value_5pct": float(critical_values[1]),
        "critical_value_10pct": float(critical_values[2]),
        "significance_level": significance_level,
        "is_cointegrated": bool(p_value < significance_level),
        "observations": len(aligned),
    }
