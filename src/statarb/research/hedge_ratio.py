import pandas as pd
import statsmodels.api as sm


def estimate_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
) -> dict:
    """Estimate the OLS relationship: A = alpha + beta * B + error."""

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

    X = sm.add_constant(aligned["B"])
    model = sm.OLS(aligned["A"], X).fit()

    return {
        "alpha": float(model.params["const"]),
        "hedge_ratio": float(model.params["B"]),
        "r_squared": float(model.rsquared),
        "observations": len(aligned),
    }
