import pandas as pd


def calculate_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Pearson correlations between stock returns."""
    if returns.empty:
        raise ValueError("Returns DataFrame cannot be empty.")

    if returns.columns.duplicated().any():
        raise ValueError("Duplicate ticker columns found.")

    if returns.isna().all().any():
        raise ValueError("One or more tickers have no usable returns.")

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    return returns.corr(method="pearson", min_periods=2)


# we're using returns for correlation, not the price series