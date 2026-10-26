import pandas as pd


def rank_pairs(
    correlation: pd.DataFrame,
    min_correlation: float = 0.5,
) -> pd.DataFrame:
    """Rank unique stock pairs by absolute correlation."""

    if correlation.empty:
        raise ValueError("Correlation matrix cannot be empty.")

    if not -1 <= min_correlation <= 1:
        raise ValueError("min_correlation must be between -1 and 1.")

    if correlation.index.has_duplicates or correlation.columns.has_duplicates:
        raise ValueError("Duplicate tickers found.")

    if not correlation.index.equals(correlation.columns):
        raise ValueError("Correlation matrix must have matching row and column labels.")

    pairs = []

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            ticker_a = correlation.columns[i]
            ticker_b = correlation.columns[j]
            corr = correlation.iloc[i, j]

            if pd.isna(corr):
                continue

            if abs(corr) >= min_correlation:
                pairs.append({
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                })

    result = pd.DataFrame(
        pairs,
        columns=["ticker_a", "ticker_b", "correlation", "abs_correlation"],
    )

    return result.sort_values(
        "abs_correlation", ascending=False
    ).reset_index(drop=True)