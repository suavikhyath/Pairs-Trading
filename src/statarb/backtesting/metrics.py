
import numpy as np
import pandas as pd


def calculate_cagr(equity: pd.Series) -> float:
    """Calculate compound annual growth rate."""

    if equity.empty or len(equity) < 2:
        raise ValueError("At least two equity observations are required.")

    if not np.isfinite(equity.to_numpy()).all():
        raise ValueError("Equity must contain finite values.")

    if (equity <= 0).any():
        raise ValueError("Equity must be positive.")

    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("Equity must have a DatetimeIndex.")

    if equity.index.has_duplicates or not equity.index.is_monotonic_increasing:
        raise ValueError("Equity dates must be unique and chronological.")

    elapsed_days = (equity.index[-1] - equity.index[0]).days

    if elapsed_days <= 0:
        raise ValueError("Equity observations must span multiple days.")

    years = elapsed_days / 365.25

    return float(
        (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    )


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility from daily returns."""

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must contain finite values.")

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio."""

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must contain finite values.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    periodic_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    )

    excess_returns = returns - periodic_risk_free_rate

    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility.")

    return float(
        excess_returns.mean() / volatility * np.sqrt(periods_per_year)
    )


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown."""

    if equity.empty:
        raise ValueError("Equity cannot be empty.")

    if not np.isfinite(equity.to_numpy()).all():
        raise ValueError("Equity must contain finite values.")

    if (equity <= 0).any():
        raise ValueError("Equity must be positive.")

    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("Equity must have a DatetimeIndex.")

    if equity.index.has_duplicates or not equity.index.is_monotonic_increasing:
        raise ValueError("Equity dates must be unique and chronological.")

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1

    return float(drawdown.min())


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio."""

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must contain finite values.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    periodic_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    )

    excess_returns = returns - periodic_risk_free_rate

    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.sqrt(
        np.mean(downside_returns ** 2)
    )

    if downside_deviation == 0:
        raise ValueError("Sortino ratio is undefined with zero downside deviation.")

    return float(
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(periods_per_year)
    )


def calculate_win_rate(trade_pnls: pd.Series) -> float:
    """Calculate proportion of profitable closed trades."""

    if trade_pnls.empty:
        raise ValueError("Trade P&Ls cannot be empty.")

    if not np.isfinite(trade_pnls.to_numpy()).all():
        raise ValueError("Trade P&Ls must contain finite values.")

    return float((trade_pnls > 0).mean())


def calculate_profit_factor(trade_pnls: pd.Series) -> float:
    """Calculate gross profits divided by gross losses."""

    if trade_pnls.empty:
        raise ValueError("Trade P&Ls cannot be empty.")

    if not np.isfinite(trade_pnls.to_numpy()).all():
        raise ValueError("Trade P&Ls must contain finite values.")

    gross_profit = trade_pnls[trade_pnls > 0].sum()
    gross_loss = abs(trade_pnls[trade_pnls < 0].sum())

    if gross_loss == 0:
        raise ValueError("Profit factor is undefined with zero gross losses.")

    return float(gross_profit / gross_loss)



def calculate_performance_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Calculate portfolio performance metrics.

    Metrics that cannot be calculated reliably are returned as None.
    """

    if not isinstance(equity_curve, pd.DataFrame):
        raise ValueError("Equity curve must be a DataFrame.")

    if "equity" not in equity_curve.columns:
        raise ValueError("Equity curve must contain an equity column.")

    equity = equity_curve["equity"]

    # CAGR and drawdown can be calculated independently.
    metrics = {
        "cagr": calculate_cagr(equity),
        "max_drawdown": calculate_max_drawdown(equity),
        "annualized_volatility": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
    }

    returns = equity.pct_change().dropna()

    # At least two observations are needed for sample volatility.
    if len(returns) < 2:
        return metrics

    volatility = calculate_volatility(
        returns, periods_per_year
    )

    metrics["annualized_volatility"] = volatility

    # Sharpe is undefined when return volatility is zero.
    if volatility > 0:
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        )

    periodic_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    )

    excess_returns = returns - periodic_risk_free_rate
    downside_deviation = np.sqrt(
        np.mean(np.minimum(excess_returns, 0) ** 2)
    )

    # Sortino is undefined when downside deviation is zero.
    if downside_deviation > 0:
        metrics["sortino_ratio"] = calculate_sortino_ratio(
            returns, risk_free_rate, periods_per_year
        )

    return metrics
