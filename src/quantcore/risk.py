"""Risk metrics for QuantCore."""

import pandas as pd


def max_drawdown(prices: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a price series.

    Maximum drawdown measures the largest percentage decline
    from a historical peak to a subsequent trough.

    Parameters
    ----------
    prices : pd.Series
        Series of asset or portfolio prices.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal value.
    """
    running_max = prices.cummax()
    drawdowns = (prices / running_max) - 1
    return drawdowns.min()
    
def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from periodic returns.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.

    periods_per_year : int, default=252
        Number of periods in one year.
        Use 252 for daily trading data.

    Returns
    -------
    float
        Annualized volatility as a decimal.
    """
    return returns.std() * (periods_per_year ** 0.5)
