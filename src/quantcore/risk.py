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
