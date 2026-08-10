"""
Distribution diagnostics for QuantCore.

This module provides deterministic statistical diagnostics for financial
return series.

Supported diagnostics
---------------------
- Observation count
- Mean return
- Periodic volatility
- Annualized volatility
- Skewness
- Excess kurtosis
- Jarque-Bera normality testing
- Structured distribution summaries

The functions in this module report statistical properties only.
Interpretation belongs in higher-level research or application layers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew

from quantcore.risk import volatility


ReturnInput = pd.Series | np.ndarray | Sequence[float]


__all__ = [
    "DistributionDiagnostics",
    "mean_return",
    "skewness",
    "excess_kurtosis",
    "normality_test",
    "distribution_summary",
]


@dataclass(frozen=True)
class DistributionDiagnostics:
    """
    Statistical summary of a return distribution.

    Attributes
    ----------
    observations : int
        Number of valid return observations.

    mean_return : float
        Arithmetic mean of periodic returns.

    volatility : float
        Sample volatility of periodic returns.

    annualized_volatility : float
        Volatility annualized using the supplied periods-per-year value.

    skewness : float
        Bias-corrected sample skewness.

    excess_kurtosis : float
        Bias-corrected excess kurtosis.
        A normal distribution has excess kurtosis of approximately zero.

    normality_statistic : float
        Jarque-Bera test statistic.

    normality_pvalue : float
        Jarque-Bera p-value.

    normality_rejected : bool
        Whether normality is rejected at the selected significance level.
    """

    observations: int
    mean_return: float
    volatility: float
    annualized_volatility: float
    skewness: float
    excess_kurtosis: float
    normality_statistic: float
    normality_pvalue: float
    normality_rejected: bool


def _clean_returns(
    returns: ReturnInput,
) -> pd.Series:
    """
    Normalize and validate return observations.

    Parameters
    ----------
    returns : ReturnInput
        One-dimensional collection of periodic returns.

    Returns
    -------
    pd.Series
        Floating-point return observations with missing and non-finite
        values removed.

    Raises
    ------
    ValueError
        If no valid observations remain after cleaning.
    """
    cleaned = (
        pd.Series(returns, dtype="float64")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if cleaned.empty:
        raise ValueError(
            "Return series contains no valid observations."
        )

    return cleaned


def _require_min_observations(
    returns: pd.Series,
    minimum: int,
    diagnostic_name: str,
) -> None:
    """
    Require a minimum sample size for a statistical diagnostic.
    """
    if len(returns) < minimum:
        raise ValueError(
            f"{diagnostic_name} requires at least "
            f"{minimum} valid observations."
        )


def mean_return(
    returns: ReturnInput,
) -> float:
    """
    Calculate arithmetic mean return.

    Parameters
    ----------
    returns : ReturnInput
        Periodic return observations.

    Returns
    -------
    float
        Arithmetic mean return.
    """
    cleaned = _clean_returns(returns)

    return float(cleaned.mean())


def skewness(
    returns: ReturnInput,
) -> float:
    """
    Calculate bias-corrected sample skewness.

    Skewness measures asymmetry in the return distribution.

    Negative values indicate a longer or heavier left tail, while
    positive values indicate a longer or heavier right tail.

    Parameters
    ----------
    returns : ReturnInput
        Periodic return observations.

    Returns
    -------
    float
        Bias-corrected sample skewness.
    """
    cleaned = _clean_returns(returns)

    _require_min_observations(
        cleaned,
        minimum=3,
        diagnostic_name="Skewness",
    )

    return float(
        skew(
            cleaned.to_numpy(),
            bias=False,
        )
    )


def excess_kurtosis(
    returns: ReturnInput,
) -> float:
    """
    Calculate bias-corrected excess kurtosis.

    Excess kurtosis measures tail weight relative to a normal
    distribution.

    A normal distribution has excess kurtosis of approximately zero.

    Parameters
    ----------
    returns : ReturnInput
        Periodic return observations.

    Returns
    -------
    float
        Bias-corrected excess kurtosis.
    """
    cleaned = _clean_returns(returns)

    _require_min_observations(
        cleaned,
        minimum=4,
        diagnostic_name="Excess kurtosis",
    )

    return float(
        kurtosis(
            cleaned.to_numpy(),
            fisher=True,
            bias=False,
        )
    )


def normality_test(
    returns: ReturnInput,
) -> tuple[float, float]:
    """
    Perform the Jarque-Bera normality test.

    The Jarque-Bera test evaluates whether sample skewness and kurtosis
    are consistent with a normal distribution.

    Parameters
    ----------
    returns : ReturnInput
        Periodic return observations.

    Returns
    -------
    tuple[float, float]
        Jarque-Bera test statistic and p-value.
    """
    cleaned = _clean_returns(returns)

    _require_min_observations(
        cleaned,
        minimum=2,
        diagnostic_name="Jarque-Bera normality test",
    )

    result = jarque_bera(
        cleaned.to_numpy()
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def distribution_summary(
    returns: ReturnInput,
    periods_per_year: int = 252,
    significance_level: float = 0.05,
) -> DistributionDiagnostics:
    """
    Create a complete statistical diagnostic summary.

    Parameters
    ----------
    returns : ReturnInput
        Periodic return observations.

    periods_per_year : int, default=252
        Number of return periods in one year.
        Use 252 for daily trading data.

    significance_level : float, default=0.05
        Significance threshold used when evaluating the Jarque-Bera
        normality test.

    Returns
    -------
    DistributionDiagnostics
        Structured statistical summary.

    Raises
    ------
    ValueError
        If periods_per_year is not positive, significance_level is
        outside the interval (0, 1), or too few observations are
        available for the complete diagnostic summary.
    """
    cleaned = _clean_returns(returns)

    _require_min_observations(
        cleaned,
        minimum=4,
        diagnostic_name="Distribution summary",
    )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero."
        )

    if not 0.0 < significance_level < 1.0:
        raise ValueError(
            "significance_level must be strictly between 0 and 1."
        )

    statistic, pvalue = normality_test(
        cleaned
    )

    periodic_volatility = volatility(
        cleaned,
        periods_per_year=1,
    )

    annualized_volatility = volatility(
        cleaned,
        periods_per_year=periods_per_year,
    )

    return DistributionDiagnostics(
        observations=len(cleaned),
        mean_return=mean_return(cleaned),
        volatility=float(periodic_volatility),
        annualized_volatility=float(
            annualized_volatility
        ),
        skewness=skewness(cleaned),
        excess_kurtosis=excess_kurtosis(
            cleaned
        ),
        normality_statistic=statistic,
        normality_pvalue=pvalue,
        normality_rejected=bool(
            pvalue < significance_level
        ),
    )
