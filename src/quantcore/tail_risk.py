"""
Tail-risk models for QuantCore.

This module provides deterministic downside-risk models for financial
return series.

Supported models
----------------
Historical Simulation
    - Value at Risk (VaR)
    - Expected Shortfall (ES)

Gaussian Parametric
    - Value at Risk (VaR)
    - Expected Shortfall (ES)

Student-t Parametric
    - Value at Risk (VaR)
    - Expected Shortfall (ES)

Conventions
-----------
Returns remain signed internally.

For example:
    -0.05 represents a 5% loss.

Risk thresholds are also returned as signed values.

For example:
    -0.054 represents a 5.4% loss threshold.

Reporting or presentation layers may convert these values into positive
loss magnitudes if desired.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, t


RiskInput = pd.Series | np.ndarray | Sequence[float]


__all__ = [
    "historical_var",
    "historical_expected_shortfall",
    "gaussian_var",
    "gaussian_expected_shortfall",
    "student_t_var",
    "student_t_expected_shortfall",
]


def _clean_returns(returns: RiskInput) -> pd.Series:
    """
    Normalize and validate a return series.

    Parameters
    ----------
    returns : RiskInput
        One-dimensional collection of periodic returns.

    Returns
    -------
    pd.Series
        Clean floating-point return observations with missing and
        non-finite values removed.

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


def _validate_confidence(confidence: float) -> None:
    """
    Validate a confidence level.

    Parameters
    ----------
    confidence : float
        Confidence level expressed as a decimal between 0 and 1.

    Raises
    ------
    ValueError
        If confidence is not strictly between 0 and 1.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "Confidence must be strictly between 0 and 1."
        )


def _require_min_observations(
    returns: pd.Series,
    minimum: int,
    model_name: str,
) -> None:
    """
    Require a minimum number of observations for a model.

    Parameters
    ----------
    returns : pd.Series
        Clean return observations.

    minimum : int
        Minimum number of required observations.

    model_name : str
        Human-readable model name used in the error message.

    Raises
    ------
    ValueError
        If the return series contains too few observations.
    """
    if len(returns) < minimum:
        raise ValueError(
            f"{model_name} requires at least "
            f"{minimum} valid observations."
        )


def historical_var(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Historical Simulation Value at Risk.

    Historical VaR estimates the loss threshold exceeded by the worst
    ``1 - confidence`` proportion of observed returns.

    Parameters
    ----------
    returns : RiskInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level expressed as a decimal.

    Returns
    -------
    float
        Historical VaR threshold as a signed return.

    Notes
    -----
    A result of ``-0.054`` represents a 5.4% loss threshold.

    Historical Simulation assumes that the observed return distribution
    is informative about future downside risk.
    """
    cleaned = _clean_returns(returns)
    _validate_confidence(confidence)

    percentile = (1.0 - confidence) * 100.0

    return float(
        np.percentile(
            cleaned.to_numpy(),
            percentile,
        )
    )


def historical_expected_shortfall(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Historical Expected Shortfall.

    Expected Shortfall measures the average return among observations
    that fall at or below the Historical VaR threshold.

    Parameters
    ----------
    returns : RiskInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level expressed as a decimal.

    Returns
    -------
    float
        Average signed return in the historical loss tail.

    Notes
    -----
    A result of ``-0.081`` represents an average tail loss of 8.1%.
    """
    cleaned = _clean_returns(returns)
    _validate_confidence(confidence)

    var_threshold = historical_var(
        cleaned,
        confidence=confidence,
    )

    tail_returns = cleaned[
        cleaned <= var_threshold
    ]

    return float(tail_returns.mean())


def gaussian_var(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Gaussian Parametric Value at Risk.

    The model assumes returns follow a normal distribution and estimates
    the downside threshold using the sample mean, sample volatility, and
    the appropriate normal-distribution quantile.

    Parameters
    ----------
    returns : RiskInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level expressed as a decimal.

    Returns
    -------
    float
        Gaussian VaR threshold as a signed return.

    Notes
    -----
    The model uses sample standard deviation with ``ddof=1``.
    """
    cleaned = _clean_returns(returns)
    _validate_confidence(confidence)

    _require_min_observations(
        cleaned,
        minimum=2,
        model_name="Gaussian VaR",
    )

    mean_return = float(cleaned.mean())

    volatility = float(
        cleaned.std(ddof=1)
    )

    z_score = float(
        norm.ppf(1.0 - confidence)
    )

    return float(
        mean_return
        + volatility * z_score
    )


def gaussian_expected_shortfall(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Gaussian Expected Shortfall.

    Gaussian Expected Shortfall estimates the expected return conditional
    on being beyond the Gaussian VaR threshold.

    Parameters
    ----------
    returns : RiskInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level expressed as a decimal.

    Returns
    -------
    float
        Gaussian Expected Shortfall as a signed return.
    """
    cleaned = _clean_returns(returns)
    _validate_confidence(confidence)

    _require_min_observations(
        cleaned,
        minimum=2,
        model_name="Gaussian Expected Shortfall",
    )

    mean_return = float(cleaned.mean())

    volatility = float(
        cleaned.std(ddof=1)
    )

    alpha = 1.0 - confidence

    z_score = float(
        norm.ppf(alpha)
    )

    tail_density = float(
        norm.pdf(z_score)
    )

    return float(
        mean_return
        - volatility
        * tail_density
        / alpha
    )


def student_t_var(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Student-t Parametric Value at Risk.

    A Student-t distribution is fitted to the observed returns. The
    resulting lower-tail quantile is used as the VaR threshold.

    Parameters
    ----------
    returns : RiskInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level expressed as a decimal.

    Returns
    -------
    float
        Student-t VaR threshold as a signed return.

    Notes
    -----
    Student-t models can represent heavier tails than a Gaussian model,
    making them useful when return distributions exhibit elevated tail
    risk.
    """
    cleaned = _clean_returns(returns)
    _validate_confidence(confidence)

    _require_min_observations(
        cleaned,
        minimum=3,
        model_name="Student-t VaR",
    )

    degrees_of_freedom, location, scale = t.fit(
        cleaned.to_numpy()
    )

    lower_tail_quantile = float(
        t.ppf(
            1.0 - confidence,
            degrees_of_freedom,
        )
    )

    return float(
        location
        + scale * lower_tail_quantile
    )


def student_t_expected_shortfall(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Student-t Expected Shortfall.

    The model fits a Student-t distribution to the observed returns and
    computes the analytical expectation of the lower tail beyond the
    selected confidence threshold.

    Parameters
    ----------
    returns : RiskInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level expressed as a decimal.

    Returns
    -------
    float
        Student-t Expected Shortfall as a signed return.

    Raises
    ------
    ValueError
        If the fitted Student-t distribution has one or fewer degrees of
        freedom, because its expected value is not finite.

    Notes
    -----
    Student-t Expected Shortfall explicitly models heavier-tail behavior
    than Gaussian Expected Shortfall.
    """
    cleaned = _clean_returns(returns)
    _validate_confidence(confidence)

    _require_min_observations(
        cleaned,
        minimum=3,
        model_name="Student-t Expected Shortfall",
    )

    degrees_of_freedom, location, scale = t.fit(
        cleaned.to_numpy()
    )

    if degrees_of_freedom <= 1.0:
        raise ValueError(
            "Student-t Expected Shortfall is undefined when the "
            "fitted degrees of freedom are less than or equal to 1."
        )

    alpha = 1.0 - confidence

    lower_tail_quantile = float(
        t.ppf(
            alpha,
            degrees_of_freedom,
        )
    )

    tail_density = float(
        t.pdf(
            lower_tail_quantile,
            degrees_of_freedom,
        )
    )

    expected_tail = (
        location
        - scale
        * (
            (
                degrees_of_freedom
                + lower_tail_quantile**2
            )
            / (
                degrees_of_freedom
                - 1.0
            )
        )
        * (
            tail_density
            / alpha
        )
    )

    return float(expected_tail)
