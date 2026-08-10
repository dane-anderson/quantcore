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
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from quantcore._validation import (
    clean_returns,
    require_min_observations,
    validate_confidence,
)


RiskInput = pd.Series | np.ndarray | Sequence[float]


__all__ = [
    "historical_var",
    "historical_expected_shortfall",
    "gaussian_var",
    "gaussian_expected_shortfall",
    "student_t_var",
    "student_t_expected_shortfall",
]


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

    Historical Simulation assumes the observed return distribution is
    informative about future downside risk.
    """
    cleaned = clean_returns(returns)

    confidence = validate_confidence(
        confidence
    )

    percentile = (
        1.0 - confidence
    ) * 100.0

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
    at or below the Historical VaR threshold.

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
    cleaned = clean_returns(returns)

    confidence = validate_confidence(
        confidence
    )

    var_threshold = historical_var(
        cleaned,
        confidence=confidence,
    )

    tail_returns = cleaned[
        cleaned <= var_threshold
    ]

    return float(
        tail_returns.mean()
    )


def gaussian_var(
    returns: RiskInput,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Gaussian Parametric Value at Risk.

    The model assumes returns follow a normal distribution and estimates
    the downside threshold using the sample mean, sample volatility, and
    the corresponding normal-distribution quantile.

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
    Sample standard deviation is calculated using ``ddof=1``.
    """
    cleaned = clean_returns(returns)

    confidence = validate_confidence(
        confidence
    )

    require_min_observations(
        cleaned,
        minimum=2,
        operation_name="Gaussian VaR",
    )

    mean_return = float(
        cleaned.mean()
    )

    volatility = float(
        cleaned.std(ddof=1)
    )

    z_score = float(
        norm.ppf(
            1.0 - confidence
        )
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
    on falling beyond the Gaussian VaR threshold.

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
    cleaned = clean_returns(returns)

    confidence = validate_confidence(
        confidence
    )

    require_min_observations(
        cleaned,
        minimum=2,
        operation_name="Gaussian Expected Shortfall",
    )

    mean_return = float(
        cleaned.mean()
    )

    volatility = float(
        cleaned.std(ddof=1)
    )

    alpha = (
        1.0 - confidence
    )

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

    A Student-t distribution is fitted to the observed return series.
    The resulting lower-tail quantile is used as the VaR threshold.

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
    Student-t distributions can represent heavier tails than Gaussian
    models and may better capture elevated tail risk.
    """
    cleaned = clean_returns(returns)

    confidence = validate_confidence(
        confidence
    )

    require_min_observations(
        cleaned,
        minimum=3,
        operation_name="Student-t VaR",
    )

    (
        degrees_of_freedom,
        location,
        scale,
    ) = t.fit(
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
        If the fitted Student-t distribution has one or fewer degrees
        of freedom, because its expected value is not finite.
    """
    cleaned = clean_returns(returns)

    confidence = validate_confidence(
        confidence
    )

    require_min_observations(
        cleaned,
        minimum=3,
        operation_name="Student-t Expected Shortfall",
    )

    (
        degrees_of_freedom,
        location,
        scale,
    ) = t.fit(
        cleaned.to_numpy()
    )

    if degrees_of_freedom <= 1.0:
        raise ValueError(
            "Student-t Expected Shortfall is undefined when "
            "the fitted degrees of freedom are less than or "
            "equal to 1."
        )

    alpha = (
        1.0 - confidence
    )

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

    return float(
        expected_tail
    )
