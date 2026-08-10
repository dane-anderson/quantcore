"""
Internal validation utilities for QuantCore.

This module centralizes input cleaning and parameter validation used
across QuantCore's statistical, risk, simulation, and portfolio modules.

The leading underscore indicates that this module is internal
infrastructure rather than part of QuantCore's public API.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


ReturnInput = pd.Series | np.ndarray | Sequence[float]


def clean_returns(
    returns: ReturnInput,
) -> pd.Series:
    """
    Normalize return observations into a clean pandas Series.

    Parameters
    ----------
    returns : ReturnInput
        One-dimensional collection of periodic returns.

    Returns
    -------
    pd.Series
        Finite floating-point return observations.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or contains no valid
        observations.
    """
    values = np.asarray(
        returns,
        dtype="float64",
    )

    if values.ndim != 1:
        raise ValueError(
            "Return data must be one-dimensional."
        )

    cleaned = pd.Series(
        values,
        dtype="float64",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if cleaned.empty:
        raise ValueError(
            "Return series contains no valid observations."
        )

    return cleaned.reset_index(drop=True)


def clean_return_array(
    returns: ReturnInput,
) -> np.ndarray:
    """
    Normalize return observations into a clean NumPy array.

    This form is useful for vectorized numerical operations such as
    Monte Carlo simulation.

    Parameters
    ----------
    returns : ReturnInput
        One-dimensional collection of periodic returns.

    Returns
    -------
    np.ndarray
        Finite floating-point return observations.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or contains no valid
        observations.
    """
    values = np.asarray(
        returns,
        dtype="float64",
    )

    if values.ndim != 1:
        raise ValueError(
            "Return data must be one-dimensional."
        )

    cleaned = values[
        np.isfinite(values)
    ]

    if cleaned.size == 0:
        raise ValueError(
            "Return series contains no valid observations."
        )

    return cleaned


def require_min_observations(
    values,
    minimum: int,
    operation_name: str,
) -> None:
    """
    Require a minimum number of observations.

    Parameters
    ----------
    values
        Collection of validated observations.

    minimum : int
        Minimum number of observations required.

    operation_name : str
        Human-readable operation name included in validation errors.

    Raises
    ------
    ValueError
        If fewer than the required observations are available.
    """
    if len(values) < minimum:
        raise ValueError(
            f"{operation_name} requires at least "
            f"{minimum} valid observations."
        )


def validate_probability(
    value: float,
    name: str,
) -> float:
    """
    Validate a probability strictly between zero and one.

    Parameters
    ----------
    value : float
        Probability expressed as a decimal.

    name : str
        Parameter name included in validation errors.

    Returns
    -------
    float
        Validated probability.

    Raises
    ------
    ValueError
        If the value is not finite or is outside the interval (0, 1).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be strictly between 0 and 1."
        ) from None

    if (
        not np.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(
            f"{name} must be strictly between 0 and 1."
        )

    return value


def validate_confidence(
    confidence: float,
) -> float:
    """
    Validate a statistical confidence level.

    Returns
    -------
    float
        Validated confidence level.
    """
    return validate_probability(
        confidence,
        "Confidence",
    )


def validate_significance_level(
    significance_level: float,
) -> float:
    """
    Validate a statistical significance level.

    Returns
    -------
    float
        Validated significance level.
    """
    return validate_probability(
        significance_level,
        "significance_level",
    )


def validate_positive_integer(
    value,
    name: str,
) -> int:
    """
    Validate a positive whole-number parameter.

    Parameters
    ----------
    value
        Value to validate.

    name : str
        Parameter name included in validation errors.

    Returns
    -------
    int
        Validated positive integer.

    Raises
    ------
    ValueError
        If the supplied value is not a positive whole number.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a positive integer."
        ) from None

    if (
        not np.isfinite(numeric_value)
        or numeric_value <= 0
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(numeric_value)


def validate_integer(
    value,
    name: str,
) -> int:
    """
    Validate an integer parameter.

    Parameters
    ----------
    value
        Value to validate.

    name : str
        Parameter name included in validation errors.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    ValueError
        If the supplied value is not a finite whole number.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be an integer."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be an integer."
        ) from None

    if (
        not np.isfinite(numeric_value)
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            f"{name} must be an integer."
        )

    return int(numeric_value)


def validate_positive_finite_number(
    value,
    name: str,
) -> float:
    """
    Validate a positive finite numeric parameter.

    Parameters
    ----------
    value
        Value to validate.

    name : str
        Parameter name included in validation errors.

    Returns
    -------
    float
        Validated positive finite number.

    Raises
    ------
    ValueError
        If the supplied value is non-numeric, non-finite, or not
        strictly positive.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a positive finite number."
        ) from None

    if (
        not np.isfinite(numeric_value)
        or numeric_value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    return numeric_value
