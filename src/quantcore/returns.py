"""
Return transformations for QuantCore.

This module provides deterministic utilities for converting price series
into returns and for compounding periodic returns into cumulative growth
or wealth-index series.

The functions preserve pandas indexes and Series names when possible and
apply explicit validation to avoid silent data-cleaning behavior.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantcore._validation import (
    validate_positive_finite_number,
    validate_positive_integer,
)


PriceInput = pd.Series | np.ndarray | Sequence[float]
ReturnInput = pd.Series | np.ndarray | Sequence[float]


__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "wealth_index",
]


def _coerce_series(
    values: pd.Series | np.ndarray | Sequence[float],
    *,
    name: str,
) -> pd.Series:
    """
    Convert one-dimensional numeric input to a float64 pandas Series.

    Existing pandas indexes and Series names are preserved.
    """
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        array = np.asarray(values)

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        series = pd.Series(array)

    if series.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    try:
        return series.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from exc


def _prepare_prices(
    prices: PriceInput,
) -> pd.Series:
    """
    Validate and prepare a financial price series.

    Missing prices are permitted because callers may choose whether to
    preserve or remove missing return observations. Infinite and
    non-positive prices are rejected.
    """
    series = _coerce_series(
        prices,
        name="Prices",
    )

    values = series.to_numpy(
        dtype="float64",
        copy=False,
    )

    if np.isinf(values).any():
        raise ValueError(
            "Prices cannot contain infinite values."
        )

    finite_prices = series.dropna()

    if finite_prices.empty:
        raise ValueError(
            "No valid price observations are available."
        )

    if (finite_prices <= 0).any():
        raise ValueError(
            "Prices must be strictly positive."
        )

    return series


def _prepare_returns(
    returns: ReturnInput,
) -> pd.Series:
    """
    Validate and prepare a periodic return series.

    Missing observations are permitted. Infinite returns and returns below
    -100% are rejected because they cannot represent valid simple returns.
    A return of exactly -100% is allowed.
    """
    series = _coerce_series(
        returns,
        name="Returns",
    )

    values = series.to_numpy(
        dtype="float64",
        copy=False,
    )

    if np.isinf(values).any():
        raise ValueError(
            "Returns cannot contain infinite values."
        )

    finite_returns = series.dropna()

    if finite_returns.empty:
        raise ValueError(
            "No valid return observations are available."
        )

    if (finite_returns < -1.0).any():
        raise ValueError(
            "Returns cannot be below -100%."
        )

    return series


def _finalize_series(
    values: pd.Series,
    *,
    dropna: bool,
    operation_name: str,
) -> pd.Series:
    """
    Apply missing-value handling and verify a usable result exists.
    """
    result = (
        values.dropna()
        if dropna
        else values
    )

    if values.notna().sum() == 0:
        raise ValueError(
            f"No valid observations were produced by {operation_name}."
        )

    return result.astype("float64")


def simple_returns(
    prices: PriceInput,
    periods: int = 1,
    dropna: bool = True,
) -> pd.Series:
    """
    Calculate simple percentage returns from a price series.

    Parameters
    ----------
    prices:
        One-dimensional sequence of strictly positive prices.
    periods:
        Number of observations between the current and comparison price.
    dropna:
        If True, remove missing return observations from the result.

    Returns
    -------
    pandas.Series
        Simple returns expressed as decimals.

    Notes
    -----
    Missing prices are not forward-filled. A missing price therefore
    prevents a return from being calculated across that observation.
    """
    periods = validate_positive_integer(
        periods,
        "periods",
    )

    price_series = _prepare_prices(
        prices
    )

    shifted = price_series.shift(
        periods
    )

    result = (
        price_series / shifted
    ) - 1.0

    return _finalize_series(
        result,
        dropna=dropna,
        operation_name="simple return calculation",
    )


def log_returns(
    prices: PriceInput,
    periods: int = 1,
    dropna: bool = True,
) -> pd.Series:
    """
    Calculate logarithmic returns from a price series.

    Parameters
    ----------
    prices:
        One-dimensional sequence of strictly positive prices.
    periods:
        Number of observations between the current and comparison price.
    dropna:
        If True, remove missing return observations from the result.

    Returns
    -------
    pandas.Series
        Log returns expressed as decimals.

    Notes
    -----
    Log returns are calculated explicitly as:

        log(current_price / previous_price)

    Missing prices are not forward-filled.
    """
    periods = validate_positive_integer(
        periods,
        "periods",
    )

    price_series = _prepare_prices(
        prices
    )

    shifted = price_series.shift(
        periods
    )

    result = np.log(
        price_series / shifted
    )

    return _finalize_series(
        result,
        dropna=dropna,
        operation_name="log return calculation",
    )


def cumulative_returns(
    returns: ReturnInput,
    dropna: bool = True,
) -> pd.Series:
    """
    Compound periodic simple returns into cumulative returns.

    Parameters
    ----------
    returns:
        One-dimensional sequence of periodic simple returns.
    dropna:
        If True, remove missing observations before compounding.

    Returns
    -------
    pandas.Series
        Cumulative return after each observation.

    Examples
    --------
    Returns of 10% followed by 10% produce cumulative growth of:

        10%
        21%
    """
    return_series = _prepare_returns(
        returns
    )

    if dropna:
        return_series = return_series.dropna()

    result = (
        1.0 + return_series
    ).cumprod() - 1.0

    return _finalize_series(
        result,
        dropna=dropna,
        operation_name="cumulative return calculation",
    )


def wealth_index(
    returns: ReturnInput,
    initial_value: float = 1.0,
    dropna: bool = True,
) -> pd.Series:
    """
    Compound periodic simple returns into a wealth-index series.

    Parameters
    ----------
    returns:
        One-dimensional sequence of periodic simple returns.
    initial_value:
        Starting portfolio or index value.
    dropna:
        If True, remove missing observations before compounding.

    Returns
    -------
    pandas.Series
        Wealth value after each periodic return.
    """
    initial_value = validate_positive_finite_number(
        initial_value,
        "initial_value",
    )

    return_series = _prepare_returns(
        returns
    )

    if dropna:
        return_series = return_series.dropna()

    result = (
        initial_value
        * (1.0 + return_series).cumprod()
    )

    return _finalize_series(
        result,
        dropna=dropna,
        operation_name="wealth index calculation",
    )
