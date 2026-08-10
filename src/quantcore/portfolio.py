"""
Portfolio analytics for QuantCore.

This module provides deterministic multi-asset portfolio analytics
using historical return data.

Supported analytics
-------------------
- Asset correlation matrices
- Asset covariance matrices
- Portfolio return series
- Annualized expected portfolio return
- Annualized portfolio volatility
- Portfolio Sharpe ratio

Design conventions
------------------
- Input data is expected to contain returns, not prices.
- DataFrame columns represent assets.
- Rows represent synchronized return observations.
- Rows containing missing values are removed before portfolio analysis.
- Portfolio weights must align exactly with the supplied assets.
- Short positions are permitted.
- Portfolio weights must sum to 1.0.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantcore._validation import (
    require_min_observations,
    validate_positive_integer,
)


WeightInput = (
    pd.Series
    | np.ndarray
    | Sequence[float]
)


__all__ = [
    "correlation_matrix",
    "covariance_matrix",
    "portfolio_returns",
    "portfolio_expected_return",
    "portfolio_volatility",
    "sharpe_ratio",
]


_ALLOWED_CORRELATION_METHODS = {
    "pearson",
    "spearman",
    "kendall",
}


def _prepare_return_frame(
    returns: pd.DataFrame,
    minimum_observations: int = 1,
    operation_name: str = "Portfolio analysis",
) -> pd.DataFrame:
    """
    Validate and clean a multi-asset return DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Return observations with assets represented by columns.

    minimum_observations : int, default=1
        Minimum number of complete observations required.

    operation_name : str
        Human-readable operation name used in validation errors.

    Returns
    -------
    pd.DataFrame
        Clean floating-point return data containing only complete rows.

    Raises
    ------
    ValueError
        If the data is empty, non-numeric, non-finite, has duplicate
        asset names, or does not contain enough complete observations.
    """
    if not isinstance(
        returns,
        pd.DataFrame,
    ):
        raise ValueError(
            "Portfolio return data must be a pandas DataFrame."
        )

    if (
        returns.empty
        or len(returns.columns) == 0
    ):
        raise ValueError(
            "Portfolio return data cannot be empty."
        )

    if not returns.columns.is_unique:
        raise ValueError(
            "Portfolio asset names must be unique."
        )

    try:
        frame = returns.astype(
            "float64"
        ).copy()
    except (TypeError, ValueError):
        raise ValueError(
            "Portfolio return data must contain numeric values."
        ) from None

    if np.isinf(
        frame.to_numpy()
    ).any():
        raise ValueError(
            "Portfolio return data cannot contain infinite values."
        )

    frame = frame.dropna(
        axis=0,
        how="any",
    )

    if frame.empty:
        raise ValueError(
            "Portfolio return data contains no complete observations."
        )

    require_min_observations(
        frame,
        minimum=minimum_observations,
        operation_name=operation_name,
    )

    return frame


def _align_weights(
    weights: WeightInput,
    asset_names: pd.Index,
) -> pd.Series:
    """
    Validate and align portfolio weights to asset columns.

    Pandas Series weights are aligned by asset label. Sequence and NumPy
    inputs are aligned positionally.

    Parameters
    ----------
    weights : WeightInput
        Portfolio weights.

    asset_names : pd.Index
        Ordered asset names from the return DataFrame.

    Returns
    -------
    pd.Series
        Floating-point weights aligned to asset order.

    Raises
    ------
    ValueError
        If weights are invalid, non-finite, incorrectly sized, do not
        match the assets, or do not sum to 1.0.
    """
    if isinstance(
        weights,
        pd.Series,
    ):
        if not weights.index.is_unique:
            raise ValueError(
                "Portfolio weight labels must be unique."
            )

        try:
            weight_series = weights.astype(
                "float64"
            ).copy()
        except (TypeError, ValueError):
            raise ValueError(
                "Portfolio weights must contain numeric values."
            ) from None

        missing_assets = [
            asset
            for asset in asset_names
            if asset not in weight_series.index
        ]

        extra_assets = [
            asset
            for asset in weight_series.index
            if asset not in asset_names
        ]

        if missing_assets:
            raise ValueError(
                "Portfolio weights are missing assets: "
                + ", ".join(
                    map(
                        str,
                        missing_assets,
                    )
                )
                + "."
            )

        if extra_assets:
            raise ValueError(
                "Portfolio weights contain unknown assets: "
                + ", ".join(
                    map(
                        str,
                        extra_assets,
                    )
                )
                + "."
            )

        weight_series = (
            weight_series.reindex(
                asset_names
            )
        )

    else:
        try:
            weight_array = np.asarray(
                weights,
                dtype="float64",
            )
        except (TypeError, ValueError):
            raise ValueError(
                "Portfolio weights must contain numeric values."
            ) from None

        if weight_array.ndim != 1:
            raise ValueError(
                "Portfolio weights must be one-dimensional."
            )

        if len(weight_array) != len(
            asset_names
        ):
            raise ValueError(
                "Number of portfolio weights must match "
                "the number of assets."
            )

        weight_series = pd.Series(
            weight_array,
            index=asset_names,
            dtype="float64",
        )

    if not np.isfinite(
        weight_series.to_numpy()
    ).all():
        raise ValueError(
            "Portfolio weights must contain only finite values."
        )

    weight_sum = float(
        weight_series.sum()
    )

    if not np.isclose(
        weight_sum,
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Portfolio weights must sum to 1.0."
        )

    return weight_series


def _validate_finite_number(
    value,
    name: str,
) -> float:
    """
    Validate a finite numeric value.

    Negative values are permitted.
    """
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be a finite number."
        )

    try:
        numeric_value = float(
            value
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a finite number."
        ) from None

    if not np.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{name} must be a finite number."
        )

    return numeric_value


def correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Calculate the correlation matrix between assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic asset returns.

    method : str, default="pearson"
        Correlation method.

        Supported values are:

        - ``"pearson"``
        - ``"spearman"``
        - ``"kendall"``

    Returns
    -------
    pd.DataFrame
        Asset correlation matrix.
    """
    frame = _prepare_return_frame(
        returns,
        minimum_observations=2,
        operation_name="Correlation matrix",
    )

    if not isinstance(
        method,
        str,
    ):
        raise ValueError(
            "Correlation method must be a string."
        )

    method = method.lower()

    if (
        method
        not in _ALLOWED_CORRELATION_METHODS
    ):
        available = ", ".join(
            sorted(
                _ALLOWED_CORRELATION_METHODS
            )
        )

        raise ValueError(
            f"Unknown correlation method: {method}. "
            f"Available methods: {available}."
        )

    return frame.corr(
        method=method
    )


def covariance_matrix(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Calculate an annualized covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic asset returns.

    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    pd.DataFrame
        Annualized sample covariance matrix.

    Notes
    -----
    Covariance is calculated using sample covariance and then multiplied
    by ``periods_per_year``.
    """
    frame = _prepare_return_frame(
        returns,
        minimum_observations=2,
        operation_name="Covariance matrix",
    )

    periods_per_year = (
        validate_positive_integer(
            periods_per_year,
            "periods_per_year",
        )
    )

    return (
        frame.cov()
        * periods_per_year
    )


def portfolio_returns(
    returns: pd.DataFrame,
    weights: WeightInput,
) -> pd.Series:
    """
    Calculate the periodic return series of a portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic asset returns.

    weights : WeightInput
        Portfolio weights.

        Pandas Series weights are aligned by asset name.
        Other one-dimensional inputs are aligned positionally.

    Returns
    -------
    pd.Series
        Periodic portfolio returns.

    Notes
    -----
    Rows containing missing asset returns are removed before portfolio
    returns are calculated.
    """
    frame = _prepare_return_frame(
        returns,
        minimum_observations=1,
        operation_name="Portfolio returns",
    )

    aligned_weights = _align_weights(
        weights,
        frame.columns,
    )

    values = (
        frame.to_numpy()
        @ aligned_weights.to_numpy()
    )

    return pd.Series(
        values,
        index=frame.index,
        name="portfolio_return",
        dtype="float64",
    )


def portfolio_expected_return(
    returns: pd.DataFrame,
    weights: WeightInput,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized expected portfolio return.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic asset returns.

    weights : WeightInput
        Portfolio weights.

    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float
        Annualized arithmetic expected return.
    """
    periods_per_year = (
        validate_positive_integer(
            periods_per_year,
            "periods_per_year",
        )
    )

    periodic_returns = (
        portfolio_returns(
            returns,
            weights,
        )
    )

    return float(
        periodic_returns.mean()
        * periods_per_year
    )


def portfolio_volatility(
    returns: pd.DataFrame,
    weights: WeightInput,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized portfolio volatility.

    Portfolio variance is calculated from the covariance matrix:

        w.T @ covariance @ w

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic asset returns.

    weights : WeightInput
        Portfolio weights.

    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float
        Annualized portfolio volatility.
    """
    frame = _prepare_return_frame(
        returns,
        minimum_observations=2,
        operation_name="Portfolio volatility",
    )

    aligned_weights = _align_weights(
        weights,
        frame.columns,
    )

    covariance = covariance_matrix(
        frame,
        periods_per_year=periods_per_year,
    )

    weight_array = (
        aligned_weights.to_numpy()
    )

    covariance_array = (
        covariance.to_numpy()
    )

    portfolio_variance = float(
        weight_array.T
        @ covariance_array
        @ weight_array
    )

    if portfolio_variance < -1e-12:
        raise ValueError(
            "Portfolio variance could not be calculated "
            "from the supplied covariance matrix."
        )

    portfolio_variance = max(
        portfolio_variance,
        0.0,
    )

    return float(
        np.sqrt(
            portfolio_variance
        )
    )


def sharpe_ratio(
    returns: pd.DataFrame,
    weights: WeightInput,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized portfolio Sharpe ratio.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic asset returns.

    weights : WeightInput
        Portfolio weights.

    risk_free_rate : float, default=0.0
        Annualized risk-free rate expressed as a decimal.

    periods_per_year : int, default=252
        Number of return periods in one year.

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Raises
    ------
    ValueError
        If portfolio volatility is zero.
    """
    risk_free_rate = (
        _validate_finite_number(
            risk_free_rate,
            "risk_free_rate",
        )
    )

    periods_per_year = (
        validate_positive_integer(
            periods_per_year,
            "periods_per_year",
        )
    )

    expected_return = (
        portfolio_expected_return(
            returns,
            weights,
            periods_per_year=periods_per_year,
        )
    )

    volatility = (
        portfolio_volatility(
            returns,
            weights,
            periods_per_year=periods_per_year,
        )
    )

    if np.isclose(
        volatility,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Sharpe ratio is undefined when "
            "portfolio volatility is zero."
        )

    return float(
        (
            expected_return
            - risk_free_rate
        )
        / volatility
    )
