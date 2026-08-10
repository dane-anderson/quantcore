"""
Tail-risk model comparison tools for QuantCore.

This module provides a standardized interface for running multiple
tail-risk models against the same return series.

The comparison layer does not reinterpret or modify model outputs.
Each model remains responsible for its own quantitative calculation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from quantcore.tail_risk import (
    gaussian_expected_shortfall,
    gaussian_var,
    historical_expected_shortfall,
    historical_var,
    student_t_expected_shortfall,
    student_t_var,
)


RiskFunction = Callable[..., float]
ReturnInput = pd.Series | np.ndarray | Sequence[float]


@dataclass(frozen=True)
class RiskModelResult:
    """
    Standardized result from a tail-risk model.

    Attributes
    ----------
    model_key : str
        Machine-readable model identifier.

    model_name : str
        Human-readable model name.

    confidence : float
        Confidence level used for the analysis.

    value_at_risk : float
        Signed Value at Risk threshold.

    expected_shortfall : float
        Signed Expected Shortfall estimate.

    observations : int
        Number of valid return observations used.
    """

    model_key: str
    model_name: str
    confidence: float
    value_at_risk: float
    expected_shortfall: float
    observations: int


MODEL_REGISTRY: dict[
    str,
    tuple[
        str,
        RiskFunction,
        RiskFunction,
    ],
] = {
    "historical": (
        "Historical Simulation",
        historical_var,
        historical_expected_shortfall,
    ),
    "gaussian": (
        "Gaussian",
        gaussian_var,
        gaussian_expected_shortfall,
    ),
    "student_t": (
        "Student-t",
        student_t_var,
        student_t_expected_shortfall,
    ),
}


__all__ = [
    "RiskModelResult",
    "available_models",
    "run_tail_risk_analysis",
]


def _clean_returns(
    returns: ReturnInput,
) -> pd.Series:
    """
    Normalize return observations for model comparison.
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


def available_models() -> tuple[str, ...]:
    """
    Return the registered tail-risk model keys.

    Returns
    -------
    tuple[str, ...]
        Available model identifiers.
    """
    return tuple(MODEL_REGISTRY.keys())


def run_tail_risk_analysis(
    returns: ReturnInput,
    confidence: float = 0.95,
    models: Sequence[str] | None = None,
) -> list[RiskModelResult]:
    """
    Run selected tail-risk models on the same return series.

    Parameters
    ----------
    returns : ReturnInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level supplied to each model.

    models : Sequence[str] or None, default=None
        Model keys to execute.

        Available values are:

        - ``"historical"``
        - ``"gaussian"``
        - ``"student_t"``

        If omitted, all registered models are executed.

    Returns
    -------
    list[RiskModelResult]
        Standardized results for each selected model.

    Raises
    ------
    ValueError
        If the return series contains no valid observations,
        no models are selected, or an unknown model key is supplied.
    """
    cleaned = _clean_returns(returns)

    if models is None:
        selected_models = list(
            MODEL_REGISTRY.keys()
        )
    else:
        selected_models = list(models)

    if not selected_models:
        raise ValueError(
            "At least one risk model must be selected."
        )

    unknown_models = [
        model
        for model in selected_models
        if model not in MODEL_REGISTRY
    ]

    if unknown_models:
        available = ", ".join(
            MODEL_REGISTRY.keys()
        )

        unknown = ", ".join(
            unknown_models
        )

        raise ValueError(
            f"Unknown risk model(s): {unknown}. "
            f"Available models: {available}."
        )

    results: list[RiskModelResult] = []

    for model_key in selected_models:
        (
            model_name,
            var_function,
            expected_shortfall_function,
        ) = MODEL_REGISTRY[model_key]

        value_at_risk = var_function(
            cleaned,
            confidence=confidence,
        )

        expected_shortfall = (
            expected_shortfall_function(
                cleaned,
                confidence=confidence,
            )
        )

        results.append(
            RiskModelResult(
                model_key=model_key,
                model_name=model_name,
                confidence=confidence,
                value_at_risk=float(
                    value_at_risk
                ),
                expected_shortfall=float(
                    expected_shortfall
                ),
                observations=len(cleaned),
            )
        )

    return results
