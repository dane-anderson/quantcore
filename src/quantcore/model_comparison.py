"""
Tail-risk model comparison tools for QuantCore.

This module provides a standardized interface for running multiple
tail-risk models against the same return series.

The comparison layer coordinates model execution only. Individual
models remain responsible for their quantitative calculations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from quantcore._validation import (
    clean_returns,
    validate_confidence,
)
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


def available_models() -> tuple[str, ...]:
    """
    Return registered tail-risk model identifiers.

    Returns
    -------
    tuple[str, ...]
        Available model keys in registry order.
    """
    return tuple(
        MODEL_REGISTRY.keys()
    )


def run_tail_risk_analysis(
    returns: ReturnInput,
    confidence: float = 0.95,
    models: Sequence[str] | None = None,
) -> list[RiskModelResult]:
    """
    Run selected tail-risk models against the same return series.

    Parameters
    ----------
    returns : ReturnInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level supplied to each selected model.

    models : Sequence[str] or None, default=None
        Model identifiers to execute.

        Available models are:

        - ``"historical"``
        - ``"gaussian"``
        - ``"student_t"``

        If omitted, all registered models are executed.

    Returns
    -------
    list[RiskModelResult]
        Standardized results in requested model order.

    Raises
    ------
    ValueError
        If return data is invalid, confidence is invalid, no models
        are selected, or an unknown model identifier is supplied.
    """
    cleaned = clean_returns(
        returns
    )

    confidence = validate_confidence(
        confidence
    )

    if models is None:
        selected_models = list(
            MODEL_REGISTRY.keys()
        )
    else:
        selected_models = list(
            models
        )

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
        unknown = ", ".join(
            unknown_models
        )

        available = ", ".join(
            MODEL_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown risk model(s): {unknown}. "
            f"Available models: {available}."
        )

    results: list[
        RiskModelResult
    ] = []

    for model_key in selected_models:
        (
            model_name,
            var_function,
            expected_shortfall_function,
        ) = MODEL_REGISTRY[
            model_key
        ]

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
                observations=len(
                    cleaned
                ),
            )
        )

    return results
