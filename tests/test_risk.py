"""Unit tests for QuantCore tail-risk models."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as student_t_distribution

import quantcore.tail_risk as tail_risk

from quantcore.tail_risk import (
    gaussian_expected_shortfall,
    gaussian_var,
    historical_expected_shortfall,
    historical_var,
    student_t_expected_shortfall,
    student_t_var,
)


@pytest.fixture
def symmetric_returns() -> pd.Series:
    """Return a simple deterministic return series for risk-model tests."""
    return pd.Series(
        [-0.10, -0.05, 0.00, 0.05, 0.10],
        dtype="float64",
    )


@pytest.fixture
def realistic_returns() -> pd.Series:
    """Return a varied synthetic return series resembling market data."""
    return pd.Series(
        [
            0.012,
            0.008,
            -0.015,
            0.021,
            -0.032,
            0.014,
            -0.045,
            0.009,
            -0.018,
            0.025,
            -0.011,
            0.017,
            -0.027,
            0.006,
            -0.009,
        ],
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------


def test_historical_var_returns_expected_loss_threshold(
    symmetric_returns,
):
    result = historical_var(
        symmetric_returns,
        confidence=0.75,
    )

    assert result == pytest.approx(-0.05)


def test_historical_expected_shortfall_returns_average_tail_loss(
    symmetric_returns,
):
    result = historical_expected_shortfall(
        symmetric_returns,
        confidence=0.75,
    )

    assert result == pytest.approx(-0.075)


def test_historical_expected_shortfall_is_at_least_as_severe_as_var(
    realistic_returns,
):
    var = historical_var(
        realistic_returns,
        confidence=0.95,
    )

    expected_shortfall = historical_expected_shortfall(
        realistic_returns,
        confidence=0.95,
    )

    assert expected_shortfall <= var


# ---------------------------------------------------------------------------
# Gaussian Parametric Model
# ---------------------------------------------------------------------------


def test_gaussian_var_matches_parametric_formula(
    symmetric_returns,
):
    confidence = 0.95

    result = gaussian_var(
        symmetric_returns,
        confidence=confidence,
    )

    mean_return = symmetric_returns.mean()
    volatility = symmetric_returns.std(ddof=1)

    expected = (
        mean_return
        + volatility
        * norm.ppf(1.0 - confidence)
    )

    assert result == pytest.approx(expected)


def test_gaussian_expected_shortfall_matches_parametric_formula(
    symmetric_returns,
):
    confidence = 0.95
    alpha = 1.0 - confidence

    result = gaussian_expected_shortfall(
        symmetric_returns,
        confidence=confidence,
    )

    mean_return = symmetric_returns.mean()
    volatility = symmetric_returns.std(ddof=1)

    z_score = norm.ppf(alpha)

    expected = (
        mean_return
        - volatility
        * norm.pdf(z_score)
        / alpha
    )

    assert result == pytest.approx(expected)


def test_gaussian_expected_shortfall_is_more_severe_than_var(
    realistic_returns,
):
    var = gaussian_var(
        realistic_returns,
        confidence=0.95,
    )

    expected_shortfall = gaussian_expected_shortfall(
        realistic_returns,
        confidence=0.95,
    )

    assert expected_shortfall <= var


# ---------------------------------------------------------------------------
# Student-t Parametric Model
# ---------------------------------------------------------------------------


def test_student_t_var_uses_fitted_distribution(
    realistic_returns,
    monkeypatch,
):
    degrees_of_freedom = 5.0
    location = 0.01
    scale = 0.02
    confidence = 0.95

    monkeypatch.setattr(
        tail_risk.t,
        "fit",
        lambda values: (
            degrees_of_freedom,
            location,
            scale,
        ),
    )

    result = student_t_var(
        realistic_returns,
        confidence=confidence,
    )

    quantile = student_t_distribution.ppf(
        1.0 - confidence,
        degrees_of_freedom,
    )

    expected = (
        location
        + scale * quantile
    )

    assert result == pytest.approx(expected)


def test_student_t_expected_shortfall_matches_analytical_formula(
    realistic_returns,
    monkeypatch,
):
    degrees_of_freedom = 5.0
    location = 0.01
    scale = 0.02
    confidence = 0.95

    monkeypatch.setattr(
        tail_risk.t,
        "fit",
        lambda values: (
            degrees_of_freedom,
            location,
            scale,
        ),
    )

    result = student_t_expected_shortfall(
        realistic_returns,
        confidence=confidence,
    )

    alpha = 1.0 - confidence

    quantile = student_t_distribution.ppf(
        alpha,
        degrees_of_freedom,
    )

    density = student_t_distribution.pdf(
        quantile,
        degrees_of_freedom,
    )

    expected = (
        location
        - scale
        * (
            (
                degrees_of_freedom
                + quantile**2
            )
            / (
                degrees_of_freedom
                - 1.0
            )
        )
        * (
            density
            / alpha
        )
    )

    assert result == pytest.approx(expected)


def test_student_t_expected_shortfall_rejects_undefined_mean(
    realistic_returns,
    monkeypatch,
):
    monkeypatch.setattr(
        tail_risk.t,
        "fit",
        lambda values: (
            1.0,
            0.0,
            0.02,
        ),
    )

    with pytest.raises(
        ValueError,
        match="degrees of freedom",
    ):
        student_t_expected_shortfall(
            realistic_returns
        )


# ---------------------------------------------------------------------------
# Input Cleaning
# ---------------------------------------------------------------------------


def test_models_ignore_nan_and_infinite_observations():
    clean_returns = pd.Series(
        [-0.10, -0.05, 0.00, 0.05, 0.10]
    )

    dirty_returns = pd.Series(
        [
            np.nan,
            np.inf,
            -np.inf,
            -0.10,
            -0.05,
            0.00,
            0.05,
            0.10,
        ]
    )

    clean_result = historical_var(
        clean_returns,
        confidence=0.75,
    )

    dirty_result = historical_var(
        dirty_returns,
        confidence=0.75,
    )

    assert dirty_result == pytest.approx(
        clean_result
    )


@pytest.mark.parametrize(
    "model",
    [
        historical_var,
        historical_expected_shortfall,
        gaussian_var,
        gaussian_expected_shortfall,
        student_t_var,
        student_t_expected_shortfall,
    ],
)
def test_models_reject_return_series_with_no_valid_observations(
    model,
):
    invalid_returns = pd.Series(
        [
            np.nan,
            np.inf,
            -np.inf,
        ]
    )

    with pytest.raises(
        ValueError,
        match="no valid observations",
    ):
        model(invalid_returns)


# ---------------------------------------------------------------------------
# Confidence-Level Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence",
    [
        -0.10,
        0.00,
        1.00,
        1.10,
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        historical_var,
        historical_expected_shortfall,
        gaussian_var,
        gaussian_expected_shortfall,
        student_t_var,
        student_t_expected_shortfall,
    ],
)
def test_models_reject_invalid_confidence_levels(
    model,
    confidence,
    realistic_returns,
):
    with pytest.raises(
        ValueError,
        match="Confidence",
    ):
        model(
            realistic_returns,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Observation Requirements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        gaussian_var,
        gaussian_expected_shortfall,
    ],
)
def test_gaussian_models_require_at_least_two_observations(
    model,
):
    returns = pd.Series([0.01])

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        model(returns)


@pytest.mark.parametrize(
    "model",
    [
        student_t_var,
        student_t_expected_shortfall,
    ],
)
def test_student_t_models_require_at_least_three_observations(
    model,
):
    returns = pd.Series(
        [0.01, -0.01]
    )

    with pytest.raises(
        ValueError,
        match="at least 3",
    ):
        model(returns)
