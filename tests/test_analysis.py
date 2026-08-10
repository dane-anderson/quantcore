"""Unit tests for QuantCore high-level risk analysis."""

import numpy as np
import pandas as pd
import pytest

from quantcore.analysis import (
    RiskAnalysisResult,
    analyze_risk,
)
from quantcore.diagnostics import (
    DistributionDiagnostics,
)
from quantcore.model_comparison import (
    RiskModelResult,
)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Provide deterministic returns for high-level analysis tests."""
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
            0.019,
            -0.013,
            0.011,
            -0.022,
            0.016,
        ],
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# Structured Result
# ---------------------------------------------------------------------------


def test_analyze_risk_returns_structured_result(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert isinstance(
        result,
        RiskAnalysisResult,
    )


def test_analysis_contains_distribution_diagnostics(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert isinstance(
        result.diagnostics,
        DistributionDiagnostics,
    )


def test_analysis_contains_structured_model_results(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert all(
        isinstance(
            model,
            RiskModelResult,
        )
        for model in result.models
    )


# ---------------------------------------------------------------------------
# Default Workflow
# ---------------------------------------------------------------------------


def test_analysis_runs_all_tail_risk_models_by_default(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert result.model_keys == (
        "historical",
        "gaussian",
        "student_t",
    )


def test_analysis_reports_correct_model_count(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert result.model_count == 3


def test_analysis_reports_valid_observation_count(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert (
        result.observations
        == len(sample_returns)
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_analysis_preserves_confidence_level(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        confidence=0.99,
    )

    assert result.confidence == pytest.approx(
        0.99
    )

    assert all(
        model.confidence
        == pytest.approx(0.99)
        for model in result.models
    )


def test_analysis_preserves_periods_per_year(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        periods_per_year=52,
    )

    assert result.periods_per_year == 52


def test_analysis_preserves_significance_level(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        significance_level=0.01,
    )

    assert (
        result.significance_level
        == pytest.approx(0.01)
    )


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------


def test_analysis_can_run_single_model(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        models=[
            "historical",
        ],
    )

    assert result.model_count == 1

    assert result.model_keys == (
        "historical",
    )


def test_analysis_can_run_selected_models(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        models=[
            "gaussian",
            "student_t",
        ],
    )

    assert result.model_keys == (
        "gaussian",
        "student_t",
    )


def test_analysis_preserves_requested_model_order(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        models=[
            "student_t",
            "historical",
        ],
    )

    assert result.model_keys == (
        "student_t",
        "historical",
    )


# ---------------------------------------------------------------------------
# Cross-Component Consistency
# ---------------------------------------------------------------------------


def test_all_models_use_same_observation_count(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    assert all(
        model.observations
        == result.observations
        for model in result.models
    )


def test_expected_shortfall_is_at_least_as_severe_as_var(
    sample_returns,
):
    result = analyze_risk(
        sample_returns
    )

    for model in result.models:
        assert (
            model.expected_shortfall
            <= model.value_at_risk
        )


def test_annualized_volatility_uses_requested_frequency(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        periods_per_year=12,
    )

    expected = (
        sample_returns.std(ddof=1)
        * np.sqrt(12)
    )

    assert (
        result.diagnostics.annualized_volatility
        == pytest.approx(expected)
    )


# ---------------------------------------------------------------------------
# Input Cleaning
# ---------------------------------------------------------------------------


def test_analysis_removes_non_finite_returns_consistently(
    sample_returns,
):
    dirty_returns = pd.concat(
        [
            sample_returns,
            pd.Series(
                [
                    np.nan,
                    np.inf,
                    -np.inf,
                ]
            ),
        ],
        ignore_index=True,
    )

    result = analyze_risk(
        dirty_returns
    )

    assert (
        result.observations
        == len(sample_returns)
    )

    assert all(
        model.observations
        == len(sample_returns)
        for model in result.models
    )


def test_analysis_rejects_no_valid_observations():
    returns = pd.Series(
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
        analyze_risk(
            returns
        )


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence",
    [
        -0.10,
        0.00,
        1.00,
        1.10,
        np.nan,
        np.inf,
    ],
)
def test_analysis_rejects_invalid_confidence(
    sample_returns,
    confidence,
):
    with pytest.raises(
        ValueError,
        match="Confidence",
    ):
        analyze_risk(
            sample_returns,
            confidence=confidence,
        )


@pytest.mark.parametrize(
    "periods_per_year",
    [
        0,
        -1,
        2.5,
        np.nan,
        np.inf,
    ],
)
def test_analysis_rejects_invalid_periods_per_year(
    sample_returns,
    periods_per_year,
):
    with pytest.raises(
        ValueError,
        match="periods_per_year",
    ):
        analyze_risk(
            sample_returns,
            periods_per_year=periods_per_year,
        )


@pytest.mark.parametrize(
    "significance_level",
    [
        -0.05,
        0.00,
        1.00,
        1.05,
        np.nan,
        np.inf,
    ],
)
def test_analysis_rejects_invalid_significance_level(
    sample_returns,
    significance_level,
):
    with pytest.raises(
        ValueError,
        match="significance_level",
    ):
        analyze_risk(
            sample_returns,
            significance_level=significance_level,
        )


def test_analysis_rejects_unknown_model(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="Unknown risk model",
    ):
        analyze_risk(
            sample_returns,
            models=[
                "fake_model",
            ],
        )


def test_analysis_rejects_empty_model_selection(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        analyze_risk(
            sample_returns,
            models=[],
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_analysis_can_convert_to_dictionary(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        models=[
            "historical",
        ],
    )

    serialized = result.to_dict()

    assert isinstance(
        serialized,
        dict,
    )

    assert serialized["confidence"] == pytest.approx(
        0.95
    )

    assert serialized["periods_per_year"] == 252

    assert serialized[
        "significance_level"
    ] == pytest.approx(
        0.05
    )


def test_serialized_analysis_contains_nested_diagnostics(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        models=[
            "historical",
        ],
    )

    serialized = result.to_dict()

    assert isinstance(
        serialized["diagnostics"],
        dict,
    )

    assert (
        serialized["diagnostics"][
            "observations"
        ]
        == len(sample_returns)
    )


def test_serialized_analysis_contains_model_results(
    sample_returns,
):
    result = analyze_risk(
        sample_returns,
        models=[
            "historical",
        ],
    )

    serialized = result.to_dict()

    assert isinstance(
        serialized["models"],
        tuple,
    )

    assert (
        serialized["models"][0][
            "model_key"
        ]
        == "historical"
    )
