"""Unit tests for QuantCore tail-risk model comparison."""

import numpy as np
import pandas as pd
import pytest

from quantcore.model_comparison import (
    RiskModelResult,
    available_models,
    run_tail_risk_analysis,
)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Provide deterministic return observations for comparison tests."""
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
# Model Registry
# ---------------------------------------------------------------------------


def test_available_models_returns_registered_model_keys():
    result = available_models()

    assert result == (
        "historical",
        "gaussian",
        "student_t",
    )


# ---------------------------------------------------------------------------
# Complete Model Comparison
# ---------------------------------------------------------------------------


def test_analysis_runs_all_models_by_default(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns
    )

    assert len(results) == 3


def test_analysis_returns_structured_results(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns
    )

    assert all(
        isinstance(
            result,
            RiskModelResult,
        )
        for result in results
    )


def test_analysis_preserves_model_order(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns
    )

    model_keys = [
        result.model_key
        for result in results
    ]

    assert model_keys == [
        "historical",
        "gaussian",
        "student_t",
    ]


def test_analysis_reports_valid_observation_count(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns
    )

    assert all(
        result.observations
        == len(sample_returns)
        for result in results
    )


def test_analysis_preserves_confidence_level(
    sample_returns,
):
    confidence = 0.99

    results = run_tail_risk_analysis(
        sample_returns,
        confidence=confidence,
    )

    assert all(
        result.confidence
        == pytest.approx(confidence)
        for result in results
    )


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------


def test_analysis_can_run_single_selected_model(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns,
        models=["historical"],
    )

    assert len(results) == 1

    assert (
        results[0].model_key
        == "historical"
    )


def test_analysis_can_run_multiple_selected_models(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns,
        models=[
            "gaussian",
            "student_t",
        ],
    )

    model_keys = [
        result.model_key
        for result in results
    ]

    assert model_keys == [
        "gaussian",
        "student_t",
    ]


def test_analysis_preserves_requested_model_order(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns,
        models=[
            "student_t",
            "historical",
        ],
    )

    model_keys = [
        result.model_key
        for result in results
    ]

    assert model_keys == [
        "student_t",
        "historical",
    ]


# ---------------------------------------------------------------------------
# Risk Relationships
# ---------------------------------------------------------------------------


def test_expected_shortfall_is_at_least_as_severe_as_var(
    sample_returns,
):
    results = run_tail_risk_analysis(
        sample_returns
    )

    for result in results:
        assert (
            result.expected_shortfall
            <= result.value_at_risk
        )


# ---------------------------------------------------------------------------
# Input Cleaning
# ---------------------------------------------------------------------------


def test_analysis_ignores_nan_and_infinite_observations(
    sample_returns,
):
    dirty_returns = pd.concat(
        [
            pd.Series(
                [
                    np.nan,
                    np.inf,
                    -np.inf,
                ]
            ),
            sample_returns,
        ],
        ignore_index=True,
    )

    results = run_tail_risk_analysis(
        dirty_returns,
        models=["historical"],
    )

    assert results[0].observations == len(
        sample_returns
    )


def test_analysis_rejects_series_with_no_valid_observations():
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
        run_tail_risk_analysis(
            invalid_returns
        )


# ---------------------------------------------------------------------------
# Model Validation
# ---------------------------------------------------------------------------


def test_analysis_rejects_unknown_model(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="Unknown risk model",
    ):
        run_tail_risk_analysis(
            sample_returns,
            models=["made_up_model"],
        )


def test_analysis_rejects_empty_model_selection(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        run_tail_risk_analysis(
            sample_returns,
            models=[],
        )
