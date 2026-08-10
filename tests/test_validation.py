"""Unit tests for QuantCore internal validation utilities."""

import numpy as np
import pandas as pd
import pytest

from quantcore._validation import (
    clean_return_array,
    clean_returns,
    require_min_observations,
    validate_confidence,
    validate_integer,
    validate_positive_finite_number,
    validate_positive_integer,
    validate_significance_level,
)


# ---------------------------------------------------------------------------
# Return Cleaning
# ---------------------------------------------------------------------------


def test_clean_returns_returns_float_series():
    result = clean_returns(
        [0.01, -0.02, 0.03]
    )

    assert isinstance(
        result,
        pd.Series,
    )

    assert result.dtype == "float64"


def test_clean_returns_removes_non_finite_values():
    result = clean_returns(
        [
            0.01,
            np.nan,
            np.inf,
            -np.inf,
            -0.02,
        ]
    )

    expected = pd.Series(
        [0.01, -0.02],
        dtype="float64",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_clean_return_array_returns_numpy_array():
    result = clean_return_array(
        [0.01, -0.02, 0.03]
    )

    assert isinstance(
        result,
        np.ndarray,
    )

    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "cleaner",
    [
        clean_returns,
        clean_return_array,
    ],
)
def test_return_cleaners_reject_multidimensional_input(
    cleaner,
):
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        cleaner(
            [
                [0.01, 0.02],
                [-0.01, 0.03],
            ]
        )


@pytest.mark.parametrize(
    "cleaner",
    [
        clean_returns,
        clean_return_array,
    ],
)
def test_return_cleaners_reject_no_valid_observations(
    cleaner,
):
    with pytest.raises(
        ValueError,
        match="no valid observations",
    ):
        cleaner(
            [
                np.nan,
                np.inf,
                -np.inf,
            ]
        )


# ---------------------------------------------------------------------------
# Observation Requirements
# ---------------------------------------------------------------------------


def test_minimum_observation_validation_accepts_valid_sample():
    require_min_observations(
        [1, 2, 3],
        minimum=3,
        operation_name="Test operation",
    )


def test_minimum_observation_validation_rejects_small_sample():
    with pytest.raises(
        ValueError,
        match="at least 3",
    ):
        require_min_observations(
            [1, 2],
            minimum=3,
            operation_name="Test operation",
        )


# ---------------------------------------------------------------------------
# Probability Validation
# ---------------------------------------------------------------------------


def test_confidence_validation_returns_valid_probability():
    result = validate_confidence(
        0.95
    )

    assert result == pytest.approx(
        0.95
    )


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
def test_confidence_validation_rejects_invalid_values(
    confidence,
):
    with pytest.raises(
        ValueError,
        match="Confidence",
    ):
        validate_confidence(
            confidence
        )


@pytest.mark.parametrize(
    "significance_level",
    [
        0.00,
        1.00,
        -0.05,
        1.05,
        np.nan,
    ],
)
def test_significance_validation_rejects_invalid_values(
    significance_level,
):
    with pytest.raises(
        ValueError,
        match="significance_level",
    ):
        validate_significance_level(
            significance_level
        )


# ---------------------------------------------------------------------------
# Integer Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        1,
        252,
        10_000,
        5.0,
    ],
)
def test_positive_integer_validation_accepts_whole_numbers(
    value,
):
    result = validate_positive_integer(
        value,
        "value",
    )

    assert isinstance(
        result,
        int,
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        2.5,
        np.nan,
        np.inf,
        True,
    ],
)
def test_positive_integer_validation_rejects_invalid_values(
    value,
):
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        validate_positive_integer(
            value,
            "value",
        )


def test_integer_validation_allows_negative_seed():
    result = validate_integer(
        -42,
        "seed",
    )

    assert result == -42


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        np.nan,
        np.inf,
        True,
        "invalid",
    ],
)
def test_integer_validation_rejects_non_integer_values(
    value,
):
    with pytest.raises(
        ValueError,
        match="integer",
    ):
        validate_integer(
            value,
            "seed",
        )


# ---------------------------------------------------------------------------
# Positive Finite Numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        1,
        100.5,
        10_000,
    ],
)
def test_positive_finite_number_accepts_valid_values(
    value,
):
    result = validate_positive_finite_number(
        value,
        "initial_value",
    )

    assert result > 0


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        np.nan,
        np.inf,
        True,
        "invalid",
    ],
)
def test_positive_finite_number_rejects_invalid_values(
    value,
):
    with pytest.raises(
        ValueError,
        match="positive finite number",
    ):
        validate_positive_finite_number(
            value,
            "initial_value",
        )
