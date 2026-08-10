"""Unit tests for QuantCore distribution diagnostics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import jarque_bera, kurtosis, skew

from quantcore.diagnostics import (
    DistributionDiagnostics,
    distribution_summary,
    excess_kurtosis,
    mean_return,
    normality_test,
    skewness,
)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Provide deterministic synthetic returns for diagnostic tests."""
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
        ],
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# Mean Return
# ---------------------------------------------------------------------------


def test_mean_return_matches_arithmetic_average(
    sample_returns,
):
    result = mean_return(sample_returns)

    expected = sample_returns.mean()

    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Skewness
# ---------------------------------------------------------------------------


def test_skewness_matches_scipy_bias_corrected_result(
    sample_returns,
):
    result = skewness(sample_returns)

    expected = skew(
        sample_returns.to_numpy(),
        bias=False,
    )

    assert result == pytest.approx(expected)


def test_skewness_requires_at_least_three_observations():
    returns = pd.Series(
        [0.01, -0.01]
    )

    with pytest.raises(
        ValueError,
        match="at least 3",
    ):
        skewness(returns)


# ---------------------------------------------------------------------------
# Kurtosis
# ---------------------------------------------------------------------------


def test_excess_kurtosis_matches_scipy_result(
    sample_returns,
):
    result = excess_kurtosis(
        sample_returns
    )

    expected = kurtosis(
        sample_returns.to_numpy(),
        fisher=True,
        bias=False,
    )

    assert result == pytest.approx(expected)


def test_excess_kurtosis_requires_four_observations():
    returns = pd.Series(
        [0.01, -0.01, 0.02]
    )

    with pytest.raises(
        ValueError,
        match="at least 4",
    ):
        excess_kurtosis(returns)


# ---------------------------------------------------------------------------
# Normality Testing
# ---------------------------------------------------------------------------


def test_normality_test_matches_jarque_bera(
    sample_returns,
):
    statistic, pvalue = normality_test(
        sample_returns
    )

    expected = jarque_bera(
        sample_returns.to_numpy()
    )

    assert statistic == pytest.approx(
        expected.statistic
    )

    assert pvalue == pytest.approx(
        expected.pvalue
    )


def test_normality_pvalue_is_valid_probability(
    sample_returns,
):
    _, pvalue = normality_test(
        sample_returns
    )

    assert 0.0 <= pvalue <= 1.0


# ---------------------------------------------------------------------------
# Distribution Summary
# ---------------------------------------------------------------------------


def test_distribution_summary_returns_structured_result(
    sample_returns,
):
    result = distribution_summary(
        sample_returns
    )

    assert isinstance(
        result,
        DistributionDiagnostics,
    )


def test_distribution_summary_reports_observation_count(
    sample_returns,
):
    result = distribution_summary(
        sample_returns
    )

    assert result.observations == len(
        sample_returns
    )


def test_distribution_summary_annualizes_volatility(
    sample_returns,
):
    result = distribution_summary(
        sample_returns,
        periods_per_year=252,
    )

    expected = (
        sample_returns.std(ddof=1)
        * np.sqrt(252)
    )

    assert result.annualized_volatility == pytest.approx(
        expected
    )


def test_distribution_summary_normality_flag_matches_pvalue(
    sample_returns,
):
    result = distribution_summary(
        sample_returns,
        significance_level=0.05,
    )

    assert result.normality_rejected == (
        result.normality_pvalue < 0.05
    )


# ---------------------------------------------------------------------------
# Input Cleaning
# ---------------------------------------------------------------------------


def test_diagnostics_ignore_nan_and_infinite_values(
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

    clean_result = mean_return(
        sample_returns
    )

    dirty_result = mean_return(
        dirty_returns
    )

    assert dirty_result == pytest.approx(
        clean_result
    )


@pytest.mark.parametrize(
    "diagnostic",
    [
        mean_return,
        skewness,
        excess_kurtosis,
        normality_test,
    ],
)
def test_diagnostics_reject_series_with_no_valid_observations(
    diagnostic,
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
        diagnostic(invalid_returns)


# ---------------------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "periods_per_year",
    [
        0,
        -1,
        -252,
    ],
)
def test_distribution_summary_rejects_invalid_annualization_period(
    sample_returns,
    periods_per_year,
):
    with pytest.raises(
        ValueError,
        match="periods_per_year",
    ):
        distribution_summary(
            sample_returns,
            periods_per_year=periods_per_year,
        )


@pytest.mark.parametrize(
    "significance_level",
    [
        -0.10,
        0.00,
        1.00,
        1.10,
    ],
)
def test_distribution_summary_rejects_invalid_significance_level(
    sample_returns,
    significance_level,
):
    with pytest.raises(
        ValueError,
        match="significance_level",
    ):
        distribution_summary(
            sample_returns,
            significance_level=significance_level,
        )
