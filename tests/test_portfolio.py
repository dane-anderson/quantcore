"""Unit tests for QuantCore portfolio analytics."""

import numpy as np
import pandas as pd
import pytest

from quantcore.portfolio import (
    correlation_matrix,
    covariance_matrix,
    portfolio_expected_return,
    portfolio_returns,
    portfolio_volatility,
    sharpe_ratio,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Provide deterministic multi-asset returns for portfolio tests."""
    return pd.DataFrame(
        {
            "AAPL": [
                0.010,
                -0.020,
                0.015,
                0.005,
                -0.010,
            ],
            "MSFT": [
                0.008,
                -0.010,
                0.012,
                0.007,
                -0.006,
            ],
            "NVDA": [
                0.020,
                -0.030,
                0.025,
                0.010,
                -0.015,
            ],
        },
        dtype="float64",
    )


@pytest.fixture
def equal_weights() -> np.ndarray:
    """Provide equal weights for three assets."""
    return np.array(
        [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------


def test_correlation_matrix_matches_pandas(
    sample_returns,
):
    result = correlation_matrix(
        sample_returns
    )

    expected = sample_returns.corr(
        method="pearson"
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_correlation_matrix_preserves_asset_labels(
    sample_returns,
):
    result = correlation_matrix(
        sample_returns
    )

    assert list(result.index) == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]

    assert list(result.columns) == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]


@pytest.mark.parametrize(
    "method",
    [
        "pearson",
        "spearman",
        "kendall",
    ],
)
def test_correlation_matrix_supports_registered_methods(
    sample_returns,
    method,
):
    result = correlation_matrix(
        sample_returns,
        method=method,
    )

    expected = sample_returns.corr(
        method=method
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_correlation_method_is_case_insensitive(
    sample_returns,
):
    lower = correlation_matrix(
        sample_returns,
        method="pearson",
    )

    upper = correlation_matrix(
        sample_returns,
        method="PEARSON",
    )

    pd.testing.assert_frame_equal(
        lower,
        upper,
    )


def test_correlation_matrix_rejects_unknown_method(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="Unknown correlation method",
    ):
        correlation_matrix(
            sample_returns,
            method="made_up_method",
        )


# ---------------------------------------------------------------------------
# Covariance Matrix
# ---------------------------------------------------------------------------


def test_covariance_matrix_matches_annualized_sample_covariance(
    sample_returns,
):
    result = covariance_matrix(
        sample_returns,
        periods_per_year=252,
    )

    expected = (
        sample_returns.cov()
        * 252
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_covariance_matrix_can_disable_annualization(
    sample_returns,
):
    result = covariance_matrix(
        sample_returns,
        periods_per_year=1,
    )

    expected = sample_returns.cov()

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Portfolio Return Series
# ---------------------------------------------------------------------------


def test_portfolio_returns_matches_weighted_asset_returns(
    sample_returns,
):
    weights = np.array(
        [
            0.50,
            0.30,
            0.20,
        ]
    )

    result = portfolio_returns(
        sample_returns,
        weights,
    )

    expected_values = (
        sample_returns.to_numpy()
        @ weights
    )

    expected = pd.Series(
        expected_values,
        index=sample_returns.index,
        name="portfolio_return",
        dtype="float64",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_portfolio_returns_preserves_observation_index(
    sample_returns,
    equal_weights,
):
    sample_returns.index = pd.date_range(
        "2026-01-01",
        periods=len(sample_returns),
        freq="D",
    )

    result = portfolio_returns(
        sample_returns,
        equal_weights,
    )

    assert result.index.equals(
        sample_returns.index
    )


# ---------------------------------------------------------------------------
# Weight Alignment
# ---------------------------------------------------------------------------


def test_labeled_weights_are_aligned_by_asset_name(
    sample_returns,
):
    weights = pd.Series(
        {
            "NVDA": 0.20,
            "AAPL": 0.50,
            "MSFT": 0.30,
        }
    )

    result = portfolio_returns(
        sample_returns,
        weights,
    )

    expected_weights = np.array(
        [
            0.50,
            0.30,
            0.20,
        ]
    )

    expected = (
        sample_returns.to_numpy()
        @ expected_weights
    )

    assert np.allclose(
        result.to_numpy(),
        expected,
    )


def test_portfolio_allows_short_positions_when_weights_net_to_one(
    sample_returns,
):
    weights = pd.Series(
        {
            "AAPL": 0.80,
            "MSFT": 0.50,
            "NVDA": -0.30,
        }
    )

    result = portfolio_returns(
        sample_returns,
        weights,
    )

    expected = (
        sample_returns.to_numpy()
        @ np.array(
            [
                0.80,
                0.50,
                -0.30,
            ]
        )
    )

    assert np.allclose(
        result.to_numpy(),
        expected,
    )


def test_portfolio_rejects_wrong_number_of_positional_weights(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="number of assets",
    ):
        portfolio_returns(
            sample_returns,
            [
                0.50,
                0.50,
            ],
        )


def test_portfolio_rejects_weights_that_do_not_sum_to_one(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        portfolio_returns(
            sample_returns,
            [
                0.50,
                0.30,
                0.30,
            ],
        )


def test_portfolio_rejects_non_finite_weights(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        portfolio_returns(
            sample_returns,
            [
                0.50,
                np.nan,
                0.50,
            ],
        )


def test_portfolio_rejects_multidimensional_weights(
    sample_returns,
):
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        portfolio_returns(
            sample_returns,
            [
                [0.50, 0.30, 0.20]
            ],
        )


def test_portfolio_rejects_missing_labeled_asset_weight(
    sample_returns,
):
    weights = pd.Series(
        {
            "AAPL": 0.60,
            "MSFT": 0.40,
        }
    )

    with pytest.raises(
        ValueError,
        match="missing assets",
    ):
        portfolio_returns(
            sample_returns,
            weights,
        )


def test_portfolio_rejects_unknown_labeled_asset_weight(
    sample_returns,
):
    weights = pd.Series(
        {
            "AAPL": 0.40,
            "MSFT": 0.30,
            "NVDA": 0.20,
            "TSLA": 0.10,
        }
    )

    with pytest.raises(
        ValueError,
        match="unknown assets",
    ):
        portfolio_returns(
            sample_returns,
            weights,
        )


# ---------------------------------------------------------------------------
# Expected Portfolio Return
# ---------------------------------------------------------------------------


def test_portfolio_expected_return_matches_weighted_mean(
    sample_returns,
):
    weights = np.array(
        [
            0.50,
            0.30,
            0.20,
        ]
    )

    result = portfolio_expected_return(
        sample_returns,
        weights,
        periods_per_year=252,
    )

    periodic_portfolio_returns = (
        sample_returns.to_numpy()
        @ weights
    )

    expected = (
        periodic_portfolio_returns.mean()
        * 252
    )

    assert result == pytest.approx(
        expected
    )


def test_portfolio_expected_return_without_annualization_matches_periodic_mean(
    sample_returns,
    equal_weights,
):
    result = portfolio_expected_return(
        sample_returns,
        equal_weights,
        periods_per_year=1,
    )

    expected = portfolio_returns(
        sample_returns,
        equal_weights,
    ).mean()

    assert result == pytest.approx(
        expected
    )


# ---------------------------------------------------------------------------
# Portfolio Volatility
# ---------------------------------------------------------------------------


def test_portfolio_volatility_matches_covariance_formula(
    sample_returns,
):
    weights = np.array(
        [
            0.50,
            0.30,
            0.20,
        ]
    )

    result = portfolio_volatility(
        sample_returns,
        weights,
        periods_per_year=252,
    )

    covariance = (
        sample_returns.cov().to_numpy()
        * 252
    )

    expected_variance = (
        weights.T
        @ covariance
        @ weights
    )

    expected = np.sqrt(
        expected_variance
    )

    assert result == pytest.approx(
        expected
    )


def test_portfolio_volatility_is_non_negative(
    sample_returns,
    equal_weights,
):
    result = portfolio_volatility(
        sample_returns,
        equal_weights,
    )

    assert result >= 0.0


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_matches_expected_formula(
    sample_returns,
):
    weights = np.array(
        [
            0.50,
            0.30,
            0.20,
        ]
    )

    risk_free_rate = 0.04

    result = sharpe_ratio(
        sample_returns,
        weights,
        risk_free_rate=risk_free_rate,
        periods_per_year=252,
    )

    expected_return = (
        portfolio_expected_return(
            sample_returns,
            weights,
            periods_per_year=252,
        )
    )

    expected_volatility = (
        portfolio_volatility(
            sample_returns,
            weights,
            periods_per_year=252,
        )
    )

    expected = (
        expected_return
        - risk_free_rate
    ) / expected_volatility

    assert result == pytest.approx(
        expected
    )


def test_sharpe_ratio_supports_negative_risk_free_rate(
    sample_returns,
    equal_weights,
):
    result = sharpe_ratio(
        sample_returns,
        equal_weights,
        risk_free_rate=-0.01,
    )

    assert np.isfinite(
        result
    )


def test_sharpe_ratio_rejects_zero_volatility_portfolio():
    returns = pd.DataFrame(
        {
            "AAPL": [
                0.01,
                0.01,
                0.01,
                0.01,
            ],
            "MSFT": [
                0.01,
                0.01,
                0.01,
                0.01,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="volatility is zero",
    ):
        sharpe_ratio(
            returns,
            [
                0.50,
                0.50,
            ],
        )


# ---------------------------------------------------------------------------
# Missing Data
# ---------------------------------------------------------------------------


def test_portfolio_analysis_removes_incomplete_rows():
    returns = pd.DataFrame(
        {
            "AAPL": [
                0.01,
                np.nan,
                0.03,
            ],
            "MSFT": [
                0.02,
                0.01,
                0.04,
            ],
        }
    )

    result = portfolio_returns(
        returns,
        [
            0.50,
            0.50,
        ],
    )

    assert list(
        result.index
    ) == [
        0,
        2,
    ]


def test_portfolio_analysis_rejects_no_complete_observations():
    returns = pd.DataFrame(
        {
            "AAPL": [
                0.01,
                np.nan,
            ],
            "MSFT": [
                np.nan,
                0.02,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="no complete observations",
    ):
        portfolio_returns(
            returns,
            [
                0.50,
                0.50,
            ],
        )


# ---------------------------------------------------------------------------
# Return Data Validation
# ---------------------------------------------------------------------------


def test_portfolio_rejects_non_dataframe_returns():
    with pytest.raises(
        ValueError,
        match="pandas DataFrame",
    ):
        portfolio_returns(
            [
                [0.01, 0.02],
                [0.03, 0.04],
            ],
            [
                0.50,
                0.50,
            ],
        )


def test_portfolio_rejects_empty_dataframe():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        portfolio_returns(
            pd.DataFrame(),
            [],
        )


def test_portfolio_rejects_duplicate_asset_names():
    returns = pd.DataFrame(
        np.array(
            [
                [0.01, 0.02],
                [0.03, 0.04],
            ]
        ),
        columns=[
            "AAPL",
            "AAPL",
        ],
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        portfolio_returns(
            returns,
            [
                0.50,
                0.50,
            ],
        )


def test_portfolio_rejects_non_numeric_return_data():
    returns = pd.DataFrame(
        {
            "AAPL": [
                0.01,
                "invalid",
            ],
            "MSFT": [
                0.02,
                0.03,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        portfolio_returns(
            returns,
            [
                0.50,
                0.50,
            ],
        )


def test_portfolio_rejects_infinite_return_data():
    returns = pd.DataFrame(
        {
            "AAPL": [
                0.01,
                np.inf,
            ],
            "MSFT": [
                0.02,
                0.03,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="infinite",
    ):
        portfolio_returns(
            returns,
            [
                0.50,
                0.50,
            ],
        )


# ---------------------------------------------------------------------------
# Annualization Validation
# ---------------------------------------------------------------------------


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
def test_portfolio_expected_return_rejects_invalid_annualization(
    sample_returns,
    equal_weights,
    periods_per_year,
):
    with pytest.raises(
        ValueError,
        match="periods_per_year",
    ):
        portfolio_expected_return(
            sample_returns,
            equal_weights,
            periods_per_year=periods_per_year,
        )


@pytest.mark.parametrize(
    "periods_per_year",
    [
        0,
        -252,
        1.5,
        np.nan,
        np.inf,
    ],
)
def test_portfolio_volatility_rejects_invalid_annualization(
    sample_returns,
    equal_weights,
    periods_per_year,
):
    with pytest.raises(
        ValueError,
        match="periods_per_year",
    ):
        portfolio_volatility(
            sample_returns,
            equal_weights,
            periods_per_year=periods_per_year,
        )


# ---------------------------------------------------------------------------
# Risk-Free Rate Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "risk_free_rate",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        "invalid",
    ],
)
def test_sharpe_ratio_rejects_invalid_risk_free_rate(
    sample_returns,
    equal_weights,
    risk_free_rate,
):
    with pytest.raises(
        ValueError,
        match="risk_free_rate",
    ):
        sharpe_ratio(
            sample_returns,
            equal_weights,
            risk_free_rate=risk_free_rate,
        )
