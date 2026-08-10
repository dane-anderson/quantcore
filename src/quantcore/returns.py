"""Unit tests for QuantCore return transformations."""

import numpy as np
import pandas as pd
import pytest

from quantcore.returns import (
    cumulative_returns,
    log_returns,
    simple_returns,
    wealth_index,
)


# ---------------------------------------------------------------------------
# Simple Returns
# ---------------------------------------------------------------------------


def test_simple_returns_calculates_expected_values():
    prices = pd.Series(
        [100.0, 105.0, 102.0]
    )

    result = simple_returns(
        prices
    )

    expected = pd.Series(
        [
            0.05,
            (102.0 / 105.0) - 1.0,
        ],
        index=[1, 2],
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_simple_returns_supports_multi_period_changes():
    prices = pd.Series(
        [100.0, 105.0, 110.0, 121.0]
    )

    result = simple_returns(
        prices,
        periods=2,
    )

    expected = pd.Series(
        [
            0.10,
            (121.0 / 105.0) - 1.0,
        ],
        index=[2, 3],
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_simple_returns_preserves_series_name():
    prices = pd.Series(
        [100.0, 105.0, 110.0],
        name="AAPL",
    )

    result = simple_returns(
        prices
    )

    assert result.name == "AAPL"


def test_simple_returns_does_not_bridge_missing_prices():
    prices = pd.Series(
        [
            100.0,
            np.nan,
            110.0,
            121.0,
        ]
    )

    result = simple_returns(
        prices,
        dropna=False,
    )

    assert np.isnan(
        result.iloc[1]
    )

    assert np.isnan(
        result.iloc[2]
    )

    assert result.iloc[3] == pytest.approx(
        0.10
    )


# ---------------------------------------------------------------------------
# Log Returns
# ---------------------------------------------------------------------------


def test_log_returns_calculates_expected_values():
    prices = pd.Series(
        [100.0, 110.0, 121.0]
    )

    result = log_returns(
        prices
    )

    expected_value = np.log(
        1.10
    )

    assert result.iloc[0] == pytest.approx(
        expected_value
    )

    assert result.iloc[1] == pytest.approx(
        expected_value
    )


def test_log_returns_are_additive_across_periods():
    prices = pd.Series(
        [100.0, 110.0, 121.0]
    )

    one_period = log_returns(
        prices
    )

    two_period = log_returns(
        prices,
        periods=2,
    )

    assert two_period.iloc[0] == pytest.approx(
        one_period.sum()
    )


# ---------------------------------------------------------------------------
# Cumulative Returns
# ---------------------------------------------------------------------------


def test_cumulative_returns_compounds_periodic_returns():
    returns = pd.Series(
        [0.10, 0.10]
    )

    result = cumulative_returns(
        returns
    )

    expected = pd.Series(
        [0.10, 0.21]
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_cumulative_returns_supports_total_loss():
    returns = pd.Series(
        [0.10, -1.00, 0.50]
    )

    result = cumulative_returns(
        returns
    )

    assert result.iloc[1] == pytest.approx(
        -1.0
    )

    assert result.iloc[2] == pytest.approx(
        -1.0
    )


# ---------------------------------------------------------------------------
# Wealth Index
# ---------------------------------------------------------------------------


def test_wealth_index_compounds_from_initial_value():
    returns = pd.Series(
        [0.10, -0.05]
    )

    result = wealth_index(
        returns,
        initial_value=100.0,
    )

    expected = pd.Series(
        [110.0, 104.5]
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_wealth_index_preserves_series_index():
    index = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    returns = pd.Series(
        [0.01, -0.02, 0.03],
        index=index,
    )

    result = wealth_index(
        returns
    )

    assert result.index.equals(
        index
    )


# ---------------------------------------------------------------------------
# Missing Data
# ---------------------------------------------------------------------------


def test_cumulative_returns_can_preserve_missing_observations():
    returns = pd.Series(
        [
            0.10,
            np.nan,
            0.05,
        ]
    )

    result = cumulative_returns(
        returns,
        dropna=False,
    )

    assert result.iloc[0] == pytest.approx(
        0.10
    )

    assert np.isnan(
        result.iloc[1]
    )

    assert result.iloc[2] == pytest.approx(
        0.155
    )


# ---------------------------------------------------------------------------
# Price Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, 0.0, 110.0],
        [100.0, -50.0, 110.0],
    ],
)
def test_price_return_functions_reject_non_positive_prices(
    prices,
):
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        simple_returns(
            prices
        )


@pytest.mark.parametrize(
    "function",
    [
        simple_returns,
        log_returns,
    ],
)
def test_price_return_functions_reject_infinite_prices(
    function,
):
    with pytest.raises(
        ValueError,
        match="infinite",
    ):
        function(
            [
                100.0,
                np.inf,
                110.0,
            ]
        )


@pytest.mark.parametrize(
    "function",
    [
        simple_returns,
        log_returns,
    ],
)
def test_price_return_functions_reject_multidimensional_input(
    function,
):
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        function(
            [
                [100.0, 101.0],
                [102.0, 103.0],
            ]
        )


@pytest.mark.parametrize(
    "function",
    [
        simple_returns,
        log_returns,
    ],
)
def test_price_return_functions_require_calculable_return(
    function,
):
    with pytest.raises(
        ValueError,
        match="No valid",
    ):
        function(
            [100.0]
        )


# ---------------------------------------------------------------------------
# Return Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        cumulative_returns,
        wealth_index,
    ],
)
def test_compounding_functions_reject_returns_below_negative_one(
    function,
):
    with pytest.raises(
        ValueError,
        match="-100%",
    ):
        function(
            [
                0.05,
                -1.01,
                0.02,
            ]
        )


@pytest.mark.parametrize(
    "function",
    [
        cumulative_returns,
        wealth_index,
    ],
)
def test_compounding_functions_reject_infinite_returns(
    function,
):
    with pytest.raises(
        ValueError,
        match="infinite",
    ):
        function(
            [
                0.01,
                np.inf,
                0.02,
            ]
        )


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "periods",
    [
        0,
        -1,
        1.5,
        np.nan,
        np.inf,
    ],
)
def test_return_period_validation_rejects_invalid_values(
    periods,
):
    with pytest.raises(
        ValueError,
        match="periods",
    ):
        simple_returns(
            [
                100.0,
                105.0,
            ],
            periods=periods,
        )


@pytest.mark.parametrize(
    "initial_value",
    [
        0,
        -100,
        np.nan,
        np.inf,
    ],
)
def test_wealth_index_rejects_invalid_initial_value(
    initial_value,
):
    with pytest.raises(
        ValueError,
        match="initial_value",
    ):
        wealth_index(
            [
                0.01,
                0.02,
            ],
            initial_value=initial_value,
        )
