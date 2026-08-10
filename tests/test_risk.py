"""Unit tests for QuantCore core risk metrics."""

import numpy as np
import pandas as pd
import pytest

from quantcore.risk import max_drawdown, volatility


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_returns_largest_peak_to_trough_decline():
    """Maximum drawdown should identify the worst peak-to-trough loss."""
    prices = pd.Series(
        [100, 120, 110, 90, 105],
        dtype="float64",
    )

    result = max_drawdown(prices)

    assert result == pytest.approx(-0.25)


def test_max_drawdown_returns_zero_when_prices_never_decline():
    """A continuously rising price series should have no drawdown."""
    prices = pd.Series(
        [100, 105, 110, 115, 120],
        dtype="float64",
    )

    result = max_drawdown(prices)

    assert result == pytest.approx(0.0)


def test_max_drawdown_tracks_new_highs_before_later_loss():
    """Drawdown should be measured from the most recent historical peak."""
    prices = pd.Series(
        [100, 110, 105, 130, 117],
        dtype="float64",
    )

    result = max_drawdown(prices)

    expected = (117 / 130) - 1

    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def test_volatility_matches_sample_standard_deviation_without_annualization():
    """One-period annualization should equal sample standard deviation."""
    returns = pd.Series(
        [0.01, -0.02, 0.015, -0.005],
        dtype="float64",
    )

    result = volatility(
        returns,
        periods_per_year=1,
    )

    expected = returns.std(ddof=1)

    assert result == pytest.approx(expected)


def test_volatility_annualizes_using_square_root_of_time():
    """Volatility should scale by the square root of periods per year."""
    returns = pd.Series(
        [0.01, -0.02, 0.015, -0.005],
        dtype="float64",
    )

    result = volatility(
        returns,
        periods_per_year=252,
    )

    expected = (
        returns.std(ddof=1)
        * np.sqrt(252)
    )

    assert result == pytest.approx(expected)


def test_volatility_is_zero_for_constant_returns():
    """A return series with no variation should have zero volatility."""
    returns = pd.Series(
        [0.01, 0.01, 0.01, 0.01],
        dtype="float64",
    )

    result = volatility(
        returns,
        periods_per_year=252,
    )

    assert result == pytest.approx(0.0)
