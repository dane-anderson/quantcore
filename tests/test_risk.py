
"""Unit tests for QuantCore risk metrics."""

import pandas as pd
import pytest

from quantcore.risk import max_drawdown, volatility


def test_max_drawdown_returns_largest_peak_to_trough_decline():
    prices = pd.Series([100, 120, 110, 90, 105])

    result = max_drawdown(prices)

    assert result == pytest.approx(-0.25)


def test_volatility_matches_standard_deviation_without_annualization():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])

    result = volatility(returns, periods_per_year=1)

    expected = returns.std()

    assert result == pytest.approx(expected)
