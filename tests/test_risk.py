import pandas as pd

from quantcore.risk import max_drawdown


def test_max_drawdown():
    prices = pd.Series([100, 120, 110, 90, 105])

    result = max_drawdown(prices)

    assert result == -0.25
