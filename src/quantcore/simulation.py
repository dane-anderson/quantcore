"""
Monte Carlo simulation tools for QuantCore.

This module provides reproducible forward-looking simulation of asset
or portfolio values using historical return behavior.

Current model
-------------
Gaussian log-return Monte Carlo

Historical simple returns are converted to log returns. Their sample
mean and volatility parameterize simulated forward return paths.

Conventions
-----------
- Simulated values remain strictly positive.
- Terminal returns are signed decimal returns.
- Maximum drawdowns are signed negative values.
- A deterministic random seed may be supplied for reproducibility.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


ReturnInput = Sequence[float] | np.ndarray


__all__ = [
    "MonteCarloResult",
    "monte_carlo_simulation",
]


@dataclass(frozen=True)
class MonteCarloResult:
    """
    Structured output from a Monte Carlo simulation.

    Attributes
    ----------
    final_values : np.ndarray
        Final simulated value for each path.

    terminal_returns : np.ndarray
        Total return from initial value to final value for each path.

    max_drawdowns : np.ndarray
        Maximum signed drawdown for each simulated path.

    initial_value : float
        Starting value used for every simulation.

    days : int
        Number of simulated forward periods.

    simulations : int
        Number of independent simulated paths.

    seed : int
        Random seed used by the simulation.

    model : str
        Machine-readable simulation model identifier.
    """

    final_values: np.ndarray
    terminal_returns: np.ndarray
    max_drawdowns: np.ndarray
    initial_value: float
    days: int
    simulations: int
    seed: int
    model: str

    def summary(self) -> dict[str, float | int | str]:
        """
        Return deterministic summary statistics.

        Returns
        -------
        dict
            Summary of terminal-value, return, probability, and
            drawdown distributions.
        """
        if self.simulations > 1:
            terminal_return_volatility = float(
                np.std(
                    self.terminal_returns,
                    ddof=1,
                )
            )
        else:
            terminal_return_volatility = 0.0

        return {
            "model": self.model,
            "initial_value": self.initial_value,
            "days": self.days,
            "simulations": self.simulations,
            "seed": self.seed,
            "average_final_value": float(
                np.mean(self.final_values)
            ),
            "median_final_value": float(
                np.median(self.final_values)
            ),
            "percentile_5_final_value": float(
                np.percentile(
                    self.final_values,
                    5,
                )
            ),
            "percentile_95_final_value": float(
                np.percentile(
                    self.final_values,
                    95,
                )
            ),
            "average_terminal_return": float(
                np.mean(self.terminal_returns)
            ),
            "median_terminal_return": float(
                np.median(self.terminal_returns)
            ),
            "terminal_return_volatility": (
                terminal_return_volatility
            ),
            "probability_profit": float(
                np.mean(
                    self.terminal_returns > 0
                )
            ),
            "probability_loss": float(
                np.mean(
                    self.terminal_returns < 0
                )
            ),
            "average_max_drawdown": float(
                np.mean(self.max_drawdowns)
            ),
            "median_max_drawdown": float(
                np.median(self.max_drawdowns)
            ),
            "percentile_5_max_drawdown": float(
                np.percentile(
                    self.max_drawdowns,
                    5,
                )
            ),
        }


def _clean_returns(
    returns: ReturnInput,
) -> np.ndarray:
    """
    Normalize and validate historical return observations.

    Parameters
    ----------
    returns : ReturnInput
        Historical simple periodic returns.

    Returns
    -------
    np.ndarray
        One-dimensional array containing valid finite returns.

    Raises
    ------
    ValueError
        If fewer than two valid observations remain or any return
        is less than or equal to -100%.
    """
    values = np.asarray(
        returns,
        dtype="float64",
    )

    if values.ndim != 1:
        raise ValueError(
            "Return data must be one-dimensional."
        )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        raise ValueError(
            "Monte Carlo simulation requires at least "
            "two valid return observations."
        )

    if np.any(values <= -1.0):
        raise ValueError(
            "Historical returns cannot contain values "
            "less than or equal to -100%."
        )

    return values


def _validate_positive_integer(
    value: int,
    name: str,
) -> int:
    """
    Validate a positive integer parameter.

    Parameters
    ----------
    value : int
        Value to validate.

    name : str
        Parameter name used in validation errors.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    ValueError
        If the supplied value is not a positive whole number.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a positive integer."
        ) from None

    if (
        not np.isfinite(numeric_value)
        or numeric_value <= 0
        or not numeric_value.is_integer()
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(numeric_value)


def _validate_seed(
    seed: int,
) -> int:
    """
    Validate the random seed.

    Returns
    -------
    int
        Validated random seed.
    """
    try:
        numeric_seed = float(seed)
    except (TypeError, ValueError):
        raise ValueError(
            "seed must be an integer."
        ) from None

    if (
        not np.isfinite(numeric_seed)
        or not numeric_seed.is_integer()
    ):
        raise ValueError(
            "seed must be an integer."
        )

    return int(numeric_seed)


def monte_carlo_simulation(
    returns: ReturnInput,
    initial_value: float = 10_000.0,
    days: int = 252,
    simulations: int = 10_000,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run a Gaussian log-return Monte Carlo simulation.

    Historical simple returns are converted to logarithmic returns.
    Their historical mean and sample standard deviation parameterize
    forward simulated log returns.

    Parameters
    ----------
    returns : ReturnInput
        Historical simple periodic returns.

    initial_value : float, default=10000.0
        Starting asset or portfolio value.

    days : int, default=252
        Number of forward periods to simulate.

    simulations : int, default=10000
        Number of independent simulated paths.

    seed : int, default=42
        Random seed used for reproducible simulations.

    Returns
    -------
    MonteCarloResult
        Structured simulation results.

    Raises
    ------
    ValueError
        If historical returns or simulation parameters are invalid.
    """
    historical_returns = _clean_returns(
        returns
    )

    try:
        initial_value = float(
            initial_value
        )
    except (TypeError, ValueError):
        raise ValueError(
            "initial_value must be a positive finite number."
        ) from None

    if (
        not np.isfinite(initial_value)
        or initial_value <= 0
    ):
        raise ValueError(
            "initial_value must be a positive finite number."
        )

    days = _validate_positive_integer(
        days,
        "days",
    )

    simulations = _validate_positive_integer(
        simulations,
        "simulations",
    )

    seed = _validate_seed(seed)

    historical_log_returns = np.log1p(
        historical_returns
    )

    mean_log_return = float(
        np.mean(
            historical_log_returns
        )
    )

    log_return_volatility = float(
        np.std(
            historical_log_returns,
            ddof=1,
        )
    )

    if not np.isfinite(
        log_return_volatility
    ):
        raise ValueError(
            "Historical return volatility could not be estimated."
        )

    random_generator = np.random.default_rng(
        seed
    )

    simulated_log_returns = (
        random_generator.normal(
            loc=mean_log_return,
            scale=log_return_volatility,
            size=(
                days,
                simulations,
            ),
        )
    )

    cumulative_log_returns = np.cumsum(
        simulated_log_returns,
        axis=0,
    )

    simulated_values = (
        initial_value
        * np.exp(
            cumulative_log_returns
        )
    )

    initial_row = np.full(
        (
            1,
            simulations,
        ),
        initial_value,
        dtype="float64",
    )

    paths = np.vstack(
        (
            initial_row,
            simulated_values,
        )
    )

    final_values = paths[-1].copy()

    terminal_returns = (
        final_values
        / initial_value
        - 1.0
    )

    running_peaks = np.maximum.accumulate(
        paths,
        axis=0,
    )

    drawdowns = (
        paths
        / running_peaks
        - 1.0
    )

    max_drawdowns = np.min(
        drawdowns,
        axis=0,
    )

    return MonteCarloResult(
        final_values=final_values,
        terminal_returns=terminal_returns,
        max_drawdowns=max_drawdowns,
        initial_value=initial_value,
        days=days,
        simulations=simulations,
        seed=seed,
        model="gaussian_log_return",
    )
