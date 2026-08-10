"""Unit tests for QuantCore Monte Carlo simulation."""

import numpy as np
import pytest

from quantcore.simulation import (
    MonteCarloResult,
    monte_carlo_simulation,
)


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Provide deterministic historical returns for simulation tests."""
    return np.array(
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
# Simulation Output
# ---------------------------------------------------------------------------


def test_simulation_returns_structured_result(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=20,
        simulations=100,
    )

    assert isinstance(
        result,
        MonteCarloResult,
    )


def test_simulation_returns_expected_number_of_paths(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=20,
        simulations=250,
    )

    assert result.final_values.shape == (
        250,
    )

    assert result.terminal_returns.shape == (
        250,
    )

    assert result.max_drawdowns.shape == (
        250,
    )


def test_simulated_final_values_remain_positive(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=50,
        simulations=500,
    )

    assert np.all(
        result.final_values > 0
    )


def test_terminal_returns_match_final_values(
    sample_returns,
):
    initial_value = 25_000.0

    result = monte_carlo_simulation(
        sample_returns,
        initial_value=initial_value,
        days=25,
        simulations=100,
    )

    expected = (
        result.final_values
        / initial_value
        - 1.0
    )

    assert np.allclose(
        result.terminal_returns,
        expected,
    )


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------


def test_max_drawdowns_are_non_positive(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=50,
        simulations=500,
    )

    assert np.all(
        result.max_drawdowns <= 0
    )


def test_max_drawdowns_cannot_exceed_total_loss(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=50,
        simulations=500,
    )

    assert np.all(
        result.max_drawdowns >= -1.0
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_simulation(
    sample_returns,
):
    first = monte_carlo_simulation(
        sample_returns,
        days=30,
        simulations=100,
        seed=123,
    )

    second = monte_carlo_simulation(
        sample_returns,
        days=30,
        simulations=100,
        seed=123,
    )

    assert np.array_equal(
        first.final_values,
        second.final_values,
    )

    assert np.array_equal(
        first.terminal_returns,
        second.terminal_returns,
    )

    assert np.array_equal(
        first.max_drawdowns,
        second.max_drawdowns,
    )


def test_different_seeds_produce_different_simulations(
    sample_returns,
):
    first = monte_carlo_simulation(
        sample_returns,
        days=30,
        simulations=100,
        seed=123,
    )

    second = monte_carlo_simulation(
        sample_returns,
        days=30,
        simulations=100,
        seed=456,
    )

    assert not np.array_equal(
        first.final_values,
        second.final_values,
    )


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------


def test_summary_contains_simulation_metadata(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        initial_value=15_000,
        days=40,
        simulations=200,
        seed=7,
    )

    summary = result.summary()

    assert summary["model"] == (
        "gaussian_log_return"
    )

    assert summary["initial_value"] == (
        pytest.approx(15_000)
    )

    assert summary["days"] == 40

    assert summary["simulations"] == 200

    assert summary["seed"] == 7


def test_summary_probabilities_are_valid(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=50,
        simulations=500,
    )

    summary = result.summary()

    assert (
        0.0
        <= summary["probability_profit"]
        <= 1.0
    )

    assert (
        0.0
        <= summary["probability_loss"]
        <= 1.0
    )


def test_summary_percentiles_are_ordered(
    sample_returns,
):
    result = monte_carlo_simulation(
        sample_returns,
        days=50,
        simulations=500,
    )

    summary = result.summary()

    assert (
        summary["percentile_5_final_value"]
        <= summary["median_final_value"]
        <= summary["percentile_95_final_value"]
    )


# ---------------------------------------------------------------------------
# Input Cleaning
# ---------------------------------------------------------------------------


def test_simulation_ignores_non_finite_returns(
    sample_returns,
):
    dirty_returns = np.concatenate(
        (
            sample_returns,
            [
                np.nan,
                np.inf,
                -np.inf,
            ],
        )
    )

    clean_result = monte_carlo_simulation(
        sample_returns,
        days=20,
        simulations=100,
        seed=42,
    )

    dirty_result = monte_carlo_simulation(
        dirty_returns,
        days=20,
        simulations=100,
        seed=42,
    )

    assert np.array_equal(
        clean_result.final_values,
        dirty_result.final_values,
    )


def test_simulation_requires_two_valid_returns():
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        monte_carlo_simulation(
            [0.01]
        )


def test_simulation_rejects_returns_at_or_below_negative_one():
    with pytest.raises(
        ValueError,
        match="-100%",
    ):
        monte_carlo_simulation(
            [
                0.01,
                -1.00,
                0.02,
            ]
        )


def test_simulation_rejects_multidimensional_returns():
    returns = np.array(
        [
            [0.01, 0.02],
            [-0.01, 0.03],
        ]
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        monte_carlo_simulation(
            returns
        )


# ---------------------------------------------------------------------------
# Parameter Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "initial_value",
    [
        0,
        -100,
        np.nan,
        np.inf,
    ],
)
def test_simulation_rejects_invalid_initial_value(
    sample_returns,
    initial_value,
):
    with pytest.raises(
        ValueError,
        match="initial_value",
    ):
        monte_carlo_simulation(
            sample_returns,
            initial_value=initial_value,
        )


@pytest.mark.parametrize(
    "days",
    [
        0,
        -1,
        2.5,
        np.nan,
    ],
)
def test_simulation_rejects_invalid_day_count(
    sample_returns,
    days,
):
    with pytest.raises(
        ValueError,
        match="days",
    ):
        monte_carlo_simulation(
            sample_returns,
            days=days,
        )


@pytest.mark.parametrize(
    "simulations",
    [
        0,
        -100,
        5.5,
        np.inf,
    ],
)
def test_simulation_rejects_invalid_simulation_count(
    sample_returns,
    simulations,
):
    with pytest.raises(
        ValueError,
        match="simulations",
    ):
        monte_carlo_simulation(
            sample_returns,
            simulations=simulations,
        )


@pytest.mark.parametrize(
    "seed",
    [
        1.5,
        np.nan,
        np.inf,
        "not-a-seed",
    ],
)
def test_simulation_rejects_invalid_seed(
    sample_returns,
    seed,
):
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        monte_carlo_simulation(
            sample_returns,
            seed=seed,
        )
