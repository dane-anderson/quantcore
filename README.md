# QuantCore

**A reusable Python library for quantitative finance, portfolio analytics, statistical risk modeling, and Monte Carlo simulation.**

QuantCore provides deterministic, testable building blocks for analyzing financial return distributions, portfolio behavior, downside risk, and simulated future outcomes.

It is designed as a computation layer: **Python performs the quantitative analysis; applications, reporting systems, and AI models can consume the structured results.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-0.1.0-informational)

---

## Overview

QuantCore grew out of a larger quantitative research system and was separated into an independent library so its mathematical and statistical components could be reused across applications.

The library focuses on four principles:

- **Deterministic computation** — quantitative results come from explicit Python models rather than language-model reasoning.
- **Reusable architecture** — analytics can be imported independently or composed into higher-level workflows.
- **Structured outputs** — model results are returned as typed objects suitable for notebooks, APIs, applications, and research systems.
- **Testability** — each major analytical module has a dedicated test suite covering calculations, validation, edge cases, and integration behavior.

QuantCore is intentionally **market-data agnostic**. Applications may supply prices or returns from any provider.

---

## Features

### Return Analysis

Transform financial price and return series with:

- Simple returns
- Logarithmic returns
- Multi-period returns
- Cumulative compounded returns
- Wealth-index construction
- Index and label preservation
- Missing-data validation

### Core Risk Metrics

Calculate:

- Annualized volatility
- Maximum drawdown

### Distribution Diagnostics

Analyze the statistical behavior of return series using:

- Arithmetic mean return
- Sample volatility
- Annualized volatility
- Skewness
- Excess kurtosis
- Jarque-Bera normality testing
- Configurable significance levels

### Tail-Risk Models

QuantCore includes three downside-risk model families:

#### Historical Simulation

- Value at Risk (VaR)
- Expected Shortfall (ES)

#### Gaussian Parametric

- Value at Risk
- Expected Shortfall

#### Student-t Parametric

- Value at Risk
- Expected Shortfall

The Student-t model provides a heavier-tailed parametric alternative to the Gaussian assumption.

### Risk Model Comparison

Run multiple tail-risk models against the same return series using a standardized result interface.

Supported model identifiers:

```text
historical
gaussian
student_t
```

Models can be executed individually or compared within the same analysis.

### Monte Carlo Simulation

The simulation engine performs reproducible Gaussian log-return Monte Carlo analysis using historical return behavior.

It supports:

- Configurable forward horizons
- Configurable simulation counts
- Reproducible random seeds
- Historical log-return parameter estimation
- Thousands of independent simulated paths
- Terminal-value distributions
- Terminal-return distributions
- Probability of profit
- Probability of loss
- Path-level maximum drawdown analysis
- Summary statistics and percentiles

### Portfolio Analytics

Analyze multi-asset portfolios with:

- Correlation matrices
- Annualized covariance matrices
- Portfolio return series
- Annualized expected return
- Annualized portfolio volatility
- Sharpe ratio
- Labeled asset-weight alignment
- Short-position support
- Missing-observation handling

Pandas Series weights can be aligned directly by asset name, reducing the risk of accidental positional weight mismatches.

### High-Level Risk Analysis

For complete return-risk research, QuantCore provides a high-level workflow that combines:

```text
Return Series
     ↓
Distribution Diagnostics
     ↓
Tail-Risk Models
     ↓
Model Comparison
     ↓
Structured RiskAnalysisResult
```

This allows applications to run an entire deterministic risk analysis through a single interface.

---

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/dane-anderson/quantcore.git
```

### Development Installation

Clone the repository:

```bash
git clone https://github.com/dane-anderson/quantcore.git
cd quantcore
```

Install QuantCore with development dependencies:

```bash
pip install -e ".[dev]"
```

QuantCore requires **Python 3.11 or newer**.

---

## Quick Start

QuantCore exposes its main functionality directly through the package API.

```python
import pandas as pd
import quantcore as qc

prices = pd.Series(
    [100, 102, 101, 105, 103, 108, 106]
)

returns = qc.simple_returns(prices)

analysis = qc.analyze_risk(
    returns,
    confidence=0.95,
)

print(analysis.diagnostics)
print(analysis.models)
```

The high-level result contains structured access to:

```python
analysis.observations
analysis.model_count
analysis.model_keys
analysis.diagnostics
analysis.models
analysis.to_dict()
```

---

## Return Transformations

```python
import pandas as pd
import quantcore as qc

prices = pd.Series(
    [100, 105, 102, 110]
)

simple = qc.simple_returns(prices)
log = qc.log_returns(prices)

cumulative = qc.cumulative_returns(simple)

wealth = qc.wealth_index(
    simple,
    initial_value=10_000,
)
```

---

## Tail-Risk Analysis

Run all available tail-risk models:

```python
import quantcore as qc

results = qc.run_tail_risk_analysis(
    returns,
    confidence=0.99,
)

for result in results:
    print(
        result.model_name,
        result.value_at_risk,
        result.expected_shortfall,
    )
```

Or select specific models:

```python
results = qc.run_tail_risk_analysis(
    returns,
    confidence=0.99,
    models=[
        "historical",
        "student_t",
    ],
)
```

---

## Monte Carlo Simulation

```python
import quantcore as qc

simulation = qc.monte_carlo_simulation(
    returns,
    initial_value=100_000,
    days=252,
    simulations=10_000,
    seed=42,
)

summary = simulation.summary()

print(summary)
```

The simulation result provides direct access to:

```python
simulation.final_values
simulation.terminal_returns
simulation.max_drawdowns
simulation.initial_value
simulation.days
simulation.simulations
simulation.seed
simulation.model
```

Because the random seed is configurable, identical inputs and seeds produce reproducible simulations.

---

## Portfolio Analytics

Create a synchronized multi-asset return DataFrame:

```python
import pandas as pd
import quantcore as qc

returns = pd.DataFrame(
    {
        "AAPL": [0.010, -0.020, 0.015, 0.005],
        "MSFT": [0.008, -0.010, 0.012, 0.007],
        "NVDA": [0.020, -0.030, 0.025, 0.010],
    }
)

weights = pd.Series(
    {
        "NVDA": 0.20,
        "AAPL": 0.50,
        "MSFT": 0.30,
    }
)
```

QuantCore aligns labeled weights by asset name:

```python
portfolio_return = qc.portfolio_expected_return(
    returns,
    weights,
)

portfolio_risk = qc.portfolio_volatility(
    returns,
    weights,
)

sharpe = qc.sharpe_ratio(
    returns,
    weights,
    risk_free_rate=0.04,
)

correlation = qc.correlation_matrix(
    returns
)

covariance = qc.covariance_matrix(
    returns
)
```

Short positions are supported as long as portfolio weights net to `1.0`.

---

## High-Level Analysis

The high-level API coordinates independently tested QuantCore components.

```python
import quantcore as qc

analysis = qc.analyze_risk(
    returns,
    confidence=0.99,
    models=[
        "historical",
        "gaussian",
        "student_t",
    ],
    periods_per_year=252,
    significance_level=0.05,
)
```

A `RiskAnalysisResult` contains:

```text
RiskAnalysisResult
├── diagnostics
│   ├── observations
│   ├── mean_return
│   ├── volatility
│   ├── annualized_volatility
│   ├── skewness
│   ├── excess_kurtosis
│   ├── normality_statistic
│   ├── normality_pvalue
│   └── normality_rejected
│
├── models
│   ├── Historical Simulation
│   │   ├── Value at Risk
│   │   └── Expected Shortfall
│   │
│   ├── Gaussian
│   │   ├── Value at Risk
│   │   └── Expected Shortfall
│   │
│   └── Student-t
│       ├── Value at Risk
│       └── Expected Shortfall
│
├── confidence
├── periods_per_year
└── significance_level
```

The result can also be serialized:

```python
result = analysis.to_dict()
```

This makes QuantCore suitable as a numerical backend for APIs, dashboards, notebooks, research systems, and AI-assisted applications.

---

## Architecture

QuantCore separates data transformation, statistical analysis, quantitative models, and workflow orchestration.

```text
                    QuantCore

                    Price Data
                        │
                        ▼
                   returns.py
                        │
                        ▼
                  Return Series
                        │
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
      risk.py     diagnostics.py   simulation.py
          │             │              │
          │             ▼              │
          │        Distribution        │
          │        Diagnostics         │
          │             │              │
          ▼             ▼              ▼
     tail_risk.py  model_comparison   Monte Carlo
          │             │
          └──────┬──────┘
                 ▼
             analysis.py
                 │
                 ▼
        Structured Results


Multi-Asset Returns
        │
        ▼
   portfolio.py
        │
        ├── Correlation
        ├── Covariance
        ├── Expected Return
        ├── Volatility
        └── Sharpe Ratio
```

Shared internal validation is centralized rather than duplicated across analytical modules.

---

## Project Structure

```text
quantcore/
├── src/
│   └── quantcore/
│       ├── __init__.py
│       ├── _validation.py
│       ├── analysis.py
│       ├── diagnostics.py
│       ├── model_comparison.py
│       ├── portfolio.py
│       ├── returns.py
│       ├── risk.py
│       ├── simulation.py
│       └── tail_risk.py
│
├── tests/
│   ├── test_analysis.py
│   ├── test_diagnostics.py
│   ├── test_model_comparison.py
│   ├── test_portfolio.py
│   ├── test_returns.py
│   ├── test_risk.py
│   ├── test_simulation.py
│   ├── test_tail_risk.py
│   └── test_validation.py
│
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Testing

QuantCore uses `pytest`.

Run the full test suite with:

```bash
pytest
```

The tests cover:

- Numerical calculations
- Statistical model behavior
- Input validation
- Missing and non-finite observations
- Tail-risk model selection
- Portfolio weight alignment
- Short positions
- Annualization behavior
- Monte Carlo reproducibility
- Structured result serialization
- Cross-component integration

The high-level analysis tests verify that independently tested QuantCore components remain consistent when assembled into a complete workflow.

---

## Design Philosophy

QuantCore follows a simple separation of responsibilities:

```text
QuantCore calculates.
Applications orchestrate.
Reporting layers present.
AI systems interpret.
```

The library does not depend on a language model to calculate financial metrics.

This allows quantitative results to remain deterministic, reproducible, inspectable, and independently testable while still making QuantCore useful inside larger AI-assisted research systems.

---

## Public API

The primary API is available directly from `quantcore`:

```python
import quantcore as qc
```

Examples include:

```python
qc.simple_returns(...)
qc.log_returns(...)
qc.cumulative_returns(...)
qc.wealth_index(...)

qc.max_drawdown(...)
qc.volatility(...)

qc.distribution_summary(...)

qc.historical_var(...)
qc.historical_expected_shortfall(...)
qc.gaussian_var(...)
qc.gaussian_expected_shortfall(...)
qc.student_t_var(...)
qc.student_t_expected_shortfall(...)

qc.run_tail_risk_analysis(...)
qc.analyze_risk(...)

qc.monte_carlo_simulation(...)

qc.correlation_matrix(...)
qc.covariance_matrix(...)
qc.portfolio_returns(...)
qc.portfolio_expected_return(...)
qc.portfolio_volatility(...)
qc.sharpe_ratio(...)
```

Internal validation utilities are intentionally excluded from the public API.

---

## Current Version

**0.1.0**

QuantCore is under active development. The current release focuses on deterministic return analysis, statistical diagnostics, downside-risk modeling, portfolio analytics, and Monte Carlo simulation.

Potential future extensions include:

- Downside deviation and Sortino ratio
- Beta and benchmark-relative analytics
- Rolling risk metrics
- Stress testing
- Scenario analysis
- Portfolio Monte Carlo simulation
- Portfolio optimization
- Additional simulation models
- Backtesting utilities

---

## Technology

QuantCore is built with:

- Python
- NumPy
- pandas
- SciPy
- pytest

---

## License

QuantCore is released under the [MIT License](LICENSE).

---

## Author

**Dane Anderson**

GitHub: [dane-anderson](https://github.com/dane-anderson)
