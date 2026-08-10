"""QuantCore: reusable quantitative finance and analytics tools."""
"""
QuantCore.

Reusable quantitative finance tools for return analysis, risk modeling,
portfolio analytics, statistical diagnostics, and Monte Carlo simulation.

QuantCore separates deterministic quantitative computation from
application, reporting, and AI interpretation layers.
"""

from importlib.metadata import PackageNotFoundError, version

from quantcore.analysis import (
    RiskAnalysisResult,
    analyze_risk,
)
from quantcore.diagnostics import (
    DistributionDiagnostics,
    distribution_summary,
    excess_kurtosis,
    mean_return,
    normality_test,
    skewness,
)
from quantcore.model_comparison import (
    RiskModelResult,
    available_models,
    run_tail_risk_analysis,
)
from quantcore.portfolio import (
    correlation_matrix,
    covariance_matrix,
    portfolio_expected_return,
    portfolio_returns,
    portfolio_volatility,
    sharpe_ratio,
)
from quantcore.returns import (
    cumulative_returns,
    log_returns,
    simple_returns,
    wealth_index,
)
from quantcore.risk import (
    max_drawdown,
    volatility,
)
from quantcore.simulation import (
    MonteCarloResult,
    monte_carlo_simulation,
)
from quantcore.tail_risk import (
    gaussian_expected_shortfall,
    gaussian_var,
    historical_expected_shortfall,
    historical_var,
    student_t_expected_shortfall,
    student_t_var,
)


try:
    __version__ = version("quantcore")
except PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = [
    # High-level analysis
    "RiskAnalysisResult",
    "analyze_risk",

    # Return transformations
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "wealth_index",

    # Core risk metrics
    "max_drawdown",
    "volatility",

    # Distribution diagnostics
    "DistributionDiagnostics",
    "mean_return",
    "skewness",
    "excess_kurtosis",
    "normality_test",
    "distribution_summary",

    # Tail-risk models
    "historical_var",
    "historical_expected_shortfall",
    "gaussian_var",
    "gaussian_expected_shortfall",
    "student_t_var",
    "student_t_expected_shortfall",

    # Risk model comparison
    "RiskModelResult",
    "available_models",
    "run_tail_risk_analysis",

    # Monte Carlo simulation
    "MonteCarloResult",
    "monte_carlo_simulation",

    # Portfolio analytics
    "correlation_matrix",
    "covariance_matrix",
    "portfolio_returns",
    "portfolio_expected_return",
    "portfolio_volatility",
    "sharpe_ratio",

    # Package metadata
    "__version__",
]
