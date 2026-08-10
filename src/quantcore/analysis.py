"""
High-level quantitative analysis workflows for QuantCore.

This module coordinates lower-level QuantCore components into
structured analytical results.

The analysis layer does not replace the underlying statistical or
risk models. It provides a convenient deterministic interface for
running related analyses together.

Current workflows
-----------------
- Return-distribution diagnostics
- Tail-risk model comparison

Interpretation, narrative reporting, and investment recommendations
belong outside the QuantCore computation layer.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from quantcore._validation import (
    ReturnInput,
    clean_returns,
    validate_confidence,
    validate_positive_integer,
    validate_significance_level,
)
from quantcore.diagnostics import (
    DistributionDiagnostics,
    distribution_summary,
)
from quantcore.model_comparison import (
    RiskModelResult,
    run_tail_risk_analysis,
)


__all__ = [
    "RiskAnalysisResult",
    "analyze_risk",
]


@dataclass(frozen=True)
class RiskAnalysisResult:
    """
    Structured result from a complete QuantCore risk analysis.

    Attributes
    ----------
    diagnostics : DistributionDiagnostics
        Statistical diagnostics describing the return distribution.

    models : tuple[RiskModelResult, ...]
        Tail-risk model results.

    confidence : float
        Confidence level used by the tail-risk models.

    periods_per_year : int
        Number of periods used for annualization.

    significance_level : float
        Significance threshold used for normality testing.
    """

    diagnostics: DistributionDiagnostics
    models: tuple[RiskModelResult, ...]
    confidence: float
    periods_per_year: int
    significance_level: float

    @property
    def observations(self) -> int:
        """
        Return the number of valid observations analyzed.
        """
        return self.diagnostics.observations

    @property
    def model_count(self) -> int:
        """
        Return the number of tail-risk models executed.
        """
        return len(
            self.models
        )

    @property
    def model_keys(self) -> tuple[str, ...]:
        """
        Return executed model identifiers in analysis order.
        """
        return tuple(
            result.model_key
            for result in self.models
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """
        Convert the structured analysis result to a dictionary.

        This is useful for serialization, APIs, reporting systems,
        notebooks, and downstream application layers.

        Returns
        -------
        dict[str, object]
            Nested representation of the analysis result.
        """
        return asdict(
            self
        )


def analyze_risk(
    returns: ReturnInput,
    confidence: float = 0.95,
    models: Sequence[str] | None = None,
    periods_per_year: int = 252,
    significance_level: float = 0.05,
) -> RiskAnalysisResult:
    """
    Run a complete deterministic return-risk analysis.

    The workflow performs:

    1. Return-data cleaning and validation.
    2. Distribution diagnostics.
    3. Tail-risk model comparison.
    4. Structured result assembly.

    Parameters
    ----------
    returns : ReturnInput
        Historical periodic returns.

    confidence : float, default=0.95
        Confidence level used by tail-risk models.

    models : Sequence[str] or None, default=None
        Tail-risk models to execute.

        Available model identifiers currently include:

        - ``"historical"``
        - ``"gaussian"``
        - ``"student_t"``

        If omitted, all registered models are executed.

    periods_per_year : int, default=252
        Number of return periods in one year.

    significance_level : float, default=0.05
        Significance threshold used by statistical normality testing.

    Returns
    -------
    RiskAnalysisResult
        Complete structured quantitative analysis.

    Raises
    ------
    ValueError
        If return data, model selection, or analysis parameters are
        invalid.

    Notes
    -----
    This function coordinates existing QuantCore models. It does not
    perform language-model interpretation or generate investment
    recommendations.
    """
    cleaned = clean_returns(
        returns
    )

    confidence = validate_confidence(
        confidence
    )

    periods_per_year = (
        validate_positive_integer(
            periods_per_year,
            "periods_per_year",
        )
    )

    significance_level = (
        validate_significance_level(
            significance_level
        )
    )

    diagnostics = distribution_summary(
        cleaned,
        periods_per_year=periods_per_year,
        significance_level=significance_level,
    )

    model_results = (
        run_tail_risk_analysis(
            cleaned,
            confidence=confidence,
            models=models,
        )
    )

    return RiskAnalysisResult(
        diagnostics=diagnostics,
        models=tuple(
            model_results
        ),
        confidence=confidence,
        periods_per_year=periods_per_year,
        significance_level=significance_level,
    )
