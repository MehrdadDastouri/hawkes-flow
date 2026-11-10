"""Goodness-of-fit verification using the Random Time Change Theorem."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from hawkes_flow.models import HawkesParams


@dataclass(slots=True)
class DiagnosticReport:
    """Statistical summary of goodness-of-fit diagnostics."""
    residuals: NDArray[np.float64]
    ks_statistic: float
    p_value: float
    is_valid_fit: bool


class HawkesDiagnostics:
    """Applies compensator transformations to test model adequacy against empirical flow."""

    @staticmethod
    def compute_rescaled_residuals(
        timestamps: NDArray[np.float64], params: HawkesParams
    ) -> NDArray[np.float64]:
        """
        Compute integrated intensity intervals tau_k = Lambda(t_k) - Lambda(t_{k-1}) in strict O(N).
        Under the true data generating process, tau_k are i.i.d. Exp(1) variates.
        """
        n = len(timestamps)
        if n < 2:
            return np.empty(0, dtype=np.float64)

        # Pre-allocate compensator values Lambda(t_k)
        lambda_vals = np.empty(n, dtype=np.float64)
        lambda_vals[0] = params.mu * timestamps[0]

        # Use recursive state R(k) = sum_{j < k} exp(-beta*(t_k - t_j))
        dt = np.diff(timestamps)
        decay_factors = np.exp(-params.beta * dt)

        r = 0.0
        for k in range(1, n):
            r = decay_factors[k - 1] * (1.0 + r)
            # Analytic compensator formula: Lambda(t_k) = mu*t_k + (alpha/beta) * (k - r)
            lambda_vals[k] = (params.mu * timestamps[k]) + (params.alpha / params.beta) * (k - r)

        residuals = np.diff(lambda_vals)
        return residuals

    @classmethod
    def evaluate_fit(
        cls, timestamps: NDArray[np.float64], params: HawkesParams, alpha_level: float = 0.05
    ) -> DiagnosticReport:
        """Run one-sample Kolmogorov-Smirnov test against standard Exponential(1) distribution."""
        residuals = cls.compute_rescaled_residuals(timestamps, params)
        if len(residuals) == 0:
            return DiagnosticReport(residuals=residuals, ks_statistic=1.0, p_value=0.0, is_valid_fit=False)

        # Test whether residuals follow Exp(scale=1.0)
        ks_res = stats.kstest(residuals, "expon", args=(0, 1.0))
        return DiagnosticReport(
            residuals=residuals,
            ks_statistic=float(ks_res.statistic),
            p_value=float(ks_res.pvalue),
            is_valid_fit=bool(ks_res.pvalue >= alpha_level),
        )
