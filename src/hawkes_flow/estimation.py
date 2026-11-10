"""Maximum Likelihood Estimation (MLE) calibrator for Hawkes process parameters."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from hawkes_flow.models import HawkesParams
from hawkes_flow.univariate import UnivariateHawkes


class HawkesEstimator:
    """Calibrates baseline, jump excitation, and memory decay from observed timestamps."""

    def __init__(self, max_iter: int = 500) -> None:
        self.max_iter = max_iter

    def fit(self, timestamps: NDArray[np.float64], max_time: float) -> HawkesParams:
        """
        Fit Hawkes parameters by minimizing the negative log-likelihood function.
        Enforces strict positivity and sub-critical stationarity (alpha / beta < 1).
        """
        if len(timestamps) < 3:
            raise ValueError("At least 3 event timestamps are required for reliable calibration.")

        # Heuristic initial guesses based on empirical arrival rate
        emp_rate = len(timestamps) / max_time
        init_mu = max(0.1 * emp_rate, 1e-3)
        init_beta = 1.0
        init_alpha = 0.5 * init_beta
        initial_params = np.array([init_mu, init_alpha, init_beta], dtype=np.float64)

        # Optimization bounds: mu > 0, alpha >= 0, beta > 0
        bounds = [(1e-5, None), (0.0, None), (1e-5, None)]

        def objective(x: NDArray[np.float64]) -> float:
            mu, alpha, beta = x[0], x[1], x[2]
            # Soft penalty barrier for explosive non-stationary parameters
            if alpha >= beta:
                penalty = 1e6 * (alpha - beta + 1.0) ** 2
                return penalty

            try:
                params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
                engine = UnivariateHawkes(params)
                ll = engine.log_likelihood(timestamps, max_time)
                if not np.isfinite(ll):
                    return 1e10
                return -ll
            except (ValueError, OverflowError):
                return 1e10

        res = minimize(
            objective,
            x0=initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )

        fitted_mu, fitted_alpha, fitted_beta = res.x
        return HawkesParams(
            mu=float(fitted_mu),
            alpha=float(fitted_alpha),
            beta=float(fitted_beta),
        )
