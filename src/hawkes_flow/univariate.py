"""Univariate self-exciting Hawkes process intensity and likelihood engines."""

import numpy as np
from numpy.typing import NDArray
from hawkes_flow.models import HawkesParams


class UnivariateHawkes:
    """Core mathematical engine for univariate exponential Hawkes point processes."""

    def __init__(self, params: HawkesParams) -> None:
        self.params = params

    def compute_intensities_recursive(self, timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute conditional intensity lambda(t_k) across all event arrivals in strict O(N).
        Uses Ogata's recursive state formulation R(k) = exp(-beta * dt) * (1 + R(k-1)).
        """
        n = len(timestamps)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        intensities = np.empty(n, dtype=np.float64)
        intensities[0] = self.params.mu

        if n == 1:
            return intensities

        # Inter-arrival time deltas
        dt = np.diff(timestamps)
        decay_factors = np.exp(-self.params.beta * dt)

        r = 0.0
        for k in range(1, n):
            r = decay_factors[k - 1] * (1.0 + r)
            intensities[k] = self.params.mu + self.params.alpha * r

        return intensities

    def compute_intensities_naive(self, timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute intensities via direct double summation in O(N^2).
        Retained as an exact mathematical baseline for deterministic testing.
        """
        n = len(timestamps)
        intensities = np.empty(n, dtype=np.float64)
        for i in range(n):
            t_curr = timestamps[i]
            history = timestamps[:i]
            intensities[i] = self.params.mu + np.sum(self.params.alpha * np.exp(-self.params.beta * (t_curr - history)))
        return intensities

    def log_likelihood(self, timestamps: NDArray[np.float64], max_time: float) -> float:
        """
        Compute exact log-likelihood ln L on interval [0, max_time] in O(N) complexity.
        ln L = sum(ln lambda(t_k)) - Lambda(max_time)
        """
        n = len(timestamps)
        if n == 0:
            return -self.params.mu * max_time

        if max_time < timestamps[-1]:
            raise ValueError(f"Observation window max_time ({max_time}) must be >= last event ({timestamps[-1]})")

        # 1. Sum of log conditional intensities: sum(ln lambda(t_k))
        intensities = self.compute_intensities_recursive(timestamps)
        if np.any(intensities <= 0):
            return -np.inf
        log_intensity_sum = np.sum(np.log(intensities))

        # 2. Integrated intensity compensator: Lambda(T) = mu*T + (alpha/beta) * sum(1 - exp(-beta*(T - t_i)))
        compensator = (self.params.mu * max_time) + (
            (self.params.alpha / self.params.beta) * np.sum(1.0 - np.exp(-self.params.beta * (max_time - timestamps)))
        )

        return float(log_intensity_sum - compensator)
