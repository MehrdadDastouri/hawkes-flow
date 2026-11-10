"""Discrete-event simulation engine for Hawkes processes via Ogata's thinning algorithm."""

import numpy as np
from numpy.typing import NDArray
from hawkes_flow.models import HawkesParams


class HawkesSimulator:
    """Simulates exact event realization traces of univariate Hawkes processes."""

    def __init__(self, params: HawkesParams, seed: int = 42) -> None:
        self.params = params
        self._rng = np.random.default_rng(seed)

    def simulate(self, max_time: float) -> NDArray[np.float64]:
        """
        Generate exact event timestamps on [0, max_time] using Ogata's recursive thinning.
        Operates in O(N) amortized complexity without re-scanning historical events.
        """
        if max_time <= 0:
            return np.empty(0, dtype=np.float64)

        events = []
        current_time = 0.0
        current_intensity = self.params.mu

        while current_time < max_time:
            # Upper bound on intensity is the intensity right at current_time
            lambda_bar = current_intensity
            if lambda_bar <= 0:
                break

            # Draw candidate inter-arrival delta from homogeneous Poisson process
            w = self._rng.exponential(scale=1.0 / lambda_bar)
            candidate_time = current_time + w

            if candidate_time > max_time:
                break

            # Decay intensity to candidate arrival time
            dt = candidate_time - current_time
            decay = np.exp(-self.params.beta * dt)
            lambda_candidate = self.params.mu + (current_intensity - self.params.mu) * decay

            # Rejection test
            d = self._rng.uniform(0.0, 1.0)
            if d <= (lambda_candidate / lambda_bar):
                # Accept event
                events.append(candidate_time)
                current_time = candidate_time
                # Shock jump applied to intensity state
                current_intensity = lambda_candidate + self.params.alpha
            else:
                # Reject event, advance time without adding event
                current_time = candidate_time
                current_intensity = lambda_candidate

        return np.array(events, dtype=np.float64)
