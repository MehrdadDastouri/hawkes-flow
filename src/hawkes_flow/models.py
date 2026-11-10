"""Data containers for Hawkes process parameters and event traces."""

from dataclasses import dataclass


@dataclass(slots=True)
class HawkesParams:
    """Parameters defining a univariate Hawkes process with exponential kernel."""
    mu: float     # Exogenous baseline intensity (> 0)
    alpha: float  # Shock jump size (>= 0)
    beta: float   # Exponential memory decay rate (> 0)

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"Baseline intensity mu must be > 0, got {self.mu}")
        if self.alpha < 0:
            raise ValueError(f"Excitation amplitude alpha must be >= 0, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"Decay rate beta must be > 0, got {self.beta}")

    @property
    def branching_ratio(self) -> float:
        """Reflexivity index / average number of daughter events per arrival."""
        return self.alpha / self.beta

    @property
    def is_stationary(self) -> bool:
        """Stability condition check: eta < 1 prevents explosion."""
        return self.branching_ratio < 1.0
