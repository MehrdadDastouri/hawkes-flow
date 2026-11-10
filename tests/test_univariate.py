import numpy as np
import pytest
from hawkes_flow.models import HawkesParams
from hawkes_flow.univariate import UnivariateHawkes


def test_hawkes_params_validation_and_branching():
    # Valid stationary parameters
    params = HawkesParams(mu=1.2, alpha=0.6, beta=1.0)
    assert params.branching_ratio == 0.6
    assert params.is_stationary is True

    # Critical / Super-critical non-stationary parameters
    params_crit = HawkesParams(mu=1.0, alpha=2.0, beta=1.5)
    assert params_crit.branching_ratio > 1.0
    assert params_crit.is_stationary is False

    # Illegal parameters
    with pytest.raises(ValueError):
        HawkesParams(mu=-0.5, alpha=1.0, beta=1.0)
    with pytest.raises(ValueError):
        HawkesParams(mu=1.0, alpha=-0.1, beta=1.0)
    with pytest.raises(ValueError):
        HawkesParams(mu=1.0, alpha=1.0, beta=0.0)


def test_recursive_vs_naive_equivalence():
    params = HawkesParams(mu=0.8, alpha=1.5, beta=2.2)
    engine = UnivariateHawkes(params)

    # Synthetic cluster timestamps
    timestamps = np.array([0.1, 0.25, 0.28, 0.31, 0.85, 1.20, 1.22, 1.23, 2.50], dtype=np.float64)

    naive_intensities = engine.compute_intensities_naive(timestamps)
    recursive_intensities = engine.compute_intensities_recursive(timestamps)

    # Must match down to machine epsilon precision
    np.testing.assert_allclose(recursive_intensities, naive_intensities, rtol=1e-12, atol=1e-12)


def test_log_likelihood_monotonicity_under_baseline():
    timestamps = np.array([0.2, 0.4, 0.45, 0.5, 0.9], dtype=np.float64)
    max_time = 1.0

    # Model A: Close baseline
    engine_a = UnivariateHawkes(HawkesParams(mu=2.0, alpha=0.5, beta=1.0))
    # Model B: Absurdly high baseline (massively penalizes compensator)
    engine_b = UnivariateHawkes(HawkesParams(mu=50.0, alpha=0.5, beta=1.0))

    ll_a = engine_a.log_likelihood(timestamps, max_time)
    ll_b = engine_b.log_likelihood(timestamps, max_time)

    # Likelihood of reasonable baseline must dominate degenerate baseline
    assert ll_a > ll_b
    assert np.isfinite(ll_a)
