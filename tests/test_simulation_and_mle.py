import numpy as np
from hawkes_flow.diagnostics import HawkesDiagnostics
from hawkes_flow.estimation import HawkesEstimator
from hawkes_flow.models import HawkesParams
from hawkes_flow.simulation import HawkesSimulator


def test_ogata_simulation_stationarity_and_rates():
    # Ground truth parameters: branching ratio = 0.5 (safe sub-critical)
    params = HawkesParams(mu=1.0, alpha=1.0, beta=2.0)
    simulator = HawkesSimulator(params, seed=123)

    horizon = 500.0
    events = simulator.simulate(max_time=horizon)

    assert len(events) > 0
    assert np.all(np.diff(events) > 0)  # Strictly ascending timestamps
    assert events[-1] <= horizon

    # Theoretical mean intensity: E[lambda] = mu / (1 - alpha/beta) = 1.0 / 0.5 = 2.0
    emp_rate = len(events) / horizon
    np.testing.assert_allclose(emp_rate, 2.0, rtol=0.20)


def test_mle_parameter_recovery():
    true_params = HawkesParams(mu=1.2, alpha=0.8, beta=2.0)
    simulator = HawkesSimulator(true_params, seed=777)

    # Generate large trace for statistical convergence
    horizon = 1000.0
    events = simulator.simulate(max_time=horizon)

    estimator = HawkesEstimator(max_iter=300)
    fitted = estimator.fit(events, max_time=horizon)

    # Recovered parameters must lie within reasonable statistical deviation
    assert np.isclose(fitted.mu, true_params.mu, rtol=0.30)
    assert np.isclose(fitted.alpha, true_params.alpha, rtol=0.35)
    assert np.isclose(fitted.beta, true_params.beta, rtol=0.35)
    assert fitted.is_stationary is True


def test_random_time_change_goodness_of_fit():
    params = HawkesParams(mu=1.5, alpha=1.2, beta=2.5)
    simulator = HawkesSimulator(params, seed=999)
    events = simulator.simulate(max_time=600.0)

    # Test diagnostics against true generating model
    report = HawkesDiagnostics.evaluate_fit(events, params, alpha_level=0.01)

    assert len(report.residuals) == len(events) - 1
    # Under the true model, p-value should not reject the null hypothesis of Exp(1)
    assert report.is_valid_fit is True
    assert report.ks_statistic < 0.10
