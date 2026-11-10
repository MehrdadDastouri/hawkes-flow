# hawkes-flow

[![CI](https://github.com/MehrdadDastouri/hawkes-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/MehrdadDastouri/hawkes-flow/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-throughput point process library for modeling order arrival self-excitation, endogenous reflexivity, and liquidity clustering in financial microstructure.

---

## Key Features

- **O(N) Vectorized Intensity Engine:** Evaluates conditional intensity lambda(t_k) and analytical log-likelihood compensators without O(N^2) double loops.
- **Exact Event Simulation:** Implements Ogata's modified thinning algorithm with recursive state caching for high-speed trajectory generation.
- **Constrained MLE Calibration:** Fits baseline arrival (mu), shock amplitude (alpha), and memory decay (beta) with strict sub-critical stationarity boundaries (alpha < beta).
- **Goodness-of-Fit Diagnostics:** Implements the Random Time Change Theorem, rescaled residual extraction, and Kolmogorov-Smirnov uniformity tests.

---

## Quickstart

```python
from hawkes_flow.models import HawkesParams
from hawkes_flow.simulation import HawkesSimulator
from hawkes_flow.estimation import HawkesEstimator
from hawkes_flow.diagnostics import HawkesDiagnostics

# 1. Simulate endogenous order flow
ground_truth = HawkesParams(mu=1.2, alpha=0.8, beta=2.0)
sim = HawkesSimulator(ground_truth, seed=42)
events = sim.simulate(max_time=500.0)

# 2. Calibrate parameters via Maximum Likelihood
estimator = HawkesEstimator()
fitted_params = estimator.fit(events, max_time=500.0)
print(f"Reflexivity Index (Branching Ratio): {fitted_params.branching_ratio:.3f}")

# 3. Verify statistical goodness-of-fit
report = HawkesDiagnostics.evaluate_fit(events, fitted_params)
print(f"KS-Test p-value: {report.p_value:.4f} | Fit Valid: {report.is_valid_fit}")
```

## Testing

```bash
pip install -e .[dev]
pytest -v
```

## License

MIT License. Designed and engineered by Seyed Mehrdad Dastouri.
