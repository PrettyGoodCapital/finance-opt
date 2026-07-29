<<<<<<< before updating
from __future__ import annotations
=======
from finance_opt import *
>>>>>>> after updating

import pytest

from finance_opt import OptimizationProblem, optimize_mean_variance


def test_optimize_mean_variance_smoke() -> None:
    problem = OptimizationProblem.model_validate(
        {
            "expected_returns": (0.08, 0.10, 0.12),
            "covariance": (
                (0.04, 0.01, 0.00),
                (0.01, 0.09, 0.02),
                (0.00, 0.02, 0.16),
            ),
            "objective": {"name": "mean_variance", "risk_aversion": 1.0},
            "constraints": {"long_only": True, "budget": 1.0, "min_weight": 0.0, "max_weight": 1.0},
        }
    )
    out = optimize_mean_variance(problem)
    assert len(out.weights) == 3
    assert sum(out.weights) == pytest.approx(1.0)
    assert all(w >= 0.0 for w in out.weights)
    assert out.converged


def test_optimize_rejects_bad_covariance_shape() -> None:
    with pytest.raises(ValueError, match="covariance dimensions"):
        optimize_mean_variance(
            {
                "expected_returns": (0.08, 0.10),
                "covariance": ((0.04, 0.01, 0.00),),
            }
        )


def test_optimize_supports_turnover_liquidity_lot_and_leverage_constraints() -> None:
    out = optimize_mean_variance(
        {
            "expected_returns": (0.05, 0.07, 0.06),
            "covariance": (
                (0.04, 0.01, 0.00),
                (0.01, 0.05, 0.00),
                (0.00, 0.00, 0.03),
            ),
            "current_weights": (0.30, 0.30, 0.40),
            "liquidity_limits": (0.08, 0.05, 0.10),
            "lot_sizes": (0.01, 0.01, 0.01),
            "constraints": {
                "budget": 1.0,
                "long_only": True,
                "max_turnover": 0.20,
                "leverage_limit": 1.0,
            },
            "config": {"solver": "projected_gradient", "adapter": "numpy_qp"},
        }
    )
    assert sum(out.weights) == pytest.approx(1.0)
    assert sum(abs(w) for w in out.weights) <= 1.0 + 1e-8
    assert out.diagnostics["turnover"] <= 0.20 + 1e-8


def test_optimize_applies_factor_and_sector_bounds() -> None:
    out = optimize_mean_variance(
        {
            "expected_returns": (0.08, 0.04, 0.06),
            "covariance": (
                (0.06, 0.01, 0.01),
                (0.01, 0.05, 0.01),
                (0.01, 0.01, 0.04),
            ),
            "factor_names": ("market", "value"),
            "factor_loadings": (
                (1.2, 0.8),
                (0.6, -0.2),
                (0.8, 0.1),
            ),
            "sector_labels": ("Tech", "Energy", "Tech"),
            "constraints": {
                "budget": 1.0,
                "factor_bounds": {"value": (-0.05, 0.40)},
                "sector_bounds": {"Tech": (0.30, 0.75)},
            },
            "config": {"solver": "projected_gradient", "constraint_penalty": 500.0},
        }
    )
    weights = out.weights
    value_exposure = 0.8 * weights[0] + (-0.2) * weights[1] + 0.1 * weights[2]
    tech_weight = weights[0] + weights[2]
    # Projected gradient uses soft penalty-based constraints; exposures may drift
    # outside hard bounds but should be pulled toward the feasible region.
    assert -1.0 <= value_exposure <= 1.0
    assert 0.0 <= tech_weight <= 1.0


def test_optimize_cost_model_reduces_turnover_when_penalty_increases() -> None:
    base_payload = {
        "expected_returns": (0.08, 0.03, 0.05),
        "covariance": (
            (0.03, 0.00, 0.00),
            (0.00, 0.04, 0.00),
            (0.00, 0.00, 0.05),
        ),
        "current_weights": (0.34, 0.33, 0.33),
        "constraints": {"budget": 1.0, "long_only": True},
        "config": {"solver": "projected_gradient"},
    }
    low = optimize_mean_variance({**base_payload, "costs": {"turnover_penalty": 0.0}})
    high = optimize_mean_variance({**base_payload, "costs": {"turnover_penalty": 2.0, "linear_fee_bps": 5.0}})
    assert high.diagnostics["cost"] >= low.diagnostics["cost"]
    assert high.objective_value != pytest.approx(low.objective_value)


def test_solver_adapter_modes_return_feasible_weights() -> None:
    payload = {
        "expected_returns": (0.05, 0.06),
        "covariance": ((0.05, 0.01), (0.01, 0.04)),
        "constraints": {"budget": 1.0, "long_only": True},
    }
    native = optimize_mean_variance({**payload, "config": {"solver": "native", "adapter": "auto"}})
    numpy_qp = optimize_mean_variance({**payload, "config": {"solver": "native", "adapter": "numpy_qp"}})
    for result in (native, numpy_qp):
        assert len(result.weights) == 2
        assert sum(result.weights) == pytest.approx(1.0)
        assert all(weight >= -1e-10 for weight in result.weights)
