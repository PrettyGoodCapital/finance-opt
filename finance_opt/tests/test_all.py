from __future__ import annotations

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
