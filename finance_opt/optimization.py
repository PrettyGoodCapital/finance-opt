from __future__ import annotations

from typing import Any

import numpy as np

from . import finance_opt as _native
from .schemas import OptimizationProblem, OptimizationResult

__all__ = ["optimize_mean_variance"]


def _project_weights(
    weights: np.ndarray,
    *,
    min_weight: float | None,
    max_weight: float | None,
    long_only: bool,
    budget: float,
) -> np.ndarray:
    out = np.asarray(weights, dtype=float).copy()
    if long_only:
        out = np.maximum(out, 0.0)
    if min_weight is not None:
        out = np.maximum(out, min_weight)
    if max_weight is not None:
        out = np.minimum(out, max_weight)

    total = float(out.sum())
    if abs(total) < 1e-12:
        out[:] = budget / max(out.size, 1)
    else:
        out *= budget / total
    return out


def optimize_mean_variance(problem: OptimizationProblem | dict[str, Any]) -> OptimizationResult:
    """Solve a single-period mean-variance allocation problem.

    This is phase-8.5 starter scope: the API supports objective,
    constraints, and solver configuration while currently using a
    quadratic closed-form approximation under the hood.
    """
    request = problem if isinstance(problem, OptimizationProblem) else OptimizationProblem.model_validate(problem)

    mu = np.asarray(request.expected_returns, dtype=float)
    cov = np.asarray(request.covariance, dtype=float)
    if cov.shape != (mu.size, mu.size):
        raise ValueError("covariance dimensions must match expected_returns length")

    if hasattr(_native, "solve_mean_variance"):
        native_weights = np.asarray(
            _native.solve_mean_variance(
                mu.tolist(),
                cov.tolist(),
                float(request.objective.risk_aversion),
                int(request.config.max_iterations),
                float(request.config.tolerance),
            ),
            dtype=float,
        )
    else:
        # Fallback keeps the Python API usable before rebuilding the native module.
        inv = np.linalg.pinv(cov)
        native_weights = inv @ mu

    weights = _project_weights(
        native_weights,
        min_weight=request.constraints.min_weight,
        max_weight=request.constraints.max_weight,
        long_only=request.constraints.long_only,
        budget=float(request.constraints.budget),
    )

    objective = float(request.objective.risk_aversion) * float(weights @ cov @ weights) - float(mu @ weights)
    return OptimizationResult(
        weights=tuple(float(w) for w in weights),
        objective_value=objective,
        converged=True,
        iterations=1,
    )
