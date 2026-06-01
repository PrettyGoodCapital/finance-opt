from __future__ import annotations

from typing import Any

import numpy as np

from . import finance_opt as _native
from .schemas import OptimizationProblem, OptimizationResult

__all__ = ["optimize_mean_variance"]


def _default_current(size: int, budget: float) -> np.ndarray:
    if size == 0:
        return np.array([], dtype=float)
    return np.full(size, budget / size, dtype=float)


def _project_weights(
    weights: np.ndarray,
    *,
    min_weight: float | None,
    max_weight: float | None,
    long_only: bool,
    budget: float,
    current: np.ndarray | None = None,
    liquidity_limits: np.ndarray | None = None,
    lot_sizes: np.ndarray | None = None,
    max_turnover: float | None = None,
    leverage_limit: float | None = None,
) -> np.ndarray:
    out = np.asarray(weights, dtype=float).copy()
    if current is None:
        current = _default_current(out.size, budget)

    if long_only:
        out = np.maximum(out, 0.0)
    if min_weight is not None:
        out = np.maximum(out, min_weight)
    if max_weight is not None:
        out = np.minimum(out, max_weight)

    if liquidity_limits is not None:
        lower = current - liquidity_limits
        upper = current + liquidity_limits
        out = np.minimum(np.maximum(out, lower), upper)

    if lot_sizes is not None:
        safe_lot = np.where(lot_sizes <= 0.0, 1.0, lot_sizes)
        out = current + np.round((out - current) / safe_lot) * safe_lot

    if max_turnover is not None:
        trades = out - current
        turnover = float(np.abs(trades).sum())
        if turnover > max_turnover and turnover > 0.0:
            out = current + trades * (max_turnover / turnover)

    total = float(out.sum())
    if abs(total) < 1e-12:
        out[:] = budget / max(out.size, 1)
    else:
        out *= budget / total

    if leverage_limit is not None:
        gross = float(np.abs(out).sum())
        if gross > leverage_limit and gross > 0.0:
            out *= leverage_limit / gross
            # Re-target budget after gross scaling while preserving relative weights.
            total = float(out.sum())
            if abs(total) > 1e-12:
                out *= budget / total
    return out


def _native_candidate(mu: np.ndarray, cov: np.ndarray, request: OptimizationProblem) -> np.ndarray:
    if hasattr(_native, "solve_mean_variance"):
        return np.asarray(
            _native.solve_mean_variance(
                mu.tolist(),
                cov.tolist(),
                float(request.objective.risk_aversion),
                int(request.config.max_iterations),
                float(request.config.tolerance),
            ),
            dtype=float,
        )
    inv = np.linalg.pinv(cov)
    return inv @ mu


def _exposure_vectors(request: OptimizationProblem, size: int) -> list[tuple[np.ndarray, float | None, float | None]]:
    vectors: list[tuple[np.ndarray, float | None, float | None]] = []
    if request.factor_loadings is not None and request.factor_names is not None:
        loadings = np.asarray(request.factor_loadings, dtype=float)
        if loadings.shape[0] != size or loadings.shape[1] != len(request.factor_names):
            raise ValueError("factor_loadings dimensions must match assets x factor_names")
        for idx, name in enumerate(request.factor_names):
            lo, hi = request.constraints.factor_bounds.get(name, (None, None))
            if lo is not None or hi is not None:
                vectors.append((loadings[:, idx], lo, hi))

    if request.sector_labels is not None:
        if len(request.sector_labels) != size:
            raise ValueError("sector_labels length must match expected_returns length")
        labels = np.asarray(request.sector_labels)
        for sector, bounds in request.constraints.sector_bounds.items():
            lo, hi = bounds
            vec = (labels == sector).astype(float)
            vectors.append((vec, lo, hi))
    return vectors


def _apply_exposure_corrections(
    weights: np.ndarray,
    exposure_vectors: list[tuple[np.ndarray, float | None, float | None]],
    *,
    penalty: float,
) -> np.ndarray:
    out = weights.copy()
    if not exposure_vectors:
        return out
    for _ in range(4):
        for vec, lo, hi in exposure_vectors:
            denom = float(vec @ vec)
            if denom <= 0.0:
                continue
            exposure = float(vec @ out)
            if hi is not None and exposure > hi:
                out -= ((exposure - hi) / denom) * vec * min(1.0, penalty / (penalty + 1.0))
            if lo is not None and exposure < lo:
                out += ((lo - exposure) / denom) * vec * min(1.0, penalty / (penalty + 1.0))
    return out


def _cost_value(weights: np.ndarray, current: np.ndarray, liquidity_limits: np.ndarray | None, request: OptimizationProblem) -> float:
    trades = weights - current
    abs_trade = np.abs(trades)
    linear_bps = float(request.costs.linear_fee_bps + request.costs.spread_bps)
    linear_cost = linear_bps / 10_000.0 * float(abs_trade.sum())
    turnover_cost = float(request.costs.turnover_penalty) * float(abs_trade.sum())

    if liquidity_limits is not None:
        impact_base = abs_trade / np.maximum(liquidity_limits, 1e-12)
    else:
        impact_base = abs_trade
    impact_cost = float(request.costs.impact_coefficient) * float(np.square(impact_base).sum())
    return linear_cost + turnover_cost + impact_cost


def _objective_value(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    current: np.ndarray,
    liquidity_limits: np.ndarray | None,
    request: OptimizationProblem,
) -> float:
    risk_term = float(request.objective.risk_aversion) * float(weights @ cov @ weights)
    return_term = float(mu @ weights)
    return risk_term - return_term + _cost_value(weights, current, liquidity_limits, request)


def _gradient(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    current: np.ndarray,
    liquidity_limits: np.ndarray | None,
    request: OptimizationProblem,
    exposure_vectors: list[tuple[np.ndarray, float | None, float | None]],
) -> np.ndarray:
    grad = 2.0 * float(request.objective.risk_aversion) * (cov @ weights) - mu

    trades = weights - current
    grad += (float(request.costs.linear_fee_bps + request.costs.spread_bps) / 10_000.0) * np.sign(trades)
    grad += float(request.costs.turnover_penalty) * np.sign(trades)

    if float(request.costs.impact_coefficient) > 0.0:
        if liquidity_limits is not None:
            scale = np.maximum(liquidity_limits, 1e-12)
            grad += 2.0 * float(request.costs.impact_coefficient) * trades / np.square(scale)
        else:
            grad += 2.0 * float(request.costs.impact_coefficient) * trades

    penalty = float(request.config.constraint_penalty)
    for vec, lo, hi in exposure_vectors:
        exposure = float(vec @ weights)
        if hi is not None and exposure > hi:
            grad += penalty * (exposure - hi) * vec
        if lo is not None and exposure < lo:
            grad -= penalty * (lo - exposure) * vec
    return grad


def _projected_gradient_solve(
    mu: np.ndarray,
    cov: np.ndarray,
    request: OptimizationProblem,
    *,
    current: np.ndarray,
    liquidity_limits: np.ndarray | None,
    lot_sizes: np.ndarray | None,
    exposure_vectors: list[tuple[np.ndarray, float | None, float | None]],
    initial: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    step = 1.0 / (2.0 * float(request.objective.risk_aversion) * max(np.linalg.norm(cov, ord=2), 1e-8) + 1.0)
    weights = initial.copy()
    converged = False
    iterations = int(request.config.max_iterations)

    for idx in range(iterations):
        grad = _gradient(
            weights,
            mu,
            cov,
            current=current,
            liquidity_limits=liquidity_limits,
            request=request,
            exposure_vectors=exposure_vectors,
        )
        candidate = weights - step * grad
        candidate = _apply_exposure_corrections(
            candidate,
            exposure_vectors,
            penalty=float(request.config.constraint_penalty),
        )
        candidate = _project_weights(
            candidate,
            min_weight=request.constraints.min_weight,
            max_weight=request.constraints.max_weight,
            long_only=request.constraints.long_only,
            budget=float(request.constraints.budget),
            current=current,
            liquidity_limits=liquidity_limits,
            lot_sizes=lot_sizes,
            max_turnover=request.constraints.max_turnover,
            leverage_limit=request.constraints.leverage_limit,
        )
        if float(np.max(np.abs(candidate - weights))) <= float(request.config.tolerance):
            weights = candidate
            converged = True
            return weights, converged, idx + 1
        weights = candidate

    return weights, converged, iterations


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

    size = mu.size
    current = (
        np.asarray(request.current_weights, dtype=float).reshape(-1)
        if request.current_weights is not None
        else _default_current(size, float(request.constraints.budget))
    )
    if current.size != size:
        raise ValueError("current_weights length must match expected_returns length")

    liquidity_limits = None
    if request.liquidity_limits is not None:
        liquidity_limits = np.asarray(request.liquidity_limits, dtype=float).reshape(-1)
        if liquidity_limits.size != size:
            raise ValueError("liquidity_limits length must match expected_returns length")
        if np.any(liquidity_limits < 0.0):
            raise ValueError("liquidity_limits must be non-negative")

    lot_sizes = None
    if request.lot_sizes is not None:
        lot_sizes = np.asarray(request.lot_sizes, dtype=float).reshape(-1)
        if lot_sizes.size != size:
            raise ValueError("lot_sizes length must match expected_returns length")
        if np.any(lot_sizes <= 0.0):
            raise ValueError("lot_sizes must be > 0")

    exposure_vectors = _exposure_vectors(request, size)

    native_available = hasattr(_native, "solve_mean_variance")
    mode = request.config.solver
    if mode == "auto":
        mode = "native" if native_available else "projected_gradient"

    native_seed = _native_candidate(mu, cov, request)
    native_seed = _apply_exposure_corrections(
        native_seed,
        exposure_vectors,
        penalty=float(request.config.constraint_penalty),
    )
    native_seed = _project_weights(
        native_seed,
        min_weight=request.constraints.min_weight,
        max_weight=request.constraints.max_weight,
        long_only=request.constraints.long_only,
        budget=float(request.constraints.budget),
        current=current,
        liquidity_limits=liquidity_limits,
        lot_sizes=lot_sizes,
        max_turnover=request.constraints.max_turnover,
        leverage_limit=request.constraints.leverage_limit,
    )

    if request.config.adapter == "numpy_qp" and mode == "native":
        mode = "projected_gradient"

    if mode == "native":
        weights = native_seed
        converged = True
        iterations = 1
    else:
        weights, converged, iterations = _projected_gradient_solve(
            mu,
            cov,
            request,
            current=current,
            liquidity_limits=liquidity_limits,
            lot_sizes=lot_sizes,
            exposure_vectors=exposure_vectors,
            initial=native_seed if request.config.adapter != "numpy_qp" else current,
        )

    objective = _objective_value(
        weights,
        mu,
        cov,
        current=current,
        liquidity_limits=liquidity_limits,
        request=request,
    )
    turnover = float(np.abs(weights - current).sum())
    leverage = float(np.abs(weights).sum())
    return OptimizationResult(
        weights=tuple(float(w) for w in weights),
        objective_value=objective,
        converged=converged,
        iterations=iterations,
        diagnostics={
            "expected_return": float(mu @ weights),
            "portfolio_variance": float(weights @ cov @ weights),
            "turnover": turnover,
            "leverage": leverage,
            "cost": _cost_value(weights, current, liquidity_limits, request),
        },
    )
