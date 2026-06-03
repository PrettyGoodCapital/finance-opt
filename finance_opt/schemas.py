from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ObjectiveSpec",
    "TradingCostSpec",
    "ConstraintSpec",
    "OptimizerConfig",
    "OptimizationProblem",
    "OptimizationResult",
]


class ObjectiveSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["min_variance", "max_sharpe", "mean_variance"] = "mean_variance"
    risk_aversion: float = Field(default=1.0, ge=0.0)


class TradingCostSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    linear_fee_bps: float = Field(default=0.0, ge=0.0)
    spread_bps: float = Field(default=0.0, ge=0.0)
    turnover_penalty: float = Field(default=0.0, ge=0.0)
    impact_coefficient: float = Field(default=0.0, ge=0.0)


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    long_only: bool = True
    budget: float = 1.0
    min_weight: float | None = 0.0
    max_weight: float | None = 1.0
    leverage_limit: float | None = Field(default=None, gt=0.0)
    max_turnover: float | None = Field(default=None, ge=0.0)
    factor_bounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)
    sector_bounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)

    @field_validator("max_weight")
    @classmethod
    def _validate_bounds(cls, value: float | None, info):
        min_weight = info.data.get("min_weight")
        if value is not None and min_weight is not None and value < min_weight:
            raise ValueError("max_weight must be >= min_weight")
        return value


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    solver: Literal["auto", "native", "projected_gradient"] = "auto"
    adapter: Literal["auto", "clarabel_like", "numpy_qp"] = "auto"
    max_iterations: int = Field(default=500, ge=1)
    tolerance: float = Field(default=1e-8, gt=0.0)
    constraint_penalty: float = Field(default=250.0, gt=0.0)


class OptimizationProblem(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_returns: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    asset_names: tuple[str, ...] | None = None
    current_weights: tuple[float, ...] | None = None
    liquidity_limits: tuple[float, ...] | None = None
    lot_sizes: tuple[float, ...] | None = None
    factor_names: tuple[str, ...] | None = None
    factor_loadings: tuple[tuple[float, ...], ...] | None = None
    sector_labels: tuple[str, ...] | None = None
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    costs: TradingCostSpec = Field(default_factory=TradingCostSpec)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    config: OptimizerConfig = Field(default_factory=OptimizerConfig)


class OptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: tuple[float, ...]
    objective_value: float
    converged: bool
    iterations: int
    diagnostics: dict[str, float] = Field(default_factory=dict)
