from collections.abc import Sequence

def solve_mean_variance(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    risk_aversion: float,
    max_iterations: int,
    tolerance: float,
) -> list[float]: ...
