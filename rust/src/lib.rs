pub fn solve_mean_variance(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    risk_aversion: f64,
    _max_iterations: usize,
    _tolerance: f64,
) -> Result<Vec<f64>, String> {
    let n = expected_returns.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if covariance.len() != n || covariance.iter().any(|row| row.len() != n) {
        return Err("covariance dimensions must match expected_returns length".to_string());
    }

    let mut diagonal_scores = Vec::with_capacity(n);
    for i in 0..n {
        let variance = covariance[i][i].max(1e-12);
        let score = expected_returns[i] / (risk_aversion.max(1e-12) * variance);
        diagonal_scores.push(score.max(1e-12));
    }

    let score_sum: f64 = diagonal_scores.iter().sum();
    if score_sum <= 0.0 {
        return Ok(vec![1.0 / n as f64; n]);
    }

    Ok(diagonal_scores
        .into_iter()
        .map(|value| value / score_sum)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_returns_weights_that_sum_to_one() {
        let mu = vec![0.08, 0.10, 0.12];
        let cov = vec![
            vec![0.04, 0.01, 0.00],
            vec![0.01, 0.09, 0.02],
            vec![0.00, 0.02, 0.16],
        ];

        let weights = solve_mean_variance(&mu, &cov, 1.0, 200, 1e-8).unwrap();
        let total: f64 = weights.iter().sum();

        assert_eq!(weights.len(), 3);
        assert!((total - 1.0).abs() < 1e-9);
        assert!(weights.iter().all(|w| *w > 0.0));
    }

    #[test]
    fn solve_rejects_bad_covariance_shape() {
        let mu = vec![0.08, 0.10];
        let cov = vec![vec![0.04, 0.01, 0.00]];
        assert!(solve_mean_variance(&mu, &cov, 1.0, 200, 1e-8).is_err());
    }
}
