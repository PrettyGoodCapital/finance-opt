use pyo3::prelude::*;

#[pyfunction]
fn solve_mean_variance(
    expected_returns: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    risk_aversion: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Vec<f64>> {
    finance_opt::solve_mean_variance(
        &expected_returns,
        &covariance,
        risk_aversion,
        max_iterations,
        tolerance,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}


#[pymodule]
fn finance_opt(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_mean_variance, m)?)?;
    Ok(())
}
