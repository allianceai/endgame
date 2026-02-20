//! Linear regression for Cubist
//!
//! Implements regression via normal equations (X^T X)^-1 X^T y
//! with model simplification and outlier handling.

use crate::types::*;

/// Regression solver for building linear models
pub struct RegressionSolver {
    /// Working environment
    env: RegressionEnv,
    /// Maximum number of attributes in model
    max_attrs: usize,
}

impl RegressionSolver {
    pub fn new(max_attrs: usize) -> Self {
        RegressionSolver {
            env: RegressionEnv::new(max_attrs.min(MAXN)),
            max_attrs: max_attrs.min(MAXN),
        }
    }

    /// Build a linear model for the given samples
    ///
    /// Returns the model and estimated error
    pub fn fit(
        &mut self,
        dataset: &Dataset,
        indices: &[usize],
        usable_attrs: &[usize],
    ) -> (LinearModel, f64) {
        let n_cases = indices.len();
        let n_attrs = usable_attrs.len().min(self.max_attrs);

        if n_cases < 2 || n_attrs == 0 {
            // Return mean model
            let (mean, _, _, _) = dataset.subset_stats(indices);
            let mut model = LinearModel::new(dataset.n_features);
            model.constant = mean;
            let err = self.compute_error(dataset, indices, &model);
            return (model, err);
        }

        // Select which attributes to use (limited to MAXN)
        let active_attrs: Vec<usize> = usable_attrs.iter()
            .take(n_attrs)
            .copied()
            .collect();

        // Build X^T X and X^T y matrices
        self.build_tables(dataset, indices, &active_attrs);

        // Solve normal equations
        let coeffs = match self.solve(active_attrs.len()) {
            Some(c) => c,
            None => {
                // Fallback to mean
                let (mean, _, _, _) = dataset.subset_stats(indices);
                let mut model = LinearModel::new(dataset.n_features);
                model.constant = mean;
                let err = self.compute_error(dataset, indices, &model);
                return (model, err);
            }
        };

        // Build initial model
        let mut model = LinearModel::new(dataset.n_features);
        model.constant = coeffs[0];
        for (i, &attr) in active_attrs.iter().enumerate() {
            model.coefficients[attr] = coeffs[i + 1];
            if coeffs[i + 1].abs() > 1e-10 {
                model.used_attrs.push(attr);
            }
        }

        // Simplify model by dropping least useful coefficients
        let (simplified_model, err) = self.simplify_model(dataset, indices, model, &active_attrs);

        (simplified_model, err)
    }

    /// Build X^T X and X^T y matrices
    fn build_tables(&mut self, dataset: &Dataset, indices: &[usize], attrs: &[usize]) {
        let n = attrs.len();
        self.env.reset(n);

        // Zero out matrices
        for i in 0..=n {
            for j in 0..=n {
                self.env.xtx[i][j] = 0.0;
            }
            self.env.xty[i] = 0.0;
        }

        for &idx in indices {
            let weight = dataset.weights[idx];
            let y = dataset.get_target(idx);

            // Build row: [1, x_1, x_2, ..., x_n]
            // XTX[i][j] += x_i * x_j * weight
            // XTy[i] += x_i * y * weight

            // Constant term (index 0)
            self.env.xtx[0][0] += weight;
            self.env.xty[0] += y * weight;

            for (i, &attr_i) in attrs.iter().enumerate() {
                let xi = dataset.get_value(idx, attr_i);
                if xi.is_nan() {
                    continue;
                }

                // X^T X[0][i+1] and X^T X[i+1][0]
                self.env.xtx[0][i + 1] += xi * weight;
                self.env.xtx[i + 1][0] += xi * weight;

                // X^T y[i+1]
                self.env.xty[i + 1] += xi * y * weight;

                for (j, &attr_j) in attrs.iter().enumerate().skip(i) {
                    let xj = dataset.get_value(idx, attr_j);
                    if xj.is_nan() {
                        continue;
                    }

                    // X^T X[i+1][j+1]
                    let prod = xi * xj * weight;
                    self.env.xtx[i + 1][j + 1] += prod;
                    if i != j {
                        self.env.xtx[j + 1][i + 1] += prod;
                    }
                }
            }
        }
    }

    /// Solve normal equations using Gaussian elimination with partial pivoting
    fn solve(&mut self, n_attrs: usize) -> Option<Vec<f64>> {
        let n = n_attrs + 1; // +1 for constant term

        // Copy to working matrices
        for i in 0..n {
            for j in 0..n {
                self.env.a[i][j] = self.env.xtx[i][j];
            }
            self.env.b[i] = self.env.xty[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = self.env.a[k][k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if self.env.a[i][k].abs() > max_val {
                    max_val = self.env.a[i][k].abs();
                    max_row = i;
                }
            }

            if max_val < 1e-12 {
                // Singular matrix
                return None;
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..n {
                    let tmp = self.env.a[k][j];
                    self.env.a[k][j] = self.env.a[max_row][j];
                    self.env.a[max_row][j] = tmp;
                }
                let tmp = self.env.b[k];
                self.env.b[k] = self.env.b[max_row];
                self.env.b[max_row] = tmp;
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = self.env.a[i][k] / self.env.a[k][k];
                for j in k..n {
                    self.env.a[i][j] -= factor * self.env.a[k][j];
                }
                self.env.b[i] -= factor * self.env.b[k];
            }
        }

        // Back substitution
        let mut result = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = self.env.b[i];
            for j in (i + 1)..n {
                sum -= self.env.a[i][j] * result[j];
            }
            if self.env.a[i][i].abs() < 1e-12 {
                return None;
            }
            result[i] = sum / self.env.a[i][i];
        }

        Some(result)
    }

    /// Simplify model by iteratively dropping least useful coefficients
    fn simplify_model(
        &mut self,
        dataset: &Dataset,
        indices: &[usize],
        mut model: LinearModel,
        active_attrs: &[usize],
    ) -> (LinearModel, f64) {
        if model.used_attrs.is_empty() {
            let err = self.compute_error(dataset, indices, &model);
            return (model, err);
        }

        let base_error = self.compute_error(dataset, indices, &model);
        let n_cases = indices.len() as f64;

        // Calculate adjusted error for comparison
        // Adjusted error = error * (n / (n - p)) where p is number of parameters
        let adjusted_base = if n_cases > model.n_terms() as f64 {
            base_error * n_cases / (n_cases - model.n_terms() as f64)
        } else {
            base_error
        };

        loop {
            if model.used_attrs.len() <= 1 {
                break;
            }

            // Try dropping each coefficient
            let mut best_drop: Option<usize> = None;
            let mut best_adj_err = adjusted_base;

            for (idx, &attr) in model.used_attrs.iter().enumerate() {
                // Temporarily drop this coefficient
                let old_coef = model.coefficients[attr];
                model.coefficients[attr] = 0.0;

                let err = self.compute_error(dataset, indices, &model);
                let n_terms = model.used_attrs.len() as f64; // -1 since we dropped one
                let adj_err = if n_cases > n_terms {
                    err * n_cases / (n_cases - n_terms)
                } else {
                    err
                };

                // Restore
                model.coefficients[attr] = old_coef;

                // Keep track of best drop (lowest adjusted error)
                if adj_err < best_adj_err - 1e-10 {
                    best_adj_err = adj_err;
                    best_drop = Some(idx);
                }
            }

            // If no improvement, stop
            if best_drop.is_none() {
                break;
            }

            // Permanently drop the coefficient
            let drop_idx = best_drop.unwrap();
            let drop_attr = model.used_attrs[drop_idx];
            model.coefficients[drop_attr] = 0.0;
            model.used_attrs.remove(drop_idx);

            // Re-fit with remaining attributes
            let remaining: Vec<usize> = active_attrs.iter()
                .filter(|&&a| model.used_attrs.contains(&a))
                .copied()
                .collect();

            if !remaining.is_empty() {
                self.build_tables(dataset, indices, &remaining);
                if let Some(coeffs) = self.solve(remaining.len()) {
                    model.constant = coeffs[0];
                    for (i, &attr) in remaining.iter().enumerate() {
                        model.coefficients[attr] = coeffs[i + 1];
                    }
                }
            }
        }

        let final_error = self.compute_error(dataset, indices, &model);
        (model, final_error)
    }

    /// Compute mean absolute error
    fn compute_error(&self, dataset: &Dataset, indices: &[usize], model: &LinearModel) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut total_err = 0.0;
        let mut total_weight = 0.0;

        for &idx in indices {
            let row = dataset.get_row(idx);
            let pred = model.predict(row);
            let actual = dataset.get_target(idx);
            let weight = dataset.weights[idx];

            total_err += (pred - actual).abs() * weight;
            total_weight += weight;
        }

        total_err / total_weight
    }

    /// Fit with outlier elimination
    pub fn fit_with_outlier_elimination(
        &mut self,
        dataset: &Dataset,
        indices: &[usize],
        usable_attrs: &[usize],
    ) -> (LinearModel, f64) {
        let (mut model, mut err) = self.fit(dataset, indices, usable_attrs);

        if indices.len() < 10 {
            return (model, err);
        }

        // Eliminate outliers iteratively
        let mut working_indices: Vec<usize> = indices.to_vec();

        for _ in 0..3 {
            // Max 3 iterations
            if working_indices.len() < 5 {
                break;
            }

            // Compute residuals
            let mut residuals: Vec<(usize, f64)> = working_indices.iter()
                .map(|&idx| {
                    let row = dataset.get_row(idx);
                    let pred = model.predict(row);
                    let actual = dataset.get_target(idx);
                    (idx, (pred - actual).abs())
                })
                .collect();

            // Compute average residual
            let avg_residual: f64 = residuals.iter().map(|(_, r)| r).sum::<f64>()
                / residuals.len() as f64;

            if avg_residual < 1e-10 {
                break;
            }

            // Remove outliers (residual > 5 * average)
            let threshold = 5.0 * avg_residual;
            let new_indices: Vec<usize> = residuals.iter()
                .filter(|(_, r)| *r <= threshold)
                .map(|(idx, _)| *idx)
                .collect();

            if new_indices.len() == working_indices.len() || new_indices.len() < 5 {
                break;
            }

            working_indices = new_indices;

            // Re-fit model
            let (new_model, new_err) = self.fit(dataset, &working_indices, usable_attrs);
            model = new_model;
            err = new_err;
        }

        (model, err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_regression() {
        // y = 2x + 1
        let mut ds = Dataset::new(10, 1);
        for i in 0..10 {
            let x = i as f64;
            ds.set_value(i, 0, x);
            ds.set_target(i, 2.0 * x + 1.0);
        }

        let mut solver = RegressionSolver::new(10);
        let indices: Vec<usize> = (0..10).collect();
        let (model, _err) = solver.fit(&ds, &indices, &[0]);

        // Should recover y = 2x + 1
        assert!((model.constant - 1.0).abs() < 0.1);
        assert!((model.coefficients[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_multivariate_regression() {
        // y = 1 + 2*x1 + 3*x2
        let mut ds = Dataset::new(20, 2);
        for i in 0..20 {
            let x1 = (i % 5) as f64;
            let x2 = (i / 5) as f64;
            ds.set_value(i, 0, x1);
            ds.set_value(i, 1, x2);
            ds.set_target(i, 1.0 + 2.0 * x1 + 3.0 * x2);
        }

        let mut solver = RegressionSolver::new(10);
        let indices: Vec<usize> = (0..20).collect();
        let (model, _err) = solver.fit(&ds, &indices, &[0, 1]);

        assert!((model.constant - 1.0).abs() < 0.1);
        assert!((model.coefficients[0] - 2.0).abs() < 0.1);
        assert!((model.coefficients[1] - 3.0).abs() < 0.1);
    }
}
