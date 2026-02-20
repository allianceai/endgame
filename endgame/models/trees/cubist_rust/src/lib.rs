//! Cubist Regression Tree Implementation
//!
//! A Rust implementation of the Cubist algorithm for rule-based regression.

mod types;
mod regress;
mod split;
mod tree;
mod rules;
mod predict;

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};

use types::*;
use tree::*;
use rules::*;
use predict::*;

/// Main Cubist model
pub struct CubistModel {
    /// Trees from each committee member
    trees: Vec<CubistTree>,
    /// Rule sets from each committee member
    rulesets: Vec<RuleSet>,
    /// Predictor
    predictor: CubistPredictor,
    /// Number of features
    n_features: usize,
    /// Configuration
    config: CubistConfig,
}

impl CubistModel {
    pub fn new(config: CubistConfig) -> Self {
        CubistModel {
            trees: Vec::new(),
            rulesets: Vec::new(),
            predictor: CubistPredictor::new(config.clone()),
            n_features: 0,
            config,
        }
    }

    /// Fit the model on training data
    pub fn fit(&mut self, features: &[f64], targets: &[f64], n_samples: usize, n_features: usize) {
        self.n_features = n_features;

        // Create dataset
        let mut dataset = Dataset::new(n_samples, n_features);
        dataset.features = features.to_vec();
        dataset.targets = targets.to_vec();
        dataset.compute_target_stats();

        // Store for instance-based correction
        if self.config.use_instance {
            self.predictor.set_training_data(TrainingData::from_dataset(&dataset));
        }

        // Store global stats
        self.predictor.target_mean = dataset.target_mean;
        self.predictor.target_sd = dataset.target_sd;
        self.predictor.target_min = dataset.target_min;
        self.predictor.target_max = dataset.target_max;

        // Clear previous trees
        self.trees.clear();
        self.rulesets.clear();
        self.predictor.rulesets.clear();

        // Build committee members
        // Cubist uses a special boosting strategy: each subsequent model predicts
        // 2 * original_target - prediction, not residuals!
        let original_targets = targets.to_vec();
        let mut working_targets = targets.to_vec();

        for member in 0..self.config.committees {
            // Update targets for boosting using Cubist's reflection approach
            // new_target = 2 * original - prediction
            if member > 0 && !self.rulesets.is_empty() {
                // Get prediction from PREVIOUS ruleset only (not cumulative)
                let prev_ruleset = &self.rulesets[member - 1];
                for i in 0..n_samples {
                    let row = dataset.get_row(i);
                    let pred = prev_ruleset.predict(row);
                    working_targets[i] = 2.0 * original_targets[i] - pred;
                }

                // Update dataset with modified targets
                dataset.targets = working_targets.clone();
                dataset.compute_target_stats();
            }

            // Build tree
            let mut builder = TreeBuilder::new(&dataset, self.config.clone());
            let tree = builder.build();

            // Extract rules
            let extractor = RuleExtractor::new(&dataset);
            let mut ruleset = extractor.extract(&tree.root, member);

            // Prune rules
            extractor.prune_rules(&mut ruleset);

            // Prune conditions
            let indices: Vec<usize> = (0..n_samples).collect();
            let pruner = RulePruner::new(&dataset);
            pruner.prune(&mut ruleset, &indices);

            // Store
            self.trees.push(tree);
            self.predictor.add_ruleset(ruleset.clone());
            self.rulesets.push(ruleset);
        }
    }

    /// Predict for a single sample using internal predictor
    fn predict_sample(&self, sample: &[f64]) -> f64 {
        self.predictor.predict(sample)
    }

    /// Predict for a single sample
    pub fn predict(&self, sample: &[f64]) -> f64 {
        self.predictor.predict(sample)
    }

    /// Predict for multiple samples
    pub fn predict_batch(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row = &samples[i * self.n_features..(i + 1) * self.n_features];
            predictions.push(self.predict(row));
        }
        predictions
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> Vec<f64> {
        if self.trees.is_empty() {
            return vec![1.0 / self.n_features as f64; self.n_features];
        }

        // Sum importances across all trees
        let mut total_imp = vec![0.0; self.n_features];
        for tree in &self.trees {
            for (i, &imp) in tree.importances.iter().enumerate() {
                total_imp[i] += imp;
            }
        }

        // Normalize
        let sum: f64 = total_imp.iter().sum();
        if sum > 0.0 {
            total_imp.iter().map(|&x| x / sum).collect()
        } else {
            vec![1.0 / self.n_features as f64; self.n_features]
        }
    }

    /// Get number of rules
    pub fn n_rules(&self) -> usize {
        self.rulesets.iter().map(|rs| rs.rules.len()).sum()
    }

    /// Get tree statistics
    pub fn tree_stats(&self) -> (usize, usize, usize) {
        let n_trees = self.trees.len();
        let total_depth: usize = self.trees.iter().map(|t| t.depth()).sum();
        let total_leaves: usize = self.trees.iter().map(|t| t.n_leaves()).sum();
        (n_trees, total_depth, total_leaves)
    }
}

// Python bindings

#[pyclass(name = "CubistRust")]
struct PyCubist {
    model: CubistModel,
}

#[pymethods]
impl PyCubist {
    #[new]
    #[pyo3(signature = (
        min_cases=2,
        max_rules=0,
        sample=1.0,
        seed=42,
        use_instance=false,
        neighbors=5,
        committees=1,
        extrapolation=0.05,
        unbiased=false
    ))]
    fn new(
        min_cases: usize,
        max_rules: usize,
        sample: f64,
        seed: u64,
        use_instance: bool,
        neighbors: usize,
        committees: usize,
        extrapolation: f64,
        unbiased: bool,
    ) -> Self {
        let config = CubistConfig {
            min_cases,
            max_rules,
            sample,
            seed,
            use_instance,
            neighbors,
            committees,
            extrapolation,
            unbiased,
        };

        PyCubist {
            model: CubistModel::new(config),
        }
    }

    /// Fit the model
    fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        let n_samples = x_arr.nrows();
        let n_features = x_arr.ncols();

        // Flatten features
        let features: Vec<f64> = x_arr.iter().copied().collect();
        let targets: Vec<f64> = y_arr.iter().copied().collect();

        self.model.fit(&features, &targets, n_samples, n_features);

        Ok(())
    }

    /// Predict for samples
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray1<f64> {
        let x_arr = x.as_array();
        let n_samples = x_arr.nrows();

        let features: Vec<f64> = x_arr.iter().copied().collect();
        let predictions = self.model.predict_batch(&features, n_samples);

        predictions.into_pyarray(py)
    }

    /// Get feature importances
    fn feature_importances<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        let imp = self.model.feature_importances();
        imp.into_pyarray(py)
    }

    /// Get number of rules
    fn n_rules(&self) -> usize {
        self.model.n_rules()
    }

    /// Get tree statistics
    fn tree_stats(&self) -> (usize, usize, usize) {
        self.model.tree_stats()
    }

    /// Get number of committee members
    fn n_committees(&self) -> usize {
        self.model.trees.len()
    }
}

/// Python module
#[pymodule]
fn cubist_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCubist>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubist_fit_predict() {
        let config = CubistConfig::default();
        let mut model = CubistModel::new(config);

        // Simple linear data: y = 2x + 1
        let features: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let targets: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();

        model.fit(&features, &targets, 20, 1);

        // Test prediction
        let pred = model.predict(&[10.0]);
        let actual = 2.0 * 10.0 + 1.0;

        // Should be reasonably close
        assert!((pred - actual).abs() < 5.0);
    }

    #[test]
    fn test_cubist_with_committees() {
        let mut config = CubistConfig::default();
        config.committees = 3;

        let mut model = CubistModel::new(config);

        // Quadratic data
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in 0..50 {
            let x = i as f64 / 10.0;
            features.push(x);
            targets.push(x * x);
        }

        model.fit(&features, &targets, 50, 1);

        assert_eq!(model.trees.len(), 3);

        // Test prediction
        let pred = model.predict(&[2.5]);
        let actual = 2.5 * 2.5;

        // Committee should improve prediction
        assert!((pred - actual).abs() < 2.0);
    }
}
