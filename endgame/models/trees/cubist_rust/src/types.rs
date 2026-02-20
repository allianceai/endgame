//! Core types for Cubist regression trees
//!
//! Based on the original Cubist C implementation by Ross Quinlan

use std::collections::HashSet;

/// Constants matching the original Cubist
pub const MINSPLIT: usize = 3;       // Minimum cases before considering a split
pub const MINFRACT: f64 = 0.001;     // Minimum fraction of cases in a branch
pub const MAXN: usize = 20;          // Maximum number of coefficients in linear model

/// Branch type for tree splits
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BranchType {
    /// Not a split (leaf node)
    None,
    /// Discrete attribute split (one branch per value)
    Discrete,
    /// Continuous attribute threshold split (<=, >)
    Threshold,
    /// Subset split for discrete attributes
    Subset,
}

impl Default for BranchType {
    fn default() -> Self {
        BranchType::None
    }
}

/// Linear regression model coefficients
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Constant term (intercept)
    pub constant: f64,
    /// Coefficients for each attribute (0.0 if not used)
    pub coefficients: Vec<f64>,
    /// Which attributes are used in the model
    pub used_attrs: Vec<usize>,
}

impl LinearModel {
    pub fn new(n_attrs: usize) -> Self {
        LinearModel {
            constant: 0.0,
            coefficients: vec![0.0; n_attrs],
            used_attrs: Vec::new(),
        }
    }

    /// Predict value for a sample
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let mut result = self.constant;
        for &attr in &self.used_attrs {
            if attr < sample.len() && !sample[attr].is_nan() {
                result += self.coefficients[attr] * sample[attr];
            }
        }
        result
    }

    /// Number of terms (including constant)
    pub fn n_terms(&self) -> usize {
        self.used_attrs.len() + 1
    }
}

/// A node in the regression tree
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Type of branch/split
    pub branch_type: BranchType,
    /// Number of cases at this node
    pub cases: f64,
    /// Mean target value at this node
    pub mean: f64,
    /// Standard deviation at this node
    pub sd: f64,
    /// Linear model for this node (if any)
    pub model: Option<LinearModel>,
    /// Attribute being tested (for split nodes)
    pub tested_attr: Option<usize>,
    /// Number of branches (forks)
    pub n_forks: usize,
    /// Threshold for continuous splits
    pub cut: Option<f64>,
    /// Subset for discrete subset splits (bit vector per branch)
    pub subset: Option<Vec<HashSet<usize>>>,
    /// Child branches
    pub branches: Vec<TreeNode>,
    /// Lower bound on predictions
    pub lo_val: f64,
    /// Upper bound on predictions
    pub hi_val: f64,
}

impl TreeNode {
    /// Create a new leaf node
    pub fn leaf(cases: f64, mean: f64, sd: f64, lo_val: f64, hi_val: f64) -> Self {
        TreeNode {
            branch_type: BranchType::None,
            cases,
            mean,
            sd,
            model: None,
            tested_attr: None,
            n_forks: 0,
            cut: None,
            subset: None,
            branches: Vec::new(),
            lo_val,
            hi_val,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.branches.is_empty() || self.branch_type == BranchType::None
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            1 + self.branches.iter().map(|b| b.depth()).max().unwrap_or(0)
        }
    }

    /// Get number of leaves
    pub fn n_leaves(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.branches.iter().map(|b| b.n_leaves()).sum()
        }
    }
}

/// A single condition in a rule
#[derive(Debug, Clone)]
pub struct Condition {
    /// Attribute being tested
    pub attribute: usize,
    /// Type of test
    pub test_type: BranchType,
    /// Value for threshold tests (<=)
    pub threshold_low: Option<f64>,
    /// Value for threshold tests (>)
    pub threshold_high: Option<f64>,
    /// Subset of allowed values for discrete tests
    pub subset: Option<HashSet<usize>>,
}

impl Condition {
    /// Create a threshold <= condition
    pub fn threshold_le(attr: usize, value: f64) -> Self {
        Condition {
            attribute: attr,
            test_type: BranchType::Threshold,
            threshold_low: None,
            threshold_high: Some(value),
            subset: None,
        }
    }

    /// Create a threshold > condition
    pub fn threshold_gt(attr: usize, value: f64) -> Self {
        Condition {
            attribute: attr,
            test_type: BranchType::Threshold,
            threshold_low: Some(value),
            threshold_high: None,
            subset: None,
        }
    }

    /// Create a discrete subset condition
    pub fn discrete_subset(attr: usize, values: HashSet<usize>) -> Self {
        Condition {
            attribute: attr,
            test_type: BranchType::Subset,
            threshold_low: None,
            threshold_high: None,
            subset: Some(values),
        }
    }

    /// Test if a sample satisfies this condition
    pub fn test(&self, sample: &[f64]) -> bool {
        let value = sample[self.attribute];

        if value.is_nan() {
            return true; // Missing values pass all tests
        }

        match self.test_type {
            BranchType::Threshold => {
                if let Some(low) = self.threshold_low {
                    if value <= low {
                        return false;
                    }
                }
                if let Some(high) = self.threshold_high {
                    if value > high {
                        return false;
                    }
                }
                true
            }
            BranchType::Subset | BranchType::Discrete => {
                if let Some(ref subset) = self.subset {
                    subset.contains(&(value as usize))
                } else {
                    true
                }
            }
            BranchType::None => true,
        }
    }
}

/// A regression rule
#[derive(Debug, Clone)]
pub struct Rule {
    /// Rule number
    pub rule_no: usize,
    /// Model number (for committee)
    pub model_no: usize,
    /// Conditions (left-hand side)
    pub conditions: Vec<Condition>,
    /// Linear model (right-hand side)
    pub model: LinearModel,
    /// Number of cases covered
    pub cover: f64,
    /// Mean target value
    pub mean: f64,
    /// Lower bound on target values seen
    pub lo_val: f64,
    /// Upper bound on target values seen
    pub hi_val: f64,
    /// Lower prediction limit
    pub lo_lim: f64,
    /// Upper prediction limit
    pub hi_lim: f64,
    /// Estimated error
    pub est_err: f64,
}

impl Rule {
    /// Create a new empty rule
    pub fn new(rule_no: usize, model_no: usize, n_attrs: usize) -> Self {
        Rule {
            rule_no,
            model_no,
            conditions: Vec::new(),
            model: LinearModel::new(n_attrs),
            cover: 0.0,
            mean: 0.0,
            lo_val: f64::MAX,
            hi_val: f64::MIN,
            lo_lim: f64::MIN,
            hi_lim: f64::MAX,
            est_err: 0.0,
        }
    }

    /// Test if a sample matches this rule
    pub fn matches(&self, sample: &[f64]) -> bool {
        self.conditions.iter().all(|c| c.test(sample))
    }

    /// Predict value for a sample (bounded)
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let raw = self.model.predict(sample);
        raw.max(self.lo_lim).min(self.hi_lim)
    }

    /// Predict raw value (unbounded)
    pub fn predict_raw(&self, sample: &[f64]) -> f64 {
        self.model.predict(sample)
    }
}

/// A ruleset (collection of rules)
#[derive(Debug, Clone)]
pub struct RuleSet {
    /// Rules in this set
    pub rules: Vec<Rule>,
    /// Default prediction (when no rules match)
    pub default_val: f64,
}

impl RuleSet {
    pub fn new() -> Self {
        RuleSet {
            rules: Vec::new(),
            default_val: 0.0,
        }
    }

    /// Predict by averaging matching rules
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let matching: Vec<&Rule> = self.rules.iter()
            .filter(|r| r.matches(sample))
            .collect();

        if matching.is_empty() {
            return self.default_val;
        }

        // Weighted average by inverse error (or simple average if errors are zero)
        let total_weight: f64 = matching.iter()
            .map(|r| if r.est_err > 0.0 { 1.0 / r.est_err } else { 1.0 })
            .sum();

        if total_weight > 0.0 {
            matching.iter()
                .map(|r| {
                    let weight = if r.est_err > 0.0 { 1.0 / r.est_err } else { 1.0 };
                    r.predict(sample) * weight
                })
                .sum::<f64>() / total_weight
        } else {
            matching.iter().map(|r| r.predict(sample)).sum::<f64>() / matching.len() as f64
        }
    }
}

/// Dataset for regression
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature values (n_samples x n_features, row-major)
    pub features: Vec<f64>,
    /// Target values
    pub targets: Vec<f64>,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Sample weights
    pub weights: Vec<f64>,
    /// Attribute types (true = continuous, false = discrete)
    pub attr_continuous: Vec<bool>,
    /// Number of discrete values for each attribute (0 for continuous)
    pub attr_n_values: Vec<usize>,
    /// Global mean of target
    pub target_mean: f64,
    /// Global SD of target
    pub target_sd: f64,
    /// Min target value
    pub target_min: f64,
    /// Max target value
    pub target_max: f64,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(n_samples: usize, n_features: usize) -> Self {
        Dataset {
            features: vec![0.0; n_samples * n_features],
            targets: vec![0.0; n_samples],
            n_samples,
            n_features,
            weights: vec![1.0; n_samples],
            attr_continuous: vec![true; n_features],
            attr_n_values: vec![0; n_features],
            target_mean: 0.0,
            target_sd: 1.0,
            target_min: 0.0,
            target_max: 0.0,
        }
    }

    /// Get feature value
    pub fn get_value(&self, sample: usize, feature: usize) -> f64 {
        self.features[sample * self.n_features + feature]
    }

    /// Set feature value
    pub fn set_value(&mut self, sample: usize, feature: usize, value: f64) {
        self.features[sample * self.n_features + feature] = value;
    }

    /// Get target value
    pub fn get_target(&self, sample: usize) -> f64 {
        self.targets[sample]
    }

    /// Set target value
    pub fn set_target(&mut self, sample: usize, value: f64) {
        self.targets[sample] = value;
    }

    /// Get a row as a slice
    pub fn get_row(&self, sample: usize) -> &[f64] {
        let start = sample * self.n_features;
        &self.features[start..start + self.n_features]
    }

    /// Compute target statistics
    pub fn compute_target_stats(&mut self) {
        if self.n_samples == 0 {
            return;
        }

        let sum: f64 = self.targets.iter()
            .zip(&self.weights)
            .map(|(&t, &w)| t * w)
            .sum();
        let total_weight: f64 = self.weights.iter().sum();

        self.target_mean = sum / total_weight;

        let var: f64 = self.targets.iter()
            .zip(&self.weights)
            .map(|(&t, &w)| {
                let diff = t - self.target_mean;
                diff * diff * w
            })
            .sum::<f64>() / total_weight;

        self.target_sd = var.sqrt();

        self.target_min = self.targets.iter().cloned().fold(f64::INFINITY, f64::min);
        self.target_max = self.targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }

    /// Compute mean and SD for a subset of samples
    pub fn subset_stats(&self, indices: &[usize]) -> (f64, f64, f64, f64) {
        if indices.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mut sum = 0.0;
        let mut total_weight = 0.0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &i in indices {
            let t = self.targets[i];
            let w = self.weights[i];
            sum += t * w;
            total_weight += w;
            min_val = min_val.min(t);
            max_val = max_val.max(t);
        }

        let mean = sum / total_weight;

        let var: f64 = indices.iter()
            .map(|&i| {
                let diff = self.targets[i] - mean;
                diff * diff * self.weights[i]
            })
            .sum::<f64>() / total_weight;

        (mean, var.sqrt(), min_val, max_val)
    }
}

/// Configuration for Cubist tree building
#[derive(Debug, Clone)]
pub struct CubistConfig {
    /// Minimum cases in a leaf
    pub min_cases: usize,
    /// Maximum number of rules (0 = unlimited)
    pub max_rules: usize,
    /// Sample fraction (for sub-sampling)
    pub sample: f64,
    /// Random seed
    pub seed: u64,
    /// Whether to use instance-based (nearest neighbor) correction
    pub use_instance: bool,
    /// Number of neighbors for instance-based correction
    pub neighbors: usize,
    /// Number of committee members
    pub committees: usize,
    /// Whether to extrapolate beyond training range
    pub extrapolation: f64,
    /// Whether to unbiased (cf=0 style)
    pub unbiased: bool,
}

impl Default for CubistConfig {
    fn default() -> Self {
        CubistConfig {
            min_cases: 2,
            max_rules: 0,
            sample: 1.0,
            seed: 42,
            use_instance: false,
            neighbors: 5,
            committees: 1,
            extrapolation: 0.05,
            unbiased: false,
        }
    }
}

/// Working environment for regression computations
#[derive(Debug)]
pub struct RegressionEnv {
    /// X^T X matrix (n+1 x n+1 for n attributes + intercept)
    pub xtx: Vec<Vec<f64>>,
    /// X^T y vector
    pub xty: Vec<f64>,
    /// Working matrix A for Gaussian elimination
    pub a: Vec<Vec<f64>>,
    /// Working vector B
    pub b: Vec<f64>,
    /// Number of attributes being used
    pub n_attrs: usize,
}

impl RegressionEnv {
    pub fn new(max_attrs: usize) -> Self {
        let size = max_attrs + 1;
        RegressionEnv {
            xtx: vec![vec![0.0; size]; size],
            xty: vec![0.0; size],
            a: vec![vec![0.0; size]; size],
            b: vec![0.0; size],
            n_attrs: 0,
        }
    }

    /// Reset for new computation
    pub fn reset(&mut self, n_attrs: usize) {
        self.n_attrs = n_attrs;
        let size = n_attrs + 1;
        for row in &mut self.xtx {
            for v in row.iter_mut().take(size) {
                *v = 0.0;
            }
        }
        for v in self.xty.iter_mut().take(size) {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model() {
        let mut model = LinearModel::new(3);
        model.constant = 1.0;
        model.coefficients[0] = 2.0;
        model.coefficients[1] = 3.0;
        model.used_attrs = vec![0, 1];

        let sample = vec![1.0, 2.0, 3.0];
        let pred = model.predict(&sample);
        assert!((pred - 9.0).abs() < 1e-10); // 1 + 2*1 + 3*2 = 9
    }

    #[test]
    fn test_condition_threshold() {
        let cond = Condition::threshold_le(0, 5.0);
        assert!(cond.test(&[3.0]));
        assert!(cond.test(&[5.0]));
        assert!(!cond.test(&[6.0]));

        let cond = Condition::threshold_gt(0, 5.0);
        assert!(!cond.test(&[3.0]));
        assert!(!cond.test(&[5.0]));
        assert!(cond.test(&[6.0]));
    }

    #[test]
    fn test_dataset_stats() {
        let mut ds = Dataset::new(5, 2);
        for i in 0..5 {
            ds.set_target(i, (i + 1) as f64);
        }
        ds.compute_target_stats();

        assert!((ds.target_mean - 3.0).abs() < 1e-10);
        assert!(ds.target_min == 1.0);
        assert!(ds.target_max == 5.0);
    }
}
