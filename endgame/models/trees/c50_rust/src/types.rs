//! Core type definitions for C5.0

use std::collections::HashMap;
use ordered_float::OrderedFloat;
use serde::{Serialize, Deserialize};

/// Special value indicating unknown/missing value
pub const UNKNOWN: f64 = f64::NAN;

/// Special value indicating N/A (not applicable)
pub const NA: f64 = f64::NEG_INFINITY;

/// Check if value is unknown
#[inline]
pub fn is_unknown(v: f64) -> bool {
    v.is_nan()
}

/// Check if value is N/A
#[inline]
pub fn is_na(v: f64) -> bool {
    v == NA
}

/// Check if value is valid (not unknown and not N/A)
#[inline]
pub fn is_valid(v: f64) -> bool {
    !is_unknown(v) && !is_na(v)
}

/// Tree configuration parameters
#[derive(Clone, Debug)]
pub struct TreeConfig {
    /// Minimum cases in a branch
    pub min_cases: usize,
    /// Confidence factor for pruning (default 0.25)
    pub cf: f64,
    /// Use subset splits for discrete attributes
    pub use_subset: bool,
    /// Apply global pruning
    pub global_pruning: bool,
    /// Use soft thresholds for continuous attributes
    pub soft_threshold: bool,
    /// Number of threads (0 = auto)
    pub n_threads: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            min_cases: 2,
            cf: 0.25,
            use_subset: true,
            global_pruning: true,
            soft_threshold: false,
            n_threads: 0,
            random_state: None,
        }
    }
}

/// Type of node in the tree
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    /// Leaf node (prediction)
    Leaf,
    /// Split on discrete attribute (one branch per value)
    Discrete,
    /// Split on continuous attribute (threshold)
    Threshold,
    /// Split on discrete attribute using subsets
    Subset,
}

/// A tree node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeNode {
    /// Type of this node
    pub node_type: NodeType,
    /// Predicted class (for all nodes, used as default)
    pub class: usize,
    /// Number of cases reaching this node
    pub cases: f64,
    /// Class distribution [n_classes]
    pub class_dist: Vec<f64>,
    /// Estimated errors at this node
    pub errors: f64,
    /// Attribute being tested (for non-leaf nodes)
    pub tested_attr: Option<usize>,
    /// Number of branches
    pub n_forks: usize,
    /// Threshold for continuous split
    pub threshold: Option<f64>,
    /// Soft threshold bounds (lower, mid, upper)
    pub soft_bounds: Option<(f64, f64, f64)>,
    /// Subsets for discrete split (each subset is a set of values)
    pub subsets: Option<Vec<Vec<usize>>>,
    /// Child nodes
    pub branches: Vec<TreeNode>,
}

impl TreeNode {
    /// Create a new leaf node
    pub fn leaf(class: usize, cases: f64, class_dist: Vec<f64>, errors: f64) -> Self {
        TreeNode {
            node_type: NodeType::Leaf,
            class,
            cases,
            class_dist,
            errors,
            tested_attr: None,
            n_forks: 0,
            threshold: None,
            soft_bounds: None,
            subsets: None,
            branches: Vec::new(),
        }
    }

    /// Create a threshold (continuous) split node
    pub fn threshold_split(
        attr: usize,
        threshold: f64,
        class: usize,
        cases: f64,
        class_dist: Vec<f64>,
        errors: f64,
        left: TreeNode,
        right: TreeNode,
    ) -> Self {
        TreeNode {
            node_type: NodeType::Threshold,
            class,
            cases,
            class_dist,
            errors,
            tested_attr: Some(attr),
            n_forks: 2,
            threshold: Some(threshold),
            soft_bounds: None,
            subsets: None,
            branches: vec![left, right],
        }
    }

    /// Create a discrete split node
    pub fn discrete_split(
        attr: usize,
        class: usize,
        cases: f64,
        class_dist: Vec<f64>,
        errors: f64,
        branches: Vec<TreeNode>,
    ) -> Self {
        let n_forks = branches.len();
        TreeNode {
            node_type: NodeType::Discrete,
            class,
            cases,
            class_dist,
            errors,
            tested_attr: Some(attr),
            n_forks,
            threshold: None,
            soft_bounds: None,
            subsets: None,
            branches,
        }
    }

    /// Create a subset split node
    pub fn subset_split(
        attr: usize,
        subsets: Vec<Vec<usize>>,
        class: usize,
        cases: f64,
        class_dist: Vec<f64>,
        errors: f64,
        branches: Vec<TreeNode>,
    ) -> Self {
        let n_forks = branches.len();
        TreeNode {
            node_type: NodeType::Subset,
            class,
            cases,
            class_dist,
            errors,
            tested_attr: Some(attr),
            n_forks,
            threshold: None,
            soft_bounds: None,
            subsets: Some(subsets),
            branches,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.node_type == NodeType::Leaf
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            1 + self.branches.iter().map(|b| b.depth()).max().unwrap_or(0)
        }
    }

    /// Count number of leaves
    pub fn n_leaves(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.branches.iter().map(|b| b.n_leaves()).sum()
        }
    }
}

/// Dataset for training
pub struct Dataset {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Feature values [n_samples][n_features]
    pub values: Vec<Vec<f64>>,
    /// Class labels [n_samples]
    pub classes: Vec<usize>,
    /// Sample weights [n_samples]
    pub weights: Vec<f64>,
    /// Which features are categorical
    pub categorical: Vec<bool>,
    /// Number of unique values per categorical feature
    pub n_values: Vec<usize>,
    /// Number of classes (computed from data)
    n_classes: Option<usize>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(n_samples: usize, n_features: usize) -> Self {
        Dataset {
            n_samples,
            n_features,
            values: vec![vec![0.0; n_features]; n_samples],
            classes: vec![0; n_samples],
            weights: vec![1.0; n_samples],
            categorical: vec![false; n_features],
            n_values: vec![0; n_features],
            n_classes: None,
        }
    }

    /// Set a feature value
    #[inline]
    pub fn set_value(&mut self, sample: usize, feature: usize, value: f64) {
        self.values[sample][feature] = value;
    }

    /// Get a feature value
    #[inline]
    pub fn get_value(&self, sample: usize, feature: usize) -> f64 {
        self.values[sample][feature]
    }

    /// Set class label
    #[inline]
    pub fn set_class(&mut self, sample: usize, class: usize) {
        self.classes[sample] = class;
        self.n_classes = None; // Invalidate cache
    }

    /// Get class label
    #[inline]
    pub fn get_class(&self, sample: usize) -> usize {
        self.classes[sample]
    }

    /// Set sample weight
    #[inline]
    pub fn set_weight(&mut self, sample: usize, weight: f64) {
        self.weights[sample] = weight;
    }

    /// Get sample weight
    #[inline]
    pub fn get_weight(&self, sample: usize) -> f64 {
        self.weights[sample]
    }

    /// Mark a feature as categorical
    pub fn set_categorical(&mut self, feature: usize, is_cat: bool) {
        self.categorical[feature] = is_cat;
        if is_cat {
            // Count unique values
            let mut values: Vec<OrderedFloat<f64>> = self.values.iter()
                .map(|row| OrderedFloat(row[feature]))
                .filter(|v| is_valid(v.0))
                .collect();
            values.sort();
            values.dedup();
            self.n_values[feature] = values.len();
        }
    }

    /// Check if feature is categorical
    #[inline]
    pub fn is_categorical(&self, feature: usize) -> bool {
        self.categorical[feature]
    }

    /// Get number of classes
    pub fn n_classes(&mut self) -> usize {
        if let Some(n) = self.n_classes {
            return n;
        }
        let n = self.classes.iter().max().map(|m| m + 1).unwrap_or(0);
        self.n_classes = Some(n);
        n
    }

    /// Get total weight of all samples
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Get class distribution (weighted)
    pub fn class_distribution(&self, indices: &[usize], n_classes: usize) -> Vec<f64> {
        let mut dist = vec![0.0; n_classes];
        for &i in indices {
            dist[self.classes[i]] += self.weights[i];
        }
        dist
    }
}

/// Result of evaluating a split
#[derive(Clone, Debug)]
pub struct SplitResult {
    /// Attribute being split
    pub attribute: usize,
    /// Type of split
    pub split_type: NodeType,
    /// Information gain
    pub gain: f64,
    /// Gain ratio (gain / split info)
    pub gain_ratio: f64,
    /// Threshold for continuous split
    pub threshold: Option<f64>,
    /// Subsets for subset split
    pub subsets: Option<Vec<Vec<usize>>>,
    /// Cases in each branch
    pub branch_cases: Vec<f64>,
    /// Class distributions in each branch [branch][class]
    pub branch_class_dist: Vec<Vec<f64>>,
    /// Indices assigned to each branch
    pub branch_indices: Vec<Vec<usize>>,
}

impl SplitResult {
    /// Check if this split is valid
    pub fn is_valid(&self) -> bool {
        self.gain > 0.0 && self.gain_ratio > 0.0
    }
}

/// Sorted record for continuous attribute evaluation
#[derive(Clone, Copy)]
pub struct SortRec {
    /// Sample index
    pub index: usize,
    /// Attribute value
    pub value: f64,
    /// Sample weight
    pub weight: f64,
    /// Class label
    pub class: usize,
}

impl SortRec {
    pub fn new(index: usize, value: f64, weight: f64, class: usize) -> Self {
        SortRec { index, value, weight, class }
    }
}
