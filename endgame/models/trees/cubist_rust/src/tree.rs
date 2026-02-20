//! Tree building for Cubist regression trees
//!
//! Constructs regression trees with linear models at the leaves.

use crate::types::*;
use crate::split::*;
use crate::regress::*;

/// Main Cubist tree
#[derive(Debug, Clone)]
pub struct CubistTree {
    /// Root node
    pub root: TreeNode,
    /// Number of features
    pub n_features: usize,
    /// Feature importances
    pub importances: Vec<f64>,
    /// Global target statistics
    pub target_mean: f64,
    pub target_sd: f64,
    pub target_min: f64,
    pub target_max: f64,
}

impl CubistTree {
    /// Predict for a single sample
    pub fn predict(&self, sample: &[f64]) -> f64 {
        self.predict_node(&self.root, sample)
    }

    /// Predict using the tree, falling through to the appropriate leaf
    fn predict_node(&self, node: &TreeNode, sample: &[f64]) -> f64 {
        if node.is_leaf() {
            // Use linear model if available, otherwise use mean
            return if let Some(ref model) = node.model {
                let pred = model.predict(sample);
                // Bound prediction
                pred.max(node.lo_val).min(node.hi_val)
            } else {
                node.mean
            };
        }

        // Find which branch to follow
        let attr = node.tested_attr.unwrap_or(0);
        let value = sample.get(attr).copied().unwrap_or(f64::NAN);

        let branch_idx = match node.branch_type {
            BranchType::Threshold => {
                let cut = node.cut.unwrap_or(0.0);
                if value.is_nan() {
                    // Missing value: use branch with more cases
                    if node.branches.len() >= 2 {
                        if node.branches[0].cases >= node.branches[1].cases { 0 } else { 1 }
                    } else {
                        0
                    }
                } else if value <= cut {
                    0
                } else {
                    1
                }
            }
            BranchType::Discrete => {
                if value.is_nan() {
                    // Missing: use largest branch
                    node.branches.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.cases.partial_cmp(&b.1.cases).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                } else {
                    let vi = value as usize;
                    vi.min(node.branches.len() - 1)
                }
            }
            BranchType::Subset => {
                if value.is_nan() {
                    0
                } else if let Some(ref subsets) = node.subset {
                    let vi = value as usize;
                    subsets.iter()
                        .position(|s| s.contains(&vi))
                        .unwrap_or(0)
                } else {
                    0
                }
            }
            BranchType::None => 0,
        };

        if branch_idx < node.branches.len() {
            self.predict_node(&node.branches[branch_idx], sample)
        } else {
            node.mean
        }
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Get number of leaves
    pub fn n_leaves(&self) -> usize {
        self.root.n_leaves()
    }

    /// Get normalized feature importances
    pub fn feature_importances(&self) -> Vec<f64> {
        let total: f64 = self.importances.iter().sum();
        if total > 0.0 {
            self.importances.iter().map(|&x| x / total).collect()
        } else {
            vec![1.0 / self.n_features as f64; self.n_features]
        }
    }
}

/// Builder for Cubist trees
pub struct TreeBuilder<'a> {
    dataset: &'a Dataset,
    config: CubistConfig,
    importances: Vec<f64>,
    regression_solver: RegressionSolver,
}

impl<'a> TreeBuilder<'a> {
    pub fn new(dataset: &'a Dataset, config: CubistConfig) -> Self {
        TreeBuilder {
            dataset,
            config,
            importances: vec![0.0; dataset.n_features],
            regression_solver: RegressionSolver::new(dataset.n_features),
        }
    }

    /// Build the tree
    pub fn build(&mut self) -> CubistTree {
        let indices: Vec<usize> = (0..self.dataset.n_samples).collect();

        // Build tree structure
        let mut root = self.build_node(&indices, 0);

        // Add linear models to all nodes
        self.add_models(&mut root, &indices);

        CubistTree {
            root,
            n_features: self.dataset.n_features,
            importances: self.importances.clone(),
            target_mean: self.dataset.target_mean,
            target_sd: self.dataset.target_sd,
            target_min: self.dataset.target_min,
            target_max: self.dataset.target_max,
        }
    }

    /// Build a node recursively
    fn build_node(&mut self, indices: &[usize], depth: usize) -> TreeNode {
        let (mean, sd, min_val, max_val) = self.dataset.subset_stats(indices);
        let total_weight: f64 = indices.iter().map(|&i| self.dataset.weights[i]).sum();

        // Check stopping conditions
        if self.should_stop(indices, sd, total_weight) {
            return TreeNode::leaf(total_weight, mean, sd, min_val, max_val);
        }

        // Find best split
        let evaluator = SplitEvaluator::new(self.dataset, indices, &self.config);
        let best_split = match evaluator.find_best_split() {
            Some(s) => s,
            None => return TreeNode::leaf(total_weight, mean, sd, min_val, max_val),
        };

        // Update importance
        self.importances[best_split.attribute] += best_split.gain * total_weight;

        // Build child nodes
        let children: Vec<TreeNode> = best_split.branch_indices.iter()
            .map(|child_indices| {
                if child_indices.is_empty() {
                    TreeNode::leaf(0.0, mean, 0.0, mean, mean)
                } else {
                    self.build_node(child_indices, depth + 1)
                }
            })
            .collect();

        // Create split node
        TreeNode {
            branch_type: best_split.split_type,
            cases: total_weight,
            mean,
            sd,
            model: None, // Model added later
            tested_attr: Some(best_split.attribute),
            n_forks: children.len(),
            cut: best_split.threshold,
            subset: best_split.subsets,
            branches: children,
            lo_val: min_val,
            hi_val: max_val,
        }
    }

    /// Check stopping conditions
    fn should_stop(&self, indices: &[usize], sd: f64, total_weight: f64) -> bool {
        // Too few cases
        if indices.len() < 2 * self.config.min_cases {
            return true;
        }

        if total_weight < 2.0 * self.config.min_cases as f64 {
            return true;
        }

        // Very low variance
        if sd < 1e-10 {
            return true;
        }

        false
    }

    /// Add linear models to all nodes
    fn add_models(&mut self, node: &mut TreeNode, indices: &[usize]) {
        // Get usable attributes (all continuous for now)
        let usable_attrs: Vec<usize> = (0..self.dataset.n_features)
            .filter(|&i| self.dataset.attr_continuous[i])
            .collect();

        // Fit model for this node
        let (model, _err) = self.regression_solver.fit_with_outlier_elimination(
            self.dataset,
            indices,
            &usable_attrs,
        );
        node.model = Some(model);

        if node.is_leaf() {
            return;
        }

        // Distribute indices to children based on split
        let child_indices = self.distribute_indices(node, indices);

        for (child, child_idx) in node.branches.iter_mut().zip(child_indices.iter()) {
            self.add_models(child, child_idx);
        }
    }

    /// Distribute indices to child nodes based on split
    fn distribute_indices(&self, node: &TreeNode, indices: &[usize]) -> Vec<Vec<usize>> {
        let attr = node.tested_attr.unwrap_or(0);
        let mut result: Vec<Vec<usize>> = vec![Vec::new(); node.branches.len()];

        for &idx in indices {
            let value = self.dataset.get_value(idx, attr);

            let branch_idx = match node.branch_type {
                BranchType::Threshold => {
                    let cut = node.cut.unwrap_or(0.0);
                    if value.is_nan() {
                        // Send to larger branch
                        if node.branches.len() >= 2 {
                            if node.branches[0].cases >= node.branches[1].cases { 0 } else { 1 }
                        } else {
                            0
                        }
                    } else if value <= cut {
                        0
                    } else {
                        1
                    }
                }
                BranchType::Discrete => {
                    if value.is_nan() {
                        0
                    } else {
                        let vi = value as usize;
                        vi.min(node.branches.len() - 1)
                    }
                }
                BranchType::Subset => {
                    if value.is_nan() {
                        0
                    } else if let Some(ref subsets) = node.subset {
                        let vi = value as usize;
                        subsets.iter()
                            .position(|s| s.contains(&vi))
                            .unwrap_or(0)
                    } else {
                        0
                    }
                }
                BranchType::None => 0,
            };

            if branch_idx < result.len() {
                result[branch_idx].push(idx);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tree() {
        // y = 2x for x <= 5, y = 3x for x > 5
        let mut ds = Dataset::new(10, 1);
        for i in 0..5 {
            ds.set_value(i, 0, (i + 1) as f64);
            ds.set_target(i, 2.0 * (i + 1) as f64);
        }
        for i in 5..10 {
            ds.set_value(i, 0, (i + 1) as f64);
            ds.set_target(i, 3.0 * (i + 1) as f64);
        }
        ds.compute_target_stats();

        let config = CubistConfig::default();
        let mut builder = TreeBuilder::new(&ds, config);
        let tree = builder.build();

        // Should have created a split
        assert!(!tree.root.is_leaf());
        assert!(tree.depth() >= 2);

        // Test predictions
        let pred1 = tree.predict(&[3.0]);
        let pred2 = tree.predict(&[8.0]);

        // Predictions should be different for left vs right
        assert!((pred1 - pred2).abs() > 1.0);
    }

    #[test]
    fn test_linear_leaf() {
        // Pure linear relationship
        let mut ds = Dataset::new(20, 1);
        for i in 0..20 {
            let x = i as f64;
            ds.set_value(i, 0, x);
            ds.set_target(i, 2.0 * x + 1.0);
        }
        ds.compute_target_stats();

        let config = CubistConfig::default();
        let mut builder = TreeBuilder::new(&ds, config);
        let tree = builder.build();

        // Should fit well
        let pred = tree.predict(&[10.0]);
        let actual = 2.0 * 10.0 + 1.0;
        assert!((pred - actual).abs() < 2.0);
    }
}
