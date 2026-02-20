//! Tree building and main tree structure for C5.0

use crate::types::*;
use crate::info::*;
use crate::split::*;
use crate::prune::*;
use crate::classify::*;

/// Main C5.0 tree classifier
#[derive(Clone)]
pub struct C50Tree {
    /// Root node of the tree
    pub root: TreeNode,
    /// Number of features
    pub n_features: usize,
    /// Number of classes
    pub n_classes: usize,
    /// Feature importances (computed from split gains)
    pub importances: Vec<f64>,
}

impl C50Tree {
    /// Classify a single sample
    pub fn classify(&self, sample: &[f64]) -> usize {
        classify_sample(&self.root, sample, self.n_classes)
    }

    /// Get class probabilities for a sample
    pub fn classify_proba(&self, sample: &[f64]) -> Vec<f64> {
        classify_proba(&self.root, sample, self.n_classes)
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> Vec<f64> {
        // Normalize importances
        let total: f64 = self.importances.iter().sum();
        if total > 0.0 {
            self.importances.iter().map(|&x| x / total).collect()
        } else {
            vec![1.0 / self.n_features as f64; self.n_features]
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

    /// Export tree as text representation
    pub fn export_text(
        &self,
        feature_names: &[String],
        class_names: &[String],
        max_depth: usize,
    ) -> String {
        let mut output = String::new();
        self.export_node(&self.root, feature_names, class_names, 0, max_depth, &mut output);
        output
    }

    fn export_node(
        &self,
        node: &TreeNode,
        feature_names: &[String],
        class_names: &[String],
        depth: usize,
        max_depth: usize,
        output: &mut String,
    ) {
        let indent = "    ".repeat(depth);

        if node.is_leaf() || depth >= max_depth {
            let class_name = class_names.get(node.class)
                .map(|s| s.as_str())
                .unwrap_or("?");
            let total: f64 = node.class_dist.iter().sum();
            let confidence = if total > 0.0 {
                node.class_dist[node.class] / total
            } else {
                0.0
            };
            output.push_str(&format!(
                "{}[{:.0} samples] class: {} (conf: {:.2})\n",
                indent, node.cases, class_name, confidence
            ));
            return;
        }

        let attr = node.tested_attr.unwrap_or(0);
        let attr_name = feature_names.get(attr)
            .map(|s| s.as_str())
            .unwrap_or("?");

        match &node.node_type {
            NodeType::Threshold => {
                let threshold = node.threshold.unwrap_or(0.0);
                output.push_str(&format!("{}{} <= {:.4}:\n", indent, attr_name, threshold));
                if let Some(left) = node.branches.get(0) {
                    self.export_node(left, feature_names, class_names, depth + 1, max_depth, output);
                }
                output.push_str(&format!("{}{} > {:.4}:\n", indent, attr_name, threshold));
                if let Some(right) = node.branches.get(1) {
                    self.export_node(right, feature_names, class_names, depth + 1, max_depth, output);
                }
            }
            NodeType::Discrete => {
                for (i, branch) in node.branches.iter().enumerate() {
                    output.push_str(&format!("{}{} == {}:\n", indent, attr_name, i));
                    self.export_node(branch, feature_names, class_names, depth + 1, max_depth, output);
                }
            }
            NodeType::Subset => {
                if let Some(subsets) = &node.subsets {
                    for (i, (subset, branch)) in subsets.iter().zip(node.branches.iter()).enumerate() {
                        output.push_str(&format!("{}{} in {:?}:\n", indent, attr_name, subset));
                        self.export_node(branch, feature_names, class_names, depth + 1, max_depth, output);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Builder for constructing C5.0 trees
pub struct TreeBuilder<'a> {
    dataset: &'a Dataset,
    n_classes: usize,
    config: TreeConfig,
    importances: Vec<f64>,
}

impl<'a> TreeBuilder<'a> {
    /// Create a new tree builder
    pub fn new(dataset: &'a Dataset, config: TreeConfig) -> Self {
        let n_classes = dataset.classes.iter().max().map(|&m| m + 1).unwrap_or(0);
        let n_features = dataset.n_features;

        TreeBuilder {
            dataset,
            n_classes,
            config,
            importances: vec![0.0; n_features],
        }
    }

    /// Build the complete tree
    pub fn build(&mut self) -> C50Tree {
        let indices: Vec<usize> = (0..self.dataset.n_samples).collect();

        // Build unpruned tree
        let mut root = self.build_node(&indices, 0);

        // Apply pruning
        if self.config.cf > 0.0 {
            let mut pruner = Pruner::new(self.n_classes, self.config.cf);
            pruner.prune(&mut root);

            if self.config.global_pruning {
                pruner.global_prune(&mut root);
            }
        }

        C50Tree {
            root,
            n_features: self.dataset.n_features,
            n_classes: self.n_classes,
            importances: self.importances.clone(),
        }
    }

    /// Build a single node recursively
    fn build_node(&mut self, indices: &[usize], depth: usize) -> TreeNode {
        // Compute class distribution
        let class_dist = self.dataset.class_distribution(indices, self.n_classes);
        let total_weight: f64 = class_dist.iter().sum();

        // Find majority class
        let (best_class, max_count) = class_dist.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &c)| (i, c))
            .unwrap_or((0, 0.0));

        // Check stopping conditions
        if self.should_stop(indices, &class_dist, total_weight, max_count) {
            let errors = total_weight - max_count;
            return TreeNode::leaf(best_class, total_weight, class_dist, errors);
        }

        // Find best split
        let evaluator = SplitEvaluator::new(
            self.dataset,
            indices,
            self.n_classes,
            &self.config,
        );

        let best_split = match evaluator.find_best_split() {
            Some(split) => split,
            None => {
                let errors = total_weight - max_count;
                return TreeNode::leaf(best_class, total_weight, class_dist, errors);
            }
        };

        // Update feature importance
        self.importances[best_split.attribute] += best_split.gain * total_weight;

        // Build child nodes recursively
        let children: Vec<TreeNode> = best_split.branch_indices.iter()
            .map(|child_indices| {
                if child_indices.is_empty() {
                    // Empty branch: create leaf with parent distribution
                    TreeNode::leaf(best_class, 0.0, vec![0.0; self.n_classes], 0.0)
                } else {
                    self.build_node(child_indices, depth + 1)
                }
            })
            .collect();

        // Compute total errors from children
        let child_errors: f64 = children.iter().map(|c| c.errors).sum();

        // Create split node
        match best_split.split_type {
            NodeType::Threshold => {
                TreeNode {
                    node_type: NodeType::Threshold,
                    class: best_class,
                    cases: total_weight,
                    class_dist,
                    errors: child_errors,
                    tested_attr: Some(best_split.attribute),
                    n_forks: 2,
                    threshold: best_split.threshold,
                    soft_bounds: None,
                    subsets: None,
                    branches: children,
                }
            }
            NodeType::Discrete => {
                TreeNode {
                    node_type: NodeType::Discrete,
                    class: best_class,
                    cases: total_weight,
                    class_dist,
                    errors: child_errors,
                    tested_attr: Some(best_split.attribute),
                    n_forks: children.len(),
                    threshold: None,
                    soft_bounds: None,
                    subsets: None,
                    branches: children,
                }
            }
            NodeType::Subset => {
                TreeNode {
                    node_type: NodeType::Subset,
                    class: best_class,
                    cases: total_weight,
                    class_dist,
                    errors: child_errors,
                    tested_attr: Some(best_split.attribute),
                    n_forks: children.len(),
                    threshold: None,
                    soft_bounds: None,
                    subsets: best_split.subsets,
                    branches: children,
                }
            }
            _ => unreachable!(),
        }
    }

    /// Check if we should stop splitting
    fn should_stop(
        &self,
        indices: &[usize],
        _class_dist: &[f64],
        total_weight: f64,
        max_count: f64,
    ) -> bool {
        // Too few cases
        if indices.len() < 2 * self.config.min_cases {
            return true;
        }

        if total_weight < 2.0 * self.config.min_cases as f64 {
            return true;
        }

        // Pure node
        if max_count >= total_weight - 1e-10 {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_tree() {
        // Create simple linearly separable dataset
        let mut ds = Dataset::new(10, 1);

        for i in 0..5 {
            ds.set_value(i, 0, i as f64);
            ds.set_class(i, 0);
        }
        for i in 5..10 {
            ds.set_value(i, 0, i as f64);
            ds.set_class(i, 1);
        }

        let config = TreeConfig::default();
        let mut builder = TreeBuilder::new(&ds, config);
        let tree = builder.build();

        // Should split around 4.5
        assert!(!tree.root.is_leaf());
        assert_eq!(tree.n_classes, 2);

        // Test classification
        assert_eq!(tree.classify(&[2.0]), 0);
        assert_eq!(tree.classify(&[7.0]), 1);
    }

    #[test]
    fn test_pure_node() {
        let mut ds = Dataset::new(5, 1);

        for i in 0..5 {
            ds.set_value(i, 0, i as f64);
            ds.set_class(i, 0); // All same class
        }

        let config = TreeConfig::default();
        let mut builder = TreeBuilder::new(&ds, config);
        let tree = builder.build();

        // Should be a leaf (pure node)
        assert!(tree.root.is_leaf());
        assert_eq!(tree.root.class, 0);
    }
}
