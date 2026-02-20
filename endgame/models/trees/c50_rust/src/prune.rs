//! Pruning algorithms for C5.0
//!
//! Implements both local (pessimistic error) and global (cost-complexity) pruning.

use crate::types::*;

/// Pruner for C5.0 trees
pub struct Pruner {
    n_classes: usize,
    /// Confidence factor (default 0.25)
    cf: f64,
    /// Z-value for confidence level
    z: f64,
}

impl Pruner {
    /// Create a new pruner
    pub fn new(n_classes: usize, cf: f64) -> Self {
        // Convert confidence level to z-value
        // For cf=0.25, z ≈ 0.6745 (upper 25% of standard normal)
        let z = Self::compute_z(cf);

        Pruner {
            n_classes,
            cf,
            z,
        }
    }

    /// Compute z-value from confidence factor using inverse normal approximation
    fn compute_z(cf: f64) -> f64 {
        // For confidence factor cf, we want the z-value such that
        // P(Z > z) = cf, i.e., z = Φ⁻¹(1 - cf)
        // The Abramowitz & Stegun formula works for p < 0.5
        // For p > 0.5, use: Φ⁻¹(p) = -Φ⁻¹(1-p)

        let p = 1.0 - cf;  // e.g., for cf=0.25, p=0.75

        // Handle the case where p > 0.5 by using symmetry
        let (adjusted_p, sign) = if p > 0.5 {
            (1.0 - p, 1.0)
        } else {
            (p, -1.0)
        };

        // Rational approximation (Abramowitz and Stegun 26.2.23)
        // Valid for 0 < p < 0.5
        let t = (-2.0 * adjusted_p.ln()).sqrt();
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        sign * z
    }

    /// Compute extra errors for pruning decision
    /// Uses continuity-corrected binomial estimate
    fn extra_errors(&self, n: f64, e: f64) -> f64 {
        if n <= 0.0 {
            return 0.0;
        }

        if e < 1e-6 {
            // Very few errors: use approximate formula
            return n * (1.0 - self.cf.powf(1.0 / n));
        }

        if e + 0.5 >= n {
            // Almost all errors
            return 0.67 * (n - e);
        }

        // Standard case: continuity-corrected Wilson score interval
        let z2 = self.z * self.z;
        let f = (e + 0.5) / n;

        // Upper confidence bound for error rate
        let pr = (f + z2 / (2.0 * n)
            + self.z * ((f * (1.0 - f) / n + z2 / (4.0 * n * n)).sqrt()))
            / (1.0 + z2 / n);

        n * pr - e
    }

    /// Estimate errors at a node (leaf or subtree)
    fn estimate_errors(&self, node: &TreeNode) -> f64 {
        if node.is_leaf() {
            let leaf_errors = node.cases - node.class_dist[node.class];
            leaf_errors + self.extra_errors(node.cases, leaf_errors)
        } else {
            // Sum of child errors (already estimated)
            node.branches.iter().map(|b| b.errors).sum()
        }
    }

    /// Perform local (bottom-up) pruning
    pub fn prune(&mut self, node: &mut TreeNode) {
        if node.is_leaf() {
            // Update error estimate for leaf
            let leaf_errors = node.cases - node.class_dist[node.class];
            node.errors = leaf_errors + self.extra_errors(node.cases, leaf_errors);
            return;
        }

        // Recursively prune children first
        for child in &mut node.branches {
            self.prune(child);
        }

        // Compute tree errors (sum of pruned children)
        let tree_errors: f64 = node.branches.iter().map(|b| b.errors).sum();

        // Compute leaf errors if we collapse to leaf
        let leaf_errors = node.cases - node.class_dist[node.class];
        let leaf_estimated = leaf_errors + self.extra_errors(node.cases, leaf_errors);

        // Find best branch (for potential replacement)
        let best_branch_idx = node.branches.iter()
            .enumerate()
            .filter(|(_, b)| b.cases >= node.cases * 0.1) // Significant size
            .min_by(|(_, a), (_, b)| {
                a.errors.partial_cmp(&b.errors).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let best_branch_errors = best_branch_idx
            .map(|i| node.branches[i].errors)
            .unwrap_or(f64::INFINITY);

        // Decision: keep tree, collapse to leaf, or replace with branch
        if leaf_estimated <= tree_errors + 0.1 && leaf_estimated <= best_branch_errors + 0.1 {
            // Collapse to leaf
            self.collapse_to_leaf(node);
        } else if best_branch_errors < tree_errors - 0.1
            && best_branch_errors < leaf_estimated - 0.1 {
            // Replace with best branch (only if not testing same continuous attribute)
            // AND only if the best branch is NOT a leaf (replacing with leaf is same as collapsing)
            if let Some(idx) = best_branch_idx {
                let branch_is_leaf = node.branches[idx].is_leaf();

                if !branch_is_leaf {
                    let should_replace = match (&node.node_type, &node.branches[idx].node_type) {
                        (NodeType::Threshold, NodeType::Threshold) => {
                            // Don't replace if testing same attribute
                            node.tested_attr != node.branches[idx].tested_attr
                        }
                        _ => true,
                    };

                    if should_replace {
                        self.replace_with_branch(node, idx);
                    }
                }
            }
        }
        // Otherwise keep as tree

        // Update node errors
        if node.is_leaf() {
            let leaf_errors = node.cases - node.class_dist[node.class];
            node.errors = leaf_errors + self.extra_errors(node.cases, leaf_errors);
        } else {
            node.errors = node.branches.iter().map(|b| b.errors).sum();
        }
    }

    /// Collapse a node to a leaf
    fn collapse_to_leaf(&self, node: &mut TreeNode) {
        node.node_type = NodeType::Leaf;
        node.tested_attr = None;
        node.n_forks = 0;
        node.threshold = None;
        node.soft_bounds = None;
        node.subsets = None;
        node.branches.clear();
    }

    /// Replace a node with one of its branches
    fn replace_with_branch(&self, node: &mut TreeNode, branch_idx: usize) {
        let branch = std::mem::replace(
            &mut node.branches[branch_idx],
            TreeNode::leaf(0, 0.0, vec![], 0.0)
        );

        // Copy branch properties to node
        node.node_type = branch.node_type;
        node.class = branch.class;
        // Keep original cases and class_dist for this node level
        node.tested_attr = branch.tested_attr;
        node.n_forks = branch.n_forks;
        node.threshold = branch.threshold;
        node.soft_bounds = branch.soft_bounds;
        node.subsets = branch.subsets;
        node.branches = branch.branches;
        node.errors = branch.errors;
    }

    /// Perform global (cost-complexity) pruning
    pub fn global_prune(&mut self, root: &mut TreeNode) {
        if root.is_leaf() {
            return;
        }

        // Calculate base error rate
        let base_errors = root.errors;
        let total_cases = root.cases;

        if total_cases <= 0.0 {
            return;
        }

        // Maximum extra errors allowed (1 SE rule)
        let error_rate = base_errors / total_cases;
        let max_extra_errs = (error_rate * (1.0 - error_rate) / total_cases).sqrt() * total_cases;

        let mut total_extra = 0.0;

        loop {
            if root.is_leaf() {
                break;
            }

            // Find all subtrees and their cost-complexity
            let mut candidates: Vec<(Vec<usize>, f64)> = Vec::new();
            self.find_prune_candidates(root, &mut vec![], &mut candidates);

            if candidates.is_empty() {
                break;
            }

            // Find minimum cost-complexity subtree
            let min_cc = candidates.iter()
                .map(|(_, cc)| *cc)
                .fold(f64::INFINITY, f64::min);

            // Find all subtrees with minimum CC
            let to_prune: Vec<Vec<usize>> = candidates.into_iter()
                .filter(|(_, cc)| (*cc - min_cc).abs() < 1e-10)
                .map(|(path, _)| path)
                .collect();

            // Check if pruning would exceed error budget
            let mut extra_if_pruned = 0.0;
            for path in &to_prune {
                if let Some(node) = self.get_node(root, path) {
                    let leaf_errors = node.cases - node.class_dist[node.class];
                    let leaf_estimated = leaf_errors + self.extra_errors(node.cases, leaf_errors);
                    extra_if_pruned += leaf_estimated - node.errors;
                }
            }

            if total_extra + extra_if_pruned > max_extra_errs {
                break;
            }

            // Prune the subtrees
            for path in to_prune {
                if let Some(node) = self.get_node_mut(root, &path) {
                    self.collapse_to_leaf(node);
                    let leaf_errors = node.cases - node.class_dist[node.class];
                    node.errors = leaf_errors + self.extra_errors(node.cases, leaf_errors);
                }
            }

            total_extra += extra_if_pruned;

            // Update errors throughout tree
            self.update_errors(root);
        }
    }

    /// Find all subtrees and their cost-complexity values
    fn find_prune_candidates(
        &self,
        node: &TreeNode,
        path: &mut Vec<usize>,
        candidates: &mut Vec<(Vec<usize>, f64)>,
    ) {
        if node.is_leaf() {
            return;
        }

        // Compute cost-complexity for this subtree
        let n_leaves = node.n_leaves();
        if n_leaves > 1 {
            let leaf_errors = node.cases - node.class_dist[node.class];
            let leaf_estimated = leaf_errors + self.extra_errors(node.cases, leaf_errors);
            let extra_errs = leaf_estimated - node.errors;
            let cc = extra_errs / (n_leaves - 1) as f64;

            candidates.push((path.clone(), cc));
        }

        // Recurse into children
        for (i, child) in node.branches.iter().enumerate() {
            path.push(i);
            self.find_prune_candidates(child, path, candidates);
            path.pop();
        }
    }

    /// Get node at path
    fn get_node<'a>(&self, root: &'a TreeNode, path: &[usize]) -> Option<&'a TreeNode> {
        let mut current = root;
        for &idx in path {
            current = current.branches.get(idx)?;
        }
        Some(current)
    }

    /// Get mutable node at path
    fn get_node_mut<'a>(&self, root: &'a mut TreeNode, path: &[usize]) -> Option<&'a mut TreeNode> {
        let mut current = root;
        for &idx in path {
            current = current.branches.get_mut(idx)?;
        }
        Some(current)
    }

    /// Update errors throughout tree after pruning
    fn update_errors(&self, node: &mut TreeNode) {
        if node.is_leaf() {
            let leaf_errors = node.cases - node.class_dist[node.class];
            node.errors = leaf_errors + self.extra_errors(node.cases, leaf_errors);
        } else {
            for child in &mut node.branches {
                self.update_errors(child);
            }
            node.errors = node.branches.iter().map(|b| b.errors).sum();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extra_errors() {
        let pruner = Pruner::new(2, 0.25);

        // No errors
        let e = pruner.extra_errors(100.0, 0.0);
        assert!(e > 0.0 && e < 10.0);

        // Half errors
        let e = pruner.extra_errors(100.0, 50.0);
        assert!(e > 0.0);

        // All errors
        let e = pruner.extra_errors(100.0, 100.0);
        assert!(e >= 0.0);
    }

    #[test]
    fn test_collapse_to_leaf() {
        let pruner = Pruner::new(2, 0.25);

        let mut node = TreeNode {
            node_type: NodeType::Threshold,
            class: 0,
            cases: 100.0,
            class_dist: vec![60.0, 40.0],
            errors: 40.0,
            tested_attr: Some(0),
            n_forks: 2,
            threshold: Some(0.5),
            soft_bounds: None,
            subsets: None,
            branches: vec![
                TreeNode::leaf(0, 50.0, vec![50.0, 0.0], 0.0),
                TreeNode::leaf(1, 50.0, vec![10.0, 40.0], 10.0),
            ],
        };

        pruner.collapse_to_leaf(&mut node);

        assert!(node.is_leaf());
        assert!(node.branches.is_empty());
        assert_eq!(node.class, 0);
    }
}
