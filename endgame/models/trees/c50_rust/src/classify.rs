//! Classification functions for C5.0
//!
//! Handles prediction with support for missing values and soft thresholds.

use crate::types::*;

/// Classify a sample, returning the predicted class
pub fn classify_sample(root: &TreeNode, sample: &[f64], n_classes: usize) -> usize {
    let proba = classify_proba(root, sample, n_classes);

    // Return class with highest probability
    proba.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(root.class)
}

/// Classify a sample, returning class probabilities
pub fn classify_proba(root: &TreeNode, sample: &[f64], n_classes: usize) -> Vec<f64> {
    let mut class_sum = vec![0.0; n_classes];

    // Traverse tree, accumulating class distributions
    find_leaf(root, sample, 1.0, &mut class_sum);

    // Normalize to probabilities
    let total: f64 = class_sum.iter().sum();
    if total > 0.0 {
        for p in &mut class_sum {
            *p /= total;
        }
    } else {
        // Fallback to root distribution
        let root_total: f64 = root.class_dist.iter().sum();
        if root_total > 0.0 {
            for (i, &c) in root.class_dist.iter().enumerate() {
                if i < class_sum.len() {
                    class_sum[i] = c / root_total;
                }
            }
        }
    }

    class_sum
}

/// Recursively traverse tree to find leaf, handling missing values
fn find_leaf(
    node: &TreeNode,
    sample: &[f64],
    weight: f64,
    class_sum: &mut [f64],
) {
    if node.is_leaf() {
        // Add weighted class distribution
        let total: f64 = node.class_dist.iter().sum();
        if total > 0.0 {
            for (i, &c) in node.class_dist.iter().enumerate() {
                if i < class_sum.len() {
                    class_sum[i] += weight * c / total;
                }
            }
        }
        return;
    }

    let attr = match node.tested_attr {
        Some(a) => a,
        None => {
            // Shouldn't happen, but fallback to leaf behavior
            for (i, &c) in node.class_dist.iter().enumerate() {
                if i < class_sum.len() {
                    class_sum[i] += weight * c / node.cases.max(1.0);
                }
            }
            return;
        }
    };

    let value = if attr < sample.len() { sample[attr] } else { UNKNOWN };

    match &node.node_type {
        NodeType::Threshold => {
            let threshold = node.threshold.unwrap_or(0.0);
            handle_threshold_split(node, sample, value, threshold, weight, class_sum);
        }
        NodeType::Discrete => {
            handle_discrete_split(node, sample, value, weight, class_sum);
        }
        NodeType::Subset => {
            handle_subset_split(node, sample, value, weight, class_sum);
        }
        _ => {}
    }
}

/// Handle threshold (continuous) split during classification
fn handle_threshold_split(
    node: &TreeNode,
    sample: &[f64],
    value: f64,
    threshold: f64,
    weight: f64,
    class_sum: &mut [f64],
) {
    if node.branches.len() < 2 {
        return;
    }

    if is_unknown(value) {
        // Unknown: distribute proportionally to branch weights
        let left_weight = node.branches[0].cases;
        let right_weight = node.branches[1].cases;
        let total = left_weight + right_weight;

        if total > 0.0 {
            find_leaf(&node.branches[0], sample, weight * left_weight / total, class_sum);
            find_leaf(&node.branches[1], sample, weight * right_weight / total, class_sum);
        }
    } else if let Some((lower, mid, upper)) = node.soft_bounds {
        // Soft threshold: interpolate
        let frac = interpolate_soft(value, lower, mid, upper);
        find_leaf(&node.branches[0], sample, weight * frac, class_sum);
        find_leaf(&node.branches[1], sample, weight * (1.0 - frac), class_sum);
    } else {
        // Hard threshold
        if value <= threshold {
            find_leaf(&node.branches[0], sample, weight, class_sum);
        } else {
            find_leaf(&node.branches[1], sample, weight, class_sum);
        }
    }
}

/// Interpolate soft threshold
fn interpolate_soft(value: f64, lower: f64, mid: f64, upper: f64) -> f64 {
    if value <= lower {
        1.0
    } else if value >= upper {
        0.0
    } else if value <= mid {
        1.0 - 0.5 * (value - lower) / (mid - lower).max(1e-10)
    } else {
        0.5 - 0.5 * (value - mid) / (upper - mid).max(1e-10)
    }
}

/// Handle discrete split during classification
fn handle_discrete_split(
    node: &TreeNode,
    sample: &[f64],
    value: f64,
    weight: f64,
    class_sum: &mut [f64],
) {
    if is_unknown(value) || is_na(value) {
        // Unknown: distribute proportionally
        let total: f64 = node.branches.iter().map(|b| b.cases).sum();
        if total > 0.0 {
            for branch in &node.branches {
                find_leaf(branch, sample, weight * branch.cases / total, class_sum);
            }
        }
    } else {
        // Find matching branch by value index
        let val_idx = value as usize;
        if val_idx < node.branches.len() {
            find_leaf(&node.branches[val_idx], sample, weight, class_sum);
        } else {
            // Value not seen in training: use largest branch
            if let Some(best) = node.branches.iter()
                .max_by(|a, b| a.cases.partial_cmp(&b.cases).unwrap_or(std::cmp::Ordering::Equal))
            {
                find_leaf(best, sample, weight, class_sum);
            }
        }
    }
}

/// Handle subset split during classification
fn handle_subset_split(
    node: &TreeNode,
    sample: &[f64],
    value: f64,
    weight: f64,
    class_sum: &mut [f64],
) {
    if is_unknown(value) || is_na(value) {
        // Unknown: distribute proportionally
        let total: f64 = node.branches.iter().map(|b| b.cases).sum();
        if total > 0.0 {
            for branch in &node.branches {
                find_leaf(branch, sample, weight * branch.cases / total, class_sum);
            }
        }
        return;
    }

    let val_idx = value as usize;

    // Find which subset contains this value
    if let Some(subsets) = &node.subsets {
        for (i, subset) in subsets.iter().enumerate() {
            if subset.contains(&val_idx) {
                if i < node.branches.len() {
                    find_leaf(&node.branches[i], sample, weight, class_sum);
                }
                return;
            }
        }
    }

    // Value not found in any subset: use largest branch
    if let Some(best) = node.branches.iter()
        .max_by(|a, b| a.cases.partial_cmp(&b.cases).unwrap_or(std::cmp::Ordering::Equal))
    {
        find_leaf(best, sample, weight, class_sum);
    }
}

/// Batch classification for efficiency
pub fn classify_batch(
    root: &TreeNode,
    samples: &[Vec<f64>],
    n_classes: usize,
) -> Vec<usize> {
    samples.iter()
        .map(|s| classify_sample(root, s, n_classes))
        .collect()
}

/// Batch probability prediction
pub fn classify_proba_batch(
    root: &TreeNode,
    samples: &[Vec<f64>],
    n_classes: usize,
) -> Vec<Vec<f64>> {
    samples.iter()
        .map(|s| classify_proba(root, s, n_classes))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tree() -> TreeNode {
        // Simple tree: if x <= 5 then class 0, else class 1
        TreeNode {
            node_type: NodeType::Threshold,
            class: 0,
            cases: 100.0,
            class_dist: vec![50.0, 50.0],
            errors: 0.0,
            tested_attr: Some(0),
            n_forks: 2,
            threshold: Some(5.0),
            soft_bounds: None,
            subsets: None,
            branches: vec![
                TreeNode::leaf(0, 50.0, vec![50.0, 0.0], 0.0),
                TreeNode::leaf(1, 50.0, vec![0.0, 50.0], 0.0),
            ],
        }
    }

    #[test]
    fn test_classify() {
        let tree = create_test_tree();

        assert_eq!(classify_sample(&tree, &[3.0], 2), 0);
        assert_eq!(classify_sample(&tree, &[7.0], 2), 1);
        assert_eq!(classify_sample(&tree, &[5.0], 2), 0); // <= threshold
    }

    #[test]
    fn test_classify_proba() {
        let tree = create_test_tree();

        let proba = classify_proba(&tree, &[3.0], 2);
        assert!((proba[0] - 1.0).abs() < 1e-10);
        assert!((proba[1] - 0.0).abs() < 1e-10);

        let proba = classify_proba(&tree, &[7.0], 2);
        assert!((proba[0] - 0.0).abs() < 1e-10);
        assert!((proba[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classify_unknown() {
        let tree = create_test_tree();

        // Unknown value should distribute proportionally
        let proba = classify_proba(&tree, &[f64::NAN], 2);
        assert!((proba[0] - 0.5).abs() < 1e-10);
        assert!((proba[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_soft_interpolation() {
        assert!((interpolate_soft(0.0, 0.0, 5.0, 10.0) - 1.0).abs() < 1e-10);
        assert!((interpolate_soft(5.0, 0.0, 5.0, 10.0) - 0.5).abs() < 1e-10);
        assert!((interpolate_soft(10.0, 0.0, 5.0, 10.0) - 0.0).abs() < 1e-10);
    }
}
