//! Split evaluation for Cubist regression trees
//!
//! Evaluates potential splits based on reduction in variance/error.

use crate::types::*;
use ordered_float::OrderedFloat;

/// Result of evaluating a split
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Attribute to split on
    pub attribute: usize,
    /// Type of split
    pub split_type: BranchType,
    /// Threshold for continuous splits
    pub threshold: Option<f64>,
    /// Subsets for discrete splits
    pub subsets: Option<Vec<std::collections::HashSet<usize>>>,
    /// Indices for each branch
    pub branch_indices: Vec<Vec<usize>>,
    /// Gain from this split (reduction in variance)
    pub gain: f64,
}

/// Evaluator for finding best splits
pub struct SplitEvaluator<'a> {
    dataset: &'a Dataset,
    indices: &'a [usize],
    config: &'a CubistConfig,
}

impl<'a> SplitEvaluator<'a> {
    pub fn new(
        dataset: &'a Dataset,
        indices: &'a [usize],
        config: &'a CubistConfig,
    ) -> Self {
        SplitEvaluator {
            dataset,
            indices,
            config,
        }
    }

    /// Find the best split
    pub fn find_best_split(&self) -> Option<SplitResult> {
        if self.indices.len() < 2 * self.config.min_cases {
            return None;
        }

        // Compute baseline variance
        let (mean, sd, _, _) = self.dataset.subset_stats(self.indices);
        let baseline_var = sd * sd;

        if baseline_var < 1e-10 {
            return None; // No variance to explain
        }

        let mut best_split: Option<SplitResult> = None;
        let mut best_gain = 0.0;

        // Evaluate each attribute
        for attr in 0..self.dataset.n_features {
            let split = if self.dataset.attr_continuous[attr] {
                self.evaluate_continuous_split(attr, baseline_var)
            } else {
                self.evaluate_discrete_split(attr, baseline_var)
            };

            if let Some(s) = split {
                if s.gain > best_gain {
                    best_gain = s.gain;
                    best_split = Some(s);
                }
            }
        }

        best_split
    }

    /// Evaluate a continuous attribute split
    fn evaluate_continuous_split(&self, attr: usize, baseline_var: f64) -> Option<SplitResult> {
        // Gather values
        let mut values: Vec<(f64, usize, f64, f64)> = self.indices.iter()
            .filter_map(|&idx| {
                let v = self.dataset.get_value(idx, attr);
                if v.is_nan() {
                    None
                } else {
                    Some((v, idx, self.dataset.get_target(idx), self.dataset.weights[idx]))
                }
            })
            .collect();

        if values.len() < 2 * self.config.min_cases {
            return None;
        }

        // Sort by value
        values.sort_by_key(|(v, _, _, _)| OrderedFloat(*v));

        // Compute total statistics
        let total_weight: f64 = values.iter().map(|(_, _, _, w)| w).sum();
        let total_sum: f64 = values.iter().map(|(_, _, t, w)| t * w).sum();
        let total_sum_sq: f64 = values.iter().map(|(_, _, t, w)| t * t * w).sum();

        // Running sums for left partition
        let mut left_weight = 0.0;
        let mut left_sum = 0.0;
        let mut left_sum_sq = 0.0;

        let mut best_gain = 0.0;
        let mut best_threshold = 0.0;
        let mut best_idx = 0;

        // Try each split point
        for i in 0..values.len() - 1 {
            let (v, _, t, w) = values[i];
            left_weight += w;
            left_sum += t * w;
            left_sum_sq += t * t * w;

            let right_weight = total_weight - left_weight;
            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;

            // Skip if partitions too small
            if left_weight < self.config.min_cases as f64
                || right_weight < self.config.min_cases as f64
            {
                continue;
            }

            // Skip if next value is same (no split between equal values)
            let next_v = values[i + 1].0;
            if (v - next_v).abs() < 1e-10 {
                continue;
            }

            // Compute variance reduction
            let left_var = (left_sum_sq - left_sum * left_sum / left_weight) / left_weight;
            let right_var = (right_sum_sq - right_sum * right_sum / right_weight) / right_weight;

            let weighted_var = (left_weight * left_var + right_weight * right_var) / total_weight;
            let gain = baseline_var - weighted_var;

            if gain > best_gain {
                best_gain = gain;
                best_threshold = (v + next_v) / 2.0;
                best_idx = i;
            }
        }

        if best_gain < 1e-10 {
            return None;
        }

        // Build branch indices
        let left_indices: Vec<usize> = values[..=best_idx].iter().map(|(_, i, _, _)| *i).collect();
        let right_indices: Vec<usize> = values[best_idx + 1..].iter().map(|(_, i, _, _)| *i).collect();

        // Add cases with missing values (split to both branches proportionally)
        // For simplicity, we send missing to larger branch
        let missing: Vec<usize> = self.indices.iter()
            .filter(|&&idx| self.dataset.get_value(idx, attr).is_nan())
            .copied()
            .collect();

        let mut left = left_indices;
        let mut right = right_indices;

        if left.len() >= right.len() {
            left.extend(missing);
        } else {
            right.extend(missing);
        }

        Some(SplitResult {
            attribute: attr,
            split_type: BranchType::Threshold,
            threshold: Some(best_threshold),
            subsets: None,
            branch_indices: vec![left, right],
            gain: best_gain,
        })
    }

    /// Evaluate a discrete attribute split
    fn evaluate_discrete_split(&self, attr: usize, baseline_var: f64) -> Option<SplitResult> {
        let n_values = self.dataset.attr_n_values[attr];
        if n_values < 2 {
            return None;
        }

        // Group by discrete value
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_values];
        let mut missing = Vec::new();

        for &idx in self.indices {
            let v = self.dataset.get_value(idx, attr);
            if v.is_nan() {
                missing.push(idx);
            } else {
                let vi = v as usize;
                if vi < n_values {
                    groups[vi].push(idx);
                }
            }
        }

        // Compute statistics for each group
        let mut group_stats: Vec<(f64, f64, f64)> = Vec::new(); // (weight, mean, variance)
        for group in &groups {
            if group.is_empty() {
                group_stats.push((0.0, 0.0, 0.0));
            } else {
                let (mean, sd, _, _) = self.dataset.subset_stats(group);
                let weight: f64 = group.iter().map(|&i| self.dataset.weights[i]).sum();
                group_stats.push((weight, mean, sd * sd));
            }
        }

        // For binary splits (2-way), try all subset combinations
        // For multi-way, just do one branch per value
        if n_values == 2 {
            // Binary: simple two-way split
            let total_weight: f64 = group_stats.iter().map(|(w, _, _)| w).sum();
            if total_weight < 2.0 * self.config.min_cases as f64 {
                return None;
            }

            let weighted_var: f64 = group_stats.iter()
                .map(|(w, _, v)| w * v)
                .sum::<f64>() / total_weight;

            let gain = baseline_var - weighted_var;
            if gain < 1e-10 {
                return None;
            }

            let branch_indices: Vec<Vec<usize>> = groups.into_iter()
                .enumerate()
                .map(|(i, mut g)| {
                    // Distribute missing proportionally
                    let ratio = group_stats[i].0 / total_weight;
                    let n_missing = (missing.len() as f64 * ratio) as usize;
                    g.extend(missing.iter().take(n_missing));
                    g
                })
                .filter(|g| !g.is_empty())
                .collect();

            if branch_indices.len() < 2 {
                return None;
            }

            return Some(SplitResult {
                attribute: attr,
                split_type: BranchType::Discrete,
                threshold: None,
                subsets: None,
                branch_indices,
                gain,
            });
        }

        // Multi-way split: try grouping into 2 subsets
        // Use greedy approach: sort by mean and find best split
        let mut sorted_groups: Vec<(usize, f64, f64, Vec<usize>)> = groups.into_iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .map(|(i, g)| (i, group_stats[i].0, group_stats[i].1, g))
            .collect();

        if sorted_groups.len() < 2 {
            return None;
        }

        sorted_groups.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Try each split point
        let mut best_gain = 0.0;
        let mut best_split_idx = 0;

        let total_weight: f64 = sorted_groups.iter().map(|(_, w, _, _)| w).sum();
        let total_sum: f64 = sorted_groups.iter()
            .map(|(_, w, m, _)| w * m)
            .sum();

        let mut left_weight = 0.0;
        let mut left_sum = 0.0;

        for i in 0..sorted_groups.len() - 1 {
            let (_, w, m, _) = &sorted_groups[i];
            left_weight += w;
            left_sum += w * m;

            let right_weight = total_weight - left_weight;
            let right_sum = total_sum - left_sum;

            if left_weight < self.config.min_cases as f64
                || right_weight < self.config.min_cases as f64
            {
                continue;
            }

            let left_mean = left_sum / left_weight;
            let right_mean = right_sum / right_weight;

            // Compute within-group variance
            let mut left_var = 0.0;
            let mut right_var = 0.0;

            for (j, (_, w, m, g)) in sorted_groups.iter().enumerate() {
                let (_, _, v) = group_stats[sorted_groups[j].0];
                if j <= i {
                    left_var += w * (v + (m - left_mean).powi(2));
                } else {
                    right_var += w * (v + (m - right_mean).powi(2));
                }
            }

            let weighted_var = (left_var + right_var) / total_weight;
            let gain = baseline_var - weighted_var;

            if gain > best_gain {
                best_gain = gain;
                best_split_idx = i;
            }
        }

        if best_gain < 1e-10 {
            return None;
        }

        // Build subsets
        let left_values: std::collections::HashSet<usize> = sorted_groups[..=best_split_idx]
            .iter()
            .map(|(i, _, _, _)| *i)
            .collect();
        let right_values: std::collections::HashSet<usize> = sorted_groups[best_split_idx + 1..]
            .iter()
            .map(|(i, _, _, _)| *i)
            .collect();

        let mut left_indices: Vec<usize> = sorted_groups[..=best_split_idx]
            .iter()
            .flat_map(|(_, _, _, g)| g.iter().copied())
            .collect();
        let mut right_indices: Vec<usize> = sorted_groups[best_split_idx + 1..]
            .iter()
            .flat_map(|(_, _, _, g)| g.iter().copied())
            .collect();

        // Distribute missing
        if left_indices.len() >= right_indices.len() {
            left_indices.extend(missing);
        } else {
            right_indices.extend(missing);
        }

        Some(SplitResult {
            attribute: attr,
            split_type: BranchType::Subset,
            threshold: None,
            subsets: Some(vec![left_values, right_values]),
            branch_indices: vec![left_indices, right_indices],
            gain: best_gain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_split() {
        let mut ds = Dataset::new(10, 1);
        for i in 0..5 {
            ds.set_value(i, 0, i as f64);
            ds.set_target(i, 10.0);
        }
        for i in 5..10 {
            ds.set_value(i, 0, i as f64);
            ds.set_target(i, 20.0);
        }

        let indices: Vec<usize> = (0..10).collect();
        let config = CubistConfig::default();
        let evaluator = SplitEvaluator::new(&ds, &indices, &config);

        let split = evaluator.find_best_split();
        assert!(split.is_some());

        let s = split.unwrap();
        assert_eq!(s.attribute, 0);
        assert!(s.threshold.unwrap() > 4.0 && s.threshold.unwrap() < 5.0);
    }
}
