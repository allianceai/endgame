//! Split evaluation for C5.0
//!
//! Implements split finding for continuous, discrete, and subset attributes.

use crate::types::*;
use crate::info::*;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

/// Evaluate all possible splits for a dataset subset
pub struct SplitEvaluator<'a> {
    dataset: &'a Dataset,
    indices: &'a [usize],
    n_classes: usize,
    config: &'a TreeConfig,
    /// Base information (entropy before any split)
    base_info: f64,
    /// Class distribution at this node
    class_dist: Vec<f64>,
    /// Total weight at this node
    total_weight: f64,
}

impl<'a> SplitEvaluator<'a> {
    /// Create a new split evaluator
    pub fn new(
        dataset: &'a Dataset,
        indices: &'a [usize],
        n_classes: usize,
        config: &'a TreeConfig,
    ) -> Self {
        // Compute class distribution
        let class_dist = dataset.class_distribution(indices, n_classes);
        let total_weight: f64 = class_dist.iter().sum();
        let base_info = entropy(&class_dist);

        SplitEvaluator {
            dataset,
            indices,
            n_classes,
            config,
            base_info,
            class_dist,
            total_weight,
        }
    }

    /// Find the best split across all attributes
    pub fn find_best_split(&self) -> Option<SplitResult> {
        let n_features = self.dataset.n_features;

        // Evaluate all attributes (potentially in parallel)
        let results: Vec<Option<SplitResult>> = if self.config.n_threads > 1 && n_features > 4 {
            (0..n_features)
                .into_par_iter()
                .map(|attr| self.evaluate_attribute(attr))
                .collect()
        } else {
            (0..n_features)
                .map(|attr| self.evaluate_attribute(attr))
                .collect()
        };

        // Find best by gain ratio
        results
            .into_iter()
            .flatten()
            .filter(|r| r.is_valid())
            .max_by(|a, b| {
                a.gain_ratio.partial_cmp(&b.gain_ratio).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Evaluate a single attribute for splitting
    fn evaluate_attribute(&self, attr: usize) -> Option<SplitResult> {
        if self.dataset.is_categorical(attr) {
            if self.config.use_subset {
                self.evaluate_subset_split(attr)
            } else {
                self.evaluate_discrete_split(attr)
            }
        } else {
            self.evaluate_continuous_split(attr)
        }
    }

    /// Evaluate continuous attribute for threshold split
    fn evaluate_continuous_split(&self, attr: usize) -> Option<SplitResult> {
        // Collect and sort valid values
        let mut records: Vec<SortRec> = self.indices
            .iter()
            .filter_map(|&i| {
                let v = self.dataset.get_value(i, attr);
                if is_valid(v) {
                    Some(SortRec::new(
                        i,
                        v,
                        self.dataset.get_weight(i),
                        self.dataset.get_class(i),
                    ))
                } else {
                    None
                }
            })
            .collect();

        if records.len() < 2 * self.config.min_cases {
            return None;
        }

        // Sort by value
        records.sort_by(|a, b| {
            OrderedFloat(a.value).cmp(&OrderedFloat(b.value))
        });

        let known_weight: f64 = records.iter().map(|r| r.weight).sum();
        let unknown_weight = self.total_weight - known_weight;
        let unknown_frac = unknown_weight / self.total_weight;

        // Compute minimum cases per branch
        let min_split = self.config.min_cases.max(
            ((0.1 * known_weight / self.n_classes as f64) as usize).min(25)
        );

        // Initialize cumulative class counts for left branch
        let mut left_counts = vec![0.0; self.n_classes];
        let mut left_weight = 0.0;

        // Right branch starts with all known cases
        let mut right_counts: Vec<f64> = records.iter()
            .fold(vec![0.0; self.n_classes], |mut acc, r| {
                acc[r.class] += r.weight;
                acc
            });
        let mut right_weight = known_weight;

        let mut best_gain = 0.0;
        let mut best_threshold = 0.0;
        let mut best_index = 0;
        let mut n_cuts = 0;
        let mut first_value = records[0].value;
        let mut last_value = records[0].value;

        // Scan through possible cut points
        for i in 0..records.len() - 1 {
            let r = &records[i];

            // Move this case from right to left
            left_counts[r.class] += r.weight;
            left_weight += r.weight;
            right_counts[r.class] -= r.weight;
            right_weight -= r.weight;

            last_value = r.value;

            // Skip if same value as next (no valid cut here)
            if (r.value - records[i + 1].value).abs() < 1e-10 {
                continue;
            }

            // Check minimum cases constraint
            if left_weight < min_split as f64 || right_weight < min_split as f64 {
                continue;
            }

            n_cuts += 1;

            // Compute gain for this split
            let branch_counts = vec![left_counts.clone(), right_counts.clone()];
            let gain = compute_gain(
                self.base_info,
                unknown_frac,
                &branch_counts,
                known_weight,
            );

            if gain > best_gain {
                best_gain = gain;
                best_threshold = (r.value + records[i + 1].value) / 2.0;
                best_index = i;
            }
        }

        if best_gain <= 0.0 || n_cuts == 0 {
            return None;
        }

        // Apply threshold penalty
        let interval_ratio = if last_value > first_value {
            (last_value - first_value) / (best_threshold - first_value).max(last_value - best_threshold)
        } else {
            1.0
        };
        let penalty = threshold_penalty(n_cuts, interval_ratio, known_weight);
        let adjusted_gain = best_gain - penalty;

        if adjusted_gain <= 0.0 {
            return None;
        }

        // Compute split info
        let branch_weights = vec![left_weight, right_weight];
        let split_info = split_info(&branch_weights);
        let gain_ratio = gain_ratio(adjusted_gain, split_info);

        // Build branch indices
        let mut left_indices = Vec::with_capacity(best_index + 1);
        let mut right_indices = Vec::with_capacity(records.len() - best_index - 1);
        let mut unknown_indices = Vec::new();

        for &idx in self.indices {
            let v = self.dataset.get_value(idx, attr);
            if is_valid(v) {
                if v <= best_threshold {
                    left_indices.push(idx);
                } else {
                    right_indices.push(idx);
                }
            } else {
                unknown_indices.push(idx);
            }
        }

        // Distribute unknown cases proportionally
        let left_frac = left_weight / known_weight;
        for &idx in &unknown_indices {
            // For simplicity, assign to larger branch (proper implementation would split weight)
            if left_frac >= 0.5 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Recompute class distributions for branches
        let left_dist = self.dataset.class_distribution(&left_indices, self.n_classes);
        let right_dist = self.dataset.class_distribution(&right_indices, self.n_classes);

        Some(SplitResult {
            attribute: attr,
            split_type: NodeType::Threshold,
            gain: adjusted_gain,
            gain_ratio,
            threshold: Some(best_threshold),
            subsets: None,
            branch_cases: vec![
                left_indices.iter().map(|&i| self.dataset.get_weight(i)).sum(),
                right_indices.iter().map(|&i| self.dataset.get_weight(i)).sum(),
            ],
            branch_class_dist: vec![left_dist, right_dist],
            branch_indices: vec![left_indices, right_indices],
        })
    }

    /// Evaluate discrete attribute for standard multi-way split
    fn evaluate_discrete_split(&self, attr: usize) -> Option<SplitResult> {
        // Count unique values
        let mut value_map: hashbrown::HashMap<OrderedFloat<f64>, usize> = hashbrown::HashMap::new();
        let mut value_list: Vec<f64> = Vec::new();

        for &i in self.indices {
            let v = self.dataset.get_value(i, attr);
            if is_valid(v) {
                let key = OrderedFloat(v);
                if !value_map.contains_key(&key) {
                    value_map.insert(key, value_list.len());
                    value_list.push(v);
                }
            }
        }

        let n_values = value_list.len();
        if n_values < 2 {
            return None;
        }

        // Build frequency table
        let mut freq_table = FrequencyTable::new(n_values, self.n_classes);
        let mut unknown_weight = 0.0;

        for &i in self.indices {
            let v = self.dataset.get_value(i, attr);
            let w = self.dataset.get_weight(i);
            let c = self.dataset.get_class(i);

            if is_valid(v) {
                let val_idx = value_map[&OrderedFloat(v)];
                freq_table.add(val_idx, c, w);
            } else {
                unknown_weight += w;
            }
        }

        // Check minimum cases per branch
        let valid_branches: Vec<usize> = (0..n_values)
            .filter(|&v| freq_table.val_freq[v] >= self.config.min_cases as f64)
            .collect();

        if valid_branches.len() < 2 {
            return None;
        }

        // Compute gain
        let unknown_frac = unknown_weight / self.total_weight;
        let gain = (1.0 - unknown_frac) * freq_table.info_gain();

        if gain <= 0.0 {
            return None;
        }

        let split_info_val = freq_table.split_info();
        let gain_ratio = gain_ratio(gain, split_info_val);

        // Build branch indices
        let mut branch_indices: Vec<Vec<usize>> = vec![Vec::new(); n_values];
        let mut unknown_indices = Vec::new();

        for &i in self.indices {
            let v = self.dataset.get_value(i, attr);
            if is_valid(v) {
                let val_idx = value_map[&OrderedFloat(v)];
                branch_indices[val_idx].push(i);
            } else {
                unknown_indices.push(i);
            }
        }

        // Distribute unknowns proportionally to largest branches
        if !unknown_indices.is_empty() {
            let max_branch = (0..n_values)
                .max_by(|&a, &b| {
                    freq_table.val_freq[a].partial_cmp(&freq_table.val_freq[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            branch_indices[max_branch].extend(unknown_indices);
        }

        // Compute branch stats
        let branch_cases: Vec<f64> = branch_indices.iter()
            .map(|indices| indices.iter().map(|&i| self.dataset.get_weight(i)).sum())
            .collect();
        let branch_class_dist: Vec<Vec<f64>> = branch_indices.iter()
            .map(|indices| self.dataset.class_distribution(indices, self.n_classes))
            .collect();

        Some(SplitResult {
            attribute: attr,
            split_type: NodeType::Discrete,
            gain,
            gain_ratio,
            threshold: None,
            subsets: None,
            branch_cases,
            branch_class_dist,
            branch_indices,
        })
    }

    /// Evaluate discrete attribute for subset split (greedy agglomerative)
    fn evaluate_subset_split(&self, attr: usize) -> Option<SplitResult> {
        // Map values to indices
        let mut value_map: hashbrown::HashMap<OrderedFloat<f64>, usize> = hashbrown::HashMap::new();
        let mut value_list: Vec<f64> = Vec::new();

        for &i in self.indices {
            let v = self.dataset.get_value(i, attr);
            if is_valid(v) {
                let key = OrderedFloat(v);
                if !value_map.contains_key(&key) {
                    value_map.insert(key, value_list.len());
                    value_list.push(v);
                }
            }
        }

        let n_values = value_list.len();
        if n_values < 2 {
            return None;
        }

        // Build frequency table
        let mut freq: Vec<Vec<f64>> = vec![vec![0.0; self.n_classes]; n_values];
        let mut val_freq: Vec<f64> = vec![0.0; n_values];
        let mut unknown_weight = 0.0;

        for &i in self.indices {
            let v = self.dataset.get_value(i, attr);
            let w = self.dataset.get_weight(i);
            let c = self.dataset.get_class(i);

            if is_valid(v) {
                let val_idx = value_map[&OrderedFloat(v)];
                freq[val_idx][c] += w;
                val_freq[val_idx] += w;
            } else {
                unknown_weight += w;
            }
        }

        let known_weight: f64 = val_freq.iter().sum();
        let unknown_frac = unknown_weight / self.total_weight;

        // Initialize: each value in its own subset
        let mut subsets: Vec<Vec<usize>> = (0..n_values).map(|v| vec![v]).collect();
        let mut subset_freq: Vec<Vec<f64>> = freq.clone();
        let mut subset_weight: Vec<f64> = val_freq.clone();
        let initial_blocks = n_values;

        // Greedy merge: combine subsets that maximize gain
        loop {
            let n_subsets = subsets.len();
            if n_subsets <= 2 {
                break;
            }

            // Find best pair to merge
            let mut best_merge: Option<(usize, usize, f64)> = None;
            let mut best_merged_gain = 0.0;

            for i in 0..n_subsets {
                for j in i + 1..n_subsets {
                    // Compute gain if we merge i and j
                    let mut merged_freq = subset_freq[i].clone();
                    for (c, &f) in subset_freq[j].iter().enumerate() {
                        merged_freq[c] += f;
                    }
                    let merged_weight = subset_weight[i] + subset_weight[j];

                    // Build temporary subset configuration
                    let mut temp_freq: Vec<Vec<f64>> = Vec::new();
                    let mut temp_weights: Vec<f64> = Vec::new();

                    for k in 0..n_subsets {
                        if k == i {
                            temp_freq.push(merged_freq.clone());
                            temp_weights.push(merged_weight);
                        } else if k != j {
                            temp_freq.push(subset_freq[k].clone());
                            temp_weights.push(subset_weight[k]);
                        }
                    }

                    // Compute gain with this configuration
                    let gain = compute_gain(
                        self.base_info,
                        unknown_frac,
                        &temp_freq,
                        known_weight,
                    );

                    // Apply penalty
                    let n_final = temp_freq.len();
                    let penalty = subset_penalty(initial_blocks, n_final, known_weight);
                    let adjusted_gain = gain - penalty;

                    // Compute gain ratio
                    let si = split_info(&temp_weights);
                    let gr = if si > 0.0 { adjusted_gain / si } else { 0.0 };

                    if gr > best_merged_gain {
                        best_merged_gain = gr;
                        best_merge = Some((i, j, gr));
                    }
                }
            }

            // Check if merging improves things
            let current_gain = compute_gain(
                self.base_info,
                unknown_frac,
                &subset_freq,
                known_weight,
            );
            let current_penalty = subset_penalty(initial_blocks, subsets.len(), known_weight);
            let current_adjusted = current_gain - current_penalty;
            let current_si = split_info(&subset_weight);
            let current_gr = if current_si > 0.0 { current_adjusted / current_si } else { 0.0 };

            match best_merge {
                Some((i, j, gr)) if gr >= current_gr * 0.999 => {
                    // Merge subsets i and j
                    let mut merged_subset = subsets[i].clone();
                    merged_subset.extend(&subsets[j]);
                    subsets[i] = merged_subset;
                    subsets.remove(j);

                    // Update frequencies
                    for c in 0..self.n_classes {
                        subset_freq[i][c] += subset_freq[j][c];
                    }
                    subset_weight[i] += subset_weight[j];
                    subset_freq.remove(j);
                    subset_weight.remove(j);
                }
                _ => break,
            }
        }

        if subsets.len() < 2 {
            return None;
        }

        // Compute final gain
        let gain = compute_gain(
            self.base_info,
            unknown_frac,
            &subset_freq,
            known_weight,
        );
        let penalty = subset_penalty(initial_blocks, subsets.len(), known_weight);
        let adjusted_gain = gain - penalty;

        if adjusted_gain <= 0.0 {
            return None;
        }

        let split_info_val = split_info(&subset_weight);
        let gain_ratio = gain_ratio(adjusted_gain, split_info_val);

        // Convert value indices back to actual values
        let subsets_with_values: Vec<Vec<usize>> = subsets.iter()
            .map(|s| s.iter().map(|&v| v).collect())
            .collect();

        // Build branch indices
        let mut branch_indices: Vec<Vec<usize>> = vec![Vec::new(); subsets.len()];
        let mut unknown_indices = Vec::new();

        for &i in self.indices {
            let v = self.dataset.get_value(i, attr);
            if is_valid(v) {
                let val_idx = value_map[&OrderedFloat(v)];
                // Find which subset contains this value
                for (s_idx, subset) in subsets.iter().enumerate() {
                    if subset.contains(&val_idx) {
                        branch_indices[s_idx].push(i);
                        break;
                    }
                }
            } else {
                unknown_indices.push(i);
            }
        }

        // Assign unknowns to largest subset
        if !unknown_indices.is_empty() {
            let max_subset = (0..subsets.len())
                .max_by(|&a, &b| {
                    subset_weight[a].partial_cmp(&subset_weight[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            branch_indices[max_subset].extend(unknown_indices);
        }

        let branch_cases: Vec<f64> = branch_indices.iter()
            .map(|indices| indices.iter().map(|&i| self.dataset.get_weight(i)).sum())
            .collect();
        let branch_class_dist: Vec<Vec<f64>> = branch_indices.iter()
            .map(|indices| self.dataset.class_distribution(indices, self.n_classes))
            .collect();

        Some(SplitResult {
            attribute: attr,
            split_type: NodeType::Subset,
            gain: adjusted_gain,
            gain_ratio,
            threshold: None,
            subsets: Some(subsets_with_values),
            branch_cases,
            branch_class_dist,
            branch_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> Dataset {
        // Simple XOR-like dataset
        let mut ds = Dataset::new(8, 2);

        // Feature 0: continuous [0, 1, 2, 3, 0, 1, 2, 3]
        // Feature 1: categorical [0, 0, 1, 1, 0, 0, 1, 1]
        // Class: XOR of (feature0 > 1.5) and feature1
        let values = [
            (0.0, 0.0, 0), (1.0, 0.0, 0),
            (2.0, 1.0, 0), (3.0, 1.0, 0),
            (0.0, 0.0, 0), (1.0, 0.0, 0),
            (2.0, 1.0, 1), (3.0, 1.0, 1),
        ];

        for (i, &(f0, f1, c)) in values.iter().enumerate() {
            ds.set_value(i, 0, f0);
            ds.set_value(i, 1, f1);
            ds.set_class(i, c);
        }

        ds.set_categorical(1, true);
        ds
    }

    #[test]
    fn test_continuous_split() {
        let ds = create_test_dataset();
        let indices: Vec<usize> = (0..8).collect();
        let config = TreeConfig::default();

        let evaluator = SplitEvaluator::new(&ds, &indices, 2, &config);
        let result = evaluator.evaluate_continuous_split(0);

        assert!(result.is_some());
        let split = result.unwrap();
        assert!(split.threshold.is_some());
        assert!(split.gain > 0.0);
    }

    #[test]
    fn test_discrete_split() {
        let ds = create_test_dataset();
        let indices: Vec<usize> = (0..8).collect();
        let config = TreeConfig::default();

        let evaluator = SplitEvaluator::new(&ds, &indices, 2, &config);
        let result = evaluator.evaluate_discrete_split(1);

        assert!(result.is_some());
    }
}
