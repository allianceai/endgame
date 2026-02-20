//! Information theory calculations for C5.0
//!
//! Implements entropy, information gain, and gain ratio calculations.

use std::f64::consts::LN_2;

/// Log base 2, handling zero
#[inline]
pub fn log2(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x.ln() / LN_2
    }
}

/// Compute entropy of a distribution
/// H = -sum(p_i * log2(p_i))
pub fn entropy(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            sum -= p * log2(p);
        }
    }
    sum
}

/// Compute total information (not normalized)
/// TotalInfo = N * log2(N) - sum(n_i * log2(n_i))
pub fn total_info(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for &c in counts {
        if c > 0.0 {
            sum += c * log2(c);
        }
    }
    total * log2(total) - sum
}

/// Compute information gain for a split
///
/// # Arguments
/// * `base_info` - Information before split
/// * `unknown_frac` - Fraction of cases with unknown values
/// * `branch_class_counts` - Class counts in each branch [branch][class]
/// * `total_cases` - Total number of cases
///
/// # Returns
/// Information gain (adjusted for unknown values)
pub fn compute_gain(
    base_info: f64,
    unknown_frac: f64,
    branch_class_counts: &[Vec<f64>],
    total_cases: f64,
) -> f64 {
    if total_cases <= 0.0 {
        return 0.0;
    }

    // Compute info after split
    let mut split_info = 0.0;
    for branch_counts in branch_class_counts {
        split_info += total_info(branch_counts);
    }
    split_info /= total_cases;

    // Adjust for unknown values
    (1.0 - unknown_frac) * (base_info - split_info)
}

/// Compute split information (entropy of the split itself)
/// Used for gain ratio calculation
///
/// SplitInfo = -sum((n_branch/N) * log2(n_branch/N))
pub fn split_info(branch_counts: &[f64]) -> f64 {
    entropy(branch_counts)
}

/// Compute gain ratio
/// GainRatio = Gain / SplitInfo
pub fn gain_ratio(gain: f64, split_info: f64) -> f64 {
    if split_info <= 0.0 {
        0.0
    } else {
        gain / split_info
    }
}

/// Compute penalty for continuous attribute threshold selection
/// Penalty = log2(min(interval_ratio, n_cuts)) / n_cases
///
/// This penalizes having many possible cut points to prevent overfitting
pub fn threshold_penalty(
    n_cuts: usize,
    interval_ratio: f64,
    n_cases: f64,
) -> f64 {
    if n_cases <= 0.0 || n_cuts == 0 {
        return 0.0;
    }

    let effective_cuts = (n_cuts as f64).min(interval_ratio);
    log2(effective_cuts.max(1.0)) / n_cases
}

/// Compute Bell number (Stirling number of second kind)
/// Used for subset splitting penalty
///
/// Bell[n][k] = number of ways to partition n items into k non-empty subsets
pub fn bell_number(n: usize, k: usize) -> f64 {
    if k == 0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if k > n {
        return 0.0;
    }
    if k == 1 || k == n {
        return 1.0;
    }

    // Use recurrence: Bell[n][k] = Bell[n-1][k-1] + k * Bell[n-1][k]
    // Dynamic programming approach
    let mut prev = vec![0.0; k + 1];
    let mut curr = vec![0.0; k + 1];

    prev[1] = 1.0;

    for i in 2..=n {
        curr[1] = 1.0;
        for j in 2..=k.min(i) {
            curr[j] = prev[j - 1] + j as f64 * prev[j];
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[k]
}

/// Compute penalty for subset splitting
/// Penalty = log2(Bell[n_initial][n_final]) / n_cases
pub fn subset_penalty(
    n_initial_subsets: usize,
    n_final_subsets: usize,
    n_cases: f64,
) -> f64 {
    if n_cases <= 0.0 {
        return 0.0;
    }

    let bell = bell_number(n_initial_subsets, n_final_subsets);
    log2(bell.max(1.0)) / n_cases
}

/// Class for tracking frequency tables during split evaluation
pub struct FrequencyTable {
    /// Frequency counts: freq[value][class]
    pub freq: Vec<Vec<f64>>,
    /// Total weight per value
    pub val_freq: Vec<f64>,
    /// Total weight per class
    pub class_freq: Vec<f64>,
    /// Total weight
    pub total: f64,
    /// Number of values
    pub n_values: usize,
    /// Number of classes
    pub n_classes: usize,
}

impl FrequencyTable {
    /// Create a new frequency table
    pub fn new(n_values: usize, n_classes: usize) -> Self {
        FrequencyTable {
            freq: vec![vec![0.0; n_classes]; n_values],
            val_freq: vec![0.0; n_values],
            class_freq: vec![0.0; n_classes],
            total: 0.0,
            n_values,
            n_classes,
        }
    }

    /// Reset all counts to zero
    pub fn reset(&mut self) {
        for row in &mut self.freq {
            row.fill(0.0);
        }
        self.val_freq.fill(0.0);
        self.class_freq.fill(0.0);
        self.total = 0.0;
    }

    /// Add a weighted observation
    #[inline]
    pub fn add(&mut self, value: usize, class: usize, weight: f64) {
        self.freq[value][class] += weight;
        self.val_freq[value] += weight;
        self.class_freq[class] += weight;
        self.total += weight;
    }

    /// Compute base information (entropy of class distribution)
    pub fn base_info(&self) -> f64 {
        entropy(&self.class_freq)
    }

    /// Compute information gain for splitting by value
    pub fn info_gain(&self) -> f64 {
        if self.total <= 0.0 {
            return 0.0;
        }

        let base = self.base_info();

        // Weighted average of branch entropies
        let mut split_entropy = 0.0;
        for v in 0..self.n_values {
            if self.val_freq[v] > 0.0 {
                let p = self.val_freq[v] / self.total;
                split_entropy += p * entropy(&self.freq[v]);
            }
        }

        base - split_entropy
    }

    /// Compute split information
    pub fn split_info(&self) -> f64 {
        split_info(&self.val_freq)
    }

    /// Compute gain ratio
    pub fn gain_ratio(&self) -> f64 {
        let gain = self.info_gain();
        let split = self.split_info();
        gain_ratio(gain, split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        // Pure distribution
        assert!((entropy(&[10.0, 0.0]) - 0.0).abs() < 1e-10);

        // Uniform distribution (2 classes)
        let e = entropy(&[5.0, 5.0]);
        assert!((e - 1.0).abs() < 1e-10);

        // Uniform distribution (4 classes)
        let e = entropy(&[5.0, 5.0, 5.0, 5.0]);
        assert!((e - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_numbers() {
        // Bell triangle values
        assert!((bell_number(3, 2) - 3.0).abs() < 1e-10);
        assert!((bell_number(4, 2) - 7.0).abs() < 1e-10);
        assert!((bell_number(5, 3) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_table() {
        let mut ft = FrequencyTable::new(3, 2);

        // Add observations: value 0 -> class 0, value 1 -> class 1, etc
        ft.add(0, 0, 5.0);
        ft.add(1, 1, 5.0);

        assert!((ft.total - 10.0).abs() < 1e-10);
        assert!(ft.info_gain() > 0.0);
    }
}
