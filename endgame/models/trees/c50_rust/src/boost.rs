//! Boosting implementation for C5.0
//!
//! Implements a modified AdaBoost algorithm as used in C5.0.

use crate::types::*;
use crate::tree::*;
use crate::classify::*;
use rayon::prelude::*;

/// Boosted ensemble of C5.0 trees
#[derive(Clone)]
pub struct C50Ensemble {
    /// Individual trees in the ensemble
    pub trees: Vec<C50Tree>,
    /// Confidence/weight for each tree
    pub confidences: Vec<f64>,
    /// Number of classes
    pub n_classes: usize,
}

impl C50Ensemble {
    /// Classify a sample using voting
    pub fn classify(&self, sample: &[f64]) -> usize {
        let votes = self.get_votes(sample);

        votes.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get class probabilities
    pub fn classify_proba(&self, sample: &[f64]) -> Vec<f64> {
        let votes = self.get_votes(sample);

        let total: f64 = votes.iter().sum();
        if total > 0.0 {
            votes.iter().map(|&v| v / total).collect()
        } else {
            vec![1.0 / self.n_classes as f64; self.n_classes]
        }
    }

    /// Get weighted votes for each class
    fn get_votes(&self, sample: &[f64]) -> Vec<f64> {
        let mut votes = vec![0.0; self.n_classes];

        for (tree, &confidence) in self.trees.iter().zip(&self.confidences) {
            let proba = tree.classify_proba(sample);
            for (c, &p) in proba.iter().enumerate() {
                if c < votes.len() {
                    votes[c] += confidence * p;
                }
            }
        }

        votes
    }
}

/// Booster for building C5.0 ensembles
pub struct Booster<'a> {
    dataset: &'a Dataset,
    config: TreeConfig,
    n_trials: usize,
    n_classes: usize,
}

impl<'a> Booster<'a> {
    /// Create a new booster
    pub fn new(dataset: &'a Dataset, config: TreeConfig, n_trials: usize) -> Self {
        let n_classes = dataset.classes.iter().max().map(|&m| m + 1).unwrap_or(0);

        Booster {
            dataset,
            config,
            n_trials,
            n_classes,
        }
    }

    /// Build the boosted ensemble
    pub fn build(&mut self) -> C50Ensemble {
        let n_samples = self.dataset.n_samples;
        let mut weights = vec![1.0; n_samples];
        let mut trees = Vec::with_capacity(self.n_trials);
        let mut confidences = Vec::with_capacity(self.n_trials);

        // Track votes for early stopping
        let mut votes: Vec<Vec<f64>> = vec![vec![0.0; self.n_classes]; n_samples];

        // Base leaf ratio (for tree size control)
        let mut leaf_ratio: Option<f64> = None;

        for trial in 0..self.n_trials {
            // Create weighted dataset
            let mut weighted_dataset = self.create_weighted_dataset(&weights);

            // Build tree
            let mut builder = TreeBuilder::new(&weighted_dataset, self.config.clone());
            let tree = builder.build();

            // Set leaf ratio from first tree
            if trial == 0 {
                let n_leaves = tree.n_leaves();
                leaf_ratio = Some(1.1 * n_leaves as f64 / n_samples as f64);
            }

            // Compute predictions and errors
            let mut correct_weight = 0.0;
            let mut error_weight = 0.0;
            let predictions: Vec<usize> = (0..n_samples)
                .map(|i| {
                    let sample: Vec<f64> = (0..self.dataset.n_features)
                        .map(|j| self.dataset.get_value(i, j))
                        .collect();
                    tree.classify(&sample)
                })
                .collect();

            for i in 0..n_samples {
                if predictions[i] == self.dataset.get_class(i) {
                    correct_weight += weights[i];
                } else {
                    error_weight += weights[i];
                }
            }

            let total_weight = correct_weight + error_weight;
            let error_rate = error_weight / total_weight;

            // Check if this tree is useful
            if error_rate >= 0.5 {
                // Tree is no better than random, stop boosting
                break;
            }

            // Compute confidence (tree weight)
            let confidence = if error_rate > 0.0 {
                0.5 * ((1.0 - error_rate) / error_rate).ln()
            } else {
                // Perfect tree
                2.0
            };

            // Update votes
            for i in 0..n_samples {
                let sample: Vec<f64> = (0..self.dataset.n_features)
                    .map(|j| self.dataset.get_value(i, j))
                    .collect();
                let proba = tree.classify_proba(&sample);
                for (c, &p) in proba.iter().enumerate() {
                    votes[i][c] += confidence * p;
                }
            }

            trees.push(tree);
            confidences.push(confidence);

            // Update weights (AdaBoost-style)
            if trial < self.n_trials - 1 {
                self.update_weights(
                    &mut weights,
                    &predictions,
                    error_rate,
                    &votes,
                    trial,
                );
            }
        }

        // Normalize confidences
        let conf_sum: f64 = confidences.iter().sum();
        if conf_sum > 0.0 {
            for c in &mut confidences {
                *c /= conf_sum;
            }
        }

        C50Ensemble {
            trees,
            confidences,
            n_classes: self.n_classes,
        }
    }

    /// Create a dataset with current weights
    fn create_weighted_dataset(&self, weights: &[f64]) -> Dataset {
        let mut ds = Dataset::new(self.dataset.n_samples, self.dataset.n_features);

        for i in 0..self.dataset.n_samples {
            for j in 0..self.dataset.n_features {
                ds.set_value(i, j, self.dataset.get_value(i, j));
            }
            ds.set_class(i, self.dataset.get_class(i));
            ds.set_weight(i, weights[i]);
        }

        for j in 0..self.dataset.n_features {
            if self.dataset.is_categorical(j) {
                ds.set_categorical(j, true);
            }
        }

        ds
    }

    /// Update weights after a boosting iteration
    fn update_weights(
        &self,
        weights: &mut [f64],
        predictions: &[usize],
        error_rate: f64,
        votes: &[Vec<f64>],
        trial: usize,
    ) {
        let n_samples = weights.len();

        // Compute weight adjustments
        let beta = error_rate / (1.0 - error_rate);

        for i in 0..n_samples {
            let true_class = self.dataset.get_class(i);

            if predictions[i] == true_class {
                // Correctly classified: decrease weight
                weights[i] *= beta;
            }
            // Misclassified: weight stays the same (relative increase)

            // Brown-boost style case dropping (after halfway)
            if trial > self.n_trials / 2 {
                // Find best voted class
                let best_vote = votes[i].iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(c, &v)| (c, v));

                if let Some((best_class, best_v)) = best_vote {
                    let true_vote = votes[i][true_class];
                    let remaining_trials = self.n_trials - trial - 1;

                    // If true class can't catch up, drop this case
                    if best_class != true_class
                        && best_v > true_vote + remaining_trials as f64 * 2.0 {
                        weights[i] = 0.0;
                    }
                }
            }
        }

        // Normalize weights
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in weights.iter_mut() {
                *w *= n_samples as f64 / total;
            }
        }

        // Trim very small weights for efficiency
        let min_weight = 0.001;
        for w in weights.iter_mut() {
            if *w < min_weight {
                *w = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_booster() {
        // Create simple dataset
        let mut ds = Dataset::new(20, 2);

        for i in 0..10 {
            ds.set_value(i, 0, i as f64);
            ds.set_value(i, 1, 0.0);
            ds.set_class(i, 0);
        }
        for i in 10..20 {
            ds.set_value(i, 0, i as f64);
            ds.set_value(i, 1, 1.0);
            ds.set_class(i, 1);
        }

        let config = TreeConfig::default();
        let mut booster = Booster::new(&ds, config, 5);
        let ensemble = booster.build();

        assert!(!ensemble.trees.is_empty());

        // Test classification
        let pred = ensemble.classify(&[5.0, 0.0]);
        assert_eq!(pred, 0);

        let pred = ensemble.classify(&[15.0, 1.0]);
        assert_eq!(pred, 1);
    }
}
