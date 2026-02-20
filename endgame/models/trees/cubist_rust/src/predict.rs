//! Prediction module for Cubist
//!
//! Handles prediction using rules, committees, and optional instance-based correction.

use crate::types::*;

/// Predictor for Cubist models
pub struct CubistPredictor {
    /// Rule sets from each committee member
    pub rulesets: Vec<RuleSet>,
    /// Training data for instance-based correction (if enabled)
    pub training_data: Option<TrainingData>,
    /// Configuration
    pub config: CubistConfig,
    /// Global target statistics
    pub target_mean: f64,
    pub target_sd: f64,
    pub target_min: f64,
    pub target_max: f64,
}

/// Stored training data for instance-based correction
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
    pub n_features: usize,
}

impl TrainingData {
    pub fn from_dataset(dataset: &Dataset) -> Self {
        let mut features = Vec::with_capacity(dataset.n_samples);
        for i in 0..dataset.n_samples {
            features.push(dataset.get_row(i).to_vec());
        }

        TrainingData {
            features,
            targets: dataset.targets.clone(),
            n_features: dataset.n_features,
        }
    }
}

impl CubistPredictor {
    pub fn new(config: CubistConfig) -> Self {
        CubistPredictor {
            rulesets: Vec::new(),
            training_data: None,
            config,
            target_mean: 0.0,
            target_sd: 1.0,
            target_min: 0.0,
            target_max: 0.0,
        }
    }

    /// Add a ruleset from a committee member
    pub fn add_ruleset(&mut self, ruleset: RuleSet) {
        self.rulesets.push(ruleset);
    }

    /// Set training data for instance-based correction
    pub fn set_training_data(&mut self, data: TrainingData) {
        self.training_data = Some(data);
    }

    /// Predict for a single sample
    pub fn predict(&self, sample: &[f64]) -> f64 {
        if self.rulesets.is_empty() {
            return self.target_mean;
        }

        // Get prediction from rules
        let rule_pred = self.predict_from_rules(sample);

        // Apply instance-based correction if enabled
        if self.config.use_instance && self.training_data.is_some() {
            self.instance_correct(sample, rule_pred)
        } else {
            rule_pred
        }
    }

    /// Predict using rule sets (committee average)
    /// Cubist averages predictions from all committee members
    fn predict_from_rules(&self, sample: &[f64]) -> f64 {
        if self.rulesets.is_empty() {
            return self.target_mean;
        }

        // Average predictions from all rulesets (committee voting)
        let sum: f64 = self.rulesets.iter()
            .map(|rs| rs.predict(sample))
            .sum();
        sum / self.rulesets.len() as f64
    }

    /// Apply instance-based (nearest neighbor) correction
    fn instance_correct(&self, sample: &[f64], rule_pred: f64) -> f64 {
        let training = match &self.training_data {
            Some(t) => t,
            None => return rule_pred,
        };

        if training.features.is_empty() {
            return rule_pred;
        }

        // Find k nearest neighbors
        let k = self.config.neighbors.min(training.features.len());
        let mut distances: Vec<(f64, usize)> = training.features.iter()
            .enumerate()
            .map(|(i, feat)| {
                let dist = self.euclidean_distance(sample, feat);
                (dist, i)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Compute weighted average of neighbor corrections
        let mut total_weight = 0.0;
        let mut total_correction = 0.0;

        for (dist, idx) in distances.iter().take(k) {
            if *dist < 1e-10 {
                // Exact match: use this target directly
                return training.targets[*idx];
            }

            // Weight by inverse distance
            let weight = 1.0 / dist;

            // Get model prediction for this training sample
            let train_pred = self.predict_from_rules(&training.features[*idx]);
            let actual = training.targets[*idx];
            let correction = actual - train_pred;

            total_weight += weight;
            total_correction += weight * correction;
        }

        if total_weight > 0.0 {
            rule_pred + total_correction / total_weight
        } else {
            rule_pred
        }
    }

    /// Compute Euclidean distance between samples
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| {
                if ai.is_nan() || bi.is_nan() {
                    0.0 // Ignore missing values
                } else {
                    (ai - bi).powi(2)
                }
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Predict for multiple samples
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<f64> {
        samples.iter()
            .map(|s| self.predict(s))
            .collect()
    }

    /// Predict with bounds enforcement
    pub fn predict_bounded(&self, sample: &[f64]) -> f64 {
        let pred = self.predict(sample);

        // Allow some extrapolation
        let range = self.target_max - self.target_min;
        let extra = range * self.config.extrapolation;

        pred.max(self.target_min - extra).min(self.target_max + extra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_prediction() {
        let config = CubistConfig::default();
        let mut predictor = CubistPredictor::new(config);

        // Create a simple ruleset
        let mut ruleset = RuleSet::new();
        ruleset.default_val = 10.0;

        let mut rule = Rule::new(0, 0, 2);
        rule.conditions.push(Condition::threshold_le(0, 5.0));
        rule.model.constant = 5.0;
        rule.model.coefficients = vec![1.0, 0.0];
        rule.model.used_attrs = vec![0];
        rule.cover = 10.0;
        rule.lo_lim = 0.0;
        rule.hi_lim = 20.0;
        ruleset.rules.push(rule);

        predictor.add_ruleset(ruleset);
        predictor.target_mean = 10.0;

        // Test prediction
        let pred = predictor.predict(&[3.0, 0.0]);
        assert!((pred - 8.0).abs() < 1e-10); // 5 + 1*3 = 8
    }

    #[test]
    fn test_instance_correction() {
        let mut config = CubistConfig::default();
        config.use_instance = true;
        config.neighbors = 2;

        let mut predictor = CubistPredictor::new(config);

        // Simple ruleset that predicts constant
        let mut ruleset = RuleSet::new();
        ruleset.default_val = 10.0;
        predictor.add_ruleset(ruleset);

        // Add training data with known values
        let training = TrainingData {
            features: vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![2.0, 0.0],
            ],
            targets: vec![0.0, 5.0, 10.0],
            n_features: 2,
        };
        predictor.set_training_data(training);

        // Predict at x=1.5 - should be corrected toward neighbors
        let pred = predictor.predict(&[1.5, 0.0]);
        // Neighbors at 1.0 and 2.0, so should be between 5 and 10
        assert!(pred > 0.0 && pred < 15.0);
    }
}
