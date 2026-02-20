//! Rule extraction from Cubist trees
//!
//! Extracts and prunes regression rules from decision trees.

use crate::types::*;
use crate::regress::*;

/// Extract rules from a tree
pub struct RuleExtractor<'a> {
    dataset: &'a Dataset,
    n_features: usize,
}

impl<'a> RuleExtractor<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        RuleExtractor {
            dataset,
            n_features: dataset.n_features,
        }
    }

    /// Extract all rules from a tree
    pub fn extract(&self, tree: &TreeNode, model_no: usize) -> RuleSet {
        let mut rules = Vec::new();
        let mut conditions = Vec::new();
        let mut rule_no = 0;

        self.scan_tree(tree, &mut conditions, &mut rules, &mut rule_no, model_no);

        // Set default value
        let default_val = tree.mean;

        RuleSet {
            rules,
            default_val,
        }
    }

    /// Recursively scan tree and extract rules
    fn scan_tree(
        &self,
        node: &TreeNode,
        conditions: &mut Vec<Condition>,
        rules: &mut Vec<Rule>,
        rule_no: &mut usize,
        model_no: usize,
    ) {
        if node.is_leaf() {
            // Create rule from current path
            if let Some(ref model) = node.model {
                let mut rule = Rule::new(*rule_no, model_no, self.n_features);
                rule.conditions = conditions.clone();
                rule.model = model.clone();
                rule.cover = node.cases;
                rule.mean = node.mean;
                rule.lo_val = node.lo_val;
                rule.hi_val = node.hi_val;
                rule.lo_lim = node.lo_val - 0.05 * (node.hi_val - node.lo_val).abs();
                rule.hi_lim = node.hi_val + 0.05 * (node.hi_val - node.lo_val).abs();

                rules.push(rule);
                *rule_no += 1;
            }
            return;
        }

        let attr = node.tested_attr.unwrap_or(0);

        // Process each branch
        match node.branch_type {
            BranchType::Threshold => {
                let cut = node.cut.unwrap_or(0.0);

                // Left branch: attr <= cut
                if node.branches.len() > 0 {
                    conditions.push(Condition::threshold_le(attr, cut));
                    self.scan_tree(&node.branches[0], conditions, rules, rule_no, model_no);
                    conditions.pop();
                }

                // Right branch: attr > cut
                if node.branches.len() > 1 {
                    conditions.push(Condition::threshold_gt(attr, cut));
                    self.scan_tree(&node.branches[1], conditions, rules, rule_no, model_no);
                    conditions.pop();
                }
            }
            BranchType::Discrete => {
                for (i, branch) in node.branches.iter().enumerate() {
                    let mut subset = std::collections::HashSet::new();
                    subset.insert(i);
                    conditions.push(Condition::discrete_subset(attr, subset));
                    self.scan_tree(branch, conditions, rules, rule_no, model_no);
                    conditions.pop();
                }
            }
            BranchType::Subset => {
                if let Some(ref subsets) = node.subset {
                    for (subset, branch) in subsets.iter().zip(node.branches.iter()) {
                        conditions.push(Condition::discrete_subset(attr, subset.clone()));
                        self.scan_tree(branch, conditions, rules, rule_no, model_no);
                        conditions.pop();
                    }
                }
            }
            BranchType::None => {}
        }
    }

    /// Prune redundant conditions from rules
    pub fn prune_rules(&self, ruleset: &mut RuleSet) {
        for rule in &mut ruleset.rules {
            self.prune_rule(rule);
        }
    }

    /// Prune a single rule by removing redundant conditions
    fn prune_rule(&self, rule: &mut Rule) {
        if rule.conditions.is_empty() {
            return;
        }

        // Merge conditions on the same attribute
        let mut merged: std::collections::HashMap<usize, (Option<f64>, Option<f64>)> =
            std::collections::HashMap::new();

        for cond in &rule.conditions {
            if cond.test_type == BranchType::Threshold {
                let entry = merged.entry(cond.attribute).or_insert((None, None));

                // Update lower bound (attr > low)
                if let Some(low) = cond.threshold_low {
                    entry.0 = Some(entry.0.map(|l| l.max(low)).unwrap_or(low));
                }

                // Update upper bound (attr <= high)
                if let Some(high) = cond.threshold_high {
                    entry.1 = Some(entry.1.map(|h| h.min(high)).unwrap_or(high));
                }
            }
        }

        // Rebuild conditions with merged thresholds
        let mut new_conditions = Vec::new();

        // Add merged threshold conditions
        for (&attr, &(low, high)) in &merged {
            if low.is_some() || high.is_some() {
                let mut cond = Condition {
                    attribute: attr,
                    test_type: BranchType::Threshold,
                    threshold_low: low,
                    threshold_high: high,
                    subset: None,
                };
                new_conditions.push(cond);
            }
        }

        // Add non-threshold conditions
        for cond in &rule.conditions {
            if cond.test_type != BranchType::Threshold {
                new_conditions.push(cond.clone());
            }
        }

        rule.conditions = new_conditions;
    }
}

/// Rule pruner that drops conditions based on error impact
pub struct RulePruner<'a> {
    dataset: &'a Dataset,
}

impl<'a> RulePruner<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        RulePruner { dataset }
    }

    /// Prune rules by dropping conditions that don't significantly increase error
    pub fn prune(&self, ruleset: &mut RuleSet, indices: &[usize]) {
        let mut solver = RegressionSolver::new(self.dataset.n_features);

        for rule in &mut ruleset.rules {
            self.prune_single_rule(rule, indices, &mut solver);
        }
    }

    fn prune_single_rule(&self, rule: &mut Rule, indices: &[usize], solver: &mut RegressionSolver) {
        if rule.conditions.is_empty() {
            return;
        }

        // Get matching indices for current rule
        let matching: Vec<usize> = indices.iter()
            .filter(|&&i| rule.matches(self.dataset.get_row(i)))
            .copied()
            .collect();

        if matching.is_empty() {
            return;
        }

        // Compute base error
        let base_err = self.compute_rule_error(rule, &matching);

        // Try dropping each condition
        let mut improved = true;
        while improved && !rule.conditions.is_empty() {
            improved = false;
            let mut best_drop: Option<usize> = None;
            let mut best_err = base_err;

            for i in 0..rule.conditions.len() {
                // Temporarily remove condition
                let cond = rule.conditions.remove(i);

                // Get new matching set (will be >= original)
                let new_matching: Vec<usize> = indices.iter()
                    .filter(|&&idx| rule.matches(self.dataset.get_row(idx)))
                    .copied()
                    .collect();

                // Compute error with this condition dropped
                if new_matching.len() >= 2 {
                    let usable_attrs: Vec<usize> = (0..self.dataset.n_features)
                        .filter(|&i| self.dataset.attr_continuous[i])
                        .collect();

                    let (new_model, new_err) = solver.fit(self.dataset, &new_matching, &usable_attrs);

                    // Adjust for complexity
                    let adj_base = base_err * (1.0 + 0.01 * rule.conditions.len() as f64);
                    let adj_new = new_err;

                    if adj_new <= adj_base {
                        if new_err < best_err || (new_err <= best_err && best_drop.is_none()) {
                            best_err = new_err;
                            best_drop = Some(i);
                        }
                    }
                }

                // Restore condition
                rule.conditions.insert(i, cond);
            }

            // If we found a condition to drop, do it permanently
            if let Some(drop_idx) = best_drop {
                rule.conditions.remove(drop_idx);

                // Re-fit model for new coverage
                let new_matching: Vec<usize> = indices.iter()
                    .filter(|&&idx| rule.matches(self.dataset.get_row(idx)))
                    .copied()
                    .collect();

                if !new_matching.is_empty() {
                    let usable_attrs: Vec<usize> = (0..self.dataset.n_features)
                        .filter(|&i| self.dataset.attr_continuous[i])
                        .collect();

                    let (new_model, _) = solver.fit(self.dataset, &new_matching, &usable_attrs);
                    rule.model = new_model;
                    rule.cover = new_matching.iter()
                        .map(|&i| self.dataset.weights[i])
                        .sum();

                    // Update stats
                    let (mean, _, lo, hi) = self.dataset.subset_stats(&new_matching);
                    rule.mean = mean;
                    rule.lo_val = lo;
                    rule.hi_val = hi;
                    rule.lo_lim = lo - 0.05 * (hi - lo).abs();
                    rule.hi_lim = hi + 0.05 * (hi - lo).abs();
                }

                improved = true;
            }
        }

        // Compute final error estimate
        let final_matching: Vec<usize> = indices.iter()
            .filter(|&&i| rule.matches(self.dataset.get_row(i)))
            .copied()
            .collect();
        rule.est_err = self.compute_rule_error(rule, &final_matching);
    }

    fn compute_rule_error(&self, rule: &Rule, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut total_err = 0.0;
        let mut total_weight = 0.0;

        for &idx in indices {
            let row = self.dataset.get_row(idx);
            let pred = rule.predict(row);
            let actual = self.dataset.get_target(idx);
            let weight = self.dataset.weights[idx];

            total_err += (pred - actual).abs() * weight;
            total_weight += weight;
        }

        total_err / total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::*;

    #[test]
    fn test_rule_extraction() {
        let mut ds = Dataset::new(10, 1);
        for i in 0..5 {
            ds.set_value(i, 0, i as f64);
            ds.set_target(i, 10.0);
        }
        for i in 5..10 {
            ds.set_value(i, 0, i as f64);
            ds.set_target(i, 20.0);
        }
        ds.compute_target_stats();

        let config = CubistConfig::default();
        let mut builder = TreeBuilder::new(&ds, config);
        let tree = builder.build();

        let extractor = RuleExtractor::new(&ds);
        let ruleset = extractor.extract(&tree.root, 0);

        // Should have at least 2 rules (one per leaf)
        assert!(ruleset.rules.len() >= 2);
    }

    #[test]
    fn test_condition_merge() {
        let ds = Dataset::new(10, 2);
        let extractor = RuleExtractor::new(&ds);

        let mut rule = Rule::new(0, 0, 2);
        // Add two conditions on same attribute that should merge
        rule.conditions.push(Condition::threshold_le(0, 10.0)); // x <= 10
        rule.conditions.push(Condition::threshold_gt(0, 5.0));  // x > 5

        extractor.prune_rule(&mut rule);

        // Should be merged into single condition: 5 < x <= 10
        assert_eq!(rule.conditions.len(), 1);
        let cond = &rule.conditions[0];
        assert_eq!(cond.attribute, 0);
        assert!((cond.threshold_low.unwrap() - 5.0).abs() < 1e-10);
        assert!((cond.threshold_high.unwrap() - 10.0).abs() < 1e-10);
    }
}
