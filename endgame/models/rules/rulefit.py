from __future__ import annotations

"""RuleFit: Rule-based models combining tree ensembles with Lasso.

RuleFit, developed by Friedman and Popescu (2008), is a hybrid model that combines
the predictive power of tree ensembles with the interpretability of linear models.

References
----------
Friedman, J. H., & Popescu, B. E. (2008). "Predictive learning via rule ensembles."
The Annals of Applied Statistics, 2(3), 916-954.
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from endgame.models.rules.extraction import extract_rules_from_ensemble
from typing import Any


def _preprocess_linear_features(
    X: np.ndarray,
    winsorize: float | None = 0.025,
    standardize: bool = True,
    fit: bool = True,
    stored_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Preprocess linear features for RuleFit.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input features.
    winsorize : float or None
        Quantile for winsorization. If 0.025, clips values
        below 2.5th percentile and above 97.5th percentile.
    standardize : bool
        Whether to standardize (zero mean, unit variance).
    fit : bool
        If True, compute parameters from X. If False, use stored_params.
    stored_params : dict or None
        Previously computed preprocessing parameters.

    Returns
    -------
    X_processed : ndarray
        Preprocessed features.
    params : dict
        Preprocessing parameters for transform.
    """
    params = stored_params.copy() if stored_params else {}
    X_processed = X.copy()

    if winsorize is not None and winsorize > 0:
        if fit:
            lower = np.percentile(X, winsorize * 100, axis=0)
            upper = np.percentile(X, (1 - winsorize) * 100, axis=0)
            params["winsorize_lower"] = lower
            params["winsorize_upper"] = upper
        else:
            lower = params["winsorize_lower"]
            upper = params["winsorize_upper"]

        X_processed = np.clip(X_processed, lower, upper)

    if standardize:
        if fit:
            mean = np.mean(X_processed, axis=0)
            std = np.std(X_processed, axis=0)
            std[std < 1e-10] = 1.0  # Prevent division by zero
            params["mean"] = mean
            params["std"] = std
        else:
            mean = params["mean"]
            std = params["std"]

        X_processed = (X_processed - mean) / std

    return X_processed, params


class RuleFitRegressor(GlassboxMixin, BaseEstimator, RegressorMixin):
    """
    RuleFit: Rule-based regression combining tree ensembles with Lasso.

    RuleFit generates interpretable models by extracting rules from a tree
    ensemble and fitting a sparse linear model on the original features
    plus binary rule features. The result is a human-readable model that
    shows exactly which rules and features drive predictions.

    Parameters
    ----------
    tree_generator : estimator or None, default=None
        The tree ensemble used to generate rules. If None, uses
        GradientBoostingRegressor with default parameters.
        Must have `estimators_` attribute after fitting (trees).
        Common choices:
        - GradientBoostingRegressor/Classifier
        - RandomForestRegressor/Classifier
        - ExtraTreesRegressor/Classifier

    n_estimators : int, default=100
        Number of trees in the ensemble (if tree_generator is None).
        Ignored if tree_generator is provided.

    tree_max_depth : int, default=3
        Maximum depth of trees (if tree_generator is None).
        Shallow trees (2-4) produce simpler, more interpretable rules.
        Ignored if tree_generator is provided.

    max_rules : int or None, default=2000
        Maximum number of rules to extract. If None, extracts all rules.
        Rules are selected by coverage (fraction of samples satisfying rule).

    min_support : float, default=0.01
        Minimum fraction of samples that must satisfy a rule for it
        to be included. Rules with lower support are discarded.

    max_support : float, default=0.99
        Maximum fraction of samples satisfying a rule. Rules with
        higher support are too general and discarded.

    alpha : float or None, default=None
        Lasso regularization strength. If None, selected via cross-validation.
        Higher values produce sparser (more interpretable) models.

    cv : int, default=5
        Number of cross-validation folds for alpha selection.
        Only used if alpha is None.

    include_linear : bool, default=True
        Whether to include original features (linear terms) in the model.
        If False, model uses only rule features.

    standardize_linear : bool, default=True
        Whether to standardize linear features before fitting Lasso.
        Recommended for fair penalization across features.

    winsorize_linear : float or None, default=0.025
        Winsorization quantile for linear features. If not None,
        clips extreme values at this quantile to reduce outlier influence.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=None
        Number of parallel jobs for cross-validation.

    Attributes
    ----------
    rules_ : list of Rule
        Extracted rules with non-zero coefficients.

    rule_ensemble_ : RuleEnsemble
        All extracted rules (before Lasso selection).

    coef_ : ndarray
        Coefficients for all features (linear + rules).

    intercept_ : float
        Model intercept.

    linear_coef_ : ndarray of shape (n_features_in_,)
        Coefficients for original linear features.

    rule_coef_ : ndarray of shape (n_rules,)
        Coefficients for rule features.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during fit.

    n_features_in_ : int
        Number of original features.

    n_rules_ : int
        Number of rules extracted (before Lasso).

    n_rules_selected_ : int
        Number of rules with non-zero coefficients.

    alpha_ : float
        Regularization strength used (selected or provided).

    cv_results_ : dict
        Cross-validation results for alpha selection.

    tree_generator_ : estimator
        Fitted tree ensemble used for rule extraction.

    feature_importances_ : ndarray
        Importance of each original feature (sum of absolute coefficients
        for linear term and rules involving that feature).

    Examples
    --------
    >>> from endgame.models import RuleFitRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    >>> model = RuleFitRegressor(tree_max_depth=3, random_state=42)
    >>> model.fit(X, y)
    >>> print(model.get_rules())  # Print selected rules
    >>> predictions = model.predict(X)

    Notes
    -----
    For best interpretability:
    - Use shallow trees (max_depth=2 or 3) for simpler rules
    - Use higher alpha (more regularization) for sparser models
    - Provide meaningful feature_names for readable rule output

    References
    ----------
    Friedman, J. H., & Popescu, B. E. (2008). "Predictive learning via
    rule ensembles." The Annals of Applied Statistics, 2(3), 916-954.
    """

    def __init__(
        self,
        tree_generator=None,
        n_estimators: int = 50,
        tree_max_depth: int = 3,
        max_rules: int | None = 300,
        min_support: float = 0.01,
        max_support: float = 0.99,
        alpha: float | None = None,
        cv: int = 3,
        include_linear: bool = True,
        standardize_linear: bool = True,
        winsorize_linear: float | None = 0.025,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.tree_generator = tree_generator
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.max_rules = max_rules
        self.min_support = min_support
        self.max_support = max_support
        self.alpha = alpha
        self.cv = cv
        self.include_linear = include_linear
        self.standardize_linear = standardize_linear
        self.winsorize_linear = winsorize_linear
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        """
        Fit the RuleFit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        feature_names : list of str, default=None
            Names for features. If None, uses ['x0', 'x1', ...].
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for fitting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Set feature names
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(
                [f"x{i}" for i in range(self.n_features_in_)]
            )

        # Store training data for support calculation
        self._X_train = X
        self._y_train = y

        # Step 1: Fit tree ensemble
        self.tree_generator_ = self._get_tree_generator()
        self.tree_generator_.fit(X, y, sample_weight=sample_weight)

        # Step 2: Extract rules (tree-based support avoids re-evaluating on X)
        rule_ensemble = extract_rules_from_ensemble(
            self.tree_generator_,
            list(self.feature_names_in_),
        )

        # Step 3: Filter and deduplicate rules
        rule_ensemble = rule_ensemble.deduplicate()
        rule_ensemble = rule_ensemble.filter_by_support(
            self.min_support, self.max_support
        )

        if self.max_rules is not None:
            rule_ensemble = rule_ensemble.limit_rules(self.max_rules)

        self.rule_ensemble_ = rule_ensemble
        self.n_rules_ = len(rule_ensemble)

        # Step 4: Build design matrix
        X_rules = rule_ensemble.transform(X)

        if self.include_linear:
            X_linear, self._linear_params = _preprocess_linear_features(
                X,
                winsorize=self.winsorize_linear,
                standardize=self.standardize_linear,
                fit=True,
            )
            X_combined = np.hstack([X_linear, X_rules])
        else:
            X_combined = X_rules
            self._linear_params = {}

        # Step 5: Fit Lasso
        self._fit_lasso(X_combined, y, sample_weight)

        # Step 6: Extract linear and rule coefficients
        if self.include_linear:
            self.linear_coef_ = self.coef_[: self.n_features_in_]
            self.rule_coef_ = self.coef_[self.n_features_in_ :]
        else:
            self.linear_coef_ = np.zeros(self.n_features_in_)
            self.rule_coef_ = self.coef_

        # Step 7: Assign coefficients to rules
        for i, rule in enumerate(self.rule_ensemble_.rules):
            rule.coefficient = self.rule_coef_[i]

        # Step 8: Get selected rules (non-zero coefficients)
        self.rules_ = [
            r for r in self.rule_ensemble_.rules if abs(r.coefficient) > 1e-10
        ]
        self.n_rules_selected_ = len(self.rules_)

        # Step 9: Compute feature importances
        self._compute_feature_importances()

        return self

    def _get_tree_generator(self):
        """Get or create the tree generator."""
        if self.tree_generator is not None:
            return clone(self.tree_generator)

        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.tree_max_depth,
            learning_rate=0.15,
            subsample=0.5,
            max_features="sqrt",
            random_state=self.random_state,
        )

    def _fit_lasso(self, X_combined, y, sample_weight):
        """Fit Lasso model on combined features."""
        from sklearn.linear_model import Lasso, LassoCV

        if self.alpha is None:
            lasso_cv = LassoCV(
                cv=self.cv,
                n_alphas=50,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                max_iter=1000,
            )
            lasso_cv.fit(X_combined, y, sample_weight=sample_weight)

            self.alpha_ = lasso_cv.alpha_
            self.cv_results_ = {
                "alphas": lasso_cv.alphas_,
                "mse_path": lasso_cv.mse_path_,
            }

            self.coef_ = lasso_cv.coef_
            self.intercept_ = lasso_cv.intercept_
        else:
            lasso = Lasso(
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=1000,
            )
            lasso.fit(X_combined, y, sample_weight=sample_weight)

            self.alpha_ = self.alpha
            self.cv_results_ = {}

            self.coef_ = lasso.coef_
            self.intercept_ = lasso.intercept_

    def predict(self, X):
        """
        Predict using the fitted RuleFit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RuleFitRegressor is expecting "
                f"{self.n_features_in_} features as input."
            )

        X_combined = self._transform_combined(X)
        return X_combined @ self.coef_ + self.intercept_

    def _transform_combined(self, X):
        """Transform X to combined linear + rule features."""
        X_rules = self.rule_ensemble_.transform(X)

        if self.include_linear:
            X_linear, _ = _preprocess_linear_features(
                X,
                winsorize=self.winsorize_linear,
                standardize=self.standardize_linear,
                fit=False,
                stored_params=self._linear_params,
            )
            return np.hstack([X_linear, X_rules])
        else:
            return X_rules

    def transform(self, X):
        """
        Transform X into rule features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_rules : ndarray of shape (n_samples, n_rules)
            Binary matrix of rule satisfactions.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        return self.rule_ensemble_.transform(X)

    def _compute_feature_importances(self):
        """Compute feature importances from coefficients."""
        importances = np.zeros(self.n_features_in_)

        # Linear term contributions
        if self.include_linear:
            importances += np.abs(self.linear_coef_)

        # Rule contributions (distributed to features in each rule)
        for rule in self.rule_ensemble_.rules:
            if abs(rule.coefficient) > 1e-10:
                feature_indices = rule.feature_indices
                n_features_in_rule = len(feature_indices)

                if n_features_in_rule > 0:
                    # Distribute rule importance equally among its features
                    importance_per_feature = abs(rule.coefficient) / n_features_in_rule

                    for feat_idx in feature_indices:
                        importances[feat_idx] += importance_per_feature

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def get_rules(
        self, exclude_zero_coef: bool = True, sort_by: str = "importance"
    ) -> list[dict]:
        """
        Get the extracted rules with their coefficients.

        Parameters
        ----------
        exclude_zero_coef : bool, default=True
            If True, only return rules with non-zero coefficients.
        sort_by : str, default='importance'
            How to sort rules:
            - 'importance': By absolute coefficient value * support (descending)
            - 'support': By rule support/coverage (descending)
            - 'coefficient': By raw coefficient value (descending)
            - 'length': By number of conditions (ascending)

        Returns
        -------
        rules : list of dict
            Each dict contains:
            - 'rule': str, human-readable rule
            - 'coefficient': float, Lasso coefficient
            - 'support': float, fraction of samples satisfying rule
            - 'importance': float, |coefficient| * support
            - 'conditions': list of Condition objects
        """
        check_is_fitted(self)

        if exclude_zero_coef:
            rules = [r for r in self.rule_ensemble_.rules if abs(r.coefficient) > 1e-10]
        else:
            rules = list(self.rule_ensemble_.rules)

        # Sort
        if sort_by == "importance":
            rules = sorted(rules, key=lambda r: r.importance, reverse=True)
        elif sort_by == "support":
            rules = sorted(rules, key=lambda r: r.support, reverse=True)
        elif sort_by == "coefficient":
            rules = sorted(rules, key=lambda r: r.coefficient, reverse=True)
        elif sort_by == "length":
            rules = sorted(rules, key=lambda r: r.length)
        else:
            raise ValueError(f"Unknown sort_by: {sort_by}")

        return [r.to_dict() for r in rules]

    def summary(self) -> str:
        """
        Return a human-readable summary of the model.

        Returns
        -------
        summary : str
            Formatted model summary including:
            - Model statistics
            - Top rules by importance
            - Linear feature coefficients
        """
        check_is_fitted(self)

        lines = []
        lines.append("=" * 60)
        lines.append("RuleFit Model Summary")
        lines.append("=" * 60)
        lines.append("")

        # Model statistics
        lines.append("Model Statistics:")
        lines.append("-" * 40)
        lines.append(f"  Total rules extracted:    {self.n_rules_}")
        lines.append(f"  Rules with non-zero coef: {self.n_rules_selected_}")
        lines.append(f"  Alpha (regularization):   {self.alpha_:.6f}")
        lines.append(f"  Intercept:                {self.intercept_:.4f}")
        lines.append("")

        # Linear coefficients
        if self.include_linear:
            lines.append("Linear Feature Coefficients:")
            lines.append("-" * 40)

            # Sort by absolute value
            linear_importance = sorted(
                zip(self.feature_names_in_, self.linear_coef_),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            for name, coef in linear_importance:
                if abs(coef) > 1e-10:
                    lines.append(f"  {name:30s} {coef:+.4f}")

            lines.append("")

        # Top rules
        lines.append("Top Rules by Importance:")
        lines.append("-" * 40)

        rules = self.get_rules(exclude_zero_coef=True, sort_by="importance")

        for i, rule_dict in enumerate(rules[:20]):  # Top 20 rules
            lines.append(f"  [{i+1}] {rule_dict['rule']}")
            lines.append(
                f"      Coef: {rule_dict['coefficient']:+.4f}, "
                f"Support: {rule_dict['support']:.3f}, "
                f"Importance: {rule_dict['importance']:.4f}"
            )
            lines.append("")

        if len(rules) > 20:
            lines.append(f"  ... and {len(rules) - 20} more rules")
            lines.append("")

        # Feature importances
        lines.append("Feature Importances:")
        lines.append("-" * 40)

        feat_importance = sorted(
            zip(self.feature_names_in_, self.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        for name, imp in feat_importance[:10]:
            lines.append(f"  {name:30s} {imp:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_equation(self, precision: int = 4) -> str:
        """
        Get the model as a human-readable equation.

        Parameters
        ----------
        precision : int
            Number of decimal places for coefficients.

        Returns
        -------
        equation : str
            Model equation string.
        """
        check_is_fitted(self)

        terms = []

        # Intercept
        terms.append(f"{self.intercept_:.{precision}f}")

        # Linear terms
        if self.include_linear:
            for name, coef in zip(self.feature_names_in_, self.linear_coef_):
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:+.{precision}f} * {name}")

        # Rule terms
        for rule in self.rules_:
            if abs(rule.coefficient) > 1e-10:
                terms.append(f"{rule.coefficient:+.{precision}f} * [{rule}]")

        # Format nicely
        equation = "y = " + terms[0]
        for term in terms[1:]:
            equation += "\n    " + term

        return equation

    def visualize_rule(self, rule_idx: int):
        """
        Visualize a specific rule's effect.

        Parameters
        ----------
        rule_idx : int
            Index of the rule to visualize.

        Returns
        -------
        fig : matplotlib Figure
            Visualization of rule effect.
        """
        check_is_fitted(self)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        if rule_idx >= len(self.rules_):
            raise ValueError(
                f"rule_idx {rule_idx} out of range. "
                f"Only {len(self.rules_)} rules available."
            )

        rule = self.rules_[rule_idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Rule satisfaction distribution
        if hasattr(self, "_X_train"):
            satisfied = rule.evaluate(self._X_train)
            ax1 = axes[0]
            ax1.bar(["Satisfied", "Not Satisfied"], [satisfied.sum(), (~satisfied).sum()])
            ax1.set_ylabel("Count")
            ax1.set_title(f"Rule Satisfaction (Support: {rule.support:.3f})")

        # Plot 2: Feature distribution for rule conditions
        ax2 = axes[1]
        if len(rule.conditions) > 0:
            condition = rule.conditions[0]
            if hasattr(self, "_X_train"):
                feat_vals = self._X_train[:, condition.feature_idx]
                ax2.hist(feat_vals, bins=30, alpha=0.7, label="All samples")
                ax2.axvline(
                    condition.threshold,
                    color="r",
                    linestyle="--",
                    label=f"Threshold: {condition.threshold:.3g}",
                )
                ax2.set_xlabel(condition.feature_name)
                ax2.set_ylabel("Count")
                ax2.set_title(f"First Condition: {condition}")
                ax2.legend()

        fig.suptitle(f"Rule {rule_idx}: {rule}\nCoefficient: {rule.coefficient:+.4f}")
        plt.tight_layout()

        return fig


class RuleFitClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """
    RuleFit for classification.

    For binary classification, uses logistic regression on rule features.
    For multiclass, uses one-vs-rest strategy.

    Parameters
    ----------
    tree_generator : estimator or None, default=None
        The tree ensemble used to generate rules. If None, uses
        GradientBoostingClassifier with default parameters.

    n_estimators : int, default=100
        Number of trees in the ensemble (if tree_generator is None).

    tree_max_depth : int, default=3
        Maximum depth of trees (if tree_generator is None).

    max_rules : int or None, default=2000
        Maximum number of rules to extract.

    min_support : float, default=0.01
        Minimum fraction of samples that must satisfy a rule.

    max_support : float, default=0.99
        Maximum fraction of samples satisfying a rule.

    alpha : float or None, default=None
        Regularization strength (1/C for logistic regression).
        If None, selected via cross-validation.

    cv : int, default=5
        Number of cross-validation folds for alpha selection.

    include_linear : bool, default=True
        Whether to include original features (linear terms).

    standardize_linear : bool, default=True
        Whether to standardize linear features.

    winsorize_linear : float or None, default=0.025
        Winsorization quantile for linear features.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    n_jobs : int, default=None
        Number of parallel jobs.

    class_weight : dict, 'balanced', or None, default=None
        Weights for classes in the logistic regression step.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of classes.

    (Plus all attributes from RuleFitRegressor)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        tree_generator=None,
        n_estimators: int = 50,
        tree_max_depth: int = 3,
        max_rules: int | None = 300,
        min_support: float = 0.01,
        max_support: float = 0.99,
        alpha: float | None = None,
        cv: int = 3,
        include_linear: bool = True,
        standardize_linear: bool = True,
        winsorize_linear: float | None = 0.025,
        random_state: int | None = None,
        n_jobs: int | None = None,
        class_weight: dict | str | None = None,
    ):
        self.tree_generator = tree_generator
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.max_rules = max_rules
        self.min_support = min_support
        self.max_support = max_support
        self.alpha = alpha
        self.cv = cv
        self.include_linear = include_linear
        self.standardize_linear = standardize_linear
        self.winsorize_linear = winsorize_linear
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        """
        Fit the RuleFit classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
        feature_names : list of str, default=None
            Names for features.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for fitting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float64)

        # Encode classes
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        self.n_features_in_ = X.shape[1]

        # Set feature names
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(
                [f"x{i}" for i in range(self.n_features_in_)]
            )

        self._X_train = X

        # Step 1: Fit tree ensemble (classifier version)
        self.tree_generator_ = self._get_tree_generator()
        self.tree_generator_.fit(X, y_encoded, sample_weight=sample_weight)

        # Step 2: Extract rules (tree-based support avoids re-evaluating on X)
        rule_ensemble = extract_rules_from_ensemble(
            self.tree_generator_,
            list(self.feature_names_in_),
        )

        # Step 3: Filter and deduplicate
        rule_ensemble = rule_ensemble.deduplicate()
        rule_ensemble = rule_ensemble.filter_by_support(
            self.min_support, self.max_support
        )

        if self.max_rules is not None:
            rule_ensemble = rule_ensemble.limit_rules(self.max_rules)

        self.rule_ensemble_ = rule_ensemble
        self.n_rules_ = len(rule_ensemble)

        # Step 4: Build design matrix
        X_rules = rule_ensemble.transform(X)

        if self.include_linear:
            X_linear, self._linear_params = _preprocess_linear_features(
                X,
                winsorize=self.winsorize_linear,
                standardize=self.standardize_linear,
                fit=True,
            )
            X_combined = np.hstack([X_linear, X_rules])
        else:
            X_combined = X_rules
            self._linear_params = {}

        # Step 5: Fit Logistic Regression with L1 penalty
        if self.n_classes_ == 2:
            self._fit_binary(X_combined, y_encoded, sample_weight)
        else:
            self._fit_multiclass(X_combined, y_encoded, sample_weight)

        # Step 6: Assign coefficients to rules and compute importances
        self._assign_coefficients()
        self._compute_feature_importances()

        return self

    def _get_tree_generator(self):
        """Get or create the tree generator."""
        if self.tree_generator is not None:
            return clone(self.tree_generator)

        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.tree_max_depth,
            learning_rate=0.15,
            subsample=0.5,
            max_features="sqrt",
            random_state=self.random_state,
        )

    def _fit_binary(self, X_combined, y, sample_weight):
        """Fit binary classification."""
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        if self.alpha is None:
            model = LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                cv=self.cv,
                Cs=5,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=500,
                n_jobs=self.n_jobs,
            )
            model.fit(X_combined, y)

            self.alpha_ = 1.0 / model.C_[0]
            self.cv_results_ = {"Cs": model.Cs_, "scores": model.scores_}
        else:
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0 / self.alpha,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000,
            )
            model.fit(X_combined, y)

            self.alpha_ = self.alpha
            self.cv_results_ = {}

        self.coef_ = model.coef_.ravel()
        self.intercept_ = model.intercept_[0]
        self._logistic_model = model

    def _fit_multiclass(self, X_combined, y, sample_weight):
        """Fit multiclass classification with multinomial loss."""
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        if self.alpha is None:
            model = LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                cv=self.cv,
                Cs=5,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=500,
                n_jobs=self.n_jobs,
            )
            model.fit(X_combined, y)

            self.alpha_ = 1.0 / np.mean(model.C_)
            self.cv_results_ = {"Cs": model.Cs_, "scores": model.scores_}
        else:
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0 / self.alpha,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000,
            )
            model.fit(X_combined, y)

            self.alpha_ = self.alpha
            self.cv_results_ = {}

        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self._logistic_model = model

    def _assign_coefficients(self):
        """Assign coefficients from fitted model to rules."""
        if self.include_linear:
            if self.coef_.ndim == 1:
                self.linear_coef_ = self.coef_[: self.n_features_in_]
                self.rule_coef_ = self.coef_[self.n_features_in_ :]
            else:
                # Multiclass: average across classes for simplicity
                self.linear_coef_ = np.mean(
                    np.abs(self.coef_[:, : self.n_features_in_]), axis=0
                )
                self.rule_coef_ = np.mean(
                    np.abs(self.coef_[:, self.n_features_in_ :]), axis=0
                )
        else:
            self.linear_coef_ = np.zeros(self.n_features_in_)
            if self.coef_.ndim == 1:
                self.rule_coef_ = self.coef_
            else:
                self.rule_coef_ = np.mean(np.abs(self.coef_), axis=0)

        # Assign to rules
        for i, rule in enumerate(self.rule_ensemble_.rules):
            rule.coefficient = self.rule_coef_[i]

        self.rules_ = [
            r for r in self.rule_ensemble_.rules if abs(r.coefficient) > 1e-10
        ]
        self.n_rules_selected_ = len(self.rules_)

    def _compute_feature_importances(self):
        """Compute feature importances from coefficients."""
        importances = np.zeros(self.n_features_in_)

        # Linear term contributions
        if self.include_linear:
            importances += np.abs(self.linear_coef_)

        # Rule contributions
        for rule in self.rule_ensemble_.rules:
            if abs(rule.coefficient) > 1e-10:
                feature_indices = rule.feature_indices
                n_features_in_rule = len(feature_indices)

                if n_features_in_rule > 0:
                    importance_per_feature = abs(rule.coefficient) / n_features_in_rule

                    for feat_idx in feature_indices:
                        importances[feat_idx] += importance_per_feature

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def _transform_combined(self, X):
        """Transform X to combined linear + rule features."""
        X_rules = self.rule_ensemble_.transform(X)

        if self.include_linear:
            X_linear, _ = _preprocess_linear_features(
                X,
                winsorize=self.winsorize_linear,
                standardize=self.standardize_linear,
                fit=False,
                stored_params=self._linear_params,
            )
            return np.hstack([X_linear, X_rules])
        else:
            return X_rules

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RuleFitClassifier is expecting "
                f"{self.n_features_in_} features as input."
            )

        X_combined = self._transform_combined(X)
        y_pred = self._logistic_model.predict(X_combined)

        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        X_combined = self._transform_combined(X)
        return self._logistic_model.predict_proba(X_combined)

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Class log-probabilities.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        X_combined = self._transform_combined(X)
        return self._logistic_model.predict_log_proba(X_combined)

    def transform(self, X):
        """
        Transform X into rule features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_rules : ndarray of shape (n_samples, n_rules)
            Binary matrix of rule satisfactions.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        return self.rule_ensemble_.transform(X)

    def get_rules(
        self, exclude_zero_coef: bool = True, sort_by: str = "importance"
    ) -> list[dict]:
        """
        Get the extracted rules with their coefficients.

        Parameters
        ----------
        exclude_zero_coef : bool, default=True
            If True, only return rules with non-zero coefficients.
        sort_by : str, default='importance'
            How to sort rules: 'importance', 'support', 'coefficient', 'length'.

        Returns
        -------
        rules : list of dict
            Each dict contains rule information.
        """
        check_is_fitted(self)

        if exclude_zero_coef:
            rules = [r for r in self.rule_ensemble_.rules if abs(r.coefficient) > 1e-10]
        else:
            rules = list(self.rule_ensemble_.rules)

        # Sort
        if sort_by == "importance":
            rules = sorted(rules, key=lambda r: r.importance, reverse=True)
        elif sort_by == "support":
            rules = sorted(rules, key=lambda r: r.support, reverse=True)
        elif sort_by == "coefficient":
            rules = sorted(rules, key=lambda r: r.coefficient, reverse=True)
        elif sort_by == "length":
            rules = sorted(rules, key=lambda r: r.length)
        else:
            raise ValueError(f"Unknown sort_by: {sort_by}")

        return [r.to_dict() for r in rules]

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        check_is_fitted(self)

        lines = []
        lines.append("=" * 60)
        lines.append("RuleFit Classifier Summary")
        lines.append("=" * 60)
        lines.append("")

        # Model statistics
        lines.append("Model Statistics:")
        lines.append("-" * 40)
        lines.append(f"  Number of classes:        {self.n_classes_}")
        lines.append(f"  Total rules extracted:    {self.n_rules_}")
        lines.append(f"  Rules with non-zero coef: {self.n_rules_selected_}")
        lines.append(f"  Alpha (regularization):   {self.alpha_:.6f}")
        lines.append("")

        # Top rules
        lines.append("Top Rules by Importance:")
        lines.append("-" * 40)

        rules = self.get_rules(exclude_zero_coef=True, sort_by="importance")

        for i, rule_dict in enumerate(rules[:20]):
            lines.append(f"  [{i+1}] {rule_dict['rule']}")
            lines.append(
                f"      Coef: {rule_dict['coefficient']:+.4f}, "
                f"Support: {rule_dict['support']:.3f}"
            )
            lines.append("")

        if len(rules) > 20:
            lines.append(f"  ... and {len(rules) - 20} more rules")
            lines.append("")

        # Feature importances
        lines.append("Feature Importances:")
        lines.append("-" * 40)

        feat_importance = sorted(
            zip(self.feature_names_in_, self.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        for name, imp in feat_importance[:10]:
            lines.append(f"  {name:30s} {imp:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_equation(self, precision: int = 4) -> str:
        """
        Get the model as a human-readable equation.

        For binary classification only.

        Parameters
        ----------
        precision : int
            Number of decimal places for coefficients.

        Returns
        -------
        equation : str
            Model equation string.
        """
        check_is_fitted(self)

        if self.n_classes_ > 2:
            return "Equation format not supported for multiclass classification."

        terms = []

        # Intercept
        terms.append(f"{self.intercept_:.{precision}f}")

        # Linear terms
        if self.include_linear:
            for name, coef in zip(self.feature_names_in_, self.linear_coef_):
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:+.{precision}f} * {name}")

        # Rule terms
        for rule in self.rules_:
            if abs(rule.coefficient) > 1e-10:
                terms.append(f"{rule.coefficient:+.{precision}f} * [{rule}]")

        # Format nicely
        equation = "logit(P(y=1)) = " + terms[0]
        for term in terms[1:]:
            equation += "\n                " + term

        return equation


def _rulefit_structure(self, *, is_classifier: bool) -> dict[str, Any]:
    check_is_fitted(self)
    rules = self.get_rules(exclude_zero_coef=False, sort_by="importance")
    linear_terms: list[dict[str, Any]] = []
    if getattr(self, "include_linear", False):
        for name, coef in zip(self.feature_names_in_, self.linear_coef_):
            if abs(coef) > 0:
                linear_terms.append({"feature": str(name), "coefficient": float(coef)})
    return {
        "intercept": float(self.intercept_),
        "link": "logit" if is_classifier else "identity",
        "linear_terms": linear_terms,
        "rules": rules,
        "n_rules": int(self.n_rules_),
        "n_rules_selected": int(getattr(self, "n_rules_selected_", 0)),
        "alpha": float(getattr(self, "alpha_", 0.0)),
        "feature_importances": [
            {"feature": str(name), "importance": float(imp)}
            for name, imp in zip(self.feature_names_in_, self.feature_importances_)
        ],
    }


RuleFitRegressor._structure_type = "rules"
RuleFitRegressor._structure_content = lambda self: _rulefit_structure(self, is_classifier=False)
RuleFitClassifier._structure_type = "rules"
RuleFitClassifier._structure_content = lambda self: _rulefit_structure(self, is_classifier=True)
