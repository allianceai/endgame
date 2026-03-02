"""Explainable Boosting Machine (EBM) Integration.

This module provides enhanced wrappers for InterpretML's Explainable Boosting
Machines (EBMs), adding additional functionality for integration with the
endgame library.

EBMs are interpretable machine learning models that combine modern ML techniques
(bagging, gradient boosting, automatic interaction detection) with the
transparency of Generalized Additive Models (GAMs).

Key features:
- Glass-box interpretability with competitive accuracy
- Automatic pairwise interaction detection
- Global and local explanations
- Editable by domain experts
- Support for mixed feature types (continuous, categorical)
- Missing value handling

Example usage:
    >>> from endgame.models import EBMClassifier, EBMRegressor
    >>> clf = EBMClassifier(interactions=10)
    >>> clf.fit(X_train, y_train)
    >>> clf.explain_global()
    >>> clf.explain_local(X_test[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    from interpret.glassbox import (
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
    )
    HAS_INTERPRET = True
except ImportError:
    HAS_INTERPRET = False
    ExplainableBoostingClassifier = None
    ExplainableBoostingRegressor = None


def _check_interpret_installed():
    """Raise ImportError if interpret is not installed."""
    if not HAS_INTERPRET:
        raise ImportError(
            "The 'interpret' package is required for EBM models. "
            "Install with: pip install endgame-ml[tabular]"
        )


class EBMBase(BaseEstimator):
    """Base class for EBM wrappers with common functionality.

    This class provides shared functionality for both classification and
    regression EBM models, including explanation methods and model inspection.
    """

    def __init__(
        self,
        # Feature configuration
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
        # Binning
        max_bins: int = 1024,
        max_interaction_bins: int = 64,
        # Interactions
        interactions: int | float | str | list = 10,
        exclude: list | None = None,
        # Training
        validation_size: float = 0.15,
        outer_bags: int = 14,
        inner_bags: int = 0,
        learning_rate: float = 0.015,
        greedy_ratio: float = 10.0,
        cyclic_progress: bool = False,
        smoothing_rounds: int = 75,
        interaction_smoothing_rounds: int = 75,
        max_rounds: int = 50000,
        early_stopping_rounds: int = 100,
        early_stopping_tolerance: float = 1e-5,
        # Regularization
        min_samples_leaf: int = 4,
        min_hessian: float = 0.0001,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        max_delta_step: float = 0.0,
        gain_scale: float = 5.0,
        # Categorical handling
        min_cat_samples: int = 10,
        cat_smooth: float = 10.0,
        missing: str = "separate",
        # Tree structure
        max_leaves: int = 2,
        monotone_constraints: list | None = None,
        # Parallelism
        n_jobs: int = -2,
        random_state: int | None = 42,
    ):
        _check_interpret_installed()

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins
        self.interactions = interactions
        self.exclude = exclude
        self.validation_size = validation_size
        self.outer_bags = outer_bags
        self.inner_bags = inner_bags
        self.learning_rate = learning_rate
        self.greedy_ratio = greedy_ratio
        self.cyclic_progress = cyclic_progress
        self.smoothing_rounds = smoothing_rounds
        self.interaction_smoothing_rounds = interaction_smoothing_rounds
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.min_samples_leaf = min_samples_leaf
        self.min_hessian = min_hessian
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_delta_step = max_delta_step
        self.gain_scale = gain_scale
        self.min_cat_samples = min_cat_samples
        self.cat_smooth = cat_smooth
        self.missing = missing
        self.max_leaves = max_leaves
        self.monotone_constraints = monotone_constraints
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _get_ebm_params(self) -> dict[str, Any]:
        """Get parameters for the underlying EBM model."""
        return {
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "max_bins": self.max_bins,
            "max_interaction_bins": self.max_interaction_bins,
            "interactions": self.interactions,
            "exclude": self.exclude,
            "validation_size": self.validation_size,
            "outer_bags": self.outer_bags,
            "inner_bags": self.inner_bags,
            "learning_rate": self.learning_rate,
            "greedy_ratio": self.greedy_ratio,
            "cyclic_progress": self.cyclic_progress,
            "smoothing_rounds": self.smoothing_rounds,
            "interaction_smoothing_rounds": self.interaction_smoothing_rounds,
            "max_rounds": self.max_rounds,
            "early_stopping_rounds": self.early_stopping_rounds,
            "early_stopping_tolerance": self.early_stopping_tolerance,
            "min_samples_leaf": self.min_samples_leaf,
            "min_hessian": self.min_hessian,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "max_delta_step": self.max_delta_step,
            "gain_scale": self.gain_scale,
            "min_cat_samples": self.min_cat_samples,
            "cat_smooth": self.cat_smooth,
            "missing": self.missing,
            "max_leaves": self.max_leaves,
            "monotone_constraints": self.monotone_constraints,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    @property
    def intercept_(self) -> NDArray:
        """Model intercept."""
        check_is_fitted(self, "_ebm")
        return self._ebm.intercept_

    @property
    def term_features_(self) -> list[tuple[int, ...]]:
        """Feature indices for each term."""
        check_is_fitted(self, "_ebm")
        return self._ebm.term_features_

    @property
    def term_scores_(self) -> list[NDArray]:
        """Scores for each term (lookup tables)."""
        check_is_fitted(self, "_ebm")
        return self._ebm.term_scores_

    @property
    def feature_names_in_(self) -> list[str]:
        """Feature names seen during fit."""
        if hasattr(self, "_feature_names_override"):
            return self._feature_names_override
        check_is_fitted(self, "_ebm")
        return self._ebm.feature_names_in_

    @feature_names_in_.setter
    def feature_names_in_(self, value):
        self._feature_names_override = value

    @property
    def feature_types_in_(self) -> list[str]:
        """Feature types detected during fit."""
        check_is_fitted(self, "_ebm")
        return self._ebm.feature_types_in_

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit."""
        check_is_fitted(self, "_ebm")
        return self._ebm.n_features_in_

    def explain_global(self, name: str | None = None):
        """Generate global explanations for the model.

        Returns an explanation object showing the contribution of each
        feature/term to the model's predictions across all data.

        Parameters
        ----------
        name : str, optional
            Name for the explanation.

        Returns
        -------
        explanation : EBMExplanation
            Global explanation object with visualization methods.
        """
        check_is_fitted(self, "_ebm")
        return self._ebm.explain_global(name=name)

    def explain_local(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        name: str | None = None,
    ):
        """Generate local explanations for specific predictions.

        Returns an explanation object showing how each feature contributed
        to the predictions for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.
        y : array-like of shape (n_samples,), optional
            True labels (for comparison in visualization).
        name : str, optional
            Name for the explanation.

        Returns
        -------
        explanation : EBMExplanation
            Local explanation object with visualization methods.
        """
        check_is_fitted(self, "_ebm")
        return self._ebm.explain_local(X, y, name=name)

    def term_importances(
        self,
        importance_type: str = "avg_weight",
    ) -> NDArray:
        """Get importance scores for each term.

        Parameters
        ----------
        importance_type : str, default="avg_weight"
            Type of importance to compute. Options:
            - "avg_weight": Average absolute contribution
            - "min_max": Range of contributions

        Returns
        -------
        importances : ndarray of shape (n_terms,)
            Importance score for each term.
        """
        check_is_fitted(self, "_ebm")
        return self._ebm.term_importances(importance_type=importance_type)

    def get_term_names(self) -> list[str]:
        """Get human-readable names for each term.

        Returns
        -------
        names : list of str
            Names like "feature_0" or "feature_0 x feature_1" for interactions.
        """
        check_is_fitted(self, "_ebm")
        names = []
        for term in self.term_features_:
            if len(term) == 1:
                names.append(self.feature_names_in_[term[0]])
            else:
                names.append(" x ".join(self.feature_names_in_[i] for i in term))
        return names

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importances aggregated across all terms.

        For interactions, importance is split equally among participating features.

        Returns
        -------
        importances : dict
            Mapping from feature name to importance score.
        """
        check_is_fitted(self, "_ebm")
        importances = {name: 0.0 for name in self.feature_names_in_}
        term_imp = self.term_importances()

        for term, imp in zip(self.term_features_, term_imp):
            share = imp / len(term)
            for idx in term:
                importances[self.feature_names_in_[idx]] += share

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    @property
    def feature_importances_(self) -> NDArray:
        """Feature importances as numpy array (sklearn compatible)."""
        check_is_fitted(self, "_ebm")
        imp_dict = self.get_feature_importances()
        return np.array([imp_dict[name] for name in self.feature_names_in_])

    def monotonize(
        self,
        term: int | str,
        increasing: bool = True,
        passthrough: float = 0.0,
    ) -> EBMBase:
        """Enforce monotonicity on a univariate term.

        Parameters
        ----------
        term : int or str
            Index or name of the term to monotonize.
        increasing : bool, default=True
            If True, enforce increasing monotonicity; if False, decreasing.
        passthrough : float, default=0.0
            Fraction of violations to allow.

        Returns
        -------
        self : EBMBase
            The model with monotonicity enforced.
        """
        check_is_fitted(self, "_ebm")
        self._ebm.monotonize(term, increasing=increasing, passthrough=passthrough)
        return self

    def remove_terms(self, terms: list[int | str]) -> EBMBase:
        """Remove terms from the model.

        Parameters
        ----------
        terms : list of int or str
            Indices or names of terms to remove.

        Returns
        -------
        self : EBMBase
            The model with terms removed.
        """
        check_is_fitted(self, "_ebm")
        self._ebm.remove_terms(terms)
        return self

    def to_json(self, file: str, detail: str = "all", indent: int = 2) -> None:
        """Export model to JSON format.

        Parameters
        ----------
        file : str
            Path to save JSON.
        detail : str, default="all"
            Level of detail ("all", "minimal", etc.).
        indent : int, default=2
            JSON indentation.

        Returns
        -------
        None
        """
        check_is_fitted(self, "_ebm")
        self._ebm.to_json(file, detail=detail, indent=indent)

    def get_histogram(self, term: int | str) -> tuple[NDArray, NDArray]:
        """Get the histogram (bin edges and scores) for a term.

        Parameters
        ----------
        term : int or str
            Index or name of the term.

        Returns
        -------
        bin_edges : ndarray
            Edges of bins for the term.
        scores : ndarray
            Score contribution for each bin.
        """
        check_is_fitted(self, "_ebm")

        if isinstance(term, str):
            term_names = self.get_term_names()
            term = term_names.index(term)

        term_features = self.term_features_[term]
        if len(term_features) > 1:
            raise ValueError("get_histogram only works for univariate terms")

        scores = self.term_scores_[term]
        bins = self._ebm.bins_[term_features[0]][0]

        return bins, scores

    def predict_contributions(self, X: ArrayLike) -> NDArray:
        """Get per-sample, per-term contributions to predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to get contributions for.

        Returns
        -------
        contributions : ndarray of shape (n_samples, n_terms + 1)
            Contribution of each term to each sample's prediction.
            Last column is the intercept.
        """
        check_is_fitted(self, "_ebm")
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")

        # Get local explanation to extract contributions
        local_exp = self._ebm.explain_local(X)

        n_samples = X.shape[0]
        n_terms = len(self.term_features_)

        contributions = np.zeros((n_samples, n_terms + 1))

        # Extract contributions from explanation data
        for i in range(n_samples):
            data = local_exp.data(i)
            if "scores" in data:
                contributions[i, :-1] = data["scores"]
            contributions[i, -1] = self.intercept_[0] if len(self.intercept_.shape) > 0 else self.intercept_

        return contributions


class EBMClassifier(ClassifierMixin, EBMBase):
    """Explainable Boosting Machine for Classification.

    An interpretable classifier that combines the accuracy of gradient boosting
    with the transparency of Generalized Additive Models (GAMs).

    Parameters
    ----------
    feature_names : list of str, optional
        Names for features. If None, uses default naming.
    feature_types : list of str, optional
        Types for features ("continuous", "nominal", "ordinal").
    max_bins : int, default=1024
        Maximum number of bins for continuous features.
    max_interaction_bins : int, default=64
        Maximum bins for interaction terms.
    interactions : int, float, str, or list, default=10
        Number or specification of interaction terms to detect.
        Can be an integer (number of interactions), float (fraction),
        string like "3x" (multiple of features), or explicit list.
    exclude : list, optional
        Features or interactions to exclude.
    validation_size : float, default=0.15
        Fraction of data to use for validation during training.
    outer_bags : int, default=14
        Number of outer bags for ensembling.
    inner_bags : int, default=0
        Number of inner bags (0 means no inner bagging).
    learning_rate : float, default=0.015
        Learning rate for boosting.
    greedy_ratio : float, default=10.0
        Ratio controlling greedy vs cyclic feature selection.
    cyclic_progress : bool, default=False
        If True, use cyclic progress; if False, use greedy.
    smoothing_rounds : int, default=75
        Number of smoothing rounds for main effects.
    interaction_smoothing_rounds : int, default=75
        Number of smoothing rounds for interactions.
    max_rounds : int, default=50000
        Maximum number of boosting rounds.
    early_stopping_rounds : int, default=100
        Stop if no improvement after this many rounds.
    early_stopping_tolerance : float, default=1e-5
        Tolerance for early stopping.
    min_samples_leaf : int, default=4
        Minimum samples in a leaf.
    min_hessian : float, default=0.0001
        Minimum hessian in a leaf.
    reg_alpha : float, default=0.0
        L1 regularization.
    reg_lambda : float, default=0.0
        L2 regularization.
    max_delta_step : float, default=0.0
        Maximum delta step (0 means no limit).
    gain_scale : float, default=5.0
        Scale factor for gain computation.
    min_cat_samples : int, default=10
        Minimum samples for categorical bins.
    cat_smooth : float, default=10.0
        Smoothing for categorical features.
    missing : str, default="separate"
        How to handle missing values ("separate", "min", "max").
    max_leaves : int, default=2
        Maximum leaves per tree (2 = stumps).
    monotone_constraints : list, optional
        Monotonicity constraints per feature (-1, 0, 1).
    n_jobs : int, default=-2
        Number of jobs for parallel processing.
    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Feature names.
    feature_types_in_ : list of str
        Detected feature types.
    intercept_ : ndarray
        Model intercept.
    term_features_ : list of tuple
        Feature indices for each term.
    term_scores_ : list of ndarray
        Score lookup tables for each term.

    Examples
    --------
    >>> from endgame.models import EBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = EBMClassifier(interactions=5)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    0.98
    >>> global_exp = clf.explain_global()
    >>> local_exp = clf.explain_local(X[:5])
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
        max_bins: int = 512,
        max_interaction_bins: int = 32,
        interactions: int | float | str | list = 10,
        exclude: list | None = None,
        validation_size: float = 0.15,
        outer_bags: int = 8,
        inner_bags: int = 0,
        learning_rate: float = 0.02,
        greedy_ratio: float = 10.0,
        cyclic_progress: bool = False,
        smoothing_rounds: int = 50,
        interaction_smoothing_rounds: int = 50,
        max_rounds: int = 25000,
        early_stopping_rounds: int = 50,
        early_stopping_tolerance: float = 1e-5,
        min_samples_leaf: int = 4,
        min_hessian: float = 0.0001,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        max_delta_step: float = 0.0,
        gain_scale: float = 5.0,
        min_cat_samples: int = 10,
        cat_smooth: float = 10.0,
        missing: str = "separate",
        max_leaves: int = 2,
        monotone_constraints: list | None = None,
        n_jobs: int = -2,
        random_state: int | None = 42,
    ):
        super().__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            greedy_ratio=greedy_ratio,
            cyclic_progress=cyclic_progress,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            min_samples_leaf=min_samples_leaf,
            min_hessian=min_hessian,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_delta_step=max_delta_step,
            gain_scale=gain_scale,
            min_cat_samples=min_cat_samples,
            cat_smooth=cat_smooth,
            missing=missing,
            max_leaves=max_leaves,
            monotone_constraints=monotone_constraints,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> EBMClassifier:
        """Fit the EBM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : EBMClassifier
            Fitted classifier.
        """
        # Extract feature names from DataFrame before check_X_y converts to array
        if self.feature_names is None and hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=None, ensure_all_finite="allow-nan")

        self._ebm = ExplainableBoostingClassifier(**self._get_ebm_params())
        self._ebm.fit(X, y, sample_weight=sample_weight)

        self.classes_ = self._ebm.classes_

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, "_ebm")
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return self._ebm.predict(X)

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, "_ebm")
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return self._ebm.predict_proba(X)

    def decision_function(self, X: ArrayLike) -> NDArray:
        """Compute decision function values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        decision : ndarray
            Decision function values.
        """
        check_is_fitted(self, "_ebm")
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return self._ebm.decision_function(X)


class EBMRegressor(EBMBase, RegressorMixin):
    """Explainable Boosting Machine for Regression.

    An interpretable regressor that combines the accuracy of gradient boosting
    with the transparency of Generalized Additive Models (GAMs).

    Parameters
    ----------
    feature_names : list of str, optional
        Names for features. If None, uses default naming.
    feature_types : list of str, optional
        Types for features ("continuous", "nominal", "ordinal").
    max_bins : int, default=1024
        Maximum number of bins for continuous features.
    max_interaction_bins : int, default=64
        Maximum bins for interaction terms.
    interactions : int, float, str, or list, default=10
        Number or specification of interaction terms to detect.
    exclude : list, optional
        Features or interactions to exclude.
    validation_size : float, default=0.15
        Fraction of data to use for validation during training.
    outer_bags : int, default=14
        Number of outer bags for ensembling.
    inner_bags : int, default=0
        Number of inner bags.
    learning_rate : float, default=0.015
        Learning rate for boosting.
    greedy_ratio : float, default=10.0
        Ratio controlling greedy vs cyclic feature selection.
    cyclic_progress : bool, default=False
        If True, use cyclic progress.
    smoothing_rounds : int, default=75
        Number of smoothing rounds.
    interaction_smoothing_rounds : int, default=75
        Number of smoothing rounds for interactions.
    max_rounds : int, default=50000
        Maximum number of boosting rounds.
    early_stopping_rounds : int, default=100
        Stop if no improvement after this many rounds.
    early_stopping_tolerance : float, default=1e-5
        Tolerance for early stopping.
    min_samples_leaf : int, default=4
        Minimum samples in a leaf.
    min_hessian : float, default=0.0001
        Minimum hessian in a leaf.
    reg_alpha : float, default=0.0
        L1 regularization.
    reg_lambda : float, default=0.0
        L2 regularization.
    max_delta_step : float, default=0.0
        Maximum delta step.
    gain_scale : float, default=5.0
        Scale factor for gain computation.
    min_cat_samples : int, default=10
        Minimum samples for categorical bins.
    cat_smooth : float, default=10.0
        Smoothing for categorical features.
    missing : str, default="separate"
        How to handle missing values.
    max_leaves : int, default=2
        Maximum leaves per tree.
    monotone_constraints : list, optional
        Monotonicity constraints per feature.
    n_jobs : int, default=-2
        Number of jobs for parallel processing.
    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Feature names.
    feature_types_in_ : list of str
        Detected feature types.
    intercept_ : float
        Model intercept.
    term_features_ : list of tuple
        Feature indices for each term.
    term_scores_ : list of ndarray
        Score lookup tables for each term.

    Examples
    --------
    >>> from endgame.models import EBMRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = EBMRegressor(interactions=10)
    >>> reg.fit(X, y)
    >>> reg.score(X, y)
    0.72
    >>> importance = reg.get_feature_importances()
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
        max_bins: int = 1024,
        max_interaction_bins: int = 64,
        interactions: int | float | str | list = 10,
        exclude: list | None = None,
        validation_size: float = 0.15,
        outer_bags: int = 8,
        inner_bags: int = 0,
        learning_rate: float = 0.02,
        greedy_ratio: float = 10.0,
        cyclic_progress: bool = False,
        smoothing_rounds: int = 50,
        interaction_smoothing_rounds: int = 50,
        max_rounds: int = 25000,
        early_stopping_rounds: int = 50,
        early_stopping_tolerance: float = 1e-5,
        min_samples_leaf: int = 4,
        min_hessian: float = 0.0001,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        max_delta_step: float = 0.0,
        gain_scale: float = 5.0,
        min_cat_samples: int = 10,
        cat_smooth: float = 10.0,
        missing: str = "separate",
        max_leaves: int = 2,
        monotone_constraints: list | None = None,
        n_jobs: int = -2,
        random_state: int | None = 42,
    ):
        super().__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            greedy_ratio=greedy_ratio,
            cyclic_progress=cyclic_progress,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            min_samples_leaf=min_samples_leaf,
            min_hessian=min_hessian,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_delta_step=max_delta_step,
            gain_scale=gain_scale,
            min_cat_samples=min_cat_samples,
            cat_smooth=cat_smooth,
            missing=missing,
            max_leaves=max_leaves,
            monotone_constraints=monotone_constraints,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> EBMRegressor:
        """Fit the EBM regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : EBMRegressor
            Fitted regressor.
        """
        # Extract feature names from DataFrame before check_X_y converts to array
        if self.feature_names is None and hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=None, y_numeric=True, ensure_all_finite="allow-nan")

        self._ebm = ExplainableBoostingRegressor(**self._get_ebm_params())
        self._ebm.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "_ebm")
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return self._ebm.predict(X)


def show_explanation(explanation, share_graphs: bool = False):
    """Display an EBM explanation in a dashboard.

    This is a convenience function that wraps interpret's show() function.

    Parameters
    ----------
    explanation : EBMExplanation
        Explanation from explain_global() or explain_local().
    share_graphs : bool, default=False
        If True, link axes across graphs.

    Returns
    -------
    None
        Opens an interactive dashboard.
    """
    try:
        from interpret import show
        show(explanation, share_graphs=share_graphs)
    except ImportError:
        raise ImportError(
            "The 'interpret' package is required. "
            "Install with: pip install endgame-ml[tabular]"
        )
