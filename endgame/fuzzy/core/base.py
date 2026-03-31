"""Base classes for fuzzy learning algorithms.

Provides common functionality for all fuzzy systems including
rule management, membership function initialization, and sklearn compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.membership import (
    BaseMembershipFunction,
    GaussianMF,
    create_uniform_mfs,
)
from endgame.fuzzy.core.operators import BaseTNorm, MinTNorm, ProductTNorm


class BaseFuzzySystem(BaseEstimator):
    """Base class for all fuzzy inference systems.

    Provides common infrastructure for fuzzy rule management,
    membership function initialization, and interpretable output.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    n_mfs : int, default=3
        Number of membership functions per input variable.
    mf_type : str, default='gaussian'
        Type of membership function: 'gaussian', 'triangular', 'trapezoidal'.
    t_norm : str, default='product'
        T-norm for antecedent combination: 'min', 'product', 'lukasiewicz'.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        n_rules: int = 10,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.t_norm = t_norm
        self.random_state = random_state
        self.verbose = verbose

    def _get_t_norm(self) -> BaseTNorm:
        """Get t-norm operator instance."""
        if self.t_norm == "min":
            return MinTNorm()
        elif self.t_norm == "product":
            return ProductTNorm()
        else:
            from endgame.fuzzy.core.operators import _TNORM_MAP
            if self.t_norm in _TNORM_MAP:
                return _TNORM_MAP[self.t_norm]()
            raise ValueError(f"Unknown t-norm: {self.t_norm}")

    def _init_membership_functions(
        self,
        X: np.ndarray,
    ) -> list[list[BaseMembershipFunction]]:
        """Initialize membership functions from data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        list of list of BaseMembershipFunction
            mfs[feature_idx][mf_idx] gives the MF for that feature and term.
        """
        n_features = X.shape[1]
        mfs = []
        for j in range(n_features):
            x_min = float(np.min(X[:, j]))
            x_max = float(np.max(X[:, j]))
            # Add small padding to avoid edge effects
            padding = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
            feature_mfs = create_uniform_mfs(
                n_mfs=self.n_mfs,
                x_min=x_min - padding,
                x_max=x_max + padding,
                mf_type=self.mf_type,
            )
            mfs.append(feature_mfs)
        return mfs

    def _compute_firing_strengths(
        self,
        X: np.ndarray,
        antecedent_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute firing strengths for all rules.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        antecedent_indices : ndarray of shape (n_rules, n_features)
            For each rule and feature, the index of the MF to use.

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
            Firing strength of each rule for each sample.
        """
        n_samples = X.shape[0]
        n_rules = antecedent_indices.shape[0]
        n_features = X.shape[1]
        tnorm = self._get_t_norm()

        # Compute membership degrees for all features and MFs
        # memberships[j][k] = MF degrees for feature j, term k
        memberships = []
        for j in range(n_features):
            feature_memberships = []
            for k in range(self.n_mfs):
                feature_memberships.append(self.mfs_[j][k](X[:, j]))
            memberships.append(feature_memberships)

        # Compute firing strengths
        strengths = np.ones((n_samples, n_rules))
        for r in range(n_rules):
            for j in range(n_features):
                mf_idx = antecedent_indices[r, j]
                strengths[:, r] = tnorm(strengths[:, r], memberships[j][mf_idx])

        return strengths

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[FuzzySystem] {message}")

    @property
    def n_rules_(self) -> int:
        """Number of rules in the fitted system."""
        check_is_fitted(self, ["antecedent_indices_"])
        return self.antecedent_indices_.shape[0]

    def get_rules_str(self) -> str:
        """Get human-readable string representation of fuzzy rules.

        Returns
        -------
        str
            Multi-line string with one rule per line.
        """
        check_is_fitted(self, ["antecedent_indices_"])
        lines = []
        n_rules = self.antecedent_indices_.shape[0]
        n_features = self.antecedent_indices_.shape[1]
        term_names = ["low", "medium", "high", "very_low", "low_med",
                      "med_high", "very_high", "extra1", "extra2", "extra3"]

        for r in range(n_rules):
            antecedents = []
            for j in range(n_features):
                mf_idx = self.antecedent_indices_[r, j]
                term = term_names[mf_idx] if mf_idx < len(term_names) else f"term_{mf_idx}"
                fname = self.feature_names_[j] if hasattr(self, "feature_names_") else f"x{j}"
                antecedents.append(f"{fname} IS {term}")
            ant_str = " AND ".join(antecedents)
            if hasattr(self, "consequent_params_"):
                cons = self.consequent_params_[r]
                if cons.ndim == 0 or len(cons) == 1:
                    cons_str = f"{float(cons):.4f}"
                else:
                    cons_str = f"f(x) = {cons[-1]:.4f}"
                    for j in range(min(len(cons) - 1, n_features)):
                        cons_str += f" + {cons[j]:.4f}*x{j}"
            else:
                cons_str = "..."
            lines.append(f"Rule {r+1}: IF {ant_str} THEN {cons_str}")
        return "\n".join(lines)


class BaseFuzzyClassifier(BaseFuzzySystem, ClassifierMixin):
    """Base class for fuzzy classification systems.

    Adds predict_proba and class label handling on top of BaseFuzzySystem.
    """

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode class labels to integers."""
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        return y_encoded

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        raise NotImplementedError("Subclasses must implement predict_proba")


class BaseFuzzyRegressor(BaseFuzzySystem, RegressorMixin):
    """Base class for fuzzy regression systems.

    Provides standard predict interface for TSK-style output.
    """

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        raise NotImplementedError("Subclasses must implement predict")


class BaseFuzzyTransformer(BaseEstimator, TransformerMixin):
    """Base class for fuzzy feature transformation / selection.

    Follows sklearn TransformerMixin pattern with fit/transform.
    """

    def fit(self, X: Any, y: Any = None) -> BaseFuzzyTransformer:
        """Fit the transformer."""
        raise NotImplementedError("Subclasses must implement fit")

    def transform(self, X: Any) -> np.ndarray:
        """Transform the data."""
        raise NotImplementedError("Subclasses must implement transform")
