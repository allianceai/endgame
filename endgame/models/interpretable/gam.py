"""pyGAM: Generalized Additive Models wrapper.

pyGAM provides fast and flexible GAM fitting with various spline types
and automatic smoothness selection. This module wraps pyGAM for
seamless integration with sklearn pipelines.

References
----------
- Hastie & Tibshirani "Generalized Additive Models" (1990)
- https://github.com/dswah/pyGAM

Example
-------
>>> from endgame.models.interpretable import GAMClassifier
>>> clf = GAMClassifier(n_splines=25, lam=0.6)
>>> clf.fit(X_train, y_train)
>>> clf.summary()
>>> proba = clf.predict_proba(X_test)
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    from pygam import GAM, LinearGAM, LogisticGAM, f, l, s, te  # spline, factor, linear, tensor
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False
    LogisticGAM = None
    LinearGAM = None
    s = None


def _check_pygam():
    if not HAS_PYGAM:
        raise ImportError(
            "The 'pygam' package is required for GAM models. "
            "Install with: pip install pygam"
        )


class GAMClassifier(ClassifierMixin, BaseEstimator):
    """Generalized Additive Model Classifier.

    GAMs model the target as a sum of smooth functions of individual
    features, providing interpretability through feature-level effect plots.

    Uses pyGAM's LogisticGAM for binary classification.

    Parameters
    ----------
    n_splines : int, default=25
        Number of splines for each smoothing term.
        More splines allow more complex shapes but risk overfitting.

    spline_order : int, default=3
        Order of spline polynomials (3 = cubic splines).

    lam : float or "auto", default="auto"
        Smoothing penalty. Higher values produce smoother fits.
        If "auto", uses grid search to find optimal lambda.

    lam_search : int, default=20
        Number of lambda values to try in grid search.

    constraints : str or None, default=None
        Global constraint for all terms:
        - None: No constraint
        - "monotonic_inc": Monotonically increasing
        - "monotonic_dec": Monotonically decreasing
        - "convex": Convex shape
        - "concave": Concave shape

    max_iter : int, default=100
        Maximum iterations for fitting.

    tol : float, default=1e-4
        Convergence tolerance.

    verbose : bool, default=False
        Whether to print fitting progress.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.

    n_features_in_ : int
        Number of features.

    gam_ : LogisticGAM
        Fitted pyGAM model.

    feature_importances_ : ndarray
        Importance scores based on deviance explained.

    Examples
    --------
    >>> from endgame.models.interpretable import GAMClassifier
    >>> clf = GAMClassifier(n_splines=25)
    >>> clf.fit(X_train, y_train)
    >>> clf.summary()
    >>> proba = clf.predict_proba(X_test)
    >>> # Plot partial dependence
    >>> clf.plot_partial_dependence(0)  # Feature 0
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_splines: int = 25,
        spline_order: int = 3,
        lam: float | str = "auto",
        lam_search: int = 20,
        constraints: str | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.lam_search = lam_search
        self.constraints = constraints
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "GAMClassifier":
        """Fit the GAM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names for features.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self : GAMClassifier
        """
        _check_pygam()

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        if len(self.classes_) != 2:
            raise ValueError("GAMClassifier only supports binary classification.")

        # Feature names
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Build term list
        terms = self._build_terms()

        # Create GAM
        self.gam_ = LogisticGAM(
            terms,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )

        # Fit with optional grid search
        if self.lam == "auto":
            lam_values = np.logspace(-3, 3, self.lam_search)
            self.gam_.gridsearch(X, y_encoded, lam=lam_values, progress=self.verbose)
        else:
            self.gam_.fit(X, y_encoded, weights=sample_weight)

        self._X_train = X
        self._y_train = y_encoded
        self._importances_computed = False

        return self

    def _build_terms(self):
        """Build term list for GAM."""
        terms = []

        for i in range(self.n_features_in_):
            term = s(
                i,
                n_splines=self.n_splines,
                spline_order=self.spline_order,
                constraints=self.constraints,
            )
            terms.append(term)

        if len(terms) == 1:
            return terms[0]

        result = terms[0]
        for t in terms[1:]:
            result = result + t

        return result

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on partial effect variance (lazy)."""
        check_is_fitted(self, "gam_")
        if not self._importances_computed:
            importances = np.zeros(self.n_features_in_)
            for i in range(self.n_features_in_):
                try:
                    effects = self.gam_.partial_dependence(i, self._X_train)
                    importances[i] = np.std(effects)
                except Exception:
                    importances[i] = 0.0
            total = importances.sum()
            if total > 0:
                importances /= total
            self._feature_importances = importances
            self._importances_computed = True
        return self._feature_importances

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self, "gam_")
        X = check_array(X)

        proba_1 = self.gam_.predict_proba(X)
        proba_0 = 1 - proba_1

        return np.column_stack([proba_0, proba_1])

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "gam_")
        X = check_array(X)

        y_pred = self.gam_.predict(X)
        return self._label_encoder.inverse_transform(y_pred.astype(int))

    def partial_dependence(self, feature_idx: int, X: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """Get partial dependence for a feature.

        Parameters
        ----------
        feature_idx : int
            Index of feature.
        X : array-like, optional
            Data to use for grid. If None, uses equally spaced grid.

        Returns
        -------
        grid : ndarray
            Feature values.
        effects : ndarray
            Partial dependence values.
        """
        check_is_fitted(self, "gam_")

        if X is not None:
            XX = self.gam_.generate_X_grid(term=feature_idx)
            grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), len(XX))
            XX[:, feature_idx] = grid
        else:
            XX = self.gam_.generate_X_grid(term=feature_idx)
            grid = XX[:, feature_idx]

        effects = self.gam_.partial_dependence(feature_idx, XX)

        return grid, effects

    def plot_partial_dependence(self, feature_idx: int, X: np.ndarray = None):
        """Plot partial dependence for a feature.

        Parameters
        ----------
        feature_idx : int
            Index of feature.
        X : array-like, optional
            Data for computing feature range.

        Returns
        -------
        fig : matplotlib Figure
        """
        check_is_fitted(self, "gam_")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        grid, effects = self.partial_dependence(feature_idx, X)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot effect and confidence interval
        ax.plot(grid, effects, "b-", linewidth=2, label="Effect")

        # Confidence interval
        XX = self.gam_.generate_X_grid(term=feature_idx)
        try:
            ci = self.gam_.confidence_intervals(XX, width=0.95)
            ax.fill_between(
                grid,
                ci[:, 0, feature_idx] if ci.ndim == 3 else ci[:, 0],
                ci[:, 1, feature_idx] if ci.ndim == 3 else ci[:, 1],
                alpha=0.2,
                label="95% CI",
            )
        except Exception:
            pass  # CI computation may fail

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(self.feature_names_[feature_idx])
        ax.set_ylabel("Partial Effect (log-odds)")
        ax.set_title(f"Partial Dependence: {self.feature_names_[feature_idx]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """Get model summary.

        Returns
        -------
        summary : str
            Formatted model summary.
        """
        check_is_fitted(self, "gam_")

        lines = []
        lines.append("=" * 60)
        lines.append("GAM Classifier Summary")
        lines.append("=" * 60)
        lines.append("")

        # Model info
        lines.append("Model Information:")
        lines.append("-" * 40)
        lines.append(f"  Number of features: {self.n_features_in_}")
        lines.append(f"  Splines per feature: {self.n_splines}")

        stats = self.gam_.statistics_
        lines.append(f"  Pseudo R-squared: {stats.get('pseudo_r2', {}).get('explained_deviance', 'N/A'):.4f}")
        lines.append(f"  Deviance: {stats.get('deviance', 'N/A'):.4f}")
        lines.append(f"  AIC: {stats.get('AIC', 'N/A'):.4f}")

        lines.append("")
        lines.append("Feature Importances:")
        lines.append("-" * 40)

        sorted_idx = np.argsort(self.feature_importances_)[::-1]
        for i in sorted_idx:
            lines.append(f"  {self.feature_names_[i]:30s} {self.feature_importances_[i]:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class GAMRegressor(RegressorMixin, BaseEstimator):
    """Generalized Additive Model Regressor.

    Uses pyGAM's LinearGAM for regression.

    Parameters
    ----------
    n_splines : int, default=25
        Number of splines per feature.

    spline_order : int, default=3
        Spline polynomial order.

    lam : float or "auto", default="auto"
        Smoothing penalty.

    lam_search : int, default=20
        Lambda grid search size.

    constraints : str or None, default=None
        Shape constraints.

    max_iter : int, default=100
        Maximum iterations.

    tol : float, default=1e-4
        Convergence tolerance.

    verbose : bool, default=False
        Verbosity.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.

    gam_ : LinearGAM
        Fitted model.

    feature_importances_ : ndarray
        Feature importance scores.

    Examples
    --------
    >>> from endgame.models.interpretable import GAMRegressor
    >>> reg = GAMRegressor(n_splines=25)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_splines: int = 25,
        spline_order: int = 3,
        lam: float | str = "auto",
        lam_search: int = 20,
        constraints: str | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.lam_search = lam_search
        self.constraints = constraints
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "GAMRegressor":
        """Fit the GAM regressor."""
        _check_pygam()

        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]

        # Feature names
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Build terms
        terms = self._build_terms()

        # Create GAM
        self.gam_ = LinearGAM(
            terms,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )

        # Fit
        if self.lam == "auto":
            lam_values = np.logspace(-3, 3, self.lam_search)
            self.gam_.gridsearch(X, y, lam=lam_values, progress=self.verbose)
        else:
            self.gam_.fit(X, y, weights=sample_weight)

        self._X_train = X
        self._importances_computed = False

        return self

    def _build_terms(self):
        """Build term list."""
        terms = []

        for i in range(self.n_features_in_):
            term = s(
                i,
                n_splines=self.n_splines,
                spline_order=self.spline_order,
                constraints=self.constraints,
            )
            terms.append(term)

        if len(terms) == 1:
            return terms[0]

        result = terms[0]
        for t in terms[1:]:
            result = result + t

        return result

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on partial effect variance (lazy)."""
        check_is_fitted(self, "gam_")
        if not self._importances_computed:
            importances = np.zeros(self.n_features_in_)
            for i in range(self.n_features_in_):
                try:
                    effects = self.gam_.partial_dependence(i, self._X_train)
                    importances[i] = np.std(effects)
                except Exception:
                    importances[i] = 0.0
            total = importances.sum()
            if total > 0:
                importances /= total
            self._feature_importances = importances
            self._importances_computed = True
        return self._feature_importances

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        check_is_fitted(self, "gam_")
        X = check_array(X)

        return self.gam_.predict(X)

    def partial_dependence(self, feature_idx: int, X: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """Get partial dependence for a feature."""
        check_is_fitted(self, "gam_")

        if X is not None:
            XX = self.gam_.generate_X_grid(term=feature_idx)
            grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), len(XX))
            XX[:, feature_idx] = grid
        else:
            XX = self.gam_.generate_X_grid(term=feature_idx)
            grid = XX[:, feature_idx]

        effects = self.gam_.partial_dependence(feature_idx, XX)

        return grid, effects

    def plot_partial_dependence(self, feature_idx: int, X: np.ndarray = None):
        """Plot partial dependence."""
        check_is_fitted(self, "gam_")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        grid, effects = self.partial_dependence(feature_idx, X)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grid, effects, "b-", linewidth=2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(self.feature_names_[feature_idx])
        ax.set_ylabel("Partial Effect")
        ax.set_title(f"Partial Dependence: {self.feature_names_[feature_idx]}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """Get model summary."""
        check_is_fitted(self, "gam_")

        lines = []
        lines.append("=" * 60)
        lines.append("GAM Regressor Summary")
        lines.append("=" * 60)
        lines.append("")

        lines.append("Model Information:")
        lines.append("-" * 40)
        lines.append(f"  Number of features: {self.n_features_in_}")
        lines.append(f"  Splines per feature: {self.n_splines}")

        stats = self.gam_.statistics_
        lines.append(f"  R-squared: {stats.get('pseudo_r2', {}).get('explained_deviance', 'N/A'):.4f}")
        lines.append(f"  GCV: {stats.get('GCV', 'N/A'):.4f}")
        lines.append(f"  AIC: {stats.get('AIC', 'N/A'):.4f}")

        lines.append("")
        lines.append("Feature Importances:")
        lines.append("-" * 40)

        sorted_idx = np.argsort(self.feature_importances_)[::-1]
        for i in sorted_idx:
            lines.append(f"  {self.feature_names_[i]:30s} {self.feature_importances_[i]:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
