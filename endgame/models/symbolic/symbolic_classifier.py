"""Symbolic Classification via logistic transformation of symbolic regression.

Binary: fits symbolic regression on log-odds, applies sigmoid.
Multiclass: one-vs-rest with softmax over symbolic regressors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from endgame.models.symbolic.symbolic_regressor import SymbolicRegressor
from typing import Any


class SymbolicClassifier(GlassboxMixin, BaseEstimator, ClassifierMixin):
    """Symbolic Classification via logistic transformation of symbolic regression.

    For binary classification, fits a symbolic regression model to the log-odds
    and applies sigmoid transformation for probabilities.

    For multiclass, uses one-vs-rest strategy with multiple symbolic regressors.

    Parameters
    ----------
    All parameters from SymbolicRegressor are accepted.

    threshold : float, default=0.5
        Classification threshold for binary classification.

    Attributes
    ----------
    model_ : SymbolicRegressor or list of SymbolicRegressor
        Underlying symbolic regressor(s).
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        preset: str = "default",
        operators: str | dict[str, list[str]] = "scientific",
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        niterations: int | None = None,
        maxsize: int | None = None,
        maxdepth: int | None = None,
        populations: int | None = None,
        population_size: int | None = None,
        parsimony: float | None = None,
        model_selection: str = "best",
        constraints: dict | None = None,
        nested_constraints: dict | None = None,
        denoise: bool = False,
        select_k_features: int | None = None,
        turbo: bool = False,
        parallelism: str = "multithreading",
        procs: int | None = None,
        random_state: int | None = None,
        verbosity: int = 0,
        temp_equation_file: bool = True,
        output_directory: str | None = None,
        threshold: float = 0.5,
    ):
        self.preset = preset
        self.operators = operators
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.niterations = niterations
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.populations = populations
        self.population_size = population_size
        self.parsimony = parsimony
        self.model_selection = model_selection
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.denoise = denoise
        self.select_k_features = select_k_features
        self.turbo = turbo
        self.parallelism = parallelism
        self.procs = procs
        self.random_state = random_state
        self.verbosity = verbosity
        self.temp_equation_file = temp_equation_file
        self.output_directory = output_directory
        self.threshold = threshold

    def _create_regressor(self) -> SymbolicRegressor:
        return SymbolicRegressor(
            preset=self.preset,
            operators=self.operators,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            niterations=self.niterations,
            maxsize=self.maxsize,
            maxdepth=self.maxdepth,
            populations=self.populations,
            population_size=self.population_size,
            parsimony=self.parsimony,
            model_selection=self.model_selection,
            loss="L2DistLoss()",
            constraints=self.constraints,
            nested_constraints=self.nested_constraints,
            denoise=self.denoise,
            select_k_features=self.select_k_features,
            turbo=self.turbo,
            parallelism=self.parallelism,
            procs=self.procs,
            random_state=self.random_state,
            verbosity=self.verbosity,
            temp_equation_file=self.temp_equation_file,
            output_directory=self.output_directory,
        )

    def fit(self, X, y, **fit_params) -> SymbolicClassifier:
        """Fit symbolic classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])

        if self.n_classes_ == 2:
            y_continuous = y_encoded * 6 - 3
            self.model_ = self._create_regressor()
            self.model_.fit(X, y_continuous, **fit_params)
        else:
            self.model_ = []
            for c in range(self.n_classes_):
                y_binary = (y_encoded == c).astype(float) * 6 - 3
                if self.verbosity > 0:
                    print(f"Fitting model for class {self.classes_[c]}...")
                model = self._create_regressor()
                model.fit(X, y_binary, **fit_params)
                self.model_.append(model)

        return self

    def predict_proba(self, X) -> NDArray:
        check_is_fitted(self, "model_")
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        if self.n_classes_ == 2:
            logits = self.model_.predict(X)
            p1 = sigmoid(logits)
            return np.column_stack([1 - p1, p1])
        else:
            scores = np.column_stack([m.predict(X) for m in self.model_])
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X) -> NDArray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def get_best_equation(self, class_idx: int = 1) -> str:
        check_is_fitted(self, "model_")
        if self.n_classes_ == 2:
            return self.model_.get_best_equation()
        if class_idx >= len(self.model_):
            raise ValueError(f"class_idx must be < {len(self.model_)}")
        return self.model_[class_idx].get_best_equation()

    def sympy(self, class_idx: int = 1):
        check_is_fitted(self, "model_")
        if self.n_classes_ == 2:
            return self.model_.sympy()
        if class_idx >= len(self.model_):
            raise ValueError(f"class_idx must be < {len(self.model_)}")
        return self.model_[class_idx].sympy()

    @property
    def feature_importances_(self) -> NDArray:
        check_is_fitted(self, "model_")
        if self.n_classes_ == 2:
            return self.model_.feature_importances_
        return np.mean([m.feature_importances_ for m in self.model_], axis=0)

    def summary(self) -> str:
        check_is_fitted(self, "model_")
        lines = ["Symbolic Classifier Results", "=" * 50]
        if self.n_classes_ == 2:
            lines.append(f"Binary classification (positive class: {self.classes_[1]})")
            lines.append(self.model_.summary())
        else:
            lines.append(f"Multiclass classification ({self.n_classes_} classes)")
            for cls, model in zip(self.classes_, self.model_):
                lines.append(f"\n--- Class {cls} ---")
                lines.append(model.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:
        if hasattr(self, "model_"):
            if self.n_classes_ == 2:
                return f"SymbolicClassifier(equation={self.model_.best_equation_})"
            return f"SymbolicClassifier(n_classes={self.n_classes_})"
        return f"SymbolicClassifier(preset={self.preset!r})"

    _structure_type = "symbolic"

    def _structure_content(self) -> dict[str, Any]:
        check_is_fitted(self, "model_")
        if self.n_classes_ == 2:
            inner = self.model_._structure_content()
            inner["link"] = "logit"
            return inner
        per_class = []
        for cls, model in zip(self.classes_, self.model_):
            payload = model._structure_content()
            payload["class"] = cls.item() if hasattr(cls, "item") else cls
            per_class.append(payload)
        return {
            "link": "softmax_ovr",
            "per_class": per_class,
        }
