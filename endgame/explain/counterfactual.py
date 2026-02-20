"""Counterfactual explanations via DiCE (Diverse Counterfactual Explanations).

Generates "what-if" explanations showing the minimal feature changes needed
to alter the model's prediction.

Reference
---------
Mothilal, R.K., Sharma, A. & Tan, C. (2020). "Explaining Machine
Learning Classifiers through Diverse Counterfactual Explanations."
*FAT* '20.

Example
-------
>>> from endgame.explain import CounterfactualExplainer
>>> cf = CounterfactualExplainer(model, X_train, feature_names=feature_names)
>>> explanation = cf.explain(X_test[:1], desired_class=1)
>>> explanation.to_dataframe()
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

from endgame.explain._base import BaseExplainer, Explanation


def _check_dice_installed() -> None:
    """Raise ImportError if the dice-ml package is unavailable."""
    try:
        import dice_ml  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'dice-ml' package is required for CounterfactualExplainer. "
            "Install it with: pip install dice-ml"
        )


class CounterfactualExplainer(BaseExplainer):
    """Counterfactual explanation generator using DiCE.

    Finds diverse sets of minimal feature perturbations that change
    the model's prediction to a desired outcome.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted classifier or regressor.
    training_data : array-like or pd.DataFrame
        Training data used to define feature ranges and constraints.
    continuous_features : list of str, optional
        Names of continuous features.  If ``None``, all features are
        assumed continuous.
    outcome_name : str, default='outcome'
        Name of the target column (used internally by DiCE).
    feature_names : list of str, optional
        Feature names.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Examples
    --------
    >>> cf = CounterfactualExplainer(model, X_train)
    >>> explanation = cf.explain(X_test[:1], desired_class=1, n_counterfactuals=3)
    >>> print(explanation.metadata['counterfactuals'])
    """

    def __init__(
        self,
        model: BaseEstimator,
        training_data: Any,
        continuous_features: list[str] | None = None,
        outcome_name: str = "outcome",
        feature_names: list[str] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            model=model,
            feature_names=feature_names,
            random_state=random_state,
            verbose=verbose,
        )
        self.training_data = training_data
        self.continuous_features = continuous_features
        self.outcome_name = outcome_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_dice_data(
        self, X: np.ndarray, names: list[str]
    ) -> Any:
        """Build a ``dice_ml.Data`` object from the training set.

        Parameters
        ----------
        X : np.ndarray
            Training data (features only).
        names : list of str
            Feature names.

        Returns
        -------
        dice_ml.Data
        """
        import dice_ml
        import pandas as pd

        df = pd.DataFrame(X, columns=names)

        # Add a dummy outcome column (DiCE requires it for data schema).
        df[self.outcome_name] = 0

        continuous = self.continuous_features
        if continuous is None:
            continuous = list(names)

        return dice_ml.Data(
            dataframe=df,
            continuous_features=continuous,
            outcome_name=self.outcome_name,
        )

    def _prepare_dice_model(self) -> Any:
        """Wrap the sklearn model for DiCE.

        Returns
        -------
        dice_ml.Model
        """
        import dice_ml

        backend = "sklearn"
        model_type = (
            "classifier"
            if hasattr(self.model, "predict_proba")
            else "regressor"
        )

        return dice_ml.Model(
            model=self.model,
            backend=backend,
            model_type=model_type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        *,
        desired_class: int | str = "opposite",
        n_counterfactuals: int = 3,
        method: Literal["random", "genetic", "kdtree"] = "random",
        features_to_vary: list[str] | None = None,
        permitted_range: dict[str, list[float]] | None = None,
    ) -> Explanation:
        """Generate counterfactual explanations.

        Parameters
        ----------
        X : array-like of shape (n_instances, n_features)
            Instance(s) to explain.  Typically a single row.
        desired_class : int or str, default='opposite'
            Target class for the counterfactual.  ``'opposite'`` flips a
            binary prediction.
        n_counterfactuals : int, default=3
            Number of diverse counterfactuals to generate per instance.
        method : str, default='random'
            DiCE generation method:
            - ``'random'``: Random perturbation search.
            - ``'genetic'``: Genetic algorithm search.
            - ``'kdtree'``: KD-tree nearest-neighbour search.
        features_to_vary : list of str, optional
            Restrict changes to these features only.  If ``None``, all
            features may be varied.
        permitted_range : dict, optional
            Per-feature permitted ranges, e.g.
            ``{'age': [18, 65], 'income': [0, 200000]}``.

        Returns
        -------
        Explanation
            An :class:`Explanation` with:
            - ``values``: Mean absolute change across counterfactuals
              (feature-level importance of change).
            - ``metadata['counterfactuals']``: DataFrame of generated
              counterfactuals.
            - ``metadata['original']``: The original instance(s).
        """
        _check_dice_installed()
        import pandas as pd

        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        train_data = self._to_numpy(self.training_data)
        names = self._resolve_feature_names(X)

        self._log(
            f"Generating {n_counterfactuals} counterfactual(s) for "
            f"{X.shape[0]} instance(s) using method='{method}' ..."
        )

        dice_data = self._prepare_dice_data(train_data, names)
        dice_model = self._prepare_dice_model()

        import dice_ml
        dice_exp = dice_ml.Dice(dice_data, dice_model, method=method)

        query_df = pd.DataFrame(X, columns=names)

        generate_kwargs: dict[str, Any] = {
            "query_instances": query_df,
            "total_CFs": n_counterfactuals,
            "desired_class": desired_class,
        }
        if features_to_vary is not None:
            generate_kwargs["features_to_vary"] = features_to_vary
        if permitted_range is not None:
            generate_kwargs["permitted_range"] = permitted_range

        result = dice_exp.generate_counterfactuals(**generate_kwargs)

        # Extract counterfactual DataFrames and compute feature changes.
        all_cfs: list[pd.DataFrame] = []
        total_changes = np.zeros(X.shape[1])

        for i, cf_example in enumerate(result.cf_examples_list):
            cf_df = cf_example.final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                # Drop the outcome column if present.
                cf_features = cf_df.drop(
                    columns=[self.outcome_name], errors="ignore"
                )
                all_cfs.append(cf_features)

                # Compute absolute changes from the original instance.
                original = X[i]
                for _, cf_row in cf_features.iterrows():
                    cf_vals = cf_row[names].values.astype(float)
                    total_changes += np.abs(cf_vals - original)

        n_total_cfs = sum(len(df) for df in all_cfs) if all_cfs else 1
        mean_changes = total_changes / max(n_total_cfs, 1)

        counterfactual_df = pd.concat(all_cfs, ignore_index=True) if all_cfs else pd.DataFrame(columns=names)

        return Explanation(
            values=mean_changes,
            base_value=None,
            feature_names=names,
            method="counterfactual",
            metadata={
                "counterfactuals": counterfactual_df,
                "original": pd.DataFrame(X, columns=names),
                "n_counterfactuals_requested": n_counterfactuals,
                "n_counterfactuals_generated": len(counterfactual_df),
                "desired_class": desired_class,
                "method": method,
            },
        )
