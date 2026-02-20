"""Heuristic-based search strategy.

This module provides a search strategy that uses data-driven heuristics
and meta-features to select models and configurations.
"""

import logging
from typing import Any

from endgame.automl.model_registry import MODEL_REGISTRY
from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)


# Heuristic rules based on data characteristics
HEURISTIC_RULES = {
    "small_data": {
        "condition": lambda mf: mf.get("nr_inst", 10000) < 1000,
        "recommended": ["tabpfn", "lgbm", "linear", "gp", "knn"],
        "avoid": ["ft_transformer", "saint", "node", "tabnet"],
        "reason": "Small data benefits from simpler models or pre-trained foundations",
    },
    "medium_data": {
        "condition": lambda mf: 1000 <= mf.get("nr_inst", 10000) < 10000,
        "recommended": ["lgbm", "catboost", "xgb", "mlp", "ebm"],
        "avoid": [],
        "reason": "Medium data works well with GBDTs and simple neural nets",
    },
    "large_data": {
        "condition": lambda mf: mf.get("nr_inst", 10000) >= 100000,
        "recommended": ["lgbm", "xgb", "catboost", "linear"],
        "avoid": ["tabpfn", "gp", "knn", "saint", "ft_transformer"],
        "reason": "Large data requires scalable models",
    },
    "high_dimensionality": {
        "condition": lambda mf: mf.get("dimensionality", 0) > 0.5,
        "recommended": ["lgbm", "xgb", "linear", "mlp"],
        "preprocessing": [("feature_selector", {"method": "adversarial"})],
        "reason": "High dimensionality benefits from feature selection",
    },
    "many_categoricals": {
        "condition": lambda mf: mf.get("cat_to_num", 0) > 0.5,
        "recommended": ["catboost", "lgbm", "ebm"],
        "preprocessing": [("encoder", {"method": "target"})],
        "reason": "Many categorical features work best with native handling",
    },
    "class_imbalance": {
        "condition": lambda mf: mf.get("class_imbalance", 1) > 10,
        "recommended": ["lgbm", "xgb", "catboost"],
        "preprocessing": [("balancer", {"method": "smote"})],
        "model_params": {"class_weight": "balanced"},
        "reason": "Severe class imbalance needs special handling",
    },
    "missing_values": {
        "condition": lambda mf: mf.get("pct_missing", 0) > 0.1,
        "recommended": ["lgbm", "catboost", "xgb"],
        "avoid": ["linear", "svm", "knn"],
        "reason": "High missing values need native handling or imputation",
    },
    "interpretability_needed": {
        "condition": lambda mf: mf.get("require_interpretable", False),
        "recommended": ["ebm", "linear", "c50", "rulefit", "nam"],
        "avoid": ["ft_transformer", "saint", "node", "tabnet"],
        "reason": "Interpretability requires glass-box models",
    },
}


class HeuristicSearch(BaseSearchStrategy):
    """Heuristic-based search strategy using data-driven rules.

    This strategy applies rules based on dataset characteristics to select
    appropriate models and preprocessing steps. It's faster than portfolio
    search as it makes informed decisions rather than trying everything.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    eval_metric : str
        Evaluation metric to optimize.
    model_pool : list of str, optional
        Explicit list of models to consider. If None, uses all registered.
    max_models : int, default=5
        Maximum number of models to suggest.
    apply_preprocessing_rules : bool, default=True
        Whether to include preprocessing recommendations.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Examples
    --------
    >>> strategy = HeuristicSearch(
    ...     task_type="classification",
    ...     max_models=3,
    ... )
    >>> meta_features = {"nr_inst": 500, "nr_attr": 20}
    >>> configs = strategy.suggest(meta_features=meta_features, n_suggestions=3)
    """

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        model_pool: list[str] | None = None,
        max_models: int = 5,
        apply_preprocessing_rules: bool = True,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
        )

        self.max_models = max_models
        self.apply_preprocessing_rules = apply_preprocessing_rules

        # Set model pool
        if model_pool is not None:
            self.model_pool = model_pool
        else:
            self.model_pool = list(MODEL_REGISTRY.keys())

        # Track suggested models
        self._suggested: set[str] = set()

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Suggest pipeline configurations based on heuristics.

        Parameters
        ----------
        meta_features : dict, optional
            Dataset meta-features for informed suggestions.
        n_suggestions : int, default=1
            Number of configurations to suggest.

        Returns
        -------
        list of PipelineConfig
            Suggested configurations.
        """
        if meta_features is None:
            meta_features = {}

        # Apply heuristic rules
        recommendations = self._apply_rules(meta_features)

        # Filter and rank models
        available = self._filter_available_models(meta_features)
        ranked = self._rank_by_recommendations(available, recommendations)

        # Select top models
        n_select = min(n_suggestions, self.max_models, len(ranked))
        selected = ranked[:n_select]

        # Create pipeline configs
        configs = []
        for model_name in selected:
            config = self._create_config(model_name, meta_features, recommendations)
            configs.append(config)
            self._suggested.add(model_name)

        if self.verbose > 0:
            print(f"Heuristic selection: {[c.model_name for c in configs]}")
            for rule_name, rule_data in recommendations.get("applied_rules", {}).items():
                print(f"  Rule '{rule_name}': {rule_data.get('reason', '')}")

        return configs

    def _apply_rules(
        self,
        meta_features: dict[str, float],
    ) -> dict[str, Any]:
        """Apply heuristic rules to determine recommendations.

        Parameters
        ----------
        meta_features : dict
            Dataset meta-features.

        Returns
        -------
        dict
            Recommendations including models, preprocessing, and params.
        """
        recommendations = {
            "recommended_models": set(),
            "avoided_models": set(),
            "preprocessing": [],
            "model_params": {},
            "applied_rules": {},
        }

        for rule_name, rule in HEURISTIC_RULES.items():
            try:
                if rule["condition"](meta_features):
                    # Add recommended models
                    for model in rule.get("recommended", []):
                        recommendations["recommended_models"].add(model)

                    # Add avoided models
                    for model in rule.get("avoid", []):
                        recommendations["avoided_models"].add(model)

                    # Add preprocessing steps
                    if "preprocessing" in rule:
                        recommendations["preprocessing"].extend(rule["preprocessing"])

                    # Add model params
                    if "model_params" in rule:
                        recommendations["model_params"].update(rule["model_params"])

                    # Track applied rule
                    recommendations["applied_rules"][rule_name] = {
                        "reason": rule.get("reason", ""),
                    }

                    if self.verbose > 1:
                        logger.debug(f"Applied rule: {rule_name}")

            except Exception as e:
                logger.warning(f"Error applying rule '{rule_name}': {e}")

        return recommendations

    def _filter_available_models(
        self,
        meta_features: dict[str, float],
    ) -> list[str]:
        """Filter models based on constraints.

        Parameters
        ----------
        meta_features : dict
            Dataset meta-features.

        Returns
        -------
        list of str
            Available model names.
        """
        available = []
        n_samples = meta_features.get("nr_inst", 10000)

        for model_name in self.model_pool:
            if model_name not in MODEL_REGISTRY:
                continue

            info = MODEL_REGISTRY[model_name]

            # Check task type compatibility
            if self.task_type == "classification":
                if "classification" not in info.task_types and "both" not in info.task_types:
                    continue
            elif self.task_type == "regression":
                if "regression" not in info.task_types and "both" not in info.task_types:
                    continue

            # Check sample size limits
            if info.max_samples and n_samples > info.max_samples:
                continue
            if info.min_samples and n_samples < info.min_samples:
                continue

            # Skip already suggested
            if model_name in self._suggested:
                continue

            available.append(model_name)

        return available

    def _rank_by_recommendations(
        self,
        models: list[str],
        recommendations: dict[str, Any],
    ) -> list[str]:
        """Rank models based on heuristic recommendations.

        Parameters
        ----------
        models : list of str
            Available models.
        recommendations : dict
            Heuristic recommendations.

        Returns
        -------
        list of str
            Ranked models (best first).
        """
        recommended = recommendations.get("recommended_models", set())
        avoided = recommendations.get("avoided_models", set())

        scores = {}
        for model_name in models:
            score = 50  # Base score

            # Boost recommended models
            if model_name in recommended:
                score += 30

            # Penalize avoided models
            if model_name in avoided:
                score -= 40

            # Add family-based heuristics
            info = MODEL_REGISTRY.get(model_name)
            if info:
                # GBDTs are generally strong
                if info.family == "gbdt":
                    score += 15

                # Fast models get a small boost
                if info.typical_fit_time == "fast":
                    score += 5

            scores[model_name] = score

        return sorted(models, key=lambda m: scores.get(m, 0), reverse=True)

    def _create_config(
        self,
        model_name: str,
        meta_features: dict[str, float],
        recommendations: dict[str, Any],
    ) -> PipelineConfig:
        """Create a pipeline configuration.

        Parameters
        ----------
        model_name : str
            Model name.
        meta_features : dict
            Dataset meta-features.
        recommendations : dict
            Heuristic recommendations.

        Returns
        -------
        PipelineConfig
            Pipeline configuration.
        """
        info = MODEL_REGISTRY.get(model_name)

        # Get default params
        params = info.default_params.copy() if info else {}

        # Apply recommended model params
        params.update(recommendations.get("model_params", {}))

        # Adjust params based on data size
        params = self._adjust_params_for_data(model_name, params, meta_features)

        # Determine preprocessing
        preprocessing = []
        if self.apply_preprocessing_rules:
            preprocessing = self._get_preprocessing(
                model_name, meta_features, recommendations
            )

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={
                "source": "heuristic_search",
                "applied_rules": list(recommendations.get("applied_rules", {}).keys()),
            },
        )

    def _adjust_params_for_data(
        self,
        model_name: str,
        params: dict[str, Any],
        meta_features: dict[str, float],
    ) -> dict[str, Any]:
        """Adjust model parameters based on data characteristics.

        Parameters
        ----------
        model_name : str
            Model name.
        params : dict
            Base parameters.
        meta_features : dict
            Dataset meta-features.

        Returns
        -------
        dict
            Adjusted parameters.
        """
        params = params.copy()
        n_samples = meta_features.get("nr_inst", 10000)
        n_features = meta_features.get("nr_attr", 10)

        # GBDT adjustments
        if model_name in ("lgbm", "xgb", "catboost"):
            if n_samples < 1000:
                params["n_estimators"] = min(params.get("n_estimators", 1000), 500)
                params["learning_rate"] = max(params.get("learning_rate", 0.05), 0.1)
            elif n_samples > 50000:
                params["n_estimators"] = max(params.get("n_estimators", 1000), 2000)

            if n_features > 100:
                params["max_depth"] = min(params.get("max_depth", 6), 8)
                params["num_leaves"] = min(params.get("num_leaves", 31), 63)

        # Neural network adjustments
        elif model_name in ("ft_transformer", "saint", "tabnet", "mlp"):
            if n_samples < 2000:
                params["n_epochs"] = min(params.get("n_epochs", 100), 50)
            if n_samples < 500:
                params["batch_size"] = min(params.get("batch_size", 64), 32)

        # Linear model adjustments
        elif model_name == "linear":
            if n_features > n_samples:
                params["penalty"] = "l1"
                params["C"] = 0.1

        return params

    def _get_preprocessing(
        self,
        model_name: str,
        meta_features: dict[str, float],
        recommendations: dict[str, Any],
    ) -> list[tuple]:
        """Get preprocessing steps for a model.

        Parameters
        ----------
        model_name : str
            Model name.
        meta_features : dict
            Dataset meta-features.
        recommendations : dict
            Heuristic recommendations.

        Returns
        -------
        list of tuple
            Preprocessing steps.
        """
        steps = []
        info = MODEL_REGISTRY.get(model_name)

        # Add recommended preprocessing
        for step in recommendations.get("preprocessing", []):
            steps.append(step)

        # Handle missing values if model doesn't support them
        pct_missing = meta_features.get("pct_missing", 0)
        if pct_missing > 0 and info and not info.handles_missing:
            if ("imputer", {}) not in steps:
                steps.append(("imputer", {"strategy": "median"}))

        # Handle categorical encoding if model doesn't support it
        n_cat = meta_features.get("nr_cat", 0)
        if n_cat > 0 and info and not info.handles_categorical:
            if not any(s[0] == "encoder" for s in steps):
                steps.append(("encoder", {"method": "target"}))

        return steps

    def update(self, result: SearchResult) -> None:
        """Update strategy with a new result.

        Parameters
        ----------
        result : SearchResult
            Result from evaluating a configuration.
        """
        super().update(result)

        # Track performance by rule for future improvements
        if result.success:
            applied_rules = result.config.metadata.get("applied_rules", [])
            if self.verbose > 1:
                logger.debug(
                    f"Model {result.config.model_name} with rules {applied_rules}: "
                    f"score={result.score:.4f}"
                )
