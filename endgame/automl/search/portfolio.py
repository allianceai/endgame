from __future__ import annotations

"""Portfolio-based search strategy.

The default search strategy that trains a diverse portfolio of models
in parallel, ensuring coverage across different model families.

After the initial model sweep exhausts all model types, the strategy
switches to **hyperparameter variant generation**: it perturbs the
hyperparameters of the top-performing models so that the continuous
optimization loop always has new configurations to try.
"""

import importlib.util
import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=64)
def _is_package_available(name: str) -> bool:
    """Check if a Python package is installed (lightweight, no import)."""
    return importlib.util.find_spec(name) is not None

from endgame.automl.model_registry import (
    MODEL_REGISTRY,
    get_interpretable_portfolio,
)
from endgame.automl.presets import MODEL_POOLS
from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)

# ── Hyperparameter perturbation tables ──────────────────────────────────────
# For each model family, define (param_name, type, range) triples.
# "log" means sample log-uniformly; "int" means integer uniform.
_GBDT_SPACE: list[tuple[str, str, float, float]] = [
    ("learning_rate", "log", 0.005, 0.3),
    ("n_estimators", "int", 200, 4000),
    ("max_depth", "int", 3, 12),
    ("subsample", "uniform", 0.5, 1.0),
    ("colsample_bytree", "uniform", 0.3, 1.0),
    ("reg_alpha", "log", 1e-8, 10.0),
    ("reg_lambda", "log", 1e-8, 10.0),
    ("min_child_weight", "log", 1e-3, 100.0),
]

_NEURAL_SPACE: list[tuple[str, str, float, float]] = [
    ("learning_rate", "log", 1e-5, 1e-2),
    ("n_epochs", "int", 20, 300),
    ("weight_decay", "log", 1e-7, 1e-2),
    ("dropout", "uniform", 0.0, 0.5),
    ("batch_size", "choice", 64, 512),
]

_TREE_SPACE: list[tuple[str, str, float, float]] = [
    ("n_estimators", "int", 50, 2000),
    ("max_depth", "int", 3, 30),
    ("min_samples_leaf", "int", 1, 50),
    ("max_features", "uniform", 0.3, 1.0),
]

_LINEAR_SPACE: list[tuple[str, str, float, float]] = [
    ("C", "log", 1e-4, 100.0),
]

_FAMILY_SPACES: dict[str, list] = {
    "gbdt": _GBDT_SPACE,
    "neural": _NEURAL_SPACE,
    "tree": _TREE_SPACE,
    "linear": _LINEAR_SPACE,
    "foundation": _NEURAL_SPACE,
}


def _perturb_params(
    base_params: dict[str, Any],
    family: str,
    rng: np.random.RandomState,
    magnitude: float = 1.0,
) -> dict[str, Any]:
    """Create a random perturbation of *base_params* using family-aware ranges.

    Only modifies parameters that already exist in *base_params* or that are
    in the family space.  Parameters not in the space are kept as-is.
    """
    space = _FAMILY_SPACES.get(family, [])
    if not space:
        return base_params

    params = base_params.copy()
    n_to_change = max(1, int(len(space) * 0.4 * magnitude))
    indices = rng.choice(len(space), size=min(n_to_change, len(space)), replace=False)

    for idx in indices:
        name, kind, lo, hi = space[idx]
        try:
            if kind == "log":
                params[name] = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
            elif kind == "int":
                params[name] = int(rng.randint(int(lo), int(hi) + 1))
            elif kind == "uniform":
                params[name] = float(rng.uniform(lo, hi))
            elif kind == "choice":
                params[name] = int(2 ** rng.randint(int(math.log2(lo)), int(math.log2(hi)) + 1))
        except Exception:
            pass

    return params


# ── Preprocessing option sampling ───────────────────────────────────────────
_IMPUTER_OPTIONS = [
    {"strategy": "median"},
    {"strategy": "mean"},
    {"strategy": "most_frequent"},
    {"strategy": "knn", "n_neighbors": 5},
]

_ENCODER_OPTIONS = [
    {"method": "target"},
    {"method": "onehot"},
    {"method": "ordinal"},
]

_SCALER_OPTIONS = [
    {"method": "standard"},
    {"method": "robust"},
    {"method": "quantile"},
    None,  # no scaling
]

_FEATURE_SELECTION_OPTIONS = [
    None,  # no selection
    {"method": "variance_threshold", "threshold": 0.01},
    {"method": "mutual_info", "k": 20},
    {"method": "boruta"},
]


def _sample_preprocessing(
    model_name: str,
    meta_features: dict[str, float] | None,
    rng: np.random.RandomState,
) -> list[tuple[str, dict]]:
    """Sample a random preprocessing pipeline for a model variant."""
    info = MODEL_REGISTRY.get(model_name)
    steps: list[tuple[str, dict]] = []

    mf = meta_features or {}

    # Imputer — only if dataset has missing values
    if mf.get("pct_missing", 0) > 0:
        if not (info and info.handles_missing):
            imp = _IMPUTER_OPTIONS[rng.randint(len(_IMPUTER_OPTIONS))]
            steps.append(("imputer", imp))

    # Encoder — only if dataset has categorical features
    if mf.get("nr_cat", 0) > 0:
        if not (info and info.handles_categorical):
            enc = _ENCODER_OPTIONS[rng.randint(len(_ENCODER_OPTIONS))]
            steps.append(("encoder", enc))

    # Scaler — randomly include or skip
    scaler = _SCALER_OPTIONS[rng.randint(len(_SCALER_OPTIONS))]
    if scaler is not None:
        steps.append(("scaler", scaler))

    # Feature selection — randomly include or skip
    fs = _FEATURE_SELECTION_OPTIONS[rng.randint(len(_FEATURE_SELECTION_OPTIONS))]
    if fs is not None:
        steps.append(("feature_selection", fs))

    return steps


class PortfolioSearch(BaseSearchStrategy):
    """Portfolio-based search strategy with iterative HPO.

    Phase 1 (initial sweep): suggests a diverse set of model types.
    Phase 2 (HPO variants):  once all model types are trained, generates
    hyperparameter variants of the top performers so the continuous
    optimization loop never runs out of candidates.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    eval_metric : str
        Evaluation metric to optimize.
    model_pool : list of str, optional
        Explicit list of models to consider. If None, uses preset.
    preset : str, default="medium_quality"
        Preset to use for model pool if model_pool not specified.
    ensure_diversity : bool, default=True
        Whether to ensure at least one model from each family.
    max_models : int, optional
        Maximum number of models to suggest.
    min_models : int, default=1
        Minimum number of models to suggest.
    meta_learner : MetaLearner, optional
        Pre-trained meta-learner for model ranking.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        model_pool: list[str] | None = None,
        preset: str = "medium_quality",
        ensure_diversity: bool = True,
        max_models: int | None = None,
        min_models: int = 1,
        meta_learner: Any | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        interpretable_only: bool = False,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
        )

        self.preset = preset
        self.ensure_diversity = ensure_diversity
        self.max_models = max_models
        self.min_models = min_models
        self.meta_learner = meta_learner
        self.interpretable_only = interpretable_only
        self._rng = np.random.RandomState(random_state or 42)

        if interpretable_only:
            self.model_pool = get_interpretable_portfolio(
                task_type=task_type,
                time_budget="high",
            )
            if verbose > 0:
                logger.info(f"Interpretable-only mode: {len(self.model_pool)} models available")
        elif model_pool is not None:
            self.model_pool = model_pool
        elif preset in MODEL_POOLS:
            self.model_pool = MODEL_POOLS[preset].copy()
        else:
            self.model_pool = MODEL_POOLS["medium_quality"].copy()

        self._trained: set[str] = set()
        self._family_scores: dict[str, list[float]] = {}
        self._variant_counter: int = 0
        self._config_hashes: set[str] = set()

    # ── Phase detection ──────────────────────────────────────────────────

    @property
    def initial_sweep_done(self) -> bool:
        """True when all candidate model types have been trained once."""
        return len(self._filter_available_models(None)) == 0

    # ── Main suggest() ───────────────────────────────────────────────────

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Suggest pipeline configurations to try.

        During the initial sweep, suggests new model types.  After all
        model types have been tried, switches to HPO variant generation
        (perturbing hyperparameters of the best-performing models).
        """
        # Phase 1: new model types
        available = self._filter_available_models(meta_features)

        if available:
            ranked = self._rank_models(available, meta_features)
            if self.ensure_diversity:
                selected = self._select_diverse(ranked, n_suggestions)
            else:
                selected = ranked[:n_suggestions]

            if self.max_models:
                selected = selected[: self.max_models]
            if len(selected) < self.min_models:
                remaining = [m for m in ranked if m not in selected]
                selected.extend(remaining[: self.min_models - len(selected)])

            configs = [self._create_config(m, meta_features) for m in selected]
            if self.verbose > 0:
                print(f"Portfolio: {[c.model_name for c in configs]}")
            return configs

        # Phase 2: HPO variants of top models
        return self._suggest_variants(meta_features, n_suggestions)

    def _filter_available_models(
        self,
        meta_features: dict[str, float] | None,
    ) -> list[str]:
        """Filter models based on constraints.

        Parameters
        ----------
        meta_features : dict, optional
            Dataset meta-features.

        Returns
        -------
        list of str
            Available model names.
        """
        available = []

        n_samples = meta_features.get("nr_inst", 10000) if meta_features else 10000

        for model_name in self.model_pool:
            if model_name not in MODEL_REGISTRY:
                logger.warning(f"Model '{model_name}' not in registry, skipping")
                continue

            info = MODEL_REGISTRY[model_name]

            # Enforce interpretable_only constraint
            if self.interpretable_only and not info.interpretable:
                if self.verbose > 1:
                    logger.debug(f"Skipping {model_name}: not interpretable")
                continue

            # Check task type compatibility
            if self.task_type == "classification":
                if "classification" not in info.task_types and "both" not in info.task_types:
                    continue
            elif self.task_type == "regression":
                if "regression" not in info.task_types and "both" not in info.task_types:
                    continue

            # Check sample size limits
            if info.max_samples and n_samples > info.max_samples:
                if self.verbose > 1:
                    logger.debug(
                        f"Skipping {model_name}: n_samples={n_samples} > max={info.max_samples}"
                    )
                continue

            if info.min_samples and n_samples < info.min_samples:
                if self.verbose > 1:
                    logger.debug(
                        f"Skipping {model_name}: n_samples={n_samples} < min={info.min_samples}"
                    )
                continue

            # Skip already trained models
            if model_name in self._trained:
                continue

            # Check that required external packages are installed
            if info.required_packages:
                missing = [p for p in info.required_packages if not _is_package_available(p)]
                if missing:
                    if self.verbose > 1:
                        logger.debug(
                            f"Skipping {model_name}: missing packages {missing}"
                        )
                    continue

            available.append(model_name)

        return available

    def _rank_models(
        self,
        models: list[str],
        meta_features: dict[str, float] | None,
    ) -> list[str]:
        """Rank models by expected performance.

        Parameters
        ----------
        models : list of str
            Models to rank.
        meta_features : dict, optional
            Dataset meta-features.

        Returns
        -------
        list of str
            Models ranked by expected performance (best first).
        """
        # Use meta-learner if available
        if self.meta_learner is not None:
            try:
                from endgame.benchmark.profiler import MetaFeatureSet

                if isinstance(meta_features, dict):
                    mf = MetaFeatureSet(features=meta_features)
                else:
                    mf = meta_features

                recommendation = self.meta_learner.recommend_from_features(mf)

                # Build ranking from recommendation
                ranked = []
                if recommendation.model_name in models:
                    ranked.append(recommendation.model_name)

                for alt_model, _ in recommendation.alternatives:
                    if alt_model in models and alt_model not in ranked:
                        ranked.append(alt_model)

                # Add remaining models
                for m in models:
                    if m not in ranked:
                        ranked.append(m)

                return ranked

            except Exception as e:
                logger.warning(f"Meta-learner ranking failed: {e}")

        # Fallback: heuristic ranking
        return self._heuristic_rank(models, meta_features)

    def _heuristic_rank(
        self,
        models: list[str],
        meta_features: dict[str, float] | None,
    ) -> list[str]:
        """Heuristic model ranking based on meta-features.

        Parameters
        ----------
        models : list of str
            Models to rank.
        meta_features : dict, optional
            Dataset meta-features.

        Returns
        -------
        list of str
            Models ranked by heuristic score.
        """
        if meta_features is None:
            meta_features = {}

        n_samples = meta_features.get("nr_inst", 10000)
        n_features = meta_features.get("nr_attr", 10)
        pct_cat = meta_features.get("cat_to_num", 0)
        imbalance = meta_features.get("class_imbalance", 1)

        scores = {}
        for model_name in models:
            info = MODEL_REGISTRY.get(model_name)
            if info is None:
                scores[model_name] = 0
                continue

            score = 50  # Base score

            # GBDTs are generally strong
            if info.family == "gbdt":
                score += 30

            # Neural networks need enough data
            if info.family == "neural":
                if n_samples < 1000:
                    score -= 20
                elif n_samples > 10000:
                    score += 10

            # Foundation models for small data
            if info.family == "foundation":
                if n_samples < 5000:
                    score += 25
                else:
                    score -= 10

            # Handle categorical features
            if pct_cat > 0.5 and info.handles_categorical:
                score += 10

            # Prefer models that handle imbalance well
            if imbalance > 5:
                if model_name in ("catboost", "lgbm", "xgb"):
                    score += 10

            # Penalize slow models for large data
            if info.typical_fit_time == "very_slow":
                if n_samples > 10000:
                    score -= 40
                elif n_samples > 5000:
                    score -= 20
            elif info.typical_fit_time == "slow":
                if n_samples > 50000:
                    score -= 15
                elif n_samples > 20000:
                    score -= 10

            scores[model_name] = score

        # Sort by score descending
        return sorted(models, key=lambda m: scores.get(m, 0), reverse=True)

    def _select_diverse(
        self,
        ranked: list[str],
        n_select: int,
    ) -> list[str]:
        """Select diverse models ensuring family coverage.

        Parameters
        ----------
        ranked : list of str
            Models ranked by expected performance.
        n_select : int
            Number of models to select.

        Returns
        -------
        list of str
            Selected models.
        """
        selected = []
        families_included: set[str] = set()

        # First pass: include top model from each family
        for model_name in ranked:
            if len(selected) >= n_select:
                break

            info = MODEL_REGISTRY.get(model_name)
            if info is None:
                continue

            family = info.family

            # Always include if family not yet represented
            if family not in families_included:
                selected.append(model_name)
                families_included.add(family)

        # Second pass: fill remaining slots with best models
        for model_name in ranked:
            if len(selected) >= n_select:
                break

            if model_name not in selected:
                selected.append(model_name)

        return selected

    def _create_config(
        self,
        model_name: str,
        meta_features: dict[str, float] | None,
    ) -> PipelineConfig:
        """Create a pipeline configuration for a model.

        Parameters
        ----------
        model_name : str
            Model name.
        meta_features : dict, optional
            Dataset meta-features.

        Returns
        -------
        PipelineConfig
            Pipeline configuration.
        """
        info = MODEL_REGISTRY.get(model_name)

        # Get default params
        params = info.default_params.copy() if info else {}

        # Adjust params based on meta-features
        if meta_features:
            params = self._adjust_params(model_name, params, meta_features)

        # Determine preprocessing
        preprocessing = self._suggest_preprocessing(model_name, meta_features)

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={"source": "portfolio_search"},
        )

    def _adjust_params(
        self,
        model_name: str,
        params: dict[str, Any],
        meta_features: dict[str, float],
    ) -> dict[str, Any]:
        """Adjust model parameters based on meta-features.

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
            # Reduce estimators for small data
            if n_samples < 1000:
                params["n_estimators"] = min(params.get("n_estimators", 1000), 500)

            # Increase for large data
            elif n_samples > 50000:
                params["n_estimators"] = max(params.get("n_estimators", 1000), 2000)

            # Adjust depth for dimensionality
            if n_features > 100:
                params["max_depth"] = params.get("max_depth", 6)

        # Neural network adjustments
        elif model_name in ("ft_transformer", "saint", "tabnet", "mlp"):
            # Reduce epochs for small data to prevent overfitting
            if n_samples < 2000:
                params["n_epochs"] = min(params.get("n_epochs", 100), 50)

        return params

    def _suggest_preprocessing(
        self,
        model_name: str,
        meta_features: dict[str, float] | None,
    ) -> list[tuple]:
        """Suggest preprocessing steps for a model.

        Parameters
        ----------
        model_name : str
            Model name.
        meta_features : dict, optional
            Dataset meta-features.

        Returns
        -------
        list of tuple
            Preprocessing steps as (name, params) tuples.
        """
        steps = []

        if meta_features is None:
            return steps

        info = MODEL_REGISTRY.get(model_name)

        # Handle missing values
        pct_missing = meta_features.get("pct_missing", 0)
        if pct_missing > 0:
            if info and info.handles_missing:
                pass  # Model handles it
            else:
                steps.append(("imputer", {"strategy": "median"}))

        # Handle categorical encoding
        n_cat = meta_features.get("nr_cat", 0)
        if n_cat > 0:
            if info and info.handles_categorical:
                pass  # Model handles it
            else:
                # Target encoding for high cardinality
                steps.append(("encoder", {"method": "target"}))

        return steps

    def _suggest_variants(
        self,
        meta_features: dict[str, float] | None,
        n_suggestions: int,
    ) -> list[PipelineConfig]:
        """Generate HPO variants of the top-performing trained models.

        Picks the best models, perturbs their hyperparameters **and
        preprocessing options**, and returns configs with unique
        ``config_id``s so the training loop treats them as new.
        """
        successful = [r for r in self.results_ if r.success]
        if not successful:
            return []

        successful.sort(key=lambda r: r.score, reverse=True)
        top_n = max(3, n_suggestions)
        top_results = successful[:top_n]

        configs: list[PipelineConfig] = []
        for result in top_results:
            if len(configs) >= n_suggestions:
                break

            model_name = result.config.model_name
            info = MODEL_REGISTRY.get(model_name)
            family = info.family if info else "unknown"
            base_params = result.config.model_params.copy()

            self._variant_counter += 1
            magnitude = min(1.0 + self._variant_counter * 0.1, 3.0)
            new_params = _perturb_params(base_params, family, self._rng, magnitude)

            # Vary preprocessing every other variant
            if self._variant_counter % 2 == 0:
                preprocessing = _sample_preprocessing(
                    model_name, meta_features, self._rng,
                )
            else:
                preprocessing = result.config.preprocessing

            config = PipelineConfig(
                model_name=model_name,
                model_params=new_params,
                preprocessing=preprocessing,
                metadata={
                    "source": "portfolio_hpo_variant",
                    "variant_of": model_name,
                    "variant_num": self._variant_counter,
                    "base_score": result.score,
                },
            )

            if config.config_id in self._config_hashes:
                continue
            self._config_hashes.add(config.config_id)
            configs.append(config)

        if self.verbose > 0 and configs:
            names = [f"{c.model_name}(v{c.metadata.get('variant_num', '?')})" for c in configs]
            print(f"HPO variants: {names}")

        return configs

    def update(self, result: SearchResult) -> None:
        """Update strategy with a new result."""
        super().update(result)

        # Mark model type as trained (for Phase 1 sweep)
        self._trained.add(result.config.model_name)

        # Track the config hash to avoid duplicates in variant generation
        if result.config.config_id:
            self._config_hashes.add(result.config.config_id)

        # Track model performance by family
        if result.success:
            info = MODEL_REGISTRY.get(result.config.model_name)
            if info:
                family = info.family
                if family not in self._family_scores:
                    self._family_scores[family] = []
                self._family_scores[family].append(result.score)
