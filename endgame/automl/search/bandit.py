"""Bandit-based search strategy (Successive Halving / ASHA).

Implements multi-fidelity optimisation: many configurations are trained
cheaply on small data fractions, and only the top performers are promoted
to progressively larger fractions.  This is far more time-efficient than
training every configuration on the full dataset.

The "budget" for each rung is controlled by a *data fraction* — the
orchestrator samples that fraction of the training data for CV.  The
reduction factor ``eta`` determines how aggressively we prune at each
rung (keep top 1/eta).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from typing import Any

import numpy as np

from endgame.automl.model_registry import MODEL_REGISTRY
from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)

# ── Preprocessing option pools (shared with genetic search) ─────────

_IMPUTER_OPTIONS: list[dict[str, Any]] = [
    {"strategy": "median"},
    {"strategy": "mean"},
    {"strategy": "most_frequent"},
    {"strategy": "knn", "n_neighbors": 5},
]

_ENCODER_OPTIONS: list[dict[str, Any]] = [
    {"method": "ordinal"},
    {"method": "onehot"},
    {"method": "target"},
]

_SCALER_OPTIONS: list[dict[str, Any] | None] = [
    None,
    {"method": "standard"},
    {"method": "robust"},
    {"method": "quantile"},
]

_FEATURE_SELECTION_OPTIONS: list[dict[str, Any] | None] = [
    None,
    None,  # extra weight toward no selection
    {"method": "variance_threshold", "threshold": 0.0},
    {"method": "mutual_info", "k": 20},
    {"method": "mutual_info", "k": 50},
]


class BanditSearch(BaseSearchStrategy):
    """Successive Halving search strategy for AutoML.

    Trains many candidate configurations on a small fraction of data,
    keeps the top performers, and promotes them to progressively larger
    fractions until the final rung uses the full dataset.

    Example rung schedule with ``max_configs=27, eta=3``::

        =====  ============  ==============
        Rung   # Configs     Data fraction
        =====  ============  ==============
        0       27            ~4%
        1       9             ~11%
        2       3             ~33%
        3       1             100%
        =====  ============  ==============

    Parameters
    ----------
    task_type : str
        ``"classification"`` or ``"regression"``.
    eval_metric : str
        Metric to optimise (``"auto"`` -> sensible default).
    model_pool : list of str, optional
        Explicit model names.  ``None`` -> all registered models
        compatible with the task.
    reduction_factor : int, default=3
        How aggressively to prune (eta).  Keep top ``1/eta`` each rung.
    max_configs : int, default=27
        Maximum configurations in the first rung.  Should ideally
        be a power of ``reduction_factor`` for even splits.
    min_fraction : float, optional
        Data fraction for the first rung.  Defaults to
        ``1 / max_configs`` (at least 0.04).
    random_state : int, optional
        Seed for reproducibility.
    verbose : int, default=0
        Verbosity level.
    excluded_models : set of str, optional
        Models to exclude from the search.

    Examples
    --------
    >>> strategy = BanditSearch(
    ...     task_type="classification",
    ...     max_configs=27,
    ...     reduction_factor=3,
    ... )
    >>> configs = strategy.suggest(meta_features=mf, n_suggestions=5)
    >>> # train configs, then:
    >>> for result in results:
    ...     strategy.update(result)
    """

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        model_pool: list[str] | None = None,
        reduction_factor: int = 3,
        max_configs: int = 27,
        min_fraction: float | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        excluded_models: set[str] | None = None,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
            excluded_models=excluded_models,
        )
        self.model_pool = model_pool
        self.reduction_factor = reduction_factor
        self.max_configs = max_configs

        # Compute rung schedule
        eta = reduction_factor
        self.n_rungs = max(
            1, int(math.floor(math.log(max_configs) / math.log(eta))) + 1
        )

        # Data fractions: geometrically increasing to 1.0
        if min_fraction is None:
            min_fraction = max(0.04, 1.0 / max_configs)
        self.fractions: list[float] = []
        for i in range(self.n_rungs):
            frac = min(1.0, min_fraction * (eta ** i))
            self.fractions.append(frac)
        # Ensure final rung is exactly 1.0
        self.fractions[-1] = 1.0

        # Configs per rung: max_configs / eta^rung
        self.configs_per_rung: list[int] = []
        for i in range(self.n_rungs):
            n = max(1, int(max_configs / (eta ** i)))
            self.configs_per_rung.append(n)

        self._rng = random.Random(random_state)
        self._np_rng = np.random.RandomState(random_state or 42)

        # State
        self._rung: int = 0
        self._rung_configs: list[PipelineConfig] = []
        self._rung_scores: dict[str, float] = {}  # config_id -> score
        self._initialized: bool = False
        self._completed: bool = False
        self._available_models: list[str] = []

    # ── Public API ──────────────────────────────────────────────────

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Return up to *n_suggestions* configs from the current rung.

        When all configs in a rung have been evaluated, automatically
        promotes the best and advances to the next rung.
        """
        if self._completed:
            return []

        if not self._initialized:
            if not self._available_models:
                self._available_models = self._get_available_models(
                    meta_features
                )
            self._init_rung_configs(meta_features)
            self._initialized = True

        # Promote if current rung is fully evaluated
        attempts = 0
        while self._rung_fully_evaluated() and attempts < self.n_rungs:
            attempts += 1
            self._promote_and_advance(meta_features)
            if self._completed:
                return []

        # Collect unevaluated configs in current rung
        unevaluated = [
            c for c in self._rung_configs
            if c.config_id not in self._rung_scores
        ]

        batch = unevaluated[:n_suggestions]

        if self.verbose > 0 and batch:
            frac = self.fractions[self._rung]
            n_total = len(self._rung_configs)
            n_done = len(self._rung_scores)
            models = [c.model_name for c in batch]
            print(
                f"  [Bandit] Rung {self._rung}/{self.n_rungs - 1} "
                f"({frac:.0%} data): suggesting {len(batch)} "
                f"({n_done}/{n_total} evaluated) "
                f"[{', '.join(models[:5])}{'...' if len(models) > 5 else ''}]"
            )

        return batch

    def update(self, result: SearchResult) -> None:
        """Record evaluation result for a configuration."""
        super().update(result)

        config_id = result.config.config_id
        if result.success:
            self._rung_scores[config_id] = result.score
        else:
            self._rung_scores[config_id] = float("-inf")

        if self.verbose > 1:
            status = f"{result.score:.4f}" if result.success else "FAIL"
            logger.debug(
                f"Bandit rung {self._rung}: "
                f"{result.config.model_name} -> {status}"
            )

    def should_stop(self, max_iterations: int | None = None) -> bool:
        """Check if search is complete."""
        if self._completed:
            return True
        if max_iterations and self.n_evaluated_ >= max_iterations:
            return True
        return False

    # ── Rung management ─────────────────────────────────────────────

    def _rung_fully_evaluated(self) -> bool:
        """True when all configs in the current rung have scores."""
        return len(self._rung_scores) >= len(self._rung_configs)

    def _init_rung_configs(
        self,
        meta_features: dict[str, float] | None,
    ) -> None:
        """Generate the initial config pool for rung 0."""
        n_configs = self.configs_per_rung[0]
        fraction = self.fractions[0]
        mf = meta_features or {}
        configs: list[PipelineConfig] = []

        # Phase 1: one config per model type (diverse coverage)
        for model_name in self._available_models:
            if len(configs) >= n_configs:
                break
            config = self._create_config(model_name, mf, fraction, rung=0)
            configs.append(config)

        # Phase 2: fill remaining with random configs
        seen_hashes: set[str] = {c.config_id for c in configs}
        attempts = 0
        while len(configs) < n_configs and attempts < n_configs * 5:
            attempts += 1
            model_name = self._rng.choice(self._available_models)
            config = self._create_config(
                model_name, mf, fraction, rung=0, randomise_params=True
            )
            if config.config_id not in seen_hashes:
                configs.append(config)
                seen_hashes.add(config.config_id)

        self._rung_configs = configs
        self._rung_scores = {}

        if self.verbose > 0:
            unique_models = len({c.model_name for c in configs})
            print(
                f"  [Bandit] Initialised: {len(configs)} configs, "
                f"{unique_models} unique model types, "
                f"{self.n_rungs} rungs "
                f"(eta={self.reduction_factor})"
            )
            for i in range(self.n_rungs):
                print(
                    f"    Rung {i}: {self.configs_per_rung[i]} configs "
                    f"x {self.fractions[i]:.0%} data"
                )

    def _promote_and_advance(
        self,
        meta_features: dict[str, float] | None = None,
    ) -> None:
        """Promote top performers and advance to the next rung."""
        if self._rung >= self.n_rungs - 1:
            self._completed = True
            if self.verbose > 0 and self._rung_scores:
                best_id = max(self._rung_scores, key=self._rung_scores.get)
                best_score = self._rung_scores[best_id]
                best_config = next(
                    c for c in self._rung_configs
                    if c.config_id == best_id
                )
                print(
                    f"  [Bandit] Complete! Best: {best_config.model_name} "
                    f"score={best_score:.4f}"
                )
            return

        # Rank configs by score
        scored = sorted(
            self._rung_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Promote top 1/eta
        n_promote = self.configs_per_rung[self._rung + 1]
        promoted_ids = {cid for cid, _ in scored[:n_promote]}

        # Get the actual config objects
        promoted_configs = [
            c for c in self._rung_configs if c.config_id in promoted_ids
        ]

        if self.verbose > 0:
            top_score = scored[0][1] if scored else 0.0
            bottom_promoted = (
                scored[n_promote - 1][1] if n_promote <= len(scored) else 0.0
            )
            print(
                f"  [Bandit] Rung {self._rung} -> {self._rung + 1}: "
                f"promoting {n_promote}/{len(self._rung_configs)} "
                f"(cutoff={bottom_promoted:.4f}, best={top_score:.4f})"
            )

        # Advance to next rung
        self._rung += 1
        new_fraction = self.fractions[self._rung]

        # Create new configs for promoted models with higher fraction
        new_configs: list[PipelineConfig] = []
        for old_config in promoted_configs:
            new_config = PipelineConfig(
                model_name=old_config.model_name,
                model_params=old_config.model_params.copy(),
                preprocessing=list(old_config.preprocessing),
                feature_engineering=list(old_config.feature_engineering),
                metadata={
                    "source": "bandit_search",
                    "data_fraction": new_fraction,
                    "rung": self._rung,
                    "promoted_from": old_config.config_id,
                    "previous_score": self._rung_scores.get(
                        old_config.config_id, 0.0
                    ),
                },
                config_id=self._make_config_id(
                    old_config.model_name,
                    old_config.model_params,
                    old_config.preprocessing,
                    self._rung,
                ),
            )
            new_configs.append(new_config)

        self._rung_configs = new_configs
        self._rung_scores = {}

    # ── Config generation helpers ───────────────────────────────────

    def _create_config(
        self,
        model_name: str,
        meta_features: dict[str, float],
        fraction: float,
        rung: int,
        randomise_params: bool = False,
    ) -> PipelineConfig:
        """Build a PipelineConfig tagged with bandit metadata."""
        info = MODEL_REGISTRY.get(model_name)
        params = info.default_params.copy() if info else {}

        if randomise_params:
            params = self._perturb_params(model_name, params)

        preprocessing = self._sample_preprocessing(model_name, meta_features)

        config_id = self._make_config_id(
            model_name, params, preprocessing, rung
        )

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            config_id=config_id,
            metadata={
                "source": "bandit_search",
                "data_fraction": fraction,
                "rung": rung,
            },
        )

    @staticmethod
    def _make_config_id(
        model_name: str,
        params: dict,
        preprocessing: list,
        rung: int,
    ) -> str:
        """Deterministic hash including rung to ensure uniqueness."""
        import json

        config_str = json.dumps(
            {
                "model": model_name,
                "params": params,
                "preproc": preprocessing,
                "rung": rung,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _perturb_params(
        self, model_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Randomly perturb hyperparameters from defaults."""
        params = params.copy()
        info = MODEL_REGISTRY.get(model_name)
        family = getattr(info, "family", "") if info else ""

        # Family-specific perturbation ranges
        if family in ("gbdt",) or model_name in ("lgbm", "xgb", "catboost"):
            perturbations = {
                "n_estimators": ("int", 100, 4000),
                "learning_rate": ("log", 0.005, 0.3),
                "max_depth": ("int", 3, 12),
                "subsample": ("float", 0.5, 1.0),
                "colsample_bytree": ("float", 0.3, 1.0),
                "reg_alpha": ("log", 1e-8, 10.0),
                "reg_lambda": ("log", 1e-8, 10.0),
            }
        elif family in ("neural", "deep_tabular"):
            perturbations = {
                "n_epochs": ("int", 20, 300),
                "learning_rate": ("log", 1e-5, 1e-2),
                "dropout": ("float", 0.0, 0.5),
                "batch_size": ("int", 16, 512),
            }
        elif family in ("tree", "forest"):
            perturbations = {
                "n_estimators": ("int", 50, 2000),
                "max_depth": ("int", 3, 30),
                "min_samples_leaf": ("int", 1, 50),
            }
        elif family in ("linear", "glm"):
            perturbations = {
                "C": ("log", 1e-4, 100.0),
                "alpha": ("log", 1e-6, 10.0),
            }
        else:
            perturbations = {
                "n_estimators": ("int", 50, 3000),
                "max_depth": ("int", 3, 15),
            }

        # Perturb each parameter that exists in params
        for hp_name, (kind, lo, hi) in perturbations.items():
            if hp_name not in params:
                continue
            if self._rng.random() < 0.5:
                continue  # only perturb ~half the params

            if kind == "int":
                params[hp_name] = self._rng.randint(int(lo), int(hi))
            elif kind == "log":
                params[hp_name] = float(
                    np.exp(
                        self._rng.uniform(np.log(float(lo)), np.log(float(hi)))
                    )
                )
            elif kind == "float":
                params[hp_name] = self._rng.uniform(float(lo), float(hi))

        return params

    def _sample_preprocessing(
        self,
        model_name: str,
        meta_features: dict[str, float],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Sample a preprocessing pipeline for a model."""
        info = MODEL_REGISTRY.get(model_name)
        steps: list[tuple[str, dict[str, Any]]] = []
        mf = meta_features or {}

        # Imputer -- only if data has missing values and model can't handle
        if mf.get("pct_missing", 0) > 0:
            if not (info and info.handles_missing):
                imp = self._rng.choice(_IMPUTER_OPTIONS)
                steps.append(("imputer", imp))

        # Encoder -- only if data has categoricals and model can't handle
        if mf.get("nr_cat", 0) > 0:
            if not (info and info.handles_categorical):
                enc = self._rng.choice(_ENCODER_OPTIONS)
                steps.append(("encoder", enc))

        # Scaler -- 50% chance
        scaler = self._rng.choice(_SCALER_OPTIONS)
        if scaler is not None:
            steps.append(("scaler", scaler))

        # Feature selection -- 25% chance
        fs = self._rng.choice(_FEATURE_SELECTION_OPTIONS)
        if fs is not None:
            steps.append(("feature_selection", fs))

        return steps

    def _get_available_models(
        self,
        meta_features: dict[str, float] | None = None,
    ) -> list[str]:
        """Return models compatible with the task."""
        mf = meta_features or {}
        n_samples = mf.get("nr_inst", 10_000)
        candidates: list[str] = []

        pool_set = set(self.model_pool) if self.model_pool else None

        for name, info in MODEL_REGISTRY.items():
            if self.excluded_models and name in self.excluded_models:
                continue
            if pool_set and name not in pool_set:
                continue

            if self.task_type == "classification":
                if (
                    "classification" not in info.task_types
                    and "both" not in info.task_types
                ):
                    continue
            elif self.task_type == "regression":
                if (
                    "regression" not in info.task_types
                    and "both" not in info.task_types
                ):
                    continue

            if info.max_samples and n_samples > info.max_samples:
                continue
            if info.min_samples and n_samples < info.min_samples:
                continue

            candidates.append(name)

        return candidates if candidates else ["lgbm"]

    # ── Introspection ───────────────────────────────────────────────

    def get_rung_summary(self) -> list[dict[str, Any]]:
        """Return a summary of all rungs and their results.

        Returns
        -------
        list of dict
            One entry per rung with keys: rung, fraction, n_configs,
            n_evaluated, status.
        """
        summary = []
        for i in range(self.n_rungs):
            entry: dict[str, Any] = {
                "rung": i,
                "fraction": self.fractions[i],
                "n_configs": self.configs_per_rung[i],
            }

            if i < self._rung:
                entry["status"] = "completed"
                entry["n_evaluated"] = self.configs_per_rung[i]
            elif i == self._rung:
                entry["status"] = "active"
                entry["n_evaluated"] = len(self._rung_scores)
            else:
                entry["status"] = "pending"
                entry["n_evaluated"] = 0

            summary.append(entry)

        return summary

    @property
    def current_rung(self) -> int:
        """Current rung index."""
        return self._rung

    @property
    def current_fraction(self) -> float:
        """Data fraction for the current rung."""
        return self.fractions[self._rung]

    @property
    def is_final_rung(self) -> bool:
        """True if currently on the final (full-data) rung."""
        return self._rung >= self.n_rungs - 1
