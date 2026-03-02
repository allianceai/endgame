from __future__ import annotations

"""Genetic algorithm-based search strategy.

Evolves full pipeline configurations — model, hyperparameters,
preprocessing, feature selection, dimensionality reduction — using
tournament selection, crossover, and mutation.  Inspired by TPOT's
approach of treating the entire pipeline as the genome.
"""

import copy
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from endgame.automl.model_registry import MODEL_REGISTRY
from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)


# ── Gene pools ──────────────────────────────────────────────────────
# Each option is a (step_name, params_dict) pair that maps directly
# to the ``PipelineConfig.preprocessing`` list and gets built by
# ``orchestrator._build_preprocessing_step``.

_IMPUTER_GENES: list[dict[str, Any]] = [
    {"strategy": "median"},
    {"strategy": "mean"},
    {"strategy": "most_frequent"},
    {"strategy": "knn", "n_neighbors": 3},
    {"strategy": "knn", "n_neighbors": 5},
    {"strategy": "knn", "n_neighbors": 10},
]

_ENCODER_GENES: list[dict[str, Any]] = [
    {"method": "ordinal"},
    {"method": "onehot"},
    {"method": "target"},
]

_SCALER_GENES: list[dict[str, Any] | None] = [
    None,  # no scaler
    {"method": "standard"},
    {"method": "robust"},
    {"method": "quantile"},
    {"method": "minmax"},
]

_FEATURE_SELECTION_GENES: list[dict[str, Any] | None] = [
    None,  # no feature selection
    None,  # extra weight toward no selection (it's safest)
    {"method": "variance_threshold", "threshold": 0.0},
    {"method": "variance_threshold", "threshold": 0.001},
    {"method": "mutual_info", "k": 10},
    {"method": "mutual_info", "k": 20},
    {"method": "mutual_info", "k": 30},
    {"method": "mutual_info", "k": 50},
]

_DIM_REDUCTION_GENES: list[dict[str, Any] | None] = [
    None,  # no dim reduction
    {"method": "pca", "n_components": 0.80},
    {"method": "pca", "n_components": 0.90},
    {"method": "pca", "n_components": 0.95},
    {"method": "pca", "n_components": 0.99},
    {"method": "truncated_svd", "n_components": 5},
    {"method": "truncated_svd", "n_components": 10},
]

# Hyperparameter ranges organised by model family so crossover
# between models of the same family produces meaningful children.
_HP_RANGES: dict[str, dict[str, tuple]] = {
    # GBDTs
    "gbdt": {
        "n_estimators": (100, 5000, "int"),
        "learning_rate": (0.001, 0.3, "log"),
        "max_depth": (3, 12, "int"),
        "num_leaves": (15, 255, "int"),
        "min_child_samples": (1, 100, "int"),
        "reg_alpha": (1e-8, 10.0, "log"),
        "reg_lambda": (1e-8, 10.0, "log"),
        "subsample": (0.5, 1.0, "float"),
        "colsample_bytree": (0.3, 1.0, "float"),
    },
    "neural": {
        "n_epochs": (10, 300, "int"),
        "learning_rate": (1e-5, 1e-2, "log"),
        "batch_size": (16, 512, "int_pow2"),
        "hidden_dim": (32, 512, "int"),
        "dropout": (0.0, 0.5, "float"),
        "weight_decay": (1e-8, 1e-2, "log"),
    },
    "tree": {
        "n_estimators": (50, 2000, "int"),
        "max_depth": (3, 30, "int"),
        "min_samples_split": (2, 50, "int"),
        "min_samples_leaf": (1, 20, "int"),
        "max_features": (0.3, 1.0, "float"),
    },
    "linear": {
        "C": (1e-4, 100.0, "log"),
        "alpha": (1e-6, 10.0, "log"),
    },
    "kernel": {
        "C": (0.01, 1000.0, "log"),
        "gamma": (1e-5, 10.0, "log"),
    },
    "default": {
        "n_estimators": (50, 3000, "int"),
        "learning_rate": (1e-4, 0.3, "log"),
        "max_depth": (3, 15, "int"),
    },
}

_MODEL_FAMILY: dict[str, str] = {}


def _get_model_family(model_name: str) -> str:
    """Map a model name to its HP family."""
    if model_name in _MODEL_FAMILY:
        return _MODEL_FAMILY[model_name]

    info = MODEL_REGISTRY.get(model_name)
    if info is None:
        return "default"
    fam = getattr(info, "family", "")
    if fam in ("gbdt",):
        return "gbdt"
    if fam in ("neural", "deep_tabular"):
        return "neural"
    if fam in ("tree", "forest"):
        return "tree"
    if fam in ("linear", "glm", "interpretable"):
        return "linear"
    if fam in ("kernel", "svm", "gp"):
        return "kernel"
    # Heuristic fallback
    lower = model_name.lower()
    if any(k in lower for k in ("lgbm", "xgb", "catboost", "ngboost")):
        return "gbdt"
    if any(k in lower for k in ("transformer", "saint", "node", "nam", "tab", "mlp", "net")):
        return "neural"
    if any(k in lower for k in ("forest", "tree", "c50", "cubist", "rf", "extra")):
        return "tree"
    if any(k in lower for k in ("linear", "elm", "mars", "slim", "lda", "qda", "naive", "gam")):
        return "linear"
    if any(k in lower for k in ("svm", "gp", "kernel")):
        return "kernel"
    return "default"


def _sample_hp(name: str, low, high, kind: str, rng: random.Random) -> Any:
    """Sample a single hyperparameter value."""
    if kind == "int":
        return rng.randint(int(low), int(high))
    if kind == "int_pow2":
        exp = rng.randint(int(np.log2(low)), int(np.log2(high)))
        return 2 ** exp
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))
    return rng.uniform(float(low), float(high))


def _perturb_hp(value: Any, low, high, kind: str, rng: random.Random, strength: float = 0.3) -> Any:
    """Perturb a value within its range.  ``strength`` ∈ (0, 1) controls
    how far from the original value the perturbation can go."""
    if kind in ("int", "int_pow2"):
        delta = max(1, int((high - low) * strength))
        new = int(value) + rng.randint(-delta, delta)
        return max(int(low), min(int(high), new))
    if kind == "log":
        log_val = np.log(float(value))
        log_lo, log_hi = np.log(float(low)), np.log(float(high))
        delta = (log_hi - log_lo) * strength
        new = log_val + rng.uniform(-delta, delta)
        return float(np.exp(max(log_lo, min(log_hi, new))))
    # float
    delta = (float(high) - float(low)) * strength
    new = float(value) + rng.uniform(-delta, delta)
    return max(float(low), min(float(high), new))


# ── Individual dataclass ────────────────────────────────────────────

@dataclass
class Individual:
    """A single genome in the population.

    The genome encodes a *complete* pipeline: preprocessing steps,
    feature selection, dimensionality reduction, model choice, and
    hyperparameters.
    """

    config: PipelineConfig
    fitness: float = 0.0
    generation: int = 0
    evaluated: bool = False
    parent_ids: list[str] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return self.config.config_id or ""


# ── GeneticSearch ───────────────────────────────────────────────────

class GeneticSearch(BaseSearchStrategy):
    """Evolutionary pipeline optimisation (TPOT-style).

    Each *individual* is a full pipeline configuration:
    ``[imputer] → [encoder] → [scaler] → [feature_selection]
      → [dim_reduction] → model(hyperparams)``

    The search evolves a population through tournament selection,
    uniform crossover, and gene-level mutation.  Every gene
    (preprocessing step, model choice, each hyperparameter) can be
    independently crossed over or mutated.

    Parameters
    ----------
    task_type : str
        ``"classification"`` or ``"regression"``.
    eval_metric : str
        Metric to optimise (``"auto"`` selects based on task).
    model_pool : list of str, optional
        Explicit model pool.  ``None`` → all models in the registry
        compatible with the task.
    population_size : int
        Individuals per generation.
    n_generations : int
        Maximum generations (may stop earlier via ``patience``).
    mutation_rate : float
        Per-gene mutation probability.
    crossover_rate : float
        Probability that two parents produce a crossed-over child
        (vs. cloning the fitter parent).
    tournament_size : int
        Participants per tournament selection round.
    elitism : int
        Top-*k* individuals copied unchanged to the next generation.
    patience : int
        Generations without improvement before early stopping.
    random_state : int, optional
        Seed for reproducibility.
    verbose : int
        0 = silent, 1 = generation summaries, 2 = per-individual logs.
    """

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        model_pool: list[str] | None = None,
        population_size: int = 30,
        n_generations: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elitism: int = 3,
        patience: int = 15,
        random_state: int | None = None,
        verbose: int = 1,
        excluded_models: set[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
            excluded_models=excluded_models,
        )
        self.model_pool = model_pool
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self._base_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.patience = patience
        self.max_model_time: float = kwargs.pop("max_model_time", 300)

        self._rng = random.Random(random_state)
        self._np_rng = np.random.RandomState(random_state)

        # State
        self._population: list[Individual] = []
        self._generation: int = 0
        self._best_fitness: float = -float("inf")
        self._stale_generations: int = 0
        self._available_models: list[str] = []
        self._evaluated_ids: set[str] = set()
        self._hall_of_fame: list[Individual] = []
        self._gen_pending: int = 0  # unevaluated count in current gen
        self._fail_counts: dict[str, int] = {}
        self._blacklisted_models: set[str] = set()
        self._importance_mask: np.ndarray | None = None
        self._importance_scores: np.ndarray | None = None

    # ────────────────────────── public API ──────────────────────────

    def set_feature_importance_feedback(
        self,
        mask: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> None:
        """Inject feature importance feedback from trained models.

        This enables iterative feature selection: future configs may
        include an importance-based feature mask step.

        Parameters
        ----------
        mask : np.ndarray of bool
            Boolean mask — True for features to keep.
        scores : np.ndarray, optional
            Normalised importance scores per feature.
        """
        self._importance_mask = mask
        self._importance_scores = scores

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 5,
    ) -> list[PipelineConfig]:
        """Return up to *n_suggestions* unevaluated configs.

        On the first call, initialises the population.  On subsequent
        calls, returns remaining unevaluated individuals or evolves a
        new generation if all have been scored.
        """
        if self.should_stop():
            return []

        if not self._available_models:
            self._available_models = self._get_available_models(meta_features)

        if not self._population:
            self._init_population(meta_features)
            self._gen_pending = len(self._population)

        # Collect unevaluated individuals
        unevaluated = [ind for ind in self._population if not ind.evaluated]

        if not unevaluated:
            # Entire generation scored — check stopping then evolve
            if self.should_stop():
                return []
            self._evolve()
            unevaluated = [ind for ind in self._population if not ind.evaluated]
            self._gen_pending = len(unevaluated)

        if not unevaluated:
            return []

        # Cap duplicates: at most 2 individuals with the same model_name
        # per batch to prevent monoculture runs.
        MAX_DUPES_PER_BATCH = 2
        batch: list[Individual] = []
        model_counts: dict[str, int] = {}
        for ind in unevaluated:
            name = ind.config.model_name
            if model_counts.get(name, 0) >= MAX_DUPES_PER_BATCH:
                continue
            batch.append(ind)
            model_counts[name] = model_counts.get(name, 0) + 1
            if len(batch) >= n_suggestions:
                break

        if self.verbose > 0 and batch:
            models = [ind.config.model_name for ind in batch]
            remaining = self._gen_pending
            print(
                f"  [Genetic] Gen {self._generation}: suggesting {len(batch)}"
                f"/{remaining} "
                f"({', '.join(models[:6])}{'…' if len(models) > 6 else ''})"
            )

        return [ind.config for ind in batch]

    def update(self, result: SearchResult) -> None:
        """Record the fitness of a config after training."""
        super().update(result)

        # Match by object identity first (suggest returns references),
        # then fall back to config_id matching only unevaluated individuals
        # to avoid accidentally matching an already-evaluated elite.
        target: Individual | None = None
        for ind in self._population:
            if ind.config is result.config:
                target = ind
                break
        if target is None:
            for ind in self._population:
                if (
                    not ind.evaluated
                    and ind.config.config_id == result.config.config_id
                ):
                    target = ind
                    break

        if target is not None:
            target.fitness = result.score if result.success else -1.0
            target.evaluated = True
            self._gen_pending = max(0, self._gen_pending - 1)

        self._evaluated_ids.add(result.config.config_id)

        name = result.config.model_name
        if not result.success:
            self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
            if self._fail_counts[name] >= 3:
                self._blacklisted_models.add(name)
                if self.verbose > 0 and self._fail_counts[name] == 3:
                    print(
                        f"  [Genetic] Blacklisted '{name}' after "
                        f"{self._fail_counts[name]} consecutive failures"
                    )
        else:
            self._fail_counts.pop(name, None)

        if result.success and result.score > 0:
            self._update_hall_of_fame(result)

    def should_stop(self, max_iterations: int | None = None) -> bool:
        if max_iterations and self.n_evaluated_ >= max_iterations:
            return True
        if self._generation >= self.n_generations:
            return True
        if self.patience > 0 and self._stale_generations >= self.patience:
            return True
        return False

    def get_best_individuals(self, n: int = 5) -> list[Individual]:
        return list(self._hall_of_fame[:n])

    # ────────────────────── population init ─────────────────────────

    # Models that are known to be extremely slow or fragile in a forked
    # child process.  They are excluded from the deterministic gen-0
    # sweep but can still appear via mutation / crossover in later gens.
    _SLOW_MODELS: set[str] = {
        "oblique_forest", "patch_oblique_forest",
        "evolutionary_tree",
        "bart",
        "symbolic_regression", "symbolic_regressor",
    }

    # Models that frequently fail due to data requirements they can't
    # communicate through the registry API (e.g. non-negative features,
    # specific feature distributions).  They can still appear via
    # mutation but won't waste a Gen-0 slot.
    _FRAGILE_MODELS: set[str] = {
        "ebmc_classifier",
        "fasterrisk",
        "slim",
    }

    def _init_population(self, meta_features: dict[str, float] | None) -> None:
        """Seed gen 0 with one individual per model type, then fill randomly.

        This guarantees every model is evaluated at least once before
        evolution begins, giving the genetic algorithm a complete fitness
        landscape to select from.  Known-slow models are deferred so
        they don't consume the entire first generation's budget.
        """
        self._population = []
        mf = meta_features or {}

        # Phase 1: one config per model type (deterministic coverage).
        # Prioritise fast models; if there are more model types than
        # population slots, the shuffled order decides which get in.
        skip = self._SLOW_MODELS | self._FRAGILE_MODELS | self._blacklisted_models
        fast_models = [
            m for m in self._available_models
            if m not in skip
        ]
        self._rng.shuffle(fast_models)

        for model_name in fast_models:
            if len(self._population) >= self.population_size:
                break
            config = self._seeded_config(model_name, mf)
            self._population.append(Individual(config=config, generation=0))

        n_seeded = len(self._population)

        # Phase 2: fill remaining slots with random configs for diversity
        while len(self._population) < self.population_size:
            config = self._random_config(mf, exclude_slow=True)
            self._population.append(Individual(config=config, generation=0))

        n_deferred = len([m for m in self._available_models if m in self._SLOW_MODELS])
        n_random = len(self._population) - n_seeded
        n_skipped = max(0, len(fast_models) - n_seeded)

        if self.verbose > 0:
            msg = (
                f"  [Genetic] Gen 0: {n_seeded} model types seeded"
            )
            if n_random:
                msg += f" + {n_random} random"
            msg += f" = {len(self._population)} total"
            if n_deferred:
                msg += f" ({n_deferred} slow models deferred)"
            if n_skipped:
                msg += f" ({n_skipped} models deferred to gen 1+)"
            print(msg)

    def _seeded_config(self, model_name: str, meta_features: dict) -> PipelineConfig:
        """Build a config for a specific model using its default params
        and a light random preprocessing pipeline."""
        info = MODEL_REGISTRY.get(model_name)
        params = info.default_params.copy() if info else {}
        preprocessing = self._random_preprocessing(model_name, meta_features)

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={
                "source": "genetic_seed",
                "generation": 0,
            },
        )

    def _random_config(
        self, meta_features: dict, exclude_slow: bool = False,
    ) -> PipelineConfig:
        """Build a fully random pipeline configuration."""
        pool = self._viable_models()
        if exclude_slow:
            pool = [m for m in pool if m not in self._SLOW_MODELS] or pool
        model_name = self._rng.choice(pool)
        params = self._random_params(model_name)
        preprocessing = self._random_preprocessing(model_name, meta_features)

        config = PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={
                "source": "genetic_search",
                "generation": self._generation,
            },
        )
        return config

    def _random_params(self, model_name: str) -> dict[str, Any]:
        info = MODEL_REGISTRY.get(model_name)
        params = info.default_params.copy() if info else {}

        family = _get_model_family(model_name)
        ranges = _HP_RANGES.get(family, _HP_RANGES["default"])

        for hp, (lo, hi, kind) in ranges.items():
            if hp in params:
                params[hp] = _sample_hp(hp, lo, hi, kind, self._rng)

        return params

    def _random_preprocessing(
        self,
        model_name: str,
        meta_features: dict,
    ) -> list[tuple[str, dict]]:
        """Sample a random preprocessing pipeline."""
        info = MODEL_REGISTRY.get(model_name)
        steps: list[tuple[str, dict]] = []

        # Imputer — if data has missing values and model doesn't handle them
        if meta_features.get("pct_missing", 0) > 0:
            if not (info and info.handles_missing):
                imp = self._rng.choice(_IMPUTER_GENES)
                steps.append(("imputer", imp))

        # Encoder — if data has categoricals and model doesn't handle them
        if meta_features.get("nr_cat", 0) > 0:
            if not (info and info.handles_categorical):
                enc = self._rng.choice(_ENCODER_GENES)
                steps.append(("encoder", enc))

        # Scaler (50% chance)
        scaler = self._rng.choice(_SCALER_GENES)
        if scaler is not None:
            steps.append(("scaler", scaler))

        # Feature selection (30% chance; higher if we have importance feedback)
        fs_prob = 0.40 if self._importance_mask is not None else 0.25
        if self._rng.random() < fs_prob:
            pool = [g for g in _FEATURE_SELECTION_GENES if g is not None]
            if self.max_model_time >= 600:
                pool.append({"method": "boruta"})
            # Add importance-based selection when feedback is available
            if self._importance_mask is not None:
                mask_list = self._importance_mask.tolist()
                pool.append({"method": "importance_mask", "mask": mask_list})
                # Give it extra weight — it's informed by actual model perf
                pool.append({"method": "importance_mask", "mask": mask_list})
            fs = self._rng.choice(pool)
            steps.append(("feature_selection", fs))

        # Dimensionality reduction (25% chance, only if enough features)
        if meta_features.get("nr_attr", 0) > 15 and self._rng.random() < 0.25:
            dr = self._rng.choice([g for g in _DIM_REDUCTION_GENES if g is not None])
            steps.append(("dim_reduction", dr))

        return steps

    # ──────────────────────── evolution ──────────────────────────────

    def _evolve(self) -> None:
        """Create the next generation via selection + crossover + mutation."""
        self._generation += 1

        evaluated = [ind for ind in self._population if ind.evaluated]
        evaluated.sort(key=lambda x: x.fitness, reverse=True)

        if not evaluated:
            self._init_population({})
            return

        gen_best = evaluated[0].fitness
        if gen_best > self._best_fitness + 1e-5:
            improvement = gen_best - self._best_fitness
            self._best_fitness = gen_best
            self._stale_generations = 0
            if self.verbose > 0:
                best = evaluated[0]
                n_preproc = len(best.config.preprocessing)
                print(
                    f"  [Genetic] Gen {self._generation}: "
                    f"★ new best {gen_best:.4f} (+{improvement:.4f}) "
                    f"— {best.config.model_name} "
                    f"({n_preproc} preproc steps)"
                )
        else:
            self._stale_generations += 1
            if self.verbose > 0:
                unique_models = len({ind.config.model_name for ind in evaluated})
                print(
                    f"  [Genetic] Gen {self._generation}: "
                    f"best={gen_best:.4f} "
                    f"(stale {self._stale_generations}/{self.patience}, "
                    f"{unique_models} unique models)"
                )

        # Adaptive mutation: ramp up when stagnating to break out of local optima
        if self._stale_generations >= 5:
            self.mutation_rate = min(0.6, self._base_mutation_rate + 0.1 * self._stale_generations)
        else:
            self.mutation_rate = self._base_mutation_rate

        # Elitism: keep top-k unchanged (diverse elites only)
        new_pop: list[Individual] = []
        elite_model_names: set[str] = set()
        for elite in evaluated[: self.elitism * 2]:
            if len(new_pop) >= self.elitism:
                break
            if elite.config.model_name in elite_model_names:
                continue
            elite_model_names.add(elite.config.model_name)
            clone_ind = Individual(
                config=copy.deepcopy(elite.config),
                fitness=elite.fitness,
                generation=self._generation,
                evaluated=True,
                parent_ids=[elite.uid],
            )
            new_pop.append(clone_ind)

        # Random immigrants: inject fresh random individuals to maintain diversity.
        # More immigrants when stagnating.
        n_immigrants = max(2, self.population_size // 5)
        if self._stale_generations >= 5:
            n_immigrants = max(n_immigrants, self.population_size // 3)
        for _ in range(n_immigrants):
            if len(new_pop) >= self.population_size:
                break
            cfg = self._random_config({}, exclude_slow=True)
            new_pop.append(Individual(
                config=cfg,
                generation=self._generation,
                parent_ids=[],
            ))

        # Fill the rest with offspring
        max_attempts = self.population_size * 5
        attempts = 0
        while len(new_pop) < self.population_size and attempts < max_attempts:
            attempts += 1
            try:
                p1 = self._tournament_select(evaluated)
                p2 = self._tournament_select(evaluated)

                if self._rng.random() < self.crossover_rate:
                    child_cfg = self._crossover(p1.config, p2.config)
                    parent_ids = [p1.uid, p2.uid]
                else:
                    child_cfg = copy.deepcopy(p1.config if p1.fitness >= p2.fitness else p2.config)
                    parent_ids = [p1.uid]

                child_cfg = self._mutate(child_cfg)

                child_cfg.config_id = None
                child_cfg.metadata["generation"] = self._generation
                child_cfg.__post_init__()

                for _ in range(3):
                    if child_cfg.config_id not in self._evaluated_ids:
                        break
                    child_cfg = self._mutate(child_cfg)
                    child_cfg.config_id = None
                    child_cfg.__post_init__()

                new_pop.append(Individual(
                    config=child_cfg,
                    generation=self._generation,
                    parent_ids=parent_ids,
                ))
            except Exception as exc:
                logger.debug(
                    f"Offspring creation failed (attempt {attempts}): {exc}"
                )

        # Enforce model diversity: cap any single model to 40% of population
        max_per_model = max(3, int(self.population_size * 0.4))
        model_counts: dict[str, int] = {}
        diverse_pop: list[Individual] = []
        for ind in new_pop:
            name = ind.config.model_name
            count = model_counts.get(name, 0)
            if count < max_per_model:
                diverse_pop.append(ind)
                model_counts[name] = count + 1
            # Excess individuals of this model are discarded

        # Fill any slots freed by the diversity cap with fresh randoms
        while len(diverse_pop) < self.population_size:
            cfg = self._random_config({}, exclude_slow=True)
            diverse_pop.append(Individual(
                config=cfg,
                generation=self._generation,
                parent_ids=[],
            ))

        self._population = diverse_pop

    def _tournament_select(self, pool: list[Individual]) -> Individual:
        k = min(self.tournament_size, len(pool))
        return max(self._rng.sample(pool, k), key=lambda x: x.fitness)

    # ──────────────────────── crossover ─────────────────────────────

    def _crossover(
        self,
        p1: PipelineConfig,
        p2: PipelineConfig,
    ) -> PipelineConfig:
        """Uniform crossover across all genes."""
        pick = self._rng.random

        # Model gene: pick from one parent
        if pick() < 0.5:
            model_name = p1.model_name
            params = copy.deepcopy(p1.model_params)
        else:
            model_name = p2.model_name
            params = copy.deepcopy(p2.model_params)

        # If both parents have the same model family, blend numeric HPs
        fam1 = _get_model_family(p1.model_name)
        fam2 = _get_model_family(p2.model_name)
        if fam1 == fam2:
            for key in set(p1.model_params) & set(p2.model_params):
                v1, v2 = p1.model_params[key], p2.model_params[key]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if pick() < 0.5:
                        alpha = pick()
                        blended = alpha * v1 + (1 - alpha) * v2
                        params[key] = int(round(blended)) if isinstance(v1, int) else blended

        # Preprocessing genes: per-step uniform crossover
        preprocessing = self._crossover_preprocessing(p1.preprocessing, p2.preprocessing)

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={"source": "genetic_crossover"},
        )

    @staticmethod
    def _crossover_preprocessing(
        steps1: list[tuple[str, dict]],
        steps2: list[tuple[str, dict]],
    ) -> list[tuple[str, dict]]:
        """Per-step uniform crossover of preprocessing pipelines."""
        all_step_names = sorted(
            {s[0] for s in steps1} | {s[0] for s in steps2}
        )
        result = []
        for name in all_step_names:
            s1 = next((s for s in steps1 if s[0] == name), None)
            s2 = next((s for s in steps2 if s[0] == name), None)

            if s1 and s2:
                result.append(random.choice([s1, s2]))
            elif s1 or s2:
                step = s1 or s2
                if random.random() < 0.7:
                    result.append(step)
            # else: neither parent has it → skip
        return result

    # ──────────────────────── mutation ───────────────────────────────

    def _mutate(self, config: PipelineConfig) -> PipelineConfig:
        """Apply gene-level mutations."""
        config = copy.deepcopy(config)
        rate = self.mutation_rate

        # Replace blacklisted model unconditionally
        if config.model_name in self._blacklisted_models:
            viable = self._viable_models()
            if viable:
                config.model_name = self._rng.choice(viable)
                config.model_params = self._random_params(config.model_name)

        # Mutate model choice with adaptive rate
        if self._rng.random() < rate:
            viable = self._viable_models()
            if viable:
                config.model_name = self._rng.choice(viable)
                config.model_params = self._random_params(config.model_name)

        # Mutate hyperparameters
        family = _get_model_family(config.model_name)
        ranges = _HP_RANGES.get(family, _HP_RANGES["default"])
        for hp, (lo, hi, kind) in ranges.items():
            if hp in config.model_params and self._rng.random() < rate:
                config.model_params[hp] = _perturb_hp(
                    config.model_params[hp], lo, hi, kind, self._rng,
                )

        # Mutate preprocessing steps
        config.preprocessing = self._mutate_preprocessing(config.preprocessing)

        return config

    def _feature_selection_pool(self) -> list[dict[str, Any] | None]:
        """Feature selection genes, including boruta only when budget allows."""
        pool = list(_FEATURE_SELECTION_GENES)
        if self.max_model_time >= 600:
            pool.append({"method": "boruta"})
        return pool

    def _mutate_preprocessing(
        self,
        steps: list[tuple[str, dict]],
    ) -> list[tuple[str, dict]]:
        steps = list(steps)  # shallow copy
        rate = self.mutation_rate

        gene_pools: dict[str, list] = {
            "imputer": _IMPUTER_GENES,
            "encoder": _ENCODER_GENES,
            "scaler": _SCALER_GENES,
            "feature_selection": self._feature_selection_pool(),
            "dim_reduction": _DIM_REDUCTION_GENES,
        }

        for step_name, pool in gene_pools.items():
            if self._rng.random() >= rate:
                continue

            existing_idx = next(
                (i for i, (n, _) in enumerate(steps) if n == step_name),
                None,
            )
            new_gene = self._rng.choice(pool)

            if new_gene is None:
                if existing_idx is not None:
                    steps.pop(existing_idx)
            elif existing_idx is not None:
                steps[existing_idx] = (step_name, new_gene)
            else:
                steps.append((step_name, new_gene))

        return steps

    # ──────────────────── helper methods ────────────────────────────

    def _viable_models(self) -> list[str]:
        """Available models minus blacklisted ones."""
        viable = [
            m for m in self._available_models
            if m not in self._blacklisted_models
        ]
        return viable or self._available_models

    def _get_available_models(
        self,
        meta_features: dict[str, float] | None = None,
    ) -> list[str]:
        """Return models compatible with the task and optional pool."""
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
                if "classification" not in info.task_types and "both" not in info.task_types:
                    continue
            elif self.task_type == "regression":
                if "regression" not in info.task_types and "both" not in info.task_types:
                    continue

            if info.max_samples and n_samples > info.max_samples:
                continue
            if info.min_samples and n_samples < info.min_samples:
                continue

            candidates.append(name)

        if not candidates:
            candidates = ["lgbm"]

        return candidates

    def _update_hall_of_fame(self, result: SearchResult, max_size: int = 20) -> None:
        """Keep a sorted list of the best-ever individuals."""
        entry = Individual(
            config=result.config,
            fitness=result.score,
            generation=self._generation,
            evaluated=True,
        )
        self._hall_of_fame.append(entry)
        self._hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        self._hall_of_fame = self._hall_of_fame[:max_size]
