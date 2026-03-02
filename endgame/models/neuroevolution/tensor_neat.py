from __future__ import annotations

"""
TensorNEAT sklearn-compatible classifiers and regressors.

TensorNEAT is a GPU-accelerated NEAT implementation using JAX.
Falls back gracefully if JAX/TensorNEAT are not installed.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

try:
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from tensorneat.problem import BaseProblem

    class _TabularClassificationProblem(BaseProblem):
        jitable = True

        def __init__(self, n_feats, n_outputs, n_samples, X_jax, y_jax):
            self._n_feats = n_feats
            self._n_outputs = n_outputs
            self._n_samples = n_samples
            self._X_jax = X_jax
            self._y_jax = y_jax

        @property
        def input_shape(self):
            return (self._n_feats,)

        @property
        def output_shape(self):
            return (self._n_outputs,)

        def evaluate(self, state, randkey, act_func, params):
            outputs = jax.vmap(
                lambda xi: act_func(state, params, xi)
            )(self._X_jax)
            preds = jnp.argmax(outputs, axis=-1)
            correct = jnp.sum(preds == self._y_jax)
            return correct / self._n_samples

    class _TabularRegressionProblem(BaseProblem):
        jitable = True

        def __init__(self, n_feats, n_samples, X_jax, y_jax):
            self._n_feats = n_feats
            self._n_samples = n_samples
            self._X_jax = X_jax
            self._y_jax = y_jax

        @property
        def input_shape(self):
            return (self._n_feats,)

        @property
        def output_shape(self):
            return (1,)

        def evaluate(self, state, randkey, act_func, params):
            outputs = jax.vmap(
                lambda xi: act_func(state, params, xi)
            )(self._X_jax)
            mse = jnp.mean((outputs[:, 0] - self._y_jax) ** 2)
            return -mse

    _HAS_TENSORNEAT = True
except ImportError:
    _HAS_TENSORNEAT = False


_MAX_NODES = 150
_MAX_CONNS = 1500
_MAX_EVAL_SAMPLES = 2000


def _safe_genome_params(n_inputs, n_outputs, requested_pop, n_samples=None):
    """Compute max_nodes, max_conns, pop_size with memory-safe caps.

    JAX pre-allocates arrays of shape (pop_size, max_nodes/max_conns, ...)
    so large input/output spaces can cause OOM.  The node/conn caps are
    the primary defence; pop_size is auto-reduced so that each generation
    takes roughly ≤5 seconds on CPU.
    """
    n_io = n_inputs + n_outputs
    initial_conns = n_inputs * n_outputs

    max_nodes = min(n_io + 20, _MAX_NODES)
    max_conns = min(initial_conns + max_nodes * 2, _MAX_CONNS)

    max_conns = max(max_conns, initial_conns + 1)
    max_nodes = max(max_nodes, n_io + 1)

    n_eval = min(n_samples or _MAX_EVAL_SAMPLES, _MAX_EVAL_SAMPLES)

    # Empirical: gen_time ≈ pop * n_eval * max_conns * 1.6e-8 s (CPU, JAX)
    target_gen_secs = 4.0
    cost_per_unit = n_eval * max_conns * 1.6e-8
    if cost_per_unit > 0:
        time_pop = int(target_gen_secs / cost_per_unit)
    else:
        time_pop = requested_pop

    pop_size = min(requested_pop, max(50, time_pop))

    return max_nodes, max_conns, pop_size


class TensorNEATClassifier(BaseEstimator, ClassifierMixin):
    """
    TensorNEAT classifier — GPU-accelerated neuroevolution via JAX.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Number of evolutionary generations.
    species_size : int
        Target number of species for speciation.
    random_state : int or None
        Random seed for reproducibility.
    verbose : int
        Verbosity level (0 = silent).
    """

    def __init__(self, population_size=1000, n_generations=100, species_size=10,
                 random_state=None, verbose=0):
        if not _HAS_TENSORNEAT:
            raise ImportError(
                "tensorneat and jax are required for TensorNEATClassifier. "
                "Install from GitHub: https://github.com/EMI-Group/tensorneat"
            )
        self.population_size = population_size
        self.n_generations = n_generations
        self.species_size = species_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the TensorNEAT classifier."""
        from tensorneat.algorithm.neat import NEAT
        from tensorneat.genome import DefaultGenome
        from tensorneat.pipeline import Pipeline

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_inputs = X.shape[1]
        n_outputs = self.n_classes_

        seed = self.random_state if self.random_state is not None else 0

        n_samples_raw = X.shape[0]
        max_nodes, max_conns, pop_size = _safe_genome_params(
            n_inputs, n_outputs, self.population_size, n_samples_raw,
        )

        X_jax = jnp.array(X)
        y_jax = jnp.array(y, dtype=jnp.int32)
        n_samples = n_samples_raw

        if n_samples > _MAX_EVAL_SAMPLES:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_samples, _MAX_EVAL_SAMPLES, replace=False)
            X_eval, y_eval = X_jax[idx], y_jax[idx]
            n_samples = _MAX_EVAL_SAMPLES
        else:
            X_eval, y_eval = X_jax, y_jax

        problem = _TabularClassificationProblem(
            n_feats=n_inputs, n_outputs=n_outputs,
            n_samples=n_samples, X_jax=X_eval, y_jax=y_eval,
        )

        genome = DefaultGenome(
            num_inputs=n_inputs,
            num_outputs=n_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
        )

        algorithm = NEAT(
            genome=genome,
            pop_size=pop_size,
            species_size=self.species_size,
        )

        pipeline = Pipeline(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            generation_limit=self.n_generations,
        )

        state = pipeline.setup()
        state, best = pipeline.auto_run(state)

        self.pipeline_ = pipeline
        self.best_state_ = state
        self.best_genome_params_ = best

        return self

    def predict_proba(self, X):
        """Predict class probabilities using the best evolved genome."""
        from scipy.special import softmax

        X = np.asarray(X, dtype=np.float32)
        state = self.best_state_
        algo = self.pipeline_.algorithm
        best_genome = self.best_genome_params_

        transformed = algo.transform(state, best_genome)
        X_jax = jnp.array(X)
        raw_outputs = jax.vmap(
            lambda xi: algo.forward(state, transformed, xi)
        )(X_jax)

        return softmax(np.array(raw_outputs), axis=1)

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class TensorNEATRegressor(BaseEstimator, RegressorMixin):
    """
    TensorNEAT regressor — GPU-accelerated neuroevolution via JAX.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Number of evolutionary generations.
    species_size : int
        Target number of species for speciation.
    random_state : int or None
        Random seed for reproducibility.
    verbose : int
        Verbosity level (0 = silent).
    """

    def __init__(self, population_size=1000, n_generations=100, species_size=10,
                 random_state=None, verbose=0):
        if not _HAS_TENSORNEAT:
            raise ImportError("tensorneat and jax are required for TensorNEATRegressor")
        self.population_size = population_size
        self.n_generations = n_generations
        self.species_size = species_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the TensorNEAT regressor."""
        from tensorneat.algorithm.neat import NEAT
        from tensorneat.common.functions import act_jnp
        from tensorneat.genome import DefaultGenome
        from tensorneat.genome.gene import DefaultNode
        from tensorneat.pipeline import Pipeline

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_inputs = X.shape[1]

        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_norm = (y - self._y_mean) / self._y_std

        seed = self.random_state if self.random_state is not None else 0

        n_samples_raw = X.shape[0]
        max_nodes, max_conns, pop_size = _safe_genome_params(
            n_inputs, 1, self.population_size, n_samples_raw,
        )

        X_jax = jnp.array(X)
        y_jax = jnp.array(y_norm)
        n_samples = n_samples_raw

        if n_samples > _MAX_EVAL_SAMPLES:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_samples, _MAX_EVAL_SAMPLES, replace=False)
            X_eval, y_eval = X_jax[idx], y_jax[idx]
            n_samples = _MAX_EVAL_SAMPLES
        else:
            X_eval, y_eval = X_jax, y_jax

        problem = _TabularRegressionProblem(
            n_feats=n_inputs, n_samples=n_samples,
            X_jax=X_eval, y_jax=y_eval,
        )

        node_gene = DefaultNode(
            activation_options=[act_jnp.tanh_, act_jnp.relu_, act_jnp.sigmoid_],
            activation_default=act_jnp.tanh_,
        )

        genome = DefaultGenome(
            num_inputs=n_inputs,
            num_outputs=1,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=node_gene,
        )

        algorithm = NEAT(
            genome=genome,
            pop_size=pop_size,
            species_size=self.species_size,
        )

        pipeline = Pipeline(
            algorithm=algorithm,
            problem=problem,
            seed=seed,
            generation_limit=self.n_generations,
        )

        state = pipeline.setup()
        state, best = pipeline.auto_run(state)

        self.pipeline_ = pipeline
        self.best_state_ = state
        self.best_genome_params_ = best

        return self

    def predict(self, X):
        """Predict continuous values using the best evolved genome."""
        X = np.asarray(X, dtype=np.float32)
        state = self.best_state_
        algo = self.pipeline_.algorithm
        best_genome = self.best_genome_params_

        transformed = algo.transform(state, best_genome)
        X_jax = jnp.array(X)
        raw_outputs = jax.vmap(
            lambda xi: algo.forward(state, transformed, xi)
        )(X_jax)

        # Clip to 5 std devs in normalized space, then denormalize
        preds = np.array(raw_outputs[:, 0])
        preds = np.clip(preds, -5.0, 5.0)
        return preds * self._y_std + self._y_mean
