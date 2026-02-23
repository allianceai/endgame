"""
TensorNEAT sklearn-compatible classifiers and regressors.

TensorNEAT is a GPU-accelerated NEAT implementation using JAX.
Falls back gracefully if JAX/TensorNEAT are not installed.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


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
        self.population_size = population_size
        self.n_generations = n_generations
        self.species_size = species_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the TensorNEAT classifier."""
        try:
            import jax
            import jax.numpy as jnp
            from tensorneat.pipeline import Pipeline
            from tensorneat.algorithm.neat import NEAT
            from tensorneat.genome import DefaultGenome
            from tensorneat.problem import BaseProblem
        except ImportError:
            raise ImportError(
                "TensorNEAT requires JAX and tensorneat. "
                "Install: pip install jax jaxlib tensorneat>=0.3"
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_inputs = X.shape[1]
        n_outputs = self.n_classes_

        # Store data for inference
        self._X_train = X
        self._y_train = y

        seed = self.random_state if self.random_state is not None else 0

        class TabularClassificationProblem(BaseProblem):
            jitable = False

            def __init__(self, X_data, y_data, n_classes):
                self.X_data = X_data
                self.y_data = y_data
                self._n_classes = n_classes

            @property
            def input_shape(self):
                return (self.X_data.shape[1],)

            @property
            def output_shape(self):
                return (self._n_classes,)

            def evaluate(self, state, network, params):
                correct = 0
                for xi, yi in zip(self.X_data, self.y_data):
                    output = network(state, params, jnp.array(xi))
                    pred = jnp.argmax(output)
                    if int(pred) == int(yi):
                        correct += 1
                return correct / len(self.y_data)

        problem = TabularClassificationProblem(X, y, self.n_classes_)

        genome = DefaultGenome(
            num_inputs=n_inputs,
            num_outputs=n_outputs,
        )

        algorithm = NEAT(
            genome=genome,
            pop_size=self.population_size,
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

        del self._X_train
        del self._y_train

        return self

    def predict_proba(self, X):
        """Predict class probabilities using the best evolved genome."""
        try:
            import jax.numpy as jnp
            from scipy.special import softmax
        except ImportError:
            raise ImportError("JAX and scipy required for TensorNEAT inference.")

        X = np.asarray(X, dtype=np.float32)
        pipeline = self.pipeline_
        state = self.best_state_
        params = self.best_genome_params_

        raw_outputs = []
        for xi in X:
            output = pipeline.algorithm.genome.network(
                state, params, jnp.array(xi)
            )
            raw_outputs.append(np.array(output))

        raw_outputs = np.array(raw_outputs)
        return softmax(raw_outputs, axis=1)

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
        self.population_size = population_size
        self.n_generations = n_generations
        self.species_size = species_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the TensorNEAT regressor."""
        try:
            import jax
            import jax.numpy as jnp
            from tensorneat.pipeline import Pipeline
            from tensorneat.algorithm.neat import NEAT
            from tensorneat.genome import DefaultGenome
            from tensorneat.problem import BaseProblem
        except ImportError:
            raise ImportError(
                "TensorNEAT requires JAX and tensorneat. "
                "Install: pip install jax jaxlib tensorneat>=0.3"
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_inputs = X.shape[1]

        seed = self.random_state if self.random_state is not None else 0

        class TabularRegressionProblem(BaseProblem):
            jitable = False

            def __init__(self, X_data, y_data):
                self.X_data = X_data
                self.y_data = y_data

            @property
            def input_shape(self):
                return (self.X_data.shape[1],)

            @property
            def output_shape(self):
                return (1,)

            def evaluate(self, state, network, params):
                mse = 0.0
                for xi, yi in zip(self.X_data, self.y_data):
                    output = network(state, params, jnp.array(xi))
                    mse += float((output[0] - yi) ** 2)
                mse /= len(self.y_data)
                return -mse  # Negative MSE (higher is better)

        problem = TabularRegressionProblem(X, y)

        genome = DefaultGenome(
            num_inputs=n_inputs,
            num_outputs=1,
        )

        algorithm = NEAT(
            genome=genome,
            pop_size=self.population_size,
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
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX required for TensorNEAT inference.")

        X = np.asarray(X, dtype=np.float32)
        pipeline = self.pipeline_
        state = self.best_state_
        params = self.best_genome_params_

        predictions = []
        for xi in X:
            output = pipeline.algorithm.genome.network(
                state, params, jnp.array(xi)
            )
            predictions.append(float(output[0]))

        return np.array(predictions)
