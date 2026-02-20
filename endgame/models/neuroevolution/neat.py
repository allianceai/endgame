"""
NEAT (NeuroEvolution of Augmenting Topologies) sklearn-compatible classifiers and regressors.

Uses neat-python as the backend. Follows the same sklearn estimator pattern as
endgame's SymbolicClassifier/SymbolicRegressor.
"""

import tempfile
import textwrap

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class NEATClassifier(BaseEstimator, ClassifierMixin):
    """
    NEAT classifier using neat-python.

    Evolves neural network topology and weights using the NEAT algorithm.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Number of evolutionary generations.
    n_hidden : int
        Initial number of hidden nodes (0 = minimal topology).
    activation_default : str
        Default activation function for new nodes.
    random_state : int or None
        Random seed for reproducibility.
    verbose : int
        Verbosity level (0 = silent).
    """

    def __init__(self, population_size=150, n_generations=100, n_hidden=0,
                 activation_default='sigmoid', random_state=None, verbose=0):
        import neat as _neat  # noqa: F401
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_hidden = n_hidden
        self.activation_default = activation_default
        self.random_state = random_state
        self.verbose = verbose

    def _create_neat_config(self, n_inputs, n_outputs):
        """Generate a neat-python INI config string."""
        config_text = textwrap.dedent(f"""\
        [NEAT]
        fitness_criterion     = max
        fitness_threshold     = 1.0
        pop_size              = {self.population_size}
        reset_on_extinction   = True
        no_fitness_termination = True

        [DefaultGenome]
        # node activation options
        activation_default      = {self.activation_default}
        activation_mutate_rate  = 0.1
        activation_options      = sigmoid tanh relu

        # node aggregation options
        aggregation_default     = sum
        aggregation_mutate_rate = 0.0
        aggregation_options     = sum

        # node bias options
        bias_init_mean          = 0.0
        bias_init_stdev         = 1.0
        bias_init_type          = gaussian
        bias_max_value          = 30.0
        bias_min_value          = -30.0
        bias_mutate_power       = 0.5
        bias_mutate_rate        = 0.7
        bias_replace_rate       = 0.1

        # genome compatibility options
        compatibility_disjoint_coefficient = 1.0
        compatibility_weight_coefficient   = 0.5

        # connection add/remove rates
        conn_add_prob           = 0.5
        conn_delete_prob        = 0.2

        # connection enable options
        enabled_default             = True
        enabled_mutate_rate         = 0.01
        enabled_rate_to_true_add    = 0.0
        enabled_rate_to_false_add   = 0.0

        feed_forward            = True
        initial_connection      = full_direct
        single_structural_mutation = False
        structural_mutation_surer = default

        # node add/remove rates
        node_add_prob           = 0.3
        node_delete_prob        = 0.1

        # network parameters
        num_hidden              = {self.n_hidden}
        num_inputs              = {n_inputs}
        num_outputs             = {n_outputs}

        # node response options
        response_init_mean      = 1.0
        response_init_stdev     = 0.0
        response_init_type      = gaussian
        response_max_value      = 30.0
        response_min_value      = -30.0
        response_mutate_power   = 0.0
        response_mutate_rate    = 0.0
        response_replace_rate   = 0.0

        # connection weight options
        weight_init_mean        = 0.0
        weight_init_stdev       = 1.0
        weight_init_type        = gaussian
        weight_max_value        = 30
        weight_min_value        = -30
        weight_mutate_power     = 0.5
        weight_mutate_rate      = 0.8
        weight_replace_rate     = 0.1

        [DefaultSpeciesSet]
        compatibility_threshold = 3.0

        [DefaultStagnation]
        species_fitness_func = max
        max_stagnation       = 20
        species_elitism      = 2

        [DefaultReproduction]
        elitism            = 2
        survival_threshold = 0.2
        min_species_size   = 1
        """)
        return config_text

    def fit(self, X, y):
        """Fit the NEAT classifier by evolving neural network topology."""
        import neat

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_inputs = X.shape[1]
        n_outputs = self.n_classes_

        # Write config to temp file
        config_text = self._create_neat_config(n_inputs, n_outputs)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_text)
            config_path = f.name

        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
        finally:
            import os
            os.unlink(config_path)

        # Set random seed
        if self.random_state is not None:
            import random
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Store training data for eval callback
        self._X_train = X
        self._y_train = y

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                correct = 0
                for xi, yi in zip(self._X_train, self._y_train):
                    output = net.activate(xi.tolist())
                    pred = np.argmax(output)
                    if pred == yi:
                        correct += 1
                genome.fitness = correct / len(self._y_train)

        pop = neat.Population(config)
        if self.verbose > 0:
            pop.add_reporter(neat.StdOutReporter(True))
            pop.add_reporter(neat.StatisticsReporter())

        winner = pop.run(eval_genomes, self.n_generations)
        self.best_genome_ = winner
        self.config_ = config
        self.best_net_ = neat.nn.FeedForwardNetwork.create(winner, config)

        # Cleanup
        del self._X_train
        del self._y_train

        return self

    def predict_proba(self, X):
        """Predict class probabilities using the best evolved network."""
        from scipy.special import softmax

        X = np.asarray(X, dtype=np.float64)
        raw_outputs = np.array([self.best_net_.activate(xi.tolist()) for xi in X])
        return softmax(raw_outputs, axis=1)

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class NEATRegressor(BaseEstimator, RegressorMixin):
    """
    NEAT regressor using neat-python.

    Evolves neural network topology and weights using the NEAT algorithm,
    optimizing for mean squared error.  Targets are normalized internally
    so that network outputs (near [-1, 1]) can match the target scale.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Number of evolutionary generations.
    n_hidden : int
        Initial number of hidden nodes (0 = minimal topology).
    activation_default : str
        Default activation function for new nodes.
    random_state : int or None
        Random seed for reproducibility.
    verbose : int
        Verbosity level (0 = silent).
    """

    def __init__(self, population_size=150, n_generations=100, n_hidden=0,
                 activation_default='tanh', random_state=None, verbose=0):
        import neat as _neat  # noqa: F401
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_hidden = n_hidden
        self.activation_default = activation_default
        self.random_state = random_state
        self.verbose = verbose

    def _create_neat_config(self, n_inputs):
        """Generate a neat-python INI config string for regression (1 output)."""
        config_text = textwrap.dedent(f"""\
        [NEAT]
        fitness_criterion     = max
        fitness_threshold     = 1e10
        pop_size              = {self.population_size}
        reset_on_extinction   = True
        no_fitness_termination = True

        [DefaultGenome]
        activation_default      = {self.activation_default}
        activation_mutate_rate  = 0.1
        activation_options      = sigmoid tanh relu

        aggregation_default     = sum
        aggregation_mutate_rate = 0.0
        aggregation_options     = sum

        bias_init_mean          = 0.0
        bias_init_stdev         = 1.0
        bias_init_type          = gaussian
        bias_max_value          = 30.0
        bias_min_value          = -30.0
        bias_mutate_power       = 0.5
        bias_mutate_rate        = 0.7
        bias_replace_rate       = 0.1

        compatibility_disjoint_coefficient = 1.0
        compatibility_weight_coefficient   = 0.5

        conn_add_prob           = 0.5
        conn_delete_prob        = 0.2

        enabled_default             = True
        enabled_mutate_rate         = 0.01
        enabled_rate_to_true_add    = 0.0
        enabled_rate_to_false_add   = 0.0

        feed_forward            = True
        initial_connection      = full_direct
        single_structural_mutation = False
        structural_mutation_surer = default

        node_add_prob           = 0.3
        node_delete_prob        = 0.1

        num_hidden              = {self.n_hidden}
        num_inputs              = {n_inputs}
        num_outputs             = 1

        response_init_mean      = 1.0
        response_init_stdev     = 0.0
        response_init_type      = gaussian
        response_max_value      = 30.0
        response_min_value      = -30.0
        response_mutate_power   = 0.0
        response_mutate_rate    = 0.0
        response_replace_rate   = 0.0

        weight_init_mean        = 0.0
        weight_init_stdev       = 1.0
        weight_init_type        = gaussian
        weight_max_value        = 30
        weight_min_value        = -30
        weight_mutate_power     = 0.5
        weight_mutate_rate      = 0.8
        weight_replace_rate     = 0.1

        [DefaultSpeciesSet]
        compatibility_threshold = 3.0

        [DefaultStagnation]
        species_fitness_func = max
        max_stagnation       = 20
        species_elitism      = 2

        [DefaultReproduction]
        elitism            = 2
        survival_threshold = 0.2
        min_species_size   = 1
        """)
        return config_text

    def fit(self, X, y):
        """Fit the NEAT regressor by evolving neural network topology."""
        import neat

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Normalize targets so network outputs (near [-1, 1]) can match
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_norm = (y - self._y_mean) / self._y_std

        n_inputs = X.shape[1]
        self._X_train = X
        self._y_train = y_norm

        config_text = self._create_neat_config(n_inputs)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_text)
            config_path = f.name

        try:
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
        finally:
            import os
            os.unlink(config_path)

        if self.random_state is not None:
            import random
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                mse = 0.0
                for xi, yi in zip(self._X_train, self._y_train):
                    output = net.activate(xi.tolist())
                    mse += (output[0] - yi) ** 2
                mse /= len(self._y_train)
                genome.fitness = -mse

        pop = neat.Population(config)
        if self.verbose > 0:
            pop.add_reporter(neat.StdOutReporter(True))
            pop.add_reporter(neat.StatisticsReporter())

        winner = pop.run(eval_genomes, self.n_generations)
        self.best_genome_ = winner
        self.config_ = config
        self.best_net_ = neat.nn.FeedForwardNetwork.create(winner, config)

        del self._X_train
        del self._y_train

        return self

    def predict(self, X):
        """Predict continuous values using the best evolved network."""
        X = np.asarray(X, dtype=np.float64)
        raw = np.array([self.best_net_.activate(xi.tolist())[0] for xi in X])
        return raw * self._y_std + self._y_mean
