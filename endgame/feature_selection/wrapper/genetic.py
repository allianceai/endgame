from __future__ import annotations

"""Genetic/evolutionary feature selection."""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GeneticSelector(TransformerMixin, BaseEstimator):
    """Genetic algorithm for feature selection.

    Evolves feature subsets using selection, crossover, and mutation
    to optimize cross-validation score.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Model to use for fitness evaluation.

    population_size : int, default=50
        Size of the population.

    n_generations : int, default=100
        Number of generations.

    mutation_rate : float, default=0.1
        Probability of mutating each gene (feature).

    crossover_rate : float, default=0.8
        Probability of crossover between parents.

    tournament_size : int, default=3
        Number of individuals in tournament selection.

    elitism : int, default=2
        Number of best individuals to keep unchanged.

    min_features : int, default=1
        Minimum number of features to select.

    max_features : int or float, optional
        Maximum features. If float, fraction of total.

    scoring : str, optional
        Scoring metric.

    cv : int, default=5
        Cross-validation folds.

    early_stopping : int, optional
        Stop if no improvement for this many generations.

    random_state : int, optional
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    selected_features_ : ndarray
        Indices of selected features.

    best_score_ : float
        Best cross-validation score achieved.

    history_ : list
        Best score at each generation.

    n_features_ : int
        Number of selected features.

    Example
    -------
    >>> from endgame.feature_selection import GeneticSelector
    >>> selector = GeneticSelector(n_generations=50, population_size=30)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        elitism: int = 2,
        min_features: int = 1,
        max_features: int | float | None = None,
        scoring: str | None = None,
        cv: int = 5,
        early_stopping: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.cv = cv
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose

    def _get_estimator(self):
        """Get the estimator to use."""
        if self.estimator is not None:
            return clone(self.estimator)

        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            n_jobs=-1,
        )

    def _init_population(self, n_features: int, rng) -> np.ndarray:
        """Initialize random population."""
        population = np.zeros((self.population_size, n_features), dtype=bool)

        for i in range(self.population_size):
            # Random number of features
            max_f = self._max_features if self._max_features else n_features
            n_select = rng.randint(self.min_features, max_f + 1)
            selected = rng.choice(n_features, size=n_select, replace=False)
            population[i, selected] = True

        return population

    def _fitness(
        self, individual: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute fitness (CV score) for an individual."""
        selected = np.where(individual)[0]

        if len(selected) == 0:
            return -np.inf

        estimator = self._get_estimator()
        try:
            scores = cross_val_score(
                estimator, X[:, selected], y,
                cv=self.cv, scoring=self.scoring
            )
            return np.mean(scores)
        except Exception:
            return -np.inf

    def _tournament_select(
        self, population: np.ndarray, fitness: np.ndarray, rng
    ) -> np.ndarray:
        """Select individual via tournament selection."""
        indices = rng.choice(
            len(population), size=self.tournament_size, replace=False
        )
        winner = indices[np.argmax(fitness[indices])]
        return population[winner].copy()

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, rng
    ) -> tuple:
        """Single-point crossover."""
        if rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = rng.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1, child2

    def _mutate(self, individual: np.ndarray, rng) -> np.ndarray:
        """Mutate individual by flipping genes."""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if rng.random() < self.mutation_rate:
                mutated[i] = not mutated[i]

        return mutated

    def _enforce_constraints(self, individual: np.ndarray, rng) -> np.ndarray:
        """Ensure individual meets min/max feature constraints."""
        n_selected = individual.sum()

        # Too few features
        while n_selected < self.min_features:
            off_indices = np.where(~individual)[0]
            if len(off_indices) == 0:
                break
            turn_on = rng.choice(off_indices)
            individual[turn_on] = True
            n_selected += 1

        # Too many features
        max_f = self._max_features if self._max_features else len(individual)
        while n_selected > max_f:
            on_indices = np.where(individual)[0]
            if len(on_indices) == 0:
                break
            turn_off = rng.choice(on_indices)
            individual[turn_off] = False
            n_selected -= 1

        return individual

    def fit(self, X, y):
        """Fit the genetic selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GeneticSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Set max features
        if isinstance(self.max_features, float):
            self._max_features = int(n_features * self.max_features)
        else:
            self._max_features = self.max_features

        rng = np.random.RandomState(self.random_state)

        # Initialize population
        population = self._init_population(n_features, rng)

        # Evaluate initial fitness
        fitness = np.array([self._fitness(ind, X, y) for ind in population])

        self.history_ = []
        best_score = -np.inf
        no_improvement = 0

        for gen in range(self.n_generations):
            # Sort by fitness
            sorted_idx = np.argsort(fitness)[::-1]
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]

            gen_best = fitness[0]
            self.history_.append(gen_best)

            if self.verbose and (gen + 1) % 10 == 0:
                print(
                    f"Generation {gen + 1}/{self.n_generations}: "
                    f"best={gen_best:.4f}, "
                    f"mean={np.mean(fitness):.4f}, "
                    f"features={population[0].sum()}"
                )

            # Check for improvement
            if gen_best > best_score:
                best_score = gen_best
                best_individual = population[0].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Early stopping
            if self.early_stopping and no_improvement >= self.early_stopping:
                if self.verbose:
                    print(f"Early stopping at generation {gen + 1}")
                break

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            for i in range(self.elitism):
                new_population.append(population[i].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population, fitness, rng)
                parent2 = self._tournament_select(population, fitness, rng)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2, rng)

                # Mutation
                child1 = self._mutate(child1, rng)
                child2 = self._mutate(child2, rng)

                # Constraints
                child1 = self._enforce_constraints(child1, rng)
                child2 = self._enforce_constraints(child2, rng)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population)

            # Evaluate new population
            fitness = np.array([self._fitness(ind, X, y) for ind in population])

        # Store results
        self.selected_features_ = np.where(best_individual)[0]
        self.best_score_ = best_score
        self.n_features_ = len(self.selected_features_)
        self._support_mask = best_individual

        return self

    def transform(self, X) -> np.ndarray:
        """Select features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray
            Data with selected features.
        """
        check_is_fitted(self, "selected_features_")
        X = check_array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        check_is_fitted(self, "_support_mask")
        if indices:
            return self.selected_features_
        return self._support_mask
