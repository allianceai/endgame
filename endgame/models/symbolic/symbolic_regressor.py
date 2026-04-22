"""Symbolic Regression — pure-Python GP engine with sklearn interface.

Discovers interpretable symbolic equations that best fit the data while
maintaining parsimony through multi-population evolutionary search
and Pareto-frontier tracking.

No Julia or PySR dependency — uses numpy for evaluation and optionally
PyTorch/scipy for constant optimization.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin

from endgame.models.symbolic._constant_optimizer import optimize_constants
from endgame.models.symbolic._expression import (
    Node,
    evaluate,
    to_sympy,
)
from endgame.models.symbolic._operators import (
    OPERATOR_SETS,
    validate_operators,
)
from endgame.models.symbolic._pareto import ParetoFrontier
from endgame.models.symbolic._population import (
    compute_fitness,
    evolve_population,
    migrate,
    ramped_half_and_half,
)

# ============================================================
# Preset configurations
# ============================================================

PRESETS = {
    "fast": {
        "niterations": 20,
        "populations": 15,
        "population_size": 33,
        "maxsize": 20,
        "ncycles_per_iteration": 100,
        "parsimony": 0.0032,
    },
    "default": {
        "niterations": 40,
        "populations": 31,
        "population_size": 27,
        "maxsize": 25,
        "ncycles_per_iteration": 380,
        "parsimony": 0.0,
    },
    "competition": {
        "niterations": 100,
        "populations": 50,
        "population_size": 50,
        "maxsize": 35,
        "ncycles_per_iteration": 550,
        "parsimony": 0.0,
        "weight_optimize": 0.02,
        "should_optimize_constants": True,
        "optimizer_nrestarts": 4,
    },
    "interpretable": {
        "niterations": 60,
        "populations": 31,
        "population_size": 27,
        "maxsize": 15,
        "maxdepth": 5,
        "ncycles_per_iteration": 380,
        "parsimony": 0.01,
        "constraints": {"/": (-1, 9)},
    },
}

# Julia loss name → Python equivalent mapping
_LOSS_MAP = {
    "L2DistLoss()": "mse",
    "L1DistLoss()": "mae",
    "HuberLoss()": "huber",
    "LogitDistLoss()": "logcosh",
}


def _get_loss_fn(loss: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a numpy loss function from a loss name."""
    key = _LOSS_MAP.get(loss, loss)
    if key == "mse":
        return lambda y, yp: float(np.mean((y - yp) ** 2))
    elif key == "mae":
        return lambda y, yp: float(np.mean(np.abs(y - yp)))
    elif key == "huber":
        def huber(y, yp, delta=1.0):
            r = np.abs(y - yp)
            return float(np.mean(np.where(r < delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))))
        return huber
    elif key == "logcosh":
        return lambda y, yp: float(np.mean(np.log(np.cosh(np.clip(y - yp, -500, 500)))))
    else:
        # Treat unknown as MSE with a deprecation warning
        warnings.warn(
            f"Unknown loss {loss!r}, falling back to MSE. "
            f"Supported: 'mse', 'mae', 'huber', 'logcosh', or Julia-style names "
            f"like 'L2DistLoss()'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return lambda y, yp: float(np.mean((y - yp) ** 2))


class SymbolicRegressor(GlassboxMixin, BaseEstimator, RegressorMixin):
    """Symbolic Regression for discovering interpretable equations.

    Uses multi-population genetic programming with Pareto-frontier
    tracking to find symbolic expressions balancing accuracy and
    complexity.

    Parameters
    ----------
    preset : str, default="default"
        Preset configuration: "fast", "default", "competition", "interpretable".
    operators : str or dict, default="scientific"
        Operator set name or dict with "binary_operators"/"unary_operators".
    binary_operators : list of str, optional
        Explicit binary operators (overrides *operators*).
    unary_operators : list of str, optional
        Explicit unary operators (overrides *operators*).
    niterations : int, optional
        Number of GP iterations.
    maxsize : int, optional
        Max tree complexity (nodes).
    maxdepth : int, optional
        Max tree depth.
    populations : int, optional
        Number of sub-populations.
    population_size : int, optional
        Individuals per population.
    parsimony : float, optional
        Complexity penalty added to loss.
    model_selection : str, default="best"
        "best" (lowest loss) or "score" (loss-complexity trade-off).
    loss : str, default="L2DistLoss()"
        Loss function name. Accepts Julia-style names for backward
        compatibility (e.g. ``"L2DistLoss()"``) or Python names
        (``"mse"``, ``"mae"``, ``"huber"``).
    constraints : dict, optional
        Reserved for API compatibility (not enforced in GP engine).
    nested_constraints : dict, optional
        Reserved for API compatibility.
    denoise : bool, default=False
        Reserved for API compatibility.
    select_k_features : int, optional
        Reserved for API compatibility.
    turbo : bool, default=False
        Reserved for API compatibility.
    parallelism : str, default="multithreading"
        Reserved for API compatibility (GP runs single-threaded).
    procs : int, optional
        Reserved for API compatibility.
    random_state : int, optional
        Random seed.
    verbosity : int, default=0
        0 = silent, 1 = progress, 2 = detailed.
    temp_equation_file : bool, default=True
        Reserved for API compatibility.
    output_directory : str, optional
        Reserved for API compatibility.

    Attributes
    ----------
    equations_ : DataFrame
        All discovered equations with loss and complexity.
    best_equation_ : str
        String of the best equation.
    best_loss_ : float
        Loss of the best equation.
    best_complexity_ : int
        Complexity of the best equation.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray
        Feature names.
    """

    _estimator_type = "regressor"

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
        loss: str = "L2DistLoss()",
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
        self.loss = loss
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

    # --------------------------------------------------------
    # Parameter resolution
    # --------------------------------------------------------

    def _get_operators(self) -> tuple[list[str], list[str]]:
        """Resolve binary/unary operators from settings."""
        if self.binary_operators is not None or self.unary_operators is not None:
            return (
                self.binary_operators or ["+", "-", "*", "/"],
                self.unary_operators or [],
            )
        if isinstance(self.operators, str):
            if self.operators not in OPERATOR_SETS:
                raise ValueError(
                    f"Unknown operator set: {self.operators}. "
                    f"Choose from: {list(OPERATOR_SETS.keys())}"
                )
            op_set = OPERATOR_SETS[self.operators]
            return op_set["binary_operators"], op_set["unary_operators"]
        if isinstance(self.operators, dict):
            return (
                self.operators.get("binary_operators", ["+", "-", "*", "/"]),
                self.operators.get("unary_operators", []),
            )
        raise ValueError(f"Invalid operators type: {type(self.operators)}")

    def _build_params(self) -> dict[str, Any]:
        """Resolve all hyperparameters (preset + overrides)."""
        if self.preset not in PRESETS:
            raise ValueError(
                f"Unknown preset: {self.preset}. "
                f"Choose from: {list(PRESETS.keys())}"
            )
        params = PRESETS[self.preset].copy()

        if self.niterations is not None:
            params["niterations"] = self.niterations
        if self.maxsize is not None:
            params["maxsize"] = self.maxsize
        if self.maxdepth is not None:
            params["maxdepth"] = self.maxdepth
        if self.populations is not None:
            params["populations"] = self.populations
        if self.population_size is not None:
            params["population_size"] = self.population_size
        if self.parsimony is not None:
            params["parsimony"] = self.parsimony

        return params

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(self, X, y, **fit_params) -> SymbolicRegressor:
        """Fit symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])

        # Resolve parameters
        params = self._build_params()
        n_iter = params["niterations"]
        n_pops = params["populations"]
        pop_size = params["population_size"]
        maxsize = params["maxsize"]
        maxdepth = params.get("maxdepth", 10)
        parsimony_val = params.get("parsimony", 0.0)
        ncycles = params.get("ncycles_per_iteration", 1)
        should_optimize = params.get("should_optimize_constants", False)
        optimizer_restarts = params.get("optimizer_nrestarts", 2)

        binary_ops, unary_ops = self._get_operators()
        validate_operators(binary_ops, unary_ops)
        loss_fn = _get_loss_fn(self.loss)

        rng = np.random.default_rng(self.random_state)

        # Initialize populations
        init_depth = min(4, maxdepth)
        all_pops: list[list[Node]] = []
        for _ in range(n_pops):
            pop = ramped_half_and_half(rng, pop_size, self.n_features_in_,
                                       binary_ops, unary_ops, 1, init_depth)
            all_pops.append(pop)

        frontier = ParetoFrontier()

        # Main loop
        for iteration in range(n_iter):
            all_fitnesses = []

            for p_idx in range(n_pops):
                pop = all_pops[p_idx]
                fitnesses = np.array([
                    compute_fitness(t, X, y, loss_fn, parsimony_val)
                    for t in pop
                ])

                # Multiple cycles per iteration
                for _ in range(ncycles):
                    pop = evolve_population(
                        pop, fitnesses, rng,
                        binary_ops, unary_ops, self.n_features_in_,
                        tournament_size=min(5, pop_size),
                        crossover_prob=0.5,
                        mutation_prob=0.3,
                        elite_frac=0.1,
                        maxsize=maxsize,
                    )
                    fitnesses = np.array([
                        compute_fitness(t, X, y, loss_fn, parsimony_val)
                        for t in pop
                    ])

                all_pops[p_idx] = pop
                all_fitnesses.append(fitnesses)

                # Update frontier with best from this population
                for t, f in zip(pop, fitnesses):
                    if f < 1e19:
                        raw_loss = loss_fn(y, evaluate(t, X))
                        frontier.update(t, raw_loss, list(self.feature_names_in_))

            # Migration every 5 iterations
            if iteration % 5 == 4 and n_pops > 1:
                migrate(all_pops, all_fitnesses, rng, n_migrants=2)

            # Optional constant optimization on best individual
            if should_optimize and iteration % 10 == 9:
                best_entry = frontier.get_best(self.model_selection)
                if best_entry is not None:
                    optimized = optimize_constants(
                        best_entry.tree, X, y, loss_fn,
                        n_restarts=optimizer_restarts,
                    )
                    opt_loss = loss_fn(y, evaluate(optimized, X))
                    frontier.update(optimized, opt_loss, list(self.feature_names_in_))

            if self.verbosity >= 1 and (iteration + 1) % max(1, n_iter // 10) == 0:
                best = frontier.get_best(self.model_selection)
                loss_str = f"{best.loss:.6f}" if best else "N/A"
                print(f"  Iteration {iteration + 1}/{n_iter} — best loss: {loss_str}")

        # Final constant optimization pass
        if should_optimize or self.preset == "competition":
            for entry in list(frontier._best.values()):
                optimized = optimize_constants(
                    entry.tree, X, y, loss_fn,
                    n_restarts=max(optimizer_restarts, 2),
                )
                opt_loss = loss_fn(y, evaluate(optimized, X))
                frontier.update(optimized, opt_loss, list(self.feature_names_in_))

        # Store results
        self._frontier = frontier
        self.equations_ = frontier.to_dataframe()
        self._best_entry = frontier.get_best(self.model_selection)

        if self._best_entry is not None:
            self.best_equation_ = self._best_entry.equation
            self.best_loss_ = self._best_entry.loss
            self.best_complexity_ = self._best_entry.complexity
            self._best_tree = self._best_entry.tree
        else:
            self.best_equation_ = "None"
            self.best_loss_ = float("inf")
            self.best_complexity_ = 0
            self._best_tree = None

        # Mark as fitted (for check_is_fitted compatibility with old API)
        self.model_ = self

        return self

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(self, X, index: int | None = None) -> NDArray:
        """Predict using the discovered equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        index : int, optional
            Complexity level to use. If None, uses best equation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "model_")
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if index is not None:
            entry = self._frontier.get_at_complexity(index)
            if entry is None:
                raise ValueError(f"No equation found at complexity {index}")
            return evaluate(entry.tree, X)

        if self._best_tree is None:
            return np.zeros(X.shape[0])
        return evaluate(self._best_tree, X)

    # --------------------------------------------------------
    # Symbolic export
    # --------------------------------------------------------

    def sympy(self, index: int | None = None):
        """Return SymPy expression of the best (or indexed) equation."""
        check_is_fitted(self, "model_")
        if index is not None:
            entry = self._frontier.get_at_complexity(index)
            if entry is None:
                raise ValueError(f"No equation at complexity {index}")
            return to_sympy(entry.tree, list(self.feature_names_in_))
        if self._best_tree is None:
            import sympy
            return sympy.Integer(0)
        return to_sympy(self._best_tree, list(self.feature_names_in_))

    def latex(self, index: int | None = None) -> str:
        """Return LaTeX string of the equation."""
        check_is_fitted(self, "model_")
        try:
            from sympy import latex as sympy_latex
            return sympy_latex(self.sympy(index=index))
        except ImportError:
            raise ImportError("SymPy is required for LaTeX export")

    # --------------------------------------------------------
    # Inspection
    # --------------------------------------------------------

    def get_best_equation(self) -> str:
        check_is_fitted(self, "model_")
        return self.best_equation_

    def get_pareto_frontier(self) -> pd.DataFrame:
        """Return Pareto-optimal equations as a DataFrame."""
        check_is_fitted(self, "model_")
        return self._frontier.get_pareto_optimal()

    def get_equation_at_complexity(self, complexity: int) -> str | None:
        check_is_fitted(self, "model_")
        entry = self._frontier.get_at_complexity(complexity)
        return entry.equation if entry else None

    @property
    def feature_importances_(self) -> NDArray:
        """Feature importances from equation structure (occurrence count)."""
        check_is_fitted(self, "model_")
        importances = np.zeros(self.n_features_in_)
        if self._best_tree is None:
            return importances
        for i, name in enumerate(self.feature_names_in_):
            importances[i] = self.best_equation_.count(name)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def summary(self) -> str:
        check_is_fitted(self, "model_")
        if self.equations_ is None or len(self.equations_) == 0:
            return "No equations discovered."
        lines = [
            "Symbolic Regression Results",
            "=" * 50,
            f"Best equation: {self.best_equation_}",
            f"Best loss: {self.best_loss_:.6f}",
            f"Best complexity: {self.best_complexity_}",
            "",
            "All equations (sorted by complexity):",
            "-" * 50,
        ]
        for _, row in self.equations_.sort_values("complexity").iterrows():
            lines.append(
                f"  [{row['complexity']:2.0f}] {row['equation']} "
                f"(loss: {row['loss']:.6f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        if hasattr(self, "best_equation_"):
            return f"SymbolicRegressor(best={self.best_equation_}, loss={self.best_loss_:.4f})"
        return f"SymbolicRegressor(preset={self.preset!r}, operators={self.operators!r})"

    _structure_type = "symbolic"

    def _structure_content(self) -> dict[str, Any]:
        check_is_fitted(self, "model_")
        frontier = []
        try:
            pf = self._frontier.get_pareto_optimal()
            for _, row in pf.iterrows():
                frontier.append({
                    "complexity": int(row["complexity"]),
                    "equation": str(row["equation"]),
                    "loss": float(row["loss"]),
                })
        except Exception:
            pass
        return {
            "equation": str(self.best_equation_),
            "best_loss": float(self.best_loss_),
            "best_complexity": int(self.best_complexity_),
            "pareto_frontier": frontier,
            "feature_importances": self.feature_importances_.tolist(),
        }
