"""Basis function classes for MARS (Multivariate Adaptive Regression Splines).

This module contains the core data structures for representing MARS basis functions:
- HingeSpec: Specification for a single hinge function
- BasisFunction: A product of hinge functions (potentially with interactions)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class HingeSpec:
    """Specification for a single hinge in a basis function.

    A hinge function is either:
    - max(0, x - knot) when direction = +1 (positive hinge)
    - max(0, knot - x) when direction = -1 (negative hinge)

    Parameters
    ----------
    feature_idx : int
        Which feature (column index) this hinge operates on.
    knot : float
        Knot location (threshold value) for the hinge.
    direction : int
        +1 for max(0, x-knot), -1 for max(0, knot-x).

    Examples
    --------
    >>> import numpy as np
    >>> hinge = HingeSpec(feature_idx=0, knot=5.0, direction=1)
    >>> x = np.array([3.0, 5.0, 7.0])
    >>> hinge.evaluate(x)
    array([0., 0., 2.])
    """
    feature_idx: int
    knot: float
    direction: int  # +1 for max(0, x-knot), -1 for max(0, knot-x)

    def __post_init__(self):
        """Validate direction is +1 or -1."""
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction}")

    def evaluate(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate hinge on feature values.

        Parameters
        ----------
        x : ndarray of shape (n_samples,)
            Values of the feature for this hinge.

        Returns
        -------
        result : ndarray of shape (n_samples,)
            Hinge function values.
        """
        if self.direction == 1:
            return np.maximum(0.0, x - self.knot)
        else:
            return np.maximum(0.0, self.knot - x)

    def __str__(self) -> str:
        """Return string representation of the hinge."""
        if self.direction == 1:
            return f"h(x{self.feature_idx} - {self.knot:.4g})"
        else:
            return f"h({self.knot:.4g} - x{self.feature_idx})"

    def to_str_with_name(self, feature_name: str) -> str:
        """Return string representation using feature name.

        Parameters
        ----------
        feature_name : str
            Name to use for the feature.

        Returns
        -------
        str
            Formatted hinge string with feature name.
        """
        if self.direction == 1:
            return f"h({feature_name} - {self.knot:.4g})"
        else:
            return f"h({self.knot:.4g} - {feature_name})"


@dataclass
class BasisFunction:
    """A MARS basis function, which is a product of hinge functions.

    For degree 1 (no interactions): single hinge or constant
    For degree 2+: product of multiple hinges on different features

    The constant (intercept) basis function has an empty hinges list.

    Parameters
    ----------
    hinges : list of HingeSpec, default=[]
        List of hinge specifications that make up this basis function.
        Empty list represents the constant/intercept term.

    Examples
    --------
    >>> import numpy as np
    >>> # Intercept term
    >>> intercept = BasisFunction()
    >>> X = np.random.randn(5, 3)
    >>> intercept.evaluate(X)
    array([1., 1., 1., 1., 1.])

    >>> # Single hinge basis function
    >>> bf = BasisFunction([HingeSpec(0, 0.5, 1)])
    >>> bf.degree
    1

    >>> # Interaction term (product of two hinges)
    >>> bf_interact = BasisFunction([
    ...     HingeSpec(0, 0.5, 1),
    ...     HingeSpec(1, -0.3, -1)
    ... ])
    >>> bf_interact.degree
    2
    """
    hinges: list[HingeSpec] = field(default_factory=list)

    @property
    def degree(self) -> int:
        """Number of hinges (interaction degree).

        Returns
        -------
        int
            0 for intercept, 1 for single hinge, 2+ for interactions.
        """
        return len(self.hinges)

    @property
    def is_intercept(self) -> bool:
        """True if this is the constant/intercept term.

        Returns
        -------
        bool
            True if no hinges (intercept term).
        """
        return len(self.hinges) == 0

    @property
    def feature_indices(self) -> list[int]:
        """List of feature indices involved in this basis function.

        Returns
        -------
        list of int
            Indices of features used in the hinges.
        """
        return [h.feature_idx for h in self.hinges]

    def evaluate(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate basis function on data matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        result : ndarray of shape (n_samples,)
            Basis function values.
        """
        if self.is_intercept:
            return np.ones(X.shape[0], dtype=np.float64)

        result = np.ones(X.shape[0], dtype=np.float64)
        for hinge in self.hinges:
            result *= hinge.evaluate(X[:, hinge.feature_idx])
        return result

    def __str__(self) -> str:
        """Return string representation of the basis function."""
        if self.is_intercept:
            return "1"
        return " * ".join(str(h) for h in self.hinges)

    def to_str_with_names(self, feature_names: list[str]) -> str:
        """Return string representation using feature names.

        Parameters
        ----------
        feature_names : list of str
            Names to use for features.

        Returns
        -------
        str
            Formatted basis function string with feature names.
        """
        if self.is_intercept:
            return "1"
        return " * ".join(
            h.to_str_with_name(feature_names[h.feature_idx])
            for h in self.hinges
        )

    def can_extend_with(self, feature_idx: int, max_degree: int) -> bool:
        """Check if this basis function can be extended with a new hinge.

        A basis function can be extended if:
        1. Adding another hinge won't exceed max_degree
        2. The feature isn't already used in this basis function

        Parameters
        ----------
        feature_idx : int
            Index of feature to potentially add.
        max_degree : int
            Maximum allowed degree (number of hinges).

        Returns
        -------
        bool
            True if extension is allowed.
        """
        if self.degree >= max_degree:
            return False
        if feature_idx in self.feature_indices:
            return False  # Can't have two hinges on same feature
        return True

    def copy(self) -> BasisFunction:
        """Create a copy of this basis function.

        Returns
        -------
        BasisFunction
            A new BasisFunction with copied hinges.
        """
        return BasisFunction(
            hinges=[
                HingeSpec(h.feature_idx, h.knot, h.direction)
                for h in self.hinges
            ]
        )

    def extend(self, hinge: HingeSpec) -> BasisFunction:
        """Create a new basis function by adding a hinge.

        Parameters
        ----------
        hinge : HingeSpec
            Hinge to add.

        Returns
        -------
        BasisFunction
            New basis function with the additional hinge.
        """
        new_hinges = [
            HingeSpec(h.feature_idx, h.knot, h.direction)
            for h in self.hinges
        ]
        new_hinges.append(hinge)
        return BasisFunction(hinges=new_hinges)


class LinearBasisFunction:
    """A linear (no-hinge) basis function for features with linear relationships.

    This is used when allow_linear=True and a feature shows a purely linear
    relationship with the target. Instead of two hinges (positive and negative),
    we use a single linear term.

    Parameters
    ----------
    feature_idx : int
        Which feature (column index) this linear term operates on.

    Examples
    --------
    >>> import numpy as np
    >>> linear = LinearBasisFunction(feature_idx=0)
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> linear.evaluate(X)
    array([1., 3., 5.])
    """

    def __init__(self, feature_idx: int):
        self.feature_idx = feature_idx
        self.hinges: list[HingeSpec] = []  # For compatibility with BasisFunction interface

    @property
    def degree(self) -> int:
        """Linear terms have degree 1."""
        return 1

    @property
    def is_intercept(self) -> bool:
        """Linear terms are not the intercept."""
        return False

    @property
    def feature_indices(self) -> list[int]:
        """Return list containing just this feature index."""
        return [self.feature_idx]

    def evaluate(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate linear term on data matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        result : ndarray of shape (n_samples,)
            Feature values (no transformation).
        """
        return X[:, self.feature_idx].astype(np.float64)

    def __str__(self) -> str:
        """Return string representation."""
        return f"x{self.feature_idx}"

    def to_str_with_names(self, feature_names: list[str]) -> str:
        """Return string representation using feature name."""
        return feature_names[self.feature_idx]

    def can_extend_with(self, feature_idx: int, max_degree: int) -> bool:
        """Linear terms cannot be extended."""
        return False

    def copy(self) -> LinearBasisFunction:
        """Create a copy of this linear basis function."""
        return LinearBasisFunction(self.feature_idx)
