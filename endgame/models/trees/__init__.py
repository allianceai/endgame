"""Custom tree-based models.

When ``treeple`` is installed, its Cython-compiled oblique / patch-oblique /
extra-oblique / honest forests are used in place of our pure-Python fallbacks.
"""

from endgame.models.trees.adtree import (
    AlternatingDecisionTreeClassifier,
    AlternatingModelTreeRegressor,
)
from endgame.models.trees.c50 import (
    C50Classifier,
    C50Ensemble,
)
from endgame.models.trees.cubist import (
    CubistRegressor,
)
from endgame.models.trees.evtree import (
    EvolutionaryTreeClassifier,
    EvolutionaryTreeRegressor,
)

# ── treeple-backed models (Cython, fast) ────────────────────────────
# Fall back to our pure-Python implementations when treeple is absent.
try:
    from treeple import (
        ExtraObliqueRandomForestClassifier,
        ExtraObliqueRandomForestRegressor,
        HonestForestClassifier,
        ObliqueRandomForestClassifier,
        PatchObliqueRandomForestClassifier,
        PatchObliqueRandomForestRegressor,
    )

    # treeple's ObliqueRandomForestRegressor has a broken __sklearn_tags__
    # on sklearn 1.7+ (sets multi_label on RegressorTags which no longer
    # has that attribute).  Use our pure-Python regressor instead.
    from endgame.models.trees.oblique_forest import (
        ObliqueRandomForestRegressor,
    )

    _HAS_TREEPLE = True
except ImportError:
    from endgame.models.trees.oblique_forest import (
        ObliqueRandomForestClassifier,
        ObliqueRandomForestRegressor,
    )

    ExtraObliqueRandomForestClassifier = None  # type: ignore[assignment,misc]
    ExtraObliqueRandomForestRegressor = None  # type: ignore[assignment,misc]
    PatchObliqueRandomForestClassifier = None  # type: ignore[assignment,misc]
    PatchObliqueRandomForestRegressor = None  # type: ignore[assignment,misc]
    HonestForestClassifier = None  # type: ignore[assignment,misc]
    _HAS_TREEPLE = False

from endgame.models.trees.oblique_tree import (
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
)
from endgame.models.trees.quantile_forest import (
    QuantileRegressorForest,
    interval_coverage,
    interval_width,
    pinball_loss,
)
from endgame.models.trees.rotation_forest import (
    RotationForestClassifier,
    RotationForestRegressor,
)

__all__ = [
    "RotationForestClassifier",
    "RotationForestRegressor",
    "C50Classifier",
    "C50Ensemble",
    "CubistRegressor",
    "ObliqueRandomForestClassifier",
    "ObliqueRandomForestRegressor",
    "ObliqueDecisionTreeClassifier",
    "ObliqueDecisionTreeRegressor",
    "AlternatingDecisionTreeClassifier",
    "AlternatingModelTreeRegressor",
    "QuantileRegressorForest",
    "pinball_loss",
    "interval_coverage",
    "interval_width",
    "EvolutionaryTreeClassifier",
    "EvolutionaryTreeRegressor",
]

if _HAS_TREEPLE:
    __all__.extend([
        "ExtraObliqueRandomForestClassifier",
        "ExtraObliqueRandomForestRegressor",
        "PatchObliqueRandomForestClassifier",
        "PatchObliqueRandomForestRegressor",
        "HonestForestClassifier",
    ])
