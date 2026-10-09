"""ChimeraBoost and CTBoost: the two new gradient boosters in TabArena's top 30 (Oct 2026), re-exported.

Both packages ship native scikit-learn estimators (``get_params``, ``fit(X, y, sample_weight=...)``,
``predict_proba``, ``feature_importances_``), so these names are the packages' own classes, imported on first access.
Name categorical columns explicitly (``cat_features``, names or indices: a ``fit`` argument for ChimeraBoost, a
constructor argument for CTBoost); both take numpy or pandas and NaN.

- ``ChimeraBoostClassifier`` / ``ChimeraBoostRegressor`` / ``ChimeraBoostQuantileRegressor``
  (https://github.com/bbstats/chimeraboost, Apache-2.0; numba, CPU). Early stopping on an internal 20 % split is on by
  default; TabArena ran ``n_estimators=10000``. ``fit``'s third positional argument is ``cat_features``, so pass
  ``sample_weight`` by keyword. ``pip install chimeraboost``
- ``CTBoostClassifier`` / ``CTBoostRegressor`` (https://github.com/captnmarkus/ctboost, Apache-2.0; C++, optional
  CUDA via ``task_type="GPU"``). Conditional-inference-test splits. Early stopping needs an explicit ``eval_set``.
  TabArena ran ``iterations=1000, learning_rate=0.05, subsample=0.8, bootstrap_type="Bernoulli", ordered_ctr=True,
  max_cat_threshold=64`` with ``early_stopping_rounds=50``. ``pip install ctboost``

Examples
--------
>>> from endgame.models.boosters import ChimeraBoostClassifier
>>> clf = ChimeraBoostClassifier(random_state=0).fit(X_df, y, cat_features=["city"])
>>> from endgame.models.boosters import CTBoostClassifier
>>> clf = CTBoostClassifier(cat_features=["city"], random_state=0).fit(X_df, y)
"""

from __future__ import annotations

import importlib

_SOURCES = {
    "ChimeraBoostClassifier": "chimeraboost",
    "ChimeraBoostRegressor": "chimeraboost",
    "ChimeraBoostQuantileRegressor": "chimeraboost",
    "CTBoostClassifier": "ctboost",
    "CTBoostRegressor": "ctboost",
}
__all__ = list(_SOURCES)


def __getattr__(name):
    if name not in _SOURCES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        cls = getattr(importlib.import_module(_SOURCES[name]), name)
    except ImportError as exc:
        raise ImportError(f"{name} needs the {_SOURCES[name]} package: pip install {_SOURCES[name]}") from exc
    globals()[name] = cls
    return cls
