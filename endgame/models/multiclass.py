"""Helpers for running binary-only classifiers against multiclass targets.

Several interpretable models in endgame (SLIM, FasterRisk, GOSDT, GAM,
CORELS) are binary-only by construction — their ``fit`` raises
``ValueError`` when more than two classes are present. To keep the
estimator surface uniform, wrap them with
:class:`sklearn.multiclass.OneVsRestClassifier` for multiclass targets.

Example
-------
    from endgame.models.interpretable import FasterRiskClassifier
    from endgame.models.multiclass import ovr_wrap, is_binary_only

    model = FasterRiskClassifier(max_coef=5, sparsity=10)
    if is_binary_only("fasterrisk") and n_classes > 2:
        model = ovr_wrap(model)
    model.fit(X, y)
"""

from __future__ import annotations

from sklearn.multiclass import OneVsRestClassifier


def ovr_wrap(estimator):
    """Wrap ``estimator`` with :class:`OneVsRestClassifier`.

    The returned estimator fits one copy of ``estimator`` per class and is
    sklearn-compatible (supports ``predict``, ``predict_proba``, etc.).
    """
    return OneVsRestClassifier(estimator)


def is_binary_only(model_id: str) -> bool:
    """Return True if the registry entry for ``model_id`` has
    ``binary_only=True``.

    Safe to call even when the model is not in the registry (returns False).
    """
    from endgame.automl.model_registry import MODEL_REGISTRY

    info = MODEL_REGISTRY.get(model_id)
    return bool(info and getattr(info, "binary_only", False))
