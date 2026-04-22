from __future__ import annotations

"""GlassboxMixin — unified structure-extraction interface.

Any estimator whose fitted form is human-inspectable (decision tree,
rule list, linear model, GAM, Bayesian network, scoring system,
symbolic expression, subgroup box, ...) should inherit
:class:`GlassboxMixin` and implement :meth:`_structure_content`.

Users then call :meth:`get_structure` on the fitted estimator to obtain
a machine-readable ``dict`` describing the learned model. The dict
always carries the same top-level metadata plus a ``structure_type``
discriminator that indicates which structure-specific keys are
present.

Recognised ``structure_type`` values
------------------------------------
``"tree"``
    Single decision / model tree. Payload: ``tree`` (nested dict with
    ``feature``, ``threshold``, ``children`` / ``value`` leaves),
    optional ``n_nodes``, ``n_leaves``, ``max_depth``, ``text``.
``"tree_ensemble"``
    Collection of trees. Payload: ``trees`` (list of tree dicts),
    optional ``weights``.
``"rules"``
    Ordered rule list. Payload: ``rules`` (list of dicts with
    ``conditions``, ``prediction``/``weight``), optional ``default``
    or ``intercept``.
``"fuzzy_rules"``
    Rules with fuzzy antecedents. Payload: ``rules``.
``"boxes"``
    Axis-aligned subgroup boxes (PRIM). Payload: ``boxes``.
``"linear"``
    Linear model. Payload: ``coefficients`` (dict or 2D list),
    ``intercept``, optional ``link``.
``"additive"``
    Sum of per-feature / per-interaction terms (GAM, EBM, MARS,
    NAM, GAMI-Net, NODE-GAM). Payload: ``terms`` (list of dicts with
    ``name``, ``features``, ``type``, ``importance`` and
    structure-specific fields), optional ``intercept``.
``"scorecard"``
    Integer scoring system (SLIM, FasterRisk). Payload: ``scorecard``
    (list of ``{feature, threshold, points}``), ``intercept``.
``"bayesian_network"``
    Discrete BN. Payload: ``nodes``, ``edges``, optional ``cpts``.
``"symbolic"``
    Symbolic expression. Payload: ``equation``, optional
    ``pareto_frontier``, ``complexity``.
"""

from abc import abstractmethod
from typing import Any

import numpy as np


class GlassboxMixin:
    """Mixin marking an estimator as structure-inspectable."""

    _structure_type: str = "generic"

    @abstractmethod
    def _structure_content(self) -> dict[str, Any]:
        """Return the structure-specific payload for this estimator.

        Called by :meth:`get_structure` after the standard metadata
        (``model_type``, ``structure_type``, ``n_features``,
        ``feature_names``, ``classes``) has been collected. Implementers
        may assume the estimator is fitted.
        """
        raise NotImplementedError

    def get_structure(
        self,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a machine-readable dict describing the fitted model.

        Parameters
        ----------
        feature_names : list of str, optional
            Override the feature names attached to the fitted model.

        Returns
        -------
        dict
            Keys always present:

            - ``model_type`` — estimator class name.
            - ``structure_type`` — discriminator (see module docstring).
            - ``n_features`` — number of input features.
            - ``feature_names`` — list of feature names.

            For classifiers, ``classes`` and ``n_classes`` are added.
            All other keys are specific to ``structure_type``.
        """
        self._glassbox_check_fitted()

        n_features = self._structure_n_features()
        names = (
            list(feature_names)
            if feature_names is not None
            else self._structure_feature_names(n_features)
        )

        result: dict[str, Any] = {
            "model_type": self.__class__.__name__,
            "structure_type": self._structure_type,
            "n_features": n_features,
            "feature_names": names,
        }

        classes = getattr(self, "classes_", None)
        if classes is not None:
            result["classes"] = np.asarray(classes).tolist()
            result["n_classes"] = len(result["classes"])

        payload = self._structure_content()
        if payload:
            for key, value in payload.items():
                result[key] = value
        return result

    def _glassbox_check_fitted(self) -> None:
        """Best-effort fitted-state verification."""
        checker = getattr(self, "_check_is_fitted", None)
        if callable(checker):
            checker()
            return
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(self)
        except Exception:
            pass

    def _structure_n_features(self) -> int:
        for attr in ("n_features_in_", "_n_features_in", "n_features_"):
            n = getattr(self, attr, None)
            if isinstance(n, (int, np.integer)):
                return int(n)
        names = getattr(self, "feature_names_in_", None) or getattr(
            self, "feature_names_", None
        )
        if names is not None:
            return len(list(names))
        return 0

    def _structure_feature_names(self, n_features: int) -> list[str]:
        for attr in ("feature_names_in_", "feature_names_"):
            names = getattr(self, attr, None)
            if names is not None:
                return [str(n) for n in list(names)]
        return [f"x{i}" for i in range(n_features)]


def sklearn_tree_to_dict(
    tree,
    feature_names: list[str] | None = None,
    class_names: list[Any] | None = None,
) -> dict[str, Any]:
    """Convert a fitted sklearn ``Tree`` to a nested dict.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        The low-level tree object (``estimator.tree_``).
    feature_names : list of str, optional
        Names used for each split feature.
    class_names : list, optional
        Class labels (for classifiers) used to decorate leaves.
    """
    from sklearn.tree._tree import TREE_LEAF

    def recurse(node: int, depth: int) -> dict[str, Any]:
        if tree.children_left[node] == TREE_LEAF:
            value = tree.value[node].tolist()
            leaf: dict[str, Any] = {
                "type": "leaf",
                "depth": depth,
                "n_samples": int(tree.n_node_samples[node]),
                "value": value,
            }
            if class_names is not None and len(value) == 1:
                counts = value[0]
                leaf["class"] = class_names[int(np.argmax(counts))]
            return leaf
        feat = int(tree.feature[node])
        return {
            "type": "split",
            "depth": depth,
            "feature_index": feat,
            "feature": (
                feature_names[feat]
                if feature_names is not None and feat < len(feature_names)
                else f"x{feat}"
            ),
            "threshold": float(tree.threshold[node]),
            "n_samples": int(tree.n_node_samples[node]),
            "impurity": float(tree.impurity[node]),
            "left": recurse(int(tree.children_left[node]), depth + 1),
            "right": recurse(int(tree.children_right[node]), depth + 1),
        }

    return {
        "root": recurse(0, 0),
        "n_nodes": int(tree.node_count),
        "max_depth": int(tree.max_depth),
    }


__all__ = ["GlassboxMixin", "sklearn_tree_to_dict"]
