"""Metadata collection for model persistence."""

import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

CURRENT_FORMAT_VERSION = 1


@dataclass
class ModelMetadata:
    """Metadata stored alongside a persisted model.

    Attributes
    ----------
    endgame_version : str
        Version of endgame that saved the model.
    format_version : int
        Persistence format version for forward compatibility.
    model_class : str
        Fully qualified class name of the estimator.
    model_params : dict
        Parameters from ``get_params()``.
    created_at : str
        ISO 8601 timestamp of when the model was saved.
    python_version : str
        Python version used to save the model.
    dependencies : dict
        Versions of key dependencies (numpy, sklearn, torch, etc.).
    n_features_in_ : int or None
        Number of input features (if fitted).
    feature_names_in_ : list of str or None
        Input feature names (if available).
    classes_ : list or None
        Class labels for classifiers.
    is_fitted : bool
        Whether the estimator was fitted when saved.
    backend : str
        Backend used for serialization ("joblib", "torch", or "pickle").
    compression : int or None
        Compression level used.
    """

    endgame_version: str = ""
    format_version: int = CURRENT_FORMAT_VERSION
    model_class: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    python_version: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    n_features_in_: int | None = None
    feature_names_in_: list[str] | None = None
    classes_: list | None = None
    is_fitted: bool = False
    backend: str = "joblib"
    compression: int | None = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        # numpy arrays aren't JSON-serializable
        if d["classes_"] is not None:
            d["classes_"] = [
                x.item() if isinstance(x, np.generic) else x
                for x in d["classes_"]
            ]
        # Clean model_params of non-serializable values
        d["model_params"] = _sanitize_params(d["model_params"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMetadata":
        """Create from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _sanitize_params(params: dict) -> dict:
    """Make params JSON-serializable by converting non-standard types."""
    clean = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            clean[k] = v.item()
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, dict):
            clean[k] = _sanitize_params(v)
        elif isinstance(v, (str, int, float, bool, type(None), list)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def _get_dependency_versions() -> dict[str, str]:
    """Collect versions of key ML dependencies."""
    deps = {"numpy": np.__version__}

    for pkg_name, import_name in [
        ("scikit-learn", "sklearn"),
        ("polars", "polars"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("lightgbm", "lightgbm"),
        ("xgboost", "xgboost"),
        ("catboost", "catboost"),
        ("joblib", "joblib"),
    ]:
        try:
            mod = __import__(import_name)
            deps[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return deps


def _check_is_fitted(estimator) -> bool:
    """Check if an estimator appears to be fitted."""
    # EndgameEstimator uses _is_fitted
    if hasattr(estimator, "_is_fitted"):
        return estimator._is_fitted
    if hasattr(estimator, "is_fitted_"):
        return estimator.is_fitted_
    # sklearn convention: fitted estimators have attributes ending in _
    from sklearn.utils.validation import check_is_fitted as sklearn_check

    try:
        sklearn_check(estimator)
        return True
    except Exception:
        return False


def collect_metadata(
    estimator,
    backend: str,
    compression: int | None,
) -> ModelMetadata:
    """Gather metadata by introspecting the estimator and environment.

    Parameters
    ----------
    estimator : estimator object
        The sklearn-compatible estimator to inspect.
    backend : str
        Backend name ("joblib", "torch", or "pickle").
    compression : int or None
        Compression level.

    Returns
    -------
    ModelMetadata
        Populated metadata object.
    """
    import endgame

    cls = type(estimator)
    model_class = f"{cls.__module__}.{cls.__qualname__}"

    # Get params safely
    params = {}
    if hasattr(estimator, "get_params"):
        try:
            params = estimator.get_params(deep=False)
        except Exception:
            pass

    # Feature metadata
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is None:
        n_features = getattr(estimator, "_n_features_in", None)

    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is None:
        feature_names = getattr(estimator, "_feature_names_in", None)
    if feature_names is not None:
        feature_names = list(feature_names)

    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        classes = list(classes)

    return ModelMetadata(
        endgame_version=endgame.__version__,
        format_version=CURRENT_FORMAT_VERSION,
        model_class=model_class,
        model_params=params,
        created_at=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        dependencies=_get_dependency_versions(),
        n_features_in_=n_features,
        feature_names_in_=feature_names,
        classes_=classes,
        is_fitted=_check_is_fitted(estimator),
        backend=backend,
        compression=compression,
    )
