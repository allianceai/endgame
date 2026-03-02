from __future__ import annotations

"""Core save/load functions for model persistence."""

from pathlib import Path
from typing import Any

from endgame.persistence._backends import (
    load_joblib,
    load_pickle,
    load_torch,
    save_joblib,
    save_pickle,
    save_torch,
)
from endgame.persistence._detection import detect_backend
from endgame.persistence._metadata import collect_metadata


def save(
    estimator: Any,
    path: str,
    backend: str = "auto",
    compress: int | None = None,
) -> str:
    """Save any sklearn-compatible estimator to disk.

    Parameters
    ----------
    estimator : estimator object
        A fitted (or unfitted) sklearn-compatible estimator.
    path : str
        Destination file or directory path. The appropriate extension
        (``.egm`` for single-file, ``.egd`` for PyTorch directory)
        will be added automatically.
    backend : str, default="auto"
        Serialization backend. ``"auto"`` selects ``"torch"`` for
        estimators containing ``nn.Module`` attributes, ``"joblib"``
        otherwise. Explicit options: ``"joblib"``, ``"torch"``,
        ``"pickle"``.
    compress : int or None
        Compression level (0-9). Only used by the joblib backend.
        ``None`` means no compression.

    Returns
    -------
    str
        The actual path where the model was saved.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> import endgame as eg
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> eg.save(model, "/tmp/my_model")
    '/tmp/my_model.egm'
    >>> loaded = eg.load("/tmp/my_model.egm")
    """
    resolved_backend = detect_backend(estimator, preferred=backend)
    metadata = collect_metadata(estimator, resolved_backend, compress)

    dest = Path(path)

    if resolved_backend == "torch":
        result = save_torch(estimator, metadata, dest, compress)
    elif resolved_backend == "pickle":
        result = save_pickle(estimator, metadata, dest, compress)
    else:
        result = save_joblib(estimator, metadata, dest, compress)

    return str(result)


def load(
    path: str,
    map_location: str | None = None,
) -> Any:
    """Load an estimator from disk.

    Parameters
    ----------
    path : str
        Path to a ``.egm`` file or ``.egd`` directory.
    map_location : str or None
        PyTorch ``map_location`` for loading tensors (e.g. ``"cpu"``).
        Only relevant for ``.egd`` (PyTorch) saves.

    Returns
    -------
    estimator
        The loaded estimator.

    Examples
    --------
    >>> import endgame as eg
    >>> model = eg.load("/tmp/my_model.egm")
    >>> model.predict(X_test)
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"No model found at: {path}")

    if p.is_dir():
        # Directory format — either .egd or a directory without extension
        if (p / "metadata.json").exists():
            estimator, _meta = load_torch(p, map_location=map_location)
            return estimator
        raise ValueError(
            f"Directory {path} does not appear to be a valid .egd model directory "
            "(missing metadata.json)."
        )

    # Single-file format
    try:
        estimator, _meta = load_joblib(p)
        return estimator
    except Exception:
        # Fall back to pickle
        try:
            estimator, _meta = load_pickle(p)
            return estimator
        except Exception as exc:
            raise ValueError(
                f"Could not load model from {path}. "
                f"The file may be corrupted or in an unsupported format."
            ) from exc
