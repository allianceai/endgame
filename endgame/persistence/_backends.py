"""Persistence backends for saving and loading estimators."""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

from endgame.persistence._detection import find_torch_modules
from endgame.persistence._metadata import CURRENT_FORMAT_VERSION, ModelMetadata

logger = logging.getLogger(__name__)

# File extensions
EGM_EXT = ".egm"   # single-file format (joblib/pickle)
EGD_EXT = ".egd"   # directory format (torch)


def _ensure_joblib():
    """Import joblib, raising a clear error if unavailable."""
    try:
        import joblib
        return joblib
    except ImportError:
        raise ImportError(
            "joblib is required for model persistence. "
            "Install it with: pip install joblib"
        )


# ---------- Single-file (.egm) backend ----------

def save_joblib(
    estimator: Any,
    metadata: ModelMetadata,
    path: Path,
    compress: int | None,
) -> Path:
    """Save an estimator to a single ``.egm`` file using joblib.

    Parameters
    ----------
    estimator : estimator object
        The estimator to save.
    metadata : ModelMetadata
        Collected metadata.
    path : Path
        Destination path (extension will be normalised to ``.egm``).
    compress : int or None
        Compression level (0-9). None or 0 means no compression.

    Returns
    -------
    Path
        The actual path where the file was written.
    """
    joblib = _ensure_joblib()

    path = path.with_suffix(EGM_EXT)
    payload = {"metadata": metadata.to_dict(), "estimator": estimator}
    joblib.dump(payload, path, compress=compress or 0)
    return path


def load_joblib(path: Path) -> tuple[Any, ModelMetadata]:
    """Load an estimator from a ``.egm`` file.

    Returns
    -------
    tuple of (estimator, ModelMetadata)
    """
    joblib = _ensure_joblib()

    payload = joblib.load(path)
    meta_dict = payload["metadata"]
    _check_format_version(meta_dict)
    metadata = ModelMetadata.from_dict(meta_dict)
    return payload["estimator"], metadata


# ---------- Pickle backend ----------

def save_pickle(
    estimator: Any,
    metadata: ModelMetadata,
    path: Path,
    compress: int | None,
) -> Path:
    """Save an estimator using plain pickle (fallback backend)."""
    path = path.with_suffix(EGM_EXT)
    payload = {"metadata": metadata.to_dict(), "estimator": estimator}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: Path) -> tuple[Any, ModelMetadata]:
    """Load an estimator from a pickle-based ``.egm`` file."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    meta_dict = payload["metadata"]
    _check_format_version(meta_dict)
    metadata = ModelMetadata.from_dict(meta_dict)
    return payload["estimator"], metadata


# ---------- Directory (.egd) backend (PyTorch) ----------

def save_torch(
    estimator: Any,
    metadata: ModelMetadata,
    path: Path,
    compress: int | None,
) -> Path:
    """Save a PyTorch-containing estimator to a ``.egd`` directory.

    Layout::

        path.egd/
            metadata.json
            estimator.joblib
            torch_state/
                <attr_name>.pt   # state_dict for each nn.Module

    Parameters
    ----------
    estimator : estimator object
        The estimator (must contain ``nn.Module`` attributes).
    metadata : ModelMetadata
        Collected metadata.
    path : Path
        Destination path (suffix normalised to ``.egd``).
    compress : int or None
        Compression level for the joblib portion.

    Returns
    -------
    Path
        The directory path.
    """
    import torch
    joblib = _ensure_joblib()

    path = path.with_suffix(EGD_EXT)
    path.mkdir(parents=True, exist_ok=True)

    # Save metadata as JSON
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Move modules to CPU and save state_dicts
    torch_dir = path / "torch_state"
    torch_dir.mkdir(exist_ok=True)

    modules = find_torch_modules(estimator)
    for attr_name, module in modules.items():
        module_cpu = module.cpu()
        torch.save(module_cpu.state_dict(), torch_dir / f"{attr_name}.pt")

    # Save the full estimator with joblib (modules already on CPU)
    joblib.dump(estimator, path / "estimator.joblib", compress=compress or 0)

    return path


def load_torch(
    path: Path,
    map_location: str | None = None,
) -> tuple[Any, ModelMetadata]:
    """Load a PyTorch-containing estimator from a ``.egd`` directory.

    First tries to load the full estimator via joblib. If that fails
    (e.g. device mismatch), falls back to loading ``torch_state/*.pt``
    files with ``map_location``.

    Parameters
    ----------
    path : Path
        The ``.egd`` directory.
    map_location : str or None
        PyTorch ``map_location`` for loading tensors (e.g. ``"cpu"``).

    Returns
    -------
    tuple of (estimator, ModelMetadata)
    """
    import torch
    joblib = _ensure_joblib()

    # Load metadata
    with open(path / "metadata.json") as f:
        meta_dict = json.load(f)
    _check_format_version(meta_dict)
    metadata = ModelMetadata.from_dict(meta_dict)

    # Try direct joblib load first
    try:
        estimator = joblib.load(path / "estimator.joblib")
        if map_location is not None:
            # Move modules to requested device
            modules = find_torch_modules(estimator)
            device = torch.device(map_location)
            for module in modules.values():
                module.to(device)
        return estimator, metadata
    except Exception as exc:
        logger.info(
            "Direct load failed (%s), falling back to state_dict loading", exc
        )

    # Fallback: load with map_location on the state_dicts
    torch_dir = path / "torch_state"
    if not torch_dir.exists():
        raise RuntimeError(
            f"Cannot load model: direct joblib load failed and no "
            f"torch_state directory found in {path}"
        )

    # Reload estimator with pickle allowing map_location
    estimator = joblib.load(path / "estimator.joblib")
    modules = find_torch_modules(estimator)
    loc = map_location or "cpu"
    for attr_name, module in modules.items():
        pt_path = torch_dir / f"{attr_name}.pt"
        if pt_path.exists():
            state = torch.load(pt_path, map_location=loc, weights_only=False)
            module.load_state_dict(state)

    return estimator, metadata


# ---------- Helpers ----------

def _check_format_version(meta_dict: dict) -> None:
    """Warn if the format version doesn't match the current one."""
    version = meta_dict.get("format_version", 0)
    if version != CURRENT_FORMAT_VERSION:
        warnings.warn(
            f"Model was saved with format version {version}, but the current "
            f"version is {CURRENT_FORMAT_VERSION}. Loading may not work correctly.",
            UserWarning,
            stacklevel=4,
        )
