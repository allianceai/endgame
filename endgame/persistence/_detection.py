"""PyTorch module detection for persistence backend selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch



def _torch_available() -> bool:
    """Check if PyTorch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def has_torch_modules(estimator) -> bool:
    """Check if any estimator attributes are ``nn.Module`` instances.

    Parameters
    ----------
    estimator : object
        The estimator to inspect.

    Returns
    -------
    bool
        True if the estimator contains PyTorch modules.
    """
    if not _torch_available():
        return False
    return len(find_torch_modules(estimator)) > 0


def find_torch_modules(estimator) -> dict[str, torch.nn.Module]:
    """Scan estimator attributes for ``nn.Module`` instances.

    Checks common attribute names (``model_``, ``module_``, ``_model``,
    ``network_``, ``net_``) and then falls back to scanning all ``vars()``.

    Parameters
    ----------
    estimator : object
        The estimator to inspect.

    Returns
    -------
    dict
        Mapping of attribute name to ``nn.Module`` instance.
    """
    if not _torch_available():
        return {}

    import torch.nn as nn

    modules: dict[str, nn.Module] = {}

    # Check common attribute names first
    common_attrs = [
        "model_", "module_", "_model", "network_", "net_",
        "_module", "_network", "_net",
    ]

    checked = set()
    for attr_name in common_attrs:
        val = getattr(estimator, attr_name, None)
        if val is not None and isinstance(val, nn.Module):
            modules[attr_name] = val
        checked.add(attr_name)

    # Scan all vars for anything we missed
    for attr_name, val in vars(estimator).items():
        if attr_name in checked:
            continue
        if isinstance(val, nn.Module):
            modules[attr_name] = val

    return modules


def detect_backend(estimator, preferred: str = "auto") -> str:
    """Determine which persistence backend to use.

    Parameters
    ----------
    estimator : object
        The estimator to save.
    preferred : str, default="auto"
        Preferred backend. ``"auto"`` selects based on estimator contents.
        Explicit values: ``"joblib"``, ``"torch"``, ``"pickle"``.

    Returns
    -------
    str
        Backend name: ``"torch"`` if the estimator has ``nn.Module``
        attributes and PyTorch is available, else ``"joblib"``.
    """
    if preferred != "auto":
        return preferred

    if _torch_available() and has_torch_modules(estimator):
        return "torch"

    return "joblib"
