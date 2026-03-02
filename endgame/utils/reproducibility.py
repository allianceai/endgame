from __future__ import annotations

"""Reproducibility utilities for consistent experiments."""

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (if available)
    - TensorFlow (if available)
    - CUDA (if available)

    Also sets environment variables for deterministic behavior.

    Parameters
    ----------
    seed : int, default=42
        Random seed to use.

    Examples
    --------
    >>> from endgame.utils import seed_everything
    >>> seed_everything(42)
    """
    # Python random
    random.seed(seed)

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # For TF 2.x
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


class SeedEverything:
    """Context manager for reproducible experiments.

    Sets random seeds on entry and optionally restores state on exit.

    Parameters
    ----------
    seed : int, default=42
        Random seed to use.
    restore : bool, default=False
        Whether to restore random state on exit.

    Examples
    --------
    >>> with SeedEverything(42):
    ...     # Reproducible code here
    ...     pass

    >>> seed_ctx = SeedEverything(42)
    >>> with seed_ctx:
    ...     result = train_model()
    """

    def __init__(self, seed: int = 42, restore: bool = False):
        self.seed = seed
        self.restore = restore
        self._python_state: tuple | None = None
        self._numpy_state: dict | None = None
        self._torch_state: any | None = None
        self._cuda_state: any | None = None

    def __enter__(self) -> SeedEverything:
        """Enter the context manager."""
        if self.restore:
            # Save current state
            self._python_state = random.getstate()
            self._numpy_state = np.random.get_state()

            try:
                import torch
                self._torch_state = torch.get_rng_state()
                if torch.cuda.is_available():
                    self._cuda_state = torch.cuda.get_rng_state_all()
            except ImportError:
                pass

        # Set seeds
        seed_everything(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        if self.restore:
            # Restore state
            if self._python_state is not None:
                random.setstate(self._python_state)

            if self._numpy_state is not None:
                np.random.set_state(self._numpy_state)

            try:
                import torch
                if self._torch_state is not None:
                    torch.set_rng_state(self._torch_state)
                if self._cuda_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(self._cuda_state)
            except ImportError:
                pass


class ReproducibleRun:
    """Context manager for fully reproducible ML experiments.

    Combines seed setting with logging of environment info.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    log_environment : bool, default=True
        Whether to log environment information.

    Examples
    --------
    >>> with ReproducibleRun(seed=42) as run:
    ...     print(run.environment_info)
    ...     # Train model
    ...     pass
    """

    def __init__(self, seed: int = 42, log_environment: bool = True):
        self.seed = seed
        self.log_environment = log_environment
        self.environment_info: dict = {}
        self._seed_ctx: SeedEverything | None = None

    def __enter__(self) -> ReproducibleRun:
        """Enter the context manager."""
        # Set seeds
        self._seed_ctx = SeedEverything(self.seed)
        self._seed_ctx.__enter__()

        # Log environment
        if self.log_environment:
            self.environment_info = self._get_environment_info()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        if self._seed_ctx is not None:
            self._seed_ctx.__exit__(exc_type, exc_val, exc_tb)

    def _get_environment_info(self) -> dict:
        """Get environment information for logging."""
        import platform
        import sys

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "seed": self.seed,
        }

        # Add package versions
        packages = ["sklearn", "pandas", "polars", "torch", "lightgbm", "xgboost", "catboost"]
        for pkg in packages:
            try:
                mod = __import__(pkg)
                info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass

        return info

    def save_config(self, filepath: str) -> None:
        """Save reproducibility configuration to file.

        Parameters
        ----------
        filepath : str
            Output file path (JSON or YAML).
        """
        import json

        config = {
            "seed": self.seed,
            "environment": self.environment_info,
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
