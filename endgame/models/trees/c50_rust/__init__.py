from __future__ import annotations

"""C5.0 Rust extension module."""

try:
    from endgame.models.trees.c50_rust.c50_rust import C50Classifier, C50Ensemble
    __all__ = ["C50Classifier", "C50Ensemble"]
except ImportError:
    # Rust extension not built
    pass
