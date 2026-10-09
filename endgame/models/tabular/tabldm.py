"""Xiaomi-TabLDM: Xiaomi's in-context tabular foundation model (arXiv 2609.03880; Apache-2.0; TabArena top 15, Oct 2026).

Re-exports ``tabldm.TabLDMEnhancedClassifier`` / ``tabldm.TabLDMEnhancedRegressor``, the estimators TabArena ran
(https://github.com/xiaomi-research/xiaomi-tabldm). They are already scikit-learn estimators that load their checkpoint
in ``fit``: ``TabLDMEnhancedClassifier(n_estimators=8, device=None, random_state=42, n_jobs=None, ...)``. String,
category and bool columns are ordinal-encoded, NaN is mean-imputed, more than 10 classes are supported natively.
Weights (``occams/Xiaomi-TabLDM``, ~0.56 GB per task, Apache-2.0, not gated) download on first fit.

Install the commit TabArena pinned; later commits removed the ``Enhanced`` classes and changed defaults::

    pip install "Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@6773a30d43e43fad3e8b474e20ca8c7ec40dcd76"
"""

try:
    from tabldm import TabLDMEnhancedClassifier, TabLDMEnhancedRegressor
except ImportError as exc:
    raise ImportError("Xiaomi-TabLDM is not installed: pip install 'Xiaomi-TabLDM @ "
                      "git+https://github.com/xiaomi-research/xiaomi-tabldm.git@6773a30d43e43fad3e8b474e20ca8c7ec40dcd76'") from exc

__all__ = ["TabLDMEnhancedClassifier", "TabLDMEnhancedRegressor"]
