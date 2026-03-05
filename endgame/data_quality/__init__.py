from __future__ import annotations

"""Data quality module: profiling, deduplication, drift detection, and data valuation."""

from endgame.data_quality.profiler import DataProfile, DataProfiler
from endgame.data_quality.duplicates import DuplicateDetector, DuplicateReport
from endgame.data_quality.drift import DriftDetector, DriftReport, FeatureDriftResult
from endgame.data_quality.valuation import DataValuator


def profile_data(X, y=None, *, correlation=False):
    """One-line convenience function to profile a dataset.

    Parameters
    ----------
    X : array-like
        Input data (pandas/polars DataFrame or numpy array).
    y : array-like, optional
        Target variable.
    correlation : bool
        Whether to compute a correlation matrix.

    Returns
    -------
    DataProfile
        Profiling results.

    Examples
    --------
    >>> import endgame as eg
    >>> profile = eg.data_quality.profile_data(X)
    >>> print(profile.summary())
    """
    return DataProfiler(correlation=correlation).profile(X, y)


__all__ = [
    "DataProfiler",
    "DataProfile",
    "DuplicateDetector",
    "DuplicateReport",
    "DriftDetector",
    "DriftReport",
    "FeatureDriftResult",
    "DataValuator",
    "profile_data",
]
