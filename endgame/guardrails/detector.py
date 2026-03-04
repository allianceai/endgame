from __future__ import annotations

"""LeakageDetector: sklearn-compatible transformer for data quality and leakage detection."""

import logging
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from endgame.guardrails.checks import CHECK_REGISTRY, COST_ORDER
from endgame.guardrails.report import GuardrailsReport
from endgame.guardrails.utils import check_time_budget, normalize_input, reconstruct_output

logger = logging.getLogger(__name__)


class LeakageDetector(TransformerMixin, BaseEstimator):
    """Detect and optionally remove data quality issues and target leakage.

    A standalone sklearn-compatible transformer that runs configurable checks
    for data health, target leakage, feature redundancy, and train/test issues.

    Parameters
    ----------
    mode : str, default="detect"
        ``"detect"`` logs issues only; ``"fix"`` removes flagged features in
        ``transform()``.
    checks : str or list of str, default="default"
        ``"default"`` runs all checks that are on by default.
        ``"all"`` runs every registered check.
        A list of check names runs exactly those checks.
    leakage_threshold : float, default=0.95
        Absolute correlation with target above which a feature is flagged.
    mi_threshold : float, default=0.95
        Normalized mutual information threshold for leakage.
    redundancy_threshold : float, default=0.98
        Pairwise correlation above which features are flagged as redundant.
    vif_threshold : float, default=10.0
        VIF above which features are flagged for multicollinearity.
    variance_low_threshold : float, default=1e-10
        Variance below which features are flagged as near-zero variance.
    adversarial_threshold : float, default=0.7
        AUC above which train/test drift is flagged.
    suspect_patterns : list of str, optional
        Regex/glob patterns for suspect feature names (e.g. ``["cohort_*"]``).
    future_features : list of str, optional
        Feature names known to contain future information.
    time_budget : float, optional
        Time budget in seconds. Expensive checks are skipped if exceeded.
    random_state : int, optional
        Random seed for reproducible MI computation.
    verbose : bool, default=False
        Whether to print progress.

    Examples
    --------
    >>> from endgame.guardrails import LeakageDetector
    >>> detector = LeakageDetector(mode="detect")
    >>> detector.fit(X_train, y_train)
    >>> print(detector.report_)

    >>> # Fix mode — drop flagged features
    >>> detector = LeakageDetector(mode="fix")
    >>> X_clean = detector.fit_transform(X_train, y_train)
    """

    def __init__(
        self,
        mode: str = "detect",
        checks: str | list[str] = "default",
        leakage_threshold: float = 0.95,
        mi_threshold: float = 0.95,
        redundancy_threshold: float = 0.98,
        vif_threshold: float = 10.0,
        variance_low_threshold: float = 1e-10,
        adversarial_threshold: float = 0.7,
        suspect_patterns: list[str] | None = None,
        future_features: list[str] | None = None,
        time_budget: float | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.mode = mode
        self.checks = checks
        self.leakage_threshold = leakage_threshold
        self.mi_threshold = mi_threshold
        self.redundancy_threshold = redundancy_threshold
        self.vif_threshold = vif_threshold
        self.variance_low_threshold = variance_low_threshold
        self.adversarial_threshold = adversarial_threshold
        self.suspect_patterns = suspect_patterns
        self.future_features = future_features
        self.time_budget = time_budget
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any = None, *, X_test: Any = None) -> LeakageDetector:
        """Run all configured checks on the data.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like, optional
            Target variable.
        X_test : array-like, optional
            Test features for train/test checks.

        Returns
        -------
        self
        """
        start = time.time()
        report = GuardrailsReport()

        X_numeric, numeric_names, X_cat, cat_names, y_arr = normalize_input(X, y)

        # Normalize X_test if provided
        X_test_numeric = None
        if X_test is not None:
            X_test_numeric, _, _, _, _ = normalize_input(X_test)

        # Determine which checks to run
        checks_to_run = self._resolve_checks()

        # Sort by cost tier: cheap → moderate → expensive
        checks_to_run.sort(key=lambda name: COST_ORDER.get(
            CHECK_REGISTRY[name][2], 99
        ))

        # Build kwargs for check functions
        check_kwargs: dict[str, Any] = {
            "X_numeric": X_numeric,
            "numeric_names": numeric_names,
            "X_cat": X_cat,
            "cat_names": cat_names,
            "y_arr": y_arr,
            "X_original": X,
            "X_test_numeric": X_test_numeric,
            "report": report,
            "leakage_threshold": self.leakage_threshold,
            "mi_threshold": self.mi_threshold,
            "redundancy_threshold": self.redundancy_threshold,
            "vif_threshold": self.vif_threshold,
            "variance_low_threshold": self.variance_low_threshold,
            "adversarial_threshold": self.adversarial_threshold,
            "suspect_patterns": self.suspect_patterns,
            "future_features": self.future_features,
            "random_state": self.random_state,
        }

        # Execute checks in order
        for check_name in checks_to_run:
            if check_time_budget(start, self.time_budget, 0.95):
                if self.verbose:
                    logger.info(f"Time budget exceeded, skipping remaining checks")
                break

            check_fn = CHECK_REGISTRY[check_name][0]
            cost = CHECK_REGISTRY[check_name][2]

            # Skip expensive checks if time is running low
            if cost == "expensive" and check_time_budget(start, self.time_budget, 0.7):
                if self.verbose:
                    logger.info(f"Skipping expensive check '{check_name}' due to time budget")
                continue

            if self.verbose:
                logger.info(f"Running check: {check_name}")

            try:
                check_fn(**check_kwargs)
            except Exception as e:
                logger.warning(f"Check '{check_name}' failed: {e}")

        # Deduplicate features_to_drop
        seen: set[str] = set()
        deduped: list[str] = []
        for f in report.features_to_drop:
            if f not in seen:
                seen.add(f)
                deduped.append(f)
        report.features_to_drop = deduped

        # Log results
        if self.verbose:
            for w in report.warnings:
                if w.severity == "critical":
                    logger.warning(f"[Guardrails CRITICAL] {w.message}")
                elif w.severity == "warning":
                    logger.warning(f"[Guardrails] {w.message}")
                else:
                    logger.debug(f"[Guardrails] {w.message}")

        self.report_ = report

        # Store column info for transform
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = numeric_names + cat_names

        self.features_to_drop_ = report.features_to_drop
        self.columns_to_keep_ = [
            c for c in self.feature_names_in_ if c not in set(self.features_to_drop_)
        ]

        return self

    def transform(self, X: Any) -> Any:
        """Drop flagged features if mode='fix', otherwise pass through.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        Filtered X (same type as input) if mode='fix', otherwise X unchanged.
        """
        if self.mode != "fix" or not self.features_to_drop_:
            return X
        return reconstruct_output(X, self.columns_to_keep_)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return feature names after transformation."""
        if self.mode != "fix":
            return self.feature_names_in_
        return self.columns_to_keep_

    def _resolve_checks(self) -> list[str]:
        """Resolve check specification to list of check names."""
        if isinstance(self.checks, list):
            valid = [c for c in self.checks if c in CHECK_REGISTRY]
            return valid

        if self.checks == "all":
            return list(CHECK_REGISTRY.keys())

        # "default" — all checks that are enabled by default
        return [name for name, (_, enabled, _, _) in CHECK_REGISTRY.items() if enabled]
