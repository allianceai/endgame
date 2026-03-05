from __future__ import annotations

"""Distribution drift detection between reference and test datasets."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from endgame.guardrails.utils import normalize_input


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature.

    Attributes
    ----------
    feature : str
        Feature name.
    method : str
        Statistical test used.
    score : float
        Test statistic or PSI value.
    p_value : float or None
        p-value (None for PSI and adversarial methods).
    is_drifted : bool
        Whether drift was detected at the configured threshold.
    """

    feature: str
    method: str
    score: float
    p_value: float | None
    is_drifted: bool


@dataclass
class DriftReport:
    """Aggregated drift detection results.

    Attributes
    ----------
    feature_results : list[FeatureDriftResult]
        Per-feature drift results.
    overall_drift_score : float
        Mean drift score across features.
    drifted_features : list[str]
        Names of features where drift was detected.
    method : str
        Detection method used.
    adversarial_auc : float or None
        AUC from adversarial validation, if that method was used.
    """

    feature_results: list[FeatureDriftResult]
    overall_drift_score: float
    drifted_features: list[str]
    method: str
    adversarial_auc: float | None = None

    def summary(self) -> str:
        n = len(self.drifted_features)
        total = len(self.feature_results)
        lines = [
            f"DriftReport ({self.method}): {n}/{total} features drifted "
            f"(overall score={self.overall_drift_score:.4f})",
        ]
        if self.adversarial_auc is not None:
            lines.append(f"  Adversarial AUC: {self.adversarial_auc:.4f}")
        if self.drifted_features:
            lines.append(f"  Drifted: {', '.join(self.drifted_features)}")
        return "\n".join(lines)

    def to_dataframe(self):
        """Return feature results as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for r in self.feature_results:
            rows.append({
                "feature": r.feature,
                "method": r.method,
                "score": r.score,
                "p_value": r.p_value,
                "is_drifted": r.is_drifted,
            })
        return pd.DataFrame(rows)

    def _repr_html_(self) -> str:
        n = len(self.drifted_features)
        total = len(self.feature_results)
        rows_html = ""
        for r in self.feature_results:
            color = "red" if r.is_drifted else "green"
            p_str = f"{r.p_value:.4f}" if r.p_value is not None else "—"
            rows_html += (
                f"<tr><td>{r.feature}</td><td>{r.method}</td>"
                f"<td>{r.score:.4f}</td><td>{p_str}</td>"
                f'<td style="color:{color}">{"Yes" if r.is_drifted else "No"}</td></tr>'
            )
        return (
            f"<div><strong>DriftReport</strong> ({self.method}): "
            f"{n}/{total} features drifted</div>"
            "<table><tr><th>Feature</th><th>Method</th><th>Score</th>"
            "<th>p-value</th><th>Drifted</th></tr>"
            f"{rows_html}</table>"
        )

    def __repr__(self) -> str:
        return self.summary()


class DriftDetector:
    """Detect distribution drift between a reference and test dataset.

    Parameters
    ----------
    method : str
        Detection method: ``'auto'``, ``'psi'``, ``'ks'``, ``'chi2'``, or
        ``'adversarial'``.
    psi_threshold : float
        PSI threshold above which drift is flagged.
    ks_alpha : float
        Significance level for the Kolmogorov-Smirnov test.
    chi2_alpha : float
        Significance level for the chi-squared test.
    adversarial_threshold : float
        AUC threshold for the adversarial method.
    n_bins : int
        Number of bins for PSI computation.

    Examples
    --------
    >>> detector = DriftDetector(method='ks')
    >>> report = detector.detect(X_train, X_test)
    >>> print(report.summary())
    """

    def __init__(
        self,
        method: str = "auto",
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        chi2_alpha: float = 0.05,
        adversarial_threshold: float = 0.7,
        n_bins: int = 10,
    ):
        self.method = method
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.chi2_alpha = chi2_alpha
        self.adversarial_threshold = adversarial_threshold
        self.n_bins = n_bins

    def detect(self, X_reference: Any, X_test: Any) -> DriftReport:
        """Run drift detection.

        Parameters
        ----------
        X_reference : array-like
            Reference (training) data.
        X_test : array-like
            Test (production) data.

        Returns
        -------
        DriftReport
        """
        if self.method == "adversarial":
            return self._detect_adversarial(X_reference, X_test)

        ref_num, ref_num_names, ref_cat, ref_cat_names, _ = normalize_input(X_reference)
        test_num, test_num_names, test_cat, test_cat_names, _ = normalize_input(X_test)

        results: list[FeatureDriftResult] = []

        method = self.method

        # Numeric features
        for i, name in enumerate(ref_num_names):
            ref_col = ref_num[:, i]
            test_col = test_num[:, i]
            # Remove NaN
            ref_valid = ref_col[~np.isnan(ref_col)]
            test_valid = test_col[~np.isnan(test_col)]

            if method in ("auto", "ks"):
                results.append(self._ks_test(name, ref_valid, test_valid))
            elif method == "psi":
                results.append(self._psi_test(name, ref_valid, test_valid))
            elif method == "chi2":
                # Skip numeric for chi2
                continue
            else:
                raise ValueError(f"Unknown method: {method!r}")

        # Categorical features
        if ref_cat is not None and test_cat is not None:
            for i, name in enumerate(ref_cat_names):
                ref_col = ref_cat[:, i]
                test_col = test_cat[:, i]

                if method in ("auto", "chi2"):
                    results.append(self._chi2_test(name, ref_col, test_col))
                elif method == "psi":
                    # PSI on categoricals: treat as label-encoded
                    results.append(self._chi2_test(name, ref_col, test_col))
                elif method == "ks":
                    # Skip categoricals for KS
                    continue

        drifted = [r.feature for r in results if r.is_drifted]
        scores = [r.score for r in results]
        overall = float(np.mean(scores)) if scores else 0.0

        return DriftReport(
            feature_results=results,
            overall_drift_score=overall,
            drifted_features=drifted,
            method=method,
        )

    # ------------------------------------------------------------------
    # Private test methods
    # ------------------------------------------------------------------

    def _ks_test(self, name: str, ref: np.ndarray, test: np.ndarray) -> FeatureDriftResult:
        if len(ref) == 0 or len(test) == 0:
            return FeatureDriftResult(name, "ks", 0.0, 1.0, False)
        stat, p = sp_stats.ks_2samp(ref, test)
        return FeatureDriftResult(name, "ks", float(stat), float(p), p < self.ks_alpha)

    def _psi_test(self, name: str, ref: np.ndarray, test: np.ndarray) -> FeatureDriftResult:
        if len(ref) == 0 or len(test) == 0:
            return FeatureDriftResult(name, "psi", 0.0, None, False)

        # Equal-frequency binning from reference
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(ref, percentiles)
        bin_edges = np.unique(bin_edges)

        ref_counts = np.histogram(ref, bins=bin_edges)[0].astype(float)
        test_counts = np.histogram(test, bins=bin_edges)[0].astype(float)

        # Normalize to proportions with small epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = ref_counts / ref_counts.sum() + eps
        test_pct = test_counts / test_counts.sum() + eps

        psi = float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))
        return FeatureDriftResult(name, "psi", psi, None, psi > self.psi_threshold)

    def _chi2_test(self, name: str, ref: np.ndarray, test: np.ndarray) -> FeatureDriftResult:
        # Build contingency table from category counts
        ref_clean = [str(v) for v in ref if v is not None and str(v) != "nan"]
        test_clean = [str(v) for v in test if v is not None and str(v) != "nan"]

        if not ref_clean or not test_clean:
            return FeatureDriftResult(name, "chi2", 0.0, 1.0, False)

        from collections import Counter

        ref_counts = Counter(ref_clean)
        test_counts = Counter(test_clean)
        all_cats = sorted(set(ref_counts) | set(test_counts))

        observed = np.array(
            [[ref_counts.get(c, 0) for c in all_cats],
             [test_counts.get(c, 0) for c in all_cats]]
        )

        # Remove columns that are all zero
        col_sums = observed.sum(axis=0)
        observed = observed[:, col_sums > 0]

        if observed.shape[1] < 2:
            return FeatureDriftResult(name, "chi2", 0.0, 1.0, False)

        stat, p, _, _ = sp_stats.chi2_contingency(observed)
        return FeatureDriftResult(name, "chi2", float(stat), float(p), p < self.chi2_alpha)

    def _detect_adversarial(self, X_reference: Any, X_test: Any) -> DriftReport:
        from endgame.validation.adversarial import AdversarialValidator

        av = AdversarialValidator()
        result = av.check_drift(X_reference, X_test)

        auc = result.auc_score
        is_drifted = auc >= self.adversarial_threshold

        # Build per-feature results from importances
        feature_results = []
        for feat in result.drifted_features:
            imp = result.feature_importances.get(feat, 0.0)
            feature_results.append(
                FeatureDriftResult(feat, "adversarial", imp, None, is_drifted)
            )

        drifted = [r.feature for r in feature_results] if is_drifted else []

        return DriftReport(
            feature_results=feature_results,
            overall_drift_score=auc,
            drifted_features=drifted,
            method="adversarial",
            adversarial_auc=auc,
        )
