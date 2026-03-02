from __future__ import annotations

"""Calibration analysis and visualization tools.

Provides metrics and diagnostic plots for assessing probability calibration.

References
----------
- Naeini et al. "Obtaining Well Calibrated Probabilities Using Bayesian Binning" (2015)
- Guo et al. "On Calibration of Modern Neural Networks" (2017)
- Nixon et al. "Measuring Calibration in Deep Learning" (2019)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    """Container for calibration analysis results."""

    # Core metrics
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    log_loss: float

    # Brier decomposition
    reliability: float  # Calibration component
    resolution: float  # How much predictions differ from base rate
    uncertainty: float  # Baseline entropy

    # Reliability diagram data
    bin_edges: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray

    # Summary statistics
    mean_confidence: float
    accuracy: float
    overconfidence: float  # Mean(confidence - accuracy)

    def __repr__(self) -> str:
        return (
            f"CalibrationReport(\n"
            f"  ECE={self.ece:.4f},\n"
            f"  MCE={self.mce:.4f},\n"
            f"  Brier={self.brier_score:.4f},\n"
            f"  Log Loss={self.log_loss:.4f},\n"
            f"  Reliability={self.reliability:.4f},\n"
            f"  Resolution={self.resolution:.4f},\n"
            f"  Overconfidence={self.overconfidence:.4f}\n"
            f")"
        )


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    n_bins : int, default=10
        Number of bins.
    strategy : str, default='uniform'
        Binning strategy: 'uniform' or 'quantile'.

    Returns
    -------
    float
        Expected Calibration Error (lower is better, 0 is perfectly calibrated).

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_proba = np.array([0.2, 0.4, 0.6, 0.9])
    >>> ece = expected_calibration_error(y_true, y_proba)
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    if len(y_true) == 0:
        return 0.0

    # Determine bin edges
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    n_samples = len(y_true)

    for i in range(len(bin_edges) - 1):
        # Find samples in this bin
        if i == len(bin_edges) - 2:
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        else:
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])

        n_bin = np.sum(mask)
        if n_bin == 0:
            continue

        # Accuracy and confidence in this bin
        accuracy = np.mean(y_true[mask])
        confidence = np.mean(y_proba[mask])

        # Weighted absolute difference
        ece += (n_bin / n_samples) * np.abs(accuracy - confidence)

    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Maximum Calibration Error (MCE).

    MCE is the maximum calibration error across all bins.
    Useful for identifying worst-case miscalibration.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins.
    strategy : str, default='uniform'
        Binning strategy.

    Returns
    -------
    float
        Maximum Calibration Error.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    if len(y_true) == 0:
        return 0.0

    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)

    mce = 0.0

    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        else:
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])

        n_bin = np.sum(mask)
        if n_bin == 0:
            continue

        accuracy = np.mean(y_true[mask])
        confidence = np.mean(y_proba[mask])

        mce = max(mce, np.abs(accuracy - confidence))

    return mce


def brier_score_decomposition(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    Brier Score = Reliability - Resolution + Uncertainty

    - Reliability: Measures calibration (lower is better)
    - Resolution: Measures how much predictions differ from base rate (higher is better)
    - Uncertainty: Base rate entropy (constant for a dataset)

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for decomposition.

    Returns
    -------
    dict
        Dictionary with 'brier_score', 'reliability', 'resolution', 'uncertainty'.

    Examples
    --------
    >>> decomp = brier_score_decomposition(y_true, y_proba)
    >>> print(f"Reliability: {decomp['reliability']:.4f}")
    >>> print(f"Resolution: {decomp['resolution']:.4f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    n_samples = len(y_true)
    base_rate = np.mean(y_true)

    # Brier score
    brier = np.mean((y_proba - y_true) ** 2)

    # Uncertainty (entropy of base rate)
    uncertainty = base_rate * (1 - base_rate)

    # Bin samples
    bin_edges = np.linspace(0, 1, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        else:
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])

        n_bin = np.sum(mask)
        if n_bin == 0:
            continue

        # Average prediction and accuracy in bin
        avg_pred = np.mean(y_proba[mask])
        avg_true = np.mean(y_true[mask])

        # Reliability contribution
        reliability += (n_bin / n_samples) * (avg_pred - avg_true) ** 2

        # Resolution contribution
        resolution += (n_bin / n_samples) * (avg_true - base_rate) ** 2

    return {
        "brier_score": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    """Compute log loss (cross-entropy).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    float
        Log loss.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    y_proba = np.clip(y_proba, eps, 1 - eps)

    return -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))


class CalibrationAnalyzer:
    """Analyze and visualize model calibration.

    Computes comprehensive calibration metrics and generates diagnostic plots.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for binning-based metrics.
    strategy : str, default='uniform'
        Binning strategy: 'uniform' or 'quantile'.

    Examples
    --------
    >>> analyzer = CalibrationAnalyzer(n_bins=15)
    >>> report = analyzer.analyze(y_true, y_proba)
    >>> print(report)
    >>>
    >>> # Visualize
    >>> analyzer.plot_reliability_diagram(y_true, y_proba)
    >>> analyzer.plot_confidence_histogram(y_proba)
    """

    def __init__(
        self,
        n_bins: int = 10,
        strategy: str = "uniform",
    ):
        self.n_bins = n_bins
        self.strategy = strategy

    def analyze(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> CalibrationReport:
        """Compute comprehensive calibration metrics.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_proba : array-like
            Predicted probabilities for positive class.

        Returns
        -------
        CalibrationReport
            Comprehensive calibration analysis results.
        """
        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba).ravel()

        # Core metrics
        ece = expected_calibration_error(y_true, y_proba, self.n_bins, self.strategy)
        mce = maximum_calibration_error(y_true, y_proba, self.n_bins, self.strategy)
        brier = np.mean((y_proba - y_true) ** 2)
        ll = log_loss(y_true, y_proba)

        # Brier decomposition
        decomp = brier_score_decomposition(y_true, y_proba, self.n_bins)

        # Reliability diagram data
        bin_data = self._compute_bin_data(y_true, y_proba)

        # Summary statistics
        mean_conf = np.mean(y_proba)
        accuracy = np.mean(y_true)

        # Overconfidence: positive means overconfident, negative means underconfident
        overconf = mean_conf - accuracy

        return CalibrationReport(
            ece=ece,
            mce=mce,
            brier_score=brier,
            log_loss=ll,
            reliability=decomp["reliability"],
            resolution=decomp["resolution"],
            uncertainty=decomp["uncertainty"],
            bin_edges=bin_data["edges"],
            bin_accuracies=bin_data["accuracies"],
            bin_confidences=bin_data["confidences"],
            bin_counts=bin_data["counts"],
            mean_confidence=mean_conf,
            accuracy=accuracy,
            overconfidence=overconf,
        )

    def _compute_bin_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute binned data for reliability diagram."""
        if self.strategy == "uniform":
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
        else:
            bin_edges = np.percentile(y_proba, np.linspace(0, 100, self.n_bins + 1))
            bin_edges = np.unique(bin_edges)

        n_bins_actual = len(bin_edges) - 1
        accuracies = np.zeros(n_bins_actual)
        confidences = np.zeros(n_bins_actual)
        counts = np.zeros(n_bins_actual, dtype=int)

        for i in range(n_bins_actual):
            if i == n_bins_actual - 1:
                mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
            else:
                mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])

            counts[i] = np.sum(mask)
            if counts[i] > 0:
                accuracies[i] = np.mean(y_true[mask])
                confidences[i] = np.mean(y_proba[mask])
            else:
                # Use bin midpoint for empty bins
                accuracies[i] = np.nan
                confidences[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

        return {
            "edges": bin_edges,
            "accuracies": accuracies,
            "confidences": confidences,
            "counts": counts,
        }

    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        ax=None,
        show_histogram: bool = True,
        show_ece: bool = True,
        title: str = "Reliability Diagram",
    ):
        """Plot reliability (calibration) diagram.

        A well-calibrated model has points close to the diagonal.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_proba : array-like
            Predicted probabilities.
        ax : matplotlib axes, optional
            Axes to plot on.
        show_histogram : bool, default=True
            Show histogram of predictions at bottom.
        show_ece : bool, default=True
            Show ECE value on plot.
        title : str, default='Reliability Diagram'
            Plot title.

        Returns
        -------
        matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba).ravel()

        # Get bin data
        bin_data = self._compute_bin_data(y_true, y_proba)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

        # Plot bins
        bin_midpoints = (bin_data["edges"][:-1] + bin_data["edges"][1:]) / 2

        # Only plot non-empty bins
        valid_mask = bin_data["counts"] > 0

        ax.bar(
            bin_midpoints[valid_mask],
            bin_data["accuracies"][valid_mask],
            width=1.0 / len(bin_midpoints),
            alpha=0.7,
            edgecolor='black',
            label='Actual accuracy',
        )

        # Gap visualization (miscalibration)
        for i, (conf, acc, count) in enumerate(zip(
            bin_data["confidences"],
            bin_data["accuracies"],
            bin_data["counts"]
        )):
            if count > 0 and not np.isnan(acc):
                color = 'red' if conf > acc else 'blue'
                ax.plot(
                    [conf, conf], [acc, conf],
                    color=color, alpha=0.5, linewidth=2
                )

        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left')

        if show_ece:
            ece = expected_calibration_error(y_true, y_proba, self.n_bins, self.strategy)
            ax.text(
                0.95, 0.05, f'ECE = {ece:.4f}',
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        if show_histogram:
            # Add histogram at bottom
            ax2 = ax.twinx()
            ax2.hist(
                y_proba, bins=self.n_bins, range=(0, 1),
                alpha=0.3, color='gray', edgecolor='gray'
            )
            ax2.set_ylabel('Count', fontsize=10, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_ylim(0, ax2.get_ylim()[1] * 3)  # Make histogram smaller

        plt.tight_layout()
        return ax

    def plot_confidence_histogram(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray | None = None,
        ax=None,
        title: str = "Confidence Distribution",
    ):
        """Plot histogram of prediction confidences.

        Parameters
        ----------
        y_proba : array-like
            Predicted probabilities.
        y_true : array-like, optional
            True labels for coloring by correctness.
        ax : matplotlib axes, optional
            Axes to plot on.
        title : str
            Plot title.

        Returns
        -------
        matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        y_proba = np.asarray(y_proba).ravel()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if y_true is not None:
            y_true = np.asarray(y_true).ravel()
            # Color by correctness
            predictions = (y_proba >= 0.5).astype(int)
            correct = predictions == y_true

            ax.hist(
                y_proba[correct], bins=self.n_bins, range=(0, 1),
                alpha=0.7, label='Correct', color='green', edgecolor='darkgreen'
            )
            ax.hist(
                y_proba[~correct], bins=self.n_bins, range=(0, 1),
                alpha=0.7, label='Incorrect', color='red', edgecolor='darkred'
            )
            ax.legend()
        else:
            ax.hist(
                y_proba, bins=self.n_bins, range=(0, 1),
                alpha=0.7, edgecolor='black'
            )

        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim([0, 1])

        # Add statistics
        mean_conf = np.mean(y_proba)
        max_conf = np.max(y_proba)
        ax.axvline(mean_conf, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')

        plt.tight_layout()
        return ax

    def compare_calibrations(
        self,
        y_true: np.ndarray,
        probas_dict: dict[str, np.ndarray],
        ax=None,
        title: str = "Calibration Comparison",
    ):
        """Compare calibration of multiple models.

        Parameters
        ----------
        y_true : array-like
            True labels.
        probas_dict : dict
            Dictionary mapping model names to predicted probabilities.
        ax : matplotlib axes, optional
            Axes to plot on.
        title : str
            Plot title.

        Returns
        -------
        matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        y_true = np.asarray(y_true).ravel()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)

        colors = plt.cm.Set1(np.linspace(0, 1, len(probas_dict)))

        for (name, y_proba), color in zip(probas_dict.items(), colors):
            y_proba = np.asarray(y_proba).ravel()
            bin_data = self._compute_bin_data(y_true, y_proba)
            ece = expected_calibration_error(y_true, y_proba, self.n_bins)

            valid_mask = bin_data["counts"] > 0
            ax.plot(
                bin_data["confidences"][valid_mask],
                bin_data["accuracies"][valid_mask],
                'o-', color=color, label=f'{name} (ECE={ece:.4f})',
                linewidth=2, markersize=6
            )

        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left')

        plt.tight_layout()
        return ax
