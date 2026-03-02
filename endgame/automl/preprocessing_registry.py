from __future__ import annotations

"""Preprocessing component registry for AutoML.

This module provides a centralized registry of feature selection and
dimensionality reduction methods with metadata about their capabilities,
computational costs, and recommended usage scenarios.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionInfo:
    """Information about a feature selection method.

    Attributes
    ----------
    name : str
        Short name (used as key).
    display_name : str
        Human-readable name.
    category : str
        Category: "filter", "wrapper", "importance", "advanced".
    class_path : str
        Full import path for the class.
    supports_regression : bool
        Whether it supports regression tasks.
    supports_classification : bool
        Whether it supports classification tasks.
    requires_fitted_model : bool
        Whether it needs a pre-fitted model.
    typical_fit_time : str
        Expected fit time: "fast", "medium", "slow", "very_slow".
    handles_high_dim : bool
        Whether it handles high-dimensional data well.
    statistical_guarantees : bool
        Whether it provides statistical guarantees (FDR control, etc.).
    handles_interactions : bool
        Whether it captures feature interactions.
    handles_redundancy : bool
        Whether it handles redundant features.
    default_params : dict
        Default hyperparameters.
    notes : str
        Additional notes.
    """

    name: str
    display_name: str
    category: str
    class_path: str
    supports_regression: bool = True
    supports_classification: bool = True
    requires_fitted_model: bool = False
    typical_fit_time: str = "medium"
    handles_high_dim: bool = True
    statistical_guarantees: bool = False
    handles_interactions: bool = False
    handles_redundancy: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DimensionalityReductionInfo:
    """Information about a dimensionality reduction method.

    Attributes
    ----------
    name : str
        Short name (used as key).
    display_name : str
        Human-readable name.
    category : str
        Category: "linear", "manifold", "deep".
    class_path : str
        Full import path for the class.
    preserves_global : bool
        Whether it preserves global structure.
    preserves_local : bool
        Whether it preserves local structure.
    supports_transform : bool
        Whether it supports transform() on new data.
    supports_inverse : bool
        Whether it supports inverse_transform().
    requires_torch : bool
        Whether it requires PyTorch.
    typical_fit_time : str
        Expected fit time.
    memory_usage : str
        Expected memory usage.
    best_for_visualization : bool
        Whether it's particularly good for 2D/3D visualization.
    default_params : dict
        Default hyperparameters.
    notes : str
        Additional notes.
    """

    name: str
    display_name: str
    category: str
    class_path: str
    preserves_global: bool = True
    preserves_local: bool = True
    supports_transform: bool = True
    supports_inverse: bool = False
    requires_torch: bool = False
    typical_fit_time: str = "medium"
    memory_usage: str = "medium"
    best_for_visualization: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# =============================================================================
# Feature Selection Registry
# =============================================================================

FEATURE_SELECTION_REGISTRY: dict[str, FeatureSelectionInfo] = {
    # ==================== Filter Methods ====================
    "univariate": FeatureSelectionInfo(
        name="univariate",
        display_name="Univariate Statistical",
        category="filter",
        class_path="endgame.feature_selection.UnivariateSelector",
        typical_fit_time="fast",
        handles_interactions=False,
        handles_redundancy=False,
        default_params={"score_func": "mutual_info_classif", "k": 50},
        notes="F-test, mutual information, chi-squared. Fast baseline.",
    ),
    "mutual_info": FeatureSelectionInfo(
        name="mutual_info",
        display_name="Mutual Information",
        category="filter",
        class_path="endgame.feature_selection.MutualInfoSelector",
        typical_fit_time="fast",
        handles_interactions=False,
        handles_redundancy=False,
        default_params={"k": 50},
        notes="Captures nonlinear relationships. Better than correlation.",
    ),
    "f_test": FeatureSelectionInfo(
        name="f_test",
        display_name="F-Test",
        category="filter",
        class_path="endgame.feature_selection.FTestSelector",
        typical_fit_time="fast",
        handles_interactions=False,
        handles_redundancy=False,
        default_params={"k": 50},
        notes="ANOVA F-test. Very fast linear baseline.",
    ),
    "chi2": FeatureSelectionInfo(
        name="chi2",
        display_name="Chi-Squared",
        category="filter",
        class_path="endgame.feature_selection.Chi2Selector",
        supports_regression=False,
        typical_fit_time="fast",
        handles_interactions=False,
        handles_redundancy=False,
        default_params={"k": 50},
        notes="For categorical features. Requires non-negative values.",
    ),
    "mrmr": FeatureSelectionInfo(
        name="mrmr",
        display_name="MRMR",
        category="filter",
        class_path="endgame.feature_selection.MRMRSelector",
        typical_fit_time="fast",
        handles_interactions=False,
        handles_redundancy=True,
        default_params={"n_features": 50},
        notes="Minimum Redundancy Maximum Relevance. Handles redundancy well.",
    ),
    "relieff": FeatureSelectionInfo(
        name="relieff",
        display_name="ReliefF",
        category="filter",
        class_path="endgame.feature_selection.ReliefFSelector",
        typical_fit_time="medium",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_features": 50, "n_neighbors": 10},
        notes="Instance-based. Naturally handles feature interactions.",
    ),
    "correlation": FeatureSelectionInfo(
        name="correlation",
        display_name="Correlation Filter",
        category="filter",
        class_path="endgame.feature_selection.CorrelationSelector",
        typical_fit_time="fast",
        handles_interactions=False,
        handles_redundancy=True,
        default_params={"threshold": 0.95},
        notes="Removes highly correlated features. Simple and effective.",
    ),

    # ==================== Wrapper Methods ====================
    "rfe": FeatureSelectionInfo(
        name="rfe",
        display_name="RFE",
        category="wrapper",
        class_path="endgame.feature_selection.RFESelector",
        typical_fit_time="slow",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_features": 50},
        notes="Recursive Feature Elimination. Classic wrapper method.",
    ),
    "boruta": FeatureSelectionInfo(
        name="boruta",
        display_name="Boruta",
        category="wrapper",
        class_path="endgame.feature_selection.BorutaSelector",
        typical_fit_time="slow",
        statistical_guarantees=True,
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"max_iter": 100},
        notes="Gold standard. Shadow feature comparison. Statistically principled.",
    ),
    "sequential": FeatureSelectionInfo(
        name="sequential",
        display_name="Sequential Selection",
        category="wrapper",
        class_path="endgame.feature_selection.SequentialSelector",
        typical_fit_time="slow",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_features": 50, "direction": "forward"},
        notes="Forward/backward/bidirectional selection.",
    ),
    "genetic": FeatureSelectionInfo(
        name="genetic",
        display_name="Genetic Algorithm",
        category="wrapper",
        class_path="endgame.feature_selection.GeneticSelector",
        typical_fit_time="very_slow",
        handles_interactions=True,
        handles_redundancy=True,
        default_params={"population_size": 50, "n_generations": 100},
        notes="Evolutionary feature selection. Good for large search spaces.",
    ),

    # ==================== Importance Methods ====================
    "permutation": FeatureSelectionInfo(
        name="permutation",
        display_name="Permutation Importance",
        category="importance",
        class_path="endgame.feature_selection.PermutationSelector",
        requires_fitted_model=True,
        typical_fit_time="medium",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_features": 50, "n_repeats": 10},
        notes="Model-agnostic. Measures actual predictive contribution.",
    ),
    "shap": FeatureSelectionInfo(
        name="shap",
        display_name="SHAP",
        category="importance",
        class_path="endgame.feature_selection.SHAPSelector",
        requires_fitted_model=True,
        typical_fit_time="slow",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_features": 50},
        notes="Theoretically grounded. Mean absolute SHAP values.",
    ),
    "tree_importance": FeatureSelectionInfo(
        name="tree_importance",
        display_name="Tree Importance",
        category="importance",
        class_path="endgame.feature_selection.TreeImportanceSelector",
        typical_fit_time="fast",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_features": "mean"},
        notes="Gini/entropy importance. Fast but biased to high-cardinality.",
    ),

    # ==================== Advanced Methods ====================
    "stability": FeatureSelectionInfo(
        name="stability",
        display_name="Stability Selection",
        category="advanced",
        class_path="endgame.feature_selection.StabilitySelector",
        typical_fit_time="very_slow",
        statistical_guarantees=True,
        handles_interactions=True,  # Depends on base selector
        handles_redundancy=True,  # Depends on base selector
        default_params={"n_bootstrap": 100, "threshold": 0.6},
        notes="Wrapper around any selector. Makes selection more stable.",
    ),
    "knockoff": FeatureSelectionInfo(
        name="knockoff",
        display_name="Knockoff Filter",
        category="advanced",
        class_path="endgame.feature_selection.KnockoffSelector",
        typical_fit_time="medium",
        statistical_guarantees=True,
        handles_interactions=False,
        handles_redundancy=True,
        default_params={"fdr": 0.1},
        notes="FDR control. Rigorous statistical guarantees.",
    ),
    "null_importance": FeatureSelectionInfo(
        name="null_importance",
        display_name="Null Importance",
        category="advanced",
        class_path="endgame.preprocessing.selection.NullImportanceSelector",
        typical_fit_time="slow",
        statistical_guarantees=True,
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"n_iterations": 100},
        notes="Features must beat shuffled-target baseline.",
    ),
    "adversarial": FeatureSelectionInfo(
        name="adversarial",
        display_name="Adversarial Selection",
        category="advanced",
        class_path="endgame.preprocessing.selection.AdversarialFeatureSelector",
        typical_fit_time="medium",
        handles_interactions=True,
        handles_redundancy=False,
        default_params={"threshold": 0.05},
        notes="Removes features that cause train/test drift.",
    ),
}


# =============================================================================
# Dimensionality Reduction Registry
# =============================================================================

DIMENSIONALITY_REDUCTION_REGISTRY: dict[str, DimensionalityReductionInfo] = {
    # ==================== Linear Methods ====================
    "pca": DimensionalityReductionInfo(
        name="pca",
        display_name="PCA",
        category="linear",
        class_path="endgame.dimensionality_reduction.PCAReducer",
        preserves_global=True,
        preserves_local=False,
        supports_transform=True,
        supports_inverse=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"n_components": 50},
        notes="Principal Component Analysis. Gold standard for linear reduction.",
    ),
    "randomized_pca": DimensionalityReductionInfo(
        name="randomized_pca",
        display_name="Randomized PCA",
        category="linear",
        class_path="endgame.dimensionality_reduction.RandomizedPCA",
        preserves_global=True,
        preserves_local=False,
        supports_transform=True,
        supports_inverse=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"n_components": 50},
        notes="Faster PCA for large datasets using randomized SVD.",
    ),
    "truncated_svd": DimensionalityReductionInfo(
        name="truncated_svd",
        display_name="Truncated SVD",
        category="linear",
        class_path="endgame.dimensionality_reduction.TruncatedSVDReducer",
        preserves_global=True,
        preserves_local=False,
        supports_transform=True,
        supports_inverse=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"n_components": 50},
        notes="Works with sparse matrices. Good for text (LSA).",
    ),
    "kernel_pca": DimensionalityReductionInfo(
        name="kernel_pca",
        display_name="Kernel PCA",
        category="linear",
        class_path="endgame.dimensionality_reduction.KernelPCAReducer",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,
        supports_inverse=False,
        typical_fit_time="medium",
        memory_usage="high",
        default_params={"n_components": 50, "kernel": "rbf"},
        notes="Nonlinear projections via kernel trick.",
    ),
    "ica": DimensionalityReductionInfo(
        name="ica",
        display_name="ICA",
        category="linear",
        class_path="endgame.dimensionality_reduction.ICAReducer",
        preserves_global=True,
        preserves_local=False,
        supports_transform=True,
        supports_inverse=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={"n_components": 50},
        notes="Independent Component Analysis. For non-Gaussian sources.",
    ),

    # ==================== Manifold Methods ====================
    "umap": DimensionalityReductionInfo(
        name="umap",
        display_name="UMAP",
        category="manifold",
        class_path="endgame.dimensionality_reduction.UMAPReducer",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,
        supports_inverse=True,
        typical_fit_time="medium",
        memory_usage="medium",
        best_for_visualization=True,
        default_params={"n_components": 2, "n_neighbors": 15},
        notes="Uniform Manifold Approximation. Better than t-SNE, faster.",
    ),
    "parametric_umap": DimensionalityReductionInfo(
        name="parametric_umap",
        display_name="Parametric UMAP",
        category="manifold",
        class_path="endgame.dimensionality_reduction.ParametricUMAP",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,
        supports_inverse=True,
        requires_torch=True,
        typical_fit_time="slow",
        memory_usage="high",
        default_params={"n_components": 2},
        notes="Neural network-based UMAP. Fast transform on new data.",
    ),
    "trimap": DimensionalityReductionInfo(
        name="trimap",
        display_name="TriMAP",
        category="manifold",
        class_path="endgame.dimensionality_reduction.TriMAPReducer",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,  # Approximation via NN
        supports_inverse=False,
        typical_fit_time="medium",
        memory_usage="medium",
        best_for_visualization=True,
        default_params={"n_components": 2},
        notes="Triplet constraints. Better global structure than t-SNE/UMAP.",
    ),
    "phate": DimensionalityReductionInfo(
        name="phate",
        display_name="PHATE",
        category="manifold",
        class_path="endgame.dimensionality_reduction.PHATEReducer",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,
        supports_inverse=False,
        typical_fit_time="medium",
        memory_usage="medium",
        best_for_visualization=True,
        default_params={"n_components": 2, "knn": 5},
        notes="Diffusion-based. Great for trajectory visualization.",
    ),
    "pacmap": DimensionalityReductionInfo(
        name="pacmap",
        display_name="PaCMAP",
        category="manifold",
        class_path="endgame.dimensionality_reduction.PaCMAPReducer",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,
        supports_inverse=False,
        typical_fit_time="fast",
        memory_usage="medium",
        best_for_visualization=True,
        default_params={"n_components": 2},
        notes="Pairwise controlled. Faster with competitive quality.",
    ),

    # ==================== Deep Learning Methods ====================
    "vae": DimensionalityReductionInfo(
        name="vae",
        display_name="VAE",
        category="deep",
        class_path="endgame.dimensionality_reduction.VAEReducer",
        preserves_global=True,
        preserves_local=True,
        supports_transform=True,
        supports_inverse=True,
        requires_torch=True,
        typical_fit_time="slow",
        memory_usage="high",
        default_params={"n_components": 10},
        notes="Variational Autoencoder. Smooth latent space, generative.",
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_feature_selection_info(name: str) -> FeatureSelectionInfo:
    """Get information about a feature selection method."""
    if name not in FEATURE_SELECTION_REGISTRY:
        available = ", ".join(sorted(FEATURE_SELECTION_REGISTRY.keys()))
        raise KeyError(f"Feature selector '{name}' not found. Available: {available}")
    return FEATURE_SELECTION_REGISTRY[name]


def get_dimensionality_reduction_info(name: str) -> DimensionalityReductionInfo:
    """Get information about a dimensionality reduction method."""
    if name not in DIMENSIONALITY_REDUCTION_REGISTRY:
        available = ", ".join(sorted(DIMENSIONALITY_REDUCTION_REGISTRY.keys()))
        raise KeyError(f"Reducer '{name}' not found. Available: {available}")
    return DIMENSIONALITY_REDUCTION_REGISTRY[name]


def list_feature_selectors(
    category: str | None = None,
    handles_interactions: bool | None = None,
    handles_redundancy: bool | None = None,
    statistical_guarantees: bool | None = None,
    max_fit_time: str | None = None,
) -> list[str]:
    """List feature selection methods matching criteria.

    Parameters
    ----------
    category : str, optional
        Filter by category: "filter", "wrapper", "importance", "advanced".
    handles_interactions : bool, optional
        Filter by interaction handling.
    handles_redundancy : bool, optional
        Filter by redundancy handling.
    statistical_guarantees : bool, optional
        Filter by statistical guarantees.
    max_fit_time : str, optional
        Maximum fit time: "fast", "medium", "slow", "very_slow".

    Returns
    -------
    list of str
        Names of matching selectors.
    """
    time_order = {"fast": 0, "medium": 1, "slow": 2, "very_slow": 3}

    results = []
    for name, info in FEATURE_SELECTION_REGISTRY.items():
        if category and info.category != category:
            continue
        if handles_interactions is not None and info.handles_interactions != handles_interactions:
            continue
        if handles_redundancy is not None and info.handles_redundancy != handles_redundancy:
            continue
        if statistical_guarantees is not None and info.statistical_guarantees != statistical_guarantees:
            continue
        if max_fit_time:
            if time_order.get(info.typical_fit_time, 3) > time_order.get(max_fit_time, 3):
                continue
        results.append(name)

    return sorted(results)


def list_dimensionality_reducers(
    category: str | None = None,
    supports_transform: bool | None = None,
    supports_inverse: bool | None = None,
    requires_torch: bool | None = None,
    best_for_visualization: bool | None = None,
) -> list[str]:
    """List dimensionality reduction methods matching criteria.

    Parameters
    ----------
    category : str, optional
        Filter by category: "linear", "manifold", "deep".
    supports_transform : bool, optional
        Filter by transform support.
    supports_inverse : bool, optional
        Filter by inverse transform support.
    requires_torch : bool, optional
        Filter by PyTorch requirement.
    best_for_visualization : bool, optional
        Filter by visualization suitability.

    Returns
    -------
    list of str
        Names of matching reducers.
    """
    results = []
    for name, info in DIMENSIONALITY_REDUCTION_REGISTRY.items():
        if category and info.category != category:
            continue
        if supports_transform is not None and info.supports_transform != supports_transform:
            continue
        if supports_inverse is not None and info.supports_inverse != supports_inverse:
            continue
        if requires_torch is not None and info.requires_torch != requires_torch:
            continue
        if best_for_visualization is not None and info.best_for_visualization != best_for_visualization:
            continue
        results.append(name)

    return sorted(results)


def get_recommended_selectors(
    n_samples: int,
    n_features: int,
    task_type: str = "classification",
    time_budget: str = "medium",
) -> list[str]:
    """Get recommended feature selection methods based on data characteristics.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    task_type : str
        "classification" or "regression".
    time_budget : str
        "fast", "medium", "slow".

    Returns
    -------
    list of str
        Recommended selector names in priority order.
    """
    recommendations = []

    # High-dimensional: prioritize fast methods
    if n_features > 1000:
        recommendations.extend(["mrmr", "correlation", "mutual_info"])
        if time_budget in ("medium", "slow"):
            recommendations.append("tree_importance")
    else:
        # Normal dimension
        recommendations.extend(["mrmr", "relieff"])
        if time_budget in ("medium", "slow"):
            recommendations.extend(["boruta", "permutation"])
        if time_budget == "slow":
            recommendations.extend(["stability", "knockoff"])

    # Small data: avoid complex methods
    if n_samples < 500:
        recommendations = [r for r in recommendations if r not in ["genetic", "stability"]]

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            # Check task compatibility
            info = FEATURE_SELECTION_REGISTRY.get(r)
            if info:
                if task_type == "classification" and not info.supports_classification:
                    continue
                if task_type == "regression" and not info.supports_regression:
                    continue
            result.append(r)

    return result


def get_recommended_reducers(
    n_samples: int,
    n_features: int,
    n_components: int = 2,
    for_visualization: bool = False,
) -> list[str]:
    """Get recommended dimensionality reduction methods.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    n_components : int
        Target dimensions.
    for_visualization : bool
        Whether the goal is visualization.

    Returns
    -------
    list of str
        Recommended reducer names in priority order.
    """
    recommendations = []

    if for_visualization and n_components <= 3:
        # For visualization
        recommendations.extend(["umap", "pacmap", "trimap", "phate"])
    elif n_components > 50:
        # For preprocessing (many components)
        recommendations.extend(["pca", "randomized_pca", "truncated_svd"])
    else:
        # General case
        recommendations.extend(["pca", "umap"])
        if n_samples > 1000:
            recommendations.append("pacmap")

    # Large data: avoid slow methods
    if n_samples > 50000:
        recommendations = [
            r for r in recommendations
            if DIMENSIONALITY_REDUCTION_REGISTRY.get(r, DimensionalityReductionInfo(
                name="", display_name="", category="", class_path=""
            )).typical_fit_time != "slow"
        ]

    return recommendations
