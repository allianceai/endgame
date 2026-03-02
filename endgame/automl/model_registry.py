from __future__ import annotations

"""Model registry for AutoML.

This module provides a centralized registry of all available models
with metadata about their capabilities, computational costs, and
recommended usage scenarios.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model in the registry.

    Attributes
    ----------
    name : str
        Short name for the model (used as key).
    display_name : str
        Human-readable name.
    family : str
        Model family (gbdt, neural, linear, tree, kernel, rules, bayesian, foundation).
    class_path : str
        Full import path for the model class.
    task_types : list of str
        Supported task types ("classification", "regression", "both").
    supports_sample_weight : bool
        Whether the model supports sample_weight in fit().
    supports_feature_importance : bool
        Whether the model provides feature_importances_.
    supports_gpu : bool
        Whether the model can use GPU acceleration.
    requires_torch : bool
        Whether PyTorch is required.
    requires_julia : bool
        Whether Julia is required.
    typical_fit_time : str
        Expected fit time category: "fast", "medium", "slow", "very_slow".
    memory_usage : str
        Expected memory usage: "low", "medium", "high".
    interpretable : bool
        Whether the model is considered interpretable.
    handles_categorical : bool
        Whether the model natively handles categorical features.
    handles_missing : bool
        Whether the model natively handles missing values.
    max_samples : int or None
        Recommended maximum samples (None = no limit).
    min_samples : int
        Recommended minimum samples.
    default_params : dict
        Default hyperparameters for the model.
    tuning_space : str or None
        Name of the predefined tuning space (if any).
    notes : str
        Additional notes about the model.
    """

    name: str
    display_name: str
    family: str
    class_path: str
    task_types: list[str] = field(default_factory=lambda: ["classification", "regression"])
    supports_sample_weight: bool = True
    supports_feature_importance: bool = True
    supports_gpu: bool = False
    requires_torch: bool = False
    requires_julia: bool = False
    typical_fit_time: str = "medium"
    memory_usage: str = "medium"
    interpretable: bool = False
    handles_categorical: bool = False
    handles_missing: bool = False
    max_samples: int | None = None
    min_samples: int = 10
    default_params: dict[str, Any] = field(default_factory=dict)
    tuning_space: str | None = None
    required_packages: list[str] = field(default_factory=list)
    notes: str = ""


# Define model families
MODEL_FAMILIES = {
    "gbdt": "Gradient Boosting Decision Trees",
    "neural": "Neural Networks",
    "linear": "Linear Models",
    "tree": "Tree-Based Models",
    "kernel": "Kernel Methods",
    "rules": "Rule-Based Models",
    "bayesian": "Bayesian Models",
    "foundation": "Foundation Models",
    "ensemble": "Ensemble Methods",
}

# The main model registry
MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ==================== GBDT Models ====================
    "lgbm": ModelInfo(
        name="lgbm",
        display_name="LightGBM",
        family="gbdt",
        class_path="endgame.models.LGBMWrapper",
        supports_gpu=True,
        handles_categorical=True,
        handles_missing=True,
        typical_fit_time="fast",
        memory_usage="medium",
        default_params={"preset": "endgame", "n_estimators": 2000},
        tuning_space="lgbm_standard",
        notes="Default choice for tabular. Fast, accurate, handles missing/categorical.",
    ),
    "xgb": ModelInfo(
        name="xgb",
        display_name="XGBoost",
        family="gbdt",
        class_path="endgame.models.XGBWrapper",
        supports_gpu=True,
        handles_missing=True,
        typical_fit_time="fast",
        memory_usage="medium",
        default_params={"preset": "endgame", "n_estimators": 2000},
        tuning_space="xgb_standard",
        notes="Robust GBDT, slightly different regularization than LightGBM.",
    ),
    "catboost": ModelInfo(
        name="catboost",
        display_name="CatBoost",
        family="gbdt",
        class_path="endgame.models.CatBoostWrapper",
        supports_gpu=True,
        handles_categorical=True,
        handles_missing=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={"preset": "endgame", "iterations": 2000},
        tuning_space="catboost_standard",
        notes="Best native categorical handling. Ordered boosting.",
    ),
    "ngboost": ModelInfo(
        name="ngboost",
        display_name="NGBoost",
        family="gbdt",
        class_path="endgame.models.NGBoostClassifier",  # or NGBoostRegressor
        typical_fit_time="slow",
        memory_usage="high",
        default_params={"n_estimators": 500},
        notes="Natural gradient boosting for probabilistic predictions.",
    ),

    # ==================== Neural Network Models ====================
    "ft_transformer": ModelInfo(
        name="ft_transformer",
        display_name="FT-Transformer",
        family="neural",
        class_path="endgame.models.tabular.FTTransformerClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="slow",
        memory_usage="high",
        min_samples=500,
        default_params={"n_epochs": 100, "batch_size": 256},
        notes="Transformer for tabular. Best deep learning for tabular.",
    ),
    "saint": ModelInfo(
        name="saint",
        display_name="SAINT",
        family="neural",
        class_path="endgame.models.tabular.SAINTClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="slow",
        memory_usage="high",
        min_samples=500,
        default_params={"n_epochs": 100},
        notes="Self-attention + intersample attention transformer.",
    ),
    "tabnet": ModelInfo(
        name="tabnet",
        display_name="TabNet",
        family="neural",
        class_path="endgame.models.neural.TabNetClassifier",
        supports_gpu=True,
        requires_torch=True,
        supports_feature_importance=True,
        typical_fit_time="slow",
        memory_usage="high",
        min_samples=500,
        default_params={"n_steps": 3, "n_d": 32, "n_a": 32},
        required_packages=["pytorch_tabnet"],
        notes="Attention-based with built-in feature selection.",
    ),
    "node": ModelInfo(
        name="node",
        display_name="NODE",
        family="neural",
        class_path="endgame.models.tabular.NODEClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="slow",
        memory_usage="high",
        min_samples=500,
        default_params={},
        notes="Neural Oblivious Decision Ensembles.",
    ),
    "nam": ModelInfo(
        name="nam",
        display_name="Neural Additive Model",
        family="neural",
        class_path="endgame.models.tabular.NAMClassifier",
        supports_gpu=True,
        requires_torch=True,
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        min_samples=200,
        max_samples=20000,
        default_params={"n_epochs": 100},
        notes="Interpretable neural network (GAM-like). Slow on large datasets.",
    ),
    "tabular_resnet": ModelInfo(
        name="tabular_resnet",
        display_name="Tabular ResNet",
        family="neural",
        class_path="endgame.models.tabular.TabularResNetClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=500,
        default_params={},
        notes="ResNet architecture for tabular data.",
    ),
    "mlp": ModelInfo(
        name="mlp",
        display_name="MLP",
        family="neural",
        class_path="endgame.models.neural.MLPClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={"hidden_dims": [256, 128], "dropout": 0.3},
        notes="Standard multi-layer perceptron.",
    ),
    "embedding_mlp": ModelInfo(
        name="embedding_mlp",
        display_name="Embedding MLP",
        family="neural",
        class_path="endgame.models.neural.EmbeddingMLPClassifier",
        supports_gpu=True,
        requires_torch=True,
        handles_categorical=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={},
        notes="MLP with entity embeddings for categoricals.",
    ),

    # ==================== Tree-Based Models ====================
    "rotation_forest": ModelInfo(
        name="rotation_forest",
        display_name="Rotation Forest",
        family="tree",
        class_path="endgame.models.RotationForestClassifier",
        typical_fit_time="slow",
        memory_usage="high",
        max_samples=50000,
        default_params={"n_estimators": 100, "n_subsets": 3},
        notes="PCA rotation for diverse trees. Good for ensemble diversity.",
    ),
    "c50": ModelInfo(
        name="c50",
        display_name="C5.0",
        family="tree",
        class_path="endgame.models.C50Classifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Quinlan's C5.0. Interpretable rules/trees.",
    ),
    "oblique_forest": ModelInfo(
        name="oblique_forest",
        display_name="Oblique Random Forest",
        family="tree",
        class_path="endgame.models.ObliqueRandomForestClassifier",
        typical_fit_time="medium",
        memory_usage="medium",
        max_samples=50000,
        default_params={
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": 2,
        },
        notes=(
            "Oblique splits (linear combos of features) for diagonal decision "
            "boundaries. Uses Cython-compiled treeple when available; falls "
            "back to pure-Python implementation otherwise."
        ),
    ),
    "extra_oblique_forest": ModelInfo(
        name="extra_oblique_forest",
        display_name="Extra Oblique Random Forest",
        family="tree",
        class_path="endgame.models.ExtraObliqueRandomForestClassifier",
        typical_fit_time="fast",
        memory_usage="medium",
        max_samples=100000,
        default_params={
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": 2,
        },
        notes=(
            "Extra-trees variant of oblique RF — maximally random oblique splits. "
            "Fastest oblique forest, excellent ensemble diversity. Requires treeple."
        ),
    ),
    "patch_oblique_forest": ModelInfo(
        name="patch_oblique_forest",
        display_name="Patch Oblique Random Forest",
        family="tree",
        class_path="endgame.models.PatchObliqueRandomForestClassifier",
        typical_fit_time="medium",
        memory_usage="medium",
        max_samples=50000,
        default_params={
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": 2,
        },
        notes=(
            "Oblique splits on contiguous feature patches — effective on spatial "
            "or structured tabular data. Requires treeple."
        ),
    ),
    "honest_forest": ModelInfo(
        name="honest_forest",
        display_name="Honest Forest",
        family="tree",
        class_path="endgame.models.HonestForestClassifier",
        task_types=["classification"],
        typical_fit_time="medium",
        memory_usage="medium",
        max_samples=50000,
        default_params={
            "n_estimators": 100,
            "n_jobs": 2,
            "honest_fraction": 0.5,
        },
        notes=(
            "Honest forests use separate data for tree structure and leaf estimates, "
            "yielding better-calibrated probabilities. Requires treeple."
        ),
    ),
    "evolutionary_tree": ModelInfo(
        name="evolutionary_tree",
        display_name="Evolutionary Tree",
        family="tree",
        class_path="endgame.models.trees.EvolutionaryTreeClassifier",
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        max_samples=10000,
        default_params={"n_generations": 100},
        notes="Genetic algorithm optimized trees.",
    ),
    "quantile_forest": ModelInfo(
        name="quantile_forest",
        display_name="Quantile Regression Forest",
        family="tree",
        class_path="endgame.models.QuantileRegressorForest",
        task_types=["regression"],
        typical_fit_time="medium",
        memory_usage="high",
        default_params={"n_estimators": 100},
        notes="For prediction intervals and uncertainty quantification.",
    ),

    # ==================== Linear Models ====================
    "linear": ModelInfo(
        name="linear",
        display_name="Linear Model",
        family="linear",
        class_path="endgame.models.baselines.LinearClassifier",
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"penalty": "l2", "C": 1.0},
        notes="Fast baseline. L1/L2/ElasticNet regularization.",
    ),
    "elm": ModelInfo(
        name="elm",
        display_name="Extreme Learning Machine",
        family="linear",
        class_path="endgame.models.baselines.ELMClassifier",
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"n_hidden": 500},
        notes="Random projection NN. Very fast, good for ensembles.",
    ),
    "mars": ModelInfo(
        name="mars",
        display_name="MARS",
        family="linear",
        class_path="endgame.models.MARSClassifier",
        interpretable=True,
        typical_fit_time="medium",
        memory_usage="low",
        max_samples=50000,
        default_params={},
        notes="Multivariate Adaptive Regression Splines. Interpretable.",
    ),

    # ==================== Kernel Methods ====================
    "svm": ModelInfo(
        name="svm",
        display_name="SVM",
        family="kernel",
        class_path="endgame.models.kernel.SVMClassifier",
        typical_fit_time="slow",
        memory_usage="high",
        max_samples=10000,
        default_params={"kernel": "rbf", "C": 1.0},
        notes="Support Vector Machine. Slow for large data.",
    ),
    "gp": ModelInfo(
        name="gp",
        display_name="Gaussian Process",
        family="kernel",
        class_path="endgame.models.kernel.GPClassifier",
        typical_fit_time="very_slow",
        memory_usage="high",
        max_samples=5000,
        default_params={},
        notes="Bayesian kernel method. Great for small data + uncertainty.",
    ),

    # ==================== Rule-Based Models ====================
    "rulefit": ModelInfo(
        name="rulefit",
        display_name="RuleFit",
        family="rules",
        class_path="endgame.models.RuleFitClassifier",
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        max_samples=15000,
        default_params={},
        notes="Rule-based ensemble. Interpretable rules. Slow on large datasets due to RF base.",
    ),
    "furia": ModelInfo(
        name="furia",
        display_name="FURIA",
        family="rules",
        class_path="endgame.models.FURIAClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        max_samples=20000,
        default_params={},
        notes="Fuzzy Unordered Rule Induction. Interpretable fuzzy rules. Slow on large datasets.",
    ),

    # ==================== Bayesian Models ====================
    "tan": ModelInfo(
        name="tan",
        display_name="TAN",
        family="bayesian",
        class_path="endgame.models.bayesian.TANClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Tree Augmented Naive Bayes. Fast probabilistic.",
    ),
    "eskdb": ModelInfo(
        name="eskdb",
        display_name="ESKDB",
        family="bayesian",
        class_path="endgame.models.bayesian.ESKDBClassifier",
        task_types=["classification"],
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={},
        notes="Ensemble of Selective K-Dependence Bayes.",
    ),
    "kdb": ModelInfo(
        name="kdb",
        display_name="KDB",
        family="bayesian",
        class_path="endgame.models.bayesian.KDBClassifier",
        task_types=["classification"],
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="K-Dependence Bayes classifier.",
    ),
    "bart": ModelInfo(
        name="bart",
        display_name="BART",
        family="bayesian",
        class_path="endgame.models.probabilistic.BARTClassifier",
        typical_fit_time="very_slow",
        memory_usage="high",
        max_samples=10000,
        default_params={},
        required_packages=["pymc_bart"],
        notes="Bayesian Additive Regression Trees. MCMC-based.",
    ),

    # ==================== Baselines ====================
    "naive_bayes": ModelInfo(
        name="naive_bayes",
        display_name="Naive Bayes",
        family="bayesian",
        class_path="endgame.models.baselines.NaiveBayesClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Auto-selecting Naive Bayes (Gaussian/Bernoulli/Multinomial).",
    ),
    "lda": ModelInfo(
        name="lda",
        display_name="LDA",
        family="linear",
        class_path="endgame.models.baselines.LDAClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Linear Discriminant Analysis.",
    ),
    "qda": ModelInfo(
        name="qda",
        display_name="QDA",
        family="linear",
        class_path="endgame.models.baselines.QDAClassifier",
        task_types=["classification"],
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Quadratic Discriminant Analysis.",
    ),
    "knn": ModelInfo(
        name="knn",
        display_name="KNN",
        family="kernel",
        class_path="endgame.models.baselines.KNNClassifier",
        typical_fit_time="fast",
        memory_usage="high",  # Stores all training data
        max_samples=50000,
        default_params={"n_neighbors": 5},
        notes="K-Nearest Neighbors. Memory scales with data size.",
    ),

    # ==================== Interpretable Models ====================
    "ebm": ModelInfo(
        name="ebm",
        display_name="EBM",
        family="ensemble",
        class_path="endgame.models.EBMClassifier",
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        default_params={"interactions": 10},
        required_packages=["interpret"],
        notes="Explainable Boosting Machine. Glass-box interpretability.",
    ),

    # ==================== Foundation Models ====================
    "tabpfn": ModelInfo(
        name="tabpfn",
        display_name="TabPFN",
        family="foundation",
        class_path="endgame.models.tabular.TabPFNClassifier",
        task_types=["classification"],
        requires_julia=False,
        typical_fit_time="fast",
        memory_usage="high",
        max_samples=10000,
        min_samples=10,
        default_params={},
        required_packages=["tabpfn"],
        notes="In-context learning. Zero training time. Great for small data.",
    ),

    # ==================== Additional Neural Models ====================
    "tab_transformer": ModelInfo(
        name="tab_transformer",
        display_name="Tab Transformer",
        family="neural",
        class_path="endgame.models.tabular.TabTransformerClassifier",
        supports_gpu=True,
        requires_torch=True,
        handles_categorical=True,
        typical_fit_time="slow",
        memory_usage="high",
        min_samples=500,
        default_params={"max_epochs": 100, "batch_size": 256},
        required_packages=["pytorch_tabular"],
        notes="Transformer for tabular with column-wise attention.",
    ),
    "gandalf": ModelInfo(
        name="gandalf",
        display_name="GANDALF",
        family="neural",
        class_path="endgame.models.tabular.GANDALFClassifier",
        supports_gpu=True,
        requires_torch=True,
        supports_feature_importance=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={},
        required_packages=["pytorch_tabular"],
        notes="Gated Adaptive Network for Deep Automated Learning of Features.",
    ),
    "modern_nca": ModelInfo(
        name="modern_nca",
        display_name="Modern NCA",
        family="neural",
        class_path="endgame.models.tabular.ModernNCAClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=100,
        default_params={},
        notes="Modern Neighborhood Component Analysis with neural networks.",
    ),
    # ==================== Additional Bayesian Models ====================
    "neural_kdb": ModelInfo(
        name="neural_kdb",
        display_name="Neural KDB",
        family="bayesian",
        class_path="endgame.models.bayesian.NeuralKDBClassifier",
        task_types=["classification"],
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={},
        notes="Neural K-Dependence Bayes classifier with learned embeddings.",
    ),

    # ==================== Additional Ensemble Models ====================
    "ebmc": ModelInfo(
        name="ebmc",
        display_name="EBM (Classification)",
        family="ensemble",
        class_path="endgame.models.EBMClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        default_params={"interactions": 10},
        notes="Explainable Boosting Machine for classification.",
    ),
    "ebmr": ModelInfo(
        name="ebmr",
        display_name="EBM (Regression)",
        family="ensemble",
        class_path="endgame.models.EBMRegressor",
        task_types=["regression"],
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        default_params={"interactions": 10},
        notes="Explainable Boosting Machine for regression.",
    ),

    # ==================== Interpretable Models ====================
    "corels": ModelInfo(
        name="corels",
        display_name="CORELS",
        family="rules",
        class_path="endgame.models.interpretable.CORELSClassifier",
        task_types=["classification"],
        supports_sample_weight=False,
        supports_feature_importance=False,
        interpretable=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={"max_card": 2, "c": 0.001},
        notes="Rule List Classifier. Greedy sequential covering with beam search.",
    ),
    "node_gam": ModelInfo(
        name="node_gam",
        display_name="NODE-GAM",
        family="neural",
        class_path="endgame.models.interpretable.NodeGAMClassifier",
        supports_gpu=True,
        requires_torch=True,
        interpretable=True,
        typical_fit_time="very_slow",
        memory_usage="high",
        min_samples=100,
        max_samples=10000,
        default_params={"n_trees_per_feature": 32, "depth": 4},
        notes="Neural Oblivious Decision Ensembles as GAMs. Differentiable tree-based GAM. Very slow on large datasets.",
    ),
    "gami_net": ModelInfo(
        name="gami_net",
        display_name="GAMI-Net",
        family="neural",
        class_path="endgame.models.interpretable.GAMINetClassifier",
        supports_gpu=True,
        requires_torch=True,
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        min_samples=100,
        max_samples=20000,
        default_params={"interact_num": 10},
        notes="Generalized Additive Models with Structured Interactions. Slow on large datasets.",
    ),
    "slim": ModelInfo(
        name="slim",
        display_name="SLIM",
        family="linear",
        class_path="endgame.models.interpretable.SLIMClassifier",
        task_types=["classification"],
        supports_sample_weight=False,
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"max_coef": 5},
        required_packages=["fasterrisk"],
        notes="Supersparse Linear Integer Models. Produces scorecards with small integers.",
    ),
    "fasterrisk": ModelInfo(
        name="fasterrisk",
        display_name="FasterRisk",
        family="linear",
        class_path="endgame.models.interpretable.FasterRiskClassifier",
        task_types=["classification"],
        supports_sample_weight=False,
        interpretable=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={"max_coef": 5, "sparsity": 10},
        notes="Fast and accurate risk scores. Optimized SLIM variant.",
    ),
    "gam": ModelInfo(
        name="gam",
        display_name="GAM",
        family="linear",
        class_path="endgame.models.interpretable.GAMClassifier",
        interpretable=True,
        typical_fit_time="medium",
        memory_usage="low",
        default_params={"n_splines": 25},
        required_packages=["pygam"],
        notes="Generalized Additive Models via pyGAM. Smooth shape functions.",
    ),
    "gosdt": ModelInfo(
        name="gosdt",
        display_name="GOSDT",
        family="tree",
        class_path="endgame.models.interpretable.GOSDTClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="slow",
        memory_usage="medium",
        max_samples=10000,
        default_params={"regularization": 0.01, "depth_budget": 5},
        required_packages=["gosdt"],
        notes="Globally Optimal Sparse Decision Trees. Provably optimal trees.",
    ),

    # ==================== Symbolic Regression ====================
    "symbolic_regression": ModelInfo(
        name="symbolic_regression",
        display_name="Symbolic Regression",
        family="rules",
        class_path="endgame.models.symbolic.SymbolicClassifier",
        interpretable=True,
        requires_julia=False,
        typical_fit_time="slow",
        memory_usage="medium",
        max_samples=50000,
        default_params={"preset": "default", "operators": "scientific"},
        notes="GP-based symbolic regression. Discovers interpretable equations.",
    ),
    "symbolic_regressor": ModelInfo(
        name="symbolic_regressor",
        display_name="Symbolic Regressor",
        family="rules",
        class_path="endgame.models.symbolic.SymbolicRegressor",
        task_types=["regression"],
        interpretable=True,
        requires_julia=False,
        typical_fit_time="slow",
        memory_usage="medium",
        max_samples=50000,
        default_params={"preset": "default", "operators": "scientific"},
        notes="GP-based symbolic regression for regression tasks.",
    ),

    # ==================== Additional Tree Models ====================
    "adtree": ModelInfo(
        name="adtree",
        display_name="Alternating Decision Tree",
        family="tree",
        class_path="endgame.models.trees.AlternatingDecisionTreeClassifier",
        task_types=["classification"],
        supports_feature_importance=True,
        interpretable=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={"n_iterations": 10},
        notes="Alternating Decision Tree. Interpretable boosted tree ensemble.",
    ),
    "model_tree": ModelInfo(
        name="model_tree",
        display_name="Alternating Model Tree",
        family="tree",
        class_path="endgame.models.trees.AlternatingModelTreeRegressor",
        task_types=["regression"],
        supports_feature_importance=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={},
        notes="Model trees with linear models at leaves. Piecewise linear regression.",
    ),
    "cubist": ModelInfo(
        name="cubist",
        display_name="Cubist",
        family="tree",
        class_path="endgame.models.CubistRegressor",
        task_types=["regression"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"committees": 1, "neighbors": 0},
        notes="Rule-based regression (Quinlan). Interpretable piecewise linear.",
    ),
    "c50_ensemble": ModelInfo(
        name="c50_ensemble",
        display_name="C5.0 Ensemble",
        family="tree",
        class_path="endgame.models.C50Ensemble",
        task_types=["classification"],
        supports_feature_importance=True,
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"n_trials": 10},
        notes="Boosted C5.0 ensemble. Interpretable boosted trees.",
    ),

    # ==================== Ordinal Regression ====================
    "ordinal": ModelInfo(
        name="ordinal",
        display_name="Ordinal Classifier",
        family="linear",
        class_path="endgame.models.ordinal.OrdinalClassifier",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={"variant": "auto"},
        notes="Auto-selecting ordinal classifier for ordered categorical targets.",
    ),
    "logistic_at": ModelInfo(
        name="logistic_at",
        display_name="Logistic All-Threshold",
        family="linear",
        class_path="endgame.models.ordinal.LogisticAT",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Ordinal regression with all-threshold loss.",
    ),
    "logistic_it": ModelInfo(
        name="logistic_it",
        display_name="Logistic Immediate-Threshold",
        family="linear",
        class_path="endgame.models.ordinal.LogisticIT",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Ordinal regression with immediate-threshold loss.",
    ),
    "logistic_se": ModelInfo(
        name="logistic_se",
        display_name="Logistic Squared-Error",
        family="linear",
        class_path="endgame.models.ordinal.LogisticSE",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Ordinal regression with squared-error loss.",
    ),
    "ordinal_ridge": ModelInfo(
        name="ordinal_ridge",
        display_name="Ordinal Ridge",
        family="linear",
        class_path="endgame.models.ordinal.OrdinalRidge",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Ordinal regression with ridge penalty.",
    ),
    "ordinal_lad": ModelInfo(
        name="ordinal_lad",
        display_name="Ordinal LAD",
        family="linear",
        class_path="endgame.models.ordinal.LAD",
        task_types=["classification"],
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Ordinal regression with least absolute deviation.",
    ),

    # ==================== Additional Bayesian Models ====================
    "auto_sle": ModelInfo(
        name="auto_sle",
        display_name="AutoSLE",
        family="bayesian",
        class_path="endgame.models.bayesian.AutoSLE",
        task_types=[],
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={},
        notes="Structure learning only — not an sklearn estimator, excluded from AutoML.",
    ),
    "ebmc_classifier": ModelInfo(
        name="ebmc_classifier",
        display_name="EBMC",
        family="bayesian",
        class_path="endgame.models.bayesian.EBMCClassifier",
        task_types=["classification"],
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Efficient Bayesian Multivariate Classifier.",
    ),

    # ==================== Modern Neural Models ====================
    "tabm": ModelInfo(
        name="tabm",
        display_name="TabM",
        family="neural",
        class_path="endgame.models.tabular.tabm.TabMClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={},
        notes="TabM: Tabular Model with batch ensembling. Strong SOTA performance.",
    ),
    "realmlp": ModelInfo(
        name="realmlp",
        display_name="RealMLP",
        family="neural",
        class_path="endgame.models.tabular.realmlp.RealMLPClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={},
        notes="RealMLP: Well-tuned MLP baseline with modern training recipe.",
    ),
    "grande": ModelInfo(
        name="grande",
        display_name="GRANDE",
        family="neural",
        class_path="endgame.models.tabular.grande.GRANDEClassifier",
        supports_gpu=True,
        requires_torch=True,
        supports_feature_importance=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={},
        notes="GRANDE: Gradient-based decision tree ensembles. Differentiable trees.",
    ),
    "tabdpt": ModelInfo(
        name="tabdpt",
        display_name="TabDPT",
        family="neural",
        class_path="endgame.models.tabular.tabdpt.TabDPTClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="slow",
        memory_usage="high",
        min_samples=200,
        default_params={},
        required_packages=["tabdpt"],
        notes="TabDPT: Tabular Data Pre-Training. In-context learning for tabular.",
    ),
    "tabr": ModelInfo(
        name="tabr",
        display_name="TabR",
        family="neural",
        class_path="endgame.models.tabular.tabr.TabRClassifier",
        supports_gpu=True,
        requires_torch=True,
        typical_fit_time="medium",
        memory_usage="medium",
        min_samples=200,
        default_params={},
        notes="TabR: Tabular Retrieval-augmented learning. kNN-enhanced deep model.",
    ),

    # ==================== Random Forest Variants ====================
    "rf": ModelInfo(
        name="rf",
        display_name="Random Forest",
        family="tree",
        class_path="sklearn.ensemble.RandomForestClassifier",
        supports_feature_importance=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={"n_estimators": 200, "max_depth": None, "n_jobs": -1},
        notes="Standard sklearn Random Forest. Good baseline ensemble.",
    ),
    "extra_trees": ModelInfo(
        name="extra_trees",
        display_name="Extra Trees",
        family="tree",
        class_path="sklearn.ensemble.ExtraTreesClassifier",
        supports_feature_importance=True,
        typical_fit_time="fast",
        memory_usage="medium",
        default_params={"n_estimators": 200, "n_jobs": -1},
        notes="Extremely Randomized Trees. Faster than RF, different bias.",
    ),

    # ==================== Foundation Models (v2) ====================
    "tabpfn_v2": ModelInfo(
        name="tabpfn_v2",
        display_name="TabPFN v2",
        family="foundation",
        class_path="endgame.models.tabular.tabpfn.TabPFNv2Classifier",
        task_types=["classification"],
        supports_sample_weight=False,
        supports_feature_importance=False,
        handles_categorical=True,
        typical_fit_time="fast",
        memory_usage="high",
        max_samples=10000,
        min_samples=10,
        default_params={},
        notes="TabPFN v2: improved in-context learning. Great for small data.",
    ),
    "tabpfn_25": ModelInfo(
        name="tabpfn_25",
        display_name="TabPFN 2025",
        family="foundation",
        class_path="endgame.models.tabular.tabpfn.TabPFN25Classifier",
        task_types=["classification"],
        supports_sample_weight=False,
        supports_feature_importance=False,
        handles_categorical=True,
        typical_fit_time="fast",
        memory_usage="high",
        max_samples=50000,
        min_samples=10,
        default_params={},
        notes="TabPFN 2025: scales to 50K samples. State-of-the-art ICL.",
    ),

    # ==================== Foundation Models (xRFM) ====================
    "xrfm": ModelInfo(
        name="xrfm",
        display_name="xRFM",
        family="foundation",
        class_path="endgame.models.tabular.xrfm.xRFMClassifier",
        interpretable=True,
        typical_fit_time="medium",
        memory_usage="medium",
        default_params={},
        notes="Explainable Random Feature Model. Interpretable foundation model.",
    ),

    # ==================== Subgroup Discovery ====================
    "prim": ModelInfo(
        name="prim",
        display_name="PRIM",
        family="rules",
        class_path="endgame.models.subgroup.prim.PRIMClassifier",
        task_types=["classification"],
        supports_sample_weight=False,
        interpretable=True,
        typical_fit_time="fast",
        memory_usage="low",
        default_params={},
        notes="Patient Rule Induction Method. Bump hunting for subgroup discovery.",
    ),
}


def register_model(
    name: str,
    display_name: str,
    family: str,
    class_path: str,
    task_types: list[str] | None = None,
    supports_sample_weight: bool = True,
    supports_feature_importance: bool = True,
    supports_gpu: bool = False,
    requires_torch: bool = False,
    requires_julia: bool = False,
    typical_fit_time: str = "medium",
    memory_usage: str = "medium",
    interpretable: bool = False,
    handles_categorical: bool = False,
    handles_missing: bool = False,
    max_samples: int | None = None,
    min_samples: int = 10,
    default_params: dict[str, Any] | None = None,
    tuning_space: str | None = None,
    notes: str = "",
    overwrite: bool = False,
) -> ModelInfo:
    """Register a new model in the registry.

    This function provides a convenient way to add new models to the registry
    at runtime, useful for plugins or custom model extensions.

    Parameters
    ----------
    name : str
        Short name for the model (used as key).
    display_name : str
        Human-readable name.
    family : str
        Model family (gbdt, neural, linear, tree, kernel, rules, bayesian, foundation, ensemble).
    class_path : str
        Full import path for the model class.
    task_types : list of str, optional
        Supported task types. Default is ["classification", "regression"].
    supports_sample_weight : bool, default=True
        Whether the model supports sample_weight in fit().
    supports_feature_importance : bool, default=True
        Whether the model provides feature_importances_.
    supports_gpu : bool, default=False
        Whether the model can use GPU acceleration.
    requires_torch : bool, default=False
        Whether PyTorch is required.
    requires_julia : bool, default=False
        Whether Julia is required.
    typical_fit_time : str, default="medium"
        Expected fit time category: "fast", "medium", "slow", "very_slow".
    memory_usage : str, default="medium"
        Expected memory usage: "low", "medium", "high".
    interpretable : bool, default=False
        Whether the model is considered interpretable.
    handles_categorical : bool, default=False
        Whether the model natively handles categorical features.
    handles_missing : bool, default=False
        Whether the model natively handles missing values.
    max_samples : int, optional
        Recommended maximum samples (None = no limit).
    min_samples : int, default=10
        Recommended minimum samples.
    default_params : dict, optional
        Default hyperparameters for the model.
    tuning_space : str, optional
        Name of the predefined tuning space (if any).
    notes : str, default=""
        Additional notes about the model.
    overwrite : bool, default=False
        If True, overwrite existing entry. If False, raise error if exists.

    Returns
    -------
    ModelInfo
        The registered model info.

    Raises
    ------
    ValueError
        If model already exists and overwrite=False.

    Examples
    --------
    >>> register_model(
    ...     name="my_model",
    ...     display_name="My Custom Model",
    ...     family="neural",
    ...     class_path="mypackage.models.MyModelClassifier",
    ...     supports_gpu=True,
    ...     requires_torch=True,
    ...     typical_fit_time="fast",
    ...     notes="My custom neural network model.",
    ... )
    """
    if name in MODEL_REGISTRY and not overwrite:
        raise ValueError(
            f"Model '{name}' already exists in registry. "
            f"Use overwrite=True to replace it."
        )

    if family not in MODEL_FAMILIES:
        logger.warning(
            f"Family '{family}' not in standard families: {list(MODEL_FAMILIES.keys())}. "
            f"Adding anyway."
        )

    info = ModelInfo(
        name=name,
        display_name=display_name,
        family=family,
        class_path=class_path,
        task_types=task_types or ["classification", "regression"],
        supports_sample_weight=supports_sample_weight,
        supports_feature_importance=supports_feature_importance,
        supports_gpu=supports_gpu,
        requires_torch=requires_torch,
        requires_julia=requires_julia,
        typical_fit_time=typical_fit_time,
        memory_usage=memory_usage,
        interpretable=interpretable,
        handles_categorical=handles_categorical,
        handles_missing=handles_missing,
        max_samples=max_samples,
        min_samples=min_samples,
        default_params=default_params or {},
        tuning_space=tuning_space,
        notes=notes,
    )

    MODEL_REGISTRY[name] = info
    logger.debug(f"Registered model: {name} ({display_name})")

    return info


def unregister_model(name: str) -> bool:
    """Remove a model from the registry.

    Parameters
    ----------
    name : str
        Model name to remove.

    Returns
    -------
    bool
        True if model was removed, False if it didn't exist.
    """
    if name in MODEL_REGISTRY:
        del MODEL_REGISTRY[name]
        logger.debug(f"Unregistered model: {name}")
        return True
    return False


def get_model_info(name: str) -> ModelInfo:
    """Get information about a model.

    Parameters
    ----------
    name : str
        Model name (key in registry).

    Returns
    -------
    ModelInfo
        Information about the model.

    Raises
    ------
    KeyError
        If model is not in registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return MODEL_REGISTRY[name]


def list_models(
    family: str | None = None,
    task_type: str | None = None,
    interpretable_only: bool = False,
    gpu_only: bool = False,
    exclude_slow: bool = False,
    max_samples: int | None = None,
) -> list[str]:
    """List models matching criteria.

    Parameters
    ----------
    family : str, optional
        Filter by model family.
    task_type : str, optional
        Filter by task type ("classification" or "regression").
    interpretable_only : bool, default=False
        Only include interpretable models.
    gpu_only : bool, default=False
        Only include GPU-capable models.
    exclude_slow : bool, default=False
        Exclude slow and very_slow models.
    max_samples : int, optional
        Only include models that can handle this many samples.

    Returns
    -------
    list of str
        Model names matching criteria.
    """
    result = []

    for name, info in MODEL_REGISTRY.items():
        # Family filter
        if family and info.family != family:
            continue

        # Task type filter
        if task_type:
            if task_type not in info.task_types and "both" not in info.task_types:
                continue

        # Interpretable filter
        if interpretable_only and not info.interpretable:
            continue

        # GPU filter
        if gpu_only and not info.supports_gpu:
            continue

        # Speed filter
        if exclude_slow and info.typical_fit_time in ("slow", "very_slow"):
            continue

        # Sample size filter
        if max_samples and info.max_samples and max_samples > info.max_samples:
            continue

        result.append(name)

    return sorted(result)


def get_model_class(name: str) -> type:
    """Get the model class for a given name.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    type
        The model class.

    Raises
    ------
    ImportError
        If the model class cannot be imported.
    """
    info = get_model_info(name)

    # Parse the class path
    parts = info.class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid class path: {info.class_path}")

    module_path, class_name = parts

    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import {info.class_path}: {e}. "
            f"You may need to install additional dependencies."
        ) from e


def instantiate_model(
    name: str,
    task_type: str = "classification",
    **override_params,
) -> Any:
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        Model name.
    task_type : str, default="classification"
        Task type ("classification" or "regression").
        Some models have different classes for different tasks.
    **override_params
        Parameters to override the defaults.

    Returns
    -------
    estimator
        Instantiated model.
    """
    info = get_model_info(name)

    # Get the class (may need to adjust for task type)
    class_path = info.class_path
    if task_type == "regression" and "Classifier" in class_path:
        class_path = class_path.replace("Classifier", "Regressor")

    parts = class_path.rsplit(".", 1)
    module_path, class_name = parts

    try:
        import importlib

        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError):
        # Fallback: try the original class path
        model_class = get_model_class(name)

    # Combine default params with overrides
    params = info.default_params.copy()
    params.update(override_params)

    return model_class(**params)


def get_models_by_family() -> dict[str, list[str]]:
    """Get models grouped by family.

    Returns
    -------
    dict
        Mapping from family name to list of model names.
    """
    result: dict[str, list[str]] = {}

    for name, info in MODEL_REGISTRY.items():
        if info.family not in result:
            result[info.family] = []
        result[info.family].append(name)

    return {k: sorted(v) for k, v in result.items()}


def get_default_portfolio(
    task_type: str = "classification",
    n_samples: int = 10000,
    time_budget: str = "medium",
    gpu_available: bool = False,
) -> list[str]:
    """Get a recommended portfolio of models based on data characteristics.

    Parameters
    ----------
    task_type : str
        Task type.
    n_samples : int
        Number of samples in the dataset.
    time_budget : str
        Time budget: "fast", "medium", "high", "unlimited".
    gpu_available : bool
        Whether GPU is available.

    Returns
    -------
    list of str
        Recommended model names.
    """
    # Start with GBDTs (always include)
    portfolio = ["lgbm", "xgb", "catboost"]

    # Add based on data size
    if n_samples < 5000:
        portfolio.append("tabpfn")
        portfolio.append("gp")

    # Add based on time budget
    if time_budget in ("high", "unlimited"):
        portfolio.extend(["ft_transformer", "saint", "rotation_forest"])
        if task_type == "classification":
            portfolio.extend(["ebm", "ngboost"])

    elif time_budget == "medium":
        portfolio.extend(["linear", "elm"])
        if gpu_available:
            portfolio.append("ft_transformer")

    else:  # fast
        # Just keep LightGBM
        portfolio = ["lgbm"]

    # Filter by task type
    portfolio = [
        m
        for m in portfolio
        if task_type in MODEL_REGISTRY[m].task_types
        or "both" in MODEL_REGISTRY[m].task_types
    ]

    # Filter by sample size
    portfolio = [
        m
        for m in portfolio
        if MODEL_REGISTRY[m].max_samples is None
        or n_samples <= MODEL_REGISTRY[m].max_samples
    ]

    return portfolio


# ==================== Interpretable Models ====================

# Canonical list of interpretable model names
# These are models that provide human-understandable explanations
INTERPRETABLE_MODELS: set[str] = {
    # Rule-based models
    "corels",           # Certifiably Optimal Rule Lists
    "rulefit",          # Rule ensemble with linear combination
    "furia",            # Fuzzy Unordered Rule Induction
    "symbolic_regression",  # Symbolic regression (GP-based)
    "symbolic_regressor",   # Symbolic regression for regression

    # GAM-style models (additive with shape functions)
    "gam",              # Generalized Additive Models (pyGAM)
    "ebm",              # Explainable Boosting Machine
    "ebmc",             # EBM for classification
    "ebmr",             # EBM for regression
    "nam",              # Neural Additive Models
    "node_gam",         # NODE-GAM (differentiable tree-based GAM)
    "gami_net",         # GAMI-Net (GAM with structured interactions)

    # Sparse linear models (scorecards)
    "slim",             # Supersparse Linear Integer Models
    "fasterrisk",       # Fast risk scores
    "linear",           # Linear/logistic regression
    "mars",             # Multivariate Adaptive Regression Splines

    # Interpretable trees
    "gosdt",            # Globally Optimal Sparse Decision Trees
    "c50",              # C5.0 decision trees/rules
    "c50_ensemble",     # Boosted C5.0 ensemble
    "evolutionary_tree", # Evolutionary optimized trees
    "adtree",           # Alternating Decision Tree
    "cubist",           # Cubist rule-based regression

    # Ordinal regression (linear, interpretable)
    "ordinal",          # Auto-selecting ordinal classifier
    "logistic_at",      # Logistic All-Threshold
    "logistic_it",      # Logistic Immediate-Threshold
    "logistic_se",      # Logistic Squared-Error
    "ordinal_ridge",    # Ordinal Ridge
    "ordinal_lad",      # Ordinal LAD

    # Bayesian models (probabilistic interpretability)
    "naive_bayes",      # Naive Bayes
    "tan",              # Tree Augmented Naive Bayes
    "lda",              # Linear Discriminant Analysis

    # Foundation models (interpretable)
    "xrfm",             # Explainable Random Feature Model

    # Subgroup discovery
    "prim",             # Patient Rule Induction Method
}


def get_interpretable_models(
    task_type: str | None = None,
    exclude_slow: bool = False,
    max_samples: int | None = None,
) -> list[str]:
    """Get list of interpretable models.

    Parameters
    ----------
    task_type : str, optional
        Filter by task type ("classification" or "regression").
    exclude_slow : bool, default=False
        Exclude slow and very_slow models.
    max_samples : int, optional
        Only include models that can handle this many samples.

    Returns
    -------
    list of str
        Interpretable model names.
    """
    return list_models(
        task_type=task_type,
        interpretable_only=True,
        exclude_slow=exclude_slow,
        max_samples=max_samples,
    )


def get_interpretable_portfolio(
    task_type: str = "classification",
    n_samples: int = 10000,
    time_budget: str = "medium",
) -> list[str]:
    """Get a recommended portfolio of interpretable models.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    n_samples : int
        Number of samples in the dataset.
    time_budget : str
        Time budget: "fast", "medium", "high", "unlimited".

    Returns
    -------
    list of str
        Recommended interpretable model names.
    """
    # Core interpretable models (always include)
    if task_type == "classification":
        portfolio = ["ebm", "gam", "linear"]
    else:
        portfolio = ["ebmr", "gam", "linear", "mars"]

    # Add based on time budget
    if time_budget in ("high", "unlimited"):
        if task_type == "classification":
            portfolio.extend(["rulefit", "furia", "corels", "gosdt", "nam", "gami_net", "node_gam"])
        else:
            portfolio.extend(["nam", "gami_net", "node_gam", "symbolic_regressor"])
    elif time_budget == "medium":
        if task_type == "classification":
            portfolio.extend(["rulefit", "slim", "node_gam"])
        else:
            portfolio.extend(["rulefit", "node_gam"])
    # fast: just keep the basics

    # Filter by task type
    portfolio = [
        m for m in portfolio
        if m in MODEL_REGISTRY and (
            task_type in MODEL_REGISTRY[m].task_types
            or "both" in MODEL_REGISTRY[m].task_types
        )
    ]

    # Filter by sample size
    portfolio = [
        m for m in portfolio
        if m in MODEL_REGISTRY and (
            MODEL_REGISTRY[m].max_samples is None
            or n_samples <= MODEL_REGISTRY[m].max_samples
        )
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_portfolio = []
    for m in portfolio:
        if m not in seen:
            seen.add(m)
            unique_portfolio.append(m)

    return unique_portfolio
