#!/usr/bin/env python
"""
Benchmark Endgame Models on OpenML Datasets
============================================

This script benchmarks all Endgame models on OpenML benchmark datasets
using the BenchmarkRunner framework. Results are saved to benchmark_results.parquet
and accumulate across runs.

Tests all Endgame models (with automatic classifier/regressor selection):
1. Tree-based: RotationForest, C5.0, Oblique Trees, ADTree, EvTree
2. GBDT wrappers: LightGBM, XGBoost, CatBoost
3. Linear models: MARS
4. Rule-based: RuleFit
5. EBM: Explainable Boosting Machines
6. Bayesian: TAN, EBM-C, ESKDB, KDB (classification only)
7. Neural: TabNet, MLP, EmbeddingMLP
8. Modern Tabular: TabPFN, FT-Transformer, SAINT, NODE, ModernNCA, GANDALF, NAM
9. TALR: TALR-SoftTree, TALR-NODE, RandomRotation (hybrid neural rotation + GBDT)
10. NGBoost: Probabilistic gradient boosting
11. Kernel Methods: SVM
12. Simple Baselines: ELM, Naive Bayes, LDA, QDA, RDA
13. Symbolic Regression (regression only, disabled by default)

Usage:
    python scripts/benchmark_endgame_models.py --suite OpenML-CC18 --max-datasets 10
    python scripts/benchmark_endgame_models.py --quick  # Quick test run
    python scripts/benchmark_endgame_models.py  # Default: sklearn-classic
"""

import argparse
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

# Endgame imports
from endgame.benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    ResultsAnalyzer,
    SuiteLoader,
    BenchmarkReportGenerator,
)

# Sklearn baseline models
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_sklearn_baselines() -> list:
    """Get standard sklearn baseline models (classifier, regressor pairs)."""
    return [
        ("sklearn.RandomForest",
         RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
         RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("sklearn.ExtraTrees",
         ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
         ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("sklearn.GradientBoosting",
         GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
         GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ("sklearn.LogisticRegression/Ridge",
         LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
         Ridge(random_state=42)),
        ("sklearn.DecisionTree",
         DecisionTreeClassifier(max_depth=10, random_state=42),
         DecisionTreeRegressor(max_depth=10, random_state=42)),
    ]


def get_endgame_tree_models() -> list:
    """Get Endgame custom tree models (classifier, regressor pairs)."""
    models = []

    # Rotation Forest
    try:
        from endgame.models.trees import RotationForestClassifier, RotationForestRegressor
        models.append(("eg.RotationForest",
                       RotationForestClassifier(n_estimators=10, n_subsets=3, random_state=42),
                       RotationForestRegressor(n_estimators=10, n_subsets=3, random_state=42)))
    except Exception as e:
        print(f"  Skipping RotationForest: {e}")

    # C5.0 (classification only)
    try:
        from endgame.models.trees import C50Classifier, C50Ensemble
        models.append(("eg.C50",
                       C50Classifier(min_cases=2, cf=0.25, use_rust=False),
                       None))  # No regressor variant
        models.append(("eg.C50Ensemble",
                       C50Ensemble(n_trials=5, min_cases=2, cf=0.25, use_rust=False),
                       None))  # No regressor variant
    except Exception as e:
        print(f"  Skipping C5.0: {e}")

    # Oblique Trees
    try:
        from endgame.models.trees import (
            ObliqueDecisionTreeClassifier,
            ObliqueDecisionTreeRegressor,
            ObliqueRandomForestClassifier,
            ObliqueRandomForestRegressor,
        )
        models.append(("eg.ObliqueTree",
                       ObliqueDecisionTreeClassifier(max_depth=8, random_state=42),
                       ObliqueDecisionTreeRegressor(max_depth=8, random_state=42)))
        models.append(("eg.ObliqueForest",
                       ObliqueRandomForestClassifier(n_estimators=5, max_depth=6, random_state=42),
                       ObliqueRandomForestRegressor(n_estimators=5, max_depth=6, random_state=42)))
    except Exception as e:
        print(f"  Skipping Oblique Trees: {e}")

    # Alternating Decision Tree (classification) / Alternating Model Tree (regression)
    try:
        from endgame.models.trees import AlternatingDecisionTreeClassifier, AlternatingModelTreeRegressor
        models.append(("eg.ADTree",
                       AlternatingDecisionTreeClassifier(n_iterations=10, random_state=42),
                       AlternatingModelTreeRegressor(n_iterations=10, random_state=42)))
    except Exception as e:
        print(f"  Skipping ADTree: {e}")

    # Evolutionary Trees
    try:
        from endgame.models.trees import EvolutionaryTreeClassifier, EvolutionaryTreeRegressor
        models.append(("eg.EvTree",
                       EvolutionaryTreeClassifier(
                           population_size=30, n_generations=30, max_depth=6,
                           patience=10, random_state=42, verbose=False),
                       EvolutionaryTreeRegressor(
                           population_size=30, n_generations=30, max_depth=6,
                           patience=10, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping EvTree: {e}")

    return models


def get_endgame_gbdt_models() -> list:
    """Get Endgame GBDT wrapper models (classifier, regressor pairs)."""
    models = []

    try:
        from endgame.models import LGBMWrapper, XGBWrapper, CatBoostWrapper

        # LightGBM
        try:
            models.append(("eg.LightGBM",
                           LGBMWrapper(
                               preset='custom', random_state=42, verbose=False, use_gpu=False,
                               n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=6,
                               verbosity=-1, task='classification'),
                           LGBMWrapper(
                               preset='custom', random_state=42, verbose=False, use_gpu=False,
                               n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=6,
                               verbosity=-1, task='regression')))
        except Exception as e:
            print(f"  Skipping LightGBM: {e}")

        # XGBoost
        try:
            models.append(("eg.XGBoost",
                           XGBWrapper(
                               preset='custom', random_state=42, verbose=False, use_gpu=False,
                               n_estimators=200, learning_rate=0.1, max_depth=6,
                               verbosity=0, task='classification'),
                           XGBWrapper(
                               preset='custom', random_state=42, verbose=False, use_gpu=False,
                               n_estimators=200, learning_rate=0.1, max_depth=6,
                               verbosity=0, task='regression')))
        except Exception as e:
            print(f"  Skipping XGBoost: {e}")

        # CatBoost
        try:
            models.append(("eg.CatBoost",
                           CatBoostWrapper(
                               preset='custom', random_state=42, use_gpu=False,
                               iterations=200, learning_rate=0.1, depth=6,
                               verbose=False, task='classification'),
                           CatBoostWrapper(
                               preset='custom', random_state=42, use_gpu=False,
                               iterations=200, learning_rate=0.1, depth=6,
                               verbose=False, task='regression')))
        except Exception as e:
            print(f"  Skipping CatBoost: {e}")

    except Exception as e:
        print(f"  Skipping GBDT wrappers: {e}")

    return models


def get_endgame_linear_models() -> list:
    """Get Endgame linear models (MARS)."""
    models = []

    try:
        from endgame.models.linear import MARSClassifier, MARSRegressor
        models.append(("eg.MARS",
                       MARSClassifier(max_terms=20, max_degree=1),
                       MARSRegressor(max_terms=20, max_degree=1)))
    except Exception as e:
        print(f"  Skipping MARS: {e}")

    return models


def get_endgame_rule_models() -> list:
    """Get Endgame rule-based models."""
    models = []

    try:
        from endgame.models.rules import RuleFitClassifier, RuleFitRegressor
        models.append(("eg.RuleFit",
                       RuleFitClassifier(n_estimators=30, max_rules=50, random_state=42),
                       RuleFitRegressor(n_estimators=30, max_rules=50, random_state=42)))
    except Exception as e:
        print(f"  Skipping RuleFit: {e}")

    return models


def get_endgame_ebm_models() -> list:
    """Get Endgame EBM models."""
    models = []

    try:
        from endgame.models.ebm import EBMClassifier, EBMRegressor
        models.append(("eg.EBM",
                       EBMClassifier(
                           max_bins=256, interactions=0, outer_bags=4,
                           max_rounds=5000, random_state=42),
                       EBMRegressor(
                           max_bins=256, interactions=0, outer_bags=4,
                           max_rounds=5000, random_state=42)))
    except Exception as e:
        print(f"  Skipping EBM: {e}")

    return models


def get_endgame_bayesian_models() -> list:
    """Get Endgame Bayesian network classifiers (classification only)."""
    models = []

    try:
        from endgame.models.bayesian import (
            TANClassifier,
            EBMCClassifier,
            ESKDBClassifier,
            KDBClassifier,
        )

        # These are classification-only models
        models.append(("eg.TAN", TANClassifier(), None))
        models.append(("eg.EBM-C", EBMCClassifier(), None))

        try:
            models.append(("eg.ESKDB", ESKDBClassifier(k=2), None))
        except Exception:
            pass

        try:
            models.append(("eg.KDB", KDBClassifier(k=2), None))
        except Exception:
            pass

    except Exception as e:
        print(f"  Skipping Bayesian models: {e}")

    # Neural Bayesian (classification only)
    try:
        from endgame.models.bayesian import NeuralKDBClassifier
        models.append(("eg.NeuralKDB",
                       NeuralKDBClassifier(k=2, epochs=10, random_state=42),
                       None))
    except Exception as e:
        print(f"  Skipping NeuralKDB: {e}")

    return models


def get_endgame_neural_models() -> list:
    """Get Endgame neural network models."""
    models = []

    # TabNet
    try:
        from endgame.models.neural import TabNetClassifier, TabNetRegressor
        models.append(("eg.TabNet",
                       TabNetClassifier(n_steps=3, n_a=8, n_d=8, n_epochs=20, random_state=42, verbose=0),
                       TabNetRegressor(n_steps=3, n_a=8, n_d=8, n_epochs=20, random_state=42, verbose=0)))
    except Exception as e:
        print(f"  Skipping TabNet: {e}")

    # MLP
    try:
        from endgame.models.neural.mlp import MLPClassifier, MLPRegressor
        models.append(("eg.MLP",
                       MLPClassifier(hidden_dims=[64, 32], n_epochs=30, random_state=42, verbose=False),
                       MLPRegressor(hidden_dims=[64, 32], n_epochs=30, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping MLP: {e}")

    # EmbeddingMLP
    try:
        from endgame.models.neural import EmbeddingMLPClassifier, EmbeddingMLPRegressor
        models.append(("eg.EmbeddingMLP",
                       EmbeddingMLPClassifier(hidden_dims=[64, 32], n_epochs=20, random_state=42, verbose=False),
                       EmbeddingMLPRegressor(hidden_dims=[64, 32], n_epochs=20, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping EmbeddingMLP: {e}")

    return models


def get_endgame_ngboost_models() -> list:
    """Get NGBoost models."""
    models = []

    try:
        from endgame.models import NGBoostClassifier, NGBoostRegressor
        models.append(("eg.NGBoost",
                       NGBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=False),
                       NGBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping NGBoost: {e}")

    return models


def get_endgame_kernel_models() -> list:
    """Get kernel-based models (SVM) for ensemble diversity."""
    models = []

    # SVM Classifier/Regressor
    try:
        from endgame.models.kernel import SVMClassifier, SVMRegressor
        models.append(("eg.SVM",
                       SVMClassifier(kernel='rbf', C=1.0, random_state=42),
                       SVMRegressor(kernel='rbf', C=1.0)))
    except Exception as e:
        print(f"  Skipping SVM: {e}")

    return models


def get_endgame_baseline_models() -> list:
    """Get simple baseline models for ensemble diversity."""
    models = []

    # Extreme Learning Machine
    try:
        from endgame.models.baselines import ELMClassifier, ELMRegressor
        models.append(("eg.ELM",
                       ELMClassifier(n_hidden=500, activation='sigmoid', random_state=42),
                       ELMRegressor(n_hidden=500, activation='sigmoid', random_state=42)))
    except Exception as e:
        print(f"  Skipping ELM: {e}")

    # Naive Bayes (classification only)
    try:
        from endgame.models.baselines import NaiveBayesClassifier
        models.append(("eg.NaiveBayes",
                       NaiveBayesClassifier(variant='auto'),
                       None))
    except Exception as e:
        print(f"  Skipping NaiveBayes: {e}")

    # Linear Discriminant Analysis (classification only)
    try:
        from endgame.models.baselines import LDAClassifier
        models.append(("eg.LDA",
                       LDAClassifier(shrinkage='auto'),
                       None))
    except Exception as e:
        print(f"  Skipping LDA: {e}")

    # Quadratic Discriminant Analysis (classification only)
    try:
        from endgame.models.baselines import QDAClassifier
        models.append(("eg.QDA",
                       QDAClassifier(reg_param=0.1),
                       None))
    except Exception as e:
        print(f"  Skipping QDA: {e}")

    # Regularized Discriminant Analysis (classification only)
    try:
        from endgame.models.baselines import RDAClassifier
        models.append(("eg.RDA",
                       RDAClassifier(alpha=0.5, shrinkage=0.1),
                       None))
    except Exception as e:
        print(f"  Skipping RDA: {e}")

    return models


def get_endgame_tabular_models() -> list:
    """Get modern deep learning models for tabular data."""
    models = []

    # TabPFN - classification only, no training required
    try:
        from endgame.models.tabular import TabPFNClassifier
        models.append(("eg.TabPFN", TabPFNClassifier(), None))
    except Exception as e:
        print(f"  Skipping TabPFN: {e}")

    # FT-Transformer
    try:
        from endgame.models.tabular import FTTransformerClassifier, FTTransformerRegressor
        models.append(("eg.FTTransformer",
                       FTTransformerClassifier(n_epochs=10, batch_size=128, random_state=42, verbose=False),
                       FTTransformerRegressor(n_epochs=10, batch_size=128, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping FT-Transformer: {e}")

    # SAINT
    try:
        from endgame.models.tabular import SAINTClassifier, SAINTRegressor
        models.append(("eg.SAINT",
                       SAINTClassifier(n_epochs=10, batch_size=128, random_state=42, verbose=False),
                       SAINTRegressor(n_epochs=10, batch_size=128, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping SAINT: {e}")

    # NODE
    try:
        from endgame.models.tabular import NODEClassifier, NODERegressor
        models.append(("eg.NODE",
                       NODEClassifier(n_layers=2, n_trees=100, n_epochs=20, random_state=42, verbose=False),
                       NODERegressor(n_layers=2, n_trees=100, n_epochs=20, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping NODE: {e}")

    # ModernNCA (classification only)
    try:
        from endgame.models.tabular import ModernNCAClassifier
        models.append(("eg.ModernNCA",
                       ModernNCAClassifier(n_epochs=20, random_state=42, verbose=False),
                       None))
    except Exception as e:
        print(f"  Skipping ModernNCA: {e}")

    # GANDALF
    try:
        from endgame.models.tabular import GANDALFClassifier, GANDALFRegressor
        models.append(("eg.GANDALF",
                       GANDALFClassifier(n_epochs=10, random_state=42, verbose=False),
                       GANDALFRegressor(n_epochs=10, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping GANDALF: {e}")

    # NAM - Neural Additive Models
    try:
        from endgame.models.tabular import NAMClassifier, NAMRegressor
        models.append(("eg.NAM",
                       NAMClassifier(n_hidden=64, n_layers=3, n_epochs=20, random_state=42, verbose=False),
                       NAMRegressor(n_hidden=64, n_layers=3, n_epochs=20, random_state=42, verbose=False)))
    except Exception as e:
        print(f"  Skipping NAM: {e}")

    # TALR - Task-Aware Linear Representations (hybrid neural + GBDT)
    # Soft Tree variant (recommended)
    try:
        from endgame.models.tabular import TALRClassifier, TALRRegressor
        models.append(("eg.TALR-SoftTree",
                       TALRClassifier(
                           surrogate_type='soft_tree',
                           tree_depth=4,
                           n_trees=5,
                           surrogate_epochs=50,
                           gbdt_backend='xgboost',
                           gbdt_preset='fast',
                           random_state=42,
                           verbose=False),
                       TALRRegressor(
                           surrogate_type='soft_tree',
                           tree_depth=4,
                           n_trees=5,
                           surrogate_epochs=50,
                           gbdt_backend='xgboost',
                           gbdt_preset='fast',
                           random_state=42,
                           verbose=False)))
    except Exception as e:
        print(f"  Skipping TALR-SoftTree: {e}")

    # TALR NODE variant (differentiable oblivious trees)
    try:
        from endgame.models.tabular import TALRClassifier, TALRRegressor
        models.append(("eg.TALR-NODE",
                       TALRClassifier(
                           surrogate_type='node',
                           tree_depth=4,
                           n_trees=5,
                           surrogate_epochs=50,
                           gbdt_backend='xgboost',
                           gbdt_preset='fast',
                           random_state=42,
                           verbose=False),
                       TALRRegressor(
                           surrogate_type='node',
                           tree_depth=4,
                           n_trees=5,
                           surrogate_epochs=50,
                           gbdt_backend='xgboost',
                           gbdt_preset='fast',
                           random_state=42,
                           verbose=False)))
    except Exception as e:
        print(f"  Skipping TALR-NODE: {e}")

    # RandomRotation baseline (control - random orthogonal rotation + GBDT)
    try:
        from endgame.models.tabular.talr import RandomRotationClassifier
        models.append(("eg.RandomRotation",
                       RandomRotationClassifier(
                           gbdt_backend='xgboost',
                           gbdt_preset='fast',
                           random_state=42,
                           verbose=False),
                       None))  # No regressor variant available
    except Exception as e:
        print(f"  Skipping RandomRotation: {e}")

    return models


def get_endgame_symbolic_models() -> list:
    """Get symbolic regression models (regression only).

    Note: Symbolic regression spawns external Julia processes that don't respect
    the timeout parameter. Only include if you're willing to wait for it to complete.
    """
    models = []

    try:
        from endgame.models.symbolic import SymbolicRegressor
        models.append(("eg.SymbolicRegression",
                       None,  # No classifier variant
                       SymbolicRegressor(
                           niterations=20,  # Reduced for benchmarking
                           populations=8,
                           population_size=20,
                           maxsize=15,
                           random_state=42,
                           timeout_in_seconds=120,  # Internal timeout
                           verbosity=0)))
    except Exception as e:
        print(f"  Skipping SymbolicRegression: {e}")

    return models


def get_all_models(quick: bool = False, include_symbolic: bool = False) -> list:
    """Get all models to benchmark.

    Parameters
    ----------
    quick : bool
        If True, use fewer/faster models for quick testing.
    include_symbolic : bool
        If True, include symbolic regression (slow, spawns external processes).
    """
    all_models = []

    # Sklearn baselines
    print("  Adding sklearn baselines...")
    sklearn_models = get_sklearn_baselines()
    if quick:
        sklearn_models = sklearn_models[:3]  # Fewer for quick run
    all_models.extend(sklearn_models)
    print(f"    Added {len(sklearn_models)} sklearn models")

    # Endgame tree models
    print("  Adding Endgame tree models...")
    tree_models = get_endgame_tree_models()
    all_models.extend(tree_models)
    print(f"    Added {len(tree_models)} tree models")

    # Endgame GBDT
    print("  Adding Endgame GBDT wrappers...")
    gbdt_models = get_endgame_gbdt_models()
    all_models.extend(gbdt_models)
    print(f"    Added {len(gbdt_models)} GBDT models")

    # Endgame linear (MARS)
    print("  Adding Endgame linear models...")
    linear_models = get_endgame_linear_models()
    all_models.extend(linear_models)
    print(f"    Added {len(linear_models)} linear models")

    # Endgame rule-based (RuleFit)
    print("  Adding Endgame rule models...")
    rule_models = get_endgame_rule_models()
    all_models.extend(rule_models)
    print(f"    Added {len(rule_models)} rule models")

    # Endgame EBM
    print("  Adding Endgame EBM...")
    ebm_models = get_endgame_ebm_models()
    all_models.extend(ebm_models)
    print(f"    Added {len(ebm_models)} EBM models")

    # Endgame Bayesian (classification only)
    print("  Adding Endgame Bayesian models...")
    bayesian_models = get_endgame_bayesian_models()
    all_models.extend(bayesian_models)
    print(f"    Added {len(bayesian_models)} Bayesian models")

    # Endgame Neural
    if not quick:  # Skip for quick runs (they're slower)
        print("  Adding Endgame neural models...")
        neural_models = get_endgame_neural_models()
        all_models.extend(neural_models)
        print(f"    Added {len(neural_models)} neural models")
    else:
        print("  Skipping neural models (quick mode)")

    # NGBoost
    print("  Adding NGBoost...")
    ngboost_models = get_endgame_ngboost_models()
    all_models.extend(ngboost_models)
    print(f"    Added {len(ngboost_models)} NGBoost models")

    # Kernel methods (SVM)
    print("  Adding kernel methods (SVM)...")
    kernel_models = get_endgame_kernel_models()
    all_models.extend(kernel_models)
    print(f"    Added {len(kernel_models)} kernel models")

    # Simple baselines (ELM, NaiveBayes, LDA, QDA, RDA)
    print("  Adding simple baselines (ELM, NaiveBayes, DA)...")
    baseline_models = get_endgame_baseline_models()
    all_models.extend(baseline_models)
    print(f"    Added {len(baseline_models)} baseline models")

    # Modern Tabular Deep Learning
    if not quick:  # Skip for quick runs (they're slower)
        print("  Adding modern tabular deep learning models...")
        tabular_models = get_endgame_tabular_models()
        all_models.extend(tabular_models)
        print(f"    Added {len(tabular_models)} tabular models")
    else:
        print("  Skipping tabular deep learning models (quick mode)")

    # Symbolic Regression (regression only) - disabled by default due to long runtimes
    if include_symbolic and not quick:
        print("  Adding symbolic regression (WARNING: may take a long time)...")
        symbolic_models = get_endgame_symbolic_models()
        all_models.extend(symbolic_models)
        print(f"    Added {len(symbolic_models)} symbolic regression models")
    else:
        print("  Skipping symbolic regression (use --include-symbolic to enable)")

    return all_models


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all Endgame models on OpenML datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sklearn classic datasets
  python scripts/benchmark_endgame_models.py --quick

  # OpenML-CC18 suite (72 datasets)
  python scripts/benchmark_endgame_models.py --suite OpenML-CC18 --max-datasets 10

  # UCI Popular datasets
  python scripts/benchmark_endgame_models.py --suite uci-popular

  # Custom OpenML suite
  python scripts/benchmark_endgame_models.py --suite sklearn-classic --cv-folds 5
        """
    )

    parser.add_argument(
        '--suite',
        type=str,
        default='sklearn-classic',
        help='Benchmark suite name (default: sklearn-classic). Options: sklearn-classic, OpenML-CC18, uci-popular, etc.'
    )
    parser.add_argument(
        '--max-datasets',
        type=int,
        default=None,
        help='Maximum number of datasets to run (default: all in suite)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='Maximum samples per dataset (default: 5000)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick run with fewer models and datasets (3 CV folds, 3 datasets)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.parquet',
        help='Output file path (default: benchmark_results.parquet). Results accumulate across runs.'
    )
    parser.add_argument(
        '--max-time',
        type=int,
        default=300,
        help='Maximum training time per model in seconds (default: 300). Models exceeding this are skipped.'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Generate HTML report at this path (e.g., benchmark_report.html)'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip HTML report generation'
    )
    parser.add_argument(
        '--include-symbolic',
        action='store_true',
        help='Include symbolic regression (slow, spawns external Julia processes)'
    )

    args = parser.parse_args()

    # Default report path if not specified and not disabled
    if args.report is None and not args.no_report:
        args.report = args.output.replace('.parquet', '_report.html')

    print("="*70)
    print("ENDGAME MODELS BENCHMARK ON OPENML DATASETS")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)

    # Show available suites
    print("\nAvailable benchmark suites:")
    for name, desc in SuiteLoader.list_suites().items():
        print(f"  - {name}: {desc}")

    # Adjust settings for quick mode
    if args.quick:
        args.suite = 'sklearn-classic'
        args.max_datasets = 3
        args.cv_folds = 3
        print("\nQuick mode enabled: using 3 datasets, 3 CV folds")

    # Collect models
    print(f"\n[1/2] Collecting models...")
    all_models = get_all_models(quick=args.quick, include_symbolic=args.include_symbolic)
    print(f"\n  Total models: {len(all_models)}")
    print(f"  Models: {[m[0] for m in all_models]}")

    # Configure and run benchmark
    print(f"\n[2/2] Running benchmark...")
    print(f"  Suite: {args.suite}")
    print(f"  Max datasets: {args.max_datasets or 'all'}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Max time per model: {args.max_time}s")
    print(f"  Output: {args.output} (append mode)")

    config = BenchmarkConfig(
        suite=args.suite,
        max_datasets=args.max_datasets,
        max_samples=args.max_samples,
        cv_folds=args.cv_folds,
        timeout_per_fit=args.max_time,
        profile_datasets=True,
        profile_groups=["simple", "statistical"],
        verbose=True,
    )

    runner = BenchmarkRunner(config=config)
    tracker = runner.run(all_models, output_file=args.output)

    # Analyze results
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)

    print("\n" + tracker.summary())

    # Model rankings
    analyzer = ResultsAnalyzer(tracker, metric='accuracy')
    print("\n" + analyzer.summary_table())

    # Separate analysis for Endgame vs Sklearn
    print("\n" + "-"*70)
    print("ENDGAME vs SKLEARN COMPARISON")
    print("-"*70)

    try:
        import polars as pl
        HAS_POLARS = True
    except ImportError:
        HAS_POLARS = False
        import pandas as pd

    df = tracker.to_dataframe()

    # Get successful experiments
    try:
        if HAS_POLARS:
            successful = df.filter(pl.col('status') == 'success')
            eg_models = successful.filter(pl.col('model_name').str.starts_with('eg.'))
            sklearn_models_df = successful.filter(pl.col('model_name').str.starts_with('sklearn.'))
        else:
            successful = df[df['status'] == 'success']
            eg_models = successful[successful['model_name'].str.startswith('eg.')]
            sklearn_models_df = successful[successful['model_name'].str.startswith('sklearn.')]

        if len(eg_models) > 0:
            if HAS_POLARS:
                eg_mean = eg_models['metric_accuracy'].mean()
            else:
                eg_mean = eg_models['metric_accuracy'].mean()
            print(f"\nEndgame models mean accuracy: {eg_mean:.4f} (n={len(eg_models)})")

        if len(sklearn_models_df) > 0:
            if HAS_POLARS:
                sklearn_mean = sklearn_models_df['metric_accuracy'].mean()
            else:
                sklearn_mean = sklearn_models_df['metric_accuracy'].mean()
            print(f"Sklearn models mean accuracy: {sklearn_mean:.4f} (n={len(sklearn_models_df)})")
    except Exception as e:
        print(f"  Analysis error: {e}")

    # Best models per dataset
    print("\n" + "-"*70)
    print("BEST MODELS PER DATASET")
    print("-"*70)

    datasets = set(r.dataset_name for r in tracker.get_successful())
    for dataset_name in sorted(datasets):
        dataset_records = tracker.get_by_dataset(dataset_name)
        successful_records = [r for r in dataset_records if r.status == 'success']

        if successful_records:
            best = max(successful_records, key=lambda r: r.metrics.get('accuracy', 0))
            print(f"\n{dataset_name}:")
            print(f"  Best: {best.model_name} ({best.metrics.get('accuracy', 0):.4f})")

            # Top 5
            sorted_records = sorted(
                successful_records,
                key=lambda r: r.metrics.get('accuracy', 0),
                reverse=True
            )[:5]
            print("  Top 5:")
            for i, r in enumerate(sorted_records, 1):
                print(f"    {i}. {r.model_name}: {r.metrics.get('accuracy', 0):.4f}")

    print("\n" + "="*70)
    print(f"Benchmark complete! {len(tracker)} experiments recorded.")
    print(f"Results saved to: {args.output}")
    print("="*70)

    # Generate HTML report
    if args.report and not args.no_report:
        print(f"\n[3/3] Generating HTML report...")
        try:
            report_gen = BenchmarkReportGenerator(
                tracker,
                title=f"Endgame Benchmark Report - {args.suite}"
            )

            # Extract interpretability outputs from fitted models
            # We'll fit interpretable models on a sample dataset to get their outputs
            print("  Extracting interpretability outputs from fitted models...")
            _add_interpretability_outputs(report_gen, all_models, runner.datasets)

            # Generate the report
            report_path = report_gen.generate(
                args.report,
                include_interpretability=True,
                include_meta_features=True,
            )
            print(f"  HTML report saved to: {report_path}")
        except ImportError as e:
            print(f"  Skipping HTML report (missing dependency): {e}")
        except Exception as e:
            print(f"  Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()


def _add_interpretability_outputs(
    report_gen: BenchmarkReportGenerator,
    model_specs: list,
    datasets: list,
) -> None:
    """Extract and add interpretability outputs from models.

    Fits interpretable models on each dataset and extracts their
    learned rules, trees, equations, etc.
    """
    from sklearn.model_selection import train_test_split

    # Models known to have interpretability outputs
    INTERPRETABLE_MODELS = {
        'eg.RuleFit': 'get_rules',
        'eg.SymbolicRegression': 'get_best_equation',
        'eg.C50': 'get_structure',
        'eg.EBM': 'term_importances',
        'eg.MARS': 'summary',
        'eg.ADTree': 'summary',
        'eg.EvTree': 'feature_importances_',
    }

    # Limit to first 3 datasets to avoid excessive computation
    for dataset in datasets[:3]:
        print(f"    Processing {dataset.name}...")

        # Determine if classification or regression
        is_classification = dataset.task_type.value != 'regression'

        for model_spec in model_specs:
            model_name = model_spec[0]

            # Check if this model has interpretability
            if model_name not in INTERPRETABLE_MODELS:
                continue

            # Get the right model variant
            if len(model_spec) == 3:
                model = model_spec[1] if is_classification else model_spec[2]
            else:
                model = model_spec[1]

            if model is None:
                continue

            try:
                # Clone the model
                from sklearn.base import clone
                model_clone = clone(model)

                # Fit on the dataset
                X, y = dataset.X, dataset.y

                # Encode target for classification
                if is_classification:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                # Fit the model (with timeout protection via smaller sample)
                if len(X) > 1000:
                    X_sample, _, y_sample, _ = train_test_split(
                        X, y, train_size=1000, random_state=42, stratify=y if is_classification else None
                    )
                else:
                    X_sample, y_sample = X, y

                model_clone.fit(X_sample, y_sample)

                # Extract interpretability output
                output = _extract_model_output(model_clone, model_name, INTERPRETABLE_MODELS)

                if output:
                    report_gen.add_interpretability_output(
                        model_name=model_name,
                        dataset_name=dataset.name,
                        output=output,
                        output_type="text",
                    )

            except Exception as e:
                # Silently skip models that fail
                pass


def _extract_model_output(model, model_name: str, interpretable_models: dict) -> str:
    """Extract interpretability output from a fitted model."""
    output = None

    try:
        # RuleFit
        if hasattr(model, 'get_rules'):
            rules = model.get_rules()
            if rules:
                lines = []
                for i, r in enumerate(rules[:20]):  # Top 20 rules
                    if isinstance(r, dict):
                        rule_str = r.get('rule', str(r))
                        coef = r.get('coef', r.get('importance', 0))
                        lines.append(f"Rule {i+1}: {rule_str}")
                        lines.append(f"         Coefficient: {coef:.4f}")
                    else:
                        lines.append(f"Rule {i+1}: {r}")
                output = "\n".join(lines)

        # Symbolic Regression
        elif hasattr(model, 'get_best_equation'):
            output = f"Best Equation:\n{model.get_best_equation()}"
            if hasattr(model, 'summary'):
                try:
                    summary = model.summary()
                    if summary:
                        output += f"\n\nAll Equations (by complexity):\n{summary}"
                except Exception:
                    pass

        # C5.0 / Decision Trees
        elif hasattr(model, 'get_structure'):
            output = model.get_structure()

        # EBM
        elif hasattr(model, 'term_importances'):
            importances = model.term_importances()
            if hasattr(model, 'get_term_names'):
                term_names = model.get_term_names()
            else:
                term_names = [f"Term {i}" for i in range(len(importances))]
            sorted_terms = sorted(zip(term_names, importances), key=lambda x: abs(x[1]), reverse=True)
            output = "EBM Term Importances:\n" + "\n".join([
                f"  {name}: {imp:.4f}" for name, imp in sorted_terms[:15]
            ])

        # MARS / ADTree with summary
        elif hasattr(model, 'summary'):
            output = model.summary()

        # Feature importances fallback
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:15]
            output = "Feature Importances:\n" + "\n".join([
                f"  Feature {i}: {importances[i]:.4f}" for i in sorted_idx
            ])

    except Exception as e:
        output = f"Error extracting output: {str(e)}"

    return output


if __name__ == "__main__":
    main()
