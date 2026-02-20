#!/usr/bin/env python
"""
Experiment 1: Tabular Model Benchmark

Benchmarks 60+ Endgame models across the full Grinsztajn NeurIPS 2022
benchmark suite (~45 datasets, classification + regression).

Purpose: Demonstrate that Endgame's unified API produces competitive results
across model families. Not "our XGBoost beats their XGBoost" — rather,
"breadth works end-to-end, presets are sensible, and glass-box models
are worth including."

Reference: "Why do tree-based models still outperform deep learning on
typical tabular data?" (Grinsztajn et al., NeurIPS 2022)

Usage:
    python benchmarks/run_tabular_benchmark.py                            # Full run (~45 datasets)
    python benchmarks/run_tabular_benchmark.py --quick                     # Quick test (3 datasets)
    python benchmarks/run_tabular_benchmark.py --suite grinsztajn-classif  # Classification only
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Timeout helper (process-based — can kill stuck C extensions)
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _run_in_process(fn, args, result_queue):
    """Worker target: run fn(*args) and put result on queue."""
    import os, sys
    warnings.filterwarnings("ignore")
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        result = fn(*args)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", e))
    finally:
        devnull.close()


def run_with_timeout(fn, args, timeout):
    """Run fn(*args) in a subprocess with a hard timeout.

    Returns the result or raises TimeoutError / the original exception.
    """
    import multiprocessing as mp

    if timeout is None or timeout <= 0:
        return fn(*args)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_process, args=(fn, args, q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.kill()
        p.join(5)
        raise TimeoutError(f"Timed out after {timeout}s")

    if q.empty():
        raise RuntimeError("Worker crashed without returning a result")

    status, payload = q.get_nowait()
    if status == "ok":
        return payload
    raise payload


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_proba, n_bins=10):
    """Compute ECE for binary classification."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

def _try_load(loader_fn, name: str = ""):
    """Try to instantiate a model; return None on any import/init error."""
    try:
        return loader_fn()
    except Exception as e:
        if name:
            print(f"    [skip] {name}: {e!s:.80s}")
        return None


def _ovr(estimator):
    """Wrap a binary-only classifier with OneVsRestClassifier for multiclass."""
    from sklearn.multiclass import OneVsRestClassifier
    return OneVsRestClassifier(estimator)


def _get_classification_models(quick=False):
    """Return classification models — as many as the environment supports."""
    import endgame as eg
    from endgame.models import tabular, trees, neural
    from endgame.models.interpretable import corels, gam, gami_net, gosdt, node_gam, slim
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    models = {}

    # ── 1. GBDTs ─────────────────────────────────────────────────────
    m = _try_load(lambda: eg.models.LGBMWrapper(preset="fast"), "LightGBM")
    if m is not None: models["LightGBM"] = m
    m = _try_load(lambda: eg.models.XGBWrapper(preset="fast"), "XGBoost")
    if m is not None: models["XGBoost"] = m
    m = _try_load(lambda: eg.models.CatBoostWrapper(preset="fast"), "CatBoost")
    if m is not None: models["CatBoost"] = m

    # ── 2. sklearn tree ensembles ────────────────────────────────────
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, random_state=42
    )
    models["AdaBoost"] = AdaBoostClassifier(n_estimators=200, random_state=42)
    models["DecisionTree"] = DecisionTreeClassifier(random_state=42)

    # ── 3. Foundation / pretrained models ────────────────────────────
    m = _try_load(lambda: tabular.TabPFN25Classifier(), "TabPFN2.5")
    if m is not None: models["TabPFN2.5"] = m
    m = _try_load(lambda: tabular.TabPFNv2Classifier(), "TabPFNv2")
    if m is not None: models["TabPFNv2"] = m
    m = _try_load(lambda: tabular.TabPFNClassifier(), "TabPFN")
    if m is not None: models["TabPFN"] = m
    m = _try_load(lambda: tabular.xRFMClassifier(), "xRFM")
    if m is not None: models["xRFM"] = m

    # ── 4. Modern deep tabular ───────────────────────────────────────
    m = _try_load(lambda: tabular.RealMLPClassifier(), "RealMLP")
    if m is not None: models["RealMLP"] = m
    m = _try_load(lambda: tabular.TabDPTClassifier(), "TabDPT")
    if m is not None: models["TabDPT"] = m
    m = _try_load(lambda: tabular.TabMClassifier(), "TabM")
    if m is not None: models["TabM"] = m
    m = _try_load(lambda: eg.models.ModernNCAClassifier(), "ModernNCA")
    if m is not None: models["ModernNCA"] = m
    m = _try_load(lambda: tabular.TabRClassifier(), "TabR")
    if m is not None: models["TabR"] = m

    # ── 5. Glass-box / interpretable ─────────────────────────────────
    m = _try_load(lambda: eg.models.EBMClassifier(), "EBM")
    if m is not None: models["EBM"] = m
    m = _try_load(lambda: eg.models.MARSClassifier(), "MARS")
    if m is not None: models["MARS"] = m
    m = _try_load(lambda: eg.models.RuleFitClassifier(), "RuleFit")
    if m is not None: models["RuleFit"] = m
m = _try_load(lambda: eg.models.NAMClassifier(n_epochs=50), "NAM")
    if m is not None: models["NAM"] = m
    m = _try_load(lambda: tabular.GRANDEClassifier(), "GRANDE")
    if m is not None: models["GRANDE"] = m
    m = _try_load(lambda: _ovr(gam.GAMClassifier(lam=0.6, n_splines=10, max_iter=30)), "GAM")
    if m is not None: models["GAM"] = m
    m = _try_load(lambda: gami_net.GAMINetClassifier(), "GAMI-Net")
    if m is not None: models["GAMI-Net"] = m
    m = _try_load(lambda: _ovr(gosdt.GOSDTClassifier()), "GOSDT")
    if m is not None: models["GOSDT"] = m
    m = _try_load(lambda: _ovr(corels.CORELSClassifier()), "CORELS")
    if m is not None: models["CORELS"] = m
    m = _try_load(lambda: _ovr(slim.SLIMClassifier()), "SLIM")
    if m is not None: models["SLIM"] = m
    m = _try_load(lambda: _ovr(slim.FasterRiskClassifier()), "FasterRisk")
    if m is not None: models["FasterRisk"] = m
    m = _try_load(lambda: node_gam.NodeGAMClassifier(), "NodeGAM")
    if m is not None: models["NodeGAM"] = m

    # ── 6. Linear / baselines (scale-sensitive models get a pipeline) ──
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models["LogisticRegression"] = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )
    models["SGD"] = make_pipeline(
        StandardScaler(), SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    )
    models["KNN"] = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    )
    m = _try_load(lambda: eg.models.LinearClassifier(), "LinearClassifier")
    if m is not None: models["LinearClassifier"] = m
    m = _try_load(lambda: eg.models.NaiveBayesClassifier(), "NaiveBayes")
    if m is not None: models["NaiveBayes"] = m
    m = _try_load(lambda: eg.models.KNNClassifier(), "KNN_eg")
    if m is not None: models["KNN_eg"] = m
    m = _try_load(lambda: eg.models.LDAClassifier(), "LDA")
    if m is not None: models["LDA"] = m
    m = _try_load(lambda: eg.models.QDAClassifier(), "QDA")
    if m is not None: models["QDA"] = m
    m = _try_load(lambda: eg.models.RDAClassifier(), "RDA")
    if m is not None: models["RDA"] = m
    m = _try_load(lambda: eg.models.ELMClassifier(random_state=42), "ELM")
    if m is not None: models["ELM"] = m

    # ── 7. Kernel methods ────────────────────────────────────────────
    models["SVM_RBF"] = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)
    )
    m = _try_load(lambda: eg.models.SVMClassifier(), "SVM_eg")
    if m is not None: models["SVM_eg"] = m

    # ── 8. Bayesian classifiers ──────────────────────────────────────
    _bayes_kw = dict(discretizer_strategy="equal_freq", discretizer_max_bins=15)
    m = _try_load(lambda: eg.models.TANClassifier(**_bayes_kw), "TAN")
    if m is not None: models["TAN"] = m
    m = _try_load(lambda: eg.models.KDBClassifier(**_bayes_kw), "KDB")
    if m is not None: models["KDB"] = m
    m = _try_load(lambda: eg.models.ESKDBClassifier(**_bayes_kw), "ESKDB")
    if m is not None: models["ESKDB"] = m
    m = _try_load(lambda: eg.models.EBMCClassifier(**_bayes_kw), "EBMC")
    if m is not None: models["EBMC"] = m
    m = _try_load(lambda: eg.models.bayesian.NeuralKDBClassifier(**_bayes_kw), "NeuralKDB")
    if m is not None: models["NeuralKDB"] = m

    # ── 9. Custom trees ──────────────────────────────────────────────
    m = _try_load(lambda: eg.models.RotationForestClassifier(random_state=42), "RotationForest")
    if m is not None: models["RotationForest"] = m
    m = _try_load(lambda: eg.models.C50Classifier(), "C5.0")
    if m is not None: models["C5.0"] = m
    m = _try_load(lambda: eg.models.ObliqueRandomForestClassifier(
        n_estimators=50, max_depth=10, min_samples_split=30, min_samples_leaf=10,
        random_state=42,
    ), "ObliqueRF")
    if m is not None: models["ObliqueRF"] = m
    m = _try_load(lambda: eg.models.HonestForestClassifier(random_state=42), "HonestForest")
    if m is not None: models["HonestForest"] = m
    m = _try_load(lambda: trees.AlternatingDecisionTreeClassifier(), "ADTree")
    if m is not None: models["ADTree"] = m
    m = _try_load(lambda: eg.models.EvolutionaryTreeClassifier(random_state=42), "EvolutionaryTree")
    if m is not None: models["EvolutionaryTree"] = m

    # ── 10. Rule-based ───────────────────────────────────────────────
    m = _try_load(lambda: eg.models.FURIAClassifier(), "FURIA")
    if m is not None: models["FURIA"] = m

    # ── 11. Probabilistic ────────────────────────────────────────────
    m = _try_load(lambda: eg.models.NGBoostClassifier(), "NGBoost")
    if m is not None: models["NGBoost"] = m

    # ── 12. Ordinal regression ───────────────────────────────────────
    m = _try_load(lambda: eg.models.OrdinalClassifier(), "Ordinal")
    if m is not None: models["Ordinal"] = m
    m = _try_load(lambda: eg.models.OrdinalRidge(), "OrdinalRidge")
    if m is not None: models["OrdinalRidge"] = m

    # ── 13. Subgroup / neuroevolution ────────────────────────────────
    m = _try_load(lambda: eg.models.PRIMClassifier(), "PRIM")
    if m is not None: models["PRIM"] = m
    m = _try_load(lambda: eg.models.NEATClassifier(n_generations=5, population_size=50), "NEAT")
    if m is not None: models["NEAT"] = m
    m = _try_load(lambda: eg.models.TensorNEATClassifier(n_generations=5), "TensorNEAT")
    if m is not None: models["TensorNEAT"] = m

    # ── 14. Deep tabular ─────────────────────────────────────────────
    m = _try_load(lambda: eg.models.FTTransformerClassifier(n_epochs=50), "FTTransformer")
    if m is not None: models["FTTransformer"] = m
    m = _try_load(lambda: eg.models.SAINTClassifier(n_epochs=50), "SAINT")
    if m is not None: models["SAINT"] = m
    m = _try_load(lambda: eg.models.NODEClassifier(n_epochs=50), "NODE")
    if m is not None: models["NODE"] = m
    m = _try_load(lambda: tabular.GANDALFClassifier(
        n_epochs=100, batch_size=32, learning_rate=1e-2,
    ), "GANDALF")
    if m is not None: models["GANDALF"] = m
    m = _try_load(lambda: neural.MLPClassifier(n_epochs=50), "MLP")
    if m is not None: models["MLP"] = m
    m = _try_load(lambda: neural.EmbeddingMLPClassifier(n_epochs=50), "EmbeddingMLP")
    if m is not None: models["EmbeddingMLP"] = m
    m = _try_load(lambda: neural.TabNetClassifier(
        n_d=8, n_a=8, n_steps=3, batch_size=128, virtual_batch_size=64, n_epochs=30,
    ), "TabNet")
    if m is not None: models["TabNet"] = m
    m = _try_load(lambda: tabular.TabularResNetClassifier(n_epochs=50), "TabResNet")
    if m is not None: models["TabResNet"] = m
    m = _try_load(lambda: __import__(
        "endgame.models.tabular.tab_transformer", fromlist=["TabTransformerClassifier"]
    ).TabTransformerClassifier(), "TabTransformer")
    if m is not None: models["TabTransformer"] = m

    return models


def _get_regression_models(quick=False):
    """Return regression models — as many as the environment supports."""
    import endgame as eg
    from endgame.models import tabular, trees, neural
    from endgame.models.interpretable import gam, gami_net, node_gam
    from sklearn.ensemble import (
        AdaBoostRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Lasso, Ridge, SGDRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    models = {}

    # ── 1. GBDTs ─────────────────────────────────────────────────────
    m = _try_load(lambda: eg.models.LGBMWrapper(preset="fast", task="regression"), "LightGBM")
    if m is not None: models["LightGBM"] = m
    m = _try_load(lambda: eg.models.XGBWrapper(preset="fast", task="regression"), "XGBoost")
    if m is not None: models["XGBoost"] = m
    m = _try_load(lambda: eg.models.CatBoostWrapper(preset="fast", task="regression"), "CatBoost")
    if m is not None: models["CatBoost"] = m

    # ── 2. sklearn tree ensembles ────────────────────────────────────
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    models["GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42
    )
    models["AdaBoost"] = AdaBoostRegressor(n_estimators=200, random_state=42)
    models["DecisionTree"] = DecisionTreeRegressor(random_state=42)

    # ── 3. Foundation / pretrained models ────────────────────────────
    m = _try_load(lambda: tabular.TabPFN25Regressor(), "TabPFN2.5")
    if m is not None: models["TabPFN2.5"] = m
    m = _try_load(lambda: tabular.TabPFNv2Regressor(), "TabPFNv2")
    if m is not None: models["TabPFNv2"] = m
    m = _try_load(lambda: tabular.xRFMRegressor(), "xRFM")
    if m is not None: models["xRFM"] = m

    # ── 4. Modern deep tabular ───────────────────────────────────────
    m = _try_load(lambda: tabular.RealMLPRegressor(), "RealMLP")
    if m is not None: models["RealMLP"] = m
    m = _try_load(lambda: tabular.TabDPTRegressor(), "TabDPT")
    if m is not None: models["TabDPT"] = m
    m = _try_load(lambda: tabular.TabMRegressor(), "TabM")
    if m is not None: models["TabM"] = m
    m = _try_load(lambda: tabular.TabRRegressor(), "TabR")
    if m is not None: models["TabR"] = m

    # ── 5. Glass-box / interpretable ─────────────────────────────────
    m = _try_load(lambda: eg.models.EBMRegressor(), "EBM")
    if m is not None: models["EBM"] = m
    m = _try_load(lambda: eg.models.MARSRegressor(), "MARS")
    if m is not None: models["MARS"] = m
    m = _try_load(lambda: eg.models.RuleFitRegressor(), "RuleFit")
    if m is not None: models["RuleFit"] = m
    m = _try_load(lambda: eg.models.NAMRegressor(n_epochs=50), "NAM")
    if m is not None: models["NAM"] = m
    m = _try_load(lambda: tabular.GRANDERegressor(), "GRANDE")
    if m is not None: models["GRANDE"] = m
    m = _try_load(lambda: gam.GAMRegressor(lam=0.6, n_splines=10, max_iter=30), "GAM")
    if m is not None: models["GAM"] = m
    m = _try_load(lambda: gami_net.GAMINetRegressor(), "GAMI-Net")
    if m is not None: models["GAMI-Net"] = m
    m = _try_load(lambda: node_gam.NodeGAMRegressor(), "NodeGAM")
    if m is not None: models["NodeGAM"] = m

    # ── 6. Linear / baselines (scale-sensitive models get a pipeline) ──
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models["Ridge"] = make_pipeline(StandardScaler(), Ridge())
    models["Lasso"] = make_pipeline(StandardScaler(), Lasso(max_iter=1000))
    from sklearn.compose import TransformedTargetRegressor
    models["SGD"] = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SGDRegressor(
            max_iter=1000, random_state=42, early_stopping=True, n_iter_no_change=10,
            eta0=0.01, learning_rate='adaptive', power_t=0.25,
            penalty='l2', alpha=1e-3,
        )),
        transformer=StandardScaler(),
    )
    models["KNN"] = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, n_jobs=-1))
    m = _try_load(lambda: eg.models.LinearRegressor(), "LinearRegressor")
    if m is not None: models["LinearRegressor"] = m
    m = _try_load(lambda: eg.models.KNNRegressor(), "KNN_eg")
    if m is not None: models["KNN_eg"] = m
    m = _try_load(lambda: eg.models.ELMRegressor(random_state=42), "ELM")
    if m is not None: models["ELM"] = m

    # ── 7. Kernel methods ────────────────────────────────────────────
    models["SVR_RBF"] = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    m = _try_load(lambda: eg.models.SVMRegressor(), "SVM_eg")
    if m is not None: models["SVM_eg"] = m

    # ── 8. Custom trees ──────────────────────────────────────────────
    m = _try_load(lambda: eg.models.RotationForestRegressor(random_state=42), "RotationForest")
    if m is not None: models["RotationForest"] = m
    m = _try_load(lambda: eg.models.CubistRegressor(), "Cubist")
    if m is not None: models["Cubist"] = m
    m = _try_load(lambda: eg.models.ObliqueRandomForestRegressor(
        n_estimators=50, max_depth=10, min_samples_split=30, min_samples_leaf=10,
        random_state=42,
    ), "ObliqueRF")
    if m is not None: models["ObliqueRF"] = m
    m = _try_load(lambda: eg.models.EvolutionaryTreeRegressor(random_state=42), "EvolutionaryTree")
    if m is not None: models["EvolutionaryTree"] = m
    m = _try_load(lambda: eg.models.QuantileRegressorForest(random_state=42), "QuantileRF")
    if m is not None: models["QuantileRF"] = m

    # ── 9. Probabilistic ────────────────────────────────────────────
    m = _try_load(lambda: eg.models.NGBoostRegressor(), "NGBoost")
    if m is not None: models["NGBoost"] = m

    # ── 10. Neuroevolution ─────────────────────────────────────────────
    # PRIM is a subgroup discovery method, not suited for general regression
    m = _try_load(lambda: eg.models.NEATRegressor(n_generations=5, population_size=50), "NEAT")
    if m is not None: models["NEAT"] = m
    m = _try_load(lambda: eg.models.TensorNEATRegressor(n_generations=5), "TensorNEAT")
    if m is not None: models["TensorNEAT"] = m

    # ── 11. Deep tabular ─────────────────────────────────────────────
    m = _try_load(lambda: eg.models.FTTransformerRegressor(n_epochs=50), "FTTransformer")
    if m is not None: models["FTTransformer"] = m
    m = _try_load(lambda: eg.models.SAINTRegressor(n_epochs=50), "SAINT")
    if m is not None: models["SAINT"] = m
    m = _try_load(lambda: eg.models.NODERegressor(n_epochs=50), "NODE")
    if m is not None: models["NODE"] = m
    m = _try_load(lambda: tabular.GANDALFRegressor(
        n_epochs=100, batch_size=32, learning_rate=1e-2,
    ), "GANDALF")
    if m is not None: models["GANDALF"] = m
    m = _try_load(lambda: neural.MLPRegressor(n_epochs=50), "MLP")
    if m is not None: models["MLP"] = m
    m = _try_load(lambda: neural.EmbeddingMLPRegressor(n_epochs=50), "EmbeddingMLP")
    if m is not None: models["EmbeddingMLP"] = m
    m = _try_load(lambda: neural.TabNetRegressor(
        n_d=8, n_a=8, n_steps=3, batch_size=128, virtual_batch_size=64, n_epochs=30,
    ), "TabNet")
    if m is not None: models["TabNet"] = m
    m = _try_load(lambda: tabular.TabularResNetRegressor(n_epochs=50), "TabResNet")
    if m is not None: models["TabResNet"] = m
    m = _try_load(lambda: __import__(
        "endgame.models.tabular.tab_transformer", fromlist=["TabTransformerRegressor"]
    ).TabTransformerRegressor(), "TabTransformer")
    if m is not None: models["TabTransformer"] = m

    return models


def get_models(quick=False, task="classification"):
    """Return dict of model_name -> model_instance."""
    if task == "regression":
        return _get_regression_models(quick)
    return _get_classification_models(quick)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(suite="grinsztajn", max_samples=5000, quick=False):
    """Load benchmark datasets using Endgame's SuiteLoader.

    Default suite is the full Grinsztajn NeurIPS 2022 benchmark (~45 datasets).
    Use 'grinsztajn-classif' for classification-only subset.
    """
    from endgame.benchmark import SuiteLoader

    if quick:
        loader = SuiteLoader("quick-test", max_samples=max_samples)
    else:
        loader = SuiteLoader(suite, max_samples=max_samples)

    return list(loader.load())


def is_classification(y):
    """Detect if target is classification (discrete) or regression (continuous)."""
    if y.dtype.kind == "f":
        n_unique = len(np.unique(y))
        return n_unique < 30  # Heuristic: <30 unique floats = likely classification
    return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_classification(model, X, y, n_splits=5):
    """Evaluate a classification model with stratified k-fold CV."""
    warnings.filterwarnings("ignore")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    is_binary = n_classes == 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    fold_probas = []
    fit_times = []
    predict_times = []
    fold_errors = []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        try:
            t0 = time.time()
            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)
            fit_times.append(time.time() - t0)

            t0 = time.time()
            y_pred = model_clone.predict(X_test)
            predict_times.append(time.time() - t0)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            if hasattr(model_clone, "predict_proba"):
                y_proba = model_clone.predict_proba(X_test)
                model_classes = getattr(model_clone, "classes_", None)
                fold_probas.append((y_proba, model_classes, len(y_test)))
        except Exception as e:
            fold_errors.append(str(e))
            continue

    if not all_y_true:
        return {"error": fold_errors[0] if fold_errors else "All folds failed"}

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    avg = "binary" if is_binary else "weighted"

    metrics = {
        "task": "classification",
        "accuracy": accuracy_score(all_y_true, all_y_pred),
        "balanced_accuracy": balanced_accuracy_score(all_y_true, all_y_pred),
        "f1_weighted": f1_score(all_y_true, all_y_pred, average="weighted"),
        "precision_weighted": precision_score(
            all_y_true, all_y_pred, average="weighted", zero_division=0,
        ),
        "recall_weighted": recall_score(
            all_y_true, all_y_pred, average="weighted", zero_division=0,
        ),
        "mcc": matthews_corrcoef(all_y_true, all_y_pred),
        "cohen_kappa": cohen_kappa_score(all_y_true, all_y_pred),
        "fit_time_mean_s": np.mean(fit_times),
        "fit_time_std_s": np.std(fit_times),
        "predict_time_mean_s": np.mean(predict_times),
        "n_classes": n_classes,
    }

    if fold_probas:
        try:
            all_y_proba = _assemble_fold_probas(fold_probas, n_classes)
            if is_binary:
                proba_pos = all_y_proba[:, 1]
                metrics["roc_auc"] = roc_auc_score(all_y_true, proba_pos)
                metrics["brier_score"] = brier_score_loss(all_y_true, proba_pos)
                metrics["log_loss"] = log_loss(all_y_true, all_y_proba)
                metrics["ece"] = expected_calibration_error(all_y_true, proba_pos)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    all_y_true, all_y_proba, multi_class="ovr", average="weighted"
                )
                metrics["log_loss"] = log_loss(all_y_true, all_y_proba)
        except Exception:
            pass

    return metrics


def _assemble_fold_probas(fold_probas, n_classes):
    """Assemble per-fold probability arrays into a single (N, n_classes) array.

    Handles the case where different folds produce different numbers of
    probability columns (e.g. when a rare class is absent from a fold's
    training set).
    """
    total = sum(n for _, _, n in fold_probas)
    result = np.zeros((total, n_classes))
    offset = 0
    for proba, model_classes, n in fold_probas:
        if proba.shape[1] == n_classes:
            result[offset:offset + n] = proba
        elif model_classes is not None and len(model_classes) == proba.shape[1]:
            for j, c in enumerate(model_classes):
                c_int = int(c)
                if 0 <= c_int < n_classes:
                    result[offset:offset + n, c_int] = proba[:, j]
        else:
            cols = min(proba.shape[1], n_classes)
            result[offset:offset + n, :cols] = proba[:, :cols]
        offset += n
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    result /= row_sums
    return result


def evaluate_model_regression(model, X, y, n_splits=5):
    """Evaluate a regression model with k-fold CV."""
    warnings.filterwarnings("ignore")
    from sklearn.metrics import (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    fit_times = []
    predict_times = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            t0 = time.time()
            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)
            fit_times.append(time.time() - t0)

            t0 = time.time()
            y_pred = model_clone.predict(X_test)
            predict_times.append(time.time() - t0)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        except Exception as e:
            return {"error": str(e)}

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    y_range = all_y_true.max() - all_y_true.min()
    nrmse = rmse / y_range if y_range > 0 else float("nan")

    return {
        "task": "regression",
        "r2": r2_score(all_y_true, all_y_pred),
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mean_absolute_error(all_y_true, all_y_pred),
        "median_ae": median_absolute_error(all_y_true, all_y_pred),
        "max_error": max_error(all_y_true, all_y_pred),
        "explained_variance": explained_variance_score(all_y_true, all_y_pred),
        "mape": mean_absolute_percentage_error(all_y_true, all_y_pred),
        "fit_time_mean_s": np.mean(fit_times),
        "fit_time_std_s": np.std(fit_times),
        "predict_time_mean_s": np.mean(predict_times),
    }


def _clone_model(model):
    """Clone a model (sklearn-compatible)."""
    from sklearn.base import clone

    try:
        return clone(model)
    except Exception:
        # Fallback: recreate from class + get_params
        return model.__class__(**model.get_params())


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataset(ds):
    """Preprocess a dataset: encode categoricals, handle NaNs."""
    X = ds.X.copy() if hasattr(ds.X, "copy") else np.array(ds.X, dtype=np.float32)
    y = ds.y.copy() if hasattr(ds.y, "copy") else np.array(ds.y)

    if hasattr(ds, "categorical_indicator") and ds.categorical_indicator:
        cat_mask = np.array(ds.categorical_indicator)
        if cat_mask.any():
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[:, cat_mask] = enc.fit_transform(X[:, cat_mask])

    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_previous_results(path):
    """Load previous results from parquet, returning (df, lookup_dict).

    The lookup maps (dataset, model, task) -> row dict for quick status checks.
    """
    if not path.exists():
        return pd.DataFrame(), {}

    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(), {}

    lookup = {}
    for _, row in df.iterrows():
        key = (row["dataset"], row["model"], row["task"])
        lookup[key] = row.to_dict()
    return df, lookup


def _print_cached_result(model_name, row, task):
    """Print a cached result inline (same format as a live result)."""
    if task == "classification":
        acc = row.get("accuracy", 0)
        auc = row.get("roc_auc")
        mcc = row.get("mcc", 0)
        auc_str = f"  auc={auc:.4f}" if auc is not None and pd.notna(auc) else ""
        mcc_str = f"  mcc={mcc:.4f}" if pd.notna(mcc) else ""
        t = row.get("fit_time_mean_s", 0)
        print(f"acc={acc:.4f}{auc_str}{mcc_str}  ({t:.1f}s) [cached]")
    else:
        r2 = row.get("r2", 0)
        rmse = row.get("rmse", 0)
        t = row.get("fit_time_mean_s", 0)
        print(f"R²={r2:.4f}  rmse={rmse:.4f}  ({t:.1f}s) [cached]")


def main():
    parser = argparse.ArgumentParser(description="Endgame Tabular Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 datasets)")
    parser.add_argument(
        "--suite",
        default="grinsztajn",
        choices=["grinsztajn", "grinsztajn-classif", "grinsztajn-regression", "uci-popular"],
        help="Dataset suite (default: full Grinsztajn NeurIPS 2022, ~45 datasets)",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-model timeout in seconds (default: 300)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore previous results and start from scratch")
    args = parser.parse_args()

    print("=" * 70)
    print("Endgame Tabular Model Benchmark")
    print("=" * 70)

    # Load datasets
    print(f"\nLoading datasets (suite={args.suite})...")
    datasets = load_datasets(
        suite=args.suite, max_samples=args.max_samples, quick=args.quick
    )
    print(f"Loaded {len(datasets)} datasets")

    # Resume support: load previous results
    output_path = RESULTS_DIR / "tabular_benchmark.parquet"
    if args.fresh:
        prev_lookup = {}
        results = []
    else:
        prev_df, prev_lookup = _load_previous_results(output_path)
        # Carry forward successful results; drop failed/error/timeout for retry
        results = []
        if len(prev_df) > 0:
            keep = prev_df[prev_df["status"] == "success"]
            results = keep.to_dict("records")
            n_success = len(keep)
            n_retry = len(prev_df) - n_success
            print(f"\nResuming: {n_success} cached successes, "
                  f"{n_retry} failed/error/timeout to retry")

    n_datasets = len(datasets)
    n_skipped = 0
    n_ran = 0

    def _save_incremental():
        """Flush current results to parquet so nothing is lost on crash."""
        if results:
            pd.DataFrame(results).to_parquet(output_path, index=False)

    seen_datasets = set()
    for ds_idx, ds in enumerate(datasets, 1):
        X, y = preprocess_dataset(ds)
        ds_name = ds.name
        # Use task type from SuiteLoader (OpenML metadata); fall back to heuristic
        if hasattr(ds, "task_type") and ds.task_type is not None:
            task = "regression" if ds.task_type.value == "regression" else "classification"
        else:
            task = "classification" if is_classification(y) else "regression"

        # Skip duplicate (dataset, task) pairs from overlapping suites
        ds_key = (ds_name, task)
        if ds_key in seen_datasets:
            continue
        seen_datasets.add(ds_key)

        models = get_models(quick=args.quick, task=task)

        print(f"\n{'=' * 70}")
        print(f"[{ds_idx}/{n_datasets}] {ds_name} [{task}]"
              f" ({X.shape[0]} samples, {X.shape[1]} features, {len(models)} models)")
        print("=" * 70)

        evaluate_fn = (
            evaluate_model_classification if task == "classification"
            else evaluate_model_regression
        )

        for model_name, model in models.items():
            key = (ds_name, model_name, task)
            prev = prev_lookup.get(key)
            print(f"  {model_name:25s}", end=" ", flush=True)

            # Skip if previous run succeeded
            if prev is not None and prev.get("status") == "success":
                _print_cached_result(model_name, prev, task)
                n_skipped += 1
                continue

            # Note previous timeout without retrying
            if prev is not None and prev.get("status") == "timeout":
                print(f"SKIP    (timed out last run: {prev.get('error', '')})")
                results.append({
                    "dataset": ds_name, "model": model_name, "task": task,
                    "n_samples": X.shape[0], "n_features": X.shape[1],
                    "status": "timeout", "error": prev.get("error", ""),
                })
                n_skipped += 1
                _save_incremental()
                continue

            # Retry failed/error, or run for first time
            if prev is not None:
                print(f"[retry] ", end="", flush=True)

            n_ran += 1
            t0 = time.time()
            try:
                metrics = run_with_timeout(
                    evaluate_fn, (model, X, y, args.cv_folds), args.timeout
                )
                elapsed = time.time() - t0

                if "error" in metrics:
                    print(f"FAILED  ({metrics['error'][:60]})")
                    results.append({
                        "dataset": ds_name, "model": model_name, "task": task,
                        "n_samples": X.shape[0], "n_features": X.shape[1],
                        "status": "failed", "error": metrics["error"],
                    })
                else:
                    if task == "classification":
                        acc = metrics.get("accuracy", 0)
                        auc = metrics.get("roc_auc")
                        mcc = metrics.get("mcc", 0)
                        auc_str = f"  auc={auc:.4f}" if auc is not None else ""
                        print(f"acc={acc:.4f}{auc_str}  mcc={mcc:.4f}  ({elapsed:.1f}s)")
                    else:
                        r2 = metrics.get("r2", 0)
                        rmse = metrics.get("rmse", 0)
                        print(f"R²={r2:.4f}  rmse={rmse:.4f}  ({elapsed:.1f}s)")
                    results.append({
                        "dataset": ds_name, "model": model_name, "task": task,
                        "n_samples": X.shape[0], "n_features": X.shape[1],
                        "status": "success", **metrics,
                    })
            except TimeoutError:
                elapsed = time.time() - t0
                print(f"TIMEOUT ({elapsed:.0f}s)")
                results.append({
                    "dataset": ds_name, "model": model_name, "task": task,
                    "n_samples": X.shape[0], "n_features": X.shape[1],
                    "status": "timeout", "error": f"Exceeded {args.timeout}s",
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR   ({str(e)[:60]})")
                results.append({
                    "dataset": ds_name, "model": model_name, "task": task,
                    "n_samples": X.shape[0], "n_features": X.shape[1],
                    "status": "error", "error": str(e),
                })

            _save_incremental()

    # Final save
    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"{n_skipped} cached, {n_ran} evaluated this run")

    # Print summary
    successful = df[df["status"] == "success"]
    failed = df[df["status"].isin(["failed", "error", "timeout"])]
    print(f"{len(successful)} successful, {len(failed)} failed/timeout")

    if len(successful) > 0:
        # Classification summary
        classif = successful[successful["task"] == "classification"]
        if len(classif) > 0:
            print("\n" + "=" * 90)
            print("CLASSIFICATION RESULTS (sorted by mean accuracy)")
            print("=" * 90)

            agg_cols = {"accuracy": ["mean", "std"]}
            if "roc_auc" in classif.columns:
                agg_cols["roc_auc"] = ["mean"]
            if "mcc" in classif.columns:
                agg_cols["mcc"] = ["mean"]
            if "fit_time_mean_s" in classif.columns:
                agg_cols["fit_time_mean_s"] = ["mean"]

            summary = classif.groupby("model").agg(agg_cols)
            summary.columns = ["_".join(c).strip("_") for c in summary.columns]
            summary["n_datasets"] = classif.groupby("model")["accuracy"].count()
            summary = summary.sort_values("accuracy_mean", ascending=False)

            header = (
                f"  {'Model':25s}  {'Acc':>10s}  {'AUC':>7s}  {'MCC':>7s}"
                f"  {'Time(s)':>8s}  {'N':>3s}"
            )
            print(header)
            print("  " + "-" * len(header.strip()))
            for model_name, row in summary.iterrows():
                acc = f"{row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.3f}"
                auc = f"{row.get('roc_auc_mean', float('nan')):.4f}" if "roc_auc_mean" in row and pd.notna(row.get("roc_auc_mean")) else "   -  "
                mcc = f"{row.get('mcc_mean', float('nan')):.4f}" if "mcc_mean" in row and pd.notna(row.get("mcc_mean")) else "   -  "
                t = f"{row.get('fit_time_mean_s_mean', 0):.1f}" if "fit_time_mean_s_mean" in row else "-"
                n = int(row["n_datasets"])
                print(f"  {model_name:25s}  {acc:>10s}  {auc:>7s}  {mcc:>7s}  {t:>8s}  {n:>3d}")

            try:
                from scipy.stats import friedmanchisquare

                pivot = classif.pivot(
                    index="dataset", columns="model", values="accuracy"
                ).dropna(axis=1)
                if pivot.shape[1] >= 3:
                    stat, p = friedmanchisquare(
                        *[pivot[col].values for col in pivot.columns]
                    )
                    print(f"\n  Friedman test: chi2={stat:.2f}, p={p:.4f}")
            except Exception:
                pass

        # Regression summary
        reg = successful[successful["task"] == "regression"]
        if len(reg) > 0:
            print("\n" + "=" * 90)
            print("REGRESSION RESULTS (sorted by mean R²)")
            print("=" * 90)

            agg_cols = {"r2": ["mean", "std"]}
            if "rmse" in reg.columns:
                agg_cols["rmse"] = ["mean"]
            if "mae" in reg.columns:
                agg_cols["mae"] = ["mean"]
            if "fit_time_mean_s" in reg.columns:
                agg_cols["fit_time_mean_s"] = ["mean"]

            summary = reg.groupby("model").agg(agg_cols)
            summary.columns = ["_".join(c).strip("_") for c in summary.columns]
            summary["n_datasets"] = reg.groupby("model")["r2"].count()
            summary = summary.sort_values("r2_mean", ascending=False)

            header = (
                f"  {'Model':25s}  {'R²':>12s}  {'RMSE':>8s}  {'MAE':>8s}"
                f"  {'Time(s)':>8s}  {'N':>3s}"
            )
            print(header)
            print("  " + "-" * len(header.strip()))
            for model_name, row in summary.iterrows():
                r2 = f"{row['r2_mean']:.4f}+/-{row['r2_std']:.3f}"
                rmse = f"{row.get('rmse_mean', float('nan')):.4f}" if "rmse_mean" in row and pd.notna(row.get("rmse_mean")) else "   -   "
                mae = f"{row.get('mae_mean', float('nan')):.4f}" if "mae_mean" in row and pd.notna(row.get("mae_mean")) else "   -   "
                t = f"{row.get('fit_time_mean_s_mean', 0):.1f}" if "fit_time_mean_s_mean" in row else "-"
                n = int(row["n_datasets"])
                print(f"  {model_name:25s}  {r2:>12s}  {rmse:>8s}  {mae:>8s}  {t:>8s}  {n:>3d}")

            try:
                from scipy.stats import friedmanchisquare

                pivot = reg.pivot(
                    index="dataset", columns="model", values="r2"
                ).dropna(axis=1)
                if pivot.shape[1] >= 3:
                    stat, p = friedmanchisquare(
                        *[pivot[col].values for col in pivot.columns]
                    )
                    print(f"\n  Friedman test: chi2={stat:.2f}, p={p:.4f}")
            except Exception:
                pass

    print("\nDone.")


if __name__ == "__main__":
    main()
