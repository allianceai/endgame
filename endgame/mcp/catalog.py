"""Catalog of Endgame's modules for agents: what each is for, how to reach it over MCP, and its API.

The purposes are written by hand; members, signatures and docstrings are read from the code, so they
stay current. Modules are imported only when their members are asked for.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

# module -> (what it is for, the MCP tools that use it). Every module is also reachable through
# transform_data (transformers), train_model / compare_models (estimators by class path) and run_python.
MODULES: dict[str, tuple[str, list[str]]] = {
    "validation": ("CV that matches how data was collected: grouped, stratified-grouped, purged time-series and "
                   "panel splits, nested CV, out-of-fold helpers, adversarial validation for train/test drift",
                   ["split_data", "train_model(time_ordered=True)", "run_python"]),
    "guardrails": ("Leakage and data-quality checks before training: target leakage, ID-like and constant "
                   "columns, train/test overlap", ["check_data_quality"]),
    "data_quality": ("Profiling, duplicate detection, feature drift between datasets, data valuation",
                     ["run_python"]),
    "preprocessing": ("Fold-safe encoders (target, CatBoost, leave-one-out, frequency), imputers (KNN, MICE, "
                      "MissForest), group aggregations, interactions, ranks, lag/rolling/temporal features, "
                      "18 class-balancing samplers, noise filters, ComBat/site harmonisation",
                      ["engineer_features", "preprocess", "transform_data"]),
    "feature_selection": ("19 selectors: filters (mutual information, F-test, chi2, MRMR, ReliefF, correlation), "
                          "wrappers (RFE, sequential, genetic, Boruta), importance-based (permutation, SHAP, "
                          "null importance, tree), stability selection, knockoffs (FDR control), adversarial",
                          ["select_features", "transform_data"]),
    "dimensionality_reduction": ("PCA variants, kernel PCA, ICA, truncated SVD, UMAP, TriMAP, PHATE, PaCMAP, VAE",
                                 ["transform_data"]),
    "signal": ("Signal processing for sensor/time-series columns: filters, FFT/Welch/multitaper spectra, band "
               "power, wavelets, entropy (sample, permutation, spectral, SVD), fractal dimension, Hurst, DFA, "
               "Hjorth, peaks, zero crossings", ["engineer_features(aggregate)", "transform_data"]),
    "timeseries": ("Forecasting (naive, theta, ETS, ARIMA, MSTL, N-BEATS, N-HiTS, TFT, PatchTST), time-series "
                   "CV, ROCKET/MiniRocket/Hydra classification, tsfresh features, forecast metrics",
                   ["forecast", "transform_data", "run_python"]),
    "models": ("90+ estimators: GBDT wrappers (LightGBM, XGBoost, CatBoost), rotation/oblique/honest forests, "
               "C5.0, Cubist, EBM, MARS, RuleFit, FURIA, Bayesian network classifiers, NGBoost, BART, GPs, "
               "SVMs, MLPs, FT-Transformer, SAINT, TabM, RealMLP, and tabular foundation models (Kumo-Tabular, "
               "LimiX, TabPFN, TabICL, Mitra, ...)", ["list_models", "recommend_models", "train_model",
                                                      "compare_models"]),
    "automl": ("AutoML predictors (tabular, text, vision, time series, audio, multimodal), the model registry "
               "and presets", ["automl", "list_models", "recommend_models"]),
    "quick": ("One-line classify / regress / compare", ["quick_compare"]),
    "ensemble": ("Hill climbing, stacking, blending, optimized and rank-average blends, super learner, Bayesian "
                 "model averaging, threshold optimisation, distillation, multi-output wrappers",
                 ["ensemble"]),
    "tune": ("Optuna hyperparameter search with competition search spaces", ["run_python"]),
    "calibration": ("Conformal prediction (sets and intervals, CQR), temperature/Platt/beta/isotonic scaling, "
                    "Venn-ABERS, calibration metrics", ["run_python"]),
    "explain": ("SHAP, LIME, partial dependence, interaction strength (H-statistic), counterfactuals",
                ["explain_model", "create_visualization"]),
    "fairness": ("Group fairness metrics, reweighing, fairness-constrained training, post-processing, reports",
                 ["run_python"]),
    "anomaly": ("Isolation forests, LOF, GritBot (rule-based, explains why), PyOD's 39 detectors",
                ["detect_anomalies", "transform_data"]),
    "clustering": ("16 clustering algorithms with automatic selection", ["cluster"]),
    "semi_supervised": ("Self-training on unlabelled rows", ["train_model", "run_python"]),
    "survival": ("Time-to-event models (Cox, AFT, random survival forests, DeepSurv/DeepHit), competing risks, "
                 "concordance and Brier metrics", ["run_python"]),
    "ranking": ("Grouped cross-sectional ranking (scores within a group, e.g. per day or per position)",
                ["run_python"]),
    "fuzzy": ("Fuzzy inference, neuro-fuzzy (ANFIS), evolving and type-2 fuzzy systems", ["train_model",
                                                                                        "run_python"]),
    "nlp": ("Transformer classifiers/regressors, domain-adaptive pretraining, pseudo-labelling, LLM wrappers, "
            "text cleaning, translation and text-generation metrics", ["run_python"]),
    "vision": ("timm backbones, test-time augmentation, weighted boxes fusion, segmentation, augmentation",
               ["run_python"]),
    "audio": ("Spectrograms, PCEN, sound-event detection, audio augmentation, pretrained audio classifiers",
              ["run_python"]),
    "benchmark": ("Benchmark suites (OpenML), meta-features, learning curves, synthetic control datasets",
                  ["run_python"]),
    "utils": ("Bootstrap CIs, paired model comparison, DeLong test, decision curves, batch-leakage check, "
              "competition metrics, seeding, Sharpe-ratio analysis", ["run_python"]),
    "visualization": ("42 interactive HTML chart types (model evaluation and data exploration)",
                      ["create_visualization", "create_report"]),
    "persistence": ("Save/load models with metadata, ONNX export, model serving", ["save_model"]),
    "tracking": ("Experiment logging (console, MLflow)", ["run_python"]),
    "kaggle": ("Kaggle competitions: details and pages, data download, public notebooks, pushing and running "
               "notebooks", ["kaggle_competition", "kaggle_download", "kaggle_notebooks"]),
}


def overview() -> list[dict[str, Any]]:
    """Every module with its purpose and the MCP tools that reach it (no imports)."""
    return [{"module": f"endgame.{name}", "purpose": purpose, "mcp_tools": tools}
            for name, (purpose, tools) in MODULES.items()]


def _summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0][:160] if doc.strip() else ""


def _role(obj: Any) -> str:
    if not inspect.isclass(obj):
        return "function" if callable(obj) else "constant"
    names = {c.__name__ for c in inspect.getmro(obj)}
    if "ClassifierMixin" in names or obj.__name__.endswith("Classifier"):
        return "classifier"
    if "RegressorMixin" in names or obj.__name__.endswith("Regressor"):
        return "regressor"
    if hasattr(obj, "transform") or hasattr(obj, "fit_transform"):
        return "transformer"
    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        return "estimator"
    return "class"


def members(module: str) -> list[dict[str, Any]]:
    """Public classes and functions of one module, with role and one-line summary."""
    name = module.removeprefix("endgame.")
    mod = importlib.import_module(f"endgame.{name}")
    exported = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]
    out = []
    for attr in exported:
        obj = getattr(mod, attr, None)
        if obj is None or inspect.ismodule(obj):
            continue
        out.append({"name": f"endgame.{name}.{attr}", "role": _role(obj), "summary": _summary(obj)})
    return out


def search(query: str, limit: int = 40) -> list[dict[str, Any]]:
    """Members of every module whose name or summary contains each word of ``query``."""
    words = query.lower().split()
    hits = []
    for name, (purpose, _) in MODULES.items():
        if all(w in purpose.lower() or w in name for w in words):
            hits.append({"name": f"endgame.{name}", "role": "module", "summary": purpose[:160]})
        try:
            for m in members(name):
                if all(w in f"{m['name']} {m['summary']}".lower() for w in words):
                    hits.append(m)
        except Exception:  # an optional dependency of that module is missing
            continue
    return hits[:limit]


def resolve(path: str) -> Any:
    """The object at 'endgame.module.Name' (a bare 'Name' is looked up across the catalog's modules)."""
    if "." in path:
        mod, attr = path.rsplit(".", 1)
        try:
            return getattr(importlib.import_module(mod), attr)
        except ModuleNotFoundError:
            if mod.startswith("endgame"):
                raise
            return getattr(importlib.import_module(f"endgame.{mod}"), attr)  # "feature_selection.BorutaSelector"
    for name in MODULES:
        try:
            mod = importlib.import_module(f"endgame.{name}")
        except Exception:
            continue
        if hasattr(mod, path):
            return getattr(mod, path)
    raise KeyError(f"'{path}' not found in endgame; use list_modules(search=...)")


def describe(path: str, max_doc_chars: int = 4000) -> dict[str, Any]:
    """Signature, role, methods and docstring of a class or function."""
    obj = resolve(path)
    try:
        signature = f"{getattr(obj, '__name__', path)}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        signature = getattr(obj, "__name__", path)
    doc = inspect.getdoc(obj) or ""
    out = {
        "name": f"{getattr(obj, '__module__', '')}.{getattr(obj, '__name__', path)}",
        "role": _role(obj),
        "signature": signature,
        "doc": doc[:max_doc_chars] + (" ..." if len(doc) > max_doc_chars else ""),
    }
    if inspect.isclass(obj):
        out["methods"] = [m for m in ("fit", "transform", "fit_transform", "predict", "predict_proba",
                                      "score", "fit_resample", "explain") if hasattr(obj, m)]
    return out
