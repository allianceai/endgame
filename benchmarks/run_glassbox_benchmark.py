#!/usr/bin/env python
"""
Experiment 3: Glass-Box vs Black-Box Benchmark

Compares interpretable models against black-box GBDTs to measure
the accuracy-interpretability trade-off.

Usage:
    python benchmarks/run_glassbox_benchmark.py          # Full run
    python benchmarks/run_glassbox_benchmark.py --quick   # Quick test
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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


def get_models(quick=False):
    """Return dict of (name, model, is_glassbox)."""
    import endgame as eg
    from sklearn.ensemble import RandomForestClassifier

    models = {}

    # Black-box models
    models["LightGBM"] = (eg.models.LGBMWrapper(preset="endgame"), False)
    models["XGBoost"] = (eg.models.XGBWrapper(preset="endgame"), False)
    models["RandomForest"] = (
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        False,
    )

    if not quick:
        models["CatBoost"] = (eg.models.CatBoostWrapper(preset="endgame"), False)

    # Glass-box models
    models["EBM"] = (eg.models.EBMClassifier(), True)

    if quick:
        return models

    try:
        models["MARS"] = (eg.models.MARSClassifier(), True)
    except Exception:
        pass
    try:
        models["RuleFit"] = (eg.models.RuleFitClassifier(), True)
    except Exception:
        pass
    try:
        models["GAM"] = (eg.models.interpretable.GAMClassifier(), True)
    except Exception:
        pass
    try:
        models["NAM"] = (eg.models.NAMClassifier(n_epochs=50), True)
    except Exception:
        pass
    try:
        models["TAN"] = (eg.models.TANClassifier(), True)
    except Exception:
        pass
    try:
        models["C5.0"] = (eg.models.C50Classifier(), True)
    except Exception:
        pass
    try:
        models["GOSDT"] = (eg.models.interpretable.GOSDTClassifier(), True)
    except Exception:
        pass
    try:
        models["CORELS"] = (eg.models.interpretable.CORELSClassifier(), True)
    except Exception:
        pass
    try:
        models["SLIM"] = (eg.models.interpretable.SLIMClassifier(), True)
    except Exception:
        pass
    try:
        models["FasterRisk"] = (eg.models.interpretable.FasterRiskClassifier(), True)
    except Exception:
        pass
    try:
        models["NodeGAM"] = (eg.models.interpretable.NodeGAMClassifier(), True)
    except Exception:
        pass

    return models


def load_datasets(quick=False, max_samples=5000):
    """Load benchmark datasets."""
    from endgame.benchmark import SuiteLoader

    if quick:
        loader = SuiteLoader("quick-test", max_samples=max_samples)
    else:
        loader = SuiteLoader("grinsztajn-classif", max_samples=max_samples)

    return list(loader.load())


def preprocess_dataset(ds):
    """Preprocess: encode categoricals, handle NaNs."""
    X = ds.X.copy() if hasattr(ds.X, "copy") else np.array(ds.X, dtype=np.float32)
    y = ds.y.copy() if hasattr(ds.y, "copy") else np.array(ds.y)

    if hasattr(ds, "categorical_indicator") and ds.categorical_indicator:
        cat_mask = np.array(ds.categorical_indicator)
        if cat_mask.any():
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[:, cat_mask] = enc.fit_transform(X[:, cat_mask])

    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X, y


def evaluate_model(model, X, y, n_splits=5):
    """Evaluate a model with stratified k-fold CV."""
    from sklearn.base import clone

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    is_binary = len(le.classes_) == 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fit_times = []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        try:
            try:
                m = clone(model)
            except Exception:
                m = model.__class__(**model.get_params())

            t0 = time.time()
            m.fit(X_train, y_train)
            fit_times.append(time.time() - t0)

            y_pred = m.predict(X_test)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            if hasattr(m, "predict_proba"):
                all_y_proba.extend(m.predict_proba(X_test))
        except Exception as e:
            return {"error": str(e)}

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    metrics = {
        "accuracy": accuracy_score(all_y_true, all_y_pred),
        "balanced_accuracy": balanced_accuracy_score(all_y_true, all_y_pred),
        "f1_weighted": f1_score(all_y_true, all_y_pred, average="weighted"),
        "fit_time_s": np.mean(fit_times),
    }

    if all_y_proba:
        all_y_proba = np.array(all_y_proba)
        try:
            if is_binary:
                proba_pos = all_y_proba[:, 1]
                metrics["brier_score"] = brier_score_loss(all_y_true, proba_pos)
                metrics["ece"] = expected_calibration_error(all_y_true, proba_pos)
                metrics["roc_auc"] = roc_auc_score(all_y_true, proba_pos)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    all_y_true, all_y_proba, multi_class="ovr", average="weighted"
                )
        except Exception:
            pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Endgame Glass-Box Benchmark")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("Endgame Glass-Box vs Black-Box Benchmark")
    print("=" * 70)

    datasets = load_datasets(quick=args.quick, max_samples=args.max_samples)
    print(f"Loaded {len(datasets)} datasets")

    model_dict = get_models(quick=args.quick)
    print(f"Loaded {len(model_dict)} models")
    for name, (_, is_gb) in model_dict.items():
        tag = "[glass-box]" if is_gb else "[black-box]"
        print(f"  {tag} {name}")

    all_results = []
    total = len(datasets) * len(model_dict)
    completed = 0

    for ds in datasets:
        X, y = preprocess_dataset(ds)
        ds_name = ds.name
        print(f"\n--- {ds_name} ({X.shape[0]} x {X.shape[1]}) ---")

        for model_name, (model, is_glassbox) in model_dict.items():
            completed += 1
            print(f"  [{completed}/{total}] {model_name}...", end=" ", flush=True)

            t0 = time.time()
            try:
                metrics = evaluate_model(model, X, y, n_splits=args.cv_folds)
                elapsed = time.time() - t0

                if "error" in metrics:
                    print(f"FAILED ({metrics['error'][:50]})")
                    all_results.append(
                        {
                            "dataset": ds_name,
                            "model": model_name,
                            "is_glassbox": is_glassbox,
                            "status": "failed",
                            "error": metrics["error"],
                        }
                    )
                else:
                    acc = metrics.get("accuracy", 0)
                    print(f"acc={acc:.4f} ({elapsed:.1f}s)")
                    all_results.append(
                        {
                            "dataset": ds_name,
                            "model": model_name,
                            "is_glassbox": is_glassbox,
                            "status": "success",
                            **metrics,
                        }
                    )
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                all_results.append(
                    {
                        "dataset": ds_name,
                        "model": model_name,
                        "is_glassbox": is_glassbox,
                        "status": "error",
                        "error": str(e),
                    }
                )

    # Save results
    df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / "glassbox_benchmark.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary
    successful = df[df["status"] == "success"]
    if len(successful) > 0:
        print("\n" + "=" * 70)
        print("SUMMARY: Glass-box vs Black-box accuracy")
        print("=" * 70)

        for is_gb in [False, True]:
            label = "Glass-box" if is_gb else "Black-box"
            subset = successful[successful["is_glassbox"] == is_gb]
            if len(subset) == 0:
                continue

            print(f"\n  {label} models:")
            model_means = (
                subset.groupby("model")["accuracy"]
                .agg(["mean", "std"])
                .sort_values("mean", ascending=False)
            )
            for model_name, row in model_means.iterrows():
                print(f"    {model_name:20s}  {row['mean']:.4f} ± {row['std']:.4f}")

        # Overall comparison
        bb_acc = successful[~successful["is_glassbox"]]["accuracy"].mean()
        gb_acc = successful[successful["is_glassbox"]]["accuracy"].mean()
        if bb_acc > 0:
            pct = gb_acc / bb_acc * 100
            print(f"\n  Glass-box achieves {pct:.1f}% of black-box accuracy on average")

    print("\nDone.")


if __name__ == "__main__":
    main()
