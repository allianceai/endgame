#!/usr/bin/env python
"""
Experiment 2: Calibration Benchmark

Measures the effect of 6 calibration methods on 3 base models across
15 binary classification datasets.

Usage:
    python benchmarks/run_calibration_benchmark.py          # Full run
    python benchmarks/run_calibration_benchmark.py --quick   # Quick test
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
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


def get_base_models():
    """Return base models to calibrate."""
    import endgame as eg
    from sklearn.ensemble import RandomForestClassifier

    return {
        "CatBoost": eg.models.CatBoostWrapper(preset="fast"),
        "EBM": eg.models.EBMClassifier(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }


def get_calibration_methods():
    """Return calibration methods to test."""
    from endgame.calibration import (
        BetaCalibration,
        HistogramBinning,
        IsotonicCalibration,
        PlattScaling,
        TemperatureScaling,
        VennABERS,
    )

    return {
        "Platt": PlattScaling,
        "Isotonic": IsotonicCalibration,
        "Beta": BetaCalibration,
        "Temperature": TemperatureScaling,
        "Histogram": HistogramBinning,
        "VennABERS": VennABERS,
    }


def load_binary_datasets(quick=False, max_samples=5000):
    """Load binary classification datasets."""
    from endgame.benchmark import SuiteLoader

    datasets = []

    if quick:
        loader = SuiteLoader("quick-test", max_samples=max_samples)
    else:
        # Use Grinsztajn classification suites for more binary datasets
        loader = SuiteLoader("grinsztajn-classif", max_samples=max_samples)

    seen_names = set()
    for ds in loader.load():
        # Skip duplicates from overlapping suites
        if ds.name in seen_names:
            continue
        seen_names.add(ds.name)

        # Only binary classification
        le = LabelEncoder()
        y = le.fit_transform(ds.y)
        if len(np.unique(y)) == 2:
            datasets.append((ds, y))

        if quick and len(datasets) >= 3:
            break

    return datasets


def preprocess_dataset(ds):
    """Preprocess: encode categoricals, handle NaNs."""
    X = ds.X.copy() if hasattr(ds.X, "copy") else np.array(ds.X, dtype=np.float32)

    if hasattr(ds, "categorical_indicator") and ds.categorical_indicator:
        cat_mask = np.array(ds.categorical_indicator)
        if cat_mask.any():
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[:, cat_mask] = enc.fit_transform(X[:, cat_mask])

    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X


def evaluate_calibration(base_model, cal_methods, X, y, random_state=42):
    """Evaluate all calibration methods on one base model + dataset."""
    from sklearn.base import clone

    from endgame.calibration import ConformalClassifier

    # 60/20/20 split: train / calibration / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=random_state, stratify=y_trainval
    )

    # Train base model
    try:
        model = clone(base_model)
    except Exception:
        model = base_model.__class__(**base_model.get_params())

    model.fit(X_train, y_train)

    # Get raw probabilities
    raw_proba = model.predict_proba(X_test)[:, 1]
    cal_proba_on_cal = model.predict_proba(X_cal)[:, 1]

    results = []

    # Uncalibrated baseline
    results.append(
        {
            "calibration_method": "Uncalibrated",
            "ece": expected_calibration_error(y_test, raw_proba),
            "brier_score": brier_score_loss(y_test, raw_proba),
            "log_loss": log_loss(y_test, np.column_stack([1 - raw_proba, raw_proba])),
        }
    )

    # Apply each calibration method
    for cal_name, cal_class in cal_methods.items():
        try:
            if cal_name == "VennABERS":
                calibrator = cal_class(model)
                calibrator.fit(X_train, y_train, X_cal, y_cal)
                cal_proba = calibrator.predict_proba(X_test)[:, 1]
            elif cal_name == "Temperature":
                # Temperature scaling works on logits (1D for binary)
                calibrator = cal_class()
                logits_cal = np.log(
                    np.clip(cal_proba_on_cal, 1e-10, 1 - 1e-10)
                    / np.clip(1 - cal_proba_on_cal, 1e-10, 1 - 1e-10)
                )
                logits_test = np.log(
                    np.clip(raw_proba, 1e-10, 1 - 1e-10)
                    / np.clip(1 - raw_proba, 1e-10, 1 - 1e-10)
                )
                calibrator.fit(logits_cal, y_cal)
                cal_proba = calibrator.transform(logits_test)[:, 1]
            else:
                calibrator = cal_class()
                calibrator.fit(cal_proba_on_cal, y_cal)
                cal_proba = calibrator.transform(raw_proba)

            cal_proba = np.clip(cal_proba, 1e-10, 1 - 1e-10)

            results.append(
                {
                    "calibration_method": cal_name,
                    "ece": expected_calibration_error(y_test, cal_proba),
                    "brier_score": brier_score_loss(y_test, cal_proba),
                    "log_loss": log_loss(
                        y_test, np.column_stack([1 - cal_proba, cal_proba])
                    ),
                }
            )
        except Exception as e:
            results.append(
                {
                    "calibration_method": cal_name,
                    "ece": np.nan,
                    "brier_score": np.nan,
                    "log_loss": np.nan,
                    "error": str(e),
                }
            )

    # Conformal prediction coverage
    try:
        conf = ConformalClassifier(model, method="lac", alpha=0.05)
        conf.fit(X_train, y_train, X_cal, y_cal)
        coverage = conf.coverage_score(X_test, y_test)
        pred_sets = conf.predict(X_test)
        avg_set_size = np.mean([len(s) for s in pred_sets])
        results.append(
            {
                "calibration_method": "Conformal (LAC)",
                "coverage_at_95": coverage,
                "avg_set_size": avg_set_size,
            }
        )
    except Exception as e:
        results.append(
            {
                "calibration_method": "Conformal (LAC)",
                "coverage_at_95": np.nan,
                "error": str(e),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Endgame Calibration Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 datasets)")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    print("=" * 70)
    print("Endgame Calibration Benchmark")
    print("=" * 70)

    # Load datasets
    print("\nLoading binary classification datasets...")
    dataset_pairs = load_binary_datasets(quick=args.quick, max_samples=args.max_samples)
    print(f"Loaded {len(dataset_pairs)} binary datasets")

    # Load models and calibration methods
    base_models = get_base_models()
    cal_methods = get_calibration_methods()
    print(f"Base models: {', '.join(base_models.keys())}")
    print(f"Calibration methods: {', '.join(cal_methods.keys())}")

    # Run benchmark
    all_results = []
    total = len(dataset_pairs) * len(base_models)
    completed = 0

    for ds, y_binary in dataset_pairs:
        X = preprocess_dataset(ds)
        ds_name = ds.name
        print(f"\n--- {ds_name} ({X.shape[0]} samples) ---")

        for model_name, model in base_models.items():
            completed += 1
            print(f"  [{completed}/{total}] {model_name}...", end=" ", flush=True)

            t0 = time.time()
            try:
                results = evaluate_calibration(model, cal_methods, X, y_binary)
                elapsed = time.time() - t0

                for r in results:
                    r["dataset"] = ds_name
                    r["base_model"] = model_name
                    r["n_samples"] = X.shape[0]
                    all_results.append(r)

                # Print uncalibrated ECE
                uncal = [
                    r for r in results if r["calibration_method"] == "Uncalibrated"
                ]
                if uncal:
                    print(f"ECE={uncal[0]['ece']:.4f} ({elapsed:.1f}s)")
                else:
                    print(f"({elapsed:.1f}s)")
            except Exception as e:
                print(f"ERROR: {str(e)[:60]}")

    # Save results
    df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / "calibration_benchmark.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    cal_rows = df[df["calibration_method"] != "Conformal (LAC)"].dropna(subset=["ece"])
    if len(cal_rows) > 0:
        print("\n" + "=" * 70)
        print("SUMMARY: Mean ECE by calibration method")
        print("=" * 70)
        summary = (
            cal_rows.groupby("calibration_method")["ece"]
            .agg(["mean", "std"])
            .sort_values("mean")
        )
        for method, row in summary.iterrows():
            print(f"  {method:20s}  ECE={row['mean']:.4f} ± {row['std']:.4f}")

        # ECE reduction
        uncal_ece = cal_rows[cal_rows["calibration_method"] == "Uncalibrated"][
            "ece"
        ].mean()
        best_ece = summary["mean"].min()
        if uncal_ece > 0:
            reduction = (uncal_ece - best_ece) / uncal_ece * 100
            print(f"\nBest calibration reduces ECE by {reduction:.1f}% on average")

    # Conformal coverage
    conf_rows = df[df["calibration_method"] == "Conformal (LAC)"].dropna(
        subset=["coverage_at_95"]
    )
    if len(conf_rows) > 0:
        mean_cov = conf_rows["coverage_at_95"].mean()
        print(f"\nConformal prediction: mean coverage = {mean_cov:.1%} (target: 95%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
