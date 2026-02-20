#!/usr/bin/env python
"""Assess all interpretable models in the Endgame AutoML pipeline.

Trains each interpretable model individually, displays learned structures
(rules, equations, scorecards, etc.), and saves results to parquet.

Usage
-----
    python scripts/run_interpretable.py [--subsample N] [--max-model-time S]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FT
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from endgame.automl.display import display_model

OUTPUT_DIR = Path("automl_interpretable_output")
PARQUET_FILE = Path("benchmark_results.parquet")

# ── Models to assess ────────────────────────────────────────────────────────
# (display_name, module_path, class_name, init_kwargs)
INTERPRETABLE_MODELS: list[tuple[str, str, str, dict]] = [
    ("EBM", "endgame.models.ebm", "EBMClassifier", {"interactions": 10}),
    ("GAM", "endgame.models.interpretable.gam", "GAMClassifier",
     {"n_splines": 10, "lam_search": 5}),
    ("NAM", "endgame.models.tabular.nam", "NAMClassifier",
     {"n_epochs": 50, "learning_rate": 0.02}),
    ("NODE-GAM", "endgame.models.interpretable.node_gam", "NodeGAMClassifier",
     {"n_epochs": 30}),
    ("GAMI-Net", "endgame.models.interpretable.gami_net", "GAMINetClassifier",
     {"n_epochs": 30}),
    ("RuleFit", "endgame.models.rules.rulefit", "RuleFitClassifier", {}),
    ("FURIA", "endgame.models.rules.furia", "FURIAClassifier", {}),
    ("CORELS", "endgame.models.interpretable.corels", "CORELSClassifier", {}),
    ("SLIM", "endgame.models.interpretable.slim", "SLIMClassifier", {}),
    ("FasterRisk", "endgame.models.interpretable.slim", "FasterRiskClassifier", {}),
    ("MARS", "endgame.models.linear.mars", "MARSClassifier", {}),
    ("Linear", "endgame.models.baselines.linear", "LinearClassifier", {}),
    ("C5.0", "endgame.models.trees.c50", "C50Classifier", {"use_rust": False}),
    ("GOSDT", "endgame.models.interpretable.gosdt", "GOSDTClassifier", {}),
    ("Symbolic", "endgame.models.symbolic", "SymbolicClassifier",
     {"preset": "default", "operators": "scientific", "niterations": 30}),
    ("NGBoost", "endgame.models.ngboost", "NGBoostClassifier",
     {"n_estimators": 200}),
    ("Naive Bayes", "endgame.models.baselines.naive_bayes", "NaiveBayesClassifier", {}),
    ("LDA", "endgame.models.baselines.discriminant", "LDAClassifier", {}),
]


def load_dataset(subsample: int | None = None):
    """Load Adult Census Income with informative feature names."""
    from sklearn.datasets import fetch_openml

    print("=" * 70)
    print("Loading Adult Census Income dataset (OpenML #1590)")
    print("=" * 70)

    try:
        data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        df, target_col = data.frame.copy(), "class"
    except Exception as e:
        print(f"  OpenML fetch failed ({e}), falling back to breast cancer")
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer(as_frame=True)
        df, target_col = bc.frame, "target"

    X = df.drop(columns=[target_col])
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[target_col].astype(str)), index=X.index)

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = pd.Categorical(X[col]).codes.astype(float)
    X = X.astype(float).fillna(X.median())

    print(f"  Samples: {len(X):,}  Features: {X.shape[1]}")
    print(f"  Feature names: {list(X.columns)}")

    if subsample and subsample < len(X):
        idx = np.random.RandomState(42).choice(len(X), subsample, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
        print(f"  Subsampled to {len(X):,} rows")
    print()
    return X, y, list(X.columns)


def _save_to_parquet(results: list[dict], dataset_name: str):
    """Append results to benchmark_results.parquet."""
    ts = datetime.now(timezone.utc).isoformat()
    rows = [{
        "timestamp": ts, "script": "run_interpretable", "dataset": dataset_name,
        "model": r["model"], "status": r["status"], "accuracy": r.get("accuracy"),
        "auc": r.get("auc"), "fit_time_s": r.get("time"), "error": r.get("error"),
    } for r in results]
    new_df = pd.DataFrame(rows)

    if PARQUET_FILE.exists():
        try:
            combined = pd.concat([pd.read_parquet(PARQUET_FILE), new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(PARQUET_FILE, index=False)
    print(f"  Results appended to: {PARQUET_FILE}  ({len(combined)} total rows)")


def main():
    parser = argparse.ArgumentParser(description="Assess interpretable models")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--max-model-time", type=float, default=300.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    X, y, feature_names = load_dataset(subsample=args.subsample)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("=" * 70)
    print(f"INTERPRETABLE MODEL ASSESSMENT — {len(INTERPRETABLE_MODELS)} models")
    print("=" * 70)
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Max time per model: {args.max_model_time:.0f}s\n")

    results = []

    for idx, (name, module_path, class_name, kwargs) in enumerate(INTERPRETABLE_MODELS, 1):
        print(f"\n{'=' * 70}")
        print(f"  [{idx}/{len(INTERPRETABLE_MODELS)}] {name}")
        print(f"{'=' * 70}")

        t0 = time.time()

        # Import + instantiate
        try:
            mod = importlib.import_module(module_path)
            model = getattr(mod, class_name)(**kwargs)
        except Exception as e:
            print(f"  SKIP — {e}")
            results.append({"model": name, "status": "init_error",
                            "accuracy": None, "auc": None, "time": time.time() - t0,
                            "error": str(e)})
            continue

        # Train with timeout
        try:
            _accepts_fn = "feature_names" in inspect.signature(model.fit).parameters

            def _fit():
                if _accepts_fn:
                    model.fit(X_train.values, y_train.values, feature_names=feature_names)
                else:
                    try:
                        model.fit(X_train, y_train.values)
                    except Exception:
                        model.fit(X_train.values, y_train.values)
                return model

            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(_fit)
            try:
                future.result(timeout=args.max_model_time)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            elapsed = time.time() - t0
            print(f"  Trained in {elapsed:.1f}s")
        except FT:
            elapsed = time.time() - t0
            print(f"  TIMED OUT after {elapsed:.0f}s")
            results.append({"model": name, "status": "timeout",
                            "accuracy": None, "auc": None, "time": elapsed})
            continue
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"model": name, "status": "fit_error",
                            "accuracy": None, "auc": None, "time": elapsed,
                            "error": str(e)})
            continue

        # Evaluate
        acc = auc = None
        try:
            y_pred = model.predict(X_test) if not _try_df(model, X_test) else model.predict(X_test.values)
            acc = accuracy_score(y_test, y_pred)
        except Exception:
            try:
                acc = accuracy_score(y_test, model.predict(X_test.values))
            except Exception as e:
                print(f"  (predict failed: {e})")
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
        except Exception:
            try:
                y_proba = model.predict_proba(X_test.values)
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
            except Exception:
                pass

        if acc is not None:
            auc_str = f"  AUC={auc:.4f}" if auc else ""
            print(f"  Accuracy={acc:.4f}{auc_str}")

        # Display using library function
        try:
            display_model(name, model, feature_names, X_test.values)
        except Exception as e:
            print(f"  (display failed: {e})")

        results.append({"model": name, "status": "ok",
                        "accuracy": acc, "auc": auc, "time": elapsed})

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False, na_position="last")
    print(f"\n  {'Model':<20s} {'Status':<14s} {'Accuracy':>10s} {'AUC':>10s} {'Time':>8s}")
    print(f"  {'─' * 20} {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 8}")
    for _, row in df.iterrows():
        a = f"{row['accuracy']:.4f}" if pd.notna(row["accuracy"]) else "   —"
        u = f"{row['auc']:.4f}" if pd.notna(row["auc"]) else "   —"
        print(f"  {row['model']:<20s} {row['status']:<14s} {a:>10s} {u:>10s} {row['time']:.1f}s".rstrip())

    ok = df[df["status"] == "ok"]
    print(f"\n  Succeeded: {len(ok)}/{len(df)}")
    if len(ok) > 0 and ok["accuracy"].notna().any():
        best = ok.loc[ok["accuracy"].idxmax()]
        print(f"  Best model: {best['model']} (Accuracy={best['accuracy']:.4f})")

    df.to_csv(OUTPUT_DIR / "interpretable_results.csv", index=False)
    _save_to_parquet(results, "adult_census_income")

    print("\n" + "=" * 70 + "\nDone!\n" + "=" * 70)


def _try_df(model, X):
    """Check if predict fails with DataFrame (return True to fall back to .values)."""
    try:
        model.predict(X[:1])
        return False
    except Exception:
        return True


if __name__ == "__main__":
    main()
