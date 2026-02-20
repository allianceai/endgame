#!/usr/bin/env python
"""End-to-end test of the Endgame AutoML pipeline.

Uses the Adult Census Income dataset (OpenML #1590): 48,842 samples,
14 mixed features, binary target.  Tests both the Quick API and the
full TabularPredictor with multiple presets.
"""

from __future__ import annotations

import time
import traceback

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_adult_dataset():
    """Load Adult Census Income, return full DataFrame."""
    print("=" * 70)
    print("Loading Adult Census Income dataset (OpenML #1590)")
    print("=" * 70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame
    X = df.drop(columns=["class"])
    print(f"  Samples: {len(df):,}  Features: {X.shape[1]}")
    print(f"  Classes: {dict(df['class'].value_counts())}\n")
    return df


def prepare_numeric(df: pd.DataFrame):
    """Convert DataFrame to numeric arrays for the Quick API."""
    y = LabelEncoder().fit_transform(df["class"])
    X = df.drop(columns=["class"]).copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.fillna(X.median()), y


def test_quick_classify(X_train, y_train, X_test, y_test):
    from endgame.quick import classify
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("=" * 70)
    print("TEST 1: quick.classify()")
    print("=" * 70)

    t0 = time.time()
    result = classify(X_train, y_train, preset="default", metric="roc_auc")
    elapsed = time.time() - t0

    y_pred = result.model.predict(X_test)
    y_proba = result.model.predict_proba(X_test)[:, 1]

    print(f"  CV ROC-AUC:   {result.cv_score:.4f}")
    print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Time: {elapsed:.1f}s\n")
    return result


def test_quick_compare(X_train, y_train, X_test, y_test):
    from endgame.quick import compare
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("=" * 70)
    print("TEST 2: quick.compare()")
    print("=" * 70)

    t0 = time.time()
    cmp = compare(X_train, y_train, task="classification", preset="default", metric="roc_auc")
    elapsed = time.time() - t0

    print(f"\n  Leaderboard ({len(cmp.leaderboard)} models, {elapsed:.1f}s):")
    for i, e in enumerate(cmp.leaderboard):
        print(f"    {i+1}. {e['model']:15s} AUC={e['score']:.4f} ({e['fit_time']:.1f}s)")

    y_pred = cmp.best_model.predict(X_test)
    y_proba = cmp.best_model.predict_proba(X_test)[:, 1]
    print(f"\n  Best on test: AUC={roc_auc_score(y_test, y_proba):.4f}  "
          f"Acc={accuracy_score(y_test, y_pred):.4f}\n")
    return cmp


def test_tabular(df_train, df_test, preset, label, time_limit=300, **kwargs):
    from endgame.automl import TabularPredictor

    print("=" * 70)
    print(f"TEST: TabularPredictor preset='{preset}'")
    print("=" * 70)

    predictor = TabularPredictor(
        label=label, presets=preset, time_limit=time_limit,
        verbosity=2, random_state=42, **kwargs,
    )

    t0 = time.time()
    predictor.fit(df_train, **({"interpretable_only": True} if preset == "interpretable" else {}))
    elapsed = time.time() - t0

    print(f"\n  Time: {elapsed:.1f}s")

    if predictor.fit_summary_ is not None:
        fs = predictor.fit_summary_
        print(f"  Models trained: {fs.n_models_trained}  Best: {fs.best_model}  "
              f"CV: {fs.best_score:.4f}")

    if predictor.leaderboard_ is not None and not predictor.leaderboard_.empty:
        print("\n  Leaderboard:")
        for _, row in predictor.leaderboard_.iterrows():
            print(f"    {row['model']:25s} score={row['score']:.4f} time={row['fit_time']:.1f}s")

    if predictor.is_fitted_:
        predictor.evaluate(df_test)

        # For interpretable preset, show learned structures
        if preset == "interpretable":
            predictor.display_models()
    print()
    return predictor


def main():
    print("\n" + "*" * 70)
    print("  ENDGAME AutoML PIPELINE — END-TO-END TEST")
    print("*" * 70 + "\n")

    overall_start = time.time()
    df = load_adult_dataset()

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["class"])
    print(f"Train: {len(df_train):,}  Test: {len(df_test):,}\n")

    X, y = prepare_numeric(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    tests = [
        ("quick_classify", lambda: test_quick_classify(X_train, y_train, X_test, y_test)),
        ("quick_compare", lambda: test_quick_compare(X_train, y_train, X_test, y_test)),
        ("tabular_good", lambda: test_tabular(df_train, df_test, "good_quality", "class", 300)),
        ("tabular_fast", lambda: test_tabular(df_train, df_test, "fast", "class")),
        ("tabular_interp", lambda: test_tabular(df_train, df_test, "interpretable", "class", 180)),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

    total = time.time() - overall_start
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total time: {total:.1f}s")
    print(f"  Tests passed: {len(results)}/{len(tests)}")
    if len(results) == len(tests):
        print("  All tests passed!")
    print()


if __name__ == "__main__":
    main()
