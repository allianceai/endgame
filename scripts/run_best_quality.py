#!/usr/bin/env python
"""End-to-end AutoML run using the ``best_quality`` preset.

Demonstrates incremental checkpointing, continuous optimization, and
per-model time budgets.

Usage
-----
    python scripts/run_best_quality.py [--time-limit SECONDS] [--subsample N]

Pass ``--time-limit 0`` for unlimited training (stop with Ctrl-C).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("automl_best_quality_output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PARQUET_FILE = Path("benchmark_results.parquet")


def load_adult_dataset(subsample: int | None = None):
    """Load Adult Census Income, return (train_df, test_df, target_col)."""
    print("=" * 70)
    print("Loading Adult Census Income dataset (OpenML #1590)")
    print("=" * 70)

    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df, target_col = data.frame, "class"
    X = df.drop(columns=[target_col])

    print(f"  Samples: {len(df):,}  Features: {X.shape[1]}")
    print(f"  Classes: {dict(df[target_col].value_counts())}")

    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=42)
        print(f"  Subsampled to {len(df):,} rows")
    print()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return train_df, test_df, target_col


def _save_to_parquet(leaderboard: pd.DataFrame, total_time: float):
    """Append leaderboard to benchmark_results.parquet."""
    ts = datetime.now(timezone.utc).isoformat()
    rows = [{
        "timestamp": ts, "script": "run_best_quality", "dataset": "adult_census_income",
        "model": row.get("model", ""), "status": "ok" if row.get("score", -np.inf) > -np.inf else "failed",
        "accuracy": row.get("score"), "auc": None, "fit_time_s": row.get("fit_time"), "error": None,
    } for _, row in leaderboard.iterrows()]

    new_df = pd.DataFrame(rows)
    if PARQUET_FILE.exists():
        try:
            combined = pd.concat([pd.read_parquet(PARQUET_FILE), new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(PARQUET_FILE, index=False)
    print(f"\n  Leaderboard appended to: {PARQUET_FILE}  ({len(combined)} total rows)")


def main():
    parser = argparse.ArgumentParser(description="Run Endgame AutoML best_quality")
    parser.add_argument("--time-limit", type=int, default=600,
                        help="Time limit in seconds (0 = unlimited). Default: 600")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--min-model-time", type=float, default=300.0)
    parser.add_argument("--max-model-time", type=float, default=600.0)
    args = parser.parse_args()

    train_df, test_df, target_col = load_adult_dataset(subsample=args.subsample)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENDGAME AutoML — best_quality PRESET")
    print("=" * 70)
    print(f"  Time limit:      {'UNLIMITED' if args.time_limit == 0 else f'{args.time_limit}s'}")
    print(f"  Min model time:  {args.min_model_time:.0f}s")
    print(f"  Max model time:  {args.max_model_time:.0f}s")
    print(f"  Keep training:   True")
    print(f"  Patience:        {args.patience}\n")

    from endgame.automl import TabularPredictor

    predictor = TabularPredictor(
        label=target_col,
        presets="best_quality",
        time_limit=args.time_limit,
        output_path=str(OUTPUT_DIR),
        verbosity=2,
        random_state=42,
        checkpoint_dir=str(CHECKPOINT_DIR),
        keep_training=True,
        patience=args.patience,
        min_improvement=1e-4,
        min_model_time=args.min_model_time,
        max_model_time=args.max_model_time,
    )

    print("\n" + "-" * 70)
    print("Starting fit()  (Ctrl-C saves a checkpoint and exits gracefully)")
    print("-" * 70 + "\n")

    start = time.time()
    try:
        predictor.fit(train_df)
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n  *** Interrupted after {elapsed:.0f}s ***")
        print(f"  Checkpoint saved to: {CHECKPOINT_DIR}\n")

    elapsed = time.time() - start
    print(f"\n  Total wall time: {elapsed:.1f}s")

    if hasattr(predictor, "fit_summary_") and predictor.fit_summary_ is not None:
        fs = predictor.fit_summary_
        print(f"\n  Models trained: {fs.n_models_trained}  |  Failed: {fs.n_models_failed}")
        print(f"  Best model: {fs.best_model}  |  Best CV score: {fs.best_score:.4f}")

    if predictor.leaderboard_ is not None and not predictor.leaderboard_.empty:
        print(f"\n  Leaderboard:")
        print(predictor.leaderboard_.to_string(index=False))

    if predictor.is_fitted_:
        # evaluate() now prints classification_report automatically
        predictor.evaluate(test_df)

        try:
            predictor.save(str(OUTPUT_DIR / "final_model"))
            print(f"\n  Predictor saved to: {OUTPUT_DIR / 'final_model'}")
        except Exception as e:
            print(f"\n  Could not save final model: {e}")

        if predictor.leaderboard_ is not None and not predictor.leaderboard_.empty:
            _save_to_parquet(predictor.leaderboard_, elapsed)
    else:
        print("\n  Predictor is not fitted — check logs above for errors")
        print(f"  Partial checkpoint may be at: {CHECKPOINT_DIR}")

    print("\n" + "=" * 70 + "\nDone!\n" + "=" * 70)


if __name__ == "__main__":
    main()
