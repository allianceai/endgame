#!/usr/bin/env python
"""Demonstrate the evolutionary AutoML pipeline.

Uses the ``exhaustive`` preset with ``GeneticSearch`` to evolve
full pipeline configurations — models, hyperparameters, preprocessing,
feature selection, dimensionality reduction — over multiple generations.

Dataset: Bank Marketing (OpenML #1461)
  ~45K clients from Portuguese bank phone campaigns.
  Task: predict whether a client subscribes to a term deposit.
  17 features: age, job, marital, education, balance, housing, loan,
  contact type, day/month, call duration, campaign stats, etc.
  Challenges: class imbalance (~12% positive), mixed types, noisy features.

Usage:
    python scripts/run_evolutionary.py [--time-limit SECONDS] [--subsample N]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_dataset(subsample: int | None = 5000) -> tuple[pd.DataFrame, str]:
    """Load Bank Marketing dataset from OpenML (#1461).

    Falls back to Adult Census (#1590), then Breast Cancer if unavailable.

    Returns (dataframe, target_column_name).
    """
    # ── Try Bank Marketing first ────────────────────────────────────
    try:
        from sklearn.datasets import fetch_openml

        print("Loading Bank Marketing dataset (OpenML #1461)…")
        data = fetch_openml(data_id=1461, as_frame=True, parser="auto")
        df = data.frame.copy()
        target_col = "Class"  # "1" = subscribed, "2" = not subscribed

        # OpenML stores generic V1–V16 names; restore the real ones
        _BANK_COLUMNS = {
            "V1": "age", "V2": "job", "V3": "marital", "V4": "education",
            "V5": "default", "V6": "balance", "V7": "housing", "V8": "loan",
            "V9": "contact", "V10": "day", "V11": "month", "V12": "duration",
            "V13": "campaign", "V14": "pdays", "V15": "previous",
            "V16": "poutcome",
        }
        df.rename(columns=_BANK_COLUMNS, inplace=True)

        # Make target human-readable
        df[target_col] = df[target_col].map({"1": 1, "2": 0}).astype(int)

        print(f"  Raw size: {df.shape[0]:,} × {df.shape[1]}")
        print(f"  Features: {', '.join(c for c in df.columns if c != target_col)}")
        pos_rate = df[target_col].mean()
        print(f"  Positive rate: {pos_rate:.1%} (class imbalance)")

        if subsample and len(df) > subsample:
            df = df.sample(n=subsample, random_state=42).reset_index(drop=True)
            print(f"  Subsampled to {len(df):,} rows")

        return df, target_col

    except Exception as e:
        print(f"  Bank Marketing failed ({e})")

    # ── Fallback: Adult Census ──────────────────────────────────────
    try:
        from sklearn.datasets import fetch_openml

        print("Falling back to Adult Census Income (OpenML #1590)…")
        data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        df = data.frame.copy()
        target_col = "class"

        if subsample and len(df) > subsample:
            df = df.sample(n=subsample, random_state=42).reset_index(drop=True)
            print(f"  Subsampled to {len(df):,} rows")

        return df, target_col

    except Exception as e:
        print(f"  Adult Census failed ({e})")

    # ── Last resort: Breast Cancer ──────────────────────────────────
    from sklearn.datasets import load_breast_cancer

    print("Falling back to Breast Cancer dataset")
    bc = load_breast_cancer(as_frame=True)
    df = bc.frame
    target_col = "target"
    return df, target_col


def print_section(title: str, width: int = 72) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary AutoML — evolve full ML pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 10 min per model, unlimited time, 5K rows
  python scripts/run_evolutionary.py

  # Unlimited time, 10K rows, patient search
  python scripts/run_evolutionary.py --time-limit 0 --subsample 10000 --patience 20

  # Quick smoke test
  python scripts/run_evolutionary.py --time-limit 300 --subsample 2000 --patience 3
""",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=0,
        help="Overall time limit in seconds (default 0 = unlimited, Ctrl-C to stop).",
    )
    parser.add_argument(
        "--min-model-time",
        type=int,
        default=600,
        help="Minimum seconds per model (default 600 = 10 min).",
    )
    parser.add_argument(
        "--max-model-time",
        type=int,
        default=600,
        help="Maximum seconds per model (default 600 = 10 min).",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default None = random order each run).")
    parser.add_argument("--output", type=str, default="automl_evolutionary_output")
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Model names to exclude (e.g. --exclude gp svm node).",
    )
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────────
    print_section("DATA")
    df, target_col = get_dataset(subsample=args.subsample)
    print(f"\n  Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Target: '{target_col}'")
    print(f"  Target distribution:")
    for val, cnt in df[target_col].value_counts().items():
        print(f"    {val}: {cnt:,} ({cnt / len(df):.1%})")

    n_numeric = df.select_dtypes(include="number").shape[1] - 1
    n_cat = df.select_dtypes(include=["object", "category"]).shape[1]
    n_missing = df.isnull().sum().sum()
    print(f"  Numeric features: {n_numeric}")
    print(f"  Categorical features: {n_cat}")
    print(f"  Missing values: {n_missing:,}")

    # ── Configure predictor ─────────────────────────────────────────
    from endgame.automl import TabularPredictor

    # Models excluded by default: O(n³) kernel methods + CUDA-only models
    # that can't run in forked child processes.
    default_excludes = ["gp", "gp_regressor", "svm", "svm_regressor"]
    excluded = list(set(default_excludes + args.exclude))

    seed = args.seed
    if seed is None:
        import random as _rng
        seed = _rng.randint(0, 2**31 - 1)
        print(f"  Random seed: {seed} (use --seed {seed} to reproduce)")

    predictor = TabularPredictor(
        label=target_col,
        presets="exhaustive",
        eval_metric="roc_auc",
        time_limit=args.time_limit,
        search_strategy="genetic",
        output_path=args.output,
        checkpoint_dir=f"{args.output}/checkpoints",
        keep_training=True,
        patience=args.patience,
        min_model_time=args.min_model_time,
        max_model_time=args.max_model_time,
        excluded_models=excluded,
        random_state=seed,
        verbosity=2,
    )

    if excluded:
        print(f"  Excluded models: {', '.join(sorted(excluded))}")

    print_section("EVOLUTIONARY AUTOML")
    print(f"  Preset: exhaustive ({len(predictor._preset_config.model_pool)} model types)")
    print(f"  Search: Genetic algorithm (evolve full pipelines)")
    print(f"  Genome: model + hyperparams + scaler + feat-select + dim-reduce")
    if args.time_limit == 0:
        tl = "unlimited (Ctrl-C to stop and save best pipelines)"
    else:
        tl = f"{args.time_limit // 60}m {args.time_limit % 60}s"
    print(f"  Overall budget: {tl}")
    print(f"  Per-model budget: {args.min_model_time}s – {args.max_model_time}s")
    print(f"  Patience: {args.patience} gens without improvement (0 = disabled)")
    print(f"  Ensemble: auto (tries hill climbing, stacking, averaging)")
    print(f"  Checkpoints: top-5 pipelines saved each iteration to {args.output}/checkpoints/")
    print()

    # ── Run ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        predictor.fit(df)
    except KeyboardInterrupt:
        # The orchestrator already saved inside its own handler, but
        # the fit() call may have propagated a second Ctrl-C.
        pass

    elapsed = time.time() - t0

    # ── Results ──────────────────────────────────────────────────────
    if not predictor.is_fitted_:
        print("\nPredictor was not fitted — no results to show.")
        print(f"Check {args.output}/checkpoints/ for any saved pipelines.")
        return

    print_section("RESULTS")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"  Best score: {predictor.fit_summary_.best_score:.4f}")
    print(f"  Best model: {predictor.fit_summary_.best_model}")
    print(f"  Models trained: {len(predictor._models)}")

    # Leaderboard
    lb = predictor.leaderboard()
    if lb is not None and not lb.empty:
        print_section("LEADERBOARD")
        print(lb.to_string(index=False))

    # Evaluation on training data (sanity check)
    print_section("EVALUATION (train set)")
    try:
        predictor.evaluate(df)
    except Exception as e:
        print(f"  Evaluation failed: {e}")

    # Model interpretations
    if predictor._models:
        print_section("MODEL STRUCTURES")
        try:
            predictor.display_models()
        except Exception as e:
            print(f"  Display failed: {e}")

    print(f"\nCheckpoints saved to: {args.output}/checkpoints/")
    print(f"All outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
