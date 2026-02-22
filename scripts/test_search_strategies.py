"""Test each AutoML search strategy with medium_quality preset.

Runs each of the 8 search strategies on a synthetic binary classification
dataset, letting each use the full medium_quality defaults (15 min budget,
5-fold CV, 10 HPO trials, auto ensemble).  The only override is
time_limit=900 so each strategy gets a real run.

Prints a summary comparison table at the end.
"""

import time
import traceback

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from endgame.automl import TabularPredictor

# ── Dataset ──────────────────────────────────────────────────────────
# 1000 samples — big enough for meaningful CV, small enough that
# medium_quality finishes in a few minutes per strategy.
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_classes=2,
    random_state=42,
)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y

# Hold out 20% for prediction smoke test
train_df = df.iloc[:800].reset_index(drop=True)
test_df = df.iloc[800:].drop(columns=["target"]).reset_index(drop=True)
y_test = df.iloc[800:]["target"].values

# ── Strategies to test ───────────────────────────────────────────────
STRATEGIES = [
    "portfolio",
    "heuristic",
    "random",
    "bayesian",
    "bandit",
    "adaptive",
    "genetic",
]

results = []

for strategy in STRATEGIES:
    print(f"\n{'='*70}")
    print(f"  Strategy: {strategy}  |  preset: medium_quality")
    print(f"{'='*70}\n")

    start = time.time()
    try:
        # Use medium_quality defaults — only override search_strategy
        predictor = TabularPredictor(
            label="target",
            presets="medium_quality",
            search_strategy=strategy,
            verbosity=2,
            random_state=42,
        )
        predictor.fit(train_df)

        elapsed = time.time() - start
        n_models = predictor.fit_summary_.n_models_trained
        best_score = predictor.fit_summary_.best_score
        best_model = predictor.fit_summary_.best_model

        # Prediction smoke test
        preds = predictor.predict(test_df)
        proba = predictor.predict_proba(test_df)
        assert len(preds) == len(test_df), "prediction length mismatch"
        assert proba.shape == (len(test_df), 2), "proba shape mismatch"

        # Accuracy on held-out set
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba[:, 1])
        except Exception:
            auc = float("nan")

        results.append({
            "strategy": strategy,
            "status": "OK",
            "n_models": n_models,
            "best_cv_score": best_score,
            "test_acc": acc,
            "test_auc": auc,
            "best_model": best_model,
            "elapsed_s": elapsed,
        })

        print(f"\n  -> {strategy}: {n_models} models, "
              f"cv={best_score:.4f}, test_acc={acc:.4f}, "
              f"test_auc={auc:.4f}, best={best_model}, "
              f"{elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "strategy": strategy,
            "status": f"FAILED: {e}",
            "n_models": 0,
            "best_cv_score": float("nan"),
            "test_acc": float("nan"),
            "test_auc": float("nan"),
            "best_model": "",
            "elapsed_s": elapsed,
        })
        print(f"\n  -> {strategy}: FAILED after {elapsed:.1f}s")
        traceback.print_exc()

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("  SUMMARY — medium_quality preset, all search strategies")
print(f"{'='*70}\n")

summary = pd.DataFrame(results)
summary = summary.sort_values("best_cv_score", ascending=False, na_position="last")

pd.set_option("display.max_colwidth", 30)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.4f}".format)
print(summary.to_string(index=False))

n_ok = sum(1 for r in results if r["status"] == "OK")
n_fail = len(results) - n_ok
print(f"\n{n_ok}/{len(results)} strategies succeeded, {n_fail} failed")
