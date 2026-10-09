"""Causilo: nums-ai's in-context tabular foundation model (arXiv 2609.22866; #7 on TabArena, Oct 2026).

Re-exports ``causilo.CausiloClassifier`` / ``causilo.CausiloRegressor`` (https://github.com/nums-ai/causilo), which are
already scikit-learn estimators: ``CausiloClassifier(n_estimators=8, *, random_state=42, device="auto",
use_kv_cache=False)``, ``fit`` / ``predict_proba`` / ``predict``; the regressor's ``predict`` also takes
``output_type="median" | "quantiles"``. Non-numeric pandas columns are treated as categorical, NaN is handled, more
than 10 classes go through built-in ECOC. Weights (``nums-ai/causilo``, ~145 MB per task, not gated) download on first
fit; they are under the Causilo License v1.0: non-commercial research only. CUDA uses fp16 (pre-Ampere is fine).

Install::

    pip install causilo                  # requires torch >= 2.13
    pip install causilo --no-deps        # keeps an older torch; checked working on torch 2.9.1
"""

try:
    from causilo import CausiloClassifier, CausiloRegressor
except ImportError as exc:
    raise ImportError("causilo is not installed: pip install causilo") from exc

__all__ = ["CausiloClassifier", "CausiloRegressor"]
