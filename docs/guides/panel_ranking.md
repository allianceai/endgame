# Date-grouped panel ranking

Added 2026-09-08. These are reusable, fixed-settings primitives, not a trading strategy or a claim of investment performance. Importing them performs no training, downloads or search.

## Rank contemporaneous alternatives

`endgame.ranking.GroupedRanker` supports two backends:

- `backend="ridge"`: regularized regression of within-group outcome percentiles, with train-fitted median imputation and standardization.
- `backend="lambdamart"`: LightGBM LambdaRank with integer relevance grades, linear label gains, contiguous query sizes and CPU-only execution. LightGBM must already be installed; import fails explicitly otherwise.

Groups are decision dates or other contemporaneous choice sets, **not ticker IDs**. Input rows may be interleaved: features, grades and weights are sorted together for training and predictions preserve input order. Every training group needs at least two observations, and at least one group needs outcome variation. Each date receives equal total sample weight; LambdaRank's additional query normalization means this is not an equal-gradient guarantee.

```python
from endgame.ranking import GroupedRanker

# X_train/y_train/date_groups must already be an allowed historical window.
model = GroupedRanker(
    backend="lambdamart", focus="bottom",
    n_estimators=300, num_leaves=7, min_child_samples=50,
    n_bins=5, truncation_level=30, n_jobs=1,
)
model.fit(X_train, y_train, groups=date_groups)
scores = model.predict(X_test)  # higher always means a better predicted outcome
ranks = model.predict_rank(X_test, test_date_groups)
```

Bottom focus makes bad outcomes more relevant during learning, then negates predictions. A low-score exclusion rule therefore keeps its original direction. It predicts relative ordering, **not downside probability, expected loss, calibrated return or portfolio Sharpe**. Ridge only supports top focus: reversing a linear target does not create a distinct model.

The truncation level is a count of ranked alternatives (default 30), not a percentage. Five relevance grades, linear gains and this cutoff are fixed hypotheses, not tuned optima. There is no implicit validation split, early stopping, GPU use, parameter search or automatic core allocation. `n_jobs` must be a positive integer; default 1.

## Forward-only validation with actual label windows

The older row-gap splitter is useful for different data layouts. For a repeated stock/date panel, use `PurgedPanelSplit`, which keeps entire decision dates together and requires explicit label-end times.

```python
from endgame.validation import PurgedPanelSplit, purged_panel_oof

cv = PurgedPanelSplit(n_splits=3, train_fraction=0.75)
oof = purged_panel_oof(
    GroupedRanker(backend="ridge", alpha=100.0),
    X, y, decision_times, label_end_times,
    cv=cv, fit_groups=True,
)
validated_predictions = oof.predictions[oof.covered]
validated_targets = y[oof.covered]
```

Decision and end times must be aligned numeric ordinals or NumPy datetime64 arrays in the same units. Explicitly convert timestamp strings before use. Pass real label ends: a fixed ordinal increment is valid only if it faithfully represents the outcome horizon.

The final quarter of distinct dates is divided into validation blocks. A training date is excluded in full if any of its labels ends at or after the first validation date. No future date enters training. Optional `max_train_groups` limits eligible history in dates rather than stock rows.

All fitted preprocessing must live inside the cloned estimator or pipeline. Never globally fit imputation, selection or transforms before this call. Empty purged folds fail; there is no random-split fallback.

Cold-start rows have `NaN` predictions, `covered=False`, and fold ID -1. They must not be filled with in-sample predictions. This helper does not fit a stacking layer. A learned meta-model would need its own chronological training/validation boundary and an independent final test; scoring meta-weights on their training OOF is not honest evaluation. Generic full-partition `cross_val_predict` is not a drop-in solution for a forward splitter with unvalidated initial history.

## Fixed blending and diagnostic semantics

`group_percentiles` uses average ties and preserves row order. Distinct endpoints map to 0/1; constant and singleton groups map to 0.5.

`group_rank_average(prediction_columns, groups, weights=None)` ranks each constituent within its date and combines predeclared nonnegative weights. It does not take outcomes, fit weights or choose constituents. All scores must be finite and aligned; missing constituents fail.

`group_rank_diagnostics(y, predictions, groups, tail_fraction=0.3)` returns per-date Spearman rank IC and equal-weight outcomes for the predicted top/bottom tails. It includes all boundary ties and publishes their counts. Constant predictions give overlapping full-group tails, zero spread and undefined IC, not ticker-order profits. These outputs are not cap-weighted portfolio returns, risk-adjusted alpha, cost-adjusted performance or an annualized information ratio.

## References

- [LightGBM LGBMRanker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html): query sizes and integer relevance.
- [LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): label gains and LambdaRank truncation.
- [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html): time-aware evaluation.
- [scikit-learn cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html): its partition/prediction contract.

Generic Endgame AutoML, stacking and blender defaults remain unchanged. Call these explicit panel APIs when those guarantees are required.
