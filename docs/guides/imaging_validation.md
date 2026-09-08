# Imaging studies: fitting boundaries and validation

The imaging components support training-only fit/transform state, explicit study
folds, and auditable fallbacks. These software contracts do not establish
scientific validity for a particular Parkinson's cohort. Freeze the cohort,
outcome, feature allowlist and split manifest before fitting or selecting models.
Record patient, visit, acquisition, scanner/protocol, decision time, label-end
time, QC and modality availability. Join predictions using those keys.

## Choose the deployment question first

| Intended use | Evaluation boundary | Evidence required before relying on results |
|---|---|---|
| New patients at scanners represented in training | All visits and acquisitions of a patient stay together in outer and inner folds. Confirm scanner coverage in each training fold. | Patient-held-out discrimination, calibration and uncertainty, broken down by scanner and modality availability. Compare clinical-only, individual modalities, early/late fusion and no-harmonization baselines on identical patients. |
| Entirely new sites or scanners | Hold out complete sites, including their unlabeled images. Keep any separate adaptation/calibration sample out of final evaluation. | Site-held-out results and external replication; adjustment coverage and calibration at every new site. Raw passthrough is zero-shot application, not estimated harmonization of that site. |
| Progression from repeated visits | Define decision time, outcome horizon and availability of every feature and label. Distinguish new-patient forecasting from future visits of known patients. | Patient holdouts for the former; forward-only, label-window-purged holdouts for the latter. Validate within-person slopes, scanner changes, dropout and censoring. Resample patients for uncertainty. |
| Group differences and biomarkers | Prespecify hypotheses, covariates, comparison families and disease-by-site overlap. Separate feature discovery from confirmation. | Site-adjusted and repeated-measures inference, multiplicity control, preprocessing uncertainty, injected-effect recovery and independent replication. Model weights and PLS separation are not significance tests. |

Diagnosis may be a prespecified covariate in an association analysis. It cannot
be supplied at inference when diagnosis is the unknown prediction target. Remove
reference flags, IDs and outcome-derived columns from predictor allowlists.

## Put harmonization inside each stacking base pipeline

An outer preprocessing step fitted on all outer-training data still shares
information across the stack's inner folds. Each cloned base pipeline must own
its harmonization, normative fit, imputation, scaling and feature selection.
`block_metadata` sends scanner/covariate columns to that pipeline without using
them as modality-availability measurements or raw meta-model predictors.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from endgame.models import BlockStackingClassifier
from endgame.preprocessing import ComBatHarmonizer

# X_train is one outer training partition; patient_id is an aligned vector.
t1_cols = ["roi_1", "roi_2", "roi_3"]
base = make_pipeline(
    ComBatHarmonizer(batch="scanner", covariates=["age"], features=t1_cols),
    ColumnTransformer([("imaging", "passthrough", t1_cols)]),
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)
stack = BlockStackingClassifier(
    blocks={"t1": t1_cols}, base_estimator={"t1": base},
    block_metadata={"t1": ["scanner", "age"]}, cv=5,
).fit(X_train, y_train, groups=patient_id)
p_test = stack.predict_proba(X_test)[:, 1]
```

Repeat the whole fit in each outer fold, passing only that fold's patient IDs.
Evaluate the entire fitted stack on outer-held-out patients. Its
`oof_predictions_` are base-model meta-training inputs, not an independent
evaluation of the fitted meta-model. Tune all choices and probability calibration
within the permitted training data, including grouping at their inner boundaries.

Missing blocks use the current training fold's class prior. An unavailable or
single-class final block is reported in `block_status_` and uses the full-training
prior. Raw passthrough means are fitted on meta-training rows and reused unchanged
for one patient or a batch. Entirely absent training columns use zero imputation;
record that fallback and do not interpret the imputed modality as observed.
Whole absent modalities at prediction are allowed; partially missing column
schemas fail. Partially observed feature rows use the block fallback.

## Longitudinal folds

```python
from endgame.validation import PurgedPanelSplit

stack.set_params(cv=PurgedPanelSplit(n_splits=3, train_fraction=0.6))
stack.fit(X_history, y_history, times=decision_times,
          label_end_times=outcome_end_times)
covered = stack.oof_coverage_
```

This example intentionally permits earlier visits of known patients in training.
Initial history remains uncovered and is excluded from meta-model training.
Training decisions and label ends must both precede the first validation time.
Passing `groups` additionally enforces disjoint patients; use a custom splitter
that satisfies both requirements when evaluating new patients over time.
An integer `cv` is rejected when temporal metadata are supplied.

At the `PurgedPanelSplit.split` API, `groups` means **decision times**. At
`BlockStackingClassifier.fit` and `batch_leakage_check`, `groups` means **patients**;
these wrappers pass their separate `times` argument to the panel splitter.
`purged_panel_oof` calls scalar `predict`, not `predict_proba`, and is not a
probability-evaluation wrapper. Ranking utilities rank within query groups;
ranking scores cannot be used as disease probabilities or decision thresholds.

ComBat here is cross-sectional: it has no subject random effect. This fold
support does not turn it into longitudinal ComBat. Validate trajectory and
scanner-change preservation using an appropriate longitudinal model.

## Numerical and missing-data contracts

- ComBat excludes globally constant features from empirical-Bayes moments.
  Nonconstant features perfectly explained by the batch/covariate design raise
  by default, since these can be scanner identifiers. Rank-deficient or saturated
  designs fail. Inspect condition diagnostics and scientific site/covariate overlap.
- Empirical Bayes requires at least two active features and valid priors and
  convergence. Use `eb=False` deliberately for plain adjustment, or explicitly
  opt into warned `eb_fallback="no_eb"`. A finite fallback is not evidence that
  the harmonization is scientifically suitable.
- `unknown_batch="passthrough"` warns and retains raw unseen-scanner measurements.
  `adjustment_report(X)` records adjustment status. For blockwise harmonization,
  inspect status by modality, including missing or rare batches and insufficient
  data. Do not pool raw and adjusted rows without checking their consequences.
- `BlockwiseHarmonizer(partial_missing="raise")` rejects partially observed
  modalities. Explicit `"passthrough"` excludes those rows from estimation and
  leaves them unchanged. Never fabricate whole missing modalities to estimate
  scanner effects. `MissingBlockIndicator.empty_features_` records columns with
  no observed training values; `empty_value` defaults to zero.
- `NormativeDeviation` accepts only nonmissing boolean/0/1 reference membership.
  Its marker is needed during fit only. Zero residual scale, insufficient residual
  degrees of freedom and rank-deficient reference designs fail. Continuous
  covariate extrapolation warns by default (`extrapolation="raise"` is available).
  W-scores use a linear, homoskedastic residual scale, not predictive uncertainty.
  Check their distribution and tails in independent controls across covariates
  and sites; training-control centering is not validation.
- PLS-DA accepts weights keyed by the original labels and checks feature identity.
  Its supervised projection and component selection belong inside training folds.
  A logistic head, especially with balanced class weights, does not guarantee
  population-calibrated probabilities. Feature importances are model weights.

## Scanner diagnostics and uncertainty

Fit the diagnostic's preprocessing inside its own folds. Compare raw and adjusted
data using the same split definition and patients. Pass an unfitted `preprocessor`
and an explicit output-feature list rather than an already globally harmonized
table. Metadata needed by the preprocessor stay in the input DataFrame.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from endgame.utils import batch_leakage_check

report = batch_leakage_check(
    X_train, X_train["scanner"], groups=patient_id,
    preprocessor=ComBatHarmonizer("scanner", covariates=["age"], features=t1_cols),
    features=t1_cols,
    models={"linear": LogisticRegression(max_iter=2000),
            "forest": RandomForestClassifier(n_estimators=200, random_state=0)},
)
```

Low linear scanner AUROC does not rule out nonlinear scanner signal. High AUROC
is a diagnostic of predictability, not proof of outcome leakage: biology and
sampling can differ across sites. Examine biological-signal preservation too.

`bootstrap_ci` now defaults to ordinary resampling (`stratified=False`), suitable
for continuous outcomes. For classification, explicit stratification conditions
on class counts. With repeated visits, supply `groups=patient_id` to resample whole
patients; group stratification requires one stratum per patient. These intervals
condition on fixed predictions and do not include model-selection/training
uncertainty. Duplicating visits does not create independent patients; the metric
itself still weights rows unless you explicitly choose patient-level aggregation.

`paired_bootstrap_diff` uses paired draws for its percentile difference interval.
Its p-value is now a paired randomization test that swaps the two models' scores
within each independent patient (or row if ungrouped), with a plus-one correction.
That p-value assumes exchangeability of model scores under the null; it is not a
universal test of equal metrics. `delong_test` assumes independent observations
and both binary classes; use patient-aware methods for repeated visits.

`decision_curve` requires a finite one-dimensional positive-class probability
vector, binary 0/1 outcomes with both classes present, and thresholds strictly
between zero and one. For a case-control sample, `prevalence=` reweights sensitivity
and false-positive rate to a specified target prevalence. This does not fix
probability calibration, selection bias or transportability. Evaluate calibrated
held-out predictions at prespecified clinically meaningful thresholds. The helper
does not handle censoring, survival horizons or continuous progression outcomes.

## Reproducible checks and remaining study gates

```bash
pip install -c requirements/scientific-ci.txt -e ".[dev,scientific]"
pytest tests/test_harmonization.py tests/test_block_stacking.py \
  tests/test_normative_plsda.py tests/test_study_additions.py \
  tests/test_panel_ranking.py tests/test_scientific_safety.py \
  -W error::RuntimeWarning
```

The scientific CI job runs this set on Python 3.10–3.12 and requires neuroCombat
and LightGBM before testing. Regression tests cover reference parity, constant
padding, prediction-batch invariance, patient and temporal boundaries, held-out
normative controls, missing modalities, clustered resampling and metric contracts.

Before freezing study conclusions, add cohort-specific subject-level null-label
simulations, site-confounding and missingness-only baselines, injected disease
effects and progression-slope recovery. Permutations must refit the entire affected
pipeline under valid exchangeability restrictions. Prespecify acceptable calibration,
effect recovery and subgroup performance before inspecting final holdouts. Software
tests cannot supply these cohort-level acceptance results.

Further reading: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[longitudinal ComBat](https://pmc.ncbi.nlm.nih.gov/articles/PMC7605103/),
[normative modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC7613648/), and
[decision-curve interpretation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6261531/).
