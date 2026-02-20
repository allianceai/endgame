# Calibration Guide

Endgame provides a comprehensive calibration module covering conformal prediction,
Venn-ABERS calibration, and classical probability calibration methods. All classes
follow the sklearn interface (`fit`, `predict`, `predict_proba`).

## Conformal Prediction (Classification)

`ConformalClassifier` wraps any classifier to produce prediction sets that contain
the true label with at least `1 - alpha` marginal coverage. No distributional
assumptions are required beyond exchangeability.

```python
from endgame.calibration import ConformalClassifier
from endgame.models import LGBMWrapper
from sklearn.model_selection import train_test_split

X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base = LGBMWrapper(preset='endgame')
base.fit(X_train, y_train)

cc = ConformalClassifier(
    estimator=base,
    alpha=0.1,          # target miscoverage rate; 90% coverage guaranteed
    method='lac',       # 'lac' (softmax-based) or 'aps' (adaptive prediction sets)
)

cc.fit(X_cal, y_cal)   # calibrate on hold-out set

# Returns a list of sets, one per test point
prediction_sets = cc.predict(X_test)
for i, pset in enumerate(prediction_sets[:5]):
    print(f"Sample {i}: possible classes = {pset}")

# Standard hard prediction uses the singleton with highest score
preds = cc.predict(X_test)

# Empirical coverage on a labelled evaluation set
cov = cc.coverage_score(X_eval, y_eval)
print(f"Empirical coverage: {cov:.3f}")  # should be >= 0.90
```

The `'aps'` score (Adaptive Prediction Sets) produces smaller, class-conditional
sets at the cost of slightly weaker marginal guarantees. Use `'lac'` (Least
Ambiguous Classifier) for standard coverage.

## Conformal Prediction (Regression)

`ConformalRegressor` produces prediction intervals with guaranteed marginal
coverage. The width of intervals adapts automatically to the local difficulty of
each test point when a difficulty estimator is provided.

```python
from endgame.calibration import ConformalRegressor
from endgame.models import LGBMWrapper

base = LGBMWrapper(preset='endgame')
base.fit(X_train, y_train)

cr = ConformalRegressor(
    estimator=base,
    alpha=0.05,          # 95% coverage
    method='split',      # 'split' (fast) or 'cv+' (cross-conformal, slower)
)

cr.fit(X_cal, y_cal)

# Returns a tuple of (lower, upper) arrays
lower, upper = cr.predict_interval(X_test)

widths = upper - lower
print(f"Median interval width: {np.median(widths):.4f}")

cov = cr.coverage_score(X_eval, y_eval)
print(f"Empirical coverage: {cov:.3f}")
```

## Conformalized Quantile Regression (CQR)

`ConformizedQuantileRegressor` combines a quantile regressor with conformal
calibration to produce adaptive intervals. Intervals are wider where the model is
less certain, unlike split conformal which uses a fixed residual threshold.

```python
from endgame.calibration import ConformizedQuantileRegressor
from endgame.models import LGBMWrapper

# Base model must support quantile regression
qr = LGBMWrapper(objective='quantile', preset='endgame')

cqr = ConformizedQuantileRegressor(
    estimator=qr,
    alpha=0.1,           # 90% coverage target
    quantile_low=0.05,   # lower quantile for the base regressor
    quantile_high=0.95,  # upper quantile for the base regressor
)

cqr.fit(X_train, y_train, X_cal=X_cal, y_cal=y_cal)

lower, upper = cqr.predict_interval(X_test)
```

CQR is the recommended method when prediction intervals of varying width are
needed. The conformity score is `max(q_low - y, y - q_high)`, so the calibration
step only stretches or shrinks the raw quantile interval by a single scalar.

## Venn-ABERS Calibration

`VennABERS` produces well-calibrated probability estimates without requiring a
specific parametric form. It is guaranteed to be calibrated in a strong sense
(individual calibration) under no distributional assumptions.

```python
from endgame.calibration import VennABERS
from endgame.models import LGBMWrapper

base = LGBMWrapper(preset='endgame')
base.fit(X_train, y_train)

va = VennABERS(estimator=base)
va.fit(X_cal, y_cal)

# Returns point probabilities (geometric mean of the interval bounds)
proba = va.predict_proba(X_test)

# Returns the full Venn-ABERS interval [p0, p1] per sample
intervals = va.predict_interval(X_test)
p0, p1 = intervals[:, 0], intervals[:, 1]

# Interval width indicates epistemic uncertainty
uncertainty = p1 - p0
```

Unlike Platt scaling or isotonic regression, Venn-ABERS does not require tuning
and is valid for small calibration sets. It is particularly useful when the base
model has poorly calibrated raw probabilities (e.g., a gradient boosting model).

## Classical Probability Calibration

### Temperature Scaling

Temperature scaling divides the logits of a neural network (or any model exposing
logits) by a single learnable scalar `T`. It is the most common post-hoc
calibration technique for deep learning.

```python
from endgame.calibration import TemperatureScaling

ts = TemperatureScaling()
ts.fit(logits_cal, y_cal)    # calibrate on logits (pre-softmax)

calibrated_proba = ts.predict_proba(logits_test)
print(f"Learned temperature: {ts.temperature_:.4f}")
```

### Platt Scaling

Platt scaling fits a logistic regression on the model's raw scores. It is
effective when the raw scores are approximately normally distributed by class.

```python
from endgame.calibration import PlattScaling

ps = PlattScaling()
ps.fit(scores_cal, y_cal)    # 1D array of decision scores

calibrated_proba = ps.predict_proba(scores_test)
```

### Beta Calibration

Beta calibration maps scores through a Beta CDF, offering more flexibility than
Platt scaling for scores bounded in [0, 1] (e.g., already-softmaxed probabilities).

```python
from endgame.calibration import BetaCalibration

bc = BetaCalibration()
bc.fit(proba_cal, y_cal)   # uncalibrated probabilities in [0, 1]

calibrated_proba = bc.predict_proba(proba_test)
```

### Isotonic Calibration

Isotonic regression fits a non-parametric monotone mapping from scores to
probabilities. It can perfectly fit calibration data but may overfit with small
calibration sets.

```python
from endgame.calibration import IsotonicCalibration

ic = IsotonicCalibration()
ic.fit(proba_cal, y_cal)

calibrated_proba = ic.predict_proba(proba_test)
```

## Evaluating Calibration Quality

`CalibrationAnalyzer` computes multiple calibration diagnostics and generates
reliability diagrams.

```python
from endgame.calibration import CalibrationAnalyzer

analyzer = CalibrationAnalyzer(n_bins=10, strategy='uniform')
analyzer.fit(proba_test, y_test)

# Scalar metrics
print(f"ECE  : {analyzer.ece_:.4f}")   # Expected Calibration Error
print(f"MCE  : {analyzer.mce_:.4f}")   # Maximum Calibration Error
print(f"Brier: {analyzer.brier_:.4f}") # Brier Score

# Reliability diagram (matplotlib figure)
fig = analyzer.plot_reliability_diagram(title="Model Calibration")
fig.savefig("reliability.png", dpi=150)

# Per-bin breakdown
print(analyzer.bin_stats_)  # DataFrame: bin_lower, bin_upper, fraction_pos, mean_conf, count
```

## Choosing a Calibration Method

| Method | Best for |
|---|---|
| `TemperatureScaling` | Neural networks with logit access; large calibration sets |
| `PlattScaling` | SVM or other margin-based models; unimodal score distributions |
| `BetaCalibration` | Models outputting probabilities; flexible boundary handling |
| `IsotonicCalibration` | Large calibration sets; non-monotone miscalibration patterns |
| `VennABERS` | Small calibration sets; no distributional assumptions; individual guarantees |
| `ConformalClassifier` | Hard prediction sets with coverage guarantees |
| `ConformalRegressor` | Prediction intervals with coverage guarantees |
| `ConformizedQuantileRegressor` | Adaptive-width intervals; heteroscedastic regression |

## See Also

- [API Reference: calibration](../api/calibration)
- [Ensembles Guide](ensembles.md) for combining calibrated models
- [Models Guide](models.md) for base model options
