# Ensembles Guide

Endgame provides a full suite of ensemble methods, from classic stacking and blending
to advanced techniques like hill climbing, optimal weight search, and knowledge
distillation. All ensemble classes follow the sklearn interface (`fit`, `predict`,
`predict_proba`).

## SuperLearner

`SuperLearner` combines arbitrary base learners using non-negative least squares (NNLS)
weighting, producing a convex combination that cannot perform worse than the best
individual model on the training data.

```python
from endgame.ensemble import SuperLearner
from endgame.models import LGBMWrapper, XGBWrapper, CatBoostWrapper

base_learners = [
    LGBMWrapper(preset='endgame'),
    XGBWrapper(preset='endgame'),
    CatBoostWrapper(preset='endgame'),
]

sl = SuperLearner(
    base_estimators=[
        ("lgbm", LGBMWrapper(preset='endgame')),
        ("xgb", XGBWrapper(preset='endgame')),
        ("cb", CatBoostWrapper(preset='endgame')),
    ],
    meta_learner="nnls",  # non-negative least squares
    cv=5,                 # inner cross-validation folds for meta-features
)

sl.fit(X_train, y_train)
proba = sl.predict_proba(X_test)
preds = sl.predict(X_test)

# Inspect learned weights
print(sl.coef_)   # non-negative, sum to 1
```

The meta-features are out-of-fold predictions from each base learner. The NNLS
solver finds the weight vector that minimises squared error on those meta-features,
guaranteeing non-negative weights without requiring regularisation.

## HillClimbingEnsemble

`HillClimbingEnsemble` uses greedy forward selection to build an ensemble that
directly optimises an arbitrary metric. At each step it adds the candidate model
(with repetition allowed) that most improves the ensemble score on the hold-out
fold. This mirrors the approach used in many competition-winning solutions.

```python
from endgame.ensemble import HillClimbingEnsemble
from sklearn.metrics import roc_auc_score

hc = HillClimbingEnsemble(
    metric=roc_auc_score,
    maximize=True,
    n_iterations=100,     # maximum greedy steps
    random_state=42,
)

# Pass a list of OOF prediction arrays
oof_preds = [lgbm_oof, xgb_oof, cb_oof, ft_oof]
hc.fit(oof_preds, y_train)

# Apply the discovered weights to test predictions
test_preds = [lgbm_test, xgb_test, cb_test, ft_test]
final = hc.predict(test_preds)

print(hc.weights_)    # float weights, sums to 1
print(hc.best_score_) # best metric achieved on OOF
```

## StackingEnsemble

`StackingEnsemble` trains base learners and a meta-learner in a single `fit` call.
Base learner out-of-fold predictions become features for the meta-learner.

```python
from endgame.ensemble import StackingEnsemble
from endgame.models import LGBMWrapper, XGBWrapper
from sklearn.linear_model import LogisticRegression

stack = StackingEnsemble(
    estimators=[
        ('lgbm', LGBMWrapper(preset='endgame')),
        ('xgb',  XGBWrapper(preset='endgame')),
    ],
    meta_learner=LogisticRegression(),
    cv=5,
    passthrough=True,   # also pass original features to meta-learner
    use_proba=True,     # use predict_proba outputs as meta-features
)

stack.fit(X_train, y_train)
preds = stack.predict(X_test)
proba = stack.predict_proba(X_test)
```

## BlendingEnsemble

`BlendingEnsemble` uses a fixed hold-out split rather than cross-validation to
generate meta-features. This is faster but uses less data for training base learners.

```python
from endgame.ensemble import BlendingEnsemble
from endgame.models import LGBMWrapper, XGBWrapper, CatBoostWrapper

blend = BlendingEnsemble(
    estimators=[
        ('lgbm', LGBMWrapper()),
        ('xgb',  XGBWrapper()),
        ('cb',   CatBoostWrapper()),
    ],
    meta_learner=LGBMWrapper(n_estimators=200),
    holdout_size=0.2,
    random_state=42,
)

blend.fit(X_train, y_train)
preds = blend.predict(X_test)
```

## OptimizedBlender

`OptimizedBlender` finds continuous blend weights by minimising a loss function
over the provided out-of-fold predictions using scipy optimisation (L-BFGS-B with
a simplex constraint).

```python
from endgame.ensemble import OptimizedBlender
from sklearn.metrics import log_loss

blender = OptimizedBlender(
    metric=log_loss,
    maximize=False,     # log_loss should be minimised
    bounds=(0.0, 1.0),  # weight bounds per model
)

blender.fit(oof_preds_matrix, y_train)  # shape (n_samples, n_models)
final = blender.predict(test_preds_matrix)
print(blender.weights_)
```

## RankAverageBlender

`RankAverageBlender` converts each model's predictions to ranks before averaging.
This is robust to scale differences between models and often outperforms simple
averaging when models produce predictions on different scales.

```python
from endgame.ensemble import RankAverageBlender

blender = RankAverageBlender(weights=[0.4, 0.35, 0.25])
final = blender.predict(test_preds_matrix)  # shape (n_samples, n_models)
```

## ThresholdOptimizer

`ThresholdOptimizer` finds the optimal classification threshold by searching over
a grid of cutoffs and maximising a target metric on out-of-fold predictions. This
is particularly useful for imbalanced datasets where the default 0.5 threshold is
suboptimal.

```python
from endgame.ensemble import ThresholdOptimizer
from sklearn.metrics import f1_score

optimizer = ThresholdOptimizer(
    metric=f1_score,
    maximize=True,
    thresholds=100,     # number of candidate thresholds to evaluate
)

optimizer.fit(oof_probabilities, y_train)
print(f"Optimal threshold: {optimizer.threshold_:.4f}")

hard_preds = optimizer.predict(test_probabilities)
```

## Knowledge Distillation

Endgame supports training a lightweight student model to mimic a heavier teacher
model. This is useful when you need a fast inference model that approximates an
expensive ensemble.

```python
from endgame.ensemble import KnowledgeDistiller
from endgame.models import LGBMWrapper
from endgame.models.baselines import LinearClassifier

teacher = LGBMWrapper(preset='endgame')
teacher.fit(X_train, y_train)

student = LinearClassifier()

kd = KnowledgeDistiller(
    teacher=teacher,
    student=student,
    temperature=3.0,       # softens teacher's probability distribution
    alpha=0.5,             # blend of hard labels vs soft labels
)

kd.fit(X_train, y_train)
preds = kd.student_.predict(X_test)
```

The `temperature` parameter controls how much the teacher's soft probabilities are
smoothed before being used as targets. Higher temperature produces softer, more
informative targets. `alpha` controls the trade-off between learning from the hard
ground-truth labels and the soft teacher labels.

## Choosing an Ensemble Strategy

| Strategy | When to use |
|---|---|
| `SuperLearner` | Strong diverse base learners; want theoretically-grounded weighting |
| `HillClimbingEnsemble` | Have OOF predictions; want to directly optimise a target metric |
| `StackingEnsemble` | Standard competition workflow; enough data for CV-based stacking |
| `BlendingEnsemble` | Limited time; large datasets where full CV is expensive |
| `OptimizedBlender` | OOF predictions already computed; want continuous weight optimisation |
| `RankAverageBlender` | Models have incompatible prediction scales |
| `ThresholdOptimizer` | Binary classification with imbalanced classes or custom metric |
| `KnowledgeDistillation` | Need fast inference; ensemble too slow for production |

## See Also

- [API Reference: ensemble](../api/ensemble)
- [Models Guide](models.md) for base learner options
- [Calibration Guide](calibration.md) for post-hoc probability calibration
