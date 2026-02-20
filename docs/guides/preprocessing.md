# Preprocessing Guide

Endgame's preprocessing module is built on Polars internally, but accepts pandas DataFrames, numpy arrays, and Polars DataFrames as input. All transformers follow the scikit-learn `fit`/`transform` API and are compatible with sklearn `Pipeline`. Output format mirrors the input format by default (`output_format='auto'`).

```python
import endgame as eg
import numpy as np
import pandas as pd
from endgame.preprocessing import SafeTargetEncoder, AutoBalancer
```

---

## Encoding

Categorical encoding transforms string or integer category columns into numeric representations that models can consume. Endgame provides four encoders covering the most common competition and production patterns.

### SafeTargetEncoder

Target encoding replaces each category with the mean target value for that category. The naive version leaks target information during training — `SafeTargetEncoder` prevents this via inner-fold cross-validation: each training sample's encoding is computed from the other folds only. An M-estimate smoothing term regularizes rare categories toward the global mean.

Formula: `S_i = (n_i * mu_i + m * mu_global) / (n_i + m)` where `m` is the `smoothing` parameter.

```python
from endgame.preprocessing import SafeTargetEncoder

encoder = SafeTargetEncoder(
    smoothing=10,   # Higher = more regularization for rare categories
    cv=5,           # Inner folds for leakage prevention
    noise_level=0.0,
    handle_unknown='global_mean',  # 'global_mean', 'nan', or 'error'
)

# fit_transform uses inner-fold encoding (no leakage)
X_train_enc = encoder.fit_transform(X_train, y_train)

# transform uses full-data statistics
X_test_enc = encoder.transform(X_test)
```

`SafeTargetEncoder` auto-detects categorical columns when `cols=None`. To target specific columns:

```python
encoder = SafeTargetEncoder(cols=['city', 'product_id'], smoothing=20)
```

### LeaveOneOutEncoder

LOO encoding excludes the current sample's own target value when computing the category mean, preventing direct self-leakage without requiring cross-validation folds. Suitable for online learning or settings where full CV is too expensive.

```python
from endgame.preprocessing import LeaveOneOutEncoder

encoder = LeaveOneOutEncoder(smoothing=1.0)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)
```

### CatBoostEncoder

Mimics CatBoost's internal ordered target statistic: for each sample, only the "preceding" samples (in a random permutation) contribute to that sample's encoding. Prevents leakage without cross-validation overhead.

```python
from endgame.preprocessing import CatBoostEncoder

encoder = CatBoostEncoder(smoothing=1.0, random_state=42)
X_train_enc = encoder.fit_transform(X_train, y_train)
```

### FrequencyEncoder

Replaces categories with their frequency (proportion or count) in the training data. Does not require target values — useful for unsupervised settings or as a complement to target encoders.

```python
from endgame.preprocessing import FrequencyEncoder

encoder = FrequencyEncoder(
    normalize=True,         # True = proportions, False = raw counts
    handle_unknown='zero',  # 'zero', 'nan', or 'error'
)
X_enc = encoder.fit_transform(X)
```

---

## Imputation

Missing value imputation fills `np.nan` entries before model training. Endgame provides four imputers from fast-and-simple to thorough-and-slow, plus an `AutoImputer` that selects a strategy based on the fraction of missing values.

### MICEImputer

Multiple Imputation by Chained Equations iteratively models each feature as a function of all other features using BayesianRidge by default. Handles arbitrary missingness patterns and is the standard choice for datasets with moderate-to-heavy missingness.

```python
from endgame.preprocessing import MICEImputer

imputer = MICEImputer(
    max_iter=10,
    initial_strategy='median',
    random_state=42,
    add_indicator=False,  # Set True to append binary missing-indicator columns
)
X_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

To use a custom estimator (e.g., a random forest):

```python
from sklearn.ensemble import RandomForestRegressor

imputer = MICEImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    max_iter=5,
)
```

### MissForestImputer

Uses a `RandomForestRegressor` as the MICE estimator. Non-parametric and robust to non-linear relationships between features. Slower than `MICEImputer` with BayesianRidge but often more accurate.

```python
from endgame.preprocessing import MissForestImputer

imputer = MissForestImputer(
    n_estimators=100,
    max_iter=10,
    n_jobs=-1,        # Use all cores
    random_state=42,
)
X_imputed = imputer.fit_transform(X_train)
```

### KNNImputer

Fills missing values using the mean of the k nearest observed neighbors. Effective when local structure in the data is informative.

```python
from endgame.preprocessing import KNNImputer

imputer = KNNImputer(
    n_neighbors=5,
    weights='uniform',   # 'uniform' or 'distance'
    add_indicator=False,
)
X_imputed = imputer.fit_transform(X_train)
```

### AutoImputer

Inspects the overall fraction of missing values and selects a strategy automatically:

- Less than 5% missing: `SimpleImputer` (median fill, fast)
- 5–30% missing: `KNNImputer`
- More than 30% missing: `MICEImputer`

Also runs an approximate Little's MCAR test and exposes the detected missingness type (`MCAR`, `MAR`, or `MNAR`) via `missingness_type_`.

```python
from endgame.preprocessing import AutoImputer

imputer = AutoImputer(strategy='auto', random_state=42)
X_imputed = imputer.fit_transform(X_train)

print(imputer.selected_strategy_)    # e.g. 'knn'
print(imputer.missingness_fraction_) # e.g. 0.12
print(imputer.missingness_type_)     # e.g. 'MAR'
```

To force a specific strategy: `strategy='simple'`, `'knn'`, `'mice'`, or `'missforest'`.

---

## Class Balancing

Imbalanced datasets require resampling before training. Endgame wraps imbalanced-learn with competition-tuned defaults. All resamplers expose a `fit_resample(X, y)` method and return `(X_resampled, y_resampled)`.

Requires `imbalanced-learn`: `pip install imbalanced-learn`.

### SMOTEResampler

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic minority samples by interpolating between a sample and one of its k nearest neighbors.

```python
from endgame.preprocessing import SMOTEResampler

smote = SMOTEResampler(
    k_neighbors=5,
    sampling_strategy='auto',  # 'auto', 'minority', float, or dict
    random_state=42,
)
X_res, y_res = smote.fit_resample(X_train, y_train)
```

### ADASYNResampler

Adaptive Synthetic Sampling generates more synthetic samples in regions where the classifier boundary is difficult. Focuses over-sampling effort on hard-to-classify minority examples.

```python
from endgame.preprocessing import ADASYNResampler

adasyn = ADASYNResampler(
    sampling_strategy='auto',
    n_neighbors=5,
    random_state=42,
)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

### BorderlineSMOTEResampler

Only generates synthetic samples from minority instances near the decision boundary (borderline instances). Avoids wasting capacity on clearly separable minority samples.

```python
from endgame.preprocessing import BorderlineSMOTEResampler

bsmote = BorderlineSMOTEResampler(
    k_neighbors=5,
    m_neighbors=10,
    kind='borderline-1',  # 'borderline-1' or 'borderline-2'
    random_state=42,
)
X_res, y_res = bsmote.fit_resample(X_train, y_train)
```

### Geometric Samplers

`MultivariateGaussianSMOTE` and `SimplicialSMOTE` generate synthetic samples that stay within the convex geometry of the minority class, avoiding extrapolation beyond the observed manifold. No additional dependencies required.

```python
from endgame.preprocessing import MultivariateGaussianSMOTE, SimplicialSMOTE

geo_smote = MultivariateGaussianSMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = geo_smote.fit_resample(X_train, y_train)
```

### AutoBalancer

Selects a resampling strategy automatically based on the imbalance ratio and dataset size. Evaluates multiple strategies and picks the best for the current dataset.

```python
from endgame.preprocessing import AutoBalancer

balancer = AutoBalancer(strategy='auto', random_state=42)
X_res, y_res = balancer.fit_resample(X_train, y_train)
```

### Pipeline Integration

Use `imblearn.pipeline.Pipeline` (not `sklearn.pipeline.Pipeline`) to combine resamplers with estimators:

```python
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from endgame.preprocessing import SMOTEResampler

pipe = Pipeline([
    ('smote', SMOTEResampler(k_neighbors=5, random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=200)),
])
pipe.fit(X_train, y_train)
```

For a complete list of available samplers, inspect `endgame.preprocessing.ALL_SAMPLERS`.

---

## Feature Engineering

### AutoAggregator

Generates group-level aggregation features ("magic features") that capture entity-level statistics. A core technique in many Kaggle winning solutions.

```python
from endgame.preprocessing import AutoAggregator

agg = AutoAggregator(
    group_cols=['customer_id'],
    agg_cols=['amount', 'quantity'],         # None = all numeric columns
    methods=['mean', 'std', 'min', 'max', 'skew'],
    rank_features=True,   # Adds within-group rank (key Optiver technique)
    diff_features=True,   # Adds deviation from group mean
    ratio_features=False, # Adds ratio to group mean
)
X_agg = agg.fit_transform(X_train)
```

Multi-key grouping:

```python
agg = AutoAggregator(
    group_cols=['store_id', 'product_category'],
    agg_cols=['sales'],
    methods=['mean', 'sum', 'count'],
)
```

Generated column names follow the pattern `{group}_{col}_{method}`, e.g. `customer_id_amount_mean`.

### InteractionFeatures

Creates pairwise interaction terms (products and ratios) between numeric features.

```python
from endgame.preprocessing import InteractionFeatures

interactions = InteractionFeatures(
    cols=['feature_a', 'feature_b', 'feature_c'],
    include_products=True,
    include_ratios=True,
)
X_inter = interactions.fit_transform(X_train)
```

### Temporal Feature Extraction

`TemporalFeatures` extracts datetime components and cyclical encodings from datetime columns. Cyclical encodings (sin/cos) are important for periodic features like hour-of-day or month so that December and January are close in feature space.

```python
from endgame.preprocessing import TemporalFeatures

tf = TemporalFeatures(
    datetime_cols=['timestamp'],  # None = auto-detect
    cyclical=True,                # Adds sin/cos for periodic features
    drop_original=False,
)
X_temporal = tf.fit_transform(X_train)
```

Extracted features include: `year`, `month`, `day`, `dayofweek`, `hour`, `minute`, `quarter`, `week_of_year`, `day_of_year`, `is_weekend`, `is_month_start`, `is_month_end`, and cyclical `sin`/`cos` variants for periodic components.

For time series contexts, `LagFeatures` and `RollingFeatures` are also available:

```python
from endgame.preprocessing import LagFeatures, RollingFeatures

lags = LagFeatures(cols=['value'], lags=[1, 7, 28])
rolls = RollingFeatures(cols=['value'], windows=[7, 28], methods=['mean', 'std'])

X = lags.fit_transform(X)
X = rolls.fit_transform(X)
```

---

## Noise Detection

### ConfidentLearningFilter

Implements the Confident Learning algorithm (Northcutt et al., 2021) to identify mislabeled training examples. Cross-validated predicted probabilities are used to estimate the joint distribution of noisy and true labels.

```python
from endgame.preprocessing import ConfidentLearningFilter

clf = ConfidentLearningFilter(
    base_estimator='rf',         # 'rf', 'xgboost', 'lgbm', or any sklearn classifier
    cv=5,
    threshold='auto',            # 'auto' = per-class average probability
    method='prune_by_class',     # 'prune_by_class', 'prune_by_noise_rate', or 'both'
    random_state=42,
)

noise_mask = clf.fit_detect(X_train, y_train)
print(f"Detected {noise_mask.sum()} noisy labels out of {len(y_train)}")

X_clean = X_train[~noise_mask]
y_clean = y_train[~noise_mask]
```

`ConsensusFilter` and `CrossValNoiseDetector` are also available for ensemble-based noise detection with multiple base estimators voting on which labels are suspect.

---

## Target Transformation

For regression tasks, transforming a skewed target distribution toward normality often improves model performance. `TargetTransformer` wraps any sklearn regressor and applies an invertible transform to `y` before fitting, then inverse-transforms predictions automatically.

### Supported Transforms

| Method | Requirements | Notes |
|---|---|---|
| `'log'` | All targets > 0 | Fast, interpretable |
| `'log1p'` | All targets >= 0 | Handles zeros |
| `'sqrt'` | All targets >= 0 | Mild compression |
| `'box_cox'` | All targets > 0 | Optimal power transform |
| `'yeo_johnson'` | Any targets | Works with negatives |
| `'quantile'` | Any targets | Maps to normal distribution |
| `'auto'` | Any targets | Selects via Shapiro-Wilk test |

```python
from endgame.preprocessing import TargetTransformer
from sklearn.ensemble import GradientBoostingRegressor

model = TargetTransformer(
    regressor=GradientBoostingRegressor(),
    method='yeo_johnson',  # or 'auto' to select automatically
)

model.fit(X_train, y_train)
preds = model.predict(X_test)  # Automatically inverse-transformed
```

Using `method='auto'` runs a Shapiro-Wilk normality test on the target and selects the transform that most improves normality:

```python
model = TargetTransformer(
    regressor=GradientBoostingRegressor(),
    method='auto',
)
model.fit(X_train, y_train)
print(model.selected_method_)  # e.g. 'log1p'
```

`TargetQuantileTransformer` applies quantile normalization as a standalone transformer (without wrapping a regressor):

```python
from endgame.preprocessing import TargetQuantileTransformer

qt = TargetQuantileTransformer(output_distribution='normal')
y_transformed = qt.fit_transform(y_train)
y_pred_original = qt.inverse_transform(y_pred_transformed)
```

---

## API Reference

See the [API Reference](../api/preprocessing) for the full parameter list of each class. The preprocessing module exports:

- **Encoding**: `SafeTargetEncoder`, `LeaveOneOutEncoder`, `CatBoostEncoder`, `FrequencyEncoder`
- **Imputation**: `SimpleImputer`, `IndicatorImputer`, `KNNImputer`, `MICEImputer`, `MissForestImputer`, `AutoImputer`
- **Class Balancing**: `SMOTEResampler`, `BorderlineSMOTEResampler`, `ADASYNResampler`, `SVMSMOTEResampler`, `KMeansSMOTEResampler`, `MultivariateGaussianSMOTE`, `SimplicialSMOTE`, `AutoBalancer`, and 10+ under-sampling and combined methods
- **Feature Engineering**: `AutoAggregator`, `InteractionFeatures`, `RankFeatures`, `TemporalFeatures`, `LagFeatures`, `RollingFeatures`
- **Noise Detection**: `ConfidentLearningFilter`, `ConsensusFilter`, `CrossValNoiseDetector`
- **Target Transformation**: `TargetTransformer`, `TargetQuantileTransformer`
- **Feature Selection**: `AdversarialFeatureSelector`, `PermutationImportanceSelector`, `NullImportanceSelector`
- **Discretization**: `BayesianDiscretizer`
