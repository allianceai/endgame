# Time Series Guide

Endgame provides forecasting and time series classification through the
`endgame.timeseries` module. Baseline, statistical, and neural forecasters share
a common interface. Time series classification models are sklearn-compatible and
support `fit` / `predict`.

## Baseline Forecasters

Baseline forecasters require no dependencies beyond numpy and are useful as
sanity-check benchmarks.

### NaiveForecaster

Predicts the last observed value for every horizon step (random walk baseline).

```python
from endgame.timeseries import NaiveForecaster
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

model = NaiveForecaster()
model.fit(y)

forecast = model.predict(horizon=7)   # predict 7 steps ahead
print(forecast)   # [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```

### SeasonalNaiveForecaster

Predicts by repeating the last observed seasonal cycle. Appropriate for strongly
seasonal data (daily, weekly, monthly patterns).

```python
from endgame.timeseries import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=7)  # weekly seasonality
model.fit(y_train)

forecast = model.predict(horizon=14)  # 2 weeks ahead
```

### ThetaForecaster

The Theta method decomposes the series into two Theta lines and forecasts each
separately before combining. It placed first in the M3 competition for monthly
and quarterly data.

```python
from endgame.timeseries import ThetaForecaster

model = ThetaForecaster(
    theta=2.0,            # standard Theta-2 decomposition
    season_length=12,     # 0 to disable seasonal adjustment
)
model.fit(y_train)

forecast = model.predict(horizon=12)
conf_int = model.predict_interval(horizon=12, alpha=0.05)  # 95% intervals
lower, upper = conf_int[:, 0], conf_int[:, 1]
```

## Statistical Models (statsforecast)

Statistical models require `statsforecast` (`pip install statsforecast`). They are
fast enough for batch forecasting across thousands of series.

### AutoARIMAForecaster

Automatically selects ARIMA order (p, d, q) and seasonal components via
information criteria. Supports exogenous regressors through the `X` argument.

```python
from endgame.timeseries import AutoARIMAForecaster

model = AutoARIMAForecaster(
    season_length=12,   # monthly data with annual seasonality
    approximation=True, # faster search for long series
    n_jobs=-1,          # parallel series fitting
)
model.fit(y_train)

forecast = model.predict(horizon=12)
conf_int = model.predict_interval(horizon=12, alpha=0.05)
```

### AutoETSForecaster

Automatically selects the best Exponential Smoothing (ETS) model including
error, trend, and seasonality components.

```python
from endgame.timeseries import AutoETSForecaster

model = AutoETSForecaster(season_length=12)
model.fit(y_train)

forecast = model.predict(horizon=6)
```

### MSTLForecaster

Multiple Seasonal and Trend decomposition using Loess (MSTL). Handles series
with multiple seasonality periods, for example hourly data with both daily and
weekly cycles.

```python
from endgame.timeseries import MSTLForecaster

model = MSTLForecaster(
    season_lengths=[24, 24 * 7],  # daily and weekly cycles in hourly data
    trend_kwargs={'window': 101},
)
model.fit(y_train)

forecast = model.predict(horizon=48)
```

## Neural Models (Darts)

Neural forecasters require `darts` and `torch` (`pip install darts`). They accept
both univariate and multivariate series.

### NBEATSForecaster

N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series) learns a
set of basis functions that decompose the forecast into trend and seasonality.

```python
from endgame.timeseries import NBEATSForecaster

model = NBEATSForecaster(
    input_chunk_length=24,    # lookback window
    output_chunk_length=12,   # forecast horizon
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    n_epochs=100,
    random_state=42,
)

model.fit(y_train)
forecast = model.predict(horizon=12)
```

### TFTForecaster

Temporal Fusion Transformer combines recurrent layers, attention, and gating to
handle static covariates, known future inputs, and observed past covariates.

```python
from endgame.timeseries import TFTForecaster

model = TFTForecaster(
    input_chunk_length=48,
    output_chunk_length=12,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    n_epochs=50,
    add_relative_index=True,   # adds a time index as a future covariate
    random_state=42,
)

model.fit(y_train, future_covariates=holidays)
forecast = model.predict(horizon=12, future_covariates=future_holidays)
```

### PatchTSTForecaster

PatchTST is a Transformer model that divides the input series into patches and
applies self-attention across them. It achieves strong results on long-horizon
benchmarks with lower memory cost than full-attention models.

```python
from endgame.timeseries import PatchTSTForecaster

model = PatchTSTForecaster(
    input_chunk_length=336,    # 14-day lookback for hourly data
    output_chunk_length=96,    # 4-day forecast
    patch_length=16,
    stride=8,
    d_model=128,
    n_heads=8,
    n_epochs=30,
    random_state=42,
)

model.fit(y_train)
forecast = model.predict(horizon=96)
```

## Time Series Classification

Time series classification models accept 3D arrays of shape
`(n_samples, n_channels, series_length)` and return class labels.

### MiniRocketClassifier

MiniROCKET transforms each series using ~10 000 random convolutional kernels and
trains a ridge classifier on the resulting features. It is extremely fast while
remaining competitive with much heavier models.

```python
from endgame.timeseries import MiniRocketClassifier

# X shape: (n_samples, n_channels, series_length)
# y shape: (n_samples,)
model = MiniRocketClassifier(
    num_kernels=10_000,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
proba = model.predict_proba(X_test)
```

### HydraClassifier

Hydra is a dictionary-based method that applies competing convolutional kernels and
counts "wins" to form features. It is faster than MiniROCKET on GPU and handles
multivariate series natively.

```python
from endgame.timeseries import HydraClassifier

model = HydraClassifier(
    k=8,              # groups of convolutional kernels
    g=64,             # kernels per group
    random_state=42,
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Cross-Validation for Time Series

Use `PurgedTimeSeriesSplit` from `endgame.validation` to avoid leakage when
evaluating forecasting or classification models on sequential data.

```python
from endgame.validation import PurgedTimeSeriesSplit
from sklearn.model_selection import cross_val_score
from endgame.timeseries import MiniRocketClassifier

cv = PurgedTimeSeriesSplit(
    n_splits=5,
    gap=10,           # samples purged between train and test fold
    embargo=0.01,     # fraction of series length embargoed after each fold
)

scores = cross_val_score(
    MiniRocketClassifier(),
    X, y,
    cv=cv,
    scoring='accuracy',
)
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
```

For panel data (multiple series), use `GroupKFold` or `StratifiedGroupKFold` from
`endgame.validation` to ensure all observations from a given series stay in the
same fold.

```python
from endgame.validation import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5)

# groups: array mapping each sample to its series ID
scores = cross_val_score(
    MiniRocketClassifier(),
    X, y,
    cv=cv,
    groups=series_ids,
    scoring='roc_auc',
)
```

## Choosing a Forecaster

| Model | When to use |
|---|---|
| `NaiveForecaster` | Sanity-check baseline; very short series |
| `SeasonalNaiveForecaster` | Strongly seasonal data; benchmark for seasonal models |
| `ThetaForecaster` | Monthly/quarterly data; strong M3 performer; no dependencies |
| `AutoARIMAForecaster` | Standard univariate forecasting; exogenous regressors needed |
| `AutoETSForecaster` | Multiplicative seasonality; automatic model selection |
| `MSTLForecaster` | Multiple seasonalities (hourly, sub-daily data) |
| `NBEATSForecaster` | Long series; interpretable trend/seasonality decomposition |
| `TFTForecaster` | Covariates (static, future, past); attention-based attribution |
| `PatchTSTForecaster` | Long-horizon benchmarks; memory-efficient Transformer |

## See Also

- [API Reference: timeseries](../api/timeseries)
- [Signal Processing Guide](signal.md) for pre-processing raw sensor data
- [Validation Guide](../api/validation) for cross-validation strategies
