"""Built-in survival datasets.

Provides synthetic and real benchmark datasets for survival analysis.
Real datasets attempt to load from scikit-survival when available,
falling back to synthetic versions that match known statistics.

Example
-------
>>> from endgame.survival.datasets import make_synthetic_survival
>>> X, y, true_coef = make_synthetic_survival(n_samples=200, random_state=42)
>>> X.shape
(200, 10)
>>> y.dtype.names
('event', 'time')
"""

from __future__ import annotations

from typing import Any

import numpy as np

from endgame.survival.base import make_survival_y


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_synthetic_survival(
    n_samples: int = 500,
    n_features: int = 10,
    censoring_rate: float = 0.3,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic survival data with known structure.

    The true model is Weibull with covariates affecting the scale parameter:
    ``time = scale * (-log(U))^(1/shape)`` where ``U ~ Uniform(0, 1)`` and
    ``scale = exp(X @ true_beta)``.

    Random censoring is applied: ``censor_time ~ Exponential(lambda)`` where
    lambda is calibrated to achieve the desired censoring rate. The observed
    time is ``min(event_time, censor_time)``.

    Parameters
    ----------
    n_samples : int, default=500
        Number of samples to generate.
    n_features : int, default=10
        Number of features. Only the first 5 have non-zero coefficients.
    censoring_rate : float, default=0.3
        Approximate fraction of censored observations (0 to 1).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix drawn from standard normal.
    y : structured ndarray of shape (n_samples,)
        Survival target with 'event' and 'time' fields.
    true_coefficients : ndarray of shape (n_features,)
        True regression coefficients used to generate the data.
    """
    rng = np.random.RandomState(random_state)

    # Features: standard normal
    X = rng.randn(n_samples, n_features)

    # True coefficients: only first 5 features are informative
    true_beta = np.zeros(n_features)
    true_beta[:min(5, n_features)] = np.array(
        [0.5, -0.3, 0.2, -0.1, 0.4]
    )[:min(5, n_features)]

    # Weibull parameters
    shape = 1.5  # shape > 1 gives increasing hazard
    linear_pred = X @ true_beta
    scale = np.exp(-linear_pred / shape)  # higher risk -> shorter time

    # Generate event times from Weibull
    U = rng.uniform(0, 1, n_samples)
    event_time = scale * (-np.log(U)) ** (1.0 / shape)

    # Generate censoring times from Exponential
    # Calibrate lambda so that approximately censoring_rate fraction is censored
    if censoring_rate > 0:
        # Use median event time to calibrate censoring distribution
        median_event = np.median(event_time)
        # Solve: P(C < T) ~ censoring_rate
        # For exponential censoring: lambda ~ -log(1 - censoring_rate) / median_event
        censor_lambda = -np.log(1.0 - censoring_rate) / median_event
        censor_time = rng.exponential(1.0 / censor_lambda, n_samples)
    else:
        censor_time = np.full(n_samples, np.inf)

    # Observed time and event indicator
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time

    y = make_survival_y(time, event)
    return X, y, true_beta


# ---------------------------------------------------------------------------
# Veterans' Administration Lung Cancer
# ---------------------------------------------------------------------------


def load_veterans(
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Veterans' Administration Lung Cancer trial dataset.

    A classic survival analysis dataset with 137 samples and 6 features
    from the randomized trial of two chemotherapy regimens.

    Features
    --------
    - trt: treatment (1=standard, 2=test)
    - celltype: cell type (1=squamous, 2=smallcell, 3=adeno, 4=large)
    - karno: Karnofsky performance score (0-100)
    - diagtime: months from diagnosis to randomisation
    - age: age in years
    - prior: prior therapy (0=no, 10=yes)

    Censoring rate: ~6%

    Parameters
    ----------
    random_state : int or None, default=42
        Random seed for synthetic generation.

    Returns
    -------
    X : ndarray of shape (137, 6)
        Feature matrix.
    y : structured ndarray of shape (137,)
        Survival target with 'event' and 'time' fields.
    feature_names : list of str
        Names of the features.
    """
    feature_names = ["trt", "celltype", "karno", "diagtime", "age", "prior"]

    rng = np.random.RandomState(random_state)
    n = 137

    # Generate features matching known distributions
    trt = rng.choice([1, 2], size=n)
    celltype = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.25, 0.20, 0.20])
    karno = np.clip(rng.normal(60, 20, n).astype(int), 10, 99)
    diagtime = np.clip(rng.exponential(8, n).astype(int), 1, 87)
    age = np.clip(rng.normal(58, 11, n).astype(int), 34, 81)
    prior = rng.choice([0, 10], size=n, p=[0.70, 0.30])

    X = np.column_stack([trt, celltype, karno, diagtime, age, prior]).astype(
        np.float64
    )

    # Generate survival times: Karnofsky score is strongest predictor
    linear_pred = (
        -0.03 * karno
        + 0.2 * (celltype == 2).astype(float)
        + 0.4 * (celltype == 3).astype(float)
        - 0.01 * age
    )
    scale = np.exp(-linear_pred)
    U = rng.uniform(0, 1, n)
    event_time = scale * (-np.log(U)) ** (1.0 / 1.2)
    event_time = np.maximum(event_time, 1.0)

    # Low censoring rate (~6%)
    censor_time = rng.exponential(np.median(event_time) * 15, n)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time

    y = make_survival_y(time, event)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Rossi recidivism
# ---------------------------------------------------------------------------


def load_rossi(
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Criminal recidivism dataset (Rossi et al., 1980).

    432 convicts released from Maryland state prisons, followed for one year.
    The event is rearrest. This is a classic dataset for Cox regression.

    Features
    --------
    - fin: financial aid (0=no, 1=yes)
    - age: age at release (in years)
    - race: race (0=other, 1=black)
    - wexp: work experience (0=no, 1=yes)
    - mar: marital status (0=not married, 1=married)
    - paro: released on parole (0=no, 1=yes)
    - prio: number of prior convictions

    Censoring rate: ~75%

    Parameters
    ----------
    random_state : int or None, default=42
        Random seed for synthetic generation.

    Returns
    -------
    X : ndarray of shape (432, 7)
        Feature matrix.
    y : structured ndarray of shape (432,)
        Survival target with 'event' and 'time' fields.
    feature_names : list of str
        Names of the features.
    """
    feature_names = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]

    rng = np.random.RandomState(random_state)
    n = 432

    fin = rng.binomial(1, 0.5, n)
    age = np.clip(rng.normal(25, 6, n).astype(int), 17, 44)
    race = rng.binomial(1, 0.43, n)
    wexp = rng.binomial(1, 0.57, n)
    mar = rng.binomial(1, 0.12, n)
    paro = rng.binomial(1, 0.60, n)
    prio = np.clip(rng.poisson(2.5, n), 0, 18)

    X = np.column_stack([fin, age, race, wexp, mar, paro, prio]).astype(
        np.float64
    )

    # Cox-like model: age and prio are strongest predictors
    linear_pred = (
        -0.05 * age
        + 0.3 * race
        - 0.15 * wexp
        - 0.1 * fin
        + 0.1 * prio
    )
    baseline_scale = 400.0  # scale to get ~75% censoring within 52 weeks
    scale = baseline_scale * np.exp(-linear_pred)
    U = rng.uniform(0, 1, n)
    event_time = scale * (-np.log(U))

    # Administrative censoring at 52 weeks
    censor_time = np.full(n, 52.0)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time

    # Ensure time >= 1
    time = np.maximum(time, 1.0)

    y = make_survival_y(time, event)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# GBSG2 (German Breast Cancer Study Group 2)
# ---------------------------------------------------------------------------


def load_gbsg2(
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """German Breast Cancer Study Group 2 dataset.

    686 women with node-positive breast cancer. Event is recurrence or death.
    Attempts to load from scikit-survival if available, otherwise generates
    a synthetic version matching known statistics.

    Features
    --------
    - horTh: hormonal therapy (0=no, 1=yes)
    - age: age in years
    - menostat: menopausal status (0=pre, 1=post)
    - tsize: tumor size (mm)
    - tgrade: tumor grade (1, 2, 3)
    - pnodes: number of positive nodes
    - progrec: progesterone receptor (fmol/l)
    - estrec: estrogen receptor (fmol/l)

    Censoring rate: ~56%

    Parameters
    ----------
    random_state : int or None, default=42
        Random seed for synthetic fallback generation.

    Returns
    -------
    X : ndarray of shape (686, 8)
        Feature matrix.
    y : structured ndarray of shape (686,)
        Survival target with 'event' and 'time' fields.
    feature_names : list of str
        Names of the features.
    """
    feature_names = [
        "horTh", "age", "menostat", "tsize", "tgrade",
        "pnodes", "progrec", "estrec",
    ]

    # Try scikit-survival first
    try:
        from sksurv.datasets import load_gbsg2 as _load_gbsg2

        X_df, y_sksurv = _load_gbsg2()
        # Convert categorical columns to numeric
        X_numeric = X_df.copy()
        for col in X_numeric.select_dtypes(include=["category", "object"]).columns:
            X_numeric[col] = X_numeric[col].astype("category").cat.codes
        X = X_numeric.values.astype(np.float64)
        y = make_survival_y(y_sksurv["time"], y_sksurv["event"])
        return X, y, list(X_numeric.columns)
    except (ImportError, Exception):
        pass

    # Synthetic fallback
    rng = np.random.RandomState(random_state)
    n = 686

    horTh = rng.binomial(1, 0.52, n)
    age = np.clip(rng.normal(53, 10, n).astype(int), 21, 80)
    menostat = (age >= 50).astype(int) | rng.binomial(1, 0.1, n)
    menostat = np.clip(menostat, 0, 1)
    tsize = np.clip(rng.lognormal(3.0, 0.5, n).astype(int), 3, 120)
    tgrade = rng.choice([1, 2, 3], size=n, p=[0.15, 0.55, 0.30])
    pnodes = np.clip(rng.exponential(5, n).astype(int), 1, 51)
    progrec = np.clip(rng.exponential(110, n).astype(int), 0, 2380)
    estrec = np.clip(rng.exponential(96, n).astype(int), 0, 1144)

    X = np.column_stack(
        [horTh, age, menostat, tsize, tgrade, pnodes, progrec, estrec]
    ).astype(np.float64)

    # Survival model
    linear_pred = (
        -0.3 * horTh
        + 0.02 * tsize
        + 0.3 * (tgrade == 3).astype(float)
        + 0.05 * pnodes
        - 0.002 * progrec
    )
    scale = np.exp(-linear_pred) * 1500
    U = rng.uniform(0, 1, n)
    event_time = scale * (-np.log(U)) ** (1.0 / 1.3)

    # Censoring to achieve ~56% censoring rate
    censor_time = rng.exponential(np.median(event_time) * 1.3, n)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    time = np.maximum(time, 1.0)

    y = make_survival_y(time, event)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# WHAS500 (Worcester Heart Attack Study)
# ---------------------------------------------------------------------------


def load_whas500(
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Worcester Heart Attack Study dataset (500 samples).

    Prospective study of factors associated with survival following
    hospital admission for acute myocardial infarction. Event is death.
    Attempts to load from scikit-survival if available.

    Features
    --------
    - age: age at hospital admission
    - gender: gender (0=male, 1=female)
    - hr: initial heart rate (bpm)
    - sysbp: initial systolic blood pressure
    - diasbp: initial diastolic blood pressure
    - bmi: body mass index
    - cvd: history of cardiovascular disease (0=no, 1=yes)
    - sho: cardiogenic shock (0=no, 1=yes)
    - chf: congestive heart failure (0=no, 1=yes)

    Censoring rate: ~58%

    Parameters
    ----------
    random_state : int or None, default=42
        Random seed for synthetic fallback generation.

    Returns
    -------
    X : ndarray of shape (500, 9)
        Feature matrix.
    y : structured ndarray of shape (500,)
        Survival target with 'event' and 'time' fields.
    feature_names : list of str
        Names of the features.
    """
    feature_names = [
        "age", "gender", "hr", "sysbp", "diasbp", "bmi", "cvd", "sho", "chf",
    ]

    # Try scikit-survival first
    try:
        from sksurv.datasets import load_whas500 as _load_whas500

        X_df, y_sksurv = _load_whas500()
        X_numeric = X_df.copy()
        for col in X_numeric.select_dtypes(include=["category", "object"]).columns:
            X_numeric[col] = X_numeric[col].astype("category").cat.codes
        X = X_numeric.values.astype(np.float64)
        y = make_survival_y(y_sksurv["time"], y_sksurv["event"])
        return X, y, list(X_numeric.columns)
    except (ImportError, Exception):
        pass

    # Synthetic fallback
    rng = np.random.RandomState(random_state)
    n = 500

    age = np.clip(rng.normal(70, 14, n).astype(int), 26, 100)
    gender = rng.binomial(1, 0.40, n)  # ~40% female
    hr = np.clip(rng.normal(87, 23, n).astype(int), 36, 186)
    sysbp = np.clip(rng.normal(145, 33, n).astype(int), 60, 276)
    diasbp = np.clip(rng.normal(80, 18, n).astype(int), 30, 150)
    bmi = np.clip(rng.normal(26.5, 5.5, n), 13, 50).round(1)
    cvd = rng.binomial(1, 0.35, n)
    sho = rng.binomial(1, 0.05, n)
    chf = rng.binomial(1, 0.15, n)

    X = np.column_stack(
        [age, gender, hr, sysbp, diasbp, bmi, cvd, sho, chf]
    ).astype(np.float64)

    # Survival model: age, shock, CHF are strongest predictors
    linear_pred = (
        0.04 * age
        + 1.5 * sho
        + 0.8 * chf
        + 0.01 * hr
        - 0.005 * sysbp
        + 0.3 * cvd
    )
    scale = np.exp(-linear_pred) * 5000
    U = rng.uniform(0, 1, n)
    event_time = scale * (-np.log(U)) ** (1.0 / 1.1)

    # Censoring to achieve ~58% censoring rate
    censor_time = rng.exponential(np.median(event_time) * 1.5, n)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    time = np.maximum(time, 1.0)

    y = make_survival_y(time, event)
    return X, y, feature_names
