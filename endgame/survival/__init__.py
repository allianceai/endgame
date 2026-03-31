"""Survival analysis module.

Provides sklearn-compatible survival estimators covering nonparametric,
parametric, semi-parametric, tree-based, gradient-boosted, deep learning,
and competing risks methods, plus survival-specific metrics, validation,
and ensembles.

Submodules
----------
nonparametric : Kaplan-Meier, Nelson-Aalen (pure NumPy, no dependencies)
parametric : Weibull, LogNormal, LogLogistic, Exponential AFT models
cox : Cox Proportional Hazards (pure NumPy + optional lifelines/sksurv)
trees : Random Survival Forest, Gradient Boosted Survival (scikit-survival)
gbdt : XGBoost/LightGBM/CatBoost with survival objectives
neural : DeepSurv, DeepHit, LogisticHazard, MTLR (PyTorch/pycox)
competing_risks : Aalen-Johansen, Cause-Specific Cox, Fine-Gray
ensemble : Survival stacking and hill climbing ensembles
metrics : Concordance index, Brier score, time-dependent AUC
validation : Survival cross-validation strategies
datasets : Benchmark survival datasets

Examples
--------
>>> import endgame as eg
>>> from endgame.survival import (
...     KaplanMeierEstimator, CoxPHRegressor,
...     concordance_index, make_survival_y,
... )
>>> import numpy as np
>>> rng = np.random.RandomState(42)
>>> X = rng.randn(200, 5)
>>> time = rng.exponential(10, 200)
>>> event = rng.binomial(1, 0.7, 200).astype(bool)
>>> y = make_survival_y(time, event)
>>>
>>> # Kaplan-Meier
>>> km = KaplanMeierEstimator()
>>> km.fit(y)
>>> print(f"Median survival: {km.median_survival_time_:.2f}")
>>>
>>> # Cox PH
>>> cox = CoxPHRegressor(penalizer=0.1)
>>> cox.fit(X, y)
>>> risk = cox.predict(X)
>>> print(f"C-index: {concordance_index(y, risk):.3f}")
"""

from __future__ import annotations

# Base classes and utilities (always available)
from endgame.survival.base import (
    BaseSurvivalEstimator,
    SurvivalMixin,
    SurvivalPrediction,
    make_survival_y,
)

# Nonparametric estimators (pure NumPy, always available)
from endgame.survival.nonparametric import (
    KaplanMeierEstimator,
    NelsonAalenEstimator,
)

# Metrics (concordance_index is pure NumPy, always available)
from endgame.survival.metrics import (
    brier_score,
    calibration_curve_survival,
    concordance_index,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

# Validation (always available)
from endgame.survival.validation import (
    SurvivalStratifiedKFold,
    SurvivalTimeSeriesSplit,
    evaluate_survival,
    survival_train_test_split,
)

# Datasets (always available, with optional fallbacks)
from endgame.survival.datasets import (
    load_gbsg2,
    load_rossi,
    load_veterans,
    load_whas500,
    make_synthetic_survival,
)

# Ensemble (always available, uses base models)
from endgame.survival.ensemble import (
    SurvivalHillClimbingEnsemble,
    SurvivalStackingEnsemble,
    SurvivalVotingEnsemble,
)

__all__ = [
    # Base
    "BaseSurvivalEstimator",
    "SurvivalMixin",
    "SurvivalPrediction",
    "make_survival_y",
    # Nonparametric
    "KaplanMeierEstimator",
    "NelsonAalenEstimator",
    # Metrics
    "concordance_index",
    "concordance_index_censored",
    "integrated_brier_score",
    "brier_score",
    "cumulative_dynamic_auc",
    "calibration_curve_survival",
    # Validation
    "SurvivalStratifiedKFold",
    "SurvivalTimeSeriesSplit",
    "evaluate_survival",
    "survival_train_test_split",
    # Datasets
    "make_synthetic_survival",
    "load_veterans",
    "load_rossi",
    "load_gbsg2",
    "load_whas500",
    # Ensemble
    "SurvivalStackingEnsemble",
    "SurvivalHillClimbingEnsemble",
    "SurvivalVotingEnsemble",
]

# ===== OPTIONAL DEPENDENCIES =====

# Parametric AFT models (scipy-based, should always work)
try:
    from endgame.survival.parametric import (
        ExponentialRegressor,
        LogLogisticAFTRegressor,
        LogNormalAFTRegressor,
        WeibullAFTRegressor,
    )

    __all__.extend(
        [
            "WeibullAFTRegressor",
            "LogNormalAFTRegressor",
            "LogLogisticAFTRegressor",
            "ExponentialRegressor",
        ]
    )
except ImportError:
    pass

# Cox models (pure NumPy baseline, optionally uses lifelines/sksurv)
try:
    from endgame.survival.cox import (
        CoxNetRegressor,
        CoxPHRegressor,
    )

    __all__.extend(
        [
            "CoxPHRegressor",
            "CoxNetRegressor",
        ]
    )
except ImportError:
    pass

# scikit-survival tree wrappers (optional)
try:
    from endgame.survival.trees import (
        ExtraSurvivalTreesRegressor,
        GradientBoostedSurvivalRegressor,
        RandomSurvivalForestRegressor,
    )

    __all__.extend(
        [
            "RandomSurvivalForestRegressor",
            "ExtraSurvivalTreesRegressor",
            "GradientBoostedSurvivalRegressor",
        ]
    )
except ImportError:
    pass

# GBDT survival wrappers (XGBoost/LightGBM/CatBoost)
try:
    from endgame.survival.gbdt import SurvivalGBDTWrapper

    __all__.append("SurvivalGBDTWrapper")
except ImportError:
    pass

# Deep learning survival models (PyTorch/pycox, optional)
try:
    from endgame.survival.neural import (
        CoxTimeRegressor,
        DeepHitRegressor,
        DeepSurvRegressor,
        LogisticHazardRegressor,
        MTLRRegressor,
    )

    __all__.extend(
        [
            "DeepSurvRegressor",
            "DeepHitRegressor",
            "LogisticHazardRegressor",
            "MTLRRegressor",
            "CoxTimeRegressor",
        ]
    )
except ImportError:
    pass

# Competing risks models
try:
    from endgame.survival.competing_risks import (
        AalenJohansenEstimator,
        CauseSpecificCoxRegressor,
        FineGrayRegressor,
    )

    __all__.extend(
        [
            "AalenJohansenEstimator",
            "CauseSpecificCoxRegressor",
            "FineGrayRegressor",
        ]
    )
except ImportError:
    pass
