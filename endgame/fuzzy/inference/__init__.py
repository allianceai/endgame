"""Fuzzy inference systems: Mamdani, TSK, ANFIS, and Wang-Mendel.

Provides classic fuzzy inference engines with sklearn-compatible APIs
for both regression and classification tasks.

Example
-------
>>> from endgame.fuzzy.inference import TSKRegressor, MamdaniFIS
>>> model = TSKRegressor(n_mfs=3, order=1)
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)
"""

from endgame.fuzzy.inference.mamdani import MamdaniClassifier, MamdaniFIS
from endgame.fuzzy.inference.tsk import TSKClassifier, TSKRegressor
from endgame.fuzzy.inference.wang_mendel import WangMendelRegressor

# ANFIS requires PyTorch; import gracefully
try:
    from endgame.fuzzy.inference.anfis import ANFISClassifier, ANFISRegressor
except ImportError:
    pass

__all__ = [
    "MamdaniFIS",
    "MamdaniClassifier",
    "TSKRegressor",
    "TSKClassifier",
    "ANFISRegressor",
    "ANFISClassifier",
    "WangMendelRegressor",
]
