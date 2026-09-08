"""Modern fuzzy learning systems.

Includes hierarchical TSK, MBGD-RDA training, FCM-based initialization,
fuzzy attention mechanisms, differentiable fuzzy systems, TSK with
privileged information, and self-evolving Type-2 networks.
"""

from endgame.fuzzy.modern.fcm_rdpa import FCMRDpAClassifier, FCMRDpARegressor
from endgame.fuzzy.modern.htsk import HTSKClassifier, HTSKRegressor
from endgame.fuzzy.modern.mbgd_rda import MBGDRDARegressor, MBGDRDATrainer
from endgame.fuzzy.modern.seit2fnn import SEIT2FNNClassifier
from endgame.fuzzy.modern.tsk_plus import TSKPlusClassifier, TSKPlusRegressor


# PyTorch-dependent imports (lazy)
def __getattr__(name: str):
    if name in ("FuzzyAttentionLayer", "FuzzyAttentionClassifier"):
        from endgame.fuzzy.modern.fuzzy_attention import (
            FuzzyAttentionClassifier,
            FuzzyAttentionLayer,
        )
        return {"FuzzyAttentionLayer": FuzzyAttentionLayer,
                "FuzzyAttentionClassifier": FuzzyAttentionClassifier}[name]
    if name == "DifferentiableFuzzySystem":
        from endgame.fuzzy.modern.differentiable import DifferentiableFuzzySystem
        return DifferentiableFuzzySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HTSKClassifier",
    "HTSKRegressor",
    "MBGDRDATrainer",
    "MBGDRDARegressor",
    "FCMRDpAClassifier",
    "FCMRDpARegressor",
    "FuzzyAttentionLayer",
    "FuzzyAttentionClassifier",
    "DifferentiableFuzzySystem",
    "TSKPlusClassifier",
    "TSKPlusRegressor",
    "SEIT2FNNClassifier",
]
