"""Preprocessing module: Feature engineering, encoding, transformations, and class balancing."""

from endgame.preprocessing.aggregation import (
    AutoAggregator,
    InteractionFeatures,
    RankFeatures,
)
from endgame.preprocessing.discretize import (
    BayesianDiscretizer,
)
from endgame.preprocessing.encoding import (
    CatBoostEncoder,
    FrequencyEncoder,
    LeaveOneOutEncoder,
    SafeTargetEncoder,
)
from endgame.preprocessing.imbalance import (
    ALL_SAMPLERS,
    COMBINED_SAMPLERS,
    GENERATIVE_SAMPLERS,
    # Modern category dicts (empty until submodules populate them)
    GEOMETRIC_SAMPLERS,
    LLM_SAMPLERS,
    # Sampler dictionaries
    OVER_SAMPLERS,
    UNDER_SAMPLERS,
    ADASYNResampler,
    AllKNNUnderSampler,
    # Auto-balancer
    AutoBalancer,
    BorderlineSMOTEResampler,
    ClusterCentroidsUnderSampler,
    CondensedNearestNeighbour,
    # Under-sampling
    EditedNearestNeighbours,
    InstanceHardnessThresholdSampler,
    KMeansSMOTEResampler,
    NearMissUnderSampler,
    NeighbourhoodCleaningRule,
    OneSidedSelectionUnderSampler,
    RandomOverSampler,
    RandomUnderSampler,
    # Combined
    SMOTEENNResampler,
    # Over-sampling
    SMOTEResampler,
    SMOTETomekResampler,
    SVMSMOTEResampler,
    TomekLinksUnderSampler,
    get_class_distribution,
    # Utilities
    get_imbalance_ratio,
)
from endgame.preprocessing.imbalance_geometric import (
    GEOMETRIC_SAMPLERS as _GEO_SAMPLERS,
)

# Geometric samplers (no optional deps — always available)
from endgame.preprocessing.imbalance_geometric import (
    CVSMOTEResampler,
    MultivariateGaussianSMOTE,
    OverlapRegionDetector,
    SimplicialSMOTE,
)
from endgame.preprocessing.imputation import (
    AutoImputer,
    IndicatorImputer,
    KNNImputer,
    MICEImputer,
    MissForestImputer,
    SimpleImputer,
)
from endgame.preprocessing.noise_detection import (
    ConfidentLearningFilter,
    ConsensusFilter,
    CrossValNoiseDetector,
)
from endgame.preprocessing.selection import (
    AdversarialFeatureSelector,
    NullImportanceSelector,
    PermutationImportanceSelector,
)
from endgame.preprocessing.target_transform import (
    TargetQuantileTransformer,
    TargetTransformer,
)
from endgame.preprocessing.temporal import (
    LagFeatures,
    RollingFeatures,
    TemporalFeatures,
)

GEOMETRIC_SAMPLERS.update(_GEO_SAMPLERS)
ALL_SAMPLERS.update(_GEO_SAMPLERS)

__all__ = [
    # Encoding
    "SafeTargetEncoder",
    "LeaveOneOutEncoder",
    "CatBoostEncoder",
    "FrequencyEncoder",
    # Aggregation
    "AutoAggregator",
    "InteractionFeatures",
    "RankFeatures",
    # Temporal
    "TemporalFeatures",
    "LagFeatures",
    "RollingFeatures",
    # Selection
    "AdversarialFeatureSelector",
    "PermutationImportanceSelector",
    "NullImportanceSelector",
    # Discretization
    "BayesianDiscretizer",
    # Imputation
    "SimpleImputer",
    "IndicatorImputer",
    "KNNImputer",
    "MICEImputer",
    "MissForestImputer",
    "AutoImputer",
    # Target transformation
    "TargetTransformer",
    "TargetQuantileTransformer",
    # Noise detection
    "ConfidentLearningFilter",
    "ConsensusFilter",
    "CrossValNoiseDetector",
    # Class Balancing - Over-sampling
    "SMOTEResampler",
    "BorderlineSMOTEResampler",
    "ADASYNResampler",
    "SVMSMOTEResampler",
    "KMeansSMOTEResampler",
    "RandomOverSampler",
    # Class Balancing - Under-sampling
    "EditedNearestNeighbours",
    "AllKNNUnderSampler",
    "TomekLinksUnderSampler",
    "RandomUnderSampler",
    "NearMissUnderSampler",
    "CondensedNearestNeighbour",
    "OneSidedSelectionUnderSampler",
    "NeighbourhoodCleaningRule",
    "InstanceHardnessThresholdSampler",
    "ClusterCentroidsUnderSampler",
    # Class Balancing - Combined
    "SMOTEENNResampler",
    "SMOTETomekResampler",
    # Class Balancing - Geometric (modern)
    "MultivariateGaussianSMOTE",
    "SimplicialSMOTE",
    "CVSMOTEResampler",
    "OverlapRegionDetector",
    # Auto-balancer
    "AutoBalancer",
    # Utilities
    "get_imbalance_ratio",
    "get_class_distribution",
    # Sampler dictionaries
    "OVER_SAMPLERS",
    "UNDER_SAMPLERS",
    "COMBINED_SAMPLERS",
    "GEOMETRIC_SAMPLERS",
    "GENERATIVE_SAMPLERS",
    "LLM_SAMPLERS",
    "ALL_SAMPLERS",
]

# DAE requires PyTorch
try:
    from endgame.preprocessing.dae import DenoisingAutoEncoder
    __all__.append("DenoisingAutoEncoder")
except (ImportError, NameError):
    pass

# Generative samplers require PyTorch/XGBoost/ctgan
try:
    from endgame.preprocessing.imbalance_generative import (
        GENERATIVE_SAMPLERS as _GEN_SAMPLERS,
    )
    from endgame.preprocessing.imbalance_generative import (
        CTGANResampler,
        ForestFlowResampler,
        TabDDPMResampler,
        TabSynResampler,
    )
    GENERATIVE_SAMPLERS.update(_GEN_SAMPLERS)
    ALL_SAMPLERS.update(_GEN_SAMPLERS)
    __all__.extend([
        "CTGANResampler",
        "ForestFlowResampler",
        "TabDDPMResampler",
        "TabSynResampler",
    ])
except (ImportError, NameError):
    pass

# LLM samplers require transformers
try:
    from endgame.preprocessing.imbalance_llm import (
        LLM_SAMPLERS as _LLM_SAMPLERS,
    )
    from endgame.preprocessing.imbalance_llm import (
        GReaTResampler,
    )
    LLM_SAMPLERS.update(_LLM_SAMPLERS)
    ALL_SAMPLERS.update(_LLM_SAMPLERS)
    __all__.append("GReaTResampler")
except (ImportError, NameError):
    pass
