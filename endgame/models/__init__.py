"""Models module: GBDT wrappers, custom trees, EBMs, linear models, neural networks, and Bayesian classifiers."""

from endgame.models.wrappers import (
    CatBoostWrapper,
    GBDTWrapper,
    LGBMWrapper,
    XGBWrapper,
)

# NGBoost for probabilistic prediction
try:
    from endgame.models.ngboost import (
        NGBoostClassifier,
        NGBoostRegressor,
    )
    _HAS_NGBOOST = True
except ImportError:
    _HAS_NGBOOST = False
from endgame.models.bayesian import (
    AutoSLE,
    EBMCClassifier,
    ESKDBClassifier,
    KDBClassifier,
    TANClassifier,
)
from endgame.models.ebm import (
    EBMClassifier,
    EBMRegressor,
    show_explanation,
)
from endgame.models.linear import (
    MARSClassifier,
    MARSRegressor,
)
from endgame.models.rules import (
    FURIAClassifier,
    FuzzyCondition,
    FuzzyRule,
    RuleFitClassifier,
    RuleFitRegressor,
)
from endgame.models.trees import _HAS_TREEPLE as _HAS_TREEPLE_TREES
from endgame.models.trees import (
    C50Classifier,
    C50Ensemble,
    CubistRegressor,
    EvolutionaryTreeClassifier,
    EvolutionaryTreeRegressor,
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    QuantileRegressorForest,
    RotationForestClassifier,
    RotationForestRegressor,
    interval_coverage,
    interval_width,
    pinball_loss,
)

if _HAS_TREEPLE_TREES:
    from endgame.models.trees import (
        ExtraObliqueRandomForestClassifier,
        ExtraObliqueRandomForestRegressor,
        HonestForestClassifier,
        PatchObliqueRandomForestClassifier,
        PatchObliqueRandomForestRegressor,
    )

__all__ = [
    # GBDT wrappers
    "GBDTWrapper",
    "LGBMWrapper",
    "XGBWrapper",
    "CatBoostWrapper",
    # Custom trees
    "RotationForestClassifier",
    "RotationForestRegressor",
    "C50Classifier",
    "C50Ensemble",
    "CubistRegressor",
    "ObliqueRandomForestClassifier",
    "ObliqueRandomForestRegressor",
    "ObliqueDecisionTreeClassifier",
    "ObliqueDecisionTreeRegressor",
    "QuantileRegressorForest",
    "pinball_loss",
    "interval_coverage",
    "interval_width",
    "EvolutionaryTreeClassifier",
    "EvolutionaryTreeRegressor",
    # Explainable Boosting Machines
    "EBMClassifier",
    "EBMRegressor",
    "show_explanation",
    # Linear models (MARS)
    "MARSRegressor",
    "MARSClassifier",
    # Rule-based models (RuleFit, FURIA)
    "RuleFitRegressor",
    "RuleFitClassifier",
    "FURIAClassifier",
    "FuzzyRule",
    "FuzzyCondition",
    # Bayesian Network Classifiers
    "TANClassifier",
    "EBMCClassifier",
    "ESKDBClassifier",
    "KDBClassifier",
    "AutoSLE",
]

if _HAS_TREEPLE_TREES:
    __all__.extend([
        "ExtraObliqueRandomForestClassifier",
        "ExtraObliqueRandomForestRegressor",
        "PatchObliqueRandomForestClassifier",
        "PatchObliqueRandomForestRegressor",
        "HonestForestClassifier",
    ])

# Add NGBoost if available
if _HAS_NGBOOST:
    __all__.extend([
        "NGBoostRegressor",
        "NGBoostClassifier",
    ])

# Neural networks require PyTorch
try:
    from endgame.models.neural import (
        EmbeddingMLPClassifier,
        EmbeddingMLPRegressor,
        MLPClassifier,
        MLPRegressor,
    )
    __all__.extend([
        "MLPClassifier",
        "MLPRegressor",
        "EmbeddingMLPClassifier",
        "EmbeddingMLPRegressor",
    ])
except ImportError:
    pass

# Optional TabNet import
try:
    from endgame.models.neural import TabNetClassifier, TabNetRegressor
    __all__.extend(["TabNetClassifier", "TabNetRegressor"])
except ImportError:
    pass

# Neural Bayesian classifiers require PyTorch
try:
    from endgame.models.bayesian import NeuralKDBClassifier
    __all__.append("NeuralKDBClassifier")
except ImportError:
    pass

# Modern tabular deep learning models (require PyTorch)
# Note: TabPFNClassifier excluded — lazy-loaded for heavyweight dependency
# Import it directly: from endgame.models.tabular import TabPFNClassifier
try:
    from endgame.models.tabular import (
        FTTransformerClassifier,
        FTTransformerRegressor,
        ModernNCAClassifier,
        NODEClassifier,
        NODERegressor,
        SAINTClassifier,
        SAINTRegressor,
    )
    __all__.extend([
        "FTTransformerClassifier",
        "FTTransformerRegressor",
        "SAINTClassifier",
        "SAINTRegressor",
        "NODEClassifier",
        "NODERegressor",
        "ModernNCAClassifier",
    ])
except ImportError:
    pass

# GANDALF requires pytorch-tabular (heavyweight optional dependency)
# Import GANDALF directly from endgame.models.tabular.gandalf if needed
# try:
#     from endgame.models.tabular import (
#         GANDALFClassifier,
#         GANDALFRegressor,
#     )
#     __all__.extend([
#         "GANDALFClassifier",
#         "GANDALFRegressor",
#     ])
# except ImportError:
#     pass

# NAM (Neural Additive Models) requires PyTorch
try:
    from endgame.models.tabular import (
        NAMClassifier,
        NAMRegressor,
    )
    __all__.extend([
        "NAMClassifier",
        "NAMRegressor",
    ])
except ImportError:
    pass

# Kernel methods (GP, SVM)
from endgame.models.kernel import (
    GPClassifier,
    GPRegressor,
    SVMClassifier,
    SVMRegressor,
)

__all__.extend([
    "GPClassifier",
    "GPRegressor",
    "SVMClassifier",
    "SVMRegressor",
])

# Simple baselines for ensemble diversity (ELM, Naive Bayes, Discriminant Analysis, KNN, Linear)
from endgame.models.baselines import (
    ELMClassifier,
    ELMRegressor,
    KNNClassifier,
    KNNRegressor,
    LDAClassifier,
    LinearClassifier,
    LinearRegressor,
    NaiveBayesClassifier,
    QDAClassifier,
    RDAClassifier,
)

__all__.extend([
    "ELMClassifier",
    "ELMRegressor",
    "NaiveBayesClassifier",
    "LDAClassifier",
    "QDAClassifier",
    "RDAClassifier",
    "KNNClassifier",
    "KNNRegressor",
    "LinearClassifier",
    "LinearRegressor",
])

# Subgroup discovery (PRIM)
from endgame.models.subgroup import (
    Box,
    PRIMClassifier,
    PRIMRegressor,
    PRIMResult,
)

__all__.extend([
    "PRIMClassifier",
    "PRIMRegressor",
    "Box",
    "PRIMResult",
])

# Ordinal regression models (require mord)
try:
    from endgame.models.ordinal import (
        LAD,
        LogisticAT,
        LogisticIT,
        LogisticSE,
        OrdinalClassifier,
        OrdinalRidge,
    )
    __all__.extend([
        "OrdinalClassifier",
        "OrdinalRidge",
        "LogisticAT",
        "LogisticIT",
        "LogisticSE",
        "LAD",
    ])
except ImportError:
    pass

# BART (Bayesian Additive Regression Trees) requires pymc and pymc-bart
try:
    from endgame.models.probabilistic import (
        BARTClassifier,
        BARTRegressor,
    )
    __all__.extend([
        "BARTClassifier",
        "BARTRegressor",
    ])
except ImportError:
    pass

# Symbolic Regression (pure-Python GP engine, no external dependencies)
try:
    from endgame.models.symbolic import (
        SymbolicClassifier,
        SymbolicRegressor,
    )
    __all__.extend([
        "SymbolicRegressor",
        "SymbolicClassifier",
    ])
except ImportError:
    pass

# Neuroevolution (NEAT, TensorNEAT)
try:
    from endgame.models.neuroevolution import (
        NEATClassifier,
        NEATRegressor,
    )
    __all__.extend([
        "NEATClassifier",
        "NEATRegressor",
    ])
except ImportError:
    pass

try:
    from endgame.models.neuroevolution import (
        TensorNEATClassifier,
        TensorNEATRegressor,
    )
    __all__.extend([
        "TensorNEATClassifier",
        "TensorNEATRegressor",
    ])
except ImportError:
    pass
