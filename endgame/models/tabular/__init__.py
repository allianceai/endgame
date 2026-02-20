"""Modern deep learning models for tabular data.

This module provides state-of-the-art neural network architectures
specifically designed for tabular data:

- TabPFN: In-context learning transformer (no training required)
- TabPFN v2: Foundation model for tabular data (classification + regression)
- TabPFN 2.5: RealTabPFN-2.5, 50K samples, 2K features (classification + regression)
- TabTransformer: Contextual embeddings for categorical features
- FT-Transformer: Feature Tokenizer Transformer (all features)
- SAINT: Self-Attention and Intersample Attention Transformer
- NODE: Neural Oblivious Decision Ensembles
- ModernNCA: Modern Neighborhood Component Analysis
- GANDALF: Gated Adaptive Network for Deep Automated Learning of Features
- NAM: Neural Additive Models (interpretable)
- GRANDE: Gradient-based Neural Decision Ensemble
- TabR: Retrieval-Augmented Tabular Deep Learning
- TabM: Parameter-Efficient MLP Ensembling (BatchEnsemble)
- TabDPT: Tabular Discriminative Pre-trained Transformer (in-context learning)
- RealMLP: Meta-Tuned MLP with Robust Preprocessing (Holzmuller et al., NeurIPS 2024)
- xRFM: Tree-Structured Recursive Feature Machines (Beaglehole & Holzmuller, 2025)

Note: Most models require PyTorch. Install with: pip install torch
"""

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# TabPFN: lazy import for heavyweight optional dependency
# from endgame.models.tabular.tabpfn import TabPFNClassifier

# PyTorch-based models - only import if torch is available
if HAS_TORCH:
    from endgame.models.tabular.ft_transformer import (
        FTTransformerClassifier,
        FTTransformerRegressor,
    )
    from endgame.models.tabular.grande import (
        GRANDEClassifier,
        GRANDERegressor,
    )
    from endgame.models.tabular.modern_nca import ModernNCAClassifier

    # GANDALF: lazy import for heavyweight optional dependency (pytorch_tabular)
    # from endgame.models.tabular.gandalf import (
    #     GANDALFClassifier,
    #     GANDALFRegressor,
    # )
    from endgame.models.tabular.nam import (
        NAMClassifier,
        NAMRegressor,
    )
    from endgame.models.tabular.node import (
        NODEClassifier,
        NODERegressor,
    )
    from endgame.models.tabular.realmlp import (
        RealMLPClassifier,
        RealMLPRegressor,
    )
    from endgame.models.tabular.saint import (
        SAINTClassifier,
        SAINTRegressor,
    )
    from endgame.models.tabular.tabm import (
        TabMClassifier,
        TabMRegressor,
    )
    from endgame.models.tabular.tabr import (
        TabRClassifier,
        TabRRegressor,
    )
    from endgame.models.tabular.tabular_resnet import (
        TabularResNetClassifier,
        TabularResNetRegressor,
    )

    # Lazy import for GANDALF, TabPFN, and TabTransformer (heavyweight optional dependencies)
    def __getattr__(name):
        if name in ("GANDALFClassifier", "GANDALFRegressor"):
            from endgame.models.tabular.gandalf import GANDALFClassifier, GANDALFRegressor
            globals()["GANDALFClassifier"] = GANDALFClassifier
            globals()["GANDALFRegressor"] = GANDALFRegressor
            return globals()[name]
        if name == "TabPFNClassifier":
            from endgame.models.tabular.tabpfn import TabPFNClassifier
            globals()["TabPFNClassifier"] = TabPFNClassifier
            return TabPFNClassifier
        if name in ("TabPFNv2Classifier", "TabPFNv2Regressor"):
            from endgame.models.tabular.tabpfn import TabPFNv2Classifier, TabPFNv2Regressor
            globals()["TabPFNv2Classifier"] = TabPFNv2Classifier
            globals()["TabPFNv2Regressor"] = TabPFNv2Regressor
            return globals()[name]
        if name in ("TabPFN25Classifier", "TabPFN25Regressor"):
            from endgame.models.tabular.tabpfn import TabPFN25Classifier, TabPFN25Regressor
            globals()["TabPFN25Classifier"] = TabPFN25Classifier
            globals()["TabPFN25Regressor"] = TabPFN25Regressor
            return globals()[name]
        if name in ("TabTransformerClassifier", "TabTransformerRegressor"):
            from endgame.models.tabular.tab_transformer import (
                TabTransformerClassifier,
                TabTransformerRegressor,
            )
            globals()["TabTransformerClassifier"] = TabTransformerClassifier
            globals()["TabTransformerRegressor"] = TabTransformerRegressor
            return globals()[name]
        if name in ("TabDPTClassifier", "TabDPTRegressor"):
            from endgame.models.tabular.tabdpt import TabDPTClassifier, TabDPTRegressor
            globals()["TabDPTClassifier"] = TabDPTClassifier
            globals()["TabDPTRegressor"] = TabDPTRegressor
            return globals()[name]
        if name in ("xRFMClassifier", "xRFMRegressor"):
            from endgame.models.tabular.xrfm import xRFMClassifier, xRFMRegressor
            globals()["xRFMClassifier"] = xRFMClassifier
            globals()["xRFMRegressor"] = xRFMRegressor
            return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    __all__ = [
        "TabPFNClassifier",
        "TabPFNv2Classifier",
        "TabPFNv2Regressor",
        "TabPFN25Classifier",
        "TabPFN25Regressor",
        "TabDPTClassifier",
        "TabDPTRegressor",
        "xRFMClassifier",
        "xRFMRegressor",
        "TabTransformerClassifier",
        "TabTransformerRegressor",
        "FTTransformerClassifier",
        "FTTransformerRegressor",
        "SAINTClassifier",
        "SAINTRegressor",
        "NODEClassifier",
        "NODERegressor",
        "ModernNCAClassifier",
        "GANDALFClassifier",
        "GANDALFRegressor",
        "NAMClassifier",
        "NAMRegressor",
        "TabularResNetClassifier",
        "TabularResNetRegressor",
        "GRANDEClassifier",
        "GRANDERegressor",
        "TabRClassifier",
        "TabRRegressor",
        "TabMClassifier",
        "TabMRegressor",
        "RealMLPClassifier",
        "RealMLPRegressor",
    ]
else:
    # Provide placeholder classes that raise helpful errors
    def _make_torch_required_class(name):
        class TorchRequiredModel:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"{name} requires PyTorch. Install with: pip install torch"
                )
        TorchRequiredModel.__name__ = name
        TorchRequiredModel.__qualname__ = name
        return TorchRequiredModel

    FTTransformerClassifier = _make_torch_required_class("FTTransformerClassifier")
    FTTransformerRegressor = _make_torch_required_class("FTTransformerRegressor")
    SAINTClassifier = _make_torch_required_class("SAINTClassifier")
    SAINTRegressor = _make_torch_required_class("SAINTRegressor")
    NODEClassifier = _make_torch_required_class("NODEClassifier")
    NODERegressor = _make_torch_required_class("NODERegressor")
    ModernNCAClassifier = _make_torch_required_class("ModernNCAClassifier")
    GANDALFClassifier = _make_torch_required_class("GANDALFClassifier")
    GANDALFRegressor = _make_torch_required_class("GANDALFRegressor")
    NAMClassifier = _make_torch_required_class("NAMClassifier")
    NAMRegressor = _make_torch_required_class("NAMRegressor")
    TabularResNetClassifier = _make_torch_required_class("TabularResNetClassifier")
    TabularResNetRegressor = _make_torch_required_class("TabularResNetRegressor")
    TabTransformerClassifier = _make_torch_required_class("TabTransformerClassifier")
    TabTransformerRegressor = _make_torch_required_class("TabTransformerRegressor")
    GRANDEClassifier = _make_torch_required_class("GRANDEClassifier")
    GRANDERegressor = _make_torch_required_class("GRANDERegressor")
    TabRClassifier = _make_torch_required_class("TabRClassifier")
    TabRRegressor = _make_torch_required_class("TabRRegressor")
    TabMClassifier = _make_torch_required_class("TabMClassifier")
    TabMRegressor = _make_torch_required_class("TabMRegressor")
    RealMLPClassifier = _make_torch_required_class("RealMLPClassifier")
    RealMLPRegressor = _make_torch_required_class("RealMLPRegressor")

    # TabPFN v2/v2.5 and TabDPT wrappers have their own graceful fallback and
    # don't strictly require torch at import time -- lazy-load them.
    def __getattr__(name):
        if name in (
            "TabPFNClassifier", "TabPFNv2Classifier", "TabPFNv2Regressor",
            "TabPFN25Classifier", "TabPFN25Regressor",
        ):
            from endgame.models.tabular.tabpfn import (
                TabPFN25Classifier,
                TabPFN25Regressor,
                TabPFNClassifier,
                TabPFNv2Classifier,
                TabPFNv2Regressor,
            )
            globals()["TabPFNClassifier"] = TabPFNClassifier
            globals()["TabPFNv2Classifier"] = TabPFNv2Classifier
            globals()["TabPFNv2Regressor"] = TabPFNv2Regressor
            globals()["TabPFN25Classifier"] = TabPFN25Classifier
            globals()["TabPFN25Regressor"] = TabPFN25Regressor
            return globals()[name]
        if name in ("TabDPTClassifier", "TabDPTRegressor"):
            from endgame.models.tabular.tabdpt import TabDPTClassifier, TabDPTRegressor
            globals()["TabDPTClassifier"] = TabDPTClassifier
            globals()["TabDPTRegressor"] = TabDPTRegressor
            return globals()[name]
        if name in ("xRFMClassifier", "xRFMRegressor"):
            from endgame.models.tabular.xrfm import xRFMClassifier, xRFMRegressor
            globals()["xRFMClassifier"] = xRFMClassifier
            globals()["xRFMRegressor"] = xRFMRegressor
            return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    __all__ = [
        "TabPFNClassifier",
        "TabPFNv2Classifier",
        "TabPFNv2Regressor",
        "TabPFN25Classifier",
        "TabPFN25Regressor",
        "TabDPTClassifier",
        "TabDPTRegressor",
        "xRFMClassifier",
        "xRFMRegressor",
        "TabTransformerClassifier",
        "TabTransformerRegressor",
        "FTTransformerClassifier",
        "FTTransformerRegressor",
        "SAINTClassifier",
        "SAINTRegressor",
        "NODEClassifier",
        "NODERegressor",
        "ModernNCAClassifier",
        "GANDALFClassifier",
        "GANDALFRegressor",
        "NAMClassifier",
        "NAMRegressor",
        "TabularResNetClassifier",
        "TabularResNetRegressor",
        "GRANDEClassifier",
        "GRANDERegressor",
        "TabRClassifier",
        "TabRRegressor",
        "TabMClassifier",
        "TabMRegressor",
        "RealMLPClassifier",
        "RealMLPRegressor",
    ]
