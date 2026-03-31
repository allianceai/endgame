"""Fuzzy learning algorithms for classification, regression, clustering, and more.

Provides 35+ fuzzy algorithms spanning classic inference systems, neuro-fuzzy
architectures, evolving/online systems, Type-2 fuzzy systems, fuzzy ensembles,
fuzzy-rough methods, and modern state-of-the-art approaches.

All estimators follow the scikit-learn API (fit/predict/predict_proba/transform).

Example
-------
>>> import endgame as eg
>>> from endgame.fuzzy import TSKRegressor, FuzzyKNNClassifier
>>> model = TSKRegressor(n_rules=10, order=1)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)

Submodules
----------
core : Membership functions, t-norms, defuzzification, base classes
inference : Mamdani, TSK, ANFIS, Wang-Mendel
classifiers : Fuzzy KNN, Fuzzy Decision Tree, NEFCLASS
neurofuzzy : FALCON, SOFNN, DENFIS, FNN-TSK
evolving : eTS, eTS+, PANFIS, AutoCloud, FLEXFIS
type2 : IT2-FLS, IT2-TSK, IT2-ANFIS, General Type-2
ensemble : Fuzzy Random Forest, Fuzzy Boosted Trees, Fuzzy Bagging, Stacked Fuzzy
rough : FRNN, FRFS
modern : HTSK, MBGD-RDA, FCM-RDpA, Fuzzy Attention, Differentiable Fuzzy, TSK+, SEIT2FNN
extraction : Fuzzy rule extraction from black-box models
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy loading for fuzzy submodules and classes."""
    import importlib

    # Submodule access
    _submodules = {
        "core", "inference", "classifiers", "neurofuzzy", "evolving",
        "type2", "ensemble", "rough", "modern", "extraction",
    }
    if name in _submodules:
        module = importlib.import_module(f"endgame.fuzzy.{name}")
        globals()[name] = module
        return module

    # Direct class imports (convenience)
    _class_map = {
        # Core
        "GaussianMF": "endgame.fuzzy.core.membership",
        "TriangularMF": "endgame.fuzzy.core.membership",
        "TrapezoidalMF": "endgame.fuzzy.core.membership",
        "GeneralizedBellMF": "endgame.fuzzy.core.membership",
        "SigmoidalMF": "endgame.fuzzy.core.membership",
        "IntervalType2GaussianMF": "endgame.fuzzy.core.membership",
        "IntervalType2TriangularMF": "endgame.fuzzy.core.membership",
        "t_norm": "endgame.fuzzy.core.operators",
        "t_conorm": "endgame.fuzzy.core.operators",
        "defuzzify": "endgame.fuzzy.core.defuzzification",
        # Inference
        "MamdaniFIS": "endgame.fuzzy.inference.mamdani",
        "MamdaniClassifier": "endgame.fuzzy.inference.mamdani",
        "TSKRegressor": "endgame.fuzzy.inference.tsk",
        "TSKClassifier": "endgame.fuzzy.inference.tsk",
        "ANFISRegressor": "endgame.fuzzy.inference.anfis",
        "ANFISClassifier": "endgame.fuzzy.inference.anfis",
        "WangMendelRegressor": "endgame.fuzzy.inference.wang_mendel",
        # Classifiers
        "FuzzyKNNClassifier": "endgame.fuzzy.classifiers.fuzzy_knn",
        "FuzzyDecisionTreeClassifier": "endgame.fuzzy.classifiers.fuzzy_decision_tree",
        "FuzzyDecisionTreeRegressor": "endgame.fuzzy.classifiers.fuzzy_decision_tree",
        "NEFCLASSClassifier": "endgame.fuzzy.classifiers.nefclass",
        # Neuro-fuzzy
        "FALCONClassifier": "endgame.fuzzy.neurofuzzy.falcon",
        "FALCONRegressor": "endgame.fuzzy.neurofuzzy.falcon",
        "SOFNNRegressor": "endgame.fuzzy.neurofuzzy.sofnn",
        "DENFISRegressor": "endgame.fuzzy.neurofuzzy.denfis",
        "DENFISClassifier": "endgame.fuzzy.neurofuzzy.denfis",
        "FNNTSKRegressor": "endgame.fuzzy.neurofuzzy.fnn_tsk",
        "FNNTSKClassifier": "endgame.fuzzy.neurofuzzy.fnn_tsk",
        # Evolving
        "EvolvingTSK": "endgame.fuzzy.evolving.ets",
        "EvolvingTSKPlus": "endgame.fuzzy.evolving.ets",
        "PANFISRegressor": "endgame.fuzzy.evolving.panfis",
        "PANFISClassifier": "endgame.fuzzy.evolving.panfis",
        "AutoCloudClassifier": "endgame.fuzzy.evolving.autocloud",
        "FLEXFISRegressor": "endgame.fuzzy.evolving.flexfis",
        # Type-2
        "IT2FLSRegressor": "endgame.fuzzy.type2.it2_fls",
        "IT2FLSClassifier": "endgame.fuzzy.type2.it2_fls",
        "IT2TSKRegressor": "endgame.fuzzy.type2.it2_tsk",
        "IT2ANFISRegressor": "endgame.fuzzy.type2.it2_anfis",
        "GeneralType2FLS": "endgame.fuzzy.type2.general_t2",
        # Ensemble
        "FuzzyRandomForestClassifier": "endgame.fuzzy.ensemble.fuzzy_random_forest",
        "FuzzyRandomForestRegressor": "endgame.fuzzy.ensemble.fuzzy_random_forest",
        "FuzzyBoostedTreesClassifier": "endgame.fuzzy.ensemble.fuzzy_boosted_trees",
        "FuzzyBoostedTreesRegressor": "endgame.fuzzy.ensemble.fuzzy_boosted_trees",
        "FuzzyBaggingClassifier": "endgame.fuzzy.ensemble.fuzzy_bagging",
        "FuzzyBaggingRegressor": "endgame.fuzzy.ensemble.fuzzy_bagging",
        "StackedFuzzySystem": "endgame.fuzzy.ensemble.stacked_fuzzy",
        # Rough
        "FuzzyRoughNNClassifier": "endgame.fuzzy.rough.frnn",
        "FuzzyRoughFeatureSelector": "endgame.fuzzy.rough.frfs",
        # Modern
        "HTSKClassifier": "endgame.fuzzy.modern.htsk",
        "HTSKRegressor": "endgame.fuzzy.modern.htsk",
        "MBGDRDARegressor": "endgame.fuzzy.modern.mbgd_rda",
        "MBGDRDATrainer": "endgame.fuzzy.modern.mbgd_rda",
        "FCMRDpAClassifier": "endgame.fuzzy.modern.fcm_rdpa",
        "FCMRDpARegressor": "endgame.fuzzy.modern.fcm_rdpa",
        "FuzzyAttentionClassifier": "endgame.fuzzy.modern.fuzzy_attention",
        "DifferentiableFuzzySystem": "endgame.fuzzy.modern.differentiable",
        "TSKPlusClassifier": "endgame.fuzzy.modern.tsk_plus",
        "TSKPlusRegressor": "endgame.fuzzy.modern.tsk_plus",
        "SEIT2FNNClassifier": "endgame.fuzzy.modern.seit2fnn",
        # Extraction
        "FuzzyRuleExtractor": "endgame.fuzzy.extraction.rule_extraction",
    }

    if name in _class_map:
        module = importlib.import_module(_class_map[name])
        obj = getattr(module, name)
        globals()[name] = obj
        return obj

    # Re-exports from existing modules
    _reexports = {
        "FURIAClassifier": ("endgame.models.rules.furia", "FURIAClassifier"),
        "FuzzyRule": ("endgame.models.rules.furia", "FuzzyRule"),
        "FuzzyCondition": ("endgame.models.rules.furia", "FuzzyCondition"),
        "FuzzyCMeansClusterer": (
            "endgame.clustering.distribution", "FuzzyCMeansClusterer"
        ),
    }

    if name in _reexports:
        mod_path, attr_name = _reexports[name]
        module = importlib.import_module(mod_path)
        obj = getattr(module, attr_name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module 'endgame.fuzzy' has no attribute {name!r}")


__all__ = [
    # Submodules
    "core",
    "inference",
    "classifiers",
    "neurofuzzy",
    "evolving",
    "type2",
    "ensemble",
    "rough",
    "modern",
    "extraction",
    # Core utilities
    "GaussianMF",
    "TriangularMF",
    "TrapezoidalMF",
    "GeneralizedBellMF",
    "SigmoidalMF",
    "IntervalType2GaussianMF",
    "IntervalType2TriangularMF",
    "t_norm",
    "t_conorm",
    "defuzzify",
    # Inference
    "MamdaniFIS",
    "MamdaniClassifier",
    "TSKRegressor",
    "TSKClassifier",
    "ANFISRegressor",
    "ANFISClassifier",
    "WangMendelRegressor",
    # Classifiers
    "FuzzyKNNClassifier",
    "FuzzyDecisionTreeClassifier",
    "FuzzyDecisionTreeRegressor",
    "NEFCLASSClassifier",
    # Neuro-fuzzy
    "FALCONClassifier",
    "FALCONRegressor",
    "SOFNNRegressor",
    "DENFISRegressor",
    "DENFISClassifier",
    "FNNTSKRegressor",
    "FNNTSKClassifier",
    # Evolving
    "EvolvingTSK",
    "EvolvingTSKPlus",
    "PANFISRegressor",
    "PANFISClassifier",
    "AutoCloudClassifier",
    "FLEXFISRegressor",
    # Type-2
    "IT2FLSRegressor",
    "IT2FLSClassifier",
    "IT2TSKRegressor",
    "IT2ANFISRegressor",
    "GeneralType2FLS",
    # Ensemble
    "FuzzyRandomForestClassifier",
    "FuzzyRandomForestRegressor",
    "FuzzyBoostedTreesClassifier",
    "FuzzyBoostedTreesRegressor",
    "FuzzyBaggingClassifier",
    "FuzzyBaggingRegressor",
    "StackedFuzzySystem",
    # Rough
    "FuzzyRoughNNClassifier",
    "FuzzyRoughFeatureSelector",
    # Modern
    "HTSKClassifier",
    "HTSKRegressor",
    "MBGDRDARegressor",
    "MBGDRDATrainer",
    "FCMRDpAClassifier",
    "FCMRDpARegressor",
    "FuzzyAttentionClassifier",
    "DifferentiableFuzzySystem",
    "TSKPlusClassifier",
    "TSKPlusRegressor",
    "SEIT2FNNClassifier",
    # Extraction
    "FuzzyRuleExtractor",
    # Re-exports
    "FURIAClassifier",
    "FuzzyRule",
    "FuzzyCondition",
    "FuzzyCMeansClusterer",
]
