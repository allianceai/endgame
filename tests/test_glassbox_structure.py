"""Tests for the unified :meth:`get_structure` API across all glassbox models."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from endgame.core.glassbox import GlassboxMixin


# Common schema keys every payload must expose.
BASE_KEYS = {"model_type", "structure_type", "n_features", "feature_names"}
ALLOWED_TYPES = {
    "tree", "tree_ensemble", "rules", "fuzzy_rules", "boxes",
    "linear", "additive", "scorecard", "bayesian_network", "symbolic",
}


def _assert_base_schema(struct: dict, model_name: str) -> None:
    assert isinstance(struct, dict), f"{model_name} get_structure did not return dict"
    missing = BASE_KEYS - struct.keys()
    assert not missing, f"{model_name} missing base keys: {missing}"
    assert struct["model_type"] == model_name
    assert struct["structure_type"] in ALLOWED_TYPES, (
        f"{model_name} has unknown structure_type={struct['structure_type']}"
    )
    assert isinstance(struct["feature_names"], list)
    assert isinstance(struct["n_features"], int)
    assert struct["n_features"] == len(struct["feature_names"])


# ---- Classification fixtures -------------------------------------------------


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4,
        n_redundant=0, random_state=0,
    )
    return X, y


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=0)
    return X, y


# ---- Trees -------------------------------------------------------------------


def test_c50_classifier_structure(clf_data):
    from endgame.models.trees.c50 import C50Classifier

    clf = C50Classifier(random_state=0).fit(*clf_data)
    assert isinstance(clf, GlassboxMixin)
    s = clf.get_structure()
    _assert_base_schema(s, "C50Classifier")
    assert s["structure_type"] == "tree"
    assert "tree" in s


def test_adt_classifier_structure(clf_data):
    from endgame.models.trees.adtree import AlternatingDecisionTreeClassifier

    clf = AlternatingDecisionTreeClassifier(n_iterations=3, random_state=0).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "AlternatingDecisionTreeClassifier")
    assert s["structure_type"] == "tree"


def test_oblique_dt_classifier_structure(clf_data):
    from endgame.models.trees.oblique_tree import ObliqueDecisionTreeClassifier

    clf = ObliqueDecisionTreeClassifier(random_state=0, max_depth=3).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "ObliqueDecisionTreeClassifier")
    assert s["structure_type"] == "tree"
    assert "oblique_method" in s


def test_oblique_forest_classifier_structure(clf_data):
    from endgame.models.trees.oblique_forest import ObliqueRandomForestClassifier

    clf = ObliqueRandomForestClassifier(
        n_estimators=3, random_state=0, max_depth=3
    ).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "ObliqueRandomForestClassifier")
    assert s["structure_type"] == "tree_ensemble"
    assert s["n_trees"] == 3


def test_rotation_forest_classifier_structure(clf_data):
    from endgame.models.trees.rotation_forest import RotationForestClassifier

    clf = RotationForestClassifier(n_estimators=3, random_state=0).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "RotationForestClassifier")
    assert s["structure_type"] == "tree_ensemble"


def test_quantile_forest_structure(reg_data):
    from endgame.models.trees.quantile_forest import QuantileRegressorForest

    reg = QuantileRegressorForest(
        n_estimators=3, random_state=0, max_depth=3
    ).fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "QuantileRegressorForest")
    assert s["structure_type"] == "tree_ensemble"


def test_cubist_structure(reg_data):
    from endgame.models.trees.cubist import CubistRegressor

    reg = CubistRegressor(use_rust=False).fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "CubistRegressor")
    assert s["structure_type"] == "rules"
    assert "committees" in s


def test_evtree_classifier_structure(clf_data):
    from endgame.models.trees.evtree.evtree import EvolutionaryTreeClassifier

    clf = EvolutionaryTreeClassifier(
        n_generations=3, population_size=8, random_state=0
    ).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "EvolutionaryTreeClassifier")
    assert s["structure_type"] == "tree"


# ---- Rules -------------------------------------------------------------------


def test_rulefit_classifier_structure(clf_data):
    from endgame.models.rules.rulefit import RuleFitClassifier

    clf = RuleFitClassifier(n_estimators=10, random_state=0).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "RuleFitClassifier")
    assert s["structure_type"] == "rules"
    assert "intercept" in s
    assert "rules" in s


def test_rulefit_regressor_structure(reg_data):
    from endgame.models.rules.rulefit import RuleFitRegressor

    reg = RuleFitRegressor(n_estimators=10, random_state=0).fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "RuleFitRegressor")
    assert s["structure_type"] == "rules"


def test_furia_structure(clf_data):
    from endgame.models.rules.furia import FURIAClassifier

    clf = FURIAClassifier().fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "FURIAClassifier")
    assert s["structure_type"] == "fuzzy_rules"


def test_prim_regressor_structure(reg_data):
    from endgame.models.subgroup.prim import PRIMRegressor

    reg = PRIMRegressor(n_boxes=1).fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "PRIMRegressor")
    assert s["structure_type"] == "boxes"


def test_prim_classifier_structure(clf_data):
    from endgame.models.subgroup.prim import PRIMClassifier

    clf = PRIMClassifier(n_boxes=1).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "PRIMClassifier")
    assert s["structure_type"] == "boxes"


# ---- Linear / additive / scorecard / symbolic -------------------------------


def test_linear_classifier_structure(clf_data):
    from endgame.models.baselines.linear import LinearClassifier

    clf = LinearClassifier().fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "LinearClassifier")
    assert s["structure_type"] == "linear"
    assert "coefficients" in s


def test_linear_regressor_structure(reg_data):
    from endgame.models.baselines.linear import LinearRegressor

    reg = LinearRegressor().fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "LinearRegressor")
    assert s["structure_type"] == "linear"


def test_naive_bayes_structure(clf_data):
    from endgame.models.baselines.naive_bayes import NaiveBayesClassifier

    clf = NaiveBayesClassifier().fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "NaiveBayesClassifier")
    assert s["structure_type"] == "linear"
    assert "variant" in s


def test_lda_structure(clf_data):
    from endgame.models.baselines.discriminant import LDAClassifier

    clf = LDAClassifier().fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "LDAClassifier")
    assert s["structure_type"] == "linear"


def test_mars_regressor_structure(reg_data):
    from endgame.models.linear.mars import MARSRegressor

    reg = MARSRegressor(max_terms=10).fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "MARSRegressor")
    assert s["structure_type"] == "additive"
    assert "basis_functions" in s


def test_ebm_classifier_structure(clf_data):
    from endgame.models.ebm import EBMClassifier

    clf = EBMClassifier(interactions=0).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "EBMClassifier")
    assert s["structure_type"] == "additive"
    assert "terms" in s


def test_slim_structure(clf_data):
    from endgame.models.interpretable.slim import SLIMClassifier

    clf = SLIMClassifier().fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "SLIMClassifier")
    assert s["structure_type"] == "scorecard"


def test_symbolic_regressor_structure(reg_data):
    from endgame.models.symbolic.symbolic_regressor import SymbolicRegressor

    reg = SymbolicRegressor(
        niterations=2, population_size=20, populations=2,
    ).fit(*reg_data)
    s = reg.get_structure()
    _assert_base_schema(s, "SymbolicRegressor")
    assert s["structure_type"] == "symbolic"
    assert "equation" in s


# ---- Bayesian ----------------------------------------------------------------


def test_tan_structure(clf_data):
    from endgame.models.bayesian import TANClassifier

    clf = TANClassifier(smoothing=1.0).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "TANClassifier")
    assert s["structure_type"] == "bayesian_network"
    assert "nodes" in s and "edges" in s


def test_kdb_structure(clf_data):
    from endgame.models.bayesian import KDBClassifier

    clf = KDBClassifier(smoothing=1.0).fit(*clf_data)
    s = clf.get_structure()
    _assert_base_schema(s, "KDBClassifier")
    assert s["structure_type"] == "bayesian_network"


# ---- Feature name override ---------------------------------------------------


def test_feature_names_override(clf_data):
    from endgame.models.baselines.linear import LinearClassifier

    clf = LinearClassifier().fit(*clf_data)
    names = [f"feat_{i}" for i in range(5)]
    s = clf.get_structure(feature_names=names)
    assert s["feature_names"] == names
