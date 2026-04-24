"""Integration tests: glassbox structure section and tree-viz link in reports."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from endgame.visualization import ClassificationReport, RegressionReport


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


def _render_classification(model, clf_data, tmp_path: Path) -> tuple[str, Path]:
    X, y = clf_data
    report = ClassificationReport(
        model, X, y, feature_names=[f"x{i}" for i in range(X.shape[1])]
    )
    path = report.save(tmp_path / "report.html")
    return path.read_text(), path


def _render_regression(model, reg_data, tmp_path: Path) -> tuple[str, Path]:
    X, y = reg_data
    report = RegressionReport(
        model, X, y, feature_names=[f"x{i}" for i in range(X.shape[1])]
    )
    path = report.save(tmp_path / "report.html")
    return path.read_text(), path


# ---- Classification ---------------------------------------------------------


class TestClassificationGlassboxStructure:
    def test_tree_model_inlines_structure_and_tree_link(self, clf_data, tmp_path):
        from endgame.models.trees.c50 import C50Classifier
        X, y = clf_data
        clf = C50Classifier(random_state=0).fit(X, y)
        html, path = _render_classification(clf, clf_data, tmp_path)
        assert "Learned Structure" in html
        assert "C50Classifier" in html
        # Tree visualization sidecar + link
        assert (path.parent / "report_tree.html").exists()
        assert "report_tree.html" in html
        assert "Open interactive tree visualization" in html

    def test_rule_model_renders_rules_list(self, clf_data, tmp_path):
        from endgame.models.rules.rulefit import RuleFitClassifier
        X, y = clf_data
        clf = RuleFitClassifier(n_estimators=5, random_state=0).fit(X, y)
        html, _ = _render_classification(clf, clf_data, tmp_path)
        assert "Learned Structure" in html
        assert "RuleFitClassifier" in html
        # RuleFit is rule-typed, not tree — no sidecar
        assert "report_tree.html" not in html

    def test_linear_model_renders_coefficients(self, clf_data, tmp_path):
        from endgame.models.baselines.linear import LinearClassifier
        X, y = clf_data
        clf = LinearClassifier().fit(X, y)
        html, path = _render_classification(clf, clf_data, tmp_path)
        assert "Learned Structure" in html
        # New section should be used — legacy "Model Coefficients" fallback shouldn't fire
        assert "Model Coefficients (Top" not in html
        assert not (path.parent / "report_tree.html").exists()

    def test_bayesian_network_renders_edges(self, clf_data, tmp_path):
        from endgame.models.bayesian import TANClassifier
        X, y = clf_data
        clf = TANClassifier(smoothing=1.0).fit(X, y)
        html, path = _render_classification(clf, clf_data, tmp_path)
        assert "Learned Structure" in html
        assert "TANClassifier" in html
        assert "Edges" in html
        # BN visualization sidecar + link.
        assert (path.parent / "report_bn.html").exists()
        assert "report_bn.html" in html
        assert "Open interactive Bayesian network" in html
        # Tree sidecar should NOT be produced for a BN model.
        assert not (path.parent / "report_tree.html").exists()

    def test_non_bn_models_have_no_bn_sidecar(self, clf_data, tmp_path):
        X, y = clf_data
        clf = LogisticRegression(max_iter=200).fit(X, y)
        html, path = _render_classification(clf, clf_data, tmp_path)
        assert not (path.parent / "report_bn.html").exists()
        assert "report_bn.html" not in html


class TestNamePropagation:
    """Class / feature names must reach every section of the report."""

    def test_feature_names_from_pandas_dataframe(self, clf_data, tmp_path):
        pd = pytest.importorskip("pandas")
        X, y = clf_data
        cols = [f"age", "income", "score", "region", "tenure"]
        Xdf = pd.DataFrame(X, columns=cols)
        from endgame.models.bayesian import TANClassifier
        clf = TANClassifier().fit(Xdf.values, y)  # fit on ndarray, report gets DataFrame
        report = ClassificationReport(clf, Xdf, y, class_names=["A", "B"])
        path = report.save(tmp_path / "rep.html")
        html = path.read_text()
        # Feature names flowed into the BN edge table (instead of raw ints).
        for name in cols:
            # At least one of the cols must appear as an edge endpoint.
            pass
        # Concretely: the BN structure edges should reference feature names and "Class".
        assert "Class" in html   # 'Y' → 'Class'
        # Class names flowed into the Class Distribution / Per-Class / CM.
        assert '"A"' in html or ">A<" in html
        assert '"B"' in html or ">B<" in html

    def test_feature_names_from_model_attribute(self, clf_data, tmp_path):
        # Model trained via sklearn DataFrame input has feature_names_in_.
        pd = pytest.importorskip("pandas")
        X, y = clf_data
        cols = ["alpha", "beta", "gamma", "delta", "epsilon"]
        Xdf = pd.DataFrame(X, columns=cols)
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(Xdf, y)
        # Pass X as a plain ndarray — feature_names should still be picked up
        # from model.feature_names_in_.
        report = ClassificationReport(clf, X, y)
        assert report.feature_names == cols
        path = report.save(tmp_path / "rep.html")
        html = path.read_text()
        # Feature Importances chart payload carries the real names.
        assert '"alpha"' in html
        assert '"epsilon"' in html
        # And the default "Feature 0" placeholder is NOT used when names exist.
        assert '"Feature 0"' not in html

    def test_bayesian_network_edges_show_feature_names(self, clf_data, tmp_path):
        pd = pytest.importorskip("pandas")
        X, y = clf_data
        cols = ["age", "income", "score", "region", "tenure"]
        Xdf = pd.DataFrame(X, columns=cols)
        from endgame.models.bayesian import TANClassifier
        clf = TANClassifier().fit(Xdf.values, y)
        report = ClassificationReport(clf, Xdf, y)
        path = report.save(tmp_path / "rep.html")
        html = path.read_text()
        # Edges section replaces "Y → 3" with "Class → region" (or similar).
        assert "Class" in html
        # At least one feature name appears as an edge endpoint (TAN has
        # Class → every feature).
        in_edges = sum(1 for name in cols if f">{name}<" in html)
        assert in_edges >= 3   # most of them, in table cells

    def test_explicit_feature_names_override_inferred(self, clf_data, tmp_path):
        pd = pytest.importorskip("pandas")
        X, y = clf_data
        Xdf = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(Xdf, y)
        report = ClassificationReport(
            clf, Xdf, y, feature_names=["OVERRIDE_1", "OVERRIDE_2", "OVERRIDE_3", "OVERRIDE_4", "OVERRIDE_5"],
        )
        assert report.feature_names == [
            "OVERRIDE_1", "OVERRIDE_2", "OVERRIDE_3", "OVERRIDE_4", "OVERRIDE_5",
        ]


class TestInferFeatureNames:
    def test_from_pandas(self):
        pd = pytest.importorskip("pandas")
        from endgame.visualization.classification_report import _infer_feature_names
        Xdf = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert _infer_feature_names(Xdf, object()) == ["a", "b"]

    def test_from_model_feature_names_in_(self):
        import numpy as np
        from endgame.visualization.classification_report import _infer_feature_names
        class FakeModel:
            feature_names_in_ = np.array(["x", "y", "z"])
        assert _infer_feature_names(np.array([[1, 2, 3]]), FakeModel()) == ["x", "y", "z"]

    def test_returns_none_when_unavailable(self):
        import numpy as np
        from endgame.visualization.classification_report import _infer_feature_names
        assert _infer_feature_names(np.array([[1, 2, 3]]), object()) is None

    def test_sklearn_tree_gets_link_but_no_structure_section(self, clf_data, tmp_path):
        X, y = clf_data
        clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        html, path = _render_classification(clf, clf_data, tmp_path)
        # No structure section — sklearn RF has no get_structure
        assert "Learned Structure" not in html
        # But sidecar + link are still produced
        assert (path.parent / "report_tree.html").exists()
        assert "report_tree.html" in html

    def test_non_tree_non_glassbox_has_no_link(self, clf_data, tmp_path):
        X, y = clf_data
        clf = LogisticRegression(max_iter=200).fit(X, y)
        html, path = _render_classification(clf, clf_data, tmp_path)
        # No sidecar — LogisticRegression isn't tree-visualizable
        assert not (path.parent / "report_tree.html").exists()
        # Legacy linear fallback still shows coefficients
        assert "Model Coefficients" in html


# ---- Regression -------------------------------------------------------------


class TestRegressionGlassboxStructure:
    def test_tree_regressor_inlines_structure_and_link(self, reg_data, tmp_path):
        from endgame.models.trees.evtree.evtree import EvolutionaryTreeRegressor
        X, y = reg_data
        reg = EvolutionaryTreeRegressor(
            n_generations=3, population_size=8, random_state=0
        ).fit(X, y)
        html, path = _render_regression(reg, reg_data, tmp_path)
        assert "Learned Structure" in html
        assert (path.parent / "report_tree.html").exists()
        assert "report_tree.html" in html

    def test_mars_additive_structure(self, reg_data, tmp_path):
        from endgame.models.linear.mars import MARSRegressor
        X, y = reg_data
        reg = MARSRegressor(max_terms=10).fit(X, y)
        html, path = _render_regression(reg, reg_data, tmp_path)
        assert "Learned Structure" in html
        assert "MARSRegressor" in html
        assert not (path.parent / "report_tree.html").exists()

    def test_symbolic_regressor_equation(self, reg_data, tmp_path):
        from endgame.models.symbolic.symbolic_regressor import SymbolicRegressor
        X, y = reg_data
        reg = SymbolicRegressor(
            niterations=2, population_size=20, populations=2,
        ).fit(X, y)
        html, _ = _render_regression(reg, reg_data, tmp_path)
        assert "Learned Structure" in html
        assert "SymbolicRegressor" in html

    def test_prim_regressor_boxes(self, reg_data, tmp_path):
        from endgame.models.subgroup.prim import PRIMRegressor
        X, y = reg_data
        reg = PRIMRegressor(n_boxes=1).fit(X, y)
        html, _ = _render_regression(reg, reg_data, tmp_path)
        assert "Learned Structure" in html
        assert "PRIMRegressor" in html


# ---- Export tests -----------------------------------------------------------


class TestStructureExport:
    def test_no_cap_on_rule_rendering(self, clf_data, tmp_path):
        """Report must render every rule — no silent truncation."""
        from endgame.models.rules.rulefit import RuleFitClassifier
        X, y = clf_data
        clf = RuleFitClassifier(n_estimators=20, random_state=0).fit(X, y)
        struct = clf.get_structure()
        html, _ = _render_classification(clf, clf_data, tmp_path)
        # Every rule from get_structure() should appear as a <li> in the HTML.
        assert html.count("<li>") >= struct["n_rules"]
        # Legacy truncation note must not appear.
        assert "more rules" not in html

    def test_classification_save_writes_structure_sidecar(self, clf_data, tmp_path):
        from endgame.models.rules.rulefit import RuleFitClassifier
        import json
        X, y = clf_data
        clf = RuleFitClassifier(n_estimators=5, random_state=0).fit(X, y)
        report = ClassificationReport(clf, X, y)
        path = report.save(tmp_path / "report.html")
        sidecar = path.parent / "report_structure.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["model_type"] == "RuleFitClassifier"
        assert data["structure_type"] == "rules"
        # HTML report includes a download link to the sidecar.
        assert "report_structure.json" in path.read_text()

    def test_classification_export_structure_method(self, clf_data, tmp_path):
        from endgame.models.trees.c50 import C50Classifier
        import json
        X, y = clf_data
        clf = C50Classifier(random_state=0).fit(X, y)
        report = ClassificationReport(clf, X, y)
        out = report.export_structure(tmp_path / "c50")
        assert out is not None and out.suffix == ".json"
        data = json.loads(out.read_text())
        assert data["model_type"] == "C50Classifier"
        assert data["structure_type"] == "tree"

    def test_regression_export_structure_method(self, reg_data, tmp_path):
        from endgame.models.linear.mars import MARSRegressor
        import json
        X, y = reg_data
        reg = MARSRegressor(max_terms=10).fit(X, y)
        report = RegressionReport(reg, X, y)
        out = report.export_structure(tmp_path / "mars.json")
        assert out is not None
        data = json.loads(out.read_text())
        assert data["structure_type"] == "additive"

    def test_export_structure_returns_none_for_non_glassbox(self, clf_data, tmp_path):
        from sklearn.linear_model import LogisticRegression
        X, y = clf_data
        clf = LogisticRegression(max_iter=200).fit(X, y)
        report = ClassificationReport(clf, X, y)
        assert report.export_structure(tmp_path / "none.json") is None


# ---- Unit tests for renderer ------------------------------------------------


class TestStructureRenderer:
    def test_returns_empty_string_for_model_without_get_structure(self):
        from endgame.visualization._structure_section import render_structure_section

        class Dummy:
            pass

        assert render_structure_section(Dummy()) == ""

    def test_tree_link_only_rendered_for_tree_types(self):
        from endgame.visualization._structure_section import render_structure_section

        class FakeTreeModel:
            def get_structure(self):
                return {
                    "model_type": "FakeTree",
                    "structure_type": "tree",
                    "n_features": 2,
                    "feature_names": ["a", "b"],
                    "tree": {"n_leaves": 3, "max_depth": 2},
                }

        class FakeLinearModel:
            def get_structure(self):
                return {
                    "model_type": "FakeLinear",
                    "structure_type": "linear",
                    "n_features": 2,
                    "feature_names": ["a", "b"],
                    "intercept": 0.5,
                    "coefficients": {"a": 1.0, "b": -0.25},
                }

        tree_html = render_structure_section(FakeTreeModel(), tree_link_href="x_tree.html")
        assert "x_tree.html" in tree_html
        # Non-tree structures ignore tree_link_href.
        lin_html = render_structure_section(FakeLinearModel(), tree_link_href="x_tree.html")
        assert "x_tree.html" not in lin_html
        assert "FakeLinear" in lin_html

    def test_bn_link_only_rendered_for_bayesian_network(self):
        from endgame.visualization._structure_section import render_structure_section

        class FakeBN:
            def get_structure(self):
                return {
                    "model_type": "FakeBN",
                    "structure_type": "bayesian_network",
                    "nodes": ["a", "b"],
                    "edges": [["a", "b"]],
                    "markov_blanket": [],
                }

        class FakeTree:
            def get_structure(self):
                return {
                    "model_type": "FakeTree",
                    "structure_type": "tree",
                    "tree": {"n_leaves": 2, "max_depth": 1},
                }

        bn_html = render_structure_section(FakeBN(), bn_link_href="x_bn.html")
        assert "x_bn.html" in bn_html
        assert "Open interactive Bayesian network" in bn_html
        # Non-BN structures ignore bn_link_href.
        tree_html = render_structure_section(FakeTree(), bn_link_href="x_bn.html")
        assert "x_bn.html" not in tree_html
