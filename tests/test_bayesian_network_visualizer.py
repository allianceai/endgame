"""Tests for BayesianNetworkVisualizer.

Covers three construction paths:
  1. From a fitted endgame Bayesian classifier (TAN) — the primary path.
  2. From manual (nodes, edges, cpts) — for ad-hoc BNs.
  3. From raw ``edges_`` / ``dag_`` attributes — the fallback path.

Also verifies the produced HTML is self-contained (no external resources).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from endgame.models.bayesian import TANClassifier
from endgame.visualization import BayesianNetworkVisualizer
from endgame.visualization.bayesian_network_visualizer import (
    VizBNNode,
    _markov_blanket,
    _topo_layers,
)


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------


@pytest.fixture
def discrete_xy():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 3, size=(300, 5))
    y = (X[:, 0] + X[:, 1]) % 2
    return X, y


@pytest.fixture
def fitted_tan(discrete_xy):
    X, y = discrete_xy
    return TANClassifier().fit(X, y)


# ---------------------------------------------------------------------------
# From a fitted TAN classifier
# ---------------------------------------------------------------------------


class TestFromFittedModel:
    def test_builds_from_fitted_tan(self, fitted_tan):
        viz = BayesianNetworkVisualizer(
            fitted_tan,
            feature_names=["age", "income", "score", "region", "tenure"],
            class_names=["no", "yes"],
            title="TAN",
        )
        p = json.loads(viz.to_json())

        # 1 class node + 5 feature nodes
        assert len(p["nodes"]) == 6
        assert p["classIndex"] >= 0
        assert p["nodes"][p["classIndex"]]["role"] == "class"

        # TAN: class → every feature, plus a tree among features.
        # That's 5 class→feature + 4 tree edges = 9.
        assert len(p["edges"]) == 9

        # Every feature lists the class as a parent.
        feature_nodes = [n for n in p["nodes"] if n["role"] == "feature"]
        assert len(feature_nodes) == 5
        for f in feature_nodes:
            assert "class" in f["parents"]

        # Class node has a marginal (from class_prior_).
        cls = p["nodes"][p["classIndex"]]
        assert len(cls["marginal"]) == 2
        assert abs(sum(cls["marginal"]) - 1.0) < 1e-6
        assert cls["states"] == ["no", "yes"]

        # Class Markov blanket = all features in TAN.
        assert len(p["classMarkovBlanket"]) == 5

        # Layers form a topological ordering.
        assert len(p["layers"]) >= 2

    def test_class_names_are_used(self, fitted_tan):
        viz = BayesianNetworkVisualizer(
            fitted_tan, class_names=["neg", "pos"],
        )
        p = json.loads(viz.to_json())
        cls = p["nodes"][p["classIndex"]]
        assert cls["states"] == ["neg", "pos"]

    def test_feature_names_are_used(self, fitted_tan):
        viz = BayesianNetworkVisualizer(
            fitted_tan,
            feature_names=["alpha", "beta", "gamma", "delta", "epsilon"],
        )
        p = json.loads(viz.to_json())
        labels = {n["label"] for n in p["nodes"] if n["role"] == "feature"}
        assert labels == {"alpha", "beta", "gamma", "delta", "epsilon"}

    def test_cpts_are_valid_probability_tables(self, fitted_tan):
        viz = BayesianNetworkVisualizer(fitted_tan)
        p = json.loads(viz.to_json())
        for n in p["nodes"]:
            if n["role"] != "feature":
                continue
            arr = np.asarray(n["cpt"])
            assert arr.ndim >= 1
            # Summed over the node-state axis (axis 0), each parent config
            # should be a valid probability distribution.
            sums = arr.sum(axis=0)
            assert np.allclose(sums, 1.0, atol=1e-4)

    def test_save_produces_self_contained_html(self, fitted_tan, tmp_path: Path):
        viz = BayesianNetworkVisualizer(
            fitted_tan, title="Self-contained test",
        )
        out = viz.save(tmp_path / "bn.html")

        assert out.exists()
        html = out.read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "Self-contained test" in html
        assert '"nodes"' in html
        # No external script/stylesheet tags:
        assert not re.search(r"<script[^>]*src=", html)
        assert not re.search(r"<link[^>]*stylesheet", html)

    def test_repr_html_embeds_payload(self, fitted_tan):
        viz = BayesianNetworkVisualizer(fitted_tan)
        html = viz._repr_html_()
        assert '"nodes"' in html
        assert '"edges"' in html
        # Embedded mode uses fixed 600px height.
        assert "600px" in html


# ---------------------------------------------------------------------------
# Manual construction
# ---------------------------------------------------------------------------


class TestManualConstruction:
    def test_wet_grass_example(self):
        viz = BayesianNetworkVisualizer(
            nodes=[
                {"id": "rain", "label": "Rain", "cardinality": 2, "states": ["no", "yes"]},
                {"id": "sprinkler", "label": "Sprinkler", "cardinality": 2},
                {"id": "wet", "label": "Wet grass", "cardinality": 2},
            ],
            edges=[("rain", "sprinkler"), ("rain", "wet"), ("sprinkler", "wet")],
            cpts={
                "rain": np.array([0.8, 0.2]),
                "sprinkler": np.array([[0.6, 0.99], [0.4, 0.01]]),
                "wet": np.array([
                    [[0.999, 0.2], [0.1, 0.01]],
                    [[0.001, 0.8], [0.9, 0.99]],
                ]),
            },
            title="Wet grass",
        )
        p = json.loads(viz.to_json())
        assert {n["id"] for n in p["nodes"]} == {"rain", "sprinkler", "wet"}
        rain = next(n for n in p["nodes"] if n["id"] == "rain")
        assert rain["parents"] == []
        assert rain["marginal"] == [0.8, 0.2]

        wet = next(n for n in p["nodes"] if n["id"] == "wet")
        assert set(wet["parents"]) == {"rain", "sprinkler"}

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty|nodes|edges"):
            BayesianNetworkVisualizer(nodes=[], edges=[])

    def test_rejects_bad_color_mode(self):
        with pytest.raises(ValueError, match="color_by"):
            BayesianNetworkVisualizer(
                nodes=[{"id": "a", "label": "A", "cardinality": 2}],
                edges=[],
                color_by="bogus",
            )

    def test_rejects_bad_layout(self):
        with pytest.raises(ValueError, match="layout"):
            BayesianNetworkVisualizer(
                nodes=[{"id": "a", "label": "A", "cardinality": 2}],
                edges=[],
                layout="spiral",
            )


# ---------------------------------------------------------------------------
# Fallback paths (edges_, dag_, parents_)
# ---------------------------------------------------------------------------


class TestFallbackPaths:
    def test_edges_attribute(self):
        class FakeModel:
            edges_ = [(0, 1), (0, 2), (1, 2)]
            classes_ = np.array([0, 1])
            feature_names_in_ = np.array(["a", "b", "c"])
            n_features_in_ = 3

        viz = BayesianNetworkVisualizer(
            FakeModel(), feature_names=["a", "b", "c"], class_names=["n", "y"],
        )
        p = json.loads(viz.to_json())
        # 3 features + 1 class = 4 nodes
        assert len(p["nodes"]) == 4
        # Every feature gets a class parent added.
        for n in p["nodes"]:
            if n["role"] == "feature":
                assert "class" in n["parents"]

    def test_dag_attribute(self):
        class FakeDag:
            dag_ = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])

        viz = BayesianNetworkVisualizer(
            FakeDag(), feature_names=["a", "b", "c"],
        )
        p = json.loads(viz.to_json())
        assert len(p["edges"]) == 3

    def test_unsupported_raises(self):
        class Unsupported:
            pass
        with pytest.raises(ValueError, match="Cannot extract"):
            BayesianNetworkVisualizer(Unsupported())


# ---------------------------------------------------------------------------
# Graph helper primitives
# ---------------------------------------------------------------------------


class TestGraphHelpers:
    def test_topo_layers_chain(self):
        nodes = [
            {"id": "a", "parents": []},
            {"id": "b", "parents": ["a"]},
            {"id": "c", "parents": ["b"]},
        ]
        layers = _topo_layers(nodes)
        assert layers == [[0], [1], [2]]

    def test_topo_layers_parallel(self):
        nodes = [
            {"id": "a", "parents": []},
            {"id": "b", "parents": []},
            {"id": "c", "parents": ["a", "b"]},
        ]
        layers = _topo_layers(nodes)
        assert layers[0] == [0, 1]
        assert layers[1] == [2]

    def test_markov_blanket_standard(self):
        # x -> y -> z, with co-parent w -> z.
        nodes = [
            {"id": "x", "parents": [], "children": ["y"]},
            {"id": "y", "parents": ["x"], "children": ["z"]},
            {"id": "z", "parents": ["y", "w"], "children": []},
            {"id": "w", "parents": [], "children": ["z"]},
        ]
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        mb = _markov_blanket(1, nodes, id_to_idx)   # MB of y
        # MB(y) = parents(y) ∪ children(y) ∪ other-parents-of-children(y)
        #       = {x} ∪ {z} ∪ {w}
        assert sorted(mb) == sorted([id_to_idx["x"], id_to_idx["z"], id_to_idx["w"]])


# ---------------------------------------------------------------------------
# VizBNNode dataclass
# ---------------------------------------------------------------------------


class TestVizBNNode:
    def test_to_dict_round_trip(self):
        n = VizBNNode(
            node_id="x0", label="age", role="feature",
            cardinality=3, states=["young", "mid", "old"],
            parents=["class"], children=[], importance=0.42,
            cpt=[[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]],
            cpt_axes=["class"],
            marginal=[0.33, 0.33, 0.34],
            description="P(age | class)",
        )
        d = n.to_dict()
        assert d["id"] == "x0"
        assert d["role"] == "feature"
        assert d["cardinality"] == 3
        assert d["importance"] == 0.42
        assert d["cptAxes"] == ["class"]

    def test_importance_omitted_when_none(self):
        n = VizBNNode(node_id="x", label="X", cardinality=2)
        d = n.to_dict()
        assert "importance" not in d
