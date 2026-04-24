"""Interactive Bayesian Network Visualization.

Generates gorgeous, self-contained HTML/JavaScript visualizations of Bayesian
Networks with layered DAG layout, Markov-blanket focus, marginal distributions,
and full conditional probability tables (CPTs) in a side panel.

Supports:
- endgame TAN, KDB, ESKDB, EBMC, AutoSLE classifiers (anything with
  ``structure_`` as a ``networkx.DiGraph`` and ``cpts_`` dict)
- Manual construction from (nodes, edges, cpts, marginals)
- Graceful fallback when CPTs or marginals aren't available (structure-only)

Example
-------
>>> from endgame.models.bayesian import TANClassifier
>>> from endgame.visualization import BayesianNetworkVisualizer
>>>
>>> clf = TANClassifier().fit(X, y)
>>> viz = BayesianNetworkVisualizer(
...     clf,
...     feature_names=['age', 'income', 'score', 'region'],
...     class_names=['no', 'yes'],
...     title="TAN: Loan Approval",
... )
>>> viz.save("bn.html")  # Open in browser for interactive visualization
"""

from __future__ import annotations

import html as html_module
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Universal BN node representation for visualization
# ---------------------------------------------------------------------------


@dataclass
class VizBNNode:
    """Universal Bayesian Network node for visualization.

    All supported model formats are converted to a list of VizBNNode before
    being serialized to JSON for the browser renderer.
    """

    node_id: str = ""
    label: str = ""
    role: str = "feature"   # "class" | "feature"

    cardinality: int = 0
    states: list[str] = field(default_factory=list)   # state labels for this node

    parents: list[str] = field(default_factory=list)   # list of node_ids
    children: list[str] = field(default_factory=list)  # list of node_ids (filled later)

    # Conditional Probability Table: shape = (cardinality, *parent_cards).
    # When there are no parents this is just a 1D marginal.
    cpt: list[Any] = field(default_factory=list)
    # Axis labels = parent node_ids in the order they index ``cpt`` (dims 1..).
    # Dim 0 is always the node itself.
    cpt_axes: list[str] = field(default_factory=list)

    marginal: list[float] = field(default_factory=list)
    importance: float | None = None           # MI(X; Y) or analogue
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": str(self.node_id),
            "label": str(self.label),
            "role": str(self.role),
            "cardinality": int(self.cardinality),
            "states": [str(s) for s in self.states],
            "parents": [str(p) for p in self.parents],
            "children": [str(c) for c in self.children],
            "cpt": self.cpt,
            "cptAxes": [str(a) for a in self.cpt_axes],
            "marginal": [round(float(v), 6) for v in self.marginal],
            "description": str(self.description),
        }
        if self.importance is not None:
            d["importance"] = round(float(self.importance), 6)
        return d


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _approx_marginal_from_cpt(cpt: np.ndarray) -> list[float]:
    """Marginal over a node's CPT assuming uniform parent distribution.

    Shape convention: ``cpt.shape == (node_card, *parent_cards)``. Averaging
    over the parent axes gives a mean-parent marginal, which is close enough
    for display purposes and avoids needing full ancestor joints.
    """
    arr = np.asarray(cpt, dtype=float)
    if arr.ndim == 0:
        return []
    if arr.ndim == 1:
        s = float(arr.sum()) or 1.0
        return [float(x / s) for x in arr]
    # Mean over all parent axes (axes 1..end).
    m = arr.mean(axis=tuple(range(1, arr.ndim)))
    s = float(m.sum()) or 1.0
    return [float(x / s) for x in m]


def _extract_endgame_bayesian(
    model: Any,
    feature_names: Sequence[str] | None,
    class_names: Sequence[str] | None,
) -> list[VizBNNode]:
    """Extract VizBNNodes from an endgame Bayesian classifier.

    The expected contract is the one implemented by
    ``endgame.models.bayesian.base.BaseBayesianClassifier``:
      * ``structure_``  : ``networkx.DiGraph`` with feature ids (ints) and 'Y'
      * ``cpts_``       : ``dict[node_id -> np.ndarray]`` (shape as above)
      * ``cardinalities_`` : ``dict[int, int]`` per feature id
      * ``class_prior_`` : 1D marginal over classes
      * ``classes_``    : class labels
      * ``feature_importances_`` : optional MI-based scores
    """
    if not hasattr(model, "structure_") or model.structure_ is None:
        raise ValueError(
            "Model has no fitted ``structure_``. Call ``fit()`` first, "
            "or pass structure via ``nodes``/``edges``/``cpts`` directly."
        )

    dag = model.structure_
    cpts = dict(getattr(model, "cpts_", {}) or {})
    cards: dict[Any, int] = dict(getattr(model, "cardinalities_", {}) or {})
    class_prior = getattr(model, "class_prior_", None)
    classes = getattr(model, "classes_", None)
    importances = getattr(model, "feature_importances_", None)

    if feature_names is None:
        feature_names = list(getattr(model, "feature_names_in_", []) or [])

    n_features = int(getattr(model, "n_features_in_", 0) or 0)

    # Resolve a display name for each node id in the DAG.
    def feat_label(idx: int) -> str:
        if feature_names and 0 <= idx < len(feature_names):
            return str(feature_names[idx])
        return f"x{idx}"

    def node_label(node: Any) -> str:
        if node == "Y":
            return "Class"
        try:
            return feat_label(int(node))
        except (TypeError, ValueError):
            return str(node)

    # Class node cardinality & state labels.
    if classes is not None:
        class_card = int(len(classes))
        if class_names is None:
            class_states = [str(c) for c in classes]
        else:
            class_states = [str(c) for c in class_names]
    else:
        class_card = int(class_prior.shape[0]) if class_prior is not None else 0
        class_states = (
            [str(c) for c in class_names]
            if class_names
            else [f"c{i}" for i in range(class_card)]
        )

    def node_cardinality(node: Any) -> int:
        if node == "Y":
            return class_card
        try:
            idx = int(node)
        except (TypeError, ValueError):
            return 0
        if idx in cards:
            return int(cards[idx])
        cpt = cpts.get(node) if cpts else None
        if cpt is not None:
            return int(np.asarray(cpt).shape[0])
        return 0

    def node_states(node: Any, card: int) -> list[str]:
        if node == "Y":
            return list(class_states)
        return [f"s{i}" for i in range(card)]

    nodes: list[VizBNNode] = []
    id_by_node: dict[Any, str] = {}

    # Preserve a natural order: class first, then by feature index.
    dag_nodes = list(dag.nodes())

    def sort_key(n: Any) -> tuple[int, Any]:
        if n == "Y":
            return (0, -1)
        try:
            return (1, int(n))
        except (TypeError, ValueError):
            return (2, str(n))

    dag_nodes.sort(key=sort_key)

    for node in dag_nodes:
        node_id = "class" if node == "Y" else f"x{int(node)}" if isinstance(node, (int, np.integer)) else str(node)
        id_by_node[node] = node_id

    for node in dag_nodes:
        node_id = id_by_node[node]
        role = "class" if node == "Y" else "feature"
        card = node_cardinality(node)
        states = node_states(node, card)

        parent_nodes = list(dag.predecessors(node))
        parent_ids = [id_by_node.get(p, str(p)) for p in parent_nodes]

        # Build CPT payload.
        cpt_arr: np.ndarray | None = None
        marginal: list[float] = []

        if node == "Y":
            if class_prior is not None:
                cpt_arr = np.asarray(class_prior, dtype=float).reshape(-1)
                marginal = [float(x) for x in cpt_arr]
        else:
            raw_cpt = cpts.get(node)
            if raw_cpt is not None:
                cpt_arr = np.asarray(raw_cpt, dtype=float)
                marginal = _approx_marginal_from_cpt(cpt_arr)

        cpt_payload: list[Any] = cpt_arr.round(6).tolist() if cpt_arr is not None else []

        # Importance (MI with class, for features).
        imp: float | None = None
        if role == "feature" and importances is not None:
            try:
                idx = int(node)
                if 0 <= idx < len(importances):
                    imp = float(importances[idx])
            except (TypeError, ValueError):
                imp = None

        description = ""
        if role == "class":
            description = "Target / class variable"
        elif parent_nodes:
            description = (
                f"P({node_label(node)} | "
                + ", ".join(node_label(p) for p in parent_nodes)
                + ")"
            )
        else:
            description = f"P({node_label(node)})"

        nodes.append(
            VizBNNode(
                node_id=node_id,
                label=node_label(node),
                role=role,
                cardinality=card,
                states=states,
                parents=parent_ids,
                cpt=cpt_payload,
                cpt_axes=parent_ids,
                marginal=marginal,
                importance=imp,
                description=description,
            )
        )

    # Back-fill children for convenience in the UI.
    by_id = {n.node_id: n for n in nodes}
    for n in nodes:
        for p in n.parents:
            if p in by_id:
                by_id[p].children.append(n.node_id)

    # Optionally include features that appear only via edges_/cardinalities_
    # but are not in structure_ (shouldn't happen, but keeps us forgiving).
    if n_features and feature_names:
        known = {n.node_id for n in nodes}
        for i, name in enumerate(feature_names):
            nid = f"x{i}"
            if nid not in known:
                nodes.append(
                    VizBNNode(
                        node_id=nid,
                        label=str(name),
                        role="feature",
                        cardinality=int(cards.get(i, 0)),
                    )
                )

    return nodes


def _extract_from_edges(
    edges: Sequence[tuple[Any, Any]],
    feature_names: Sequence[str] | None,
    class_names: Sequence[str] | None,
    has_class: bool,
) -> list[VizBNNode]:
    """Fallback: build a structure-only BN from an edge list."""
    idx_nodes: dict[str, VizBNNode] = {}

    def label_for(node: Any) -> str:
        if node == "Y" or (isinstance(node, str) and node.lower() in {"y", "class"}):
            return "Class"
        try:
            idx = int(node)
            if feature_names and 0 <= idx < len(feature_names):
                return str(feature_names[idx])
            return f"x{idx}"
        except (TypeError, ValueError):
            return str(node)

    def id_for(node: Any) -> str:
        if node == "Y" or (isinstance(node, str) and node.lower() in {"y", "class"}):
            return "class"
        try:
            return f"x{int(node)}"
        except (TypeError, ValueError):
            return str(node)

    def role_for(node: Any) -> str:
        return "class" if id_for(node) == "class" else "feature"

    for u, v in edges:
        for node in (u, v):
            nid = id_for(node)
            if nid not in idx_nodes:
                idx_nodes[nid] = VizBNNode(
                    node_id=nid, label=label_for(node), role=role_for(node),
                )

    # Optionally insert an explicit class node.
    if has_class and "class" not in idx_nodes:
        idx_nodes["class"] = VizBNNode(
            node_id="class", label="Class", role="class",
        )

    for u, v in edges:
        src = id_for(u)
        tgt = id_for(v)
        if src in idx_nodes and tgt in idx_nodes:
            idx_nodes[tgt].parents.append(src)
            idx_nodes[src].children.append(tgt)

    return list(idx_nodes.values())


# ---------------------------------------------------------------------------
# Public visualizer class
# ---------------------------------------------------------------------------


_COLOR_MODES = ("role", "importance", "cardinality", "indegree")
_LAYOUTS = ("layered", "force", "circular")


class BayesianNetworkVisualizer:
    """Interactive Bayesian Network visualizer.

    Produces a self-contained HTML file with:

    * Layered / force-directed / circular DAG layouts with curved edges +
      arrowheads.
    * Class node rendered distinctly (crown glyph, halo ring).
    * Click a node to focus on its **Markov blanket** (parents + children +
      children's other parents); non-blanket nodes fade.
    * Side panel showing the selected node's CPT, marginal bar, parents, and
      children. CPTs with > 2 parents are flattened to a parent-configuration
      × node-state table.
    * Color modes: ``role`` (class vs feature vs root), ``importance``
      (feature-importance heat), ``cardinality``, ``indegree``.
    * Search, minimap, zoom/pan, keyboard shortcuts.

    Parameters
    ----------
    model : estimator, optional
        A fitted Bayesian Network model (TAN / KDB / ESKDB / EBMC / AutoSLE)
        or any object with a ``structure_`` attribute exposing a
        ``networkx.DiGraph``. Mutually exclusive with passing explicit
        ``nodes``/``edges``.
    feature_names : list of str, optional
        Human-readable feature names. If omitted, uses the model's
        ``feature_names_in_`` or falls back to ``x0, x1, ...``.
    class_names : list of str, optional
        Class label names (classification only). Defaults to ``classes_``.
    nodes, edges, cpts : optional
        Manual construction path. Use ``nodes`` as a list of dicts
        (``{'id', 'label', 'role', 'cardinality', 'states'}``), ``edges`` as
        ``[(parent_id, child_id), ...]``, and ``cpts`` as a dict of
        ``{node_id: np.ndarray}``.
    title : str, optional
        Title displayed above the visualization.
    color_by : {'role', 'importance', 'cardinality', 'indegree'}, default='role'
        Initial node coloring mode.
    layout : {'layered', 'force', 'circular'}, default='layered'
        Initial DAG layout.
    palette : str, default='tableau'
        Color palette: 'tableau', 'viridis', 'pastel', or 'dark'.

    Examples
    --------
    From a fitted model::

        viz = BayesianNetworkVisualizer(
            tan_clf,
            feature_names=["age", "income"],
            class_names=["no", "yes"],
            title="TAN",
        )
        viz.save("bn.html")

    Manual construction::

        viz = BayesianNetworkVisualizer(
            nodes=[
                {"id": "rain", "label": "Rain", "role": "feature", "cardinality": 2},
                {"id": "sprinkler", "label": "Sprinkler", "role": "feature", "cardinality": 2},
                {"id": "wet", "label": "Wet grass", "role": "feature", "cardinality": 2},
            ],
            edges=[("rain", "sprinkler"), ("rain", "wet"), ("sprinkler", "wet")],
            title="Wet Grass",
        )
    """

    def __init__(
        self,
        model: Any = None,
        *,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        nodes: Sequence[dict[str, Any]] | None = None,
        edges: Sequence[tuple[Any, Any]] | None = None,
        cpts: dict[str, Any] | None = None,
        title: str | None = None,
        color_by: str = "role",
        layout: str = "layered",
        palette: str = "tableau",
    ):
        if color_by not in _COLOR_MODES:
            raise ValueError(
                f"color_by must be one of {_COLOR_MODES}, got {color_by!r}"
            )
        if layout not in _LAYOUTS:
            raise ValueError(
                f"layout must be one of {_LAYOUTS}, got {layout!r}"
            )

        self.model = model
        self.feature_names = list(feature_names) if feature_names else None
        self.class_names = list(class_names) if class_names else None
        self.title = title
        self.color_by = color_by
        self.layout = layout
        self.palette = palette

        if model is not None:
            self._nodes = self._extract_from_model(model)
        elif nodes is not None:
            self._nodes = self._from_manual(
                nodes=nodes, edges=edges or [], cpts=cpts or {},
            )
        elif edges is not None:
            self._nodes = _extract_from_edges(
                edges, self.feature_names, self.class_names,
                has_class=any(
                    str(e).lower() in {"y", "class"}
                    for pair in edges for e in pair
                ),
            )
        else:
            raise ValueError(
                "Provide either a fitted ``model`` or explicit "
                "``nodes``/``edges``/``cpts``."
            )

        # Sanity check.
        if not self._nodes:
            raise ValueError(
                "No nodes extracted; network is empty."
            )

    # ------------------------------------------------------------------
    # Extraction paths
    # ------------------------------------------------------------------

    def _extract_from_model(self, model: Any) -> list[VizBNNode]:
        # Primary path: endgame Bayesian classifiers with structure_.
        if hasattr(model, "structure_") and model.structure_ is not None:
            return _extract_endgame_bayesian(
                model, self.feature_names, self.class_names,
            )

        # Fallbacks compatible with NetworkDiagramVisualizer contract.
        if hasattr(model, "edges_"):
            edges = [(int(p), int(c)) for p, c in model.edges_]
            has_class = hasattr(model, "classes_")
            nodes = _extract_from_edges(
                edges, self.feature_names, self.class_names, has_class,
            )
            # Add class→feature edges (TAN/KDB convention).
            if has_class:
                feature_ids = [n.node_id for n in nodes if n.role == "feature"]
                for fid in feature_ids:
                    class_node = next((n for n in nodes if n.role == "class"), None)
                    if class_node and class_node.node_id not in next(
                        (n.parents for n in nodes if n.node_id == fid), []
                    ):
                        fnode = next(n for n in nodes if n.node_id == fid)
                        fnode.parents.insert(0, class_node.node_id)
                        class_node.children.append(fid)
            return nodes

        if hasattr(model, "dag_"):
            dag = np.asarray(model.dag_)
            n = dag.shape[0]
            edges = [
                (i, j) for i in range(n) for j in range(n) if dag[i, j] != 0
            ]
            return _extract_from_edges(
                edges, self.feature_names, self.class_names, has_class=False,
            )

        if hasattr(model, "parents_"):
            parents = list(model.parents_)
            edges = [(p, i) for i, p in enumerate(parents) if p is not None and int(p) >= 0]
            return _extract_from_edges(
                edges, self.feature_names, self.class_names, has_class=False,
            )

        raise ValueError(
            f"Cannot extract Bayesian network from {type(model).__name__}. "
            "Model must expose ``structure_`` (networkx.DiGraph), "
            "``edges_``, ``dag_``, or ``parents_``."
        )

    def _from_manual(
        self,
        nodes: Sequence[dict[str, Any]],
        edges: Sequence[tuple[str, str]],
        cpts: dict[str, Any],
    ) -> list[VizBNNode]:
        viz_nodes: list[VizBNNode] = []
        by_id: dict[str, VizBNNode] = {}
        for n in nodes:
            nid = str(n["id"])
            role = str(n.get("role", "feature"))
            card = int(n.get("cardinality") or 0)
            states = [str(s) for s in n.get("states", [])] or [
                f"s{i}" for i in range(card)
            ] if card else []
            viz = VizBNNode(
                node_id=nid,
                label=str(n.get("label", nid)),
                role=role,
                cardinality=card,
                states=states,
                description=str(n.get("description", "")),
            )
            viz_nodes.append(viz)
            by_id[nid] = viz

        # Parent/child wiring.
        for src, tgt in edges:
            if src in by_id and tgt in by_id:
                by_id[tgt].parents.append(src)
                by_id[src].children.append(tgt)

        # Attach CPTs.
        for nid, cpt in (cpts or {}).items():
            if nid in by_id:
                arr = np.asarray(cpt, dtype=float)
                by_id[nid].cpt = arr.round(6).tolist()
                by_id[nid].cpt_axes = list(by_id[nid].parents)
                if arr.size:
                    if not by_id[nid].cardinality:
                        by_id[nid].cardinality = int(arr.shape[0])
                    if not by_id[nid].states:
                        by_id[nid].states = [
                            f"s{i}" for i in range(by_id[nid].cardinality)
                        ]
                    by_id[nid].marginal = _approx_marginal_from_cpt(arr)

        return viz_nodes

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _build_payload(self) -> dict[str, Any]:
        """Build the JSON blob consumed by the browser renderer."""
        nodes = [n.to_dict() for n in self._nodes]

        # Importance normalisation (for heat coloring in the UI).
        imps = [n.get("importance") for n in nodes if "importance" in n]
        if imps:
            imp_max = max(imps) or 1.0
            for n in nodes:
                if "importance" in n:
                    n["importanceNorm"] = round(n["importance"] / imp_max, 6)

        # Compute topological layers for layered layout.
        layers = _topo_layers(nodes)

        # Edge list (child-indexed).
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        edges: list[dict[str, Any]] = []
        for node in nodes:
            for p in node["parents"]:
                if p in id_to_idx:
                    edges.append({
                        "source": id_to_idx[p],
                        "target": id_to_idx[node["id"]],
                    })

        # Markov blanket of the class node, if one exists.
        class_idx = next(
            (i for i, n in enumerate(nodes) if n["role"] == "class"), None
        )
        class_mb: list[int] = []
        if class_idx is not None:
            class_mb = _markov_blanket(class_idx, nodes, id_to_idx)

        return {
            "nodes": nodes,
            "edges": edges,
            "layers": layers,
            "classIndex": class_idx if class_idx is not None else -1,
            "classMarkovBlanket": class_mb,
        }

    def to_json(self) -> str:
        """Export the BN structure + CPTs as a JSON string."""
        return json.dumps(self._build_payload(), indent=2)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def save(self, filepath: str | Path, open_browser: bool = False) -> Path:
        """Save the interactive visualization as a self-contained HTML file.

        Parameters
        ----------
        filepath : str or Path
            Output path. ``.html`` is appended if missing.
        open_browser : bool, default=False
            If True, open the file in the default web browser.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".html")

        html_content = _generate_html(
            payload=self._build_payload(),
            title=self.title or "Bayesian Network",
            color_by=self.color_by,
            layout=self.layout,
            palette=self.palette,
        )
        filepath.write_text(html_content, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(filepath.resolve().as_uri())

        return filepath.resolve()

    def _repr_html_(self) -> str:
        """Jupyter notebook display support."""
        return _generate_html(
            payload=self._build_payload(),
            title=self.title or "Bayesian Network",
            color_by=self.color_by,
            layout=self.layout,
            palette=self.palette,
            embedded=True,
        )


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _topo_layers(nodes: list[dict[str, Any]]) -> list[list[int]]:
    """Compute topological layers (each node = max parent layer + 1)."""
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)
    in_deg = [len(nd["parents"]) for nd in nodes]
    children = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for p in nd["parents"]:
            pi = id_to_idx.get(p)
            if pi is not None:
                children[pi].append(i)

    layer = [0] * n
    visited = [False] * n
    queue: list[int] = [i for i in range(n) if in_deg[i] == 0]
    for i in queue:
        visited[i] = True
    while queue:
        cur = queue.pop(0)
        for ch in children[cur]:
            layer[ch] = max(layer[ch], layer[cur] + 1)
            in_deg[ch] -= 1
            if in_deg[ch] <= 0 and not visited[ch]:
                visited[ch] = True
                queue.append(ch)
    # Cycle safety: unvisited nodes → last layer + 1.
    max_layer = max(layer) if layer else 0
    for i in range(n):
        if not visited[i]:
            layer[i] = max_layer + 1

    grouped: dict[int, list[int]] = {}
    for i, l in enumerate(layer):
        grouped.setdefault(l, []).append(i)
    return [grouped[k] for k in sorted(grouped)]


def _markov_blanket(
    idx: int,
    nodes: list[dict[str, Any]],
    id_to_idx: dict[str, int],
) -> list[int]:
    """Compute Markov blanket of ``nodes[idx]``.

    MB = parents ∪ children ∪ (other parents of children) \\ {node}.
    """
    node = nodes[idx]
    mb: set[int] = set()
    for p in node["parents"]:
        if p in id_to_idx:
            mb.add(id_to_idx[p])
    children_idx = [
        i for i, n in enumerate(nodes) if node["id"] in n["parents"]
    ]
    for ci in children_idx:
        mb.add(ci)
        for p in nodes[ci]["parents"]:
            if p in id_to_idx and id_to_idx[p] != idx:
                mb.add(id_to_idx[p])
    mb.discard(idx)
    return sorted(mb)


# ---------------------------------------------------------------------------
# HTML/CSS/JS template
# ---------------------------------------------------------------------------


_PALETTES = {
    "tableau": [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ],
    "viridis": [
        "#440154", "#482777", "#3e4989", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
    ],
    "pastel": [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
    ],
    "dark": [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#e6ab02", "#a6761d", "#666666", "#e41a1c", "#377eb8",
    ],
}


def _generate_html(
    payload: dict[str, Any],
    title: str,
    color_by: str,
    layout: str,
    palette: str,
    embedded: bool = False,
) -> str:
    """Generate the complete self-contained HTML visualization."""
    palette_colors = _PALETTES.get(palette, _PALETTES["tableau"])
    payload_json = json.dumps(payload, default=float)
    colors_json = json.dumps(palette_colors)
    title_escaped = html_module.escape(title)

    container_height = "600px" if embedded else "100vh"

    return _HTML_TEMPLATE.format(
        title=title_escaped,
        container_height=container_height,
        payload_json=payload_json,
        colors_json=colors_json,
        color_by=color_by,
        layout=layout,
    )


# Note: all literal braces in CSS/JS are doubled so ``.format()`` works.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #0f1117;
  color: #e0e0e0;
  overflow: hidden;
}}

#app {{
  width: 100vw;
  height: {container_height};
  display: flex;
  flex-direction: column;
}}

/* Header */
.header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 24px;
  background: linear-gradient(135deg, #1a1d29, #252836);
  border-bottom: 1px solid rgba(255,255,255,0.06);
  z-index: 100;
  flex-shrink: 0;
}}
.header h1 {{
  font-size: 18px; font-weight: 600; color: #f0f0f0; letter-spacing: -0.3px;
}}
.controls {{ display: flex; gap: 8px; align-items: center; }}
.btn {{
  padding: 6px 14px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 6px;
  background: rgba(255,255,255,0.05);
  color: #c0c0c0;
  font-size: 13px; cursor: pointer;
  transition: all 0.2s; user-select: none;
}}
.btn:hover {{
  background: rgba(255,255,255,0.1); color: #fff;
  border-color: rgba(255,255,255,0.2);
}}
.btn.active {{
  background: rgba(78, 121, 167, 0.3);
  border-color: #4e79a7; color: #a0c4e8;
}}
.stats {{ font-size: 12px; color: #888; padding: 0 12px; }}

/* Workspace: canvas + side panel */
#workspace {{
  flex: 1;
  display: flex;
  min-height: 0;
  position: relative;
}}
#canvas-wrap {{
  flex: 1;
  overflow: hidden;
  position: relative;
  cursor: grab;
  background:
    radial-gradient(ellipse at 30% 20%, rgba(78,121,167,0.07), transparent 60%),
    radial-gradient(ellipse at 80% 80%, rgba(225,87,89,0.05), transparent 55%),
    #0f1117;
}}
#canvas-wrap:active {{ cursor: grabbing; }}
#bn-svg {{ width: 100%; height: 100%; }}

/* Links */
.link {{
  fill: none;
  stroke: rgba(255,255,255,0.22);
  stroke-width: 1.6px;
  transition: stroke 0.25s, stroke-width 0.25s, opacity 0.25s;
}}
.link.dim {{ opacity: 0.15; }}
.link.mb-edge {{ stroke: #f4a261; stroke-width: 2.4px; opacity: 1; }}
.link.selected-edge {{ stroke: #ffffff; stroke-width: 2.6px; opacity: 1; }}

/* Nodes */
.node {{ cursor: pointer; }}
.node-shape {{
  stroke-width: 2px;
  transition: all 0.25s;
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.45));
}}
.node:hover .node-shape {{
  stroke-width: 3px;
  filter: drop-shadow(0 4px 14px rgba(0,0,0,0.7));
}}
.node.dim .node-shape {{ opacity: 0.28; }}
.node.dim .node-label {{ opacity: 0.35; }}
.node.selected .node-shape {{
  stroke-width: 3.5px; filter: drop-shadow(0 0 14px rgba(255,255,255,0.5));
}}
.node.mb .node-shape {{
  stroke: #f4a261;
  filter: drop-shadow(0 0 10px rgba(244,162,97,0.55));
}}

.node-label {{
  font-size: 12px; fill: #f0f0f0; text-anchor: middle;
  pointer-events: none; font-weight: 600;
}}
.node-sublabel {{
  font-size: 9.5px; fill: #98a1b3; text-anchor: middle;
  pointer-events: none;
}}

/* Class-node halo (static dashed ring; SVG-native animate keeps it anchored) */
.class-halo {{
  fill: none; stroke: rgba(237, 201, 72, 0.55); stroke-width: 2px;
  stroke-dasharray: 4 3;
}}

/* Marginal bar under node */
.marg-bar .seg {{ transition: width 0.3s; }}

/* Side panel */
#side-panel {{
  width: 340px;
  min-width: 280px;
  max-width: 420px;
  background: linear-gradient(180deg, #181b26, #141620);
  border-left: 1px solid rgba(255,255,255,0.07);
  padding: 18px 18px 22px;
  overflow-y: auto;
  flex-shrink: 0;
  display: flex; flex-direction: column; gap: 14px;
}}
#side-panel h2 {{
  font-size: 13px; font-weight: 600; color: #dbe1ee;
  letter-spacing: 0.3px; text-transform: uppercase;
  margin-bottom: 4px;
}}
#side-panel h3 {{
  font-size: 11px; font-weight: 600; color: #98a1b3;
  letter-spacing: 0.4px; text-transform: uppercase;
  margin-bottom: 6px;
}}
#side-panel .node-title {{
  font-size: 18px; font-weight: 600; color: #f0f0f0;
  letter-spacing: -0.2px;
}}
#side-panel .node-role {{
  display: inline-block; padding: 2px 8px; font-size: 10px;
  font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase;
  border-radius: 4px; margin-left: 6px; vertical-align: middle;
}}
#side-panel .node-role.class {{ background: rgba(237,201,72,0.18); color: #edc948; }}
#side-panel .node-role.feature {{ background: rgba(118,183,178,0.18); color: #76b7b2; }}
.panel-section {{
  padding: 12px 14px; border-radius: 10px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.05);
}}
.meta-row {{
  display: flex; justify-content: space-between; font-size: 12px;
  padding: 3px 0; color: #b9bfcc;
}}
.meta-row .l {{ color: #7f8597; }}
.meta-row .v {{ color: #e6e8ef; font-weight: 500; }}
.chip {{
  display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0;
  font-size: 11px; border-radius: 999px;
  background: rgba(255,255,255,0.06); color: #d0d4df;
  border: 1px solid rgba(255,255,255,0.06);
  cursor: pointer; transition: all 0.15s;
}}
.chip:hover {{ background: rgba(78,121,167,0.25); color: #fff; border-color: #4e79a7; }}

/* Marginal (side-panel) */
.marg-bar-big {{
  display: flex; height: 14px; border-radius: 5px; overflow: hidden;
  margin: 6px 0 8px;
  background: rgba(255,255,255,0.04);
}}
.marg-legend {{
  display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 4px;
}}
.marg-legend .item {{
  display: flex; align-items: center; gap: 5px;
  font-size: 11px; color: #c5cad6;
}}
.marg-legend .swatch {{
  width: 9px; height: 9px; border-radius: 2px;
}}

/* CPT */
.cpt-table {{
  width: 100%; border-collapse: collapse;
  font-size: 11.5px; color: #d7dbe5;
}}
.cpt-table th, .cpt-table td {{
  padding: 5px 6px; text-align: right;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  font-variant-numeric: tabular-nums;
}}
.cpt-table th {{
  font-weight: 600; color: #98a1b3; text-align: right;
  font-size: 10.5px; letter-spacing: 0.2px; text-transform: uppercase;
}}
.cpt-table td:first-child, .cpt-table th:first-child {{ text-align: left; color: #c5cad6; }}
.cpt-cell {{ position: relative; }}
.cpt-cell .fill {{
  position: absolute; left: 0; top: 0; bottom: 0;
  background: rgba(78,121,167,0.22);
  border-radius: 2px; z-index: 0;
}}
.cpt-cell span {{ position: relative; z-index: 1; }}

/* Tooltip */
.tooltip {{
  position: absolute;
  background: linear-gradient(135deg, #252836, #1e2130);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 12px 16px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
  z-index: 1000;
  min-width: 220px;
  max-width: 320px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  backdrop-filter: blur(10px);
}}
.tooltip.visible {{ opacity: 1; }}
.tooltip h3 {{
  font-size: 13px; font-weight: 600; color: #f0f0f0;
  margin-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.08);
  padding-bottom: 5px;
}}
.tooltip .r {{
  display: flex; justify-content: space-between;
  font-size: 11.5px; padding: 2px 0;
}}
.tooltip .r .l {{ color: #8a90a0; }}
.tooltip .r .v {{ color: #ddd; font-weight: 500; }}

/* Zoom indicator */
.zoom-level {{
  position: absolute; bottom: 16px; left: 16px;
  background: rgba(15, 17, 23, 0.8);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 6px; padding: 6px 12px;
  font-size: 11px; color: #888; z-index: 50;
}}

.search-box input {{
  padding: 6px 12px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 6px;
  background: rgba(255,255,255,0.05);
  color: #e0e0e0; font-size: 13px;
  width: 160px; outline: none;
  transition: all 0.2s;
}}
.search-box input:focus {{
  border-color: #4e79a7;
  background: rgba(255,255,255,0.08);
  width: 200px;
}}
.search-box input::placeholder {{ color: #666; }}

/* Minimap */
.minimap {{
  position: absolute; bottom: 16px; right: 16px;
  width: 180px; height: 120px;
  background: rgba(15, 17, 23, 0.85);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px; overflow: hidden; z-index: 50;
}}
.minimap-viewport {{
  stroke: #4e79a7; stroke-width: 1.5px;
  fill: rgba(78, 121, 167, 0.1);
}}

/* Scrollbars */
#side-panel::-webkit-scrollbar {{ width: 8px; }}
#side-panel::-webkit-scrollbar-track {{ background: transparent; }}
#side-panel::-webkit-scrollbar-thumb {{
  background: rgba(255,255,255,0.1); border-radius: 4px;
}}
</style>
</head>
<body>
<div id="app">
  <div class="header">
    <h1>{title}</h1>
    <div class="controls">
      <div class="search-box">
        <input type="text" id="search-input" placeholder="Search nodes..." />
      </div>
      <button class="btn" id="btn-layout" onclick="cycleLayout()" title="Change layout">Layout: {layout}</button>
      <button class="btn" id="btn-color" onclick="cycleColor()" title="Change color mode">Color: {color_by}</button>
      <button class="btn" onclick="focusClass()" title="Highlight the class node's Markov blanket">Focus class MB</button>
      <button class="btn" onclick="resetView()" title="Reset view">Reset</button>
      <span class="stats" id="stats-text"></span>
    </div>
  </div>
  <div id="workspace">
    <div id="canvas-wrap">
      <svg id="bn-svg"></svg>
      <div class="tooltip" id="tooltip"></div>
      <div class="zoom-level" id="zoom-level">100%</div>
      <svg class="minimap" id="minimap" viewBox="0 0 180 120"></svg>
    </div>
    <aside id="side-panel">
      <div id="panel-empty">
        <h2>Network Summary</h2>
        <div class="panel-section">
          <div class="meta-row"><span class="l">Nodes</span><span class="v" id="sum-nodes">—</span></div>
          <div class="meta-row"><span class="l">Edges</span><span class="v" id="sum-edges">—</span></div>
          <div class="meta-row"><span class="l">Layers</span><span class="v" id="sum-layers">—</span></div>
          <div class="meta-row"><span class="l">Class MB size</span><span class="v" id="sum-mb">—</span></div>
        </div>
        <p style="font-size:12px;color:#8b91a2;line-height:1.5;margin-top:4px;">
          <b>Click a node</b> to see its conditional probability table, marginal,
          and Markov blanket. <b>Scroll</b> to zoom, <b>drag</b> to pan. Press
          <code style="background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:3px;">/</code>
          to search.
        </p>
      </div>
      <div id="panel-selected" style="display:none;"></div>
    </aside>
  </div>
</div>

<script>
(function() {{
"use strict";

// ===== Data =====
const PAYLOAD = {payload_json};
const PALETTE = {colors_json};
let colorMode = "{color_by}";
let layoutMode = "{layout}";
const COLOR_MODES = ['role', 'importance', 'cardinality', 'indegree'];
const LAYOUTS = ['layered', 'force', 'circular'];

// ===== State =====
const nodes = PAYLOAD.nodes.map(function(n, i) {{
  return Object.assign({{}}, n, {{
    index: i, x: 0, y: 0, vx: 0, vy: 0, fixed: false,
  }});
}});
const edges = PAYLOAD.edges.slice();
const classIndex = PAYLOAD.classIndex;
let selectedIdx = -1;
let focusMB = []; // indices of nodes to keep fully-opaque (selected + MB)
let transform = {{ x: 0, y: 0, k: 1 }};
let dragging = false;
let dragNodeIdx = -1;
let dragStart = {{ x: 0, y: 0 }};

// ===== DOM =====
const svg = document.getElementById('bn-svg');
const minimap = document.getElementById('minimap');
const wrap = document.getElementById('canvas-wrap');
const tooltip = document.getElementById('tooltip');
const zoomLabel = document.getElementById('zoom-level');
const panelEmpty = document.getElementById('panel-empty');
const panelSelected = document.getElementById('panel-selected');
const searchInput = document.getElementById('search-input');

function ns(tag, attrs) {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}}

function escHtml(s) {{
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

// ===== Layouts =====
function layoutLayered(W, H) {{
  const layers = PAYLOAD.layers;
  const L = layers.length;
  const pad = 80;
  for (let li = 0; li < L; li++) {{
    const layer = layers[li];
    const x = pad + (W - 2 * pad) * (L === 1 ? 0.5 : li / (L - 1));
    // Sort nodes within layer: class first, then alphabetical.
    const sorted = layer.slice().sort(function(a, b) {{
      const na = nodes[a], nb = nodes[b];
      if (na.role === 'class' && nb.role !== 'class') return -1;
      if (nb.role === 'class' && na.role !== 'class') return 1;
      return na.label.localeCompare(nb.label);
    }});
    for (let pi = 0; pi < sorted.length; pi++) {{
      const ni = sorted[pi];
      const y = pad + (H - 2 * pad) * ((pi + 0.5) / sorted.length);
      nodes[ni].x = x; nodes[ni].y = y;
    }}
  }}
}}

function layoutCircular(W, H) {{
  const cx = W / 2, cy = H / 2;
  const r = Math.min(W, H) * 0.38;
  const n = nodes.length;
  for (let i = 0; i < n; i++) {{
    // Put class at the top.
    let ord = i;
    if (classIndex >= 0) {{
      if (i === 0) ord = classIndex;
      else if (i <= classIndex) ord = i - 1;
    }}
    const angle = 2 * Math.PI * ord / n - Math.PI / 2;
    nodes[i].x = cx + r * Math.cos(angle);
    nodes[i].y = cy + r * Math.sin(angle);
  }}
}}

function layoutForce(W, H) {{
  // Start from layered layout for a stable seed.
  layoutLayered(W, H);
  const n = nodes.length;
  if (n <= 1) return;
  const area = W * H;
  const k = Math.sqrt(area / n) * 0.7;
  const iters = 160;
  const temp0 = Math.min(W, H) / 6;
  for (let it = 0; it < iters; it++) {{
    const temp = temp0 * (1 - it / iters);
    const disp = nodes.map(function() {{ return {{ x: 0, y: 0 }}; }});
    for (let i = 0; i < n; i++) {{
      for (let j = i + 1; j < n; j++) {{
        let dx = nodes[i].x - nodes[j].x;
        let dy = nodes[i].y - nodes[j].y;
        let d = Math.sqrt(dx*dx + dy*dy) || 0.1;
        const f = k*k / d;
        disp[i].x += dx/d * f; disp[i].y += dy/d * f;
        disp[j].x -= dx/d * f; disp[j].y -= dy/d * f;
      }}
    }}
    for (const e of edges) {{
      let dx = nodes[e.target].x - nodes[e.source].x;
      let dy = nodes[e.target].y - nodes[e.source].y;
      let d = Math.sqrt(dx*dx + dy*dy) || 0.1;
      const f = d*d / k;
      disp[e.source].x += dx/d * f; disp[e.source].y += dy/d * f;
      disp[e.target].x -= dx/d * f; disp[e.target].y -= dy/d * f;
    }}
    for (let i = 0; i < n; i++) {{
      if (nodes[i].fixed) continue;
      let dx = disp[i].x, dy = disp[i].y;
      let d = Math.sqrt(dx*dx + dy*dy) || 0.1;
      nodes[i].x += dx/d * Math.min(d, temp);
      nodes[i].y += dy/d * Math.min(d, temp);
      nodes[i].x = Math.max(40, Math.min(W - 40, nodes[i].x));
      nodes[i].y = Math.max(40, Math.min(H - 40, nodes[i].y));
    }}
  }}
}}

function relayout() {{
  const rect = wrap.getBoundingClientRect();
  const W = Math.max(400, rect.width), H = Math.max(300, rect.height);
  if (layoutMode === 'circular') layoutCircular(W, H);
  else if (layoutMode === 'force') layoutForce(W, H);
  else layoutLayered(W, H);
}}

// ===== Coloring =====
function nodeColor(node) {{
  if (colorMode === 'role') {{
    if (node.role === 'class') return '#edc948';
    if ((node.parents || []).length === 0) return PALETTE[3 % PALETTE.length];
    if ((node.children || []).length === 0) return PALETTE[2 % PALETTE.length];
    return PALETTE[0];
  }}
  if (colorMode === 'importance') {{
    const v = (node.importanceNorm != null) ? node.importanceNorm : 0.2;
    // Heat: cool → hot
    const r = Math.round(50 + 200 * v);
    const g = Math.round(90 + 60 * (1 - v));
    const b = Math.round(180 - 150 * v);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }}
  if (colorMode === 'cardinality') {{
    const c = Math.min(node.cardinality || 1, 10);
    return PALETTE[(c - 1 + PALETTE.length) % PALETTE.length];
  }}
  // indegree
  const deg = (node.parents || []).length;
  return PALETTE[Math.min(deg, PALETTE.length - 1)];
}}

function darken(color, factor) {{
  factor = factor == null ? 0.28 : factor;
  if (color.startsWith('rgb')) {{
    const m = color.match(/[0-9.]+/g);
    if (m) return 'rgba(' + Math.round(m[0]*factor) + ',' + Math.round(m[1]*factor) + ',' + Math.round(m[2]*factor) + ',0.85)';
  }}
  const r = parseInt(color.slice(1,3),16), g = parseInt(color.slice(3,5),16), b = parseInt(color.slice(5,7),16);
  return 'rgba(' + Math.round(r*factor) + ',' + Math.round(g*factor) + ',' + Math.round(b*factor) + ',0.85)';
}}

// ===== Rendering =====
function clear(el) {{ while (el.firstChild) el.removeChild(el.firstChild); }}

function nodeRadius(node) {{
  const base = node.role === 'class' ? 34 : 26;
  const bonus = Math.min((node.children || []).length + (node.parents || []).length, 8) * 1.2;
  return base + bonus;
}}

function render() {{
  clear(svg);
  // <defs> with arrow markers (normal + highlighted)
  const defs = ns('defs');
  function makeMarker(id, fill) {{
    const m = ns('marker', {{
      id: id, markerWidth: 10, markerHeight: 7,
      refX: 9.5, refY: 3.5, orient: 'auto', 'markerUnits': 'userSpaceOnUse',
    }});
    m.appendChild(ns('polygon', {{ points: '0 0, 10 3.5, 0 7', fill: fill }}));
    return m;
  }}
  defs.appendChild(makeMarker('bn-arrow', 'rgba(255,255,255,0.45)'));
  defs.appendChild(makeMarker('bn-arrow-hl', '#f4a261'));
  defs.appendChild(makeMarker('bn-arrow-sel', '#ffffff'));
  svg.appendChild(defs);

  const g = ns('g', {{ id: 'root-g' }});
  svg.appendChild(g);
  updateTransform();

  const linkGroup = ns('g'); g.appendChild(linkGroup);
  const nodeGroup = ns('g'); g.appendChild(nodeGroup);

  // Edges (bezier)
  for (let ei = 0; ei < edges.length; ei++) {{
    const e = edges[ei];
    const s = nodes[e.source], t = nodes[e.target];
    const path = ns('path', {{
      class: 'link',
      'marker-end': 'url(#bn-arrow)',
      'data-edge': ei,
    }});
    linkGroup.appendChild(path);
    e._el = path;
  }}

  // Nodes
  for (let i = 0; i < nodes.length; i++) {{
    const nd = nodes[i];
    const grp = ns('g', {{ class: 'node', 'data-idx': i }});
    grp.addEventListener('click', function(ev) {{ ev.stopPropagation(); selectNode(i); }});
    grp.addEventListener('mouseenter', function(ev) {{ showTooltip(ev, nd); }});
    grp.addEventListener('mouseleave', hideTooltip);
    grp.addEventListener('mousedown', function(ev) {{
      if (ev.button !== 0) return;
      dragNodeIdx = i;
      nd.fixed = true;
      ev.stopPropagation();
    }});

    const r = nodeRadius(nd);
    const color = nodeColor(nd);
    const fill = darken(color, nd.role === 'class' ? 0.32 : 0.26);

    // Halo for class node — inner decorative ring + outer soft ring
    if (nd.role === 'class') {{
      const halo = ns('circle', {{
        class: 'class-halo', cx: 0, cy: 0, r: r + 8,
      }});
      grp.appendChild(halo);
      const halo2 = ns('circle', {{
        cx: 0, cy: 0, r: r + 16,
        fill: 'none', stroke: 'rgba(237,201,72,0.18)', 'stroke-width': 1,
      }});
      grp.appendChild(halo2);
    }}

    // Shape: circle for features, rounded square for class.
    let shape;
    if (nd.role === 'class') {{
      shape = ns('rect', {{
        class: 'node-shape',
        x: -r, y: -r, width: r*2, height: r*2,
        rx: 10, ry: 10,
        fill: fill, stroke: color,
      }});
    }} else {{
      shape = ns('circle', {{
        class: 'node-shape', r: r,
        fill: fill, stroke: color,
      }});
    }}
    grp.appendChild(shape);
    nd._shape = shape;

    // Marginal mini-bar inside the node
    if (nd.marginal && nd.marginal.length > 0) {{
      const barW = r * 1.5;
      const barH = 4;
      const barY = r * 0.35;
      const bg = ns('rect', {{
        x: -barW/2, y: barY, width: barW, height: barH,
        rx: 2, ry: 2, fill: 'rgba(0,0,0,0.4)',
      }});
      grp.appendChild(bg);
      let offset = 0;
      for (let k = 0; k < nd.marginal.length; k++) {{
        const w = Math.max(0, nd.marginal[k]) * barW;
        const seg = ns('rect', {{
          x: -barW/2 + offset, y: barY,
          width: w, height: barH,
          fill: PALETTE[k % PALETTE.length],
          opacity: 0.9,
        }});
        grp.appendChild(seg);
        offset += w;
      }}
    }}

    // Label
    const label = ns('text', {{ class: 'node-label', y: -r - 8 }});
    label.textContent = nd.label.length > 18 ? nd.label.slice(0, 16) + '…' : nd.label;
    grp.appendChild(label);

    // Sub-label: cardinality
    if (nd.cardinality) {{
      const sub = ns('text', {{ class: 'node-sublabel', y: r + 14 }});
      sub.textContent = '|' + nd.cardinality + '|' + (nd.role === 'class' ? ' classes' : ' states');
      grp.appendChild(sub);
    }}

    nd._group = grp;
    nodeGroup.appendChild(grp);
  }}

  updatePositions();
  updateStats();
  updateFocus();
  renderMinimap();
}}

function updatePositions() {{
  // Edges
  for (let ei = 0; ei < edges.length; ei++) {{
    const e = edges[ei];
    const s = nodes[e.source], t = nodes[e.target];
    const rs = nodeRadius(s), rt = nodeRadius(t);
    const dx = t.x - s.x, dy = t.y - s.y;
    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
    const ux = dx / dist, uy = dy / dist;
    const sx = s.x + ux * rs;
    const sy = s.y + uy * rs;
    const tx = t.x - ux * (rt + 6);
    const ty = t.y - uy * (rt + 6);
    // Quadratic curve; offset control point perpendicular to the line.
    const mx = (sx + tx) / 2, my = (sy + ty) / 2;
    const perpX = -uy, perpY = ux;
    const curve = 18;
    const cx = mx + perpX * curve, cy = my + perpY * curve;
    e._el.setAttribute('d',
      'M' + sx + ',' + sy + ' Q' + cx + ',' + cy + ' ' + tx + ',' + ty
    );
  }}
  // Nodes
  for (let i = 0; i < nodes.length; i++) {{
    nodes[i]._group.setAttribute('transform', 'translate(' + nodes[i].x + ',' + nodes[i].y + ')');
  }}
  renderMinimap();
}}

// ===== Focus / Markov blanket =====
function markovBlanket(idx) {{
  if (idx < 0 || idx >= nodes.length) return [];
  const n = nodes[idx];
  const ids = new Set();
  for (const p of n.parents || []) ids.add(p);
  for (const c of n.children || []) {{
    ids.add(c);
    const child = nodes.find(function(nd) {{ return nd.id === c; }});
    if (child) for (const p of child.parents || []) if (p !== n.id) ids.add(p);
  }}
  const out = [];
  for (let i = 0; i < nodes.length; i++) {{
    if (ids.has(nodes[i].id)) out.push(i);
  }}
  return out;
}}

function selectNode(idx) {{
  selectedIdx = idx;
  focusMB = markovBlanket(idx);
  updateFocus();
  renderPanel();
}}

function clearSelection() {{
  selectedIdx = -1;
  focusMB = [];
  updateFocus();
  renderPanel();
}}

function updateFocus() {{
  const hasFocus = selectedIdx >= 0;
  const mbSet = new Set(focusMB);
  const inBlanket = function(i) {{
    return i === selectedIdx || mbSet.has(i);
  }};
  // Nodes
  for (let i = 0; i < nodes.length; i++) {{
    const grp = nodes[i]._group;
    grp.classList.remove('selected', 'mb', 'dim');
    if (!hasFocus) continue;
    if (i === selectedIdx) grp.classList.add('selected');
    else if (mbSet.has(i)) grp.classList.add('mb');
    else grp.classList.add('dim');
  }}
  // Edges
  for (const e of edges) {{
    e._el.classList.remove('mb-edge', 'dim', 'selected-edge');
    e._el.setAttribute('marker-end', 'url(#bn-arrow)');
    if (!hasFocus) continue;
    const si = e.source, ti = e.target;
    const touchesSel = (si === selectedIdx || ti === selectedIdx);
    const betweenMB = inBlanket(si) && inBlanket(ti);
    if (touchesSel) {{
      e._el.classList.add('selected-edge');
      e._el.setAttribute('marker-end', 'url(#bn-arrow-sel)');
    }} else if (betweenMB) {{
      e._el.classList.add('mb-edge');
      e._el.setAttribute('marker-end', 'url(#bn-arrow-hl)');
    }} else {{
      e._el.classList.add('dim');
    }}
  }}
}}

// ===== Side panel =====
function renderPanel() {{
  if (selectedIdx < 0) {{
    panelEmpty.style.display = '';
    panelSelected.style.display = 'none';
    return;
  }}
  panelEmpty.style.display = 'none';
  panelSelected.style.display = '';
  const nd = nodes[selectedIdx];
  const color = nodeColor(nd);

  let html = '';
  html += '<div>';
  html += '<div class="node-title">' + escHtml(nd.label);
  html += '<span class="node-role ' + nd.role + '">' + nd.role + '</span>';
  html += '</div>';
  if (nd.description) {{
    html += '<p style="font-size:12px;color:#98a1b3;margin-top:4px;">' + escHtml(nd.description) + '</p>';
  }}
  html += '</div>';

  // Meta
  html += '<div class="panel-section">';
  html += '<h3>Metadata</h3>';
  html += '<div class="meta-row"><span class="l">Cardinality</span><span class="v">' + (nd.cardinality || '—') + '</span></div>';
  html += '<div class="meta-row"><span class="l">Parents</span><span class="v">' + ((nd.parents && nd.parents.length) || 0) + '</span></div>';
  html += '<div class="meta-row"><span class="l">Children</span><span class="v">' + ((nd.children && nd.children.length) || 0) + '</span></div>';
  if (nd.importance != null) {{
    html += '<div class="meta-row"><span class="l">Importance</span><span class="v">' + nd.importance.toFixed(4) + '</span></div>';
  }}
  html += '</div>';

  // Marginal
  if (nd.marginal && nd.marginal.length) {{
    html += '<div class="panel-section"><h3>Marginal</h3>';
    html += '<div class="marg-bar-big">';
    for (let k = 0; k < nd.marginal.length; k++) {{
      const pct = (nd.marginal[k] * 100);
      html += '<div class="seg" style="width:' + pct.toFixed(2) + '%;background:' + PALETTE[k % PALETTE.length] + '"></div>';
    }}
    html += '</div>';
    html += '<div class="marg-legend">';
    for (let k = 0; k < nd.marginal.length; k++) {{
      const lbl = (nd.states && nd.states[k]) || ('s' + k);
      html += '<span class="item"><span class="swatch" style="background:' + PALETTE[k % PALETTE.length] + '"></span>' + escHtml(lbl) + ': ' + (nd.marginal[k]*100).toFixed(1) + '%</span>';
    }}
    html += '</div>';
    html += '</div>';
  }}

  // CPT
  html += renderCPT(nd);

  // Parents / children chips
  if (nd.parents && nd.parents.length) {{
    html += '<div class="panel-section"><h3>Parents</h3><div>';
    for (const pid of nd.parents) {{
      const p = nodes.find(function(x) {{ return x.id === pid; }});
      if (p) html += '<span class="chip" data-go="' + p.index + '">' + escHtml(p.label) + '</span>';
    }}
    html += '</div></div>';
  }}
  if (nd.children && nd.children.length) {{
    html += '<div class="panel-section"><h3>Children</h3><div>';
    for (const cid of nd.children) {{
      const c = nodes.find(function(x) {{ return x.id === cid; }});
      if (c) html += '<span class="chip" data-go="' + c.index + '">' + escHtml(c.label) + '</span>';
    }}
    html += '</div></div>';
  }}

  // Markov blanket
  const mb = focusMB;
  if (mb.length) {{
    html += '<div class="panel-section"><h3>Markov Blanket</h3><div>';
    for (const mi of mb) {{
      html += '<span class="chip" data-go="' + mi + '">' + escHtml(nodes[mi].label) + '</span>';
    }}
    html += '</div></div>';
  }}

  html += '<div><button class="btn" onclick="clearSel()" style="width:100%">Clear selection</button></div>';
  panelSelected.innerHTML = html;

  // Wire up chips
  panelSelected.querySelectorAll('[data-go]').forEach(function(el) {{
    el.addEventListener('click', function() {{
      selectNode(parseInt(el.getAttribute('data-go'), 10));
    }});
  }});
}}

function renderCPT(nd) {{
  if (!nd.cpt || (Array.isArray(nd.cpt) && nd.cpt.length === 0)) return '';
  const axes = nd.cptAxes || [];
  const axisLabels = axes.map(function(a) {{
    const p = nodes.find(function(x) {{ return x.id === a; }});
    return p ? p.label : a;
  }});
  const axisStates = axes.map(function(a) {{
    const p = nodes.find(function(x) {{ return x.id === a; }});
    return p && p.states && p.states.length ? p.states : null;
  }});

  // Flatten the CPT to a (parent-config × node-state) matrix.
  const flat = flattenCPT(nd.cpt);
  const nodeStates = nd.states && nd.states.length ? nd.states : [];
  const nStates = flat.length ? flat[0].length : nd.cardinality;

  // Build parent configuration labels.
  const configs = [];
  if (axes.length === 0) {{
    configs.push('—');
  }} else {{
    const parentCards = [];
    for (let ai = 0; ai < axes.length; ai++) {{
      const p = nodes.find(function(x) {{ return x.id === axes[ai]; }});
      parentCards.push((p && p.cardinality) || (axisStates[ai] ? axisStates[ai].length : 1));
    }}
    const total = parentCards.reduce(function(a, b) {{ return a * b; }}, 1);
    for (let idx = 0; idx < total; idx++) {{
      const parts = [];
      let rem = idx;
      for (let ai = 0; ai < axes.length; ai++) {{
        const step = parentCards.slice(ai + 1).reduce(function(a, b) {{ return a * b; }}, 1);
        const coord = Math.floor(rem / step);
        rem = rem % step;
        const lbl = axisStates[ai] ? axisStates[ai][coord] : ('s' + coord);
        parts.push(axisLabels[ai] + '=' + lbl);
      }}
      configs.push(parts.join(', '));
    }}
  }}

  let html = '<div class="panel-section"><h3>CPT · P(' + escHtml(nd.label) +
    (axes.length ? ' | ' + axes.map(function(a, i) {{ return escHtml(axisLabels[i]); }}).join(', ') : '') +
    ')</h3>';
  html += '<div style="overflow-x:auto"><table class="cpt-table"><thead><tr>';
  html += '<th>Parents</th>';
  for (let s = 0; s < nStates; s++) {{
    const lbl = nodeStates[s] || ('s' + s);
    html += '<th>' + escHtml(lbl) + '</th>';
  }}
  html += '</tr></thead><tbody>';
  for (let ci = 0; ci < configs.length; ci++) {{
    html += '<tr><td>' + escHtml(configs[ci]) + '</td>';
    for (let s = 0; s < nStates; s++) {{
      const v = (flat[ci] && flat[ci][s] != null) ? flat[ci][s] : 0;
      const pct = Math.max(0, Math.min(1, v)) * 100;
      html += '<td class="cpt-cell"><span class="fill" style="width:' + pct.toFixed(1) + '%"></span><span>' + v.toFixed(3) + '</span></td>';
    }}
    html += '</tr>';
  }}
  html += '</tbody></table></div></div>';
  return html;
}}

function flattenCPT(cpt) {{
  // CPT shape: (node_card, *parent_cards). We want (parent_config, node_card).
  // Approach: rebuild as rectangular array.
  if (!Array.isArray(cpt)) return [[cpt]];
  const shape = [];
  let cur = cpt;
  while (Array.isArray(cur)) {{
    shape.push(cur.length);
    cur = cur[0];
  }}
  if (shape.length === 1) {{
    // No parents → single row.
    return [cpt.slice()];
  }}
  const nodeCard = shape[0];
  const parentShape = shape.slice(1);
  const parentTotal = parentShape.reduce(function(a, b) {{ return a * b; }}, 1);

  // Flat read in (parent-config-major, state-minor) order.
  // For each parent config, collect cpt[state][p0][p1]...[pN].
  function readAt(node, coords) {{
    let v = node;
    for (const c of coords) v = v[c];
    return v;
  }}
  const out = [];
  for (let pc = 0; pc < parentTotal; pc++) {{
    const coords = [];
    let rem = pc;
    for (let ai = 0; ai < parentShape.length; ai++) {{
      const step = parentShape.slice(ai + 1).reduce(function(a, b) {{ return a * b; }}, 1);
      coords.push(Math.floor(rem / step));
      rem = rem % step;
    }}
    const row = [];
    for (let s = 0; s < nodeCard; s++) {{
      row.push(readAt(cpt[s], coords));
    }}
    out.push(row);
  }}
  return out;
}}

// ===== Tooltip =====
function showTooltip(ev, nd) {{
  let html = '<h3>' + escHtml(nd.label) + '</h3>';
  html += '<div class="r"><span class="l">Role</span><span class="v">' + nd.role + '</span></div>';
  html += '<div class="r"><span class="l">Cardinality</span><span class="v">' + (nd.cardinality || '—') + '</span></div>';
  html += '<div class="r"><span class="l">Parents</span><span class="v">' + ((nd.parents || []).length) + '</span></div>';
  html += '<div class="r"><span class="l">Children</span><span class="v">' + ((nd.children || []).length) + '</span></div>';
  if (nd.importance != null) {{
    html += '<div class="r"><span class="l">Importance</span><span class="v">' + nd.importance.toFixed(4) + '</span></div>';
  }}
  if (nd.marginal && nd.marginal.length) {{
    html += '<div style="margin-top:8px"><div class="marg-bar-big" style="height:10px">';
    for (let k = 0; k < nd.marginal.length; k++) {{
      html += '<div style="width:' + (nd.marginal[k]*100).toFixed(2) + '%;background:' + PALETTE[k % PALETTE.length] + '"></div>';
    }}
    html += '</div></div>';
  }}
  tooltip.innerHTML = html;
  tooltip.classList.add('visible');
  const rect = wrap.getBoundingClientRect();
  let left = ev.clientX - rect.left + 18;
  let top = ev.clientY - rect.top - 10;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  if (left + tw > rect.width - 16) left = ev.clientX - rect.left - tw - 16;
  if (top + th > rect.height - 16) top = rect.height - th - 16;
  if (top < 8) top = 8;
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
}}
function hideTooltip() {{ tooltip.classList.remove('visible'); }}

// ===== Zoom & Pan =====
function updateTransform() {{
  const g = document.getElementById('root-g');
  if (g) g.setAttribute('transform', 'translate(' + transform.x + ',' + transform.y + ') scale(' + transform.k + ')');
  if (zoomLabel) zoomLabel.textContent = Math.round(transform.k * 100) + '%';
}}

wrap.addEventListener('wheel', function(e) {{
  e.preventDefault();
  const rect = wrap.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newK = Math.max(0.1, Math.min(5, transform.k * delta));
  transform.x = mx - (mx - transform.x) * (newK / transform.k);
  transform.y = my - (my - transform.y) * (newK / transform.k);
  transform.k = newK;
  updateTransform();
}}, {{ passive: false }});

wrap.addEventListener('mousedown', function(e) {{
  if (dragNodeIdx >= 0) return; // node drag takes priority
  dragging = true;
  dragStart.x = e.clientX - transform.x;
  dragStart.y = e.clientY - transform.y;
}});

window.addEventListener('mousemove', function(e) {{
  if (dragNodeIdx >= 0) {{
    // Convert screen → canvas coords accounting for transform.
    const rect = wrap.getBoundingClientRect();
    const mx = (e.clientX - rect.left - transform.x) / transform.k;
    const my = (e.clientY - rect.top - transform.y) / transform.k;
    nodes[dragNodeIdx].x = mx;
    nodes[dragNodeIdx].y = my;
    updatePositions();
    return;
  }}
  if (!dragging) return;
  transform.x = e.clientX - dragStart.x;
  transform.y = e.clientY - dragStart.y;
  updateTransform();
}});

window.addEventListener('mouseup', function() {{
  dragging = false;
  dragNodeIdx = -1;
}});

wrap.addEventListener('click', function(e) {{
  // Click on empty canvas → clear selection.
  if (e.target === svg || e.target === wrap) clearSelection();
}});

function fitToView() {{
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const n of nodes) {{
    const r = nodeRadius(n);
    minX = Math.min(minX, n.x - r); maxX = Math.max(maxX, n.x + r);
    minY = Math.min(minY, n.y - r); maxY = Math.max(maxY, n.y + r);
  }}
  if (!isFinite(minX)) return;
  const rect = wrap.getBoundingClientRect();
  const pad = 60;
  const w = (maxX - minX) + pad * 2;
  const h = (maxY - minY) + pad * 2;
  const k = Math.min(rect.width / w, rect.height / h, 1.4);
  transform.k = k;
  transform.x = (rect.width - (maxX + minX) * k) / 2;
  transform.y = (rect.height - (maxY + minY) * k) / 2;
  updateTransform();
}}

// ===== Minimap =====
function renderMinimap() {{
  clear(minimap);
  if (!nodes.length) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const n of nodes) {{
    minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
    minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
  }}
  const w = (maxX - minX) || 1, h = (maxY - minY) || 1;
  const sx = 160 / w, sy = 100 / h;
  const s = Math.min(sx, sy);
  const ox = (180 - w * s) / 2 - minX * s;
  const oy = (120 - h * s) / 2 - minY * s;
  for (const e of edges) {{
    const a = nodes[e.source], b = nodes[e.target];
    minimap.appendChild(ns('line', {{
      x1: a.x * s + ox, y1: a.y * s + oy,
      x2: b.x * s + ox, y2: b.y * s + oy,
      stroke: 'rgba(255,255,255,0.15)', 'stroke-width': 0.6,
    }}));
  }}
  for (const n of nodes) {{
    minimap.appendChild(ns('circle', {{
      cx: n.x * s + ox, cy: n.y * s + oy,
      r: n.role === 'class' ? 3.5 : 2.2,
      fill: nodeColor(n), opacity: 0.85,
    }}));
  }}
}}

// ===== Controls =====
function cycleColor() {{
  const idx = COLOR_MODES.indexOf(colorMode);
  colorMode = COLOR_MODES[(idx + 1) % COLOR_MODES.length];
  document.getElementById('btn-color').textContent = 'Color: ' + colorMode;
  render();
}}
function cycleLayout() {{
  const idx = LAYOUTS.indexOf(layoutMode);
  layoutMode = LAYOUTS[(idx + 1) % LAYOUTS.length];
  document.getElementById('btn-layout').textContent = 'Layout: ' + layoutMode;
  for (const n of nodes) n.fixed = false;
  relayout();
  render();
  fitToView();
}}
function focusClass() {{
  if (classIndex >= 0) selectNode(classIndex);
}}
function resetView() {{
  clearSelection();
  for (const n of nodes) n.fixed = false;
  relayout();
  render();
  fitToView();
}}
window.cycleColor = cycleColor;
window.cycleLayout = cycleLayout;
window.focusClass = focusClass;
window.resetView = resetView;
window.clearSel = clearSelection;

// Search
searchInput.addEventListener('input', function(e) {{
  const q = e.target.value.trim().toLowerCase();
  if (!q) {{
    for (const n of nodes) n._group.classList.remove('dim');
    updateFocus();
    return;
  }}
  for (const n of nodes) {{
    const match = n.label.toLowerCase().includes(q) || n.id.toLowerCase().includes(q);
    n._group.classList.toggle('dim', !match);
  }}
}});

// Keyboard
document.addEventListener('keydown', function(e) {{
  if (e.target === searchInput) {{
    if (e.key === 'Escape') searchInput.blur();
    return;
  }}
  if (e.key === '/' || e.key === 's') {{
    e.preventDefault();
    searchInput.focus();
  }} else if (e.key === '0') {{ fitToView(); }}
  else if (e.key === '+' || e.key === '=') {{
    transform.k = Math.min(5, transform.k * 1.2); updateTransform();
  }} else if (e.key === '-') {{
    transform.k = Math.max(0.1, transform.k / 1.2); updateTransform();
  }} else if (e.key === 'l') {{ cycleLayout(); }}
  else if (e.key === 'c') {{ cycleColor(); }}
  else if (e.key === 'm') {{ focusClass(); }}
  else if (e.key === 'Escape') {{ clearSelection(); }}
}});

// Stats
function updateStats() {{
  document.getElementById('sum-nodes').textContent = nodes.length;
  document.getElementById('sum-edges').textContent = edges.length;
  document.getElementById('sum-layers').textContent = PAYLOAD.layers.length;
  document.getElementById('sum-mb').textContent =
    (PAYLOAD.classMarkovBlanket && PAYLOAD.classMarkovBlanket.length) || 0;
  const stats = document.getElementById('stats-text');
  if (stats) stats.textContent = nodes.length + ' nodes · ' + edges.length + ' edges · ' + PAYLOAD.layers.length + ' layers';
}}

// ===== Init =====
relayout();
render();
requestAnimationFrame(fitToView);
window.addEventListener('resize', function() {{ relayout(); render(); fitToView(); }});

}})();
</script>
</body>
</html>
"""
