"""Render :meth:`get_structure` output as an HTML section for reports.

Works for every estimator inheriting :class:`endgame.core.glassbox.GlassboxMixin`.
Returns empty string for opaque models so callers can unconditionally invoke
this helper from :class:`ClassificationReport` / :class:`RegressionReport`.
"""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Any

_COEF_EPS = 1e-10


def render_structure_section(
    model: Any,
    *,
    feature_names: list[str] | None = None,
    tree_link_href: str | None = None,
    bn_link_href: str | None = None,
    structure_export_href: str | None = None,
) -> str:
    """Build the interpretability section HTML for a fitted model.

    Parameters
    ----------
    model : estimator
        Any fitted estimator. Models without ``get_structure()`` produce ``""``.
    feature_names : list of str, optional
        Override the feature names used throughout the rendered structure.
        When provided, passed to ``model.get_structure(feature_names=...)``
        so Bayesian-network edge tables, linear coefficient tables, etc.
        display human-readable labels instead of ``x0, x1, ...`` placeholders.
    tree_link_href : str, optional
        Relative URL to a sidecar tree visualization. When provided and the
        model's ``structure_type`` is tree-shaped, a "Visualize tree →" link
        is prepended to the section body.
    bn_link_href : str, optional
        Relative URL to a sidecar Bayesian network visualization. When
        provided and the model's ``structure_type`` is ``bayesian_network``,
        an "Open interactive Bayesian network →" link is prepended.
    structure_export_href : str, optional
        Relative URL to a sidecar JSON export of the full structure dict.
        When provided, a "Download JSON" link is shown in the section header.
    """
    if not hasattr(model, "get_structure"):
        return ""
    try:
        if feature_names is not None:
            struct = model.get_structure(feature_names=list(feature_names))
        else:
            struct = model.get_structure()
    except Exception:
        return ""
    if not isinstance(struct, dict):
        return ""

    stype = struct.get("structure_type", "generic")
    renderer = _RENDERERS.get(stype, _render_generic)
    try:
        body = renderer(struct)
    except Exception as exc:
        body = f'<p class="rules-empty">Structure extraction failed: {_esc(str(exc))}</p>'

    links: list[str] = []
    if tree_link_href and stype in {"tree", "tree_ensemble"}:
        links.append(
            f'<a class="tree-link" href="{_esc(tree_link_href)}" target="_blank" rel="noopener">'
            "Open interactive tree visualization →"
            "</a>"
        )
    if bn_link_href and stype == "bayesian_network":
        links.append(
            f'<a class="tree-link" href="{_esc(bn_link_href)}" target="_blank" rel="noopener">'
            "Open interactive Bayesian network →"
            "</a>"
        )
    if structure_export_href:
        links.append(
            f'<a class="tree-link tree-link-secondary" href="{_esc(structure_export_href)}" '
            f'download>Download structure (JSON) ↓</a>'
        )
    link_html = (
        f'<p class="tree-link-wrapper">{" ".join(links)}</p>' if links else ""
    )

    heading = _esc(f"Learned Structure — {struct.get('model_type', 'model')}")
    subtitle = f" <span class=\"struct-type\">{_esc(stype)}</span>"
    return (
        '<div class="interp-section struct-section">'
        f'<h2>{heading}{subtitle}</h2>'
        f"{link_html}{body}"
        "</div>"
    )


def try_save_tree_sidecar(
    model: Any,
    report_path: Path,
    *,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
) -> str | None:
    """Save an interactive tree visualization next to the report.

    Returns the sidecar's filename (relative to the report) on success, or
    ``None`` if the model isn't tree-renderable.
    """
    try:
        from endgame.visualization.tree_visualizer import TreeVisualizer
    except Exception:
        return None
    try:
        viz = TreeVisualizer(
            model,
            feature_names=feature_names,
            class_names=class_names,
        )
    except Exception:
        return None
    report_path = Path(report_path)
    sidecar = report_path.with_name(f"{report_path.stem}_tree.html")
    try:
        viz.save(sidecar)
    except Exception:
        return None
    return sidecar.name


def try_save_bn_sidecar(
    model: Any,
    report_path: Path,
    *,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
) -> str | None:
    """Save an interactive Bayesian network visualization next to the report.

    Returns the sidecar's filename (relative to the report) on success, or
    ``None`` if the model isn't a Bayesian network (no ``structure_``,
    ``edges_``, ``dag_``, or ``parents_``).
    """
    try:
        from endgame.visualization.bayesian_network_visualizer import (
            BayesianNetworkVisualizer,
        )
    except Exception:
        return None
    try:
        viz = BayesianNetworkVisualizer(
            model,
            feature_names=feature_names,
            class_names=class_names,
        )
    except Exception:
        return None
    report_path = Path(report_path)
    sidecar = report_path.with_name(f"{report_path.stem}_bn.html")
    try:
        viz.save(sidecar)
    except Exception:
        return None
    return sidecar.name


def save_structure_json(
    model: Any,
    path: str | Path,
    *,
    feature_names: list[str] | None = None,
    indent: int = 2,
) -> Path | None:
    """Write ``model.get_structure()`` to ``path`` as JSON.

    Returns the resolved path on success, ``None`` if the model has no
    ``get_structure()`` method. Numpy scalars/arrays are coerced via
    ``default=str`` so exports stay pure-JSON even for exotic payloads.
    """
    if not hasattr(model, "get_structure"):
        return None
    import json as _json

    try:
        if feature_names is not None:
            struct = model.get_structure(feature_names=feature_names)
        else:
            struct = model.get_structure()
    except Exception:
        return None
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    try:
        path.write_text(
            _json.dumps(struct, indent=indent, default=_json_default),
            encoding="utf-8",
        )
    except Exception:
        return None
    return path.resolve()


def try_save_structure_sidecar(
    model: Any,
    report_path: Path,
    *,
    feature_names: list[str] | None = None,
) -> str | None:
    """Save a JSON structure export next to the report, returning its filename."""
    if not hasattr(model, "get_structure"):
        return None
    report_path = Path(report_path)
    sidecar = report_path.with_name(f"{report_path.stem}_structure.json")
    resolved = save_structure_json(model, sidecar, feature_names=feature_names)
    return sidecar.name if resolved is not None else None


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy types, Path, and other non-serialisables."""
    try:
        import numpy as _np

        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


# --------------------------------------------------------------------------
# Renderers per structure_type
# --------------------------------------------------------------------------


def _esc(val: Any) -> str:
    return _html.escape(str(val))


def _fmt(val: Any, precision: int = 4) -> str:
    if isinstance(val, (int,)):
        return str(val)
    if isinstance(val, float):
        return f"{val:.{precision}g}"
    return _esc(val)


def _render_tree(struct: dict[str, Any]) -> str:
    tree = struct.get("tree") or {}
    n_leaves = tree.get("n_leaves")
    n_nodes = tree.get("n_nodes")
    max_depth = tree.get("max_depth")
    summary_bits: list[str] = []
    if max_depth is not None:
        summary_bits.append(f"depth {max_depth}")
    if n_leaves is not None:
        summary_bits.append(f"{n_leaves} leaves")
    if n_nodes is not None:
        summary_bits.append(f"{n_nodes} nodes")
    note = struct.get("note") or tree.get("note")
    parts: list[str] = []
    if summary_bits:
        parts.append(f"<p class=\"struct-stat\">{_esc(' · '.join(summary_bits))}</p>")
    if note:
        parts.append(f"<p class=\"struct-note\">{_esc(note)}</p>")
    imps = struct.get("feature_importances")
    if isinstance(imps, list) and imps:
        parts.append(_render_feature_importances(imps, struct.get("feature_names", [])))
    return "\n".join(parts) or '<p class="rules-empty">Empty tree.</p>'


def _render_tree_ensemble(struct: dict[str, Any]) -> str:
    n_trees = struct.get("n_trees") or len(struct.get("trees", []))
    parts = [f'<p class="struct-stat">{_esc(n_trees)} trees</p>']
    if struct.get("note"):
        parts.append(f'<p class="struct-note">{_esc(struct["note"])}</p>')
    imps = struct.get("feature_importances")
    if isinstance(imps, list) and imps:
        parts.append(_render_feature_importances(imps, struct.get("feature_names", [])))
    return "\n".join(parts)


def _render_rules(struct: dict[str, Any]) -> str:
    rules = struct.get("rules", [])
    if not rules:
        return '<p class="rules-empty">No rules extracted.</p>'
    lines: list[str] = []
    lines.append("<ol class=\"rules-list\">")
    for r in rules:
        if isinstance(r, str):
            lines.append(f"<li>{_esc(r)}</li>")
            continue
        if not isinstance(r, dict):
            continue
        # RuleFit style: rule + coefficient + support
        if "rule" in r:
            coef = r.get("coefficient", r.get("coef"))
            support = r.get("support")
            importance = r.get("importance")
            extras = []
            if coef is not None:
                extras.append(f"coef={float(coef):+.4f}")
            if support is not None:
                extras.append(f"support={float(support):.3f}")
            if importance is not None:
                extras.append(f"imp={float(importance):.4f}")
            suffix = f" <span class=\"rule-meta\">[{_esc(' · '.join(extras))}]</span>" if extras else ""
            lines.append(f"<li>{_esc(r['rule'])}{suffix}</li>")
            continue
        # CORELS style: antecedent + consequent
        if "antecedent" in r:
            cons = r.get("consequent_class", r.get("consequent"))
            pos = r.get("position")
            prefix = "IF" if pos == 0 else "ELSE IF"
            lines.append(
                f"<li>{_esc(prefix)} {_esc(r['antecedent'])} "
                f"THEN <b>{_esc(cons)}</b></li>"
            )
            continue
        # GOSDT style: conditions list + prediction
        if "conditions" in r and "prediction" in r:
            conds = " AND ".join(_esc(c) for c in r["conditions"]) or "TRUE"
            lines.append(f"<li>IF {conds} THEN <b>{_esc(r['prediction'])}</b></li>")
            continue
        # Cubist-ish: generic key=value dict
        lines.append(f"<li>{_esc(r)}</li>")
    lines.append("</ol>")
    # Append default / intercept if present
    if struct.get("default"):
        d = struct["default"]
        cons = d.get("consequent_class", d.get("consequent")) if isinstance(d, dict) else d
        lines.append(f'<p class="struct-note">Default: <b>{_esc(cons)}</b></p>')
    if struct.get("intercept") is not None:
        lines.append(f'<p class="struct-note">Intercept: {_fmt(struct["intercept"])}</p>')
    return "\n".join(lines)


def _render_fuzzy_rules(struct: dict[str, Any]) -> str:
    rules = struct.get("rules", [])
    if not rules:
        return '<p class="rules-empty">No fuzzy rules extracted.</p>'
    lines = ["<ol class=\"rules-list\">"]
    for r in rules:
        conds = r.get("conditions", [])
        ant = " AND ".join(c.get("text", _esc(c)) for c in conds) if conds else "TRUE"
        cons = r.get("consequent_class")
        weight = r.get("weight")
        support = r.get("support")
        meta = []
        if weight is not None:
            meta.append(f"w={float(weight):.3f}")
        if support is not None:
            meta.append(f"n={int(support)}")
        suffix = f" <span class=\"rule-meta\">[{_esc(' · '.join(meta))}]</span>" if meta else ""
        lines.append(f"<li>IF {_esc(ant)} THEN <b>{_esc(cons)}</b>{suffix}</li>")
    lines.append("</ol>")
    return "\n".join(lines)


def _render_linear(struct: dict[str, Any]) -> str:
    intercept = struct.get("intercept")
    link = struct.get("link")
    coefs = struct.get("coefficients")
    parts: list[str] = []
    header_bits = []
    if link:
        header_bits.append(f"link = <b>{_esc(link)}</b>")
    if intercept is not None:
        if isinstance(intercept, list):
            header_bits.append(f"intercept = {_esc(intercept)}")
        else:
            header_bits.append(f"intercept = {_fmt(intercept)}")
    variant = struct.get("variant")
    if variant:
        header_bits.append(f"variant = <b>{_esc(variant)}</b>")
    if header_bits:
        parts.append(f'<p class="struct-stat">{" · ".join(header_bits)}</p>')

    def _coef_table(mapping: dict[str, float], *, title: str | None = None) -> str:
        if not mapping:
            return ""
        pairs = sorted(
            ((k, float(v)) for k, v in mapping.items() if abs(float(v)) > _COEF_EPS),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        rows = "\n".join(
            f'<tr><td>{_esc(name)}</td><td class="num">{"+" if v >= 0 else ""}{v:.4f}</td></tr>'
            for name, v in pairs
        )
        head = f'<h3 class="struct-subhead">{_esc(title)}</h3>' if title else ""
        return f'{head}<table class="struct-table"><thead><tr><th>Feature</th><th>Coefficient</th></tr></thead><tbody>{rows}</tbody></table>'

    if isinstance(coefs, dict):
        parts.append(_coef_table(coefs))
    elif isinstance(coefs, list):
        for i, row in enumerate(coefs):
            if isinstance(row, dict):
                parts.append(_coef_table(row, title=f"Class {i}"))
    return "\n".join(p for p in parts if p) or '<p class="rules-empty">No coefficients.</p>'


def _render_additive(struct: dict[str, Any]) -> str:
    terms = struct.get("terms", [])
    if not terms:
        return '<p class="rules-empty">No terms.</p>'
    link = struct.get("link")
    intercept = struct.get("intercept")
    parts: list[str] = []
    header_bits = []
    if link:
        header_bits.append(f"link = <b>{_esc(link)}</b>")
    if intercept is not None:
        if isinstance(intercept, list):
            header_bits.append(f"intercept = {_esc(intercept)}")
        else:
            header_bits.append(f"intercept = {_fmt(intercept)}")
    if header_bits:
        parts.append(f'<p class="struct-stat">{" · ".join(header_bits)}</p>')

    def _sortkey(t: dict) -> float:
        imp = t.get("importance") or t.get("coefficient") or 0
        try:
            return abs(float(imp))
        except (TypeError, ValueError):
            return 0.0

    rows = []
    for term in sorted(terms, key=_sortkey, reverse=True):
        name = term.get("name", "")
        kind = term.get("type", "")
        imp = term.get("importance")
        coef = term.get("coefficient")
        val_cell = ""
        if imp is not None:
            val_cell = f"{float(imp):.4f}"
        elif coef is not None:
            val_cell = f"{float(coef):+.4f}"
        rows.append(
            f"<tr><td>{_esc(name)}</td><td>{_esc(kind)}</td>"
            f'<td class="num">{val_cell}</td></tr>'
        )
    parts.append(
        '<table class="struct-table">'
        "<thead><tr><th>Term</th><th>Type</th><th>Importance</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    return "\n".join(parts)


def _render_scorecard(struct: dict[str, Any]) -> str:
    card = struct.get("scorecard", [])
    if not card:
        return '<p class="rules-empty">Empty scorecard.</p>'
    rows = "\n".join(
        f'<tr><td>{_esc(item.get("feature", ""))}</td>'
        f'<td class="num">{int(item.get("points", 0)):+d}</td></tr>'
        for item in card
    )
    intercept = struct.get("intercept")
    inter_row = (
        f'<tr><td><i>Intercept</i></td><td class="num">{int(intercept):+d}</td></tr>'
        if intercept is not None else ""
    )
    extras = []
    if struct.get("variant"):
        extras.append(f"variant = <b>{_esc(struct['variant'])}</b>")
    if struct.get("max_coef") is not None:
        extras.append(f"max_coef = {_esc(struct['max_coef'])}")
    header = (
        f'<p class="struct-stat">{" · ".join(extras)}</p>' if extras else ""
    )
    return (
        f"{header}"
        '<table class="struct-table">'
        "<thead><tr><th>Condition</th><th>Points</th></tr></thead>"
        f"<tbody>{inter_row}{rows}</tbody></table>"
    )


def _render_bayesian_network(struct: dict[str, Any]) -> str:
    nodes = struct.get("nodes", [])
    edges = struct.get("edges", [])
    mb = struct.get("markov_blanket", [])
    feature_names = struct.get("feature_names", []) or []

    def _label(ident: Any) -> str:
        """Translate a BN node identifier to a human-readable label.

        Feature indices (ints or int-like strings) resolve against
        ``feature_names``. The target placeholder ``"Y"`` becomes ``"Class"``.
        Anything else is returned verbatim.
        """
        s = str(ident)
        if s == "Y" or s.lower() in {"class", "target"}:
            return "Class"
        try:
            idx = int(s)
        except (TypeError, ValueError):
            return s
        if 0 <= idx < len(feature_names):
            return str(feature_names[idx])
        return s

    parts = [
        f'<p class="struct-stat">{len(nodes)} nodes · {len(edges)} edges</p>',
    ]
    if mb:
        mb_labels = [_label(m) for m in mb]
        parts.append(
            f'<p class="struct-note">Markov blanket of target: '
            f'{_esc(", ".join(mb_labels))}</p>'
        )
    if edges:
        rows = "\n".join(
            f'<tr><td>{_esc(_label(u))}</td><td>→</td><td>{_esc(_label(v))}</td></tr>'
            for u, v in edges
        )
        parts.append(
            '<h3 class="struct-subhead">Edges</h3>'
            '<table class="struct-table"><tbody>'
            f"{rows}"
            "</tbody></table>"
        )
    if struct.get("note"):
        parts.append(f'<p class="struct-note">{_esc(struct["note"])}</p>')
    return "\n".join(parts)


def _render_symbolic(struct: dict[str, Any]) -> str:
    eq = struct.get("equation")
    parts: list[str] = []
    if eq:
        parts.append(f'<pre class="struct-equation">{_esc(eq)}</pre>')
    if struct.get("best_loss") is not None:
        parts.append(
            f'<p class="struct-stat">loss = {float(struct["best_loss"]):.4g} · '
            f"complexity = {_esc(struct.get('best_complexity', '?'))}</p>"
        )
    pf = struct.get("pareto_frontier") or []
    if pf:
        rows = "\n".join(
            f"<tr><td>{_esc(e.get('complexity'))}</td>"
            f"<td class=\"num\">{float(e.get('loss', 0)):.4g}</td>"
            f"<td><code>{_esc(e.get('equation'))}</code></td></tr>"
            for e in pf
        )
        parts.append(
            '<h3 class="struct-subhead">Pareto frontier</h3>'
            '<table class="struct-table">'
            "<thead><tr><th>Complexity</th><th>Loss</th><th>Equation</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    if struct.get("per_class"):
        parts.append('<h3 class="struct-subhead">Per-class equations</h3>')
        for entry in struct["per_class"]:
            parts.append(
                f'<p class="struct-stat">Class <b>{_esc(entry.get("class"))}</b>:'
                f' <code>{_esc(entry.get("equation"))}</code></p>'
            )
    return "\n".join(parts) or '<p class="rules-empty">No symbolic expression.</p>'


def _render_boxes(struct: dict[str, Any]) -> str:
    boxes = struct.get("boxes") or []
    per_class = struct.get("per_class") or []
    parts: list[str] = []
    if boxes:
        parts.append(_boxes_html(boxes, heading=None))
    for entry in per_class:
        heading = f"Class {_esc(entry.get('class'))}"
        parts.append(_boxes_html(entry.get("boxes", []), heading=heading))
    return "\n".join(parts) or '<p class="rules-empty">No boxes.</p>'


def _boxes_html(boxes: list[dict], *, heading: str | None) -> str:
    if not boxes:
        return ""
    rows = []
    for b in boxes:
        rules_str = " AND ".join(b.get("rules", [])) or "∅"
        coverage = b.get("coverage")
        density = b.get("density")
        support = b.get("support")
        meta = []
        if coverage is not None:
            meta.append(f"coverage={float(coverage):.3f}")
        if density is not None:
            meta.append(f"density={float(density):.4f}")
        if support is not None:
            meta.append(f"support={int(support)}")
        rows.append(
            f"<li><code>{_esc(rules_str)}</code>"
            f' <span class="rule-meta">[{_esc(" · ".join(meta))}]</span></li>'
        )
    head = f'<h3 class="struct-subhead">{heading}</h3>' if heading else ""
    return f'{head}<ol class="rules-list">{"".join(rows)}</ol>'


def _render_generic(struct: dict[str, Any]) -> str:
    """Fallback for unknown / exotic structure_type values."""
    import json as _json

    snippet = {
        k: v
        for k, v in struct.items()
        if k not in {"model_type", "structure_type", "feature_names", "n_features"}
    }
    try:
        text = _json.dumps(snippet, indent=2, default=str)[:4000]
    except Exception:
        text = repr(snippet)[:4000]
    return f'<pre class="struct-equation">{_esc(text)}</pre>'


def _render_feature_importances(values: list, feature_names: list[str]) -> str:
    try:
        pairs = list(zip(feature_names, [float(v) for v in values]))
    except (TypeError, ValueError):
        return ""
    pairs = [(n, v) for n, v in pairs if abs(v) > _COEF_EPS]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    if not pairs:
        return ""
    rows = "\n".join(
        f'<tr><td>{_esc(n)}</td><td class="num">{v:.4f}</td></tr>' for n, v in pairs
    )
    return (
        '<h3 class="struct-subhead">Feature importances</h3>'
        '<table class="struct-table">'
        "<thead><tr><th>Feature</th><th>Importance</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


_RENDERERS = {
    "tree": _render_tree,
    "tree_ensemble": _render_tree_ensemble,
    "rules": _render_rules,
    "fuzzy_rules": _render_fuzzy_rules,
    "linear": _render_linear,
    "additive": _render_additive,
    "scorecard": _render_scorecard,
    "bayesian_network": _render_bayesian_network,
    "symbolic": _render_symbolic,
    "boxes": _render_boxes,
}


STRUCTURE_SECTION_CSS = """
.struct-section h2 .struct-type {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  background: var(--bg-elevated, rgba(255,255,255,0.05));
  border: 1px solid var(--border);
  padding: 2px 8px;
  border-radius: 999px;
  margin-left: 8px;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  vertical-align: middle;
}

.tree-link-wrapper { margin: 0 0 14px 0; display: flex; gap: 10px; flex-wrap: wrap; }

.tree-link {
  display: inline-block;
  padding: 7px 14px;
  background: linear-gradient(135deg, #4e79a7, #6b96cc);
  color: #fff !important;
  text-decoration: none;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.tree-link:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(78,121,167,0.35);
}
.tree-link.tree-link-secondary {
  background: transparent;
  color: var(--text-primary) !important;
  border: 1px solid var(--border);
}
.tree-link.tree-link-secondary:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.struct-stat {
  font-size: 12px;
  color: var(--text-secondary);
  margin: 6px 0;
}

.struct-note {
  font-size: 11px;
  color: var(--text-muted);
  margin: 6px 0;
  font-style: italic;
}

.struct-subhead {
  font-size: 13px;
  font-weight: 600;
  margin: 16px 0 8px 0;
  color: var(--text-primary);
}

.struct-table {
  width: 100%;
  border-collapse: collapse;
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  font-size: 12px;
  margin-top: 6px;
}
.struct-table th, .struct-table td {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
}
.struct-table th {
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  font-size: 10px;
}
.struct-table td.num { text-align: right; }

.struct-equation {
  background: var(--bg-elevated, rgba(255,255,255,0.04));
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  font-size: 12px;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
}

.rule-meta {
  font-size: 11px;
  color: var(--text-muted);
  font-family: "SF Mono", "Fira Code", monospace;
}

.rules-empty {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
  margin: 8px 0;
}
"""


__all__ = [
    "render_structure_section",
    "try_save_tree_sidecar",
    "try_save_structure_sidecar",
    "save_structure_json",
    "STRUCTURE_SECTION_CSS",
]
