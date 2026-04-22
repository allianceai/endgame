"""Interpretable model display engine.

Provides rich, formatted display of learned model structures: rules, trees,
equations, scorecards, shape functions, coefficients, and feature importances.

Works with any sklearn-compatible estimator — detects model type by probing
for known attributes (``get_rules``, ``get_scorecard``, ``coef_``, etc.)
and renders the appropriate representation.

Example
-------
>>> from endgame.automl.display import display_model, display_models
>>> display_model("EBM", fitted_ebm, feature_names)
>>> display_models({"EBM": ebm, "RuleFit": rulefit}, feature_names)
"""

from __future__ import annotations

import re
import textwrap

import numpy as np


def replace_feature_indices(text: str, feature_names: list[str]) -> str:
    """Replace generic feature references (x0, X0, feature_0000) with real names.

    Processes longest indices first to avoid partial replacement (x13 before x1).
    """
    for i in sorted(range(len(feature_names)), reverse=True):
        name = feature_names[i]
        text = re.sub(rf"\bx{i}\b", name, text)
        text = re.sub(rf"\bX{i}\b", name, text)
        text = re.sub(rf"\bfeature_{i:04d}\b", name, text)
        text = re.sub(rf"\bfeature_{i}\b", name, text)
    return text


def _indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(str(text), prefix)


def _bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    return "█" * int(value / max_value * width)


def _show_importances(
    model,
    feature_names: list[str],
    title: str,
    top_n: int = 10,
) -> list[str]:
    """Format feature importances as lines of text."""
    lines: list[str] = []
    try:
        fi = np.asarray(model.feature_importances_).ravel()
        if len(fi) != len(feature_names):
            return lines
        lines.append(f"\n  {title}:")
        pairs = sorted(zip(feature_names, fi), key=lambda t: abs(t[1]), reverse=True)
        mx = max(fi) if max(fi) > 0 else 1
        for fn, imp in pairs[:top_n]:
            lines.append(f"    {fn:35s}  {imp:.4f}  {_bar(imp, mx)}")
    except Exception:
        pass
    return lines


def display_model(
    name: str,
    model,
    feature_names: list[str] | None = None,
    X_sample: np.ndarray | None = None,
    *,
    top_rules: int = 15,
    top_features: int = 10,
    print_output: bool = True,
) -> str:
    """Display the learned structure of a fitted interpretable model.

    Parameters
    ----------
    name : str
        Display name for the model (e.g. "EBM", "RuleFit").
    model : estimator
        A fitted sklearn-compatible estimator.
    feature_names : list of str, optional
        Feature names. If None, generic names are not replaced.
    X_sample : ndarray, optional
        Sample data for computing per-sample contributions.
    top_rules : int, default=15
        Maximum number of rules/terms to display.
    top_features : int, default=10
        Maximum number of features in importance displays.
    print_output : bool, default=True
        If True, print to stdout. Always returns the full text.

    Returns
    -------
    str
        The complete formatted display text.
    """
    lines: list[str] = []
    fn = feature_names or []

    def R(t: str) -> str:
        return replace_feature_indices(t, fn) if fn else t

    lines.append(f"\n  {'─' * 64}")
    lines.append(f"  LEARNED MODEL: {name}")
    lines.append(f"  {'─' * 64}")

    displayed = False

    # ── CORELS: rule list ────────────────────────────────────────────
    if hasattr(model, "rule_list_") and model.rule_list_:
        lines.append("\n  Rule List:")
        lines.append(_indent(R(model.rule_list_)))
        displayed = True
    if hasattr(model, "summary") and name == "CORELS":
        try:
            lines.append("\n  Summary:")
            lines.append(_indent(R(model.summary())))
        except Exception:
            pass
        displayed = True

    # ── GAM: summary ─────────────────────────────────────────────────
    if hasattr(model, "summary") and name == "GAM":
        try:
            lines.append("\n  GAM Summary:")
            lines.append(_indent(R(model.summary())))
        except Exception as e:
            lines.append(f"    (summary failed: {e})")
        displayed = True

    # ── EBM: term names + importances ────────────────────────────────
    if hasattr(model, "get_term_names"):
        try:
            term_names = model.get_term_names()
            importances = model.term_importances()
            lines.append(f"\n  EBM Terms (top {top_rules} by importance):")
            pairs = sorted(
                zip(term_names, importances),
                key=lambda t: abs(t[1]),
                reverse=True,
            )
            for tname, imp in pairs[:top_rules]:
                lines.append(f"    {R(tname):40s}  importance={imp:.4f}")
        except Exception as e:
            lines.append(f"    (term display failed: {e})")
        displayed = True

    # ── RuleFit: rules ───────────────────────────────────────────────
    if hasattr(model, "get_rules") and "rulefit" in type(model).__name__.lower():
        try:
            rules = model.get_rules(exclude_zero_coef=True, sort_by="importance")
            lines.append(f"\n  RuleFit: {len(rules)} rules with non-zero coefficients")
            lines.append(f"  Top {top_rules} rules:")
            for i, r in enumerate(rules[:top_rules]):
                rule_str = R(r.get("rule", r.get("description", "")))
                coef = r.get("coefficient", r.get("coef", 0))
                imp = r.get("importance", abs(coef))
                lines.append(f"    [{i+1:2d}] coef={coef:+.4f}  imp={imp:.4f}")
                lines.append(f"         {rule_str}")
        except Exception as e:
            lines.append(f"    (rule display failed: {e})")
        displayed = True

    # ── FURIA: fuzzy rules ───────────────────────────────────────────
    if hasattr(model, "get_rules_str"):
        try:
            lines.append("\n  FURIA Fuzzy Rules:")
            lines.append(_indent(R(model.get_rules_str())))
        except Exception as e:
            lines.append(f"    (rule display failed: {e})")
        displayed = True

    # ── SLIM / FasterRisk: scorecard ─────────────────────────────────
    if hasattr(model, "get_scorecard"):
        try:
            lines.append("\n  Scoring System (Scorecard):")
            lines.append(_indent(R(model.get_scorecard())))
        except Exception as e:
            lines.append(f"    (scorecard display failed: {e})")
        displayed = True

    # ── MARS: summary with basis functions ───────────────────────────
    if hasattr(model, "summary") and "mars" in type(model).__name__.lower():
        try:
            lines.append("\n  MARS Summary:")
            lines.append(_indent(R(model.summary())))
        except Exception as e:
            lines.append(f"    (summary failed: {e})")
        displayed = True

    # ── GOSDT: tree structure ────────────────────────────────────────
    if hasattr(model, "get_tree_structure"):
        try:
            lines.append("\n  Optimal Decision Tree:")
            lines.append(_indent(R(model.get_tree_structure())))
        except Exception as e:
            lines.append(f"    (tree display failed: {e})")
        displayed = True

    # ── C5.0 / ADT: tree structure ───────────────────────────────────
    if hasattr(model, "summary") and (
        "c50" in type(model).__name__.lower() or "alternating" in type(model).__name__.lower()
    ):
        try:
            tree_str = model.summary(feature_names=fn or None)
            lines.append("\n  Decision Tree:")
            lines.append(_indent(R(tree_str)))
        except Exception as e:
            lines.append(f"    (tree display failed: {e})")
        displayed = True

    # ── Symbolic: discovered equations ───────────────────────────────
    if hasattr(model, "get_best_equation"):
        try:
            eq = model.get_best_equation()
            lines.append("\n  Best Symbolic Equation:")
            lines.append(f"    {R(str(eq))}")
        except Exception as e:
            lines.append(f"    (equation display failed: {e})")
        if hasattr(model, "get_pareto_frontier"):
            try:
                frontier = model.get_pareto_frontier()
                if frontier is not None and len(frontier) > 0:
                    lines.append("\n  Pareto Frontier (complexity vs loss):")
                    for _, row in frontier.head(10).iterrows():
                        lines.append(
                            f"    complexity={int(row.get('complexity', 0)):3d}  "
                            f"loss={row.get('loss', 0):.4f}  "
                            f"eq: {R(str(row.get('equation', '')))}"
                        )
            except Exception:
                pass
        displayed = True

    # ── GAMI-Net: interactions ───────────────────────────────────────
    if hasattr(model, "interaction_pairs_"):
        try:
            pairs = model.interaction_pairs_
            if pairs and fn:
                lines.append("\n  GAMI-Net Interaction Pairs:")
                for p in pairs:
                    f1 = fn[p[0]] if p[0] < len(fn) else str(p[0])
                    f2 = fn[p[1]] if p[1] < len(fn) else str(p[1])
                    lines.append(f"    {f1} x {f2}")
        except Exception:
            pass
        displayed = True

    # ── NODE-GAM: per-feature contributions ──────────────────────────
    if hasattr(model, "get_feature_contributions") and X_sample is not None:
        clsname = type(model).__name__.lower()
        if "nodegam" in clsname or "node_gam" in clsname:
            try:
                contribs = model.get_feature_contributions(X_sample[:5])
                lines.append("\n  NODE-GAM Feature Contributions (first 5 samples):")
                for i in range(min(5, contribs.shape[0])):
                    top_idx = np.argsort(np.abs(contribs[i]))[::-1][:5]
                    parts = [
                        f"{fn[j]}={contribs[i, j]:+.3f}"
                        for j in top_idx
                        if fn and j < len(fn)
                    ]
                    lines.append(f"    sample {i}: {', '.join(parts)}")
            except Exception as e:
                lines.append(f"    (contribution display failed: {e})")
            displayed = True

    # ── NAM: shape-function importances ──────────────────────────────
    if "nam" in type(model).__name__.lower() and hasattr(model, "feature_importances_"):
        lines.extend(
            _show_importances(model, fn, "NAM Feature Importances (shape-function based)", top_features)
        )
        displayed = True

    # ── Linear / LDA: coefficients ───────────────────────────────────
    if hasattr(model, "coef_") and fn:
        clsname = type(model).__name__.lower()
        is_linear = any(k in clsname for k in ("linear", "lda", "logistic"))
        if is_linear:
            try:
                coef = np.asarray(model.coef_).ravel()
                if len(coef) == len(fn):
                    lines.append(f"\n  {name} Coefficients:")
                    pairs = sorted(zip(fn, coef), key=lambda t: abs(t[1]), reverse=True)
                    for fname, c in pairs:
                        lines.append(f"    {fname:35s}  {c:+.4f}")
                    if hasattr(model, "intercept_"):
                        intercept = np.asarray(model.intercept_).ravel()
                        lines.append(f"    {'(intercept)':35s}  {intercept[0]:+.4f}")
            except Exception as e:
                lines.append(f"    (coefficient display failed: {e})")
            displayed = True

    # ── NGBoost / generic: feature importances (if nothing else shown) ─
    if hasattr(model, "feature_importances_") and not displayed and fn:
        lines.extend(_show_importances(model, fn, "Feature Importances", top_features))
        displayed = True

    # ── Naive Bayes: class priors + feature means ────────────────────
    clsname = type(model).__name__.lower()
    if "naivebayes" in clsname or "naive_bayes" in clsname or "gaussiannb" in clsname:
        try:
            inner = getattr(model, "model_", model)
            if hasattr(inner, "class_prior_"):
                lines.append(f"\n  Class Priors: {inner.class_prior_}")
            if hasattr(inner, "theta_") and fn:
                lines.append("\n  Feature Means per Class:")
                for cls_idx in range(inner.theta_.shape[0]):
                    pairs = sorted(
                        zip(fn, inner.theta_[cls_idx]),
                        key=lambda t: t[1],
                        reverse=True,
                    )
                    lines.append(f"    Class {cls_idx}:")
                    for fname, val in pairs[:8]:
                        lines.append(f"      {fname:33s}  mean={val:.4f}")
        except Exception as e:
            lines.append(f"    (Naive Bayes display failed: {e})")
        displayed = True

    if not displayed:
        lines.append("    (No specific display method found for this model)")

    # ── Always show feature importances (unless already fully covered) ─
    skip_fi = {"nam", "linear", "lda", "naivebayes", "naive_bayes", "ngboost"}
    if hasattr(model, "feature_importances_") and displayed and fn:
        if not any(k in clsname for k in skip_fi):
            lines.extend(_show_importances(model, fn, "Feature Importances", top_features))

    text = "\n".join(lines)

    if print_output:
        print(text)

    return text


def display_models(
    models: dict[str, object],
    feature_names: list[str] | None = None,
    X_sample: np.ndarray | None = None,
    *,
    top_rules: int = 15,
    top_features: int = 10,
    print_output: bool = True,
) -> str:
    """Display learned structures for multiple models.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping of model names to fitted estimators.
    feature_names : list of str, optional
        Feature names for readable output.
    X_sample : ndarray, optional
        Sample data for per-sample contribution displays.
    top_rules : int, default=15
        Max rules/terms per model.
    top_features : int, default=10
        Max features per importance display.
    print_output : bool, default=True
        If True, print to stdout.

    Returns
    -------
    str
        Complete formatted text for all models.
    """
    parts: list[str] = []
    for name, model in models.items():
        text = display_model(
            name,
            model,
            feature_names=feature_names,
            X_sample=X_sample,
            top_rules=top_rules,
            top_features=top_features,
            print_output=False,
        )
        parts.append(text)

    full_text = "\n".join(parts)

    if print_output:
        print(full_text)

    return full_text
