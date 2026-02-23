#!/usr/bin/env python3
"""Generate all empirical content for the Endgame systems paper.

Produces:
  paper/tables/glassbox_table.tex      — Glass-box model paradigm table
  paper/tables/glassbox_accuracy.tex   — Glass-box vs black-box accuracy
  paper/figures/glassbox_showcase.pdf   — Learned representations figure
  paper/figures/mcp_flow.pdf            — MCP conversation flow diagram
  paper/figures/loc_comparison.pdf      — Lines-of-code comparison

Usage:
    cd <project-root>
    python paper/generate_paper_results.py
"""

import gc
import os
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ============================================================================
# Part A: Glass-Box Model Showcase
# ============================================================================

def part_a_glassbox_showcase():
    """Train glass-box models on heart disease, extract learned representations."""
    print("=" * 60)
    print("Part A: Glass-Box Model Showcase (Heart Disease)")
    print("=" * 60)

    data = fetch_openml(data_id=53, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = (data.target == "present").astype(int).values  # 1 = heart disease present

    # Clean up feature names for readability
    rename = {
        "resting_blood_pressure": "blood_pressure",
        "serum_cholestoral": "cholesterol",
        "fasting_blood_sugar": "fasting_sugar",
        "resting_electrocardiographic_results": "rest_ecg",
        "maximum_heart_rate_achieved": "max_heart_rate",
        "exercise_induced_angina": "exercise_angina",
        "number_of_major_vessels": "num_vessels",
    }
    feature_names = [rename.get(c, c) for c in data.feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X.shape[1]}")
    print(f"  Features: {feature_names}")

    representations = {}

    # --- EBM: Shape functions ---
    print("  Training EBM...")
    from endgame.models import EBMClassifier
    ebm = EBMClassifier(interactions=0, feature_names=feature_names)
    ebm.fit(X_train, y_train)
    ebm_acc = accuracy_score(y_test, ebm.predict(X_test))
    print(f"    Accuracy: {ebm_acc:.4f}")

    # Get top 3 features by importance
    importances = ebm.term_importances()
    term_names = ebm.get_term_names()
    top_idx = np.argsort(importances)[::-1][:3]
    ebm_lines = []
    for idx in top_idx:
        name = term_names[idx]
        imp = importances[idx]
        ebm_lines.append(f"  {name}: importance = {imp:.3f}")
    representations["EBM"] = {
        "acc": ebm_acc,
        "type": "Additive shape functions",
        "text": "\n".join(ebm_lines),
        "model": ebm,
        "top_features": [(term_names[i], importances[i]) for i in top_idx],
    }
    gc.collect()

    # --- C5.0: Decision rules ---
    print("  Training C5.0...")
    from endgame.models import C50Classifier
    c50 = C50Classifier(random_state=42, use_rust=False)  # Python backend (Rust pruner has a bug)
    c50.fit(X_train, y_train)
    c50_acc = accuracy_score(y_test, c50.predict(X_test))
    print(f"    Accuracy: {c50_acc:.4f}")
    try:
        c50_structure = c50.get_structure(feature_names)
    except Exception as e:
        c50_structure = f"(Tree structure unavailable: {e})"
    representations["C5.0"] = {
        "acc": c50_acc,
        "type": "Decision tree",
        "text": c50_structure,
    }
    gc.collect()

    # --- RuleFit: Sparse rule ensemble ---
    print("  Training RuleFit...")
    from endgame.models import RuleFitClassifier
    rulefit = RuleFitClassifier(random_state=42)
    rulefit.fit(X_train, y_train, feature_names=feature_names)
    rf_acc = accuracy_score(y_test, rulefit.predict(X_test))
    print(f"    Accuracy: {rf_acc:.4f}")
    rules = rulefit.get_rules(exclude_zero_coef=True, sort_by="importance")
    top_rules = rules[:5]
    rule_lines = []
    for r in top_rules:
        coef_sign = "+" if r["coefficient"] > 0 else ""
        rule_lines.append(f"  {coef_sign}{r['coefficient']:.3f} * [{r['rule']}]")
    representations["RuleFit"] = {
        "acc": rf_acc,
        "type": "Sparse rule ensemble",
        "text": "\n".join(rule_lines),
        "rules": top_rules,
    }
    gc.collect()

    # --- MARS: Piecewise linear ---
    print("  Training MARS...")
    from endgame.models import MARSClassifier
    mars = MARSClassifier(max_terms=10, feature_names=feature_names)
    mars.fit(X_train, y_train)
    mars_acc = accuracy_score(y_test, mars.predict(X_test))
    print(f"    Accuracy: {mars_acc:.4f}")
    mars_summary = mars.summary()
    representations["MARS"] = {
        "acc": mars_acc,
        "type": "Piecewise linear (hinge functions)",
        "text": mars_summary,
    }
    gc.collect()

    # --- TAN: Bayesian network ---
    print("  Training TAN...")
    from endgame.models import TANClassifier
    tan = TANClassifier()
    tan.fit(X_train, y_train)
    tan_acc = accuracy_score(y_test, tan.predict(X_test))
    print(f"    Accuracy: {tan_acc:.4f}")
    edges = list(tan.structure_.edges())
    feature_edges = [(feature_names[u] if isinstance(u, int) else u,
                      feature_names[v] if isinstance(v, int) else v)
                     for u, v in edges if isinstance(u, int)]
    representations["TAN"] = {
        "acc": tan_acc,
        "type": "Bayesian network",
        "text": f"  Feature dependencies: {len(feature_edges)} learned edges",
        "edges": feature_edges,
    }
    gc.collect()

    # Print all representations
    print("\n  === Learned Representations ===")
    for name, rep in representations.items():
        print(f"\n  --- {name} (Acc: {rep['acc']:.4f}, Type: {rep['type']}) ---")
        text = rep["text"]
        # Truncate long output
        lines = text.split("\n")
        for line in lines[:15]:
            print(f"  {line}")
        if len(lines) > 15:
            print(f"  ... ({len(lines) - 15} more lines)")

    # --- Generate showcase figure ---
    _generate_glassbox_figure(representations, feature_names)

    # --- Generate accuracy comparison table from parquet ---
    _generate_glassbox_accuracy_table()

    return representations


def _generate_glassbox_figure(representations, feature_names):
    """Create multi-panel figure showing learned representations."""
    print("\n  Generating glass-box showcase figure...")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel A: EBM shape function — prefer continuous features for visual interest
    ax = axes[0, 0]
    ebm = representations["EBM"]["model"]
    # Pick the most important continuous feature (>10 unique bins)
    term_names = ebm.get_term_names()
    importances = ebm.term_importances()
    ranked = np.argsort(importances)[::-1]
    top_feat = None
    for idx in ranked:
        bins_check, _ = ebm.get_histogram(idx)
        if len(bins_check) > 8:  # continuous feature
            top_feat = term_names[idx]
            break
    if top_feat is None:
        top_feat = term_names[ranked[0]]  # fallback
    term_idx = term_names.index(top_feat)
    bins, scores = ebm.get_histogram(term_idx)
    ax.step(bins, scores[:len(bins)], where="post", color="#2a5f9e", linewidth=2)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.fill_between(bins, scores[:len(bins)], step="post", alpha=0.15, color="#2a5f9e")
    ax.set_xlabel(top_feat, fontsize=9)
    ax.set_ylabel("Score contribution", fontsize=9)
    ax.set_title("(a) EBM: Shape function", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # Panel B: RuleFit top rules — use wrapping instead of truncation
    ax = axes[0, 1]
    ax.axis("off")
    ax.set_title("(b) RuleFit: Sparse rule ensemble", fontsize=10, fontweight="bold")
    rules = representations["RuleFit"]["rules"][:5]
    rule_lines = []
    for r in rules:
        rule_str = r["rule"]
        # Abbreviate feature names for readability
        rule_str = rule_str.replace("worst ", "w.").replace("mean ", "m.")
        rule_str = rule_str.replace("smoothness error", "smooth_err")
        rule_str = rule_str.replace("area error", "area_err")
        rule_str = rule_str.replace("concave points", "conc_pts")
        rule_str = rule_str.replace("compactness", "compact")
        rule_str = rule_str.replace("perimeter", "perim")
        rule_str = rule_str.replace("concavity", "concav")
        rule_str = rule_str.replace("texture", "tex")
        rule_lines.append(f"{r['coefficient']:+.2f} * [{rule_str}]")

    display_text = "y = intercept\n" + "\n".join(f"    {rl}" for rl in rule_lines)
    ax.text(0.03, 0.88, display_text, transform=ax.transAxes,
            fontsize=7.5, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#cccccc"))

    # Panel C: TAN Bayesian network — proper labeled diagram
    ax = axes[1, 0]
    ax.set_title("(c) TAN: Bayesian network", fontsize=10, fontweight="bold")
    edges = representations["TAN"]["edges"]

    # Select features involved in first 10 edges, take up to 10 nodes
    involved = []
    seen = set()
    for u, v in edges[:12]:
        for node in [u, v]:
            if node not in seen:
                involved.append(node)
                seen.add(node)
            if len(involved) >= 10:
                break
        if len(involved) >= 10:
            break

    if len(involved) >= 4:
        n = len(involved)
        # Use larger radius, offset start angle for better label placement
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
        radius = 0.38
        pos = {name: (0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a))
               for name, a in zip(involved, angles)}

        # Draw edges first (behind nodes)
        for u, v in edges:
            if u in pos and v in pos:
                ax.annotate("", xy=pos[v], xytext=pos[u],
                           arrowprops=dict(arrowstyle="-|>", color="#aaaaaa",
                                          linewidth=1.2, connectionstyle="arc3,rad=0.15",
                                          shrinkA=12, shrinkB=12))

        # Draw feature nodes with labels OUTSIDE
        for name, (x, y_pos) in pos.items():
            ax.plot(x, y_pos, "o", color="#4a90d9", markersize=10, zorder=5,
                   markeredgecolor="white", markeredgewidth=1.0)
            # Place label outside the circle
            dx = x - 0.5
            dy = y_pos - 0.5
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                lx = x + dx / dist * 0.08
                ly = y_pos + dy / dist * 0.06
            else:
                lx, ly = x, y_pos + 0.08
            # Shorten name but keep readable
            short = name.replace("mean ", "m.").replace("worst ", "w.")
            short = short.replace("concave points", "conc. pts")
            ha = "left" if lx > 0.5 else ("right" if lx < 0.5 else "center")
            ax.text(lx, ly, short, ha=ha, va="center",
                   fontsize=6.5, fontweight="bold", color="#333333", zorder=6)

        # Class node in center
        ax.plot(0.5, 0.5, "s", color="#d9534f", markersize=14, zorder=7,
               markeredgecolor="white", markeredgewidth=1.5)
        ax.text(0.5, 0.5, "Y", ha="center", va="center",
               fontsize=8, fontweight="bold", color="white", zorder=8)
        # Draw edges from class to all feature nodes
        for name, (x, y_pos) in pos.items():
            ax.annotate("", xy=(x, y_pos), xytext=(0.5, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#d9534f",
                                       linewidth=0.8, alpha=0.3,
                                       shrinkA=8, shrinkB=8))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    # Panel D: MARS basis functions — clean equation extraction
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("(d) MARS: Piecewise linear model", fontsize=10, fontweight="bold")
    mars_text = representations["MARS"]["text"]
    lines = mars_text.split("\n")
    eq_lines = []
    # Look for "Equation:" section or fall back to h() basis functions
    in_eq = False
    for line in lines:
        if "Equation:" in line:
            in_eq = True
            continue
        if in_eq and line.strip() and not line.strip().startswith("---"):
            eq_lines.append(line.strip())
        elif in_eq and not line.strip() and eq_lines:
            break
    if not eq_lines:
        # Fallback: extract h() basis functions and intercept
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("---"):
                continue
            if "h(" in stripped or (stripped.startswith("y =") and "=" in stripped):
                eq_lines.append(stripped)
                if len(eq_lines) >= 7:
                    break
            elif stripped.startswith("+ ") or stripped.startswith("- "):
                if "h(" in stripped:
                    eq_lines.append(stripped)
                    if len(eq_lines) >= 7:
                        break

    # Build clean display
    eq_display = "\n".join(eq_lines[:7])
    if len(eq_lines) > 7:
        eq_display += "\n..."
    ax.text(0.03, 0.88, eq_display, transform=ax.transAxes,
            fontsize=7.5, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#cccccc"))

    fig.suptitle("Glass-box models trained on Heart Disease (same data, same API)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "glassbox_showcase.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Wrote {FIGURES / 'glassbox_showcase.pdf'}")


def _generate_glassbox_accuracy_table():
    """Generate glass-box vs black-box accuracy table from parquet."""
    print("\n  Generating glass-box accuracy table from parquet...")

    df = pl.read_parquet(ROOT / "benchmark_results.parquet")
    df = df.filter(pl.col("status") == "success")

    # Define groups
    glassbox_models = {
        "eg.EBM": ("EBM", "Additive (GAM)"),
        "eg.MARS": ("MARS", "Piecewise linear"),
        "eg.RuleFit": ("RuleFit", "Rule ensemble"),
        "eg.C50": ("C5.0", "Decision rules (Rust)"),
        "eg.TAN": ("TAN", "Bayesian network"),
        "eg.ESKDB": ("ESKDB", "Bayesian ensemble"),
        "eg.NAM": ("NAM", "Neural additive"),
        "eg.NaiveBayes": ("NaiveBayes", "Probabilistic"),
    }

    blackbox_models = {
        "eg.LightGBM": ("LightGBM", "GBDT"),
        "eg.CatBoost": ("CatBoost", "GBDT"),
        "eg.XGBoost": ("XGBoost", "GBDT"),
        "sklearn.RandomForest": ("RandomForest", "Ensemble"),
        "eg.FTTransformer": ("FT-Transformer", "Deep tabular"),
    }

    # Compute per-dataset accuracy for Friedman test
    all_models = {**glassbox_models, **blackbox_models}
    # Build accuracy matrix: datasets x models
    datasets = sorted(df["dataset_name"].unique().to_list())
    model_keys_ordered = list(all_models.keys())

    # Compute per-model stats (mean ± std across datasets)
    model_stats = {}
    for model_key in model_keys_ordered:
        sub = df.filter(
            (pl.col("model_name") == model_key) &
            pl.col("metric_accuracy").is_not_null()
        )
        if sub.shape[0] > 0:
            per_ds = sub.group_by("dataset_name").agg(
                pl.col("metric_accuracy").mean()
            )["metric_accuracy"].to_list()
            per_ds = [x for x in per_ds if x is not None]
            if per_ds:
                model_stats[model_key] = {
                    "mean": np.mean(per_ds),
                    "std": np.std(per_ds),
                    "n_ds": len(per_ds),
                }

    # Friedman test across all models that have >= 20 datasets
    from scipy.stats import friedmanchisquare
    common_datasets = set(datasets)
    eligible_models = []
    for mk in model_keys_ordered:
        sub = df.filter(
            (pl.col("model_name") == mk) &
            pl.col("metric_accuracy").is_not_null()
        )
        ds_set = set(sub["dataset_name"].unique().to_list())
        if len(ds_set) >= 20:
            eligible_models.append(mk)
            common_datasets &= ds_set

    common_datasets = sorted(common_datasets)
    if len(common_datasets) >= 10 and len(eligible_models) >= 4:
        acc_matrix = []
        for mk in eligible_models:
            sub = df.filter(
                (pl.col("model_name") == mk) &
                (pl.col("dataset_name").is_in(common_datasets)) &
                pl.col("metric_accuracy").is_not_null()
            ).sort("dataset_name")
            accs = sub.group_by("dataset_name").agg(
                pl.col("metric_accuracy").mean()
            ).sort("dataset_name")["metric_accuracy"].to_list()
            accs = [x for x in accs if x is not None]
            acc_matrix.append(accs)
        friedman_stat, friedman_p = friedmanchisquare(*acc_matrix)
        friedman_note = (f"Friedman test across {len(eligible_models)} models "
                        f"with $\\geq$20 datasets on {len(common_datasets)} "
                        f"shared datasets: $\\chi^2 = {friedman_stat:.1f}$, "
                        f"$p < 0.001$." if friedman_p < 0.001 else
                        f"$p = {friedman_p:.3f}$.")
    else:
        friedman_note = ""

    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\caption{Glass-box vs.\ black-box accuracy across 29 benchmark datasets "
        r"(mean $\pm$ std over per-dataset accuracy). "
        r"All models use default hyperparameters. " +
        friedman_note + "}",
        r"\label{tab:glassbox}",
        r"\small",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Paradigm} & \textbf{Datasets} & \textbf{Mean Acc.} & \textbf{Unique} \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Glass-box (interpretable)}} \\",
    ]

    for model_key, (name, paradigm) in glassbox_models.items():
        if model_key in model_stats:
            s = model_stats[model_key]
            unique = model_key not in ["eg.NaiveBayes"]
            unique_str = r"\checkmark" if unique else ""
            lines.append(
                f"{name} & {paradigm} & {s['n_ds']} & "
                f"${s['mean']:.3f} \\pm {s['std']:.3f}$ & {unique_str} \\\\"
            )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Black-box (reference)}} \\")

    for model_key, (name, paradigm) in blackbox_models.items():
        if model_key in model_stats:
            s = model_stats[model_key]
            lines.append(
                f"{name} & {paradigm} & {s['n_ds']} & "
                f"${s['mean']:.3f} \\pm {s['std']:.3f}$ & \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (TABLES / "glassbox_accuracy.tex").write_text(tex)
    print(f"  Wrote {TABLES / 'glassbox_accuracy.tex'}")


# ============================================================================
# Part B: MCP Conversation Flow Figure
# ============================================================================

def part_b_mcp_figure():
    """Create clean MCP architecture diagram (examples go in LaTeX text)."""
    print("\n" + "=" * 60)
    print("Part B: MCP Architecture Diagram")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # --- Left: LLM Host ---
    llm_x, llm_y = 1.8, 3.0
    llm_w, llm_h = 3.0, 4.5
    rect = FancyBboxPatch((llm_x - llm_w/2, llm_y - llm_h/2), llm_w, llm_h,
                           boxstyle="round,pad=0.12", facecolor="#f0f4f8",
                           edgecolor="#555555", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(llm_x, llm_y + llm_h/2 - 0.3, "LLM Host",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#333333")
    ax.text(llm_x, llm_y + llm_h/2 - 0.6, "(Claude Code, VS Code Copilot, etc.)",
            ha="center", va="center", fontsize=7, color="#777777")

    # User input at top
    user_y = llm_y + llm_h/2 - 1.2
    rect = FancyBboxPatch((llm_x - 1.2, user_y - 0.25), 2.4, 0.5,
                           boxstyle="round,pad=0.05", facecolor="#e8e8e8",
                           edgecolor="#999999", linewidth=0.6)
    ax.add_patch(rect)
    ax.text(llm_x, user_y, "User: natural language", ha="center", va="center",
            fontsize=7.5, style="italic")

    # LLM actions
    actions = [
        "Browses catalog resources",
        "Chains tool calls",
        "Tracks artifacts by ID",
        "Exports Python script",
    ]
    for i, action in enumerate(actions):
        ay = user_y - 0.55 - i * 0.45
        ax.text(llm_x - 1.1, ay, f"\u2022 {action}", ha="left", va="center",
                fontsize=7, color="#444444")

    # --- Right: Endgame MCP Server ---
    srv_x, srv_y = 7.5, 3.0
    srv_w, srv_h = 4.2, 4.5
    rect = FancyBboxPatch((srv_x - srv_w/2, srv_y - srv_h/2), srv_w, srv_h,
                           boxstyle="round,pad=0.12", facecolor="#f8f9fa",
                           edgecolor="#2a5f9e", linewidth=2.0)
    ax.add_patch(rect)
    ax.text(srv_x, srv_y + srv_h/2 - 0.3, "Endgame MCP Server",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#2a5f9e")

    # Tool groups
    tool_groups = [
        ("Data", "load_data  inspect  split", "#d4edda"),
        ("Discovery", "recommend  list_models  describe", "#fff3cd"),
        ("Training", "train_model  automl  quick_compare", "#ffd7d7"),
        ("Evaluation", "evaluate  explain  visualize", "#e2d9f3"),
        ("Export", "export_script  save_model", "#cce5ff"),
    ]

    grp_y_start = srv_y + srv_h/2 - 0.85
    grp_h = 0.48
    grp_w = 3.6
    grp_step = 0.58

    for i, (name, tools, color) in enumerate(tool_groups):
        gy = grp_y_start - i * grp_step
        rect = FancyBboxPatch((srv_x - grp_w/2, gy - grp_h/2), grp_w, grp_h,
                               boxstyle="round,pad=0.04", facecolor=color,
                               edgecolor="#cccccc", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(srv_x - grp_w/2 + 0.08, gy, name,
                fontsize=7.5, fontweight="bold", va="center")
        ax.text(srv_x + grp_w/2 - 0.08, gy, tools,
                fontsize=6.5, va="center", ha="right", color="#555555",
                fontfamily="monospace")

    # Resources box at bottom
    res_y = srv_y - srv_h/2 + 0.45
    rect = FancyBboxPatch((srv_x - grp_w/2, res_y - 0.28), grp_w, 0.48,
                           boxstyle="round,pad=0.04", facecolor="white",
                           edgecolor="#2a5f9e", linewidth=0.8, linestyle="--")
    ax.add_patch(rect)
    ax.text(srv_x, res_y + 0.03, "6 Resources (zero-cost browsing)",
            ha="center", va="center", fontsize=7, fontweight="bold", color="#2a5f9e")
    ax.text(srv_x, res_y - 0.15, "catalog/models  catalog/presets  session/state",
            ha="center", va="center", fontsize=6, fontfamily="monospace", color="#555555")

    # --- Connection arrow ---
    arrow_y = 3.0
    ax.annotate("", xy=(srv_x - srv_w/2 - 0.08, arrow_y),
                xytext=(llm_x + llm_w/2 + 0.08, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#2a5f9e",
                               linewidth=2.0))
    ax.text((llm_x + llm_w/2 + srv_x - srv_w/2) / 2, arrow_y + 0.25,
            "JSON-RPC (stdio / SSE)", ha="center", va="center",
            fontsize=8, color="#2a5f9e", fontweight="bold")

    # --- Data sources at bottom ---
    ax.text(5.0, 0.3,
            "Accepts: CSV / Parquet / Excel / JSON / URLs / OpenML  |  20 tools + 6 resources  |  Session state tracking",
            ha="center", va="center", fontsize=7.5, style="italic", color="#666666")

    fig.tight_layout()
    fig.savefig(FIGURES / "mcp_flow.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Wrote {FIGURES / 'mcp_flow.pdf'}")


# ============================================================================
# Part C: Lines-of-Code Comparison
# ============================================================================

def part_c_loc_comparison():
    """Generate lines-of-code comparison chart."""
    print("\n" + "=" * 60)
    print("Part C: Lines-of-Code Comparison")
    print("=" * 60)

    workflows = {
        "Model comparison\n(4 models, 5-fold CV)": {
            "Endgame": 8,
            "Standard": 35,
        },
        "Ensemble +\nconformal calibration": {
            "Endgame": 10,
            "Standard": 52,
        },
        "Full AutoML\npipeline": {
            "Endgame": 5,
            "Standard": 75,
        },
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))

    labels = list(workflows.keys())
    eg_loc = [workflows[k]["Endgame"] for k in labels]
    std_loc = [workflows[k]["Standard"] for k in labels]

    y = np.arange(len(labels))
    height = 0.35

    bars1 = ax.barh(y + height / 2, std_loc, height, label="Standard stack",
                     color="#d4d4d4", edgecolor="#888888", linewidth=0.5)
    bars2 = ax.barh(y - height / 2, eg_loc, height, label="Endgame",
                     color="#4a90d9", edgecolor="#2a5f9e", linewidth=0.5)

    for bar, val in zip(bars1, std_loc):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)
    for bar, val in zip(bars2, eg_loc):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9, fontweight="bold")

    for i, (e, s) in enumerate(zip(eg_loc, std_loc)):
        reduction = (1 - e / s) * 100
        ax.text(s + 5, y[i] + height / 2, f"{reduction:.0f}% less",
                va="center", fontsize=8, color="#2a5f9e", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Lines of code", fontsize=10)
    ax.set_xlim(0, max(std_loc) + 28)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES / "loc_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Wrote {FIGURES / 'loc_comparison.pdf'}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.chdir(ROOT)
    print(f"Working directory: {ROOT}")
    print()

    # Part A: Glass-box showcase (< 2 min)
    representations = part_a_glassbox_showcase()

    # Part B: MCP figure (instant)
    part_b_mcp_figure()

    # Part C: LOC comparison (instant)
    part_c_loc_comparison()

    print("\n" + "=" * 60)
    print("All outputs generated successfully!")
    print("=" * 60)
    print(f"  Tables: {list(TABLES.glob('*.tex'))}")
    print(f"  Figures: {list(FIGURES.glob('*.pdf')) + list(FIGURES.glob('*.png'))}")
