#!/usr/bin/env python3
"""Generate PNG images for all 40 visualization types in the docs guide.

Uses diverse datasets (iris, wine, breast cancer, california housing, digits,
synthetic hard/imbalanced classification, and 20newsgroups NLP data) to
showcase each chart at its best. Screenshots are captured with headless
Chrome at viewport sizes matched to each chart.

Usage:
    python scripts/generate_viz_images.py
"""

import os
import sys
import tempfile

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# --- Setup paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "guides", "images", "viz")
HTML_DIR = tempfile.mkdtemp(prefix="endgame_viz_")
os.makedirs(IMG_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
from endgame.visualization import (
    ArcDiagramVisualizer,
    BarChartVisualizer,
    BoxPlotVisualizer,
    BumpChartVisualizer,
    CalibrationPlotVisualizer,
    ChordDiagramVisualizer,
    ConfusionMatrixVisualizer,
    DonutChartVisualizer,
    DotMatrixVisualizer,
    DumbbellChartVisualizer,
    ErrorBarsVisualizer,
    FlowChartVisualizer,
    FunnelChartVisualizer,
    GaugeChartVisualizer,
    HeatmapVisualizer,
    HistogramVisualizer,
    LiftChartVisualizer,
    LineChartVisualizer,
    LollipopChartVisualizer,
    NetworkDiagramVisualizer,
    NightingaleRoseVisualizer,
    PDP2DVisualizer,
    PDPVisualizer,
    PRCurveVisualizer,
    ParallelCoordinatesVisualizer,
    ROCCurveVisualizer,
    RadarChartVisualizer,
    RadialBarVisualizer,
    RidgelinePlotVisualizer,
    SankeyVisualizer,
    ScatterplotVisualizer,
    SpiralPlotVisualizer,
    StreamGraphVisualizer,
    SunburstVisualizer,
    TreeVisualizer,
    TreemapVisualizer,
    VennDiagramVisualizer,
    ViolinPlotVisualizer,
    WaterfallVisualizer,
    WordCloudVisualizer,
)


def save_viz(viz, name: str, width: int = 1100, height: int = 750):
    """Save a visualizer as both HTML and PNG."""
    html_path = os.path.join(HTML_DIR, f"{name}.html")
    png_path = os.path.join(IMG_DIR, f"{name}.png")
    viz.save(html_path)
    viz.to_png(png_path, width=width, height=height)
    size = os.path.getsize(png_path) if os.path.exists(png_path) else 0
    print(f"  [{name}] -> {size:,} bytes")
    return png_path


def main():
    print("=" * 60)
    print("Generating 40 visualization images (diverse datasets)")
    print("=" * 60)

    # =================================================================
    # LOAD DATASETS
    # =================================================================
    print("\n[1/3] Loading datasets...")

    # Breast cancer - binary classification with 30 features
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    fn_bc = list(bc.feature_names)
    X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(
        X_bc, y_bc, test_size=0.3, random_state=42, stratify=y_bc
    )

    # Iris - 3-class classification with 4 features
    iris = load_iris()
    X_ir, y_ir = iris.data, iris.target
    fn_ir = list(iris.feature_names)
    cn_ir = list(iris.target_names)
    X_ir_tr, X_ir_te, y_ir_tr, y_ir_te = train_test_split(
        X_ir, y_ir, test_size=0.3, random_state=42, stratify=y_ir
    )

    # Wine - 3-class, 13 features with interesting correlations
    wine = load_wine()
    X_wn, y_wn = wine.data, wine.target
    fn_wn = list(wine.feature_names)
    cn_wn = list(wine.target_names)
    X_wn_tr, X_wn_te, y_wn_tr, y_wn_te = train_test_split(
        X_wn, y_wn, test_size=0.3, random_state=42, stratify=y_wn
    )

    # Digits - 10-class, 64 features (for multiclass flow charts)
    digits = load_digits()
    X_dg, y_dg = digits.data, digits.target
    X_dg_tr, X_dg_te, y_dg_tr, y_dg_te = train_test_split(
        X_dg, y_dg, test_size=0.3, random_state=42, stratify=y_dg
    )

    # California housing - regression
    try:
        from sklearn.datasets import fetch_california_housing
        cal = fetch_california_housing()
        X_cal, y_cal = cal.data[:2000], cal.target[:2000]
        fn_cal = list(cal.feature_names)
        X_cal_tr, X_cal_te, y_cal_tr, y_cal_te = train_test_split(
            X_cal, y_cal, test_size=0.3, random_state=42
        )
        has_cal = True
    except Exception:
        has_cal = False

    # Hard synthetic classification (for ROC, radar, gauge, dumbbell)
    X_hard, y_hard = make_classification(
        n_samples=2000, n_features=20, n_informative=8, n_redundant=4,
        flip_y=0.15, class_sep=0.6, random_state=42,
    )
    X_hard_tr, X_hard_te, y_hard_tr, y_hard_te = train_test_split(
        X_hard, y_hard, test_size=0.3, random_state=42, stratify=y_hard,
    )

    # Imbalanced classification (for PR curve, nightingale rose)
    X_imb, y_imb = make_classification(
        n_samples=3000, n_features=20, n_informative=8, n_redundant=4,
        flip_y=0.05, class_sep=0.5, weights=[0.92, 0.08], random_state=42,
    )
    X_imb_tr, X_imb_te, y_imb_tr, y_imb_te = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb,
    )

    # NLP data for word cloud
    has_nlp = False
    nlp_word_dict = {}
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("  Loading 20newsgroups for word cloud...")
        news = fetch_20newsgroups(
            subset='train',
            categories=['sci.space', 'sci.med', 'comp.graphics', 'rec.sport.baseball'],
            remove=('headers', 'footers', 'quotes'),
        )
        tfidf = TfidfVectorizer(max_features=80, stop_words='english', min_df=5)
        tfidf_matrix = tfidf.fit_transform(news.data)
        nlp_words_arr = tfidf.get_feature_names_out()
        nlp_weights = np.array(tfidf_matrix.mean(axis=0)).flatten()
        nlp_word_dict = {w: float(s) for w, s in zip(nlp_words_arr, nlp_weights)}
        has_nlp = True
        print(f"  Loaded {len(nlp_word_dict)} NLP terms")
    except Exception as e:
        print(f"  Warning: Could not load 20newsgroups: {e}")

    # =================================================================
    # TRAIN MODELS
    # =================================================================
    print("[2/3] Training models...")

    # Binary models on breast cancer
    rf_bc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_bc.fit(X_bc_tr, y_bc_tr)
    lr_bc = LogisticRegression(max_iter=5000, random_state=42)
    lr_bc.fit(X_bc_tr, y_bc_tr)
    gb_bc = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    gb_bc.fit(X_bc_tr, y_bc_tr)
    svm_bc = SVC(probability=True, random_state=42)
    svm_bc.fit(X_bc_tr, y_bc_tr)

    # Multiclass on iris
    rf_ir = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_ir.fit(X_ir_tr, y_ir_tr)

    # Wine classifier
    rf_wn = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_wn.fit(X_wn_tr, y_wn_tr)
    gb_wn = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb_wn.fit(X_wn_tr, y_wn_tr)

    # Digits classifier
    rf_dg = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf_dg.fit(X_dg_tr, y_dg_tr)

    # Regression on california housing
    if has_cal:
        rf_cal = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_cal.fit(X_cal_tr, y_cal_tr)
        gb_cal = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_cal.fit(X_cal_tr, y_cal_tr)

    # Decision tree on iris (for tree viz)
    dt_ir = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_ir.fit(X_ir_tr, y_ir_tr)

    # Models on hard synthetic dataset (for ROC, radar, gauge)
    rf_hard = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_hard.fit(X_hard_tr, y_hard_tr)
    lr_hard = LogisticRegression(max_iter=5000, random_state=42)
    lr_hard.fit(X_hard_tr, y_hard_tr)
    gb_hard = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    gb_hard.fit(X_hard_tr, y_hard_tr)
    dt_hard = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_hard.fit(X_hard_tr, y_hard_tr)

    # Overfitting model on hard dataset (for dumbbell)
    rf_overfit = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42,
    )
    rf_overfit.fit(X_hard_tr, y_hard_tr)

    # Models on imbalanced dataset (for PR curve, nightingale rose)
    rf_imb = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_imb.fit(X_imb_tr, y_imb_tr)
    lr_imb = LogisticRegression(max_iter=5000, random_state=42)
    lr_imb.fit(X_imb_tr, y_imb_tr)
    gb_imb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    gb_imb.fit(X_imb_tr, y_imb_tr)

    # =================================================================
    # GENERATE VISUALIZATIONS
    # =================================================================
    print(f"\n[3/3] Generating visualizations -> {IMG_DIR}")
    count = 0

    # ===== ML EVALUATION CHARTS (Tier A) =====
    print("\n--- ML Evaluation Charts (Tier A) ---")

    # 1. ROC Curve — hard synthetic, 4 models with spread AUCs
    print("  Dataset: Hard synthetic classification (4 models)")
    from sklearn.metrics import auc, roc_curve
    y_scores_hard = {
        "Gradient Boosting": gb_hard.predict_proba(X_hard_te)[:, 1],
        "Random Forest": rf_hard.predict_proba(X_hard_te)[:, 1],
        "Logistic Regression": lr_hard.predict_proba(X_hard_te)[:, 1],
        "Decision Tree": dt_hard.predict_proba(X_hard_te)[:, 1],
    }
    curves = []
    for label, scores in y_scores_hard.items():
        fpr, tpr, thresholds = roc_curve(y_hard_te, scores)
        roc_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        curves.append({
            "fpr": fpr[::max(1, len(fpr) // 200)].tolist(),
            "tpr": tpr[::max(1, len(tpr) // 200)].tolist(),
            "auc": round(float(roc_auc), 4),
            "label": f"{label} (AUC = {roc_auc:.3f})",
            "optimalPoint": {
                "fpr": round(float(fpr[best_idx]), 4),
                "tpr": round(float(tpr[best_idx]), 4),
                "threshold": round(float(thresholds[best_idx]), 4),
            },
        })
    viz = ROCCurveVisualizer(curves, title="ROC Curve — Noisy Classification Task", width=750, height=600)
    save_viz(viz, "roc_curve", 850, 700)
    count += 1

    # 2. PR Curve — imbalanced dataset (8% positive class)
    print("  Dataset: Imbalanced synthetic (92/8 class split)")
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_imb_scores = {
        "Random Forest": rf_imb.predict_proba(X_imb_te)[:, 1],
        "Gradient Boosting": gb_imb.predict_proba(X_imb_te)[:, 1],
        "Logistic Regression": lr_imb.predict_proba(X_imb_te)[:, 1],
    }
    pr_curves = []
    for label, scores in y_imb_scores.items():
        prec, rec, thresholds = precision_recall_curve(y_imb_te, scores)
        ap = average_precision_score(y_imb_te, scores)
        # Find F1-optimal point
        f1_scores = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-8)
        best_idx = int(np.argmax(f1_scores))
        pr_curves.append({
            "precision": prec[::max(1, len(prec) // 200)].tolist(),
            "recall": rec[::max(1, len(rec) // 200)].tolist(),
            "ap": round(float(ap), 4),
            "label": f"{label} (AP = {ap:.3f})",
            "optimalPoint": {
                "precision": round(float(prec[best_idx]), 4),
                "recall": round(float(rec[best_idx]), 4),
                "threshold": round(float(thresholds[best_idx]), 4),
                "f1": round(float(f1_scores[best_idx]), 4),
            },
        })
    prevalence = float(y_imb_te.mean())
    viz = PRCurveVisualizer(
        pr_curves, prevalence=prevalence,
        title="Precision-Recall — Imbalanced Classification (8% Positive)",
        width=750, height=600,
    )
    save_viz(viz, "pr_curve", 850, 700)
    count += 1

    # 3. Calibration Plot — breast cancer, multiple models
    print("  Dataset: Breast Cancer (calibration, 4 models)")
    viz = CalibrationPlotVisualizer.from_multiple(
        y_bc_te,
        {
            "Random Forest": rf_bc.predict_proba(X_bc_te)[:, 1],
            "Logistic Reg.": lr_bc.predict_proba(X_bc_te)[:, 1],
            "Gradient Boosting": gb_bc.predict_proba(X_bc_te)[:, 1],
            "SVM": svm_bc.predict_proba(X_bc_te)[:, 1],
        },
        n_bins=10,
        width=750, height=650,
    )
    save_viz(viz, "calibration_plot", 850, 780)
    count += 1

    # 4. Lift Chart — breast cancer
    viz = LiftChartVisualizer.from_estimator(rf_bc, X_bc_te, y_bc_te, width=900, height=500)
    save_viz(viz, "lift_chart", 1000, 620)
    count += 1

    # 5. PDP (1D) — california housing (regression, continuous features)
    print("  Dataset: California Housing (regression, PDP)")
    if has_cal:
        viz = PDPVisualizer.from_estimator(
            rf_cal, X_cal_tr, feature=0, n_ice_lines=50,
            width=800, height=500,
        )
        save_viz(viz, "pdp_1d", 900, 620)
    count += 1

    # 6. PDP 2D — california housing
    if has_cal:
        viz = PDP2DVisualizer.from_estimator(
            rf_cal, X_cal_tr, features=(0, 6), grid_resolution=25,
            width=800, height=550,
        )
        save_viz(viz, "pdp_2d", 900, 680)
    count += 1

    # 7. Waterfall — breast cancer feature contributions
    print("  Dataset: Breast Cancer (waterfall)")
    top_idx = np.argsort(rf_bc.feature_importances_)[::-1][:12]
    top_names = [fn_bc[i] for i in top_idx]
    top_vals = [float(rf_bc.feature_importances_[i]) for i in top_idx]
    signed_vals = [v * (1 if i % 2 == 0 else -0.6) for i, v in enumerate(top_vals)]
    viz = WaterfallVisualizer.from_contributions(
        top_names, signed_vals, base_value=0.63,
        width=850, height=500,
    )
    save_viz(viz, "waterfall", 950, 620)
    count += 1

    # ===== CORE CHARTS (Tier 1) =====
    print("\n--- Core Charts (Tier 1) ---")

    # 8. Bar Chart — wine feature importances (13 well-named features)
    print("  Dataset: Wine (bar chart, 13 features)")
    viz = BarChartVisualizer.from_importances(
        rf_wn, feature_names=fn_wn, top_n=13,
        width=850, height=500,
    )
    save_viz(viz, "bar_chart", 950, 620)
    count += 1

    # 9. Heatmap — wine correlation matrix (13 features)
    viz = HeatmapVisualizer.from_correlation(
        X_wn, feature_names=fn_wn,
        width=850, height=700,
    )
    save_viz(viz, "heatmap", 950, 820)
    count += 1

    # 10. Confusion Matrix — digits 10×10 (rich misclassification patterns)
    print("  Dataset: Digits (10×10 confusion matrix)")
    viz = ConfusionMatrixVisualizer.from_estimator(
        rf_dg, X_dg_te, y_dg_te,
        class_names=[str(i) for i in range(10)],
        width=700, height=650,
    )
    save_viz(viz, "confusion_matrix", 820, 780)
    count += 1

    # 11. Histogram — synthetic bimodal distribution
    print("  Data: Synthetic bimodal distribution")
    np.random.seed(42)
    bimodal = np.concatenate([
        np.random.normal(0.3, 0.08, 300),
        np.random.normal(0.75, 0.1, 200),
    ])
    viz = HistogramVisualizer(
        [bimodal.tolist()],
        series_names=["Prediction Scores"],
        kde=True,
        title="Bimodal Prediction Distribution",
        width=800, height=450,
    )
    save_viz(viz, "histogram", 900, 570)
    count += 1

    # 12. Line Chart — learning curve (breast cancer)
    print("  Dataset: Breast Cancer (learning curve)")
    sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42),
        X_bc, y_bc, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy",
    )
    viz = LineChartVisualizer.from_learning_curve(
        sizes.tolist(), train_scores.tolist(), test_scores.tolist(),
        width=850, height=480,
    )
    save_viz(viz, "line_chart", 950, 600)
    count += 1

    # 13. Scatterplot — iris PCA embedding (3 well-separated clusters)
    print("  Dataset: Iris (PCA scatterplot)")
    emb = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_ir))
    label_names = [cn_ir[yi] for yi in y_ir]
    viz = ScatterplotVisualizer.from_embedding(
        emb, labels=label_names, title="PCA of Iris Dataset",
        width=800, height=550,
    )
    save_viz(viz, "scatterplot", 900, 670)
    count += 1

    # ===== STATISTICAL CHARTS (Tier 2) =====
    print("\n--- Statistical Charts (Tier 2) ---")

    # 14. Box Plot — 5-model CV comparison on breast cancer
    print("  Dataset: Breast Cancer (5-model box plot)")
    cv_scores_5 = {
        "Random Forest": cross_val_score(rf_bc, X_bc, y_bc, cv=10, scoring="accuracy").tolist(),
        "Logistic Reg.": cross_val_score(lr_bc, X_bc, y_bc, cv=10, scoring="accuracy").tolist(),
        "Grad. Boosting": cross_val_score(gb_bc, X_bc, y_bc, cv=10, scoring="accuracy").tolist(),
        "SVM": cross_val_score(svm_bc, X_bc, y_bc, cv=10, scoring="accuracy").tolist(),
        "Decision Tree": cross_val_score(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            X_bc, y_bc, cv=10, scoring="accuracy"
        ).tolist(),
    }
    viz = BoxPlotVisualizer.from_cv_results(cv_scores_5, width=850, height=480)
    save_viz(viz, "box_plot", 950, 600)
    count += 1

    # 15. Violin Plot — same 5-model comparison
    viz = ViolinPlotVisualizer(
        cv_scores_5, title="CV Score Distributions (10-fold)",
        width=850, height=480,
    )
    save_viz(viz, "violin_plot", 950, 600)
    count += 1

    # 16. Error Bars — same 5-model comparison
    viz = ErrorBarsVisualizer.from_cv_results(cv_scores_5, width=850, height=450)
    save_viz(viz, "error_bars", 950, 570)
    count += 1

    # 17. Parallel Coordinates — synthetic HPO with 10 trials
    print("  Data: Synthetic HPO (parallel coordinates)")
    np.random.seed(42)
    trials = []
    for _ in range(10):
        lr_val = 10 ** np.random.uniform(-4, -1)
        depth = np.random.choice([3, 4, 5, 6, 8, 10, 12])
        n_est = np.random.choice([50, 100, 150, 200, 300, 500])
        min_split = np.random.choice([2, 5, 10, 20])
        score = 0.85 + 0.05 * np.exp(-((depth - 5) ** 2) / 8) + 0.03 * np.exp(-((n_est - 200) ** 2) / 20000)
        score += np.random.normal(0, 0.005)
        trials.append({
            "learning_rate": round(lr_val, 5),
            "max_depth": int(depth),
            "n_estimators": int(n_est),
            "min_samples_split": int(min_split),
            "score": round(float(score), 4),
        })
    viz = ParallelCoordinatesVisualizer(trials, color_by="score", width=950, height=480)
    save_viz(viz, "parallel_coordinates", 1050, 600)
    count += 1

    # 18. Radar Chart — 4 models on hard dataset with distinct profiles
    print("  Dataset: Hard synthetic (4-model radar)")
    radar_models = {
        "Grad. Boosting": gb_hard,
        "Random Forest": rf_hard,
        "Logistic Reg.": lr_hard,
        "Decision Tree": dt_hard,
    }
    radar_series = {}
    for name, mdl in radar_models.items():
        yp = mdl.predict(X_hard_te)
        yproba = mdl.predict_proba(X_hard_te)[:, 1]
        radar_series[name] = [
            round(accuracy_score(y_hard_te, yp), 3),
            round(precision_score(y_hard_te, yp), 3),
            round(recall_score(y_hard_te, yp), 3),
            round(f1_score(y_hard_te, yp), 3),
            round(roc_auc_score(y_hard_te, yproba), 3),
        ]
    viz = RadarChartVisualizer(
        dimensions=["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"],
        series=radar_series,
        title="Model Comparison — Noisy Classification",
        width=700, height=650,
    )
    save_viz(viz, "radar_chart", 800, 770)
    count += 1

    # 19. Treemap — hierarchical breast cancer features (Mean / SE / Worst)
    print("  Dataset: Breast Cancer (hierarchical treemap)")
    measurements = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave pts", "symmetry", "fractal dim",
    ]
    group_names = ["Mean", "Std Error", "Worst"]
    importances = rf_bc.feature_importances_

    tm_labels = ["All Features"]
    tm_parents = [""]
    tm_values = [float(importances.sum())]

    for gi, group in enumerate(group_names):
        group_imp = float(importances[gi * 10:(gi + 1) * 10].sum())
        tm_labels.append(group)
        tm_parents.append("All Features")
        tm_values.append(group_imp)

        for mi, meas in enumerate(measurements):
            feat_idx = gi * 10 + mi
            tm_labels.append(f"{group}: {meas}")
            tm_parents.append(group)
            tm_values.append(float(importances[feat_idx]))

    viz = TreemapVisualizer(
        labels=tm_labels, values=tm_values, parents=tm_parents,
        title="Breast Cancer Features — Grouped by Statistic",
        width=950, height=600,
    )
    save_viz(viz, "treemap", 1050, 720)
    count += 1

    # 20. Sunburst — AutoML pipeline time allocation hierarchy
    print("  Data: AutoML pipeline hierarchy (sunburst)")
    viz = SunburstVisualizer(
        labels=[
            "AutoML Pipeline",
            # Level 1
            "Data Preparation", "Model Training", "Post-Processing",
            # Level 2: Data Prep
            "Cleaning", "Feature Eng.", "Augmentation",
            # Level 2: Model Training
            "GBDTs", "Trees", "Linear", "Neural Nets",
            # Level 2: Post-Processing
            "Ensembling", "Calibration", "Threshold Opt.",
            # Level 3: GBDTs
            "XGBoost", "LightGBM", "CatBoost",
            # Level 3: Trees
            "Random Forest", "Extra Trees",
            # Level 3: Linear
            "Logistic Reg.", "Ridge",
            # Level 3: Neural
            "MLP", "TabNet",
        ],
        parents=[
            "",
            "AutoML Pipeline", "AutoML Pipeline", "AutoML Pipeline",
            "Data Preparation", "Data Preparation", "Data Preparation",
            "Model Training", "Model Training", "Model Training", "Model Training",
            "Post-Processing", "Post-Processing", "Post-Processing",
            "GBDTs", "GBDTs", "GBDTs",
            "Trees", "Trees",
            "Linear", "Linear",
            "Neural Nets", "Neural Nets",
        ],
        values=[
            1.0,
            0.15, 0.65, 0.20,
            0.05, 0.07, 0.03,
            0.25, 0.15, 0.10, 0.15,
            0.10, 0.05, 0.05,
            0.10, 0.08, 0.07,
            0.10, 0.05,
            0.06, 0.04,
            0.08, 0.07,
        ],
        title="AutoML Pipeline — Time Allocation",
        width=700, height=650,
    )
    save_viz(viz, "sunburst", 800, 770)
    count += 1

    # ===== FLOW & COMPOSITION (Tier 3) =====
    print("\n--- Flow & Composition (Tier 3) ---")

    # 21. Sankey — digits 10-class prediction flow
    print("  Dataset: Digits (10-class Sankey)")
    cm_dg = confusion_matrix(y_dg_te, rf_dg.predict(X_dg_te))
    dg_classes = [str(i) for i in range(10)]
    nodes = [f"True {c}" for c in dg_classes] + [f"Pred {c}" for c in dg_classes]
    links = []
    for i in range(10):
        for j in range(10):
            if cm_dg[i][j] > 0:
                links.append((f"True {i}", f"Pred {j}", int(cm_dg[i][j])))
    viz = SankeyVisualizer(
        nodes=nodes, links=links,
        title="Digit Classification — Prediction Flow",
        width=950, height=600,
    )
    save_viz(viz, "sankey", 1050, 720)
    count += 1

    # 22. Dot Matrix — breast cancer accuracy breakdown
    print("  Dataset: Breast Cancer (dot matrix)")
    bc_acc = accuracy_score(y_bc_te, rf_bc.predict(X_bc_te))
    viz = DotMatrixVisualizer(
        labels=["Correct", "Incorrect"],
        values=[int(bc_acc * 100), 100 - int(bc_acc * 100)],
        n_dots=100,
        title=f"Random Forest Accuracy — {bc_acc:.1%}",
        width=700, height=500,
    )
    save_viz(viz, "dot_matrix", 800, 620)
    count += 1

    # 23. Venn — 3-model agreement on breast cancer
    print("  Dataset: Breast Cancer (3-model Venn)")
    rf_correct = set(np.where(rf_bc.predict(X_bc_te) == y_bc_te)[0])
    lr_correct = set(np.where(lr_bc.predict(X_bc_te) == y_bc_te)[0])
    gb_correct = set(np.where(gb_bc.predict(X_bc_te) == y_bc_te)[0])
    viz = VennDiagramVisualizer(
        sets={"RF": len(rf_correct), "LR": len(lr_correct), "GB": len(gb_correct)},
        intersections={
            "RF&LR": len(rf_correct & lr_correct),
            "RF&GB": len(rf_correct & gb_correct),
            "LR&GB": len(lr_correct & gb_correct),
            "RF&LR&GB": len(rf_correct & lr_correct & gb_correct),
        },
        title="Correct Predictions — 3 Model Agreement",
        width=700, height=550,
    )
    save_viz(viz, "venn_diagram", 800, 670)
    count += 1

    # 24. Word Cloud — NLP vocabulary from 20newsgroups
    print("  Data: 20newsgroups NLP vocabulary (word cloud)")
    if has_nlp and nlp_word_dict:
        viz = WordCloudVisualizer(
            words=nlp_word_dict,
            max_words=80,
            title="Top Terms — Science & Technology Newsgroups",
            width=900, height=550,
        )
    else:
        # Fallback: rich synthetic NLP vocabulary
        nlp_fallback = {
            "machine": 0.95, "learning": 0.93, "neural": 0.90, "network": 0.88,
            "algorithm": 0.85, "training": 0.83, "model": 0.92, "data": 0.91,
            "prediction": 0.80, "classification": 0.78, "regression": 0.75,
            "optimization": 0.73, "gradient": 0.70, "descent": 0.55,
            "transformer": 0.82, "attention": 0.79, "embedding": 0.76,
            "convolution": 0.72, "recurrent": 0.65, "generative": 0.68,
            "accuracy": 0.77, "precision": 0.74, "recall": 0.71,
            "ensemble": 0.69, "boosting": 0.67, "bagging": 0.60,
            "feature": 0.86, "selection": 0.63, "extraction": 0.58,
            "regularization": 0.62, "dropout": 0.57, "normalization": 0.64,
            "hyperparameter": 0.53, "validation": 0.66, "cross-validation": 0.61,
            "backpropagation": 0.52, "softmax": 0.48, "sigmoid": 0.46,
            "stochastic": 0.50, "batch": 0.54, "epoch": 0.56,
            "inference": 0.59, "deployment": 0.44, "pipeline": 0.51,
            "tabular": 0.47, "vision": 0.49, "language": 0.55,
        }
        viz = WordCloudVisualizer(
            words=nlp_fallback,
            max_words=80,
            title="Machine Learning Vocabulary",
            width=900, height=550,
        )
    save_viz(viz, "word_cloud", 1000, 670)
    count += 1

    # ===== EXTENDED CHARTS (Tier 4) =====
    print("\n--- Extended Charts (Tier 4) ---")

    # 25. Arc Diagram — wine feature correlations (13 well-named features)
    print("  Dataset: Wine (arc diagram)")
    corr_wn = np.corrcoef(X_wn.T)
    viz = ArcDiagramVisualizer.from_correlation_matrix(
        corr_wn, fn_wn, threshold=0.5,
        title="Wine Feature Correlations (|r| > 0.5)",
        width=950, height=450,
    )
    save_viz(viz, "arc_diagram", 1050, 570)
    count += 1

    # 26. Chord Diagram — iris 3-class confusion
    print("  Dataset: Iris (chord diagram)")
    cm_ir = confusion_matrix(y_ir_te, rf_ir.predict(X_ir_te))
    viz = ChordDiagramVisualizer.from_confusion_matrix(
        cm_ir, cn_ir,
        title="Iris Misclassification Flow",
        width=650, height=600,
    )
    save_viz(viz, "chord_diagram", 750, 720)
    count += 1

    # 27. Donut Chart — iris class distribution (3 balanced classes)
    print("  Dataset: Iris (donut chart)")
    unique, counts = np.unique(y_ir, return_counts=True)
    viz = DonutChartVisualizer(
        labels=[cn_ir[i] for i in unique],
        values=counts.tolist(),
        title="Iris Class Distribution",
        width=600, height=500,
    )
    save_viz(viz, "donut_chart", 700, 620)
    count += 1

    # 28. Flow Chart — synthetic ML pipeline
    print("  Data: ML pipeline flow chart")
    viz = FlowChartVisualizer.from_pipeline([
        ("Load Data", "sklearn.load_iris()"),
        ("Preprocess", "StandardScaler + PCA"),
        ("Split", "70/30 stratified"),
        ("Train", "RandomForest, SVM, GB"),
        ("Ensemble", "Hill Climbing"),
        ("Evaluate", "ROC, PR, Confusion Matrix"),
        ("Deploy", "ONNX export"),
    ], width=900, height=500)
    save_viz(viz, "flow_chart", 1000, 620)
    count += 1

    # 29. Network Diagram — synthetic Bayesian network
    print("  Data: Bayesian network (network diagram)")
    viz = NetworkDiagramVisualizer(
        nodes=[
            {"id": "Genetics", "label": "Genetics", "group": "cause"},
            {"id": "Diet", "label": "Diet", "group": "cause"},
            {"id": "Exercise", "label": "Exercise", "group": "cause"},
            {"id": "Blood Pressure", "label": "Blood Pressure", "group": "intermediate"},
            {"id": "Cholesterol", "label": "Cholesterol", "group": "intermediate"},
            {"id": "BMI", "label": "BMI", "group": "intermediate"},
            {"id": "Heart Disease", "label": "Heart Disease", "group": "outcome"},
            {"id": "Diabetes", "label": "Diabetes", "group": "outcome"},
        ],
        edges=[
            ("Genetics", "Blood Pressure"), ("Genetics", "Cholesterol"),
            ("Diet", "Cholesterol"), ("Diet", "BMI"),
            ("Exercise", "BMI"), ("Exercise", "Blood Pressure"),
            ("Blood Pressure", "Heart Disease"), ("Cholesterol", "Heart Disease"),
            ("BMI", "Diabetes"), ("BMI", "Heart Disease"),
            ("Cholesterol", "Diabetes"),
        ],
        directed=True, layout="hierarchical",
        title="Health Risk Bayesian Network",
        width=850, height=550,
    )
    save_viz(viz, "network_diagram", 950, 670)
    count += 1

    # 30. Nightingale Rose — imbalanced dataset metrics (high variation)
    print("  Dataset: Imbalanced synthetic (nightingale rose)")
    y_imb_pred = rf_imb.predict(X_imb_te)
    y_imb_proba = rf_imb.predict_proba(X_imb_te)[:, 1]
    tn = int(((y_imb_te == 0) & (y_imb_pred == 0)).sum())
    fp = int(((y_imb_te == 0) & (y_imb_pred == 1)).sum())
    specificity = tn / max(tn + fp, 1)
    rose_metrics = {
        "Accuracy": round(accuracy_score(y_imb_te, y_imb_pred), 3),
        "Bal. Accuracy": round(balanced_accuracy_score(y_imb_te, y_imb_pred), 3),
        "Precision": round(precision_score(y_imb_te, y_imb_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_imb_te, y_imb_pred), 3),
        "F1 Score": round(f1_score(y_imb_te, y_imb_pred), 3),
        "Specificity": round(specificity, 3),
        "AUC-ROC": round(roc_auc_score(y_imb_te, y_imb_proba), 3),
        "MCC": round(max(0, matthews_corrcoef(y_imb_te, y_imb_pred)), 3),
    }
    print(f"    Metrics: {rose_metrics}")
    viz = NightingaleRoseVisualizer.from_metrics(
        rose_metrics, width=700, height=650,
    )
    save_viz(viz, "nightingale_rose", 800, 770)
    count += 1

    # 31. Radial Bar — breast cancer top 15 features
    print("  Dataset: Breast Cancer (radial bar)")
    top_idx = np.argsort(rf_bc.feature_importances_)[::-1][:15]
    viz = RadialBarVisualizer(
        labels=[fn_bc[i] for i in top_idx],
        values=[float(rf_bc.feature_importances_[i]) for i in top_idx],
        sort=True,
        title="Top 15 Feature Importances",
        width=700, height=650,
    )
    save_viz(viz, "radial_bar", 800, 770)
    count += 1

    # 32. Spiral Plot — synthetic seasonal model performance (365 days)
    print("  Data: Synthetic daily performance (spiral)")
    np.random.seed(42)
    days = np.arange(365)
    seasonal = 0.92 + 0.03 * np.sin(2 * np.pi * days / 365) + np.random.randn(365) * 0.005
    seasonal = np.clip(seasonal, 0.85, 0.98)
    viz = SpiralPlotVisualizer.from_time_series(
        seasonal, title="Daily Model Accuracy (1 Year)",
        width=700, height=650,
    )
    save_viz(viz, "spiral_plot", 800, 770)
    count += 1

    # 33. Stream Graph — ensemble weight evolution with descriptive labels
    print("  Data: Ensemble weight evolution (stream graph)")
    viz = StreamGraphVisualizer(
        x=[f"Round {i}" for i in range(1, 11)],
        series={
            "XGBoost":     [35, 32, 30, 28, 25, 23, 22, 20, 19, 18],
            "LightGBM":    [25, 28, 30, 32, 33, 34, 35, 36, 37, 38],
            "CatBoost":    [20, 22, 23, 24, 25, 26, 26, 27, 27, 28],
            "Neural Net":  [10, 10, 9, 9, 10, 10, 10, 10, 10, 10],
            "Linear":      [10, 8, 8, 7, 7, 7, 7, 7, 7, 6],
        },
        baseline="center",
        title="Ensemble Weight (%) by Hill-Climbing Round",
        width=950, height=500,
    )
    save_viz(viz, "stream_graph", 1050, 620)
    count += 1

    # ===== GENERAL-PURPOSE CHARTS (Tier B) =====
    print("\n--- General-Purpose Charts (Tier B) ---")

    # 34. Ridgeline — iris features by species (shows separation patterns)
    print("  Dataset: Iris (ridgeline by species × feature)")
    ridgeline_data = {}
    features_to_show = [
        (2, "Petal Length"),
        (3, "Petal Width"),
        (0, "Sepal Length"),
    ]
    for fi, fname in features_to_show:
        for ci, cname in enumerate(cn_ir):
            ridgeline_data[f"{cname} — {fname}"] = X_ir[y_ir == ci, fi].tolist()
    viz = RidgelinePlotVisualizer(
        ridgeline_data,
        title="Iris Feature Distributions by Species",
        width=900, height=550,
    )
    save_viz(viz, "ridgeline_plot", 1000, 670)
    count += 1

    # 35. Bump Chart — 5-model ranking across 5 CV folds
    print("  Dataset: Breast Cancer (bump chart)")
    cv_5fold = {
        "Random Forest": cross_val_score(rf_bc, X_bc, y_bc, cv=5, scoring="accuracy").tolist(),
        "Logistic Reg.": cross_val_score(lr_bc, X_bc, y_bc, cv=5, scoring="accuracy").tolist(),
        "Grad. Boosting": cross_val_score(gb_bc, X_bc, y_bc, cv=5, scoring="accuracy").tolist(),
        "SVM": cross_val_score(svm_bc, X_bc, y_bc, cv=5, scoring="accuracy").tolist(),
        "Decision Tree": cross_val_score(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            X_bc, y_bc, cv=5, scoring="accuracy"
        ).tolist(),
    }
    viz = BumpChartVisualizer.from_cv_scores(cv_5fold, width=850, height=480)
    save_viz(viz, "bump_chart", 950, 600)
    count += 1

    # 36. Lollipop — wine feature importances
    print("  Dataset: Wine (lollipop)")
    viz = LollipopChartVisualizer.from_importances(
        rf_wn, feature_names=fn_wn, top_n=13,
        width=850, height=480,
    )
    save_viz(viz, "lollipop_chart", 950, 600)
    count += 1

    # 37. Dumbbell — overfitting model: train vs test gap on hard dataset
    print("  Dataset: Hard synthetic (dumbbell — overfitting check)")
    y_of_tr_pred = rf_overfit.predict(X_hard_tr)
    y_of_te_pred = rf_overfit.predict(X_hard_te)
    metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall"]
    train_metrics = [
        round(accuracy_score(y_hard_tr, y_of_tr_pred), 3),
        round(f1_score(y_hard_tr, y_of_tr_pred), 3),
        round(precision_score(y_hard_tr, y_of_tr_pred), 3),
        round(recall_score(y_hard_tr, y_of_tr_pred), 3),
    ]
    test_metrics = [
        round(accuracy_score(y_hard_te, y_of_te_pred), 3),
        round(f1_score(y_hard_te, y_of_te_pred), 3),
        round(precision_score(y_hard_te, y_of_te_pred), 3),
        round(recall_score(y_hard_te, y_of_te_pred), 3),
    ]
    print(f"    Train: {train_metrics}")
    print(f"    Test:  {test_metrics}")
    viz = DumbbellChartVisualizer(
        labels=metric_labels,
        values_start=train_metrics, values_end=test_metrics,
        start_label="Train", end_label="Test",
        title="Overfitting Check — Train vs Test Gap",
        width=850, height=450,
    )
    save_viz(viz, "dumbbell_chart", 950, 570)
    count += 1

    # 38. Funnel — realistic ML pipeline attrition
    print("  Data: Pipeline attrition (funnel)")
    viz = FunnelChartVisualizer.from_pipeline(
        stages=[
            "Raw Dataset", "After Cleaning", "After Feature Engineering",
            "After Feature Selection", "Training Samples", "Post-Augmentation",
        ],
        sample_counts=[50000, 47500, 47500, 47500, 33250, 42000],
        width=800, height=520,
    )
    save_viz(viz, "funnel_chart", 900, 640)
    count += 1

    # 39. Gauge — hard dataset AUC (moderate value shows gauge zones well)
    print("  Dataset: Hard synthetic (gauge)")
    hard_auc = roc_auc_score(y_hard_te, rf_hard.predict_proba(X_hard_te)[:, 1])
    print(f"    AUC: {hard_auc:.4f}")
    viz = GaugeChartVisualizer(
        value=round(hard_auc, 4), label="AUC-ROC",
        zones=[(0, 0.7, "#d62728"), (0.7, 0.85, "#ff7f0e"), (0.85, 1.0, "#2ca02c")],
        format_str=".1%",
        title="Random Forest — Noisy Classification",
        width=600, height=480,
    )
    save_viz(viz, "gauge_chart", 700, 600)
    count += 1

    # ===== DECISION TREE =====
    print("\n--- Decision Tree ---")

    # 40. Tree Visualizer — iris (3-class, interpretable feature names)
    print("  Dataset: Iris (decision tree)")
    viz = TreeVisualizer(dt_ir, feature_names=fn_ir, class_names=cn_ir)
    save_viz(viz, "decision_tree", 1200, 780)
    count += 1

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {count} visualization images in {IMG_DIR}")

    generated = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))
    print(f"PNG files: {len(generated)}")
    total_bytes = 0
    for f in generated:
        size = os.path.getsize(os.path.join(IMG_DIR, f))
        total_bytes += size
        print(f"  {f}: {size:,} bytes")
    print(f"Total: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
