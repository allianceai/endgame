"""Tests for the endgame visualization suite.

Tests all chart types following the pattern from test_tree_visualization.py:
- File creation (save() returns valid Path, file exists)
- HTML content (contains chart-container, no CDN URLs)
- JSON validity (to_json() parses cleanly)
- Jupyter repr (_repr_html_() returns non-empty string)
- Edge cases (empty data, single point, large data)
- Classmethod constructors work with typical ML outputs
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from endgame.visualization import (
    BaseVisualizer,
    get_palette,
    CATEGORICAL,
    SEQUENTIAL,
    DIVERGING,
    BarChartVisualizer,
    HeatmapVisualizer,
    ConfusionMatrixVisualizer,
    HistogramVisualizer,
    LineChartVisualizer,
    ScatterplotVisualizer,
    BoxPlotVisualizer,
    ViolinPlotVisualizer,
    ErrorBarsVisualizer,
    ParallelCoordinatesVisualizer,
    RadarChartVisualizer,
    TreemapVisualizer,
    SunburstVisualizer,
    SankeyVisualizer,
    DotMatrixVisualizer,
    VennDiagramVisualizer,
    WordCloudVisualizer,
    ArcDiagramVisualizer,
    ChordDiagramVisualizer,
    DonutChartVisualizer,
    FlowChartVisualizer,
    NetworkDiagramVisualizer,
    NightingaleRoseVisualizer,
    RadialBarVisualizer,
    SpiralPlotVisualizer,
    StreamGraphVisualizer,
    ROCCurveVisualizer,
    PRCurveVisualizer,
    CalibrationPlotVisualizer,
    LiftChartVisualizer,
    PDPVisualizer,
    PDP2DVisualizer,
    WaterfallVisualizer,
    RidgelinePlotVisualizer,
    BumpChartVisualizer,
    LollipopChartVisualizer,
    DumbbellChartVisualizer,
    FunnelChartVisualizer,
    GaugeChartVisualizer,
    ClassificationReport,
    RegressionReport,
)


# ===== Fixtures =====

@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y, load_iris().feature_names, load_iris().target_names.tolist()


@pytest.fixture
def binary_clf(iris_data):
    X, y, _, _ = iris_data
    # Binary subset
    mask = y < 2
    X, y = X[mask], y[mask]
    clf = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
    return clf, X, y


@pytest.fixture
def fitted_rf(iris_data):
    X, y, _, _ = iris_data
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
    return rf


# ===== Helpers =====

def _check_html_basics(html: str):
    """Check common HTML properties for all charts."""
    assert len(html) > 500, "HTML too short"
    assert "chart-container" in html
    assert "cdn." not in html
    assert "unpkg.com" not in html
    assert "jsdelivr" not in html


def _check_save_and_html(viz, name="test"):
    """Common checks for save, to_json, and _repr_html_."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = viz.save(Path(tmpdir) / name)
        assert path.exists()
        assert path.suffix == ".html"
        content = path.read_text()
        _check_html_basics(content)

    j = viz.to_json()
    data = json.loads(j)
    assert isinstance(data, dict)

    html = viz._repr_html_()
    assert isinstance(html, str)
    _check_html_basics(html)


# ===== Palette Tests =====

class TestPalettes:
    def test_get_categorical(self):
        colors = get_palette("tableau")
        assert len(colors) == 10
        assert all(c.startswith("#") for c in colors)

    def test_get_sequential(self):
        colors = get_palette("blues")
        assert len(colors) == 9

    def test_get_diverging(self):
        colors = get_palette("rdbu")
        assert len(colors) == 11

    def test_get_n_colors(self):
        colors = get_palette("tableau", n=5)
        assert len(colors) == 5

    def test_unknown_palette(self):
        with pytest.raises(ValueError, match="Unknown palette"):
            get_palette("nonexistent")

    def test_all_palettes_exist(self):
        for name in list(CATEGORICAL) + list(SEQUENTIAL) + list(DIVERGING):
            colors = get_palette(name)
            assert len(colors) > 0


# ===== BarChart Tests =====

class TestBarChart:
    def test_basic(self):
        viz = BarChartVisualizer(
            labels=["a", "b", "c"],
            values=[10, 20, 30],
            title="Test Bar",
        )
        _check_save_and_html(viz, "bar_basic")

    def test_horizontal(self):
        viz = BarChartVisualizer(
            labels=["a", "b", "c"],
            values=[10, 20, 30],
            orientation="horizontal",
        )
        _check_save_and_html(viz, "bar_horiz")

    def test_stacked(self):
        viz = BarChartVisualizer(
            labels=["a", "b", "c"],
            values=[[10, 20, 30], [5, 10, 15]],
            series_names=["S1", "S2"],
        )
        _check_save_and_html(viz, "bar_stacked")

    def test_sorted(self):
        viz = BarChartVisualizer(
            labels=["c", "a", "b"],
            values=[30, 10, 20],
            sort=True,
        )
        data = json.loads(viz.to_json())
        # First value should be largest
        assert data["values"][0][0] >= data["values"][0][1]

    def test_from_dict(self):
        viz = BarChartVisualizer.from_dict({"x": 1, "y": 2, "z": 3})
        _check_save_and_html(viz, "bar_dict")

    def test_from_importances(self, fitted_rf, iris_data):
        _, _, fnames, _ = iris_data
        viz = BarChartVisualizer.from_importances(
            fitted_rf, feature_names=fnames, top_n=4,
        )
        _check_save_and_html(viz, "bar_imp")

    def test_single_bar(self):
        viz = BarChartVisualizer(labels=["only"], values=[42])
        _check_save_and_html(viz, "bar_single")

    def test_empty_labels(self):
        viz = BarChartVisualizer(labels=[], values=[])
        j = json.loads(viz.to_json())
        assert j["labels"] == []

    def test_light_theme(self):
        viz = BarChartVisualizer(
            labels=["a", "b"], values=[1, 2], theme="light",
        )
        _check_save_and_html(viz, "bar_light")


# ===== Heatmap Tests =====

class TestHeatmap:
    def test_basic(self):
        data = np.random.randn(5, 5)
        viz = HeatmapVisualizer(data, title="Test Heatmap")
        _check_save_and_html(viz, "heatmap_basic")

    def test_with_labels(self):
        data = np.eye(3)
        viz = HeatmapVisualizer(
            data,
            x_labels=["a", "b", "c"],
            y_labels=["x", "y", "z"],
        )
        _check_save_and_html(viz, "heatmap_labels")

    def test_from_correlation(self, iris_data):
        X, _, fnames, _ = iris_data
        viz = HeatmapVisualizer.from_correlation(X, feature_names=fnames)
        _check_save_and_html(viz, "heatmap_corr")
        data = json.loads(viz.to_json())
        assert data["vmin"] == -1.0
        assert data["vmax"] == 1.0

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            HeatmapVisualizer(np.array([1, 2, 3]))

    def test_nan_handling(self):
        data = np.array([[1.0, np.nan], [np.nan, 2.0]])
        viz = HeatmapVisualizer(data)
        j = json.loads(viz.to_json())
        assert j["matrix"][0][1] is None

    def test_custom_range(self):
        data = np.random.randn(3, 3)
        viz = HeatmapVisualizer(data, vmin=-2, vmax=2)
        j = json.loads(viz.to_json())
        assert j["vmin"] == -2
        assert j["vmax"] == 2


# ===== ConfusionMatrix Tests =====

class TestConfusionMatrix:
    def test_basic(self):
        cm = [[50, 3], [7, 40]]
        viz = ConfusionMatrixVisualizer(
            cm, class_names=["neg", "pos"],
        )
        _check_save_and_html(viz, "cm_basic")

    def test_3class(self):
        cm = np.array([[40, 5, 2], [3, 45, 1], [1, 2, 50]])
        viz = ConfusionMatrixVisualizer(cm, class_names=["a", "b", "c"])
        data = json.loads(viz.to_json())
        assert len(data["classNames"]) == 3
        assert "accuracy" in data
        assert "precision" in data

    def test_normalized(self):
        cm = [[50, 10], [5, 35]]
        viz = ConfusionMatrixVisualizer(cm, normalize=True)
        data = json.loads(viz.to_json())
        assert data["normalize"] is True

    def test_from_estimator(self, binary_clf):
        clf, X, y = binary_clf
        viz = ConfusionMatrixVisualizer.from_estimator(clf, X, y)
        _check_save_and_html(viz, "cm_estimator")

    def test_from_predictions(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        viz = ConfusionMatrixVisualizer.from_predictions(y_true, y_pred)
        _check_save_and_html(viz, "cm_pred")

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            ConfusionMatrixVisualizer(np.ones((2, 3)))


# ===== Histogram Tests =====

class TestHistogram:
    def test_basic(self):
        data = np.random.randn(500)
        viz = HistogramVisualizer(data, title="Test Histogram")
        _check_save_and_html(viz, "hist_basic")

    def test_with_kde(self):
        data = np.random.randn(200)
        viz = HistogramVisualizer(data, kde=True)
        j = json.loads(viz.to_json())
        assert j["kde"] is True
        assert len(j["series"][0]["kde"]) > 0

    def test_density_mode(self):
        data = np.random.randn(200)
        viz = HistogramVisualizer(data, density=True)
        j = json.loads(viz.to_json())
        assert j["density"] is True

    def test_multi_series(self):
        d1 = np.random.randn(200)
        d2 = np.random.randn(200) + 2
        viz = HistogramVisualizer([d1, d2], series_names=["Train", "Test"])
        j = json.loads(viz.to_json())
        assert len(j["series"]) == 2

    def test_custom_bins(self):
        data = np.random.randn(100)
        viz = HistogramVisualizer(data, bins=20)
        j = json.loads(viz.to_json())
        assert len(j["series"][0]["counts"]) == 20

    def test_from_residuals(self):
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        viz = HistogramVisualizer.from_residuals(y_true, y_pred)
        _check_save_and_html(viz, "hist_residuals")

    def test_from_predictions(self):
        proba = np.random.uniform(0, 1, 100)
        viz = HistogramVisualizer.from_predictions(proba)
        _check_save_and_html(viz, "hist_pred")

    def test_single_value(self):
        viz = HistogramVisualizer([5.0])
        j = json.loads(viz.to_json())
        assert len(j["series"]) == 1


# ===== LineChart Tests =====

class TestLineChart:
    def test_basic(self):
        viz = LineChartVisualizer(
            x=[1, 2, 3, 4, 5],
            series={"Model": [0.8, 0.85, 0.88, 0.90, 0.91]},
        )
        _check_save_and_html(viz, "line_basic")

    def test_multi_series(self):
        viz = LineChartVisualizer(
            x=[1, 2, 3],
            series={"A": [0.9, 0.91, 0.92], "B": [0.85, 0.87, 0.88]},
        )
        j = json.loads(viz.to_json())
        assert len(j["series"]) == 2

    def test_with_error_bands(self):
        viz = LineChartVisualizer(
            x=[1, 2, 3],
            series={"Train": [0.9, 0.92, 0.95]},
            error_bands={"Train": ([0.88, 0.90, 0.93], [0.92, 0.94, 0.97])},
        )
        j = json.loads(viz.to_json())
        assert "errorLo" in j["series"][0]

    def test_area_mode(self):
        viz = LineChartVisualizer(
            x=[1, 2, 3],
            series={"loss": [1.0, 0.5, 0.2]},
            area=True,
        )
        j = json.loads(viz.to_json())
        assert j["area"] is True

    def test_categorical_x(self):
        viz = LineChartVisualizer(
            x=["Fold 1", "Fold 2", "Fold 3"],
            series={"Model": [0.9, 0.91, 0.92]},
        )
        _check_save_and_html(viz, "line_categorical")

    def test_from_learning_curve(self):
        sizes = [50, 100, 150, 200]
        train = np.array([[0.95, 0.94], [0.93, 0.92], [0.92, 0.91], [0.91, 0.90]])
        test = np.array([[0.80, 0.79], [0.83, 0.82], [0.85, 0.84], [0.86, 0.85]])
        viz = LineChartVisualizer.from_learning_curve(sizes, train, test)
        j = json.loads(viz.to_json())
        assert len(j["series"]) == 2
        assert "errorLo" in j["series"][0]

    def test_from_cv_scores(self):
        scores = {"LGBM": [0.91, 0.92, 0.90], "XGB": [0.89, 0.91, 0.88]}
        viz = LineChartVisualizer.from_cv_scores(scores)
        _check_save_and_html(viz, "line_cv")


# ===== Scatterplot Tests =====

class TestScatterplot:
    def test_basic(self):
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.3
        viz = ScatterplotVisualizer(x, y)
        _check_save_and_html(viz, "scatter_basic")

    def test_with_labels(self):
        x = np.random.randn(100)
        y = np.random.randn(100)
        labels = np.array(["A"] * 50 + ["B"] * 50)
        viz = ScatterplotVisualizer(x, y, labels=labels)
        j = json.loads(viz.to_json())
        assert "uniqueLabels" in j

    def test_bubble_mode(self):
        x = np.random.randn(50)
        y = np.random.randn(50)
        sizes = np.abs(np.random.randn(50)) * 10
        viz = ScatterplotVisualizer(x, y, sizes=sizes)
        j = json.loads(viz.to_json())
        assert "sizes" in j

    def test_from_predictions(self):
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.2
        viz = ScatterplotVisualizer.from_predictions(y_true, y_pred)
        j = json.loads(viz.to_json())
        assert j["showDiagonal"] is True
        assert "regression" in j

    def test_from_embedding(self):
        emb = np.random.randn(200, 2)
        labels = np.array(["A"] * 100 + ["B"] * 100)
        viz = ScatterplotVisualizer.from_embedding(emb, labels=labels)
        _check_save_and_html(viz, "scatter_emb")

    def test_nan_filtering(self):
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([2.0, 3.0, np.nan])
        viz = ScatterplotVisualizer(x, y)
        j = json.loads(viz.to_json())
        assert len(j["x"]) == 1  # Only (1, 2) survives


# ===== BoxPlot Tests =====

class TestBoxPlot:
    def test_basic(self):
        data = {
            "Model A": np.random.randn(20).tolist(),
            "Model B": (np.random.randn(20) + 1).tolist(),
        }
        viz = BoxPlotVisualizer(data, title="Box Plot")
        _check_save_and_html(viz, "box_basic")

    def test_with_points(self):
        data = {"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]}
        viz = BoxPlotVisualizer(data, show_points=True)
        j = json.loads(viz.to_json())
        assert j["showPoints"] is True

    def test_from_cv_results(self):
        results = {
            "LGBM": [0.91, 0.92, 0.90, 0.93, 0.89],
            "XGB": [0.88, 0.89, 0.87, 0.90, 0.86],
        }
        viz = BoxPlotVisualizer.from_cv_results(results)
        _check_save_and_html(viz, "box_cv")

    def test_single_group(self):
        viz = BoxPlotVisualizer({"A": [1, 2, 3, 4, 5]})
        _check_save_and_html(viz, "box_single")

    def test_outlier_detection(self):
        data = {"A": [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]}
        viz = BoxPlotVisualizer(data)
        j = json.loads(viz.to_json())
        assert len(j["groups"][0]["outliers"]) > 0


# ===== ViolinPlot Tests =====

class TestViolinPlot:
    def test_basic(self):
        data = {
            "A": np.random.randn(100).tolist(),
            "B": (np.random.randn(100) + 1).tolist(),
        }
        viz = ViolinPlotVisualizer(data, title="Violin")
        _check_save_and_html(viz, "violin_basic")

    def test_kde_computed(self):
        data = {"A": np.random.randn(50).tolist()}
        viz = ViolinPlotVisualizer(data)
        j = json.loads(viz.to_json())
        assert len(j["groups"][0]["kde"]) > 0


# ===== ErrorBars Tests =====

class TestErrorBars:
    def test_basic(self):
        viz = ErrorBarsVisualizer(
            labels=["A", "B", "C"],
            means=[0.9, 0.85, 0.92],
            errors=[0.02, 0.03, 0.01],
        )
        _check_save_and_html(viz, "errbar_basic")

    def test_asymmetric(self):
        viz = ErrorBarsVisualizer(
            labels=["A", "B"],
            means=[0.9, 0.85],
            errors=[(0.01, 0.03), (0.02, 0.04)],
        )
        j = json.loads(viz.to_json())
        assert isinstance(j["errors"][0], list)

    def test_vertical(self):
        viz = ErrorBarsVisualizer(
            labels=["A", "B"],
            means=[0.9, 0.85],
            errors=[0.02, 0.03],
            orientation="vertical",
        )
        _check_save_and_html(viz, "errbar_vert")

    def test_from_cv_results(self):
        results = {
            "A": [0.91, 0.92, 0.90],
            "B": [0.88, 0.89, 0.87],
        }
        viz = ErrorBarsVisualizer.from_cv_results(results)
        _check_save_and_html(viz, "errbar_cv")


# ===== ParallelCoordinates Tests =====

class TestParallelCoordinates:
    def test_basic(self):
        data = [
            {"lr": 0.001, "depth": 6, "n_est": 100, "score": 0.92},
            {"lr": 0.01, "depth": 4, "n_est": 200, "score": 0.89},
            {"lr": 0.1, "depth": 3, "n_est": 50, "score": 0.85},
        ]
        viz = ParallelCoordinatesVisualizer(data, color_by="score")
        _check_save_and_html(viz, "parcoord_basic")

    def test_categorical_dims(self):
        data = [
            {"model": "lgbm", "lr": 0.01, "score": 0.9},
            {"model": "xgb", "lr": 0.1, "score": 0.85},
        ]
        viz = ParallelCoordinatesVisualizer(data)
        j = json.loads(viz.to_json())
        # "model" axis should be categorical
        model_axis = next(a for a in j["axes"] if a["name"] == "model")
        assert model_axis["type"] == "categorical"

    def test_custom_dimensions(self):
        data = [
            {"a": 1, "b": 2, "c": 3, "d": 4},
            {"a": 5, "b": 6, "c": 7, "d": 8},
        ]
        viz = ParallelCoordinatesVisualizer(data, dimensions=["a", "c"])
        j = json.loads(viz.to_json())
        assert len(j["axes"]) == 2


# ===== RadarChart Tests =====

class TestRadarChart:
    def test_basic(self):
        viz = RadarChartVisualizer(
            dimensions=["Acc", "Prec", "Rec", "F1", "AUC"],
            series={"Model": [0.92, 0.88, 0.95, 0.91, 0.96]},
        )
        _check_save_and_html(viz, "radar_basic")

    def test_multi_series(self):
        viz = RadarChartVisualizer(
            dimensions=["A", "B", "C", "D"],
            series={"M1": [0.9, 0.8, 0.7, 0.6], "M2": [0.7, 0.9, 0.8, 0.7]},
        )
        j = json.loads(viz.to_json())
        assert len(j["series"]) == 2

    def test_custom_ranges(self):
        viz = RadarChartVisualizer(
            dimensions=["A", "B", "C"],
            series={"M": [50, 75, 100]},
            ranges=[(0, 100), (0, 100), (0, 100)],
        )
        j = json.loads(viz.to_json())
        assert j["ranges"][0] == [0, 100]


# ===== Treemap Tests =====

class TestTreemap:
    def test_basic(self):
        viz = TreemapVisualizer(
            labels=["a", "b", "c", "d"],
            values=[40, 30, 20, 10],
        )
        _check_save_and_html(viz, "treemap_basic")

    def test_from_importances(self, fitted_rf, iris_data):
        _, _, fnames, _ = iris_data
        viz = TreemapVisualizer.from_importances(
            fitted_rf, feature_names=fnames,
        )
        _check_save_and_html(viz, "treemap_imp")

    def test_single_item(self):
        viz = TreemapVisualizer(labels=["only"], values=[100])
        _check_save_and_html(viz, "treemap_single")


# ===== Sunburst Tests =====

class TestSunburst:
    def test_basic(self):
        viz = SunburstVisualizer(
            labels=["Root", "A", "B", "A1", "A2"],
            parents=["", "Root", "Root", "A", "A"],
            values=[100, 60, 40, 30, 30],
        )
        _check_save_and_html(viz, "sunburst_basic")


# ===== Sankey Tests =====

class TestSankey:
    def test_basic(self):
        viz = SankeyVisualizer(
            nodes=["Train", "Valid", "Pass", "Fail"],
            links=[("Train", "Pass", 80), ("Train", "Fail", 20),
                   ("Valid", "Pass", 70), ("Valid", "Fail", 30)],
        )
        _check_save_and_html(viz, "sankey_basic")

    def test_single_link(self):
        viz = SankeyVisualizer(
            nodes=["A", "B"],
            links=[("A", "B", 100)],
        )
        _check_save_and_html(viz, "sankey_single")


# ===== DotMatrix Tests =====

class TestDotMatrix:
    def test_basic(self):
        viz = DotMatrixVisualizer(
            labels=["Correct", "Incorrect"],
            values=[85, 15],
        )
        _check_save_and_html(viz, "dotmatrix_basic")

    def test_three_categories(self):
        viz = DotMatrixVisualizer(
            labels=["A", "B", "C"],
            values=[50, 30, 20],
            n_dots=100,
        )
        j = json.loads(viz.to_json())
        assert sum(j["counts"]) == 100

    def test_square_dots(self):
        viz = DotMatrixVisualizer(
            labels=["X", "Y"],
            values=[60, 40],
            dot_shape="square",
        )
        j = json.loads(viz.to_json())
        assert j["shape"] == "square"


# ===== VennDiagram Tests =====

class TestVennDiagram:
    def test_two_sets(self):
        viz = VennDiagramVisualizer(
            sets={"A": 100, "B": 80},
            intersections={"A&B": 30},
        )
        _check_save_and_html(viz, "venn_basic")

    def test_three_sets(self):
        viz = VennDiagramVisualizer(
            sets={"A": 100, "B": 80, "C": 60},
            intersections={"A&B": 30, "A&C": 20, "B&C": 15, "A&B&C": 10},
        )
        _check_save_and_html(viz, "venn_three")


# ===== WordCloud Tests =====

class TestWordCloud:
    def test_basic(self):
        words = {"accuracy": 50, "precision": 35, "recall": 40, "f1": 45,
                 "auc": 30, "logloss": 25, "rmse": 20}
        viz = WordCloudVisualizer(words)
        _check_save_and_html(viz, "wc_basic")

    def test_from_feature_names(self):
        names = [f"feature_{i}" for i in range(20)]
        importances = np.random.uniform(0, 1, 20)
        viz = WordCloudVisualizer.from_feature_names(names, importances)
        _check_save_and_html(viz, "wc_features")

    def test_max_words(self):
        words = {f"word_{i}": i for i in range(200)}
        viz = WordCloudVisualizer(words, max_words=50)
        j = json.loads(viz.to_json())
        assert len(j["words"]) == 50


# ===== Theme Tests =====

# ===== ArcDiagram Tests =====

class TestArcDiagram:
    def test_basic(self):
        viz = ArcDiagramVisualizer(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B", 0.9), ("B", "C", 0.7), ("A", "D", 0.5)],
        )
        _check_save_and_html(viz, "arc_basic")

    def test_from_correlation(self):
        corr = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.5], [0.1, 0.5, 1.0]])
        viz = ArcDiagramVisualizer.from_correlation_matrix(
            corr, ["f1", "f2", "f3"], threshold=0.4,
        )
        j = json.loads(viz.to_json())
        assert len(j["edges"]) == 2  # only f1-f2 (0.8) and f2-f3 (0.5)

    def test_sort_by_degree(self):
        viz = ArcDiagramVisualizer(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 1), ("A", "C", 1)],
            sort_by="degree",
        )
        j = json.loads(viz.to_json())
        assert j["nodes"][0] == "A"  # highest degree

    def test_empty_edges(self):
        viz = ArcDiagramVisualizer(nodes=["A", "B"], edges=[])
        _check_save_and_html(viz, "arc_empty")


# ===== ChordDiagram Tests =====

class TestChordDiagram:
    def test_basic(self):
        matrix = np.array([[0, 5, 3], [5, 0, 4], [3, 4, 0]])
        viz = ChordDiagramVisualizer(matrix, labels=["A", "B", "C"])
        _check_save_and_html(viz, "chord_basic")

    def test_asymmetric(self):
        matrix = np.array([[0, 10, 2], [3, 0, 8], [1, 4, 0]])
        viz = ChordDiagramVisualizer(matrix, labels=["X", "Y", "Z"])
        j = json.loads(viz.to_json())
        assert len(j["groups"]) == 3
        assert len(j["chords"]) > 0

    def test_from_confusion(self):
        cm = np.array([[40, 5], [10, 45]])
        viz = ChordDiagramVisualizer.from_confusion_matrix(cm, ["neg", "pos"])
        _check_save_and_html(viz, "chord_cm")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            ChordDiagramVisualizer(np.array([1, 2]), ["A"])


# ===== DonutChart Tests =====

class TestDonutChart:
    def test_basic(self):
        viz = DonutChartVisualizer(
            labels=["A", "B", "C"],
            values=[45, 35, 20],
            title="Test Donut",
        )
        _check_save_and_html(viz, "donut_basic")

    def test_with_center_text(self):
        viz = DonutChartVisualizer(
            labels=["Pass", "Fail"],
            values=[85, 15],
            center_text="85%",
        )
        j = json.loads(viz.to_json())
        assert j["centerText"] == "85%"

    def test_from_class_distribution(self):
        y = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        viz = DonutChartVisualizer.from_class_distribution(y)
        j = json.loads(viz.to_json())
        assert len(j["slices"]) == 3

    def test_pie_mode(self):
        viz = DonutChartVisualizer(
            labels=["A", "B"], values=[60, 40],
            inner_radius_ratio=0,
        )
        _check_save_and_html(viz, "donut_pie")


# ===== FlowChart Tests =====

class TestFlowChart:
    def test_basic(self):
        viz = FlowChartVisualizer(
            nodes=[
                {"id": "a", "label": "Data", "type": "input"},
                {"id": "b", "label": "Preprocess", "type": "process"},
                {"id": "c", "label": "Model", "type": "process"},
                {"id": "d", "label": "Output", "type": "output"},
            ],
            edges=[("a", "b"), ("b", "c"), ("c", "d")],
        )
        _check_save_and_html(viz, "flow_basic")

    def test_from_pipeline(self):
        steps = [
            ("Load Data", "CSV loader"),
            ("Clean", "Handle missing"),
            ("Feature Eng", "Create features"),
            ("Train", "LightGBM"),
            ("Predict", "Generate output"),
        ]
        viz = FlowChartVisualizer.from_pipeline(steps)
        j = json.loads(viz.to_json())
        assert len(j["nodes"]) == 5
        assert len(j["edges"]) == 4

    def test_top_bottom(self):
        viz = FlowChartVisualizer(
            nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            edges=[("a", "b")],
            direction="TB",
        )
        _check_save_and_html(viz, "flow_tb")

    def test_edge_labels(self):
        viz = FlowChartVisualizer(
            nodes=[{"id": "a", "label": "Start"}, {"id": "b", "label": "End"}],
            edges=[("a", "b", "yes")],
        )
        j = json.loads(viz.to_json())
        assert j["edges"][0]["label"] == "yes"


# ===== NetworkDiagram Tests =====

class TestNetworkDiagram:
    def test_basic(self):
        viz = NetworkDiagramVisualizer(
            nodes=["Rain", "Sprinkler", "Wet Grass"],
            edges=[("Rain", "Wet Grass"), ("Sprinkler", "Wet Grass"),
                   ("Rain", "Sprinkler")],
            title="Bayesian Network",
        )
        _check_save_and_html(viz, "network_basic")

    def test_undirected(self):
        viz = NetworkDiagramVisualizer(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
            directed=False,
        )
        j = json.loads(viz.to_json())
        assert j["directed"] is False

    def test_weighted_edges(self):
        viz = NetworkDiagramVisualizer(
            nodes=["X", "Y", "Z"],
            edges=[("X", "Y", 0.9), ("Y", "Z", 0.5)],
        )
        j = json.loads(viz.to_json())
        assert j["edges"][0]["weight"] == 0.9

    def test_from_adjacency_matrix(self):
        matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        viz = NetworkDiagramVisualizer.from_adjacency_matrix(
            matrix, ["A", "B", "C"],
        )
        j = json.loads(viz.to_json())
        assert len(j["edges"]) == 3

    def test_from_edge_list(self):
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]
        viz = NetworkDiagramVisualizer.from_edge_list(edges)
        j = json.loads(viz.to_json())
        assert len(j["nodes"]) == 4
        assert len(j["edges"]) == 4

    def test_hierarchical_layout(self):
        viz = NetworkDiagramVisualizer(
            nodes=["root", "child1", "child2", "leaf"],
            edges=[("root", "child1"), ("root", "child2"), ("child1", "leaf")],
            layout="hierarchical",
        )
        j = json.loads(viz.to_json())
        assert j["layout"] == "hierarchical"
        # root should be at leftmost position
        assert j["positions"][0]["x"] < j["positions"][3]["x"]

    def test_circular_layout(self):
        viz = NetworkDiagramVisualizer(
            nodes=["A", "B", "C", "D"],
            edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
            layout="circular",
        )
        _check_save_and_html(viz, "network_circular")

    def test_force_layout(self):
        viz = NetworkDiagramVisualizer(
            nodes=["A", "B", "C"],
            edges=[("A", "B"), ("B", "C")],
            layout="force",
        )
        _check_save_and_html(viz, "network_force")

    def test_dict_nodes(self):
        viz = NetworkDiagramVisualizer(
            nodes=[
                {"id": "X1", "label": "Feature 1", "group": "features"},
                {"id": "X2", "label": "Feature 2", "group": "features"},
                {"id": "Y", "label": "Target", "group": "target"},
            ],
            edges=[("X1", "Y"), ("X2", "Y")],
        )
        _check_save_and_html(viz, "network_dict")


# ===== NightingaleRose Tests =====

class TestNightingaleRose:
    def test_basic(self):
        viz = NightingaleRoseVisualizer(
            labels=["Acc", "Prec", "Rec", "F1", "AUC"],
            values=[0.95, 0.88, 0.92, 0.90, 0.97],
        )
        _check_save_and_html(viz, "rose_basic")

    def test_from_metrics(self):
        metrics = {"accuracy": 0.92, "precision": 0.88, "recall": 0.95,
                   "f1": 0.91, "auc": 0.96}
        viz = NightingaleRoseVisualizer.from_metrics(metrics)
        j = json.loads(viz.to_json())
        assert len(j["series"][0]["values"]) == 5

    def test_multi_series(self):
        viz = NightingaleRoseVisualizer(
            labels=["A", "B", "C", "D"],
            values={"Model 1": [0.9, 0.8, 0.7, 0.6],
                    "Model 2": [0.7, 0.9, 0.8, 0.7]},
        )
        j = json.loads(viz.to_json())
        assert len(j["series"]) == 2


# ===== RadialBar Tests =====

class TestRadialBar:
    def test_basic(self):
        viz = RadialBarVisualizer(
            labels=["A", "B", "C", "D", "E", "F"],
            values=[0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        )
        _check_save_and_html(viz, "radbar_basic")

    def test_sorted(self):
        viz = RadialBarVisualizer(
            labels=["C", "A", "B"],
            values=[0.7, 0.9, 0.8],
            sort=True,
        )
        j = json.loads(viz.to_json())
        assert j["values"][0] >= j["values"][1]

    def test_single_bar(self):
        viz = RadialBarVisualizer(labels=["Only"], values=[1.0])
        _check_save_and_html(viz, "radbar_single")


# ===== SpiralPlot Tests =====

class TestSpiralPlot:
    def test_basic(self):
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + 1.5
        viz = SpiralPlotVisualizer(values, title="Spiral")
        _check_save_and_html(viz, "spiral_basic")

    def test_with_labels(self):
        values = [1, 2, 3, 4, 5]
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        viz = SpiralPlotVisualizer(values, labels=labels)
        j = json.loads(viz.to_json())
        assert j["points"][0]["label"] == "Mon"

    def test_from_time_series(self):
        values = np.random.randn(50).cumsum()
        viz = SpiralPlotVisualizer.from_time_series(values)
        _check_save_and_html(viz, "spiral_ts")

    def test_nan_handling(self):
        values = [1.0, np.nan, 3.0, 4.0]
        viz = SpiralPlotVisualizer(values)
        j = json.loads(viz.to_json())
        assert j["points"][1]["value"] is None


# ===== StreamGraph Tests =====

class TestStreamGraph:
    def test_basic(self):
        viz = StreamGraphVisualizer(
            x=[1, 2, 3, 4, 5],
            series={"A": [5, 8, 12, 15, 18],
                    "B": [3, 5, 7, 8, 10],
                    "C": [4, 6, 9, 11, 14]},
        )
        _check_save_and_html(viz, "stream_basic")

    def test_zero_baseline(self):
        viz = StreamGraphVisualizer(
            x=[1, 2, 3],
            series={"X": [1, 2, 3], "Y": [3, 2, 1]},
            baseline="zero",
        )
        j = json.loads(viz.to_json())
        # With zero baseline, first layer y0 should be 0
        assert j["layers"][0]["y0"][0] == 0.0

    def test_center_baseline(self):
        viz = StreamGraphVisualizer(
            x=[1, 2, 3],
            series={"A": [2, 4, 6]},
            baseline="center",
        )
        j = json.loads(viz.to_json())
        # Center baseline should have negative y0
        assert j["layers"][0]["y0"][0] < 0

    def test_categorical_x(self):
        viz = StreamGraphVisualizer(
            x=["Q1", "Q2", "Q3", "Q4"],
            series={"Revenue": [100, 150, 200, 180]},
        )
        _check_save_and_html(viz, "stream_cat")

    def test_single_series(self):
        viz = StreamGraphVisualizer(
            x=[1, 2, 3],
            series={"Only": [10, 20, 30]},
        )
        _check_save_and_html(viz, "stream_single")


# ===== Theme Tests =====

class TestThemes:
    def test_invalid_theme(self):
        with pytest.raises(ValueError, match="theme must be"):
            BarChartVisualizer(labels=["a"], values=[1], theme="invalid")

    def test_dark_theme_default(self):
        viz = BarChartVisualizer(labels=["a"], values=[1])
        html = viz._repr_html_()
        assert 'data-theme="dark"' in html

    def test_light_theme(self):
        viz = BarChartVisualizer(labels=["a"], values=[1], theme="light")
        html = viz._repr_html_()
        assert 'data-theme="light"' in html


# ===== ROC Curve Tests =====

class TestROCCurve:
    def test_basic(self):
        curves = [{
            "fpr": [0.0, 0.1, 0.3, 0.5, 1.0],
            "tpr": [0.0, 0.5, 0.7, 0.9, 1.0],
            "auc": 0.85,
            "label": "Model (AUC = 0.850)",
        }]
        viz = ROCCurveVisualizer(curves)
        _check_save_and_html(viz, "roc_basic")

    def test_from_estimator_binary(self, binary_clf):
        clf, X, y = binary_clf
        viz = ROCCurveVisualizer.from_estimator(clf, X, y)
        j = json.loads(viz.to_json())
        assert len(j["curves"]) == 1
        assert j["curves"][0]["auc"] > 0.5
        assert "optimalPoint" in j["curves"][0]
        _check_save_and_html(viz, "roc_estimator")

    def test_from_estimator_multiclass(self, iris_data):
        X, y, _, cnames = iris_data
        clf = LogisticRegression(max_iter=300, random_state=42).fit(X, y)
        viz = ROCCurveVisualizer.from_estimator(clf, X, y, class_names=cnames)
        j = json.loads(viz.to_json())
        # 3 classes + micro-average
        assert len(j["curves"]) == 4
        _check_save_and_html(viz, "roc_multi")

    def test_from_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.2])
        viz = ROCCurveVisualizer.from_predictions(y_true, y_score, label="Test")
        j = json.loads(viz.to_json())
        assert j["curves"][0]["auc"] > 0.5
        assert "optimalPoint" in j["curves"][0]
        _check_save_and_html(viz, "roc_pred")

    def test_multiple_curves(self):
        curves = [
            {"fpr": [0, 0.2, 1], "tpr": [0, 0.8, 1], "auc": 0.9, "label": "A"},
            {"fpr": [0, 0.3, 1], "tpr": [0, 0.7, 1], "auc": 0.8, "label": "B"},
        ]
        viz = ROCCurveVisualizer(curves)
        j = json.loads(viz.to_json())
        assert len(j["curves"]) == 2


# ===== PR Curve Tests =====

class TestPRCurve:
    def test_basic(self):
        curves = [{
            "precision": [1.0, 0.9, 0.8, 0.7],
            "recall": [0.0, 0.5, 0.8, 1.0],
            "ap": 0.85,
            "label": "Model (AP = 0.850)",
        }]
        viz = PRCurveVisualizer(curves)
        _check_save_and_html(viz, "pr_basic")

    def test_from_estimator(self, binary_clf):
        clf, X, y = binary_clf
        viz = PRCurveVisualizer.from_estimator(clf, X, y)
        j = json.loads(viz.to_json())
        assert len(j["curves"]) == 1
        assert j["curves"][0]["ap"] > 0.5
        assert j["prevalence"] is not None
        _check_save_and_html(viz, "pr_estimator")

    def test_from_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6])
        viz = PRCurveVisualizer.from_predictions(y_true, y_score, label="XGB")
        j = json.loads(viz.to_json())
        assert "optimalPoint" in j["curves"][0]
        assert j["curves"][0]["optimalPoint"]["f1"] > 0
        _check_save_and_html(viz, "pr_pred")

    def test_multiclass(self, iris_data):
        X, y, _, cnames = iris_data
        clf = LogisticRegression(max_iter=300, random_state=42).fit(X, y)
        viz = PRCurveVisualizer.from_estimator(clf, X, y, class_names=cnames)
        j = json.loads(viz.to_json())
        assert len(j["curves"]) == 3  # one per class


# ===== Calibration Plot Tests =====

class TestCalibrationPlot:
    def test_basic(self):
        curves = [{
            "probTrue": [0.1, 0.3, 0.5, 0.7, 0.9],
            "probPred": [0.1, 0.3, 0.5, 0.7, 0.9],
            "counts": [20, 20, 20, 20, 20],
            "histBins": [0.1, 0.3, 0.5, 0.7, 0.9],
            "ece": 0.0,
            "mce": 0.0,
            "label": "Perfect",
        }]
        viz = CalibrationPlotVisualizer(curves)
        _check_save_and_html(viz, "cal_basic")

    def test_from_estimator(self, binary_clf):
        clf, X, y = binary_clf
        viz = CalibrationPlotVisualizer.from_estimator(clf, X, y, n_bins=5)
        j = json.loads(viz.to_json())
        assert len(j["curves"]) == 1
        assert "ece" in j["curves"][0]
        assert "mce" in j["curves"][0]
        _check_save_and_html(viz, "cal_estimator")

    def test_from_predictions(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.4, 200)
        y_prob = np.clip(y_true + np.random.randn(200) * 0.3, 0, 1)
        viz = CalibrationPlotVisualizer.from_predictions(y_true, y_prob, n_bins=10)
        j = json.loads(viz.to_json())
        assert j["curves"][0]["ece"] >= 0
        _check_save_and_html(viz, "cal_pred")

    def test_from_multiple(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.4, 200)
        preds = {
            "Model A": np.clip(y_true + np.random.randn(200) * 0.2, 0, 1),
            "Model B": np.clip(y_true + np.random.randn(200) * 0.5, 0, 1),
        }
        viz = CalibrationPlotVisualizer.from_multiple(y_true, preds, n_bins=5)
        j = json.loads(viz.to_json())
        assert len(j["curves"]) == 2
        _check_save_and_html(viz, "cal_multi")


# ===== Lift Chart Tests =====

class TestLiftChart:
    def test_basic(self):
        curves = [{
            "percentiles": [0.1, 0.2, 0.5, 1.0],
            "gains": [0.3, 0.5, 0.8, 1.0],
            "lift": [3.0, 2.5, 1.6, 1.0],
            "label": "Model",
        }]
        viz = LiftChartVisualizer(curves)
        _check_save_and_html(viz, "lift_basic")

    def test_from_predictions(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 500)
        y_score = y_true * 0.7 + np.random.uniform(0, 0.3, 500)
        viz = LiftChartVisualizer.from_predictions(y_true, y_score)
        j = json.loads(viz.to_json())
        # Gains should end at 1.0
        assert abs(j["curves"][0]["gains"][-1] - 1.0) < 0.02
        # Initial lift should be > 1
        assert j["curves"][0]["lift"][0] > 1.0
        _check_save_and_html(viz, "lift_pred")

    def test_from_estimator(self, binary_clf):
        clf, X, y = binary_clf
        viz = LiftChartVisualizer.from_estimator(clf, X, y)
        _check_save_and_html(viz, "lift_est")

    def test_gains_only_mode(self):
        curves = [{
            "percentiles": [0.25, 0.5, 0.75, 1.0],
            "gains": [0.6, 0.8, 0.95, 1.0],
            "lift": [2.4, 1.6, 1.27, 1.0],
            "label": "Test",
        }]
        viz = LiftChartVisualizer(curves, mode="gains")
        j = json.loads(viz.to_json())
        assert j["mode"] == "gains"
        _check_save_and_html(viz, "lift_gains")

    def test_lift_only_mode(self):
        curves = [{
            "percentiles": [0.25, 0.5, 0.75, 1.0],
            "gains": [0.6, 0.8, 0.95, 1.0],
            "lift": [2.4, 1.6, 1.27, 1.0],
            "label": "Test",
        }]
        viz = LiftChartVisualizer(curves, mode="lift")
        j = json.loads(viz.to_json())
        assert j["mode"] == "lift"


# ===== PDP / ICE Tests =====

class TestPDP:
    def test_basic(self):
        viz = PDPVisualizer(
            grid_values=[0, 1, 2, 3, 4],
            pdp_values=[0.2, 0.3, 0.5, 0.6, 0.7],
            feature_name="feature_0",
        )
        _check_save_and_html(viz, "pdp_basic")

    def test_with_ice(self):
        grid = [0, 1, 2, 3, 4]
        pdp = [0.2, 0.3, 0.5, 0.6, 0.7]
        ice = [[0.1, 0.25, 0.4, 0.55, 0.65],
               [0.3, 0.35, 0.6, 0.65, 0.75]]
        viz = PDPVisualizer(grid, pdp, ice_lines=ice, feature_name="test_feat")
        j = json.loads(viz.to_json())
        assert j["ice"] is not None
        assert len(j["ice"]) == 2
        _check_save_and_html(viz, "pdp_ice")

    def test_from_estimator(self):
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
        viz = PDPVisualizer.from_estimator(clf, X, feature=0, n_ice_lines=10, grid_resolution=20)
        j = json.loads(viz.to_json())
        assert len(j["grid"]) == 20
        assert len(j["pdp"]) == 20
        assert j["ice"] is not None
        _check_save_and_html(viz, "pdp_estimator")

    def test_from_precomputed(self):
        viz = PDPVisualizer.from_precomputed(
            [0, 0.5, 1.0], [0.3, 0.5, 0.7], feature_name="x1",
        )
        j = json.loads(viz.to_json())
        assert j["featureName"] == "x1"

    def test_categorical_feature(self):
        viz = PDPVisualizer(
            grid_values=[0, 1, 2],
            pdp_values=[0.3, 0.6, 0.8],
            feature_name="color",
            is_categorical=True,
        )
        j = json.loads(viz.to_json())
        assert j["isCategorical"] is True

    def test_no_ice(self):
        np.random.seed(42)
        X, y = make_classification(n_samples=60, n_features=5, random_state=42)
        clf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42).fit(X, y)
        viz = PDPVisualizer.from_estimator(clf, X, feature=0, ice=False, grid_resolution=10)
        j = json.loads(viz.to_json())
        assert j["ice"] is None


# ===== PDP2D Tests =====

class TestPDP2D:
    def test_basic(self):
        grid_x = [0, 1, 2]
        grid_y = [0, 1, 2]
        values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        viz = PDP2DVisualizer(grid_x, grid_y, values, feature_x="A", feature_y="B")
        _check_save_and_html(viz, "pdp2d_basic")

    def test_from_estimator(self):
        np.random.seed(42)
        X, y = make_classification(n_samples=80, n_features=5, random_state=42)
        clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42).fit(X, y)
        viz = PDP2DVisualizer.from_estimator(clf, X, features=(0, 1), grid_resolution=8)
        j = json.loads(viz.to_json())
        assert len(j["gridX"]) == 8
        assert len(j["gridY"]) == 8
        assert len(j["values"]) == 8
        _check_save_and_html(viz, "pdp2d_estimator")


# ===== Waterfall Chart Tests =====

class TestWaterfallChart:
    def test_basic(self):
        viz = WaterfallVisualizer(
            categories=["Revenue", "COGS", "Tax", "Other"],
            values=[100, -40, -15, -5],
            base_value=0,
        )
        _check_save_and_html(viz, "waterfall_basic")

    def test_from_shap(self):
        np.random.seed(42)
        shap_values = np.array([0.3, -0.2, 0.15, -0.1, 0.05])
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        viz = WaterfallVisualizer.from_shap(shap_values, feature_names, base_value=0.5)
        j = json.loads(viz.to_json())
        assert j["baseValue"] == 0.5
        assert abs(j["finalValue"] - (0.5 + sum(shap_values))) < 1e-4
        _check_save_and_html(viz, "waterfall_shap")

    def test_from_contributions(self):
        viz = WaterfallVisualizer.from_contributions(
            ["Start", "Step 1", "Step 2", "Step 3"],
            [50, -20, 30, -10],
            base_value=100,
        )
        j = json.loads(viz.to_json())
        assert j["baseValue"] == 100
        assert j["finalValue"] == 100 + 50 - 20 + 30 - 10
        _check_save_and_html(viz, "waterfall_contrib")

    def test_sort_by_abs(self):
        viz = WaterfallVisualizer(
            categories=["a", "b", "c"],
            values=[0.1, -0.5, 0.3],
            sort_by="abs",
        )
        j = json.loads(viz.to_json())
        # First value should have largest absolute magnitude
        assert abs(j["values"][0]) >= abs(j["values"][-1])

    def test_max_display(self):
        cats = [f"f{i}" for i in range(20)]
        vals = np.random.randn(20).tolist()
        viz = WaterfallVisualizer(cats, vals, max_display=5)
        j = json.loads(viz.to_json())
        # 5 displayed + 1 "Other" = 6
        assert len(j["categories"]) == 6
        assert "Other" in j["categories"][-1]

    def test_connectors_off(self):
        viz = WaterfallVisualizer(
            categories=["a", "b"], values=[1, -0.5],
            show_connectors=False,
        )
        j = json.loads(viz.to_json())
        assert j["showConnectors"] is False

    def test_shap_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            WaterfallVisualizer.from_shap([0.1, 0.2], ["a", "b", "c"], base_value=0)


# ===== Ridgeline Plot Tests =====

class TestRidgelinePlot:
    def test_basic(self):
        data = {
            "Model A": np.random.randn(100).tolist(),
            "Model B": (np.random.randn(100) + 1).tolist(),
            "Model C": (np.random.randn(100) - 0.5).tolist(),
        }
        viz = RidgelinePlotVisualizer(data)
        _check_save_and_html(viz, "ridge_basic")

    def test_from_cv_results(self):
        results = {
            "LGBM": [0.91, 0.92, 0.90, 0.93, 0.89],
            "XGB": [0.88, 0.89, 0.87, 0.90, 0.86],
            "CatBoost": [0.90, 0.91, 0.89, 0.92, 0.88],
        }
        viz = RidgelinePlotVisualizer.from_cv_results(results)
        _check_save_and_html(viz, "ridge_cv")

    def test_from_feature_distributions(self, iris_data):
        X, _, fnames, _ = iris_data
        viz = RidgelinePlotVisualizer.from_feature_distributions(
            X, feature_names=fnames, max_features=4,
        )
        j = json.loads(viz.to_json())
        assert len(j["ridges"]) == 4

    def test_overlap_parameter(self):
        data = {"A": np.random.randn(50).tolist(), "B": np.random.randn(50).tolist()}
        viz = RidgelinePlotVisualizer(data, overlap=0.8)
        j = json.loads(viz.to_json())
        assert j["overlap"] == 0.8

    def test_single_group(self):
        viz = RidgelinePlotVisualizer({"Only": [1, 2, 3, 4, 5]})
        _check_save_and_html(viz, "ridge_single")

    def test_quantiles_shown(self):
        viz = RidgelinePlotVisualizer(
            {"A": np.random.randn(100).tolist()},
            show_quantiles=True,
        )
        j = json.loads(viz.to_json())
        assert j["showQuantiles"] is True
        assert "median" in j["ridges"][0]
        assert "q1" in j["ridges"][0]
        assert "q3" in j["ridges"][0]

    def test_empty_group_skipped(self):
        data = {"Good": [1, 2, 3], "Empty": []}
        viz = RidgelinePlotVisualizer(data)
        j = json.loads(viz.to_json())
        assert len(j["ridges"]) == 1


# ===== Bump Chart Tests =====

class TestBumpChart:
    def test_basic(self):
        rankings = {
            "LGBM": [1, 1, 2, 1],
            "XGBoost": [2, 3, 1, 2],
            "CatBoost": [3, 2, 3, 3],
        }
        viz = BumpChartVisualizer(
            x=["Fold 1", "Fold 2", "Fold 3", "Fold 4"],
            rankings=rankings,
        )
        _check_save_and_html(viz, "bump_basic")

    def test_from_scores(self):
        scores = {
            "A": [0.91, 0.89, 0.93],
            "B": [0.88, 0.92, 0.90],
            "C": [0.85, 0.87, 0.88],
        }
        viz = BumpChartVisualizer.from_scores(["R1", "R2", "R3"], scores)
        j = json.loads(viz.to_json())
        # At R1: A best (rank 1), B second, C third
        a_ranks = next(s for s in j["series"] if s["name"] == "A")
        assert a_ranks["ranks"][0] == 1

    def test_from_cv_scores(self):
        scores = {
            "LGBM": [0.91, 0.92, 0.90, 0.93, 0.89],
            "XGB": [0.88, 0.93, 0.87, 0.90, 0.86],
        }
        viz = BumpChartVisualizer.from_cv_scores(scores)
        j = json.loads(viz.to_json())
        assert len(j["x"]) == 5
        assert j["x"][0] == "Fold 1"

    def test_two_series(self):
        viz = BumpChartVisualizer(
            x=["A", "B"], rankings={"X": [1, 2], "Y": [2, 1]},
        )
        j = json.loads(viz.to_json())
        assert j["maxRank"] == 2

    def test_lower_is_better(self):
        scores = {"M1": [0.1, 0.2], "M2": [0.3, 0.05]}
        viz = BumpChartVisualizer.from_scores(
            ["T1", "T2"], scores, higher_is_better=False,
        )
        j = json.loads(viz.to_json())
        m1_ranks = next(s for s in j["series"] if s["name"] == "M1")
        assert m1_ranks["ranks"][0] == 1  # 0.1 < 0.3 → rank 1


# ===== Lollipop Chart Tests =====

class TestLollipopChart:
    def test_basic(self):
        viz = LollipopChartVisualizer(
            labels=["a", "b", "c", "d"],
            values=[0.5, 0.3, 0.8, 0.2],
        )
        _check_save_and_html(viz, "lollipop_basic")

    def test_sorted(self):
        viz = LollipopChartVisualizer(
            labels=["a", "b", "c"],
            values=[0.3, 0.8, 0.5],
            sort=True,
        )
        j = json.loads(viz.to_json())
        assert j["values"][0] >= j["values"][1] >= j["values"][2]

    def test_vertical(self):
        viz = LollipopChartVisualizer(
            labels=["x", "y", "z"], values=[1, 2, 3],
            orientation="vertical",
        )
        j = json.loads(viz.to_json())
        assert j["orientation"] == "vertical"
        _check_save_and_html(viz, "lollipop_vert")

    def test_from_importances(self, fitted_rf, iris_data):
        _, _, fnames, _ = iris_data
        viz = LollipopChartVisualizer.from_importances(
            fitted_rf, feature_names=fnames, top_n=3,
        )
        j = json.loads(viz.to_json())
        assert len(j["labels"]) == 3

    def test_from_dict(self):
        viz = LollipopChartVisualizer.from_dict({"x": 1, "y": 2, "z": 3})
        _check_save_and_html(viz, "lollipop_dict")

    def test_highlight_top(self):
        viz = LollipopChartVisualizer(
            labels=["a", "b", "c"], values=[3, 2, 1],
            highlight_top=2,
        )
        j = json.loads(viz.to_json())
        assert j["highlightTop"] == 2

    def test_custom_baseline(self):
        viz = LollipopChartVisualizer(
            labels=["a", "b", "c"], values=[0.5, -0.3, 0.8],
            baseline=0,
        )
        j = json.loads(viz.to_json())
        assert j["baseline"] == 0

    def test_single_item(self):
        viz = LollipopChartVisualizer(labels=["only"], values=[42])
        _check_save_and_html(viz, "lollipop_single")


# ===== Dumbbell Chart Tests =====

class TestDumbbellChart:
    def test_basic(self):
        viz = DumbbellChartVisualizer(
            labels=["Model A", "Model B", "Model C"],
            values_start=[0.85, 0.82, 0.88],
            values_end=[0.91, 0.89, 0.93],
        )
        _check_save_and_html(viz, "dumbbell_basic")

    def test_sort_by_diff(self):
        viz = DumbbellChartVisualizer(
            labels=["A", "B", "C"],
            values_start=[0.8, 0.7, 0.9],
            values_end=[0.85, 0.9, 0.92],
            sort_by="diff",
        )
        j = json.loads(viz.to_json())
        # B has biggest diff (0.2), should be first
        assert j["labels"][0] == "B"

    def test_custom_labels(self):
        viz = DumbbellChartVisualizer(
            labels=["X", "Y"],
            values_start=[1, 2],
            values_end=[3, 4],
            start_label="Baseline",
            end_label="Tuned",
        )
        j = json.loads(viz.to_json())
        assert j["startLabel"] == "Baseline"
        assert j["endLabel"] == "Tuned"

    def test_from_train_test(self):
        viz = DumbbellChartVisualizer.from_train_test(
            labels=["Acc", "F1", "AUC"],
            train_scores=[0.95, 0.93, 0.97],
            test_scores=[0.91, 0.89, 0.94],
        )
        j = json.loads(viz.to_json())
        assert j["startLabel"] == "Train"
        assert j["endLabel"] == "Test"
        _check_save_and_html(viz, "dumbbell_tt")

    def test_from_metrics(self):
        before = {"Acc": 0.85, "F1": 0.80, "AUC": 0.88}
        after = {"Acc": 0.92, "F1": 0.88, "AUC": 0.95}
        viz = DumbbellChartVisualizer.from_metrics(
            ["Acc", "F1", "AUC"], before, after,
        )
        _check_save_and_html(viz, "dumbbell_metrics")

    def test_single_category(self):
        viz = DumbbellChartVisualizer(
            labels=["Only"], values_start=[1.0], values_end=[2.0],
        )
        _check_save_and_html(viz, "dumbbell_single")


# ===== Funnel Chart Tests =====

class TestFunnelChart:
    def test_basic(self):
        viz = FunnelChartVisualizer(
            stages=["Raw", "Cleaned", "Featured", "Trained"],
            values=[10000, 8500, 7200, 6800],
        )
        _check_save_and_html(viz, "funnel_basic")

    def test_from_pipeline(self):
        viz = FunnelChartVisualizer.from_pipeline(
            stages=["Load", "Filter", "Transform", "Train"],
            sample_counts=[50000, 42000, 38000, 35000],
        )
        _check_save_and_html(viz, "funnel_pipeline")

    def test_from_feature_selection(self):
        viz = FunnelChartVisualizer.from_feature_selection(
            stages=["All Features", "Variance", "Correlation", "RFE"],
            feature_counts=[500, 350, 120, 50],
        )
        j = json.loads(viz.to_json())
        assert len(j["stages"]) == 4
        _check_save_and_html(viz, "funnel_feat")

    def test_percentages_shown(self):
        viz = FunnelChartVisualizer(
            stages=["A", "B", "C"], values=[100, 50, 25],
            show_percentages=True,
        )
        j = json.loads(viz.to_json())
        assert j["showPercentages"] is True

    def test_percentages_off(self):
        viz = FunnelChartVisualizer(
            stages=["A", "B"], values=[100, 80],
            show_percentages=False,
        )
        j = json.loads(viz.to_json())
        assert j["showPercentages"] is False

    def test_single_stage(self):
        viz = FunnelChartVisualizer(stages=["Only"], values=[100])
        _check_save_and_html(viz, "funnel_single")


# ===== Gauge Chart Tests =====

class TestGaugeChart:
    def test_basic(self):
        viz = GaugeChartVisualizer(value=0.75, label="Score")
        _check_save_and_html(viz, "gauge_basic")

    def test_from_score(self):
        viz = GaugeChartVisualizer.from_score(0.92, "Accuracy")
        j = json.loads(viz.to_json())
        assert j["value"] == 0.92
        assert j["label"] == "Accuracy"
        _check_save_and_html(viz, "gauge_score")

    def test_from_accuracy(self):
        viz = GaugeChartVisualizer.from_accuracy(0.873)
        j = json.loads(viz.to_json())
        assert j["displayValue"] == "87.3%"
        _check_save_and_html(viz, "gauge_acc")

    def test_custom_range(self):
        viz = GaugeChartVisualizer(
            value=75, min_value=0, max_value=100,
            label="Percentage",
        )
        j = json.loads(viz.to_json())
        assert j["minValue"] == 0
        assert j["maxValue"] == 100

    def test_custom_zones(self):
        viz = GaugeChartVisualizer(
            value=0.6,
            zones=[(0, 0.4, "#ff0000"), (0.4, 0.7, "#ffaa00"), (0.7, 1.0, "#00ff00")],
        )
        j = json.loads(viz.to_json())
        assert len(j["zones"]) == 3
        assert j["zones"][0]["color"] == "#ff0000"

    def test_format_string(self):
        viz = GaugeChartVisualizer(value=0.923, format_str=".1%")
        j = json.loads(viz.to_json())
        assert j["displayValue"] == "92.3%"

    def test_edge_at_min(self):
        viz = GaugeChartVisualizer(value=0, min_value=0, max_value=1)
        _check_save_and_html(viz, "gauge_min")

    def test_edge_at_max(self):
        viz = GaugeChartVisualizer(value=1, min_value=0, max_value=1)
        _check_save_and_html(viz, "gauge_max")


# ===== Classification Report Tests =====

class TestClassificationReport:
    def test_binary(self, binary_clf):
        clf, X, y = binary_clf
        report = ClassificationReport(clf, X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "clf_report")
            assert path.exists()
            assert path.suffix == ".html"
            content = path.read_text()
            _check_html_basics(content)
            assert "Accuracy" in content
            assert "Confusion Matrix" in content
            assert "ROC Curve" in content
            assert "Precision-Recall" in content
            assert "Calibration" in content

    def test_multiclass(self, iris_data):
        X, y, fnames, cnames = iris_data
        clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
        report = ClassificationReport(clf, X, y, feature_names=fnames, class_names=list(cnames))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "clf_multi_report")
            assert path.exists()
            content = path.read_text()
            _check_html_basics(content)
            assert "Feature Importances" in content
            assert "Class Distribution" in content

    def test_metrics_accessible(self, binary_clf):
        clf, X, y = binary_clf
        report = ClassificationReport(clf, X, y)
        m = report.metrics
        assert "accuracy" in m
        assert "f1" in m
        assert "mcc" in m
        assert "auc" in m
        assert "per_class" in m
        assert m["accuracy"] > 0.5
        assert m["n_classes"] == 2

    def test_decision_tree_rules(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        report = ClassificationReport(dt, X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "dt_report")
            content = path.read_text()
            assert "Decision Tree Rules" in content

    def test_linear_model_coefficients(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        lr = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
        report = ClassificationReport(lr, X, y, feature_names=[f"f{i}" for i in range(5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "lr_report")
            content = path.read_text()
            assert "Coefficients" in content

    def test_repr_html(self, binary_clf):
        clf, X, y = binary_clf
        report = ClassificationReport(clf, X, y)
        html = report._repr_html_()
        assert isinstance(html, str)
        _check_html_basics(html)

    def test_custom_names(self, binary_clf):
        clf, X, y = binary_clf
        report = ClassificationReport(
            clf, X, y,
            model_name="MyModel",
            dataset_name="TestData",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "named_report")
            content = path.read_text()
            assert "MyModel" in content
            assert "TestData" in content

    def test_light_theme(self, binary_clf):
        clf, X, y = binary_clf
        report = ClassificationReport(clf, X, y, theme="light")
        html = report._repr_html_()
        assert 'data-theme="light"' in html


# ===== Regression Report Tests =====

class TestRegressionReport:
    def test_basic(self):
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=5, max_depth=4, random_state=42).fit(X, y)
        report = RegressionReport(reg, X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "reg_report")
            assert path.exists()
            content = path.read_text()
            _check_html_basics(content)
            assert "Predicted vs Actual" in content
            assert "Residual Distribution" in content
            assert "Residuals vs Predicted" in content
            assert "Q-Q Plot" in content
            assert "Feature Importances" in content

    def test_metrics(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, y)
        report = RegressionReport(reg, X, y)
        m = report.metrics
        assert "r2" in m
        assert "rmse" in m
        assert "mae" in m
        assert "max_error" in m
        assert "residual_mean" in m

    def test_decision_tree_regressor(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
        report = RegressionReport(dt, X, y, feature_names=[f"f{i}" for i in range(5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "dt_reg_report")
            content = path.read_text()
            assert "Decision Tree Rules" in content

    def test_linear_regressor(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(X, y)
        report = RegressionReport(lr, X, y, feature_names=[f"f{i}" for i in range(5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "lr_reg_report")
            content = path.read_text()
            assert "Coefficients" in content
            assert "Intercept" in content

    def test_repr_html(self):
        X, y = make_regression(n_samples=50, n_features=3, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=3, random_state=42).fit(X, y)
        report = RegressionReport(reg, X, y)
        html = report._repr_html_()
        assert isinstance(html, str)
        _check_html_basics(html)

    def test_custom_names(self):
        X, y = make_regression(n_samples=50, n_features=3, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=3, random_state=42).fit(X, y)
        report = RegressionReport(reg, X, y, model_name="RF Reg", dataset_name="Synthetic")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(Path(tmpdir) / "named_reg")
            content = path.read_text()
            assert "RF Reg" in content
            assert "Synthetic" in content
