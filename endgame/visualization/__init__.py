from __future__ import annotations

"""Visualization module for Endgame models.

Interactive, publication-quality visualizations for machine learning models.
All visualizations are self-contained HTML files with no external dependencies.

Example
-------
>>> from endgame.visualization import TreeVisualizer, BarChartVisualizer
>>> from sklearn.tree import DecisionTreeClassifier
>>>
>>> clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
>>> viz = TreeVisualizer(clf, feature_names=feature_names, class_names=class_names)
>>> viz.save("my_tree.html")
>>>
>>> bar = BarChartVisualizer.from_importances(clf, feature_names=feature_names)
>>> bar.save("importances.html")
"""

# Shared infrastructure
from endgame.visualization._base import BaseVisualizer
from endgame.visualization._palettes import CATEGORICAL, DIVERGING, SEQUENTIAL, get_palette

# Tier 4 — Extended charts
from endgame.visualization.arc_diagram import ArcDiagramVisualizer

# Tier 1 — Critical charts
from endgame.visualization.bar_chart import BarChartVisualizer

# Tier 2 — Important charts
from endgame.visualization.box_violin import BoxPlotVisualizer, ViolinPlotVisualizer
from endgame.visualization.bump_chart import BumpChartVisualizer
from endgame.visualization.calibration_plot import CalibrationPlotVisualizer
from endgame.visualization.chord_diagram import ChordDiagramVisualizer

# Reports
from endgame.visualization.classification_report import ClassificationReport
from endgame.visualization.confusion_matrix import ConfusionMatrixVisualizer
from endgame.visualization.donut_chart import DonutChartVisualizer
from endgame.visualization.dot_matrix import DotMatrixVisualizer
from endgame.visualization.dumbbell_chart import DumbbellChartVisualizer
from endgame.visualization.error_bars import ErrorBarsVisualizer
from endgame.visualization.flow_chart import FlowChartVisualizer
from endgame.visualization.funnel_chart import FunnelChartVisualizer
from endgame.visualization.gauge_chart import GaugeChartVisualizer
from endgame.visualization.heatmap import HeatmapVisualizer
from endgame.visualization.histogram import HistogramVisualizer
from endgame.visualization.lift_chart import LiftChartVisualizer
from endgame.visualization.line_chart import LineChartVisualizer
from endgame.visualization.lollipop_chart import LollipopChartVisualizer
from endgame.visualization.network_diagram import NetworkDiagramVisualizer
from endgame.visualization.nightingale_rose import NightingaleRoseVisualizer
from endgame.visualization.parallel_coordinates import ParallelCoordinatesVisualizer
from endgame.visualization.pdp_plot import PDP2DVisualizer, PDPVisualizer
from endgame.visualization.pr_curve import PRCurveVisualizer
from endgame.visualization.radar_chart import RadarChartVisualizer
from endgame.visualization.radial_bar import RadialBarVisualizer
from endgame.visualization.regression_report import RegressionReport

# Tier B — General-purpose charts
from endgame.visualization.ridgeline_plot import RidgelinePlotVisualizer

# Tier A — ML evaluation charts
from endgame.visualization.roc_curve import ROCCurveVisualizer

# Tier 3 — Nice-to-have charts
from endgame.visualization.sankey import SankeyVisualizer
from endgame.visualization.scatterplot import ScatterplotVisualizer
from endgame.visualization.spiral_plot import SpiralPlotVisualizer
from endgame.visualization.stream_graph import StreamGraphVisualizer
from endgame.visualization.tree_visualizer import TreeVisualizer
from endgame.visualization.treemap import SunburstVisualizer, TreemapVisualizer
from endgame.visualization.venn import VennDiagramVisualizer
from endgame.visualization.waterfall_chart import WaterfallVisualizer
from endgame.visualization.word_cloud import WordCloudVisualizer

__all__ = [
    # Infrastructure
    "BaseVisualizer",
    "get_palette",
    "CATEGORICAL",
    "SEQUENTIAL",
    "DIVERGING",
    # Existing
    "TreeVisualizer",
    # Tier 1
    "BarChartVisualizer",
    "HeatmapVisualizer",
    "ConfusionMatrixVisualizer",
    "HistogramVisualizer",
    "LineChartVisualizer",
    "ScatterplotVisualizer",
    # Tier 2
    "BoxPlotVisualizer",
    "ViolinPlotVisualizer",
    "ErrorBarsVisualizer",
    "ParallelCoordinatesVisualizer",
    "RadarChartVisualizer",
    "TreemapVisualizer",
    "SunburstVisualizer",
    # Tier 3
    "SankeyVisualizer",
    "DotMatrixVisualizer",
    "VennDiagramVisualizer",
    "WordCloudVisualizer",
    # Tier 4 — Extended
    "ArcDiagramVisualizer",
    "ChordDiagramVisualizer",
    "DonutChartVisualizer",
    "FlowChartVisualizer",
    "NetworkDiagramVisualizer",
    "NightingaleRoseVisualizer",
    "RadialBarVisualizer",
    "SpiralPlotVisualizer",
    "StreamGraphVisualizer",
    # Tier A — ML evaluation
    "ROCCurveVisualizer",
    "PRCurveVisualizer",
    "CalibrationPlotVisualizer",
    "LiftChartVisualizer",
    "PDPVisualizer",
    "PDP2DVisualizer",
    "WaterfallVisualizer",
    # Tier B — General-purpose
    "RidgelinePlotVisualizer",
    "BumpChartVisualizer",
    "LollipopChartVisualizer",
    "DumbbellChartVisualizer",
    "FunnelChartVisualizer",
    "GaugeChartVisualizer",
    # Reports
    "ClassificationReport",
    "RegressionReport",
]
