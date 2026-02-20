"""HTML Report Generator for Benchmark Results.

Generates beautiful, interactive HTML reports with:
- Performance comparison charts (accuracy, F1, AUC, etc.)
- Training time comparisons
- Model interpretability outputs (rules, trees, equations)
- Interactive Plotly charts
- Sortable tables
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from endgame.benchmark.tracker import ExperimentTracker

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text))


def _format_number(value: float, precision: int = 4) -> str:
    """Format a number for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if abs(value) < 0.0001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def _get_color_scale(n_colors: int) -> list[str]:
    """Get a list of distinct colors for charts."""
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1
    return colors[:n_colors] if n_colors <= len(colors) else colors * (n_colors // len(colors) + 1)


class BenchmarkReportGenerator:
    """Generate HTML reports from benchmark results.

    Parameters
    ----------
    tracker : ExperimentTracker
        The experiment tracker with benchmark results.
    title : str, optional
        Report title.

    Examples
    --------
    >>> from endgame.benchmark import BenchmarkRunner, BenchmarkReportGenerator
    >>> runner = BenchmarkRunner(suite="sklearn-classic")
    >>> tracker = runner.run(models)
    >>> report = BenchmarkReportGenerator(tracker)
    >>> report.generate("benchmark_report.html")
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        title: str = "Endgame Benchmark Report",
    ):
        self.tracker = tracker
        self.title = title
        self._df = tracker.to_dataframe()
        self._interpretability_outputs: dict[str, dict[str, str]] = {}

    def add_interpretability_output(
        self,
        model_name: str,
        dataset_name: str,
        output: str,
        output_type: str = "text",
    ) -> None:
        """Add interpretability output for a model.

        Parameters
        ----------
        model_name : str
            Name of the model.
        dataset_name : str
            Name of the dataset.
        output : str
            The interpretability output (rules, tree structure, equation, etc.)
        output_type : str
            Type of output: "text", "html", "latex", "code"
        """
        key = f"{model_name}_{dataset_name}"
        self._interpretability_outputs[key] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "output": output,
            "output_type": output_type,
        }

    def generate(
        self,
        output_path: str,
        include_interpretability: bool = True,
        include_meta_features: bool = False,
    ) -> str:
        """Generate the HTML report.

        Parameters
        ----------
        output_path : str
            Path to save the HTML report.
        include_interpretability : bool
            Include interpretability outputs section.
        include_meta_features : bool
            Include dataset meta-features section.

        Returns
        -------
        str
            Path to the generated report.
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for report generation. Install with: pip install plotly")

        sections = []

        # Header
        sections.append(self._generate_header())

        # Summary statistics
        sections.append(self._generate_summary_section())

        # Performance comparison charts
        sections.append(self._generate_performance_section())

        # Speed comparison
        sections.append(self._generate_speed_section())

        # Per-dataset results
        sections.append(self._generate_dataset_section())

        # Model rankings
        sections.append(self._generate_rankings_section())

        # Interpretability outputs
        if include_interpretability and self._interpretability_outputs:
            sections.append(self._generate_interpretability_section())

        # Meta-features
        if include_meta_features:
            sections.append(self._generate_meta_features_section())

        # Failed experiments
        sections.append(self._generate_failures_section())

        # Footer
        sections.append(self._generate_footer())

        # Combine all sections
        html_content = self._wrap_html(sections)

        # Write to file
        Path(output_path).write_text(html_content, encoding="utf-8")

        return output_path

    def _wrap_html(self, sections: list[str]) -> str:
        """Wrap sections in HTML document structure."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(self.title)}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --success-color: #22c55e;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #e2e8f0;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 0 0 1rem 1rem;
        }}

        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        header .subtitle {{
            color: rgba(255, 255, 255, 0.8);
            font-size: 1.1rem;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}

        .card h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .card h3 {{
            font-size: 1.2rem;
            margin: 1.5rem 0 1rem 0;
            color: var(--text-color);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .stat-card {{
            background: rgba(99, 102, 241, 0.1);
            border-radius: 0.75rem;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }}

        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
        }}

        .stat-card .label {{
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}

        .chart-container {{
            width: 100%;
            min-height: 400px;
            margin: 1rem 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: rgba(99, 102, 241, 0.1);
            font-weight: 600;
            color: var(--primary-color);
            cursor: pointer;
            user-select: none;
        }}

        th:hover {{
            background: rgba(99, 102, 241, 0.2);
        }}

        tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-success {{
            background: rgba(34, 197, 94, 0.2);
            color: var(--success-color);
        }}

        .badge-warning {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning-color);
        }}

        .badge-danger {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger-color);
        }}

        .interpretability-output {{
            background: #0d1117;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            overflow-x: auto;
            font-family: 'Fira Code', 'Monaco', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            white-space: pre-wrap;
            border: 1px solid var(--border-color);
        }}

        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}

        .model-name {{
            font-weight: 600;
            color: var(--secondary-color);
        }}

        .dataset-name {{
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        .tab {{
            padding: 0.5rem 1rem;
            border-radius: 0.5rem 0.5rem 0 0;
            cursor: pointer;
            transition: all 0.2s;
            color: var(--text-muted);
        }}

        .tab:hover {{
            background: rgba(99, 102, 241, 0.1);
        }}

        .tab.active {{
            background: var(--primary-color);
            color: white;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .rank-1 {{ color: #ffd700; font-weight: 700; }}
        .rank-2 {{ color: #c0c0c0; font-weight: 600; }}
        .rank-3 {{ color: #cd7f32; font-weight: 600; }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        .collapsible {{
            cursor: pointer;
            padding: 0.75rem;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .collapsible:hover {{
            background: rgba(99, 102, 241, 0.2);
        }}

        .collapsible-content {{
            display: none;
            padding: 1rem;
            animation: fadeIn 0.3s ease;
        }}

        .collapsible-content.show {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
        }}

        .progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transition: width 0.3s ease;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            header h1 {{
                font-size: 1.75rem;
            }}

            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {''.join(sections)}
    </div>

    <script>
        // Table sorting
        document.querySelectorAll('th[data-sort]').forEach(th => {{
            th.addEventListener('click', () => {{
                const table = th.closest('table');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const idx = Array.from(th.parentNode.children).indexOf(th);
                const asc = th.dataset.order !== 'asc';

                rows.sort((a, b) => {{
                    const aVal = a.children[idx].textContent;
                    const bVal = b.children[idx].textContent;
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);

                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return asc ? aNum - bNum : bNum - aNum;
                    }}
                    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }});

                th.dataset.order = asc ? 'asc' : 'desc';
                rows.forEach(row => tbody.appendChild(row));
            }});
        }});

        // Collapsible sections
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.addEventListener('click', () => {{
                const content = el.nextElementSibling;
                content.classList.toggle('show');
                el.querySelector('.arrow').textContent = content.classList.contains('show') ? '▼' : '▶';
            }});
        }});

        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const tabGroup = tab.closest('.tabs-container');
                tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                tabGroup.querySelector(tab.dataset.target).classList.add('active');
            }});
        }});
    </script>
</body>
</html>"""

    def _generate_header(self) -> str:
        """Generate the report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_experiments = len(self.tracker)
        n_successful = len(self.tracker.get_successful())

        return f"""
<header>
    <h1>🏆 {_escape_html(self.title)}</h1>
    <p class="subtitle">Generated on {timestamp} • {n_experiments} experiments • {n_successful} successful</p>
</header>
"""

    def _generate_summary_section(self) -> str:
        """Generate summary statistics section."""
        df = self._df

        if HAS_POLARS:
            successful = df.filter(pl.col('status') == 'success')
            n_total = len(df)
            n_success = len(successful)
            n_failed = n_total - n_success
            n_datasets = df['dataset_name'].n_unique()
            n_models = df['model_name'].n_unique()

            # Best accuracy
            if 'metric_accuracy' in successful.columns and len(successful) > 0:
                best_acc_row = successful.sort('metric_accuracy', descending=True).head(1)
                best_acc = best_acc_row['metric_accuracy'][0]
                best_acc_model = best_acc_row['model_name'][0]
            else:
                best_acc = 0
                best_acc_model = "N/A"

            # Average fit time
            if 'fit_time' in successful.columns:
                avg_fit_time = successful['fit_time'].mean()
            else:
                avg_fit_time = 0
        else:
            successful = df[df['status'] == 'success']
            n_total = len(df)
            n_success = len(successful)
            n_failed = n_total - n_success
            n_datasets = df['dataset_name'].nunique()
            n_models = df['model_name'].nunique()

            if 'metric_accuracy' in successful.columns and len(successful) > 0:
                best_acc = successful['metric_accuracy'].max()
                best_acc_model = successful.loc[successful['metric_accuracy'].idxmax(), 'model_name']
            else:
                best_acc = 0
                best_acc_model = "N/A"

            avg_fit_time = successful['fit_time'].mean() if 'fit_time' in successful.columns else 0

        success_rate = (n_success / n_total * 100) if n_total > 0 else 0

        return f"""
<section class="card">
    <h2>📊 Summary</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{n_total}</div>
            <div class="label">Total Experiments</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_success}</div>
            <div class="label">Successful</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_datasets}</div>
            <div class="label">Datasets</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_models}</div>
            <div class="label">Models</div>
        </div>
        <div class="stat-card">
            <div class="value">{_format_number(best_acc)}</div>
            <div class="label">Best Accuracy ({_escape_html(best_acc_model)})</div>
        </div>
        <div class="stat-card">
            <div class="value">{_format_number(avg_fit_time, 2)}s</div>
            <div class="label">Avg Fit Time</div>
        </div>
    </div>
    <div class="progress-bar">
        <div class="progress-bar-fill" style="width: {success_rate}%"></div>
    </div>
    <p style="text-align: center; color: var(--text-muted); margin-top: 0.5rem;">
        {_format_number(success_rate, 1)}% success rate
    </p>
</section>
"""

    def _generate_performance_section(self) -> str:
        """Generate performance comparison charts."""
        df = self._df

        if HAS_POLARS:
            successful = df.filter(pl.col('status') == 'success')
        else:
            successful = df[df['status'] == 'success']

        if len(successful) == 0:
            return """<section class="card"><h2>📈 Performance</h2><p>No successful experiments to display.</p></section>"""

        charts_html = []

        # Accuracy comparison chart
        accuracy_chart = self._create_metric_comparison_chart(successful, 'metric_accuracy', 'Accuracy')
        if accuracy_chart:
            charts_html.append('<div class="chart-container" id="accuracy-chart"></div>')
            charts_html.append(f'<script>{accuracy_chart}</script>')

        # F1 score chart (if available)
        f1_col = 'metric_f1' if 'metric_f1' in (successful.columns if HAS_POLARS else successful.columns) else 'metric_f1_weighted'
        if f1_col in (successful.columns if HAS_POLARS else successful.columns):
            f1_chart = self._create_metric_comparison_chart(successful, f1_col, 'F1 Score')
            if f1_chart:
                charts_html.append('<div class="chart-container" id="f1-chart"></div>')
                charts_html.append(f'<script>{f1_chart}</script>')

        # ROC-AUC chart (if available)
        auc_col = 'metric_roc_auc' if 'metric_roc_auc' in (successful.columns if HAS_POLARS else successful.columns) else 'metric_roc_auc_ovr_weighted'
        if auc_col in (successful.columns if HAS_POLARS else successful.columns):
            auc_chart = self._create_metric_comparison_chart(successful, auc_col, 'ROC-AUC')
            if auc_chart:
                charts_html.append('<div class="chart-container" id="auc-chart"></div>')
                charts_html.append(f'<script>{auc_chart}</script>')

        return f"""
<section class="card">
    <h2>📈 Performance Comparison</h2>
    {''.join(charts_html)}
</section>
"""

    def _create_metric_comparison_chart(self, df, metric_col: str, metric_name: str) -> str | None:
        """Create a bar chart comparing model performance on a metric."""
        if metric_col not in (df.columns if HAS_POLARS else df.columns):
            return None

        # Aggregate by model
        if HAS_POLARS:
            agg = df.group_by('model_name').agg([
                pl.col(metric_col).mean().alias('mean'),
                pl.col(metric_col).std().alias('std'),
                pl.col(metric_col).count().alias('count'),
            ]).sort('mean', descending=True)

            models = agg['model_name'].to_list()
            means = agg['mean'].to_list()
            stds = [s if s is not None else 0 for s in agg['std'].to_list()]
        else:
            agg = df.groupby('model_name')[metric_col].agg(['mean', 'std', 'count']).reset_index()
            agg = agg.sort_values('mean', ascending=False)

            models = agg['model_name'].tolist()
            means = agg['mean'].tolist()
            stds = agg['std'].fillna(0).tolist()

        # Create Plotly chart
        colors = _get_color_scale(len(models))

        chart_id = metric_col.replace('metric_', '') + '-chart'

        fig_json = {
            'data': [{
                'type': 'bar',
                'x': models,
                'y': means,
                'error_y': {
                    'type': 'data',
                    'array': stds,
                    'visible': True,
                },
                'marker': {'color': colors[:len(models)]},
                'hovertemplate': '<b>%{x}</b><br>' + metric_name + ': %{y:.4f}<extra></extra>',
            }],
            'layout': {
                'title': {'text': f'{metric_name} by Model (Mean ± Std)', 'font': {'color': '#e2e8f0'}},
                'xaxis': {'title': 'Model', 'tickangle': -45, 'color': '#94a3b8'},
                'yaxis': {'title': metric_name, 'color': '#94a3b8'},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#e2e8f0'},
                'margin': {'b': 150},
            }
        }

        return f"Plotly.newPlot('{chart_id}', {json.dumps(fig_json['data'])}, {json.dumps(fig_json['layout'])});"

    def _generate_speed_section(self) -> str:
        """Generate training speed comparison section."""
        df = self._df

        if HAS_POLARS:
            successful = df.filter(pl.col('status') == 'success')
            if 'fit_time' not in successful.columns or len(successful) == 0:
                return ""

            agg = successful.group_by('model_name').agg([
                pl.col('fit_time').mean().alias('mean_time'),
                pl.col('fit_time').min().alias('min_time'),
                pl.col('fit_time').max().alias('max_time'),
            ]).sort('mean_time')

            models = agg['model_name'].to_list()
            mean_times = agg['mean_time'].to_list()
            min_times = agg['min_time'].to_list()
            max_times = agg['max_time'].to_list()
        else:
            successful = df[df['status'] == 'success']
            if 'fit_time' not in successful.columns or len(successful) == 0:
                return ""

            agg = successful.groupby('model_name')['fit_time'].agg(['mean', 'min', 'max']).reset_index()
            agg = agg.sort_values('mean')

            models = agg['model_name'].tolist()
            mean_times = agg['mean'].tolist()
            min_times = agg['min'].tolist()
            max_times = agg['max'].tolist()

        # Create speed chart
        colors = ['#22c55e' if t < 10 else '#f59e0b' if t < 60 else '#ef4444' for t in mean_times]

        fig_json = {
            'data': [{
                'type': 'bar',
                'x': mean_times,
                'y': models,
                'orientation': 'h',
                'marker': {'color': colors},
                'hovertemplate': '<b>%{y}</b><br>Avg: %{x:.2f}s<extra></extra>',
            }],
            'layout': {
                'title': {'text': 'Training Time by Model (seconds)', 'font': {'color': '#e2e8f0'}},
                'xaxis': {'title': 'Time (seconds)', 'color': '#94a3b8', 'type': 'log'},
                'yaxis': {'title': '', 'color': '#94a3b8', 'autorange': 'reversed'},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#e2e8f0'},
                'margin': {'l': 200},
                'height': max(400, len(models) * 25),
            }
        }

        return f"""
<section class="card">
    <h2>⚡ Training Speed</h2>
    <div class="chart-container" id="speed-chart" style="min-height: {max(400, len(models) * 25)}px"></div>
    <script>Plotly.newPlot('speed-chart', {json.dumps(fig_json['data'])}, {json.dumps(fig_json['layout'])});</script>
    <p style="color: var(--text-muted); text-align: center; margin-top: 1rem;">
        <span style="color: #22c55e;">●</span> Fast (&lt;10s) &nbsp;
        <span style="color: #f59e0b;">●</span> Medium (&lt;60s) &nbsp;
        <span style="color: #ef4444;">●</span> Slow (&gt;60s)
    </p>
</section>
"""

    def _generate_dataset_section(self) -> str:
        """Generate per-dataset results section."""
        df = self._df

        if HAS_POLARS:
            successful = df.filter(pl.col('status') == 'success')
            datasets = successful['dataset_name'].unique().to_list()
        else:
            successful = df[df['status'] == 'success']
            datasets = successful['dataset_name'].unique().tolist()

        if not datasets:
            return ""

        dataset_sections = []

        for dataset in sorted(datasets):
            if HAS_POLARS:
                dataset_df = successful.filter(pl.col('dataset_name') == dataset)
                dataset_df = dataset_df.sort('metric_accuracy', descending=True, nulls_last=True)
            else:
                dataset_df = successful[successful['dataset_name'] == dataset]
                dataset_df = dataset_df.sort_values('metric_accuracy', ascending=False)

            # Create table rows
            rows = []
            for i, row in enumerate(dataset_df.iter_rows(named=True) if HAS_POLARS else dataset_df.itertuples()):
                if HAS_POLARS:
                    model_name = row['model_name']
                    accuracy = row.get('metric_accuracy', None)
                    f1 = row.get('metric_f1', row.get('metric_f1_weighted', None))
                    fit_time = row.get('fit_time', None)
                else:
                    model_name = row.model_name
                    accuracy = getattr(row, 'metric_accuracy', None)
                    f1 = getattr(row, 'metric_f1', getattr(row, 'metric_f1_weighted', None))
                    fit_time = getattr(row, 'fit_time', None)

                rank_class = f"rank-{i+1}" if i < 3 else ""
                rows.append(f"""
                    <tr>
                        <td class="{rank_class}">{i+1}</td>
                        <td>{_escape_html(model_name)}</td>
                        <td>{_format_number(accuracy)}</td>
                        <td>{_format_number(f1)}</td>
                        <td>{_format_number(fit_time, 2)}s</td>
                    </tr>
                """)

            dataset_sections.append(f"""
            <div class="collapsible">
                <span><strong>{_escape_html(dataset)}</strong></span>
                <span class="arrow">▶</span>
            </div>
            <div class="collapsible-content">
                <table>
                    <thead>
                        <tr>
                            <th data-sort>Rank</th>
                            <th data-sort>Model</th>
                            <th data-sort>Accuracy</th>
                            <th data-sort>F1</th>
                            <th data-sort>Fit Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
            """)

        return f"""
<section class="card">
    <h2>📁 Results by Dataset</h2>
    {''.join(dataset_sections)}
</section>
"""

    def _generate_rankings_section(self) -> str:
        """Generate model rankings section."""
        df = self._df

        if HAS_POLARS:
            successful = df.filter(pl.col('status') == 'success')
            if len(successful) == 0:
                return ""

            # Compute mean rank per model
            rankings = successful.group_by('model_name').agg([
                pl.col('metric_accuracy').mean().alias('mean_accuracy'),
                pl.col('metric_accuracy').count().alias('n_experiments'),
                pl.col('fit_time').mean().alias('mean_time'),
            ]).sort('mean_accuracy', descending=True)

            rows_data = list(rankings.iter_rows(named=True))
        else:
            successful = df[df['status'] == 'success']
            if len(successful) == 0:
                return ""

            rankings = successful.groupby('model_name').agg({
                'metric_accuracy': ['mean', 'count'],
                'fit_time': 'mean',
            }).reset_index()
            rankings.columns = ['model_name', 'mean_accuracy', 'n_experiments', 'mean_time']
            rankings = rankings.sort_values('mean_accuracy', ascending=False)

            rows_data = rankings.to_dict('records')

        # Build table
        rows = []
        for i, row in enumerate(rows_data):
            model_name = row['model_name']
            mean_acc = row['mean_accuracy']
            n_exp = row['n_experiments']
            mean_time = row['mean_time']

            rank_class = f"rank-{i+1}" if i < 3 else ""
            badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""

            rows.append(f"""
                <tr>
                    <td class="{rank_class}">{badge} {i+1}</td>
                    <td><strong>{_escape_html(model_name)}</strong></td>
                    <td>{_format_number(mean_acc)}</td>
                    <td>{n_exp}</td>
                    <td>{_format_number(mean_time, 2)}s</td>
                </tr>
            """)

        return f"""
<section class="card">
    <h2>🏅 Model Rankings</h2>
    <table>
        <thead>
            <tr>
                <th data-sort>Rank</th>
                <th data-sort>Model</th>
                <th data-sort>Mean Accuracy</th>
                <th data-sort>Experiments</th>
                <th data-sort>Mean Fit Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</section>
"""

    def _generate_interpretability_section(self) -> str:
        """Generate interpretability outputs section."""
        if not self._interpretability_outputs:
            return ""

        outputs_html = []

        for key, data in self._interpretability_outputs.items():
            model_name = data['model_name']
            dataset_name = data['dataset_name']
            output = data['output']
            output_type = data['output_type']

            if output_type == "latex":
                # Wrap in math delimiters for potential MathJax rendering
                formatted_output = f"$${_escape_html(output)}$$"
            elif output_type == "html":
                formatted_output = output  # Already HTML
            else:
                formatted_output = _escape_html(output)

            outputs_html.append(f"""
            <div class="collapsible">
                <span>
                    <span class="model-name">{_escape_html(model_name)}</span>
                    <span class="dataset-name"> on {_escape_html(dataset_name)}</span>
                </span>
                <span class="arrow">▶</span>
            </div>
            <div class="collapsible-content">
                <div class="interpretability-output">{formatted_output}</div>
            </div>
            """)

        return f"""
<section class="card">
    <h2>🔍 Model Interpretability</h2>
    <p style="color: var(--text-muted); margin-bottom: 1rem;">
        Learned rules, trees, equations, and other interpretable model outputs.
    </p>
    {''.join(outputs_html)}
</section>
"""

    def _generate_meta_features_section(self) -> str:
        """Generate dataset meta-features section."""
        # Get meta-features from tracker records
        meta_features_data = []

        for record in self.tracker.get_successful():
            if record.meta_features:
                meta_features_data.append({
                    'dataset': record.dataset_name,
                    **record.meta_features,
                })

        if not meta_features_data:
            return ""

        # Deduplicate by dataset
        seen = set()
        unique_data = []
        for mf in meta_features_data:
            if mf['dataset'] not in seen:
                seen.add(mf['dataset'])
                unique_data.append(mf)

        # Build table
        if not unique_data:
            return ""

        # Get common columns
        all_cols = set()
        for d in unique_data:
            all_cols.update(d.keys())
        all_cols.discard('dataset')
        cols = ['dataset'] + sorted(all_cols)[:10]  # Limit to 10 columns

        header = ''.join(f'<th data-sort>{_escape_html(c)}</th>' for c in cols)

        rows = []
        for d in unique_data:
            cells = []
            for c in cols:
                val = d.get(c, 'N/A')
                if isinstance(val, float):
                    val = _format_number(val, 2)
                cells.append(f'<td>{_escape_html(str(val))}</td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')

        return f"""
<section class="card">
    <h2>📐 Dataset Meta-Features</h2>
    <div style="overflow-x: auto;">
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
</section>
"""

    def _generate_failures_section(self) -> str:
        """Generate failed experiments section."""
        df = self._df

        if HAS_POLARS:
            failed = df.filter(pl.col('status') == 'failed')
            if len(failed) == 0:
                return ""

            rows_data = list(failed.select(['model_name', 'dataset_name', 'error_message']).iter_rows(named=True))
        else:
            failed = df[df['status'] == 'failed']
            if len(failed) == 0:
                return ""

            rows_data = failed[['model_name', 'dataset_name', 'error_message']].to_dict('records')

        rows = []
        for row in rows_data[:50]:  # Limit to 50 failures
            rows.append(f"""
                <tr>
                    <td>{_escape_html(row['model_name'])}</td>
                    <td>{_escape_html(row['dataset_name'])}</td>
                    <td style="color: var(--danger-color); font-size: 0.8rem;">
                        {_escape_html(str(row.get('error_message', 'Unknown error'))[:200])}
                    </td>
                </tr>
            """)

        return f"""
<section class="card">
    <h2>❌ Failed Experiments ({len(rows_data)} total)</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Dataset</th>
                <th>Error</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</section>
"""

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
<footer>
    <p>Generated by Endgame Benchmark Suite</p>
    <p>© {datetime.now().year} • Built with 🔬 for ML research</p>
</footer>
"""


def extract_interpretability_outputs(
    models: list[tuple],
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    dataset_name: str,
    feature_names: list[str] | None = None,
) -> dict[str, str]:
    """Extract interpretability outputs from fitted models.

    Parameters
    ----------
    models : List[Tuple]
        List of (name, fitted_model) tuples.
    X_sample : np.ndarray
        Sample data used for fitting.
    y_sample : np.ndarray
        Sample targets.
    dataset_name : str
        Name of the dataset.
    feature_names : List[str], optional
        Feature names for better output.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model names to their interpretability outputs.
    """
    outputs = {}

    for name, model in models:
        if model is None:
            continue

        output = None
        output_type = "text"

        try:
            # RuleFit
            if hasattr(model, 'get_rules'):
                rules = model.get_rules()
                if rules:
                    output = "\n".join([
                        f"Rule {i+1}: {r.get('rule', r)} (coef={r.get('coef', 'N/A'):.4f})"
                        for i, r in enumerate(rules[:20])  # Top 20 rules
                    ])

            # Symbolic Regression
            elif hasattr(model, 'get_best_equation'):
                output = model.get_best_equation()
                if hasattr(model, 'latex'):
                    try:
                        output = model.latex()
                        output_type = "latex"
                    except Exception:
                        pass

            # C5.0 / Decision Trees
            elif hasattr(model, 'get_structure'):
                output = model.get_structure()

            # FURIA
            elif hasattr(model, 'get_rules_str'):
                output = model.get_rules_str()

            # EBM
            elif hasattr(model, 'term_importances'):
                try:
                    importances = model.term_importances()
                    term_names = model.get_term_names() if hasattr(model, 'get_term_names') else [f"Term {i}" for i in range(len(importances))]
                    sorted_terms = sorted(zip(term_names, importances), key=lambda x: abs(x[1]), reverse=True)
                    output = "Top Feature Contributions:\n" + "\n".join([
                        f"  {name}: {imp:.4f}" for name, imp in sorted_terms[:15]
                    ])
                except Exception:
                    pass

            # MARS
            elif hasattr(model, 'summary') and 'MARS' in type(model).__name__:
                output = model.summary()

            # Generic summary method
            elif hasattr(model, 'summary'):
                try:
                    output = model.summary()
                except Exception:
                    pass

            # Feature importances fallback
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if feature_names is None:
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                output = "Feature Importances:\n" + "\n".join([
                    f"  {name}: {imp:.4f}" for name, imp in sorted_features[:15]
                ])

        except Exception as e:
            output = f"Error extracting interpretability: {str(e)}"

        if output:
            outputs[name] = {"output": output, "type": output_type, "dataset": dataset_name}

    return outputs
