"""HTML template for multi-section model reports.

Renders a full-page report with a metrics summary panel, multiple
embedded chart sections, and an optional interpretability section.
"""

from __future__ import annotations

from endgame.visualization._html_template import _BASE_CSS, _SHARED_JS


def render_report(
    *,
    title: str,
    subtitle: str,
    theme: str,
    hero_html: str = "",
    metrics_html: str,
    sections: list[dict],
    footer_html: str = "",
) -> str:
    """Build a complete multi-section HTML report.

    Parameters
    ----------
    title : str
        Report title.
    subtitle : str
        Subtitle (e.g., model name + dataset info).
    theme : str
        'dark' or 'light'.
    hero_html : str
        HTML for the hero metric card at top.
    metrics_html : str
        HTML for the metrics summary panel.
    sections : list of dict
        Each dict has 'title', 'chart_id', 'width', 'height',
        'data_json', 'config_json', 'chart_js'.
    footer_html : str
        Optional footer HTML.
    """
    section_html_parts = []
    section_js_parts = []

    for i, sec in enumerate(sections):
        cid = sec["chart_id"]
        section_html_parts.append(f"""
<div class="report-section">
  <h2 class="section-title">{sec['title']}</h2>
  <div id="{cid}" class="chart-panel" style="width:{sec['width']}px; height:{sec['height']}px;"></div>
</div>""")

        section_js_parts.append(f"""
(function() {{
  const data = {sec['data_json']};
  const config = {sec['config_json']};
  const container = document.getElementById('{cid}');
  {sec['chart_js']}
  renderChart_{cid}(data, config, container);
}})();""")

    sections_html = "\n".join(section_html_parts)
    sections_js = "\n".join(section_js_parts)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_BASE_CSS}
{_REPORT_CSS}
</style>
</head>
<body>
<div class="report-container">
  <header class="report-header">
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
  </header>

  {hero_html}

  <div class="metrics-panel">
    {metrics_html}
  </div>

  <div class="charts-grid">
    {sections_html}
  </div>

  {footer_html}
</div>
<div id="tooltip" class="tooltip"></div>
<script>
{_SHARED_JS}
{sections_js}
</script>
</body>
</html>"""


_REPORT_CSS = """
body {
  display: block;
  padding: 0;
  align-items: initial;
  justify-content: initial;
}

.report-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 28px;
}

.report-header {
  text-align: center;
  margin-bottom: 36px;
}

.report-header h1 {
  font-size: 30px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
  letter-spacing: -0.3px;
}

.report-header .subtitle {
  font-size: 14px;
  color: var(--text-muted);
  line-height: 1.6;
}

/* Hero metric card — the primary KPI */
.hero-metric {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  margin-bottom: 24px;
  padding: 28px 36px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
}

.hero-metric .hero-ring {
  position: relative;
  width: 96px;
  height: 96px;
  flex-shrink: 0;
}

.hero-metric .hero-ring svg { display: block; }

.hero-metric .hero-info {
  text-align: left;
}

.hero-metric .hero-value {
  font-size: 38px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  line-height: 1.1;
}

.hero-metric .hero-label {
  font-size: 13px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 4px;
}

.hero-metric .hero-desc {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 6px;
}

/* Metric cards grid */
.metrics-panel {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 14px;
  margin-bottom: 36px;
}

.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 16px;
  text-align: center;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.metric-card.metric-good {
  border-color: rgba(76, 175, 80, 0.3);
  background: linear-gradient(135deg, var(--bg-card), rgba(76, 175, 80, 0.05));
}

.metric-card.metric-ok {
  border-color: rgba(255, 193, 7, 0.3);
  background: linear-gradient(135deg, var(--bg-card), rgba(255, 193, 7, 0.04));
}

.metric-card.metric-poor {
  border-color: rgba(244, 67, 54, 0.3);
  background: linear-gradient(135deg, var(--bg-card), rgba(244, 67, 54, 0.04));
}

.metric-card .metric-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  line-height: 1.2;
}

.metric-card .metric-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 6px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Charts grid */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
  gap: 20px;
}

.report-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px;
  overflow: hidden;
  transition: box-shadow 0.15s ease;
}

.report-section:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 14px;
  letter-spacing: -0.2px;
}

.chart-panel {
  position: relative;
  overflow: hidden;
}

.chart-panel svg {
  display: block;
}

.chart-panel .legend {
  margin-top: 10px;
}

/* Footer & interpretability */
.report-footer {
  margin-top: 28px;
  padding: 18px 22px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.6;
}

.report-footer h3 {
  font-size: 15px;
  color: var(--text-primary);
  margin-bottom: 10px;
}

.report-footer pre {
  background: var(--bg-secondary);
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 11px;
  line-height: 1.5;
  color: var(--text-secondary);
  margin-top: 8px;
}

.interp-section {
  margin-top: 24px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px;
}

.interp-section h2 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 14px;
  color: var(--text-primary);
}

.rules-list {
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 12px;
  line-height: 1.7;
  color: var(--text-secondary);
  padding-left: 16px;
}

.rules-list li {
  margin-bottom: 4px;
}
"""
