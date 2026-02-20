"""Regression report — comprehensive single-page evaluation.

Generates a self-contained HTML report with performance metrics,
predicted vs actual scatter, residual analysis, feature importances,
and model interpretability (decision tree rules, linear coefficients).

Example
-------
>>> from endgame.visualization import RegressionReport
>>> report = RegressionReport(model, X_test, y_test, feature_names=fnames)
>>> report.save("regression_report.html", open_browser=True)
"""

from __future__ import annotations

import html as html_module
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from endgame.visualization._palettes import DEFAULT_CATEGORICAL, get_palette
from endgame.visualization._report_template import render_report
from endgame.visualization.classification_report import (
    _extract_linear_coefs,
    _extract_tree_rules,
    _is_decision_tree,
    _is_linear,
)


class RegressionReport:
    """Comprehensive regression model evaluation report.

    Generates a multi-section HTML report with metrics, charts, and
    model interpretability for any sklearn-compatible regressor.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible regressor.
    X : array-like
        Test features.
    y : array-like
        True target values.
    feature_names : list of str, optional
        Feature names.
    model_name : str, optional
        Display name for the model.
    dataset_name : str, optional
        Display name for the dataset.
    palette : str, default='tableau'
        Color palette.
    theme : str, default='dark'
        'dark' or 'light'.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> reg = RandomForestRegressor().fit(X_train, y_train)
    >>> report = RegressionReport(reg, X_test, y_test)
    >>> report.save("report.html")
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any,
        *,
        feature_names: Sequence[str] | None = None,
        model_name: str | None = None,
        dataset_name: str | None = None,
        palette: str = DEFAULT_CATEGORICAL,
        theme: str = "dark",
    ):
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y).ravel().astype(float)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.model_name = model_name or type(model).__name__
        self.dataset_name = dataset_name or ""
        self.palette = palette
        self.theme = theme

        # Predictions
        self.y_pred = np.asarray(model.predict(self.X)).ravel().astype(float)
        self.residuals = self.y - self.y_pred

        # Compute metrics
        self._metrics = self._compute_metrics()

    def _compute_metrics(self) -> dict[str, Any]:
        m = {}
        m["mae"] = round(float(mean_absolute_error(self.y, self.y_pred)), 4)
        m["mse"] = round(float(mean_squared_error(self.y, self.y_pred)), 4)
        m["rmse"] = round(float(np.sqrt(m["mse"])), 4)
        m["r2"] = round(float(r2_score(self.y, self.y_pred)), 4)
        m["explained_var"] = round(float(explained_variance_score(self.y, self.y_pred)), 4)
        m["median_ae"] = round(float(median_absolute_error(self.y, self.y_pred)), 4)
        m["max_error"] = round(float(max_error(self.y, self.y_pred)), 4)

        try:
            m["mape"] = round(float(mean_absolute_percentage_error(self.y, self.y_pred)), 4)
        except Exception:
            pass

        # Adjusted R²
        n = len(self.y)
        p = self.X.shape[1] if self.X.ndim > 1 else 1
        if n - p - 1 > 0:
            m["adj_r2"] = round(1 - (1 - m["r2"]) * (n - 1) / (n - p - 1), 4)

        m["n_samples"] = n
        m["n_features"] = p

        # Residual statistics
        m["residual_mean"] = round(float(np.mean(self.residuals)), 4)
        m["residual_std"] = round(float(np.std(self.residuals)), 4)

        return m

    @property
    def metrics(self) -> dict[str, Any]:
        """Access computed metrics dictionary."""
        return self._metrics

    def save(self, filepath: str | Path, open_browser: bool = False) -> Path:
        """Save report as self-contained HTML."""
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".html")

        html = self._render()
        filepath.write_text(html, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(filepath.resolve().as_uri())

        return filepath.resolve()

    def _repr_html_(self) -> str:
        """Jupyter inline display."""
        return self._render()

    def _render(self) -> str:
        colors = get_palette(self.palette)
        m = self._metrics

        parts = [self.model_name]
        if self.dataset_name:
            parts.append(self.dataset_name)
        parts.append(f"{m['n_samples']} samples · {m['n_features']} features")
        subtitle = html_module.escape(" — ".join(parts))

        # Metrics panel
        metrics_cards = [
            ("R²", f"{m['r2']:.4f}"),
            ("RMSE", f"{m['rmse']:.4f}"),
            ("MAE", f"{m['mae']:.4f}"),
            ("Median AE", f"{m['median_ae']:.4f}"),
            ("Max Error", f"{m['max_error']:.4f}"),
            ("Explained Var", f"{m['explained_var']:.4f}"),
        ]
        if "adj_r2" in m:
            metrics_cards.append(("Adj R²", f"{m['adj_r2']:.4f}"))
        if "mape" in m:
            metrics_cards.append(("MAPE", f"{m['mape']:.2%}"))

        metrics_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>'
            for lbl, val in metrics_cards
        )

        sections = []
        chart_w, chart_h = 600, 420

        # 1. Predicted vs Actual
        sections.append(self._section_pred_vs_actual(chart_w, chart_h, colors))

        # 2. Residual Distribution
        sections.append(self._section_residual_hist(chart_w, chart_h, colors))

        # 3. Residuals vs Predicted
        sections.append(self._section_residuals_vs_predicted(chart_w, chart_h, colors))

        # 4. Residuals vs Index (order)
        sections.append(self._section_residuals_vs_index(chart_w, chart_h, colors))

        # 5. Feature importances
        if hasattr(self.model, "feature_importances_"):
            sections.append(self._section_importances(chart_w, chart_h, colors))

        # 6. QQ plot (residuals)
        sections.append(self._section_qq(chart_w, chart_h, colors))

        footer_html = self._build_interpretability_footer()

        return render_report(
            title="Regression Report",
            subtitle=subtitle,
            theme=self.theme,
            metrics_html=metrics_html,
            sections=sections,
            footer_html=footer_html,
        )

    # ------------------------------------------------------------------
    # Chart sections
    # ------------------------------------------------------------------

    def _section_pred_vs_actual(self, w, h, colors):
        # Subsample for performance
        n = len(self.y)
        max_pts = 1000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
        else:
            idx = np.arange(n)

        y_t = self.y[idx]
        y_p = self.y_pred[idx]

        # Regression line
        lo = float(min(y_t.min(), y_p.min()))
        hi = float(max(y_t.max(), y_p.max()))

        data = {
            "yTrue": [round(float(v), 6) for v in y_t],
            "yPred": [round(float(v), 6) for v in y_p],
            "lo": round(lo, 6),
            "hi": round(hi, 6),
            "r2": self._metrics["r2"],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Predicted vs Actual",
            "chart_id": "predact",
            "width": w, "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _PREDACT_JS,
        }

    def _section_residual_hist(self, w, h, colors):
        n_bins = 40
        counts, edges = np.histogram(self.residuals, bins=n_bins)
        bins = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        data = {
            "bins": [round(float(b), 6) for b in bins],
            "counts": [int(c) for c in counts],
            "mean": round(float(np.mean(self.residuals)), 4),
            "std": round(float(np.std(self.residuals)), 4),
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Residual Distribution",
            "chart_id": "reshist",
            "width": w, "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _RESHIST_JS,
        }

    def _section_residuals_vs_predicted(self, w, h, colors):
        n = len(self.y_pred)
        max_pts = 1000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
        else:
            idx = np.arange(n)

        data = {
            "yPred": [round(float(v), 6) for v in self.y_pred[idx]],
            "residuals": [round(float(v), 6) for v in self.residuals[idx]],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Residuals vs Predicted",
            "chart_id": "respred",
            "width": w, "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _RESPRED_JS,
        }

    def _section_residuals_vs_index(self, w, h, colors):
        n = len(self.residuals)
        max_pts = 1000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
            idx.sort()
        else:
            idx = np.arange(n)

        data = {
            "indices": [int(i) for i in idx],
            "residuals": [round(float(self.residuals[i]), 6) for i in idx],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Residuals vs Sample Index",
            "chart_id": "residx",
            "width": w, "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _RESIDX_JS,
        }

    def _section_importances(self, w, h, colors):
        imp = self.model.feature_importances_
        names = self.feature_names or [f"Feature {i}" for i in range(len(imp))]
        top_n = min(20, len(imp))
        idx = np.argsort(imp)[::-1][:top_n]

        data = {
            "labels": [names[i] for i in idx],
            "values": [round(float(imp[i]), 6) for i in idx],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": f"Feature Importances (Top {top_n})",
            "chart_id": "imp",
            "width": w, "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _IMP_SECTION_JS,
        }

    def _section_qq(self, w, h, colors):
        """Normal Q-Q plot of residuals."""
        sorted_res = np.sort(self.residuals)
        n = len(sorted_res)
        # Theoretical quantiles
        from scipy.stats import norm
        theoretical = norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))

        # Subsample
        max_pts = 500
        if n > max_pts:
            idx = np.linspace(0, n - 1, max_pts, dtype=int)
        else:
            idx = np.arange(n)

        data = {
            "theoretical": [round(float(theoretical[i]), 4) for i in idx],
            "observed": [round(float(sorted_res[i]), 4) for i in idx],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Q-Q Plot (Residuals)",
            "chart_id": "qq",
            "width": w, "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _QQ_JS,
        }

    def _build_interpretability_footer(self) -> str:
        parts = []

        if _is_decision_tree(self.model):
            rules = _extract_tree_rules(self.model, self.feature_names, None)
            if rules:
                parts.append('<div class="interp-section">')
                parts.append("<h2>Decision Tree Rules</h2>")
                parts.append('<ol class="rules-list">')
                for rule in rules[:30]:
                    parts.append(f"<li>{html_module.escape(rule)}</li>")
                if len(rules) > 30:
                    parts.append(f"<li>... and {len(rules) - 30} more rules</li>")
                parts.append("</ol></div>")

        if _is_linear(self.model):
            coefs = _extract_linear_coefs(self.model, self.feature_names)
            if coefs:
                parts.append('<div class="interp-section">')
                parts.append("<h2>Model Coefficients (Top 20 by |coef|)</h2>")
                parts.append('<ol class="rules-list">')
                for name, coef in coefs[:20]:
                    sign = "+" if coef >= 0 else ""
                    parts.append(f"<li>{html_module.escape(name)}: {sign}{coef:.4f}</li>")
                if hasattr(self.model, "intercept_"):
                    intercept = float(np.asarray(self.model.intercept_).ravel()[0])
                    parts.append(f"<li>Intercept: {intercept:.4f}</li>")
                parts.append("</ol></div>")

        # Residual summary
        parts.append('<div class="report-footer">')
        parts.append("<h3>Residual Statistics</h3>")
        parts.append("<pre>")
        m = self._metrics
        parts.append(f"Mean:     {m['residual_mean']:.4f}")
        parts.append(f"Std Dev:  {m['residual_std']:.4f}")
        parts.append(f"Min:      {float(np.min(self.residuals)):.4f}")
        parts.append(f"Max:      {float(np.max(self.residuals)):.4f}")
        parts.append(f"Median:   {float(np.median(self.residuals)):.4f}")
        parts.append("</pre></div>")

        return "\n".join(parts)


# ===================================================================
# Section JavaScript
# ===================================================================

_PREDACT_JS = r"""
function renderChart_predact(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:55};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;

  const lo=data.lo,hi=data.hi;
  const pad=(hi-lo)*0.05||1;
  const xS=EG.scaleLinear([lo-pad,hi+pad],[0,iW]);
  const yS=EG.scaleLinear([lo-pad,hi+pad],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Actual');
  EG.drawYAxis(g,yS,iW,'Predicted');

  // Diagonal y=x
  g.appendChild(EG.svg('line',{x1:xS(lo-pad),y1:yS(lo-pad),x2:xS(hi+pad),y2:yS(hi+pad),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));

  // Points
  const color=config.palette[0];
  for(let i=0;i<data.yTrue.length;i++){
    const dot=EG.svg('circle',{cx:xS(data.yTrue[i]),cy:yS(data.yPred[i]),r:3,fill:color,opacity:0.5});
    dot.addEventListener('mouseenter',e=>{dot.setAttribute('r','5');dot.setAttribute('opacity','1');EG.tooltip.show(e,'Actual: '+EG.fmt(data.yTrue[i],3)+'<br>Predicted: '+EG.fmt(data.yPred[i],3)+'<br>Error: '+EG.fmt(data.yPred[i]-data.yTrue[i],3));});
    dot.addEventListener('mouseleave',()=>{dot.setAttribute('r','3');dot.setAttribute('opacity','0.5');EG.tooltip.hide();});
    g.appendChild(dot);
  }

  // R² annotation
  g.appendChild(EG.svg('text',{x:iW-5,y:16,'text-anchor':'end',fill:'var(--text-primary)','font-size':'13px','font-weight':'600'})).textContent='R² = '+data.r2.toFixed(4);
}
"""

_RESHIST_JS = r"""
function renderChart_reshist(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:50};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;

  const bins=data.bins,counts=data.counts;
  const maxC=Math.max.apply(null,counts)||1;
  const xMin=bins[0],xMax=bins[bins.length-1];
  const xPad=(xMax-xMin)*0.05||1;
  const xS=EG.scaleLinear([xMin-xPad,xMax+xPad],[0,iW]);
  const yS=EG.scaleLinear([0,maxC*1.1],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Residual');
  EG.drawYAxis(g,yS,iW,'Count');

  const barW=iW/bins.length*0.85;
  const color=config.palette[0];
  bins.forEach((b,i)=>{
    const x=xS(b)-barW/2;
    const bH=iH-yS(counts[i]);
    const rect=EG.svg('rect',{x:x,y:iH-bH,width:barW,height:Math.max(bH,0),fill:color,opacity:0.7,rx:2});
    rect.addEventListener('mouseenter',e=>{rect.setAttribute('opacity','1');EG.tooltip.show(e,'Residual ≈ '+EG.fmt(b,3)+'<br>Count: '+counts[i]);});
    rect.addEventListener('mouseleave',()=>{rect.setAttribute('opacity','0.7');EG.tooltip.hide();});
    g.appendChild(rect);
  });

  // Zero line
  if(xMin-xPad<=0 && xMax+xPad>=0){
    g.appendChild(EG.svg('line',{x1:xS(0),y1:0,x2:xS(0),y2:iH,stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'4,3',opacity:0.5}));
  }

  // Annotation
  g.appendChild(EG.svg('text',{x:iW-5,y:16,'text-anchor':'end',fill:'var(--text-secondary)','font-size':'11px'})).textContent='μ='+data.mean.toFixed(3)+' σ='+data.std.toFixed(3);
}
"""

_RESPRED_JS = r"""
function renderChart_respred(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:55};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;

  const yp=data.yPred,res=data.residuals;
  let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity;
  yp.forEach(v=>{if(v<xMin)xMin=v;if(v>xMax)xMax=v;});
  res.forEach(v=>{if(v<yMin)yMin=v;if(v>yMax)yMax=v;});
  const xPad=(xMax-xMin)*0.05||1,yPad=(yMax-yMin)*0.05||1;
  const xS=EG.scaleLinear([xMin-xPad,xMax+xPad],[0,iW]);
  const yS=EG.scaleLinear([yMin-yPad,yMax+yPad],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Predicted');
  EG.drawYAxis(g,yS,iW,'Residual');

  // Zero line
  g.appendChild(EG.svg('line',{x1:0,y1:yS(0),x2:iW,y2:yS(0),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));

  const color=config.palette[0];
  for(let i=0;i<yp.length;i++){
    const dot=EG.svg('circle',{cx:xS(yp[i]),cy:yS(res[i]),r:3,fill:color,opacity:0.5});
    dot.addEventListener('mouseenter',e=>{dot.setAttribute('r','5');dot.setAttribute('opacity','1');EG.tooltip.show(e,'Predicted: '+EG.fmt(yp[i],3)+'<br>Residual: '+EG.fmt(res[i],3));});
    dot.addEventListener('mouseleave',()=>{dot.setAttribute('r','3');dot.setAttribute('opacity','0.5');EG.tooltip.hide();});
    g.appendChild(dot);
  }
}
"""

_RESIDX_JS = r"""
function renderChart_residx(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:55};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;

  const idx=data.indices,res=data.residuals;
  const xS=EG.scaleLinear([idx[0],idx[idx.length-1]],[0,iW]);
  let yMin=Infinity,yMax=-Infinity;
  res.forEach(v=>{if(v<yMin)yMin=v;if(v>yMax)yMax=v;});
  const yPad=(yMax-yMin)*0.05||1;
  const yS=EG.scaleLinear([yMin-yPad,yMax+yPad],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Sample Index');
  EG.drawYAxis(g,yS,iW,'Residual');

  g.appendChild(EG.svg('line',{x1:0,y1:yS(0),x2:iW,y2:yS(0),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));

  const color=config.palette[0];
  for(let i=0;i<idx.length;i++){
    g.appendChild(EG.svg('circle',{cx:xS(idx[i]),cy:yS(res[i]),r:2.5,fill:color,opacity:0.45}));
  }
}
"""

_QQ_JS = r"""
function renderChart_qq(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:55};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;

  const th=data.theoretical,obs=data.observed;
  let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity;
  th.forEach(v=>{if(v<xMin)xMin=v;if(v>xMax)xMax=v;});
  obs.forEach(v=>{if(v<yMin)yMin=v;if(v>yMax)yMax=v;});
  const xPad=(xMax-xMin)*0.05||1,yPad=(yMax-yMin)*0.05||1;
  const xS=EG.scaleLinear([xMin-xPad,xMax+xPad],[0,iW]);
  const yS=EG.scaleLinear([yMin-yPad,yMax+yPad],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Theoretical Quantiles');
  EG.drawYAxis(g,yS,iW,'Observed Residuals');

  // Reference line (fit through Q1 and Q3)
  const lo=Math.min(xMin-xPad,yMin-yPad),hi=Math.max(xMax+xPad,yMax+yPad);
  // Simple diagonal for normal reference
  g.appendChild(EG.svg('line',{x1:xS(xMin-xPad),y1:yS(xMin-xPad),x2:xS(xMax+xPad),y2:yS(xMax+xPad),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));

  const color=config.palette[0];
  for(let i=0;i<th.length;i++){
    g.appendChild(EG.svg('circle',{cx:xS(th[i]),cy:yS(obs[i]),r:3,fill:color,opacity:0.5}));
  }
}
"""

# Reuse importance JS from classification report
_IMP_SECTION_JS = r"""
function renderChart_imp(data, config, container) {
  const margin={top:10,right:30,bottom:30,left:140};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const n=data.labels.length;
  const rowH=iH/n;
  const maxV=Math.max.apply(null,data.values)||1;
  const xS=EG.scaleLinear([0,maxV],[0,iW]);

  for(let i=0;i<n;i++){
    const y=i*rowH,v=data.values[i];
    const color=config.palette[i%config.palette.length];
    const bW=xS(v);
    const rect=EG.svg('rect',{x:0,y:y+2,width:Math.max(bW,2),height:rowH-4,fill:color,rx:3,opacity:0.8});
    rect.addEventListener('mouseenter',e=>{rect.setAttribute('opacity','1');EG.tooltip.show(e,'<b>'+EG.esc(data.labels[i])+'</b><br>'+EG.fmt(v,4));});
    rect.addEventListener('mouseleave',()=>{rect.setAttribute('opacity','0.8');EG.tooltip.hide();});
    g.appendChild(rect);
    g.appendChild(EG.svg('text',{x:bW+5,y:y+rowH/2+4,fill:'var(--text-secondary)','font-size':'10px'})).textContent=EG.fmt(v,4);
    g.appendChild(EG.svg('text',{x:-6,y:y+rowH/2+4,'text-anchor':'end',fill:'var(--text-primary)','font-size':'11px'})).textContent=data.labels[i].length>20?data.labels[i].slice(0,18)+'…':data.labels[i];
  }
}
"""
