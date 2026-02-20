# MCP Server

Endgame ships an [MCP](https://modelcontextprotocol.io/) server that lets any MCP-compatible LLM host (Claude Code, Claude Desktop, VS Code Copilot, etc.) build ML pipelines through natural language.

Instead of registering 300+ tools, the server exposes **20 meta-tools** and **6 resources** — keeping schema overhead under 2K tokens while giving the LLM full access to the toolkit.

## Installation

```bash
pip install endgame-ml[mcp]
# or, if already installed:
pip install "mcp>=1.2.0"
```

## Setup

### Claude Code

Add `.mcp.json` to your project root (Endgame ships one by default):

```json
{
  "mcpServers": {
    "endgame": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["-m", "endgame.mcp"]
    }
  }
}
```

Restart Claude Code. The server auto-starts on first tool call.

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "endgame": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["-m", "endgame.mcp"]
    }
  }
}
```

### Manual / Standalone

```bash
# stdio transport (default — used by MCP hosts)
python -m endgame.mcp

# SSE transport (for web-based clients)
python -m endgame.mcp --sse
```

## How It Works

The LLM never sees 300+ model definitions. Instead:

1. **Resources** (zero-cost) let the LLM browse the model catalog, presets, metrics, and visualizers without a tool call
2. **Discovery tools** help the LLM find the right model for the dataset
3. **Action tools** load data, train, evaluate, visualize, and export
4. A **SessionManager** tracks loaded datasets, trained models, and artifacts across tool calls via short IDs (`ds_a1b2c3d4`, `model_e5f6g7h8`)

```
User: "Build a classifier to predict loan defaults"
  → LLM reads endgame://catalog/models        (browse 97 models)
  → LLM calls load_data(source="loans.csv", target_column="default")
  → LLM calls recommend_models(dataset_id="ds_...", time_budget="medium")
  → LLM calls train_model(dataset_id="ds_...", model_name="lgbm")
  → LLM calls evaluate_model(model_id="model_...")
  → LLM calls create_visualization(chart_type="roc_curve", model_id="model_...")
  → LLM calls export_script(model_id="model_...")
```

## Tools Reference

### Data (3 tools)

| Tool | Purpose |
|------|---------|
| `load_data` | Load from CSV/Parquet/URL/OpenML. Auto-detects task type. Returns dataset ID. |
| `inspect_data` | Explore a dataset: summary, describe, correlations, missing, distribution, head, dtypes. |
| `split_data` | Create stratified train/test splits. Returns two new dataset IDs. |

**load_data** parameters:
- `source` — File path, URL, or `"openml:31"` / `"openml:credit-g"`
- `target_column` — Name of the target column
- `name` — Optional display name
- `sample_n` — Subsample to N rows

**inspect_data** operations:
- `summary` — Shape, dtypes, missing values, meta-features
- `describe` — Statistical summary (mean, std, quartiles)
- `correlations` — Top 20 pairwise correlations
- `missing` — Missing value counts and percentages
- `distribution` — Value counts or quantile stats for a column
- `head` — First 10 rows
- `dtypes` — Column data types

### Discovery (3 tools)

| Tool | Purpose |
|------|---------|
| `list_models` | Search available models by task type, family, interpretability, speed. |
| `recommend_models` | Smart recommendations based on dataset meta-features and time budget. |
| `describe_model` | Full metadata for a model (params, capabilities, speed, notes). |

**list_models** filters:
- `task_type` — `"classification"` or `"regression"`
- `family` — `"gbdt"`, `"neural"`, `"tree"`, `"linear"`, `"kernel"`, `"rules"`, `"bayesian"`, `"foundation"`, `"ensemble"`
- `interpretable_only` — Only glass-box models
- `fast_only` — Exclude slow/very_slow models
- `max_samples` — Only models that scale to N samples

### Training (3 tools)

| Tool | Purpose |
|------|---------|
| `train_model` | Train a single model with cross-validation. Returns model ID + metrics. |
| `automl` | Full AutoML pipeline (preprocessing → training → ensembling). |
| `quick_compare` | Quick multi-model comparison with leaderboard. |

**train_model** parameters:
- `dataset_id` — From `load_data`
- `model_name` — Registry key (e.g. `"lgbm"`, `"xgb"`, `"ebm"`)
- `params` — JSON string of hyperparameter overrides: `'{"n_estimators": 500}'`
- `cv_folds` — Number of CV folds (default 5)
- `metric` — Evaluation metric (default `"auto"`)

**automl** presets: `best_quality`, `high_quality`, `good_quality`, `medium_quality`, `fast`, `interpretable`

### Evaluation (2 tools)

| Tool | Purpose |
|------|---------|
| `evaluate_model` | Compute metrics on test data or OOF predictions. |
| `explain_model` | Feature importance (`importance`) or permutation importance (`permutation`). |

**evaluate_model** metrics (comma-separated string):
- Classification: `accuracy`, `roc_auc`, `f1`, `precision`, `recall`, `balanced_accuracy`, `log_loss`, `matthews_corrcoef`, `cohen_kappa`
- Regression: `rmse`, `r2`, `mae`, `mape`, `median_ae`, `max_error`, `explained_variance`

### Prediction (1 tool)

| Tool | Purpose |
|------|---------|
| `predict` | Generate predictions, optionally save to CSV. Supports probabilities. |

### Preprocessing (1 tool)

| Tool | Purpose |
|------|---------|
| `preprocess` | Chain preprocessing operations. Returns a new dataset ID. |

**Operations** (JSON array):
```json
[
  {"type": "impute", "strategy": "median"},
  {"type": "scale", "method": "standard"},
  {"type": "encode", "method": "label"},
  {"type": "balance", "method": "smote"},
  {"type": "select_features", "method": "mutual_info", "top_k": 20},
  {"type": "drop_columns", "columns": ["id", "name"]}
]
```

### Visualization (2 tools)

| Tool | Purpose |
|------|---------|
| `create_visualization` | Generate a self-contained HTML chart. |
| `create_report` | Full classification or regression evaluation report. |

**Chart types:**
- ML evaluation: `roc_curve`, `pr_curve`, `confusion_matrix`, `calibration_plot`, `lift_chart`, `feature_importance`
- Data exploration: `histogram`, `scatterplot`, `heatmap`, `box_plot`, `bar_chart`, `line_chart`

### Export (2 tools)

| Tool | Purpose |
|------|---------|
| `export_script` | Generate a standalone Python script reproducing the pipeline. |
| `save_model` | Save trained model to disk (`.egm` format). |

### Advanced (3 tools)

| Tool | Purpose |
|------|---------|
| `cluster` | Clustering: `auto`, `kmeans`, `hdbscan`, `dbscan`, `agglomerative`, `gaussian_mixture`. |
| `detect_anomalies` | Outlier detection: `isolation_forest`, `lof`, `elliptic_envelope`. |
| `forecast` | Time series forecasting: `auto`/`arima`, `ets`, `theta`, `naive`. |

## Resources Reference

Resources are read-only catalogs the LLM can browse without making a tool call — zero overhead for discovery.

| URI | Content |
|-----|---------|
| `endgame://catalog/models` | All 97 models grouped by family with name, fit time, and description |
| `endgame://catalog/presets` | 6 AutoML presets with time limits, model pools, and settings |
| `endgame://catalog/visualizers` | Available chart types with required inputs |
| `endgame://catalog/metrics` | Classification + regression metrics with descriptions |
| `endgame://session/state` | Current loaded datasets, trained models, and visualizations |
| `endgame://guide/examples` | Example workflows for common ML tasks |

## Example Workflows

### Train a single model

```
You: Load iris.csv and train a LightGBM classifier

LLM calls:
  load_data(source="iris.csv", target_column="species")
  train_model(dataset_id="ds_...", model_name="lgbm")
  evaluate_model(model_id="model_...")
```

### Full AutoML

```
You: Run AutoML on my dataset with high quality

LLM calls:
  load_data(source="data.csv", target_column="label")
  automl(dataset_id="ds_...", preset="high_quality")
```

### Interpretable pipeline

```
You: I need an interpretable model for regulatory compliance

LLM calls:
  load_data(source="loans.csv", target_column="default")
  list_models(task_type="classification", interpretable_only=true)
  train_model(dataset_id="ds_...", model_name="ebm")
  explain_model(model_id="model_...", method="importance")
  create_report(model_id="model_...")
  export_script(model_id="model_...")
```

### Data exploration

```
You: Explore this dataset and show me the correlations

LLM calls:
  load_data(source="housing.csv", target_column="price")
  inspect_data(dataset_id="ds_...", operation="summary")
  inspect_data(dataset_id="ds_...", operation="correlations")
  create_visualization(chart_type="heatmap", dataset_id="ds_...")
  create_visualization(chart_type="histogram", dataset_id="ds_...", params='{"column": "price"}')
```

### Preprocessing + training

```
You: Impute missing values, scale features, then train XGBoost

LLM calls:
  load_data(source="messy_data.csv", target_column="outcome")
  preprocess(dataset_id="ds_...", operations='[{"type":"impute","strategy":"median"},{"type":"scale","method":"standard"}]')
  train_model(dataset_id="ds_preprocessed_...", model_name="xgb")
```

## Session Management

Every artifact gets a short ID:

- **Datasets**: `ds_a1b2c3d4`
- **Models**: `model_e5f6g7h8`
- **Visualizations**: `viz_i9j0k1l2`

These IDs are passed between tools to chain operations. The `endgame://session/state` resource shows all current artifacts at any time.

Artifacts live in memory for the duration of the server process. Files (visualizations, exported scripts, saved models) are written to the working directory (`/tmp/endgame_mcp` by default, configurable via `ENDGAME_MCP_WORKDIR`).

## Error Handling

All tools return structured JSON with consistent format:

```text
// Success
{"status": "ok", "dataset_id": "ds_a1b2c3d4", "shape": [1000, 15], ...}

// Error
{"status": "error", "error_type": "not_found", "message": "Dataset 'ds_xxx' not found", "hint": "Use load_data() first"}
```

Error types: `not_found`, `validation`, `missing_dependency`, `timeout`, `internal`.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ENDGAME_MCP_WORKDIR` | `/tmp/endgame_mcp` | Working directory for output files |
| `ENDGAME_MCP_TIMEOUT` | `600` | Max seconds for training operations before timeout |

## Troubleshooting

### Categorical features produce wrong predictions

Endgame's MCP server stores fitted label encoders from training and reuses them during evaluation and prediction. This ensures that categorical values like `"red" -> 0, "green" -> 1` are encoded consistently across the entire pipeline. If you see unexpected predictions on categorical data, verify you are using a model trained through the MCP server (which stores encoders automatically).

Categories that appear in test data but were not seen during training are encoded as `-1`.

### Forecasting fails with "missing_dependency"

The `forecast` tool requires `statsforecast` for ARIMA, ETS, and Theta methods. Install it with:

```bash
pip install statsforecast
```

The `naive` method works without extra dependencies and returns the last observed value repeated for the forecast horizon.

### Training hangs or takes too long

Training operations have a configurable timeout (default: 10 minutes). Set a custom timeout via the `ENDGAME_MCP_TIMEOUT` environment variable:

```bash
ENDGAME_MCP_TIMEOUT=300 python -m endgame.mcp  # 5-minute timeout
```

If training consistently times out, try:
- A simpler model (e.g., `lgbm` instead of `ft_transformer`)
- A smaller dataset (use `sample_n` parameter in `load_data`)
- The `fast` preset for `automl`

### ROC/PR curves fail on multiclass problems

ROC curves and PR curves require binary classification. For multiclass problems, use `confusion_matrix` instead:

```
create_visualization(chart_type="confusion_matrix", model_id="model_...")
```

### Server stdout corruption

If you see garbled output or JSON parse errors, ensure no Endgame code is printing to stdout. The MCP server redirects stdout to stderr during tool calls, but any code running outside tool calls could corrupt the stdio transport. Use the `--sse` flag for debugging:

```bash
python -m endgame.mcp --sse
```
