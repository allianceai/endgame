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

The agent never has to read 300+ class definitions, but it can reach all of them:

1. **Server instructions** (sent at connection) name all 31 modules and the experiment workflow
2. **Guidance and discovery**: `guide` (how to run a strong, honest experiment), `list_modules` / `describe_api` (every module, class and signature), `recommend_models` (which models to compare for this table)
3. **Dedicated tools** for the common steps: load, check, engineer features, select features, compare models (GBDTs, tabular foundation models, other families), ensemble, evaluate, explain, visualize, export
4. **General access** to every module: `transform_data` (any transformer by class path), `train_model(model_name="endgame.<module>.<Class>")` (any estimator), `run_python` (anything, with the session's datasets and models)
5. A **SessionManager** tracks datasets, models (with their out-of-fold predictions) and artifacts across tool calls via short IDs (`ds_a1b2c3d4`, `model_e5f6g7h8`)

```
User: "Predict which rookies earn starting roles from their combine tracking"
  → guide()                                                     (workflow)
  → load_data(players.csv, target_column=...) ; load_data(tracking.csv)
  → check_data_quality(...)
  → engineer_features(ds_players, [aggregate tracking per player with signal features,
                                   join combine results, z-score within position])
  → select_features(ds_features, method="mrmr", n_features=20)
  → recommend_models(ds_sel)                                    (GBDTs + Kumo-Tabular, LimiX, TabPFN, ...)
  → compare_models(ds_sel, group_column="team")                 (same folds, out-of-fold predictions kept)
  → ensemble(model_ids=[...], method="hill_climbing")
  → evaluate_model / explain_model / export_script
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
- `columns` — Read only these columns (large files)

**inspect_data** operations:
- `summary` — Shape, dtypes, missing values, meta-features
- `describe` — Statistical summary (mean, std, quartiles)
- `correlations` — Top 20 pairwise correlations
- `missing` — Missing value counts and percentages
- `distribution` — Value counts or quantile stats for a column
- `head` — First 10 rows
- `dtypes` — Column data types

### Discovery and guidance (6 tools)

| Tool | Purpose |
|------|---------|
| `guide` | How to run the experiment: workflow, validation, features, selection, models, ensembling, small data, time series, beyond tabular, code. |
| `list_modules` | Every module with its purpose and the tools that reach it; `module=` lists its classes and functions; `search=` finds them across modules. |
| `describe_api` | Signature, parameters, docstring and methods of any Endgame (or sklearn) class or function. |
| `list_models` | Search available models by task type, family, interpretability, speed. |
| `recommend_models` | What to compare on this table: GBDTs; tabular foundation models ranked by TabArena Elo on tables up to 50k rows (without a GPU only the faster ones); neural and interpretable models with more time; a linear baseline. Lists models that need a package or licence. |
| `describe_model` | Full metadata for a model (params, capabilities, speed, notes). |

**list_models** filters:
- `task_type` — `"classification"` or `"regression"`
- `family` — `"gbdt"`, `"neural"`, `"tree"`, `"linear"`, `"kernel"`, `"rules"`, `"bayesian"`, `"foundation"`, `"ensemble"`
- `interpretable_only` — Only glass-box models
- `fast_only` — Exclude slow/very_slow models
- `max_samples` — Only models that scale to N samples

### Training (4 tools)

| Tool | Purpose |
|------|---------|
| `train_model` | Train one model with cross-validation; keeps its out-of-fold predictions and a final fit. |
| `compare_models` | Train several models on the same folds and rank them (default: `recommend_models`' picks). A model that fails is reported, not fatal. |
| `automl` | Full AutoML pipeline (preprocessing → training → ensembling). |
| `quick_compare` | Quick leaderboard from `eg.quick` presets; models are not kept. |

**train_model** parameters:
- `dataset_id` — From `load_data`
- `model_name` — Registry key (e.g. `"lgbm"`, `"kumo_tabular"`, `"ebm"`) or the class path of any estimator (`"endgame.models.trees.RotationForestClassifier"`, `"sklearn.svm.SVC"`)
- `params` — Hyperparameter overrides, a dict or JSON string: `{"n_estimators": 500}`
- `cv_folds` — Number of CV folds (default 5)
- `time_ordered` — Rows are in time order: each fold trains on earlier rows, is scored on later ones
- `group_column` — Rows sharing this value (a player, patient, site) stay in one fold; the column is not a feature

Integer columns named like an id (`player_id`, `nfl_id`, `ID`) with a different value in every row are left out of the features, like unique text ids.

**automl** presets: `best_quality`, `high_quality`, `good_quality`, `medium_quality`, `fast`, `interpretable`. `high_quality` and above include Kumo-Tabular, TabPFN-3.5, TabICL and Causilo on tables up to 50k rows.

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

### Features (4 tools)

| Tool | Purpose |
|------|---------|
| `engineer_features` | Build features: aggregate a long table per entity (statistics and signal features: entropy, fractal dimension, spectra, Hjorth, ...), join tables, within-group normalisation, formulas, interactions, lags/rolling, out-of-fold target encoding, frequency encoding, datetime parts, ranks. |
| `select_features` | 18 methods from `eg.feature_selection` (mrmr, boruta, stability, knockoff, null importance, SHAP, ...); `apply_to` keeps the same columns in held-out datasets. |
| `transform_data` | Apply any transformer by class path (imputers, encoders, resamplers, PCA/UMAP, signal transforms, sklearn). |
| `preprocess` | Chain basic operations (impute, scale, encode, balance, quick filter, drop). Returns a new dataset ID. |

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

**engineer_features** example (one row per player from 10 Hz tracking frames):
```json
[
  {"type": "aggregate", "source": "ds_tracking", "by": ["player_id"], "columns": ["speed", "accel"],
   "aggs": ["mean", "max", "q90", "sample_entropy", "higuchi_fd"], "order_by": "time",
   "filter": "drill == 'shuttle'", "prefix": "shuttle_"},
  {"type": "group_normalize", "by": "position", "method": "zscore"},
  {"type": "formula", "name": "speed_per_lb", "expr": "shuttle_speed_max / weight"}
]
```

### Ensembling (1 tool)

| Tool | Purpose |
|------|---------|
| `ensemble` | Combine models trained on the same dataset and folds from their out-of-fold predictions: `hill_climbing`, `stacking` (scored with nested CV), `optimized`, `mean`, `rank_average`. Reports the blend next to each member; the result works with `predict` and `evaluate_model`. |

### Code (1 tool)

| Tool | Purpose |
|------|---------|
| `run_python` | Run Python in the session: `eg`, `np`, `pd`, `pl`, plus `dataset(id)`, `add_dataset(df, name, target)`, `model(id)`, `add_model(...)`. Variables persist. For modules without a dedicated tool (survival, calibration, fairness, NLP, vision, tuning, custom CV). |

`run_python` runs arbitrary code with the server's permissions and is annotated as destructive, so clients can ask before each call. Set `ENDGAME_MCP_ALLOW_CODE=0` to remove it.

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
| `export_script` | Generate a standalone Python script that reruns the model's own cross-validation (same folds, features, encoding) and prints the metrics the server reported. |
| `save_model` | Save trained model to disk (`.egm` format). |

### Advanced (3 tools)

| Tool | Purpose |
|------|---------|
| `cluster` | Clustering: `auto`, `kmeans`, `hdbscan`, `dbscan`, `agglomerative`, `gaussian_mixture`. |
| `detect_anomalies` | Outlier detection: `isolation_forest`, `lof`, `elliptic_envelope`. |
| `forecast` | Time series forecasting: `auto`/`arima`, `ets`, `theta`, `naive`. |

### Kaggle (6 tools)

Uses your Kaggle API credentials (`~/.kaggle/`). Joining a competition and accepting its rules happens on kaggle.com; there is no API for it.

| Tool | Description |
|------|-------------|
| `kaggle_competition` | Competition details (deadline, prize, whether you've joined), its data files, and the text of its overview, evaluation, timeline and data pages (`include_rules=True` adds the rules). |
| `kaggle_download` | Download and unzip a competition's data (default `~/.endgame/competitions/<slug>/raw`). |
| `kaggle_notebooks` | List a competition's public notebooks, hottest first (or by votes, date, score). |
| `kaggle_read_notebook` | Read a public notebook as markdown plus fenced code, without outputs. |
| `kaggle_push_notebook` | Upload a local `.ipynb`/`.py` and run it on Kaggle with the competition data attached. Private unless `public=True`. |
| `kaggle_notebook_status` | Whether a pushed notebook's run is queued, running, complete or failed, with the end of its log; optionally downloads its output files. |

## Resources Reference

Resources are read-only catalogs the LLM can browse without making a tool call — zero overhead for discovery.

| URI | Content |
|-----|---------|
| `endgame://catalog/modules` | All 31 modules: purpose and the tools that reach each |
| `endgame://guide/workflow` | The experiment guide (same text as the `guide` tool) |
| `endgame://catalog/models` | Every registry model grouped by family with name, fit time, and description |
| `endgame://catalog/presets` | 6 AutoML presets with time limits, model pools, and settings |
| `endgame://catalog/visualizers` | Available chart types with required inputs |
| `endgame://catalog/metrics` | Classification + regression metrics with descriptions |
| `endgame://session/state` | Current loaded datasets, trained models, and visualizations |
| `endgame://guide/examples` | Example workflows: a full tabular experiment, entity-level prediction from sensor data, time-ordered rows, using any module |

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

### GPU memory and foundation models

On a CUDA machine, foundation models (family `foundation`: Kumo-Tabular, LimiX-2, TabPFN, TabFM, Mitra, iLTM, ...) never hold GPU memory in the server process. `train_model` cross-validates them in a child process that exits when it is done, and the stored model fits itself in a single worker process the first time it is used (`predict`, `evaluate_model`, charts) and keeps serving calls from there. At most one worker lives at a time: training anything, or using a different foundation model, stops it, which frees all of its GPU memory. A session can therefore train any number of them on one small GPU; the price is a refit when you go back to an earlier model (seconds for in-context models, a minute or two for Mitra and iLTM, same parameters and seeds).

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
| `ENDGAME_MCP_ALLOW_CODE` | `1` | `0` removes the `run_python` tool |

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
