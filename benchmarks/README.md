# Endgame Benchmarks

Reproducible benchmarks for the Endgame ML framework.

## Experiments

### Experiment 1: Tabular Model Benchmark

**Script:** `run_tabular_benchmark.py`

Benchmarks 20+ models across the full Grinsztajn NeurIPS 2022 benchmark suite (~45 datasets, classification + regression).

**Reference:** Grinsztajn et al., "Why do tree-based models still outperform deep learning on typical tabular data?" (NeurIPS 2022)

**OpenML Suites:** 337 (classif/numerical), 334 (classif/categorical), 336 (regression/numerical), 335 (regression/categorical)

**Models:** LightGBM, XGBoost, CatBoost, EBM, MARS, RuleFit, GAM, NAM, TAN, ESKDB, FT-Transformer, NODE, TabPFN, Rotation Forest, C5.0, RandomForest, LogisticRegression, NaiveBayes, TabNet

**Metrics:** accuracy, balanced_accuracy, f1_weighted, roc_auc, brier_score, log_loss, ECE, fit_time_s

**Protocol:** 5-fold StratifiedKFold, max 5000 samples, 600s timeout per model

```bash
# Full run (~45 datasets, ~8-12 hours)
python benchmarks/run_tabular_benchmark.py

# Quick test (3 datasets, fast models only)
python benchmarks/run_tabular_benchmark.py --quick

# Classification datasets only
python benchmarks/run_tabular_benchmark.py --suite grinsztajn-classif
```

### Experiment 2: Calibration Benchmark

**Script:** `run_calibration_benchmark.py`

Measures the effect of 6 calibration methods on 3 base models across 15 binary classification datasets.

**Calibration methods:** PlattScaling, IsotonicCalibration, BetaCalibration, TemperatureScaling, HistogramBinning, VennABERS

**Metrics:** ECE, Brier score, log loss, conformal coverage at alpha=0.05

```bash
# Full run (~1 hour)
python benchmarks/run_calibration_benchmark.py

# Quick test
python benchmarks/run_calibration_benchmark.py --quick
```

### Experiment 3: Glass-Box vs Black-Box

**Script:** `run_glassbox_benchmark.py`

Compares interpretable models against black-box GBDTs to measure the accuracy-interpretability trade-off.

```bash
# Full run (~2-3 hours)
python benchmarks/run_glassbox_benchmark.py
```

## Output

Results are saved to `results/` as Parquet files:

```
results/
├── tabular_benchmark.parquet
├── calibration_benchmark.parquet
└── glassbox_benchmark.parquet
```

## Requirements

```bash
pip install endgame-ml[tabular,benchmark]
```

## Reproduction

All experiments use fixed random seeds (42) and deterministic splits. Results should be identical across runs on the same hardware. GPU is not required but will accelerate neural models.
