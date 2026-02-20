# AutoML Framework Development Tracker

## Overview

This document tracks the development progress of Endgame's AutoML framework, which aims to match AutoGluon's 3-line simplicity while leveraging Endgame's 100+ models, meta-learning infrastructure, and competition-winning techniques.

**Target API:**
```python
from endgame.automl import AutoMLPredictor
predictor = AutoMLPredictor(label="target").fit("train.csv", presets="best_quality")
predictions = predictor.predict("test.csv")
```

**Priority Order:** Tabular → Text → Vision → TimeSeries → Audio → MultiModal

---

## Sprint Status

### Sprint 1: Core Infrastructure (Foundation) - COMPLETED

| Task | Status | File(s) |
|------|--------|---------|
| Create module structure | Done | `endgame/automl/` |
| Implement `presets.py` | Done | `presets.py` |
| Implement `base.py` with `BasePredictor` | Done | `base.py` |
| Implement `time_manager.py` | Done | `time_manager.py` |
| Implement `model_registry.py` | Done | `model_registry.py` (34 models) |
| Implement `search/base.py` | Done | `search/base.py` |
| Implement `search/portfolio.py` | Done | `search/portfolio.py` |
| Implement `orchestrator.py` | Done | `orchestrator.py` |
| Implement `utils/data_loader.py` | Done | `utils/data_loader.py` |
| Implement `tabular.py` | Done | `tabular.py` |
| Implement `predictor.py` | Done | `predictor.py` |
| Implement `__init__.py` | Done | `__init__.py` |
| Write tests | Done | `tests/test_automl.py` (32 tests) |

### Sprint 2: TabularPredictor MVP - COMPLETED

| Task | Status | File(s) |
|------|--------|---------|
| Implement `search/heuristic.py` | Done | `search/heuristic.py` |
| Implement `search/genetic.py` | Done | `search/genetic.py` |
| Implement `search/random.py` | Done | `search/random.py` |
| Implement `search/bayesian.py` | Done | `search/bayesian.py` |
| Implement `preprocessing/auto_preprocessor.py` | Done | `preprocessing/auto_preprocessor.py` |
| Implement `stacker.py` (AutoMLStacker) | Done | `stacker.py` |
| Integration tests for TabularPredictor | Pending | `tests/test_automl.py` |

### Sprint 3: TextPredictor - COMPLETED

| Task | Status | File(s) |
|------|--------|---------|
| Implement `text.py` | Done | `text.py` |
| Integrate TransformerClassifier | Done | `text.py` |
| Integrate DomainAdaptivePretrainer | Done | `text.py` |
| Integrate PseudoLabelTrainer | Done | `text.py` |
| Text-specific preprocessing | Done | `text.py` |
| Auto-detect text columns | Done | `text.py` |
| Write NLP tests | Done | `tests/test_automl.py` (20 tests) |

### Sprint 4: Domain Predictors - COMPLETED

| Task | Status | File(s) |
|------|--------|---------|
| Implement `VisionPredictor` | Done | `vision.py` |
| Integrate VisionBackbone + TTA | Done | `vision.py` |
| Implement `TimeSeriesPredictor` | Done | `timeseries.py` |
| Integrate ROCKET classifiers | Done | `timeseries.py` |
| Implement `AudioPredictor` | Done | `audio.py` |
| Integrate SEDModel + SpectrogramTransformer | Done | `audio.py` |
| Implement `MultiModalPredictor` | Done | `multimodal.py` |
| Multi-modal fusion strategies | Done | `multimodal.py` |
| Write domain predictor tests | Done | `tests/test_automl.py` (40+ tests) |

### Sprint 5: Meta-Learning Bootstrap - NOT STARTED

| Task | Status | File(s) |
|------|--------|---------|
| Run benchmarks on OpenML-CC18 | Pending | - |
| Train and save meta-learner | Pending | - |
| Package pre-trained meta-learner | Pending | - |
| Add exploration mode | Pending | - |

---

## Available Presets

| Preset | Time Limit | CV Folds | Models | Tuning | Ensemble |
|--------|-----------|----------|--------|--------|----------|
| `best_quality` | None | 8 | All (12+) | 100 trials | 2-level stack |
| `high_quality` | 4 hours | 5 | 7 models | 50 trials | 1-level stack |
| `good_quality` | 1 hour | 5 | 5 models | 25 trials | Hill climbing |
| `medium_quality` | 15 min | 5 | 4 models | 10 trials | Hill climbing |
| `fast` | 5 min | 3 | 1 model (LGBM) | No | No |
| `interpretable` | 15 min | 5 | EBM, NAM, Linear | 25 trials | No |

---

## Search Strategies

| Strategy | Description | Status |
|----------|-------------|--------|
| `portfolio` | Train diverse model portfolio in parallel (default) | Done |
| `heuristic` | Data-driven rules based on meta-features | Done |
| `genetic` | Evolutionary optimization of pipelines | Done |
| `random` | Random valid pipeline sampling | Done |
| `bayesian` | Optuna-based Bayesian optimization | Done |

---

## Model Registry

41 models registered across families:
- **GBDT**: lgbm, xgb, catboost, ngboost
- **Neural**: mlp, ft_transformer, tab_transformer, tabnet, saint, node, nam, tabular_resnet, gandalf, embedding_mlp, modern_nca
- **Linear**: linear, lda, qda, elm, mars
- **Tree**: c50, rotation_forest, oblique_forest, evolutionary_tree, quantile_forest
- **Kernel**: svm, gp, knn
- **Rules**: furia, rulefit
- **Bayesian**: naive_bayes, kdb, tan, eskdb, bart, neural_kdb
- **Foundation**: tabpfn
- **Ensemble**: ebm, ebmc, ebmr

**Note:** Use `register_model()` and `unregister_model()` to dynamically add/remove models at runtime.

---

## File Structure

```
endgame/automl/
├── __init__.py              # Public API exports
├── predictor.py             # AutoMLPredictor (unified entry)
├── base.py                  # BasePredictor abstract class
├── tabular.py               # TabularPredictor
├── text.py                  # TextPredictor (Sprint 3)
├── vision.py                # VisionPredictor (Sprint 4)
├── timeseries.py            # TimeSeriesPredictor (Sprint 4)
├── audio.py                 # AudioPredictor (Sprint 4)
├── multimodal.py            # MultiModalPredictor (Sprint 4) - NEW
├── orchestrator.py          # PipelineOrchestrator
├── time_manager.py          # TimeBudgetManager
├── model_registry.py        # Centralized model registry (64 models)
├── presets.py               # Preset configurations
├── preprocessing_registry.py # Preprocessing method registry
├── stacker.py               # AutoMLStacker (Sprint 2)
├── DEVELOPMENT.md           # This file
├── search/
│   ├── __init__.py
│   ├── base.py              # BaseSearchStrategy
│   ├── portfolio.py         # PortfolioSearch (default)
│   ├── heuristic.py         # HeuristicSearch (Sprint 2)
│   ├── genetic.py           # GeneticSearch (Sprint 2)
│   ├── random.py            # RandomSearch (Sprint 2)
│   └── bayesian.py          # BayesianSearch (Sprint 2)
├── preprocessing/
│   ├── __init__.py          # (Sprint 2)
│   └── auto_preprocessor.py # AutoPreprocessor (Sprint 2)
└── utils/
    ├── __init__.py
    └── data_loader.py       # DataLoader, load_data, infer_task_type
```

---

## Usage Examples

### Basic Usage
```python
from endgame.automl import AutoMLPredictor

predictor = AutoMLPredictor(label="target").fit("train.csv")
predictions = predictor.predict("test.csv")
```

### With Options
```python
from endgame.automl import TabularPredictor

predictor = TabularPredictor(
    label="price",
    presets="best_quality",
    time_limit=3600,
    search_strategy="portfolio",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)
```

### Different Search Strategies
```python
# Genetic algorithm search
predictor = AutoMLPredictor(label="target", search_strategy="genetic")

# Bayesian optimization
predictor = AutoMLPredictor(label="target", search_strategy="bayesian")
```

### Text/NLP Classification
```python
from endgame.automl import TextPredictor

# Basic text classification
predictor = TextPredictor(
    label="sentiment",
    text_column="review",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)

# With domain-adaptive pretraining and pseudo-labeling
predictor = TextPredictor(
    label="category",
    text_column="description",
    use_dapt=True,               # Domain-adaptive pretraining
    use_pseudo_labeling=True,    # Use test data for pseudo-labeling
    pseudo_threshold=0.95,       # Confidence threshold
    presets="best_quality",
)
predictor.fit(train_df, unlabeled_data=test_df)
predictions = predictor.predict(test_df)
```

### Image Classification
```python
from endgame.automl import VisionPredictor

# Basic image classification
predictor = VisionPredictor(
    label="species",
    image_column="image_path",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)

# With custom backbones and TTA
predictor = VisionPredictor(
    label="category",
    image_column="filepath",
    model_presets=["efficientnet_b3", "swin_tiny_patch4_window7_224"],
    augmentation="heavy",
    use_tta=True,
    tta_augmentations=["identity", "hflip", "vflip"],
    presets="best_quality",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)
```

### Time Series Classification
```python
from endgame.automl import TimeSeriesPredictor

# With numpy arrays (n_samples, n_channels, n_timesteps)
predictor = TimeSeriesPredictor(label="activity")
predictor.fit(X_train, y_train)
predictions = predictor.predict(X_test)

# With DataFrame (wide format) and multiple classifiers
predictor = TimeSeriesPredictor(
    label="class",
    classifier_presets=["minirocket", "hydra_minirocket"],
    use_feature_extraction=True,
    presets="best_quality",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)
```

### Audio/Sound Classification
```python
from endgame.automl import AudioPredictor

# Basic audio classification
predictor = AudioPredictor(
    label="species",
    audio_column="audio_path",
    sample_rate=32000,
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)

# With PCEN spectrograms and mixup augmentation
predictor = AudioPredictor(
    label="event",
    spectrogram_type="pcen",     # PCEN normalization (better for birds)
    use_mixup=True,
    mixup_alpha=0.4,
    model_presets=["cnn14", "efficientnet"],
    presets="best_quality",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)
frame_probs = predictor.predict_frames(test_df)  # Sound event detection
```

### Multi-Modal Classification
```python
from endgame.automl import MultiModalPredictor

# Basic multi-modal (auto-detects text, image, tabular columns)
predictor = MultiModalPredictor(label="category")
predictor.fit(train_df)  # df has text, image paths, and numeric columns
predictions = predictor.predict(test_df)

# With explicit column specification and weighted fusion
predictor = MultiModalPredictor(
    label="sentiment",
    text_columns=["review", "title"],
    image_columns=["product_image"],
    tabular_columns=["price", "rating", "num_reviews"],
    fusion_strategy="weighted",  # late, weighted, stacking, attention
    modality_weights={"text": 0.5, "tabular": 0.3, "image": 0.2},
    presets="best_quality",
)
predictor.fit(train_df)
predictions = predictor.predict(test_df)

# Get modality contributions
contributions = predictor.get_modality_contributions()
leaderboard = predictor.leaderboard()
```

---

## Tests

Location: `tests/test_automl.py`

| Test Class | Tests | Status |
|------------|-------|--------|
| TestPresets | 6 | Passing |
| TestTimeBudgetManager | 4 | Passing |
| TestModelRegistry | 7 | Passing |
| TestPipelineConfig | 4 | Passing |
| TestSearchResult | 1 | Passing |
| TestPortfolioSearch | 3 | Passing |
| TestDataLoader | 6 | Passing |
| TestFitSummary | 1 | Passing |
| TestTabularPredictor | 6 | Passing |
| TestAutoMLPredictor | 4 | Passing |
| TestPipelineOrchestrator | 2 | Passing |
| TestTextPredictor | 20 | Passing |
| TestVisionPredictor | 8 | Passing |
| TestTimeSeriesPredictor | 9 | Passing |
| TestAudioPredictor | 8 | Passing |
| TestMultiModalPredictor | 15 | Passing |

**Total: 100 unit tests passing**

---

## Changelog

### 2026-01-23 (Sprint 4 - MultiModal)
- Implemented `MultiModalPredictor` for multi-modal datasets:
  - Auto-detects text, image, audio columns by content/extension
  - Supports explicit column specification
  - Fusion strategies: late (equal), weighted, stacking, attention
  - Auto-tuned weights based on modality validation scores
  - Time budget allocation across modalities
  - Leaderboard and contribution analysis methods
- Added 15 unit tests for MultiModalPredictor
- Sprint 4 fully completed (all domain predictors done)
- Total tests: 100 (all passing)
- Model registry expanded to 64 models

### 2026-01-23 (Sprint 4 - Domain Predictors)
- Implemented `VisionPredictor` for image classification:
  - Integrates with VisionBackbone from endgame.vision
  - Default backbones: EfficientNet-B0/B3, EVA02, ConvNeXt, Swin Transformer
  - Data augmentation via AugmentationPipeline
  - Test-time augmentation (TTA) support
  - Auto-detection of image columns by file extension
- Implemented `TimeSeriesPredictor` for time series classification:
  - Integrates with ROCKET-family classifiers (state-of-the-art)
  - Classifiers: MiniROCKET, ROCKET, MultiROCKET, HYDRA, HYDRA+MiniROCKET
  - Optional statistical feature extraction
  - Supports numpy arrays (3D/2D) and wide-format DataFrames
- Implemented `AudioPredictor` for audio/sound classification:
  - Integrates with SEDModel from endgame.audio
  - Encoders: CNN6, CNN10, CNN14, EfficientNet
  - Spectrogram conversion via SpectrogramTransformer/PCENTransformer
  - Audio augmentation support (noise, gain, timeshift, mixup)
  - Frame-level predictions for sound event detection
- Added 25 unit tests for domain predictors
- Total tests: 85 (all passing)

### 2026-01-23 (Sprint 3)
- Sprint 3 completed
- Implemented `TextPredictor` for NLP tasks:
  - Auto-detection of text columns in DataFrames
  - Integration with `TransformerClassifier` from endgame.nlp
  - Support for Domain-Adaptive Pretraining (DAPT) via `DomainAdaptivePretrainer`
  - Support for pseudo-labeling via `PseudoLabelTrainer`
  - Text preprocessing integration with `TextPreprocessor`
  - Weighted ensemble for multiple transformer models (`_TextEnsemble`)
- Default transformer models:
  - Fast: distilbert-base-uncased
  - Medium: microsoft/deberta-v3-base
  - Best: microsoft/deberta-v3-large
  - Multilingual: xlm-roberta-base
- Added 20 unit tests for TextPredictor
- Total tests: 60 (all passing)

### 2026-01-22 (Sprint 2)
- Sprint 2 completed
- Implemented all search strategies:
  - `HeuristicSearch`: Data-driven rules based on meta-features
  - `GeneticSearch`: Evolutionary pipeline optimization with crossover and mutation
  - `RandomSearch`: Random valid pipeline sampling
  - `BayesianSearch`: Optuna-based TPE optimization
- Implemented `AutoPreprocessor` for intelligent preprocessing selection
- Implemented `AutoMLStacker` and `HillClimbingStacker` for ensemble building

### 2026-01-22 (Sprint 1)
- Sprint 1 completed
- Created core infrastructure: presets, base predictor, time manager, model registry
- Implemented search module with portfolio search strategy
- Implemented TabularPredictor and AutoMLPredictor
- Created data loading utilities
- Added 32 unit tests

---

## Next Steps

1. ~~Implement `MultiModalPredictor` for combined data types~~ ✅ DONE
2. Sprint 5: Meta-learning bootstrap with OpenML benchmarks
3. Write comprehensive integration tests with actual model training
4. Add model persistence (save/load functionality)
5. Performance profiling and optimization
6. Add support for object detection (VisionPredictor)
7. Add forecasting mode for TimeSeriesPredictor
8. Knowledge distillation for ensemble compression
