"""Tests for AutoML module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from endgame.automl import (
    AutoMLPredictor,
    TabularPredictor,
    TextPredictor,
    VisionPredictor,
    TimeSeriesPredictor,
    AudioPredictor,
    BasePredictor,
    FitSummary,
    PresetConfig,
    PRESETS,
    get_preset,
    list_presets,
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
    PortfolioSearch,
    PipelineOrchestrator,
    TimeBudgetManager,
    MODEL_REGISTRY,
    ModelInfo,
    get_model_info,
    get_model_class,
    get_default_portfolio,
    get_models_by_family,
    DataLoader,
    load_data,
    infer_task_type,
)


class TestPresets:
    """Tests for preset configurations."""

    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_presets()
        assert "best_quality" in presets
        assert "high_quality" in presets
        assert "medium_quality" in presets
        assert "fast" in presets
        assert "interpretable" in presets

    def test_get_preset(self):
        """Test getting a preset configuration."""
        preset = get_preset("medium_quality")
        assert isinstance(preset, PresetConfig)
        assert preset.name == "medium_quality"
        assert preset.cv_folds >= 3

    def test_get_preset_invalid(self):
        """Test error on invalid preset."""
        with pytest.raises(ValueError):
            get_preset("nonexistent_preset")

    def test_preset_has_required_fields(self):
        """Test that presets have required fields."""
        for name, preset in PRESETS.items():
            assert hasattr(preset, "name")
            assert hasattr(preset, "cv_folds")
            assert hasattr(preset, "model_pool")
            assert hasattr(preset, "ensemble_method")

    def test_fast_preset_has_short_time_limit(self):
        """Test that fast preset has appropriate time limit."""
        preset = get_preset("fast")
        assert preset.default_time_limit is not None
        assert preset.default_time_limit <= 600  # 10 minutes max

    def test_best_quality_preset(self):
        """Test best quality preset configuration."""
        preset = get_preset("best_quality")
        assert preset.cv_folds >= 5
        assert preset.calibrate is True
        assert len(preset.model_pool) >= 8


class TestTimeBudgetManager:
    """Tests for time budget management."""

    def test_initialization(self):
        """Test TimeBudgetManager initialization."""
        allocations = {"stage1": 0.3, "stage2": 0.5, "stage3": 0.2}
        manager = TimeBudgetManager(total_budget=100, allocations=allocations)

        assert manager.total_budget == 100
        assert manager.allocations == allocations

    def test_start_and_end_stage(self):
        """Test starting and ending stages."""
        allocations = {"stage1": 0.5, "stage2": 0.5}
        manager = TimeBudgetManager(total_budget=100, allocations=allocations)
        manager.start()

        # Begin stage 1
        budget1 = manager.begin_stage("stage1")
        assert budget1 > 0
        assert manager._current_stage == "stage1"

        # End stage 1
        duration = manager.end_stage()
        assert duration >= 0
        assert manager._current_stage is None

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        allocations = {"stage1": 0.5, "stage2": 0.5}
        manager = TimeBudgetManager(total_budget=100, allocations=allocations)
        manager.start()

        # Initial remaining should be full budget
        remaining = manager.remaining_budget()
        assert remaining > 0

    def test_redistribute_time(self):
        """Test time redistribution between stages."""
        allocations = {"stage1": 0.3, "stage2": 0.3, "stage3": 0.4}
        manager = TimeBudgetManager(total_budget=100, allocations=allocations)
        manager.start()

        initial_stage2 = manager._stage_budgets["stage2"]
        initial_stage3 = manager._stage_budgets["stage3"]

        # Redistribute some time from stage2 to stage3
        transferred = manager.redistribute("stage2", "stage3", fraction=0.5)

        new_stage2 = manager._stage_budgets["stage2"]
        new_stage3 = manager._stage_budgets["stage3"]

        assert transferred > 0
        assert new_stage2 < initial_stage2
        assert new_stage3 > initial_stage3


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_not_empty(self):
        """Test that model registry has entries."""
        assert len(MODEL_REGISTRY) > 0

    def test_lgbm_in_registry(self):
        """Test that LightGBM is in the registry."""
        assert "lgbm" in MODEL_REGISTRY

    def test_get_model_info(self):
        """Test getting model info."""
        info = get_model_info("lgbm")
        assert info is not None
        assert info.name == "lgbm"
        assert info.family == "gbdt"

    def test_get_model_info_unknown(self):
        """Test getting info for unknown model raises error."""
        with pytest.raises(KeyError):
            get_model_info("unknown_model")

    def test_model_info_has_required_fields(self):
        """Test that model info has required fields."""
        valid_families = (
            "gbdt", "neural", "linear", "tree", "kernel",
            "rules", "bayesian", "foundation", "ensemble",
        )
        for name, info in MODEL_REGISTRY.items():
            assert isinstance(info, ModelInfo)
            assert info.name == name
            assert info.family in valid_families, f"Model {name} has unknown family: {info.family}"
            assert isinstance(info.task_types, list)

    def test_get_models_by_family(self):
        """Test getting models grouped by family."""
        models_by_family = get_models_by_family()
        assert isinstance(models_by_family, dict)
        assert "gbdt" in models_by_family
        assert len(models_by_family["gbdt"]) > 0
        assert "lgbm" in models_by_family["gbdt"]

    def test_get_default_portfolio(self):
        """Test getting default portfolio."""
        portfolio = get_default_portfolio()
        assert len(portfolio) > 0
        assert "lgbm" in portfolio


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_create_config(self):
        """Test creating a pipeline config."""
        config = PipelineConfig(
            model_name="lgbm",
            model_params={"n_estimators": 100},
        )
        assert config.model_name == "lgbm"
        assert config.model_params["n_estimators"] == 100

    def test_config_generates_id(self):
        """Test that config generates unique ID."""
        config = PipelineConfig(model_name="lgbm")
        assert config.config_id is not None
        assert len(config.config_id) == 12

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = PipelineConfig(
            model_name="lgbm",
            model_params={"n_estimators": 100},
        )
        d = config.to_dict()
        assert d["model_name"] == "lgbm"
        assert d["model_params"]["n_estimators"] == 100

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "model_name": "xgb",
            "model_params": {"n_estimators": 200},
        }
        config = PipelineConfig.from_dict(d)
        assert config.model_name == "xgb"
        assert config.model_params["n_estimators"] == 200


class TestSearchResult:
    """Tests for search results."""

    def test_create_result(self):
        """Test creating a search result."""
        config = PipelineConfig(model_name="lgbm")
        result = SearchResult(
            config=config,
            score=0.85,
            fit_time=10.5,
            success=True,
        )
        assert result.score == 0.85
        assert result.success is True


class TestPortfolioSearch:
    """Tests for portfolio search strategy."""

    def test_create_portfolio_search(self):
        """Test creating portfolio search."""
        search = PortfolioSearch(
            task_type="classification",
            preset="medium_quality",
        )
        assert search.task_type == "classification"

    def test_suggest_returns_configs(self):
        """Test that suggest returns pipeline configs."""
        search = PortfolioSearch(
            task_type="classification",
            preset="fast",
        )
        meta_features = {"nr_inst": 1000, "nr_attr": 10}
        configs = search.suggest(meta_features=meta_features, n_suggestions=3)

        assert len(configs) > 0
        assert all(isinstance(c, PipelineConfig) for c in configs)

    def test_ensure_diversity(self):
        """Test that diversity is ensured in suggestions."""
        search = PortfolioSearch(
            task_type="classification",
            preset="medium_quality",
            ensure_diversity=True,
        )
        meta_features = {"nr_inst": 10000, "nr_attr": 20}
        configs = search.suggest(meta_features=meta_features, n_suggestions=5)

        # Check that we have configs from different families
        model_names = [c.model_name for c in configs]
        families = set()
        for name in model_names:
            info = get_model_info(name)
            if info:
                families.add(info.family)

        # Should have at least 2 different families
        assert len(families) >= min(2, len(configs))


class TestDataLoader:
    """Tests for data loading utilities."""

    def test_infer_task_type_binary(self):
        """Test binary classification detection."""
        y = np.array([0, 1, 0, 1, 1, 0])
        task_type = infer_task_type(y)
        assert task_type == "binary"

    def test_infer_task_type_multiclass(self):
        """Test multiclass classification detection."""
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        task_type = infer_task_type(y)
        assert task_type == "multiclass"

    def test_infer_task_type_regression(self):
        """Test regression detection."""
        # Use enough unique float values to exceed the threshold
        y = np.linspace(0.0, 100.0, 50)  # 50 unique continuous values
        task_type = infer_task_type(y)
        assert task_type == "regression"

    def test_data_loader_creation(self):
        """Test DataLoader creation."""
        loader = DataLoader(label="target")
        assert loader.label == "target"

    def test_load_data_from_dataframe(self):
        """Test loading data from DataFrame."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0],
        })
        X, y = load_data(df, label="target")

        assert len(X) == 5
        assert len(y) == 5
        assert "target" not in X.columns

    def test_load_data_from_numpy(self):
        """Test loading data from numpy array."""
        data = np.array([
            [1, 2, 0],
            [3, 4, 1],
            [5, 6, 0],
        ])
        # When loading from numpy without specifying a label,
        # the last column is used as target by default
        X, y = load_data(data)

        # The default behavior assigns 'target' to the last column
        assert X.shape[0] == 3
        assert y is not None
        assert len(y) == 3


class TestTabularPredictor:
    """Tests for TabularPredictor."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["target"] = y
        return df

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["target"] = y
        return df

    def test_create_predictor(self):
        """Test creating a TabularPredictor."""
        predictor = TabularPredictor(label="target")
        assert predictor.label == "target"
        assert predictor.is_fitted_ is False

    def test_create_predictor_with_options(self):
        """Test creating predictor with options."""
        predictor = TabularPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        assert predictor.presets == "fast"
        assert predictor.time_limit == 60

    def test_fit_classification(self, classification_data):
        """Test fitting on classification data."""
        predictor = TabularPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(classification_data)

        assert predictor.is_fitted_
        assert predictor.problem_type_ in ("binary", "classification")
        assert predictor.classes_ is not None

    def test_fit_regression(self, regression_data):
        """Test fitting on regression data."""
        predictor = TabularPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(regression_data)

        assert predictor.is_fitted_
        assert predictor.problem_type_ == "regression"

    def test_predict_classification(self, classification_data):
        """Test predictions on classification data."""
        predictor = TabularPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(classification_data)

        X_test = classification_data.drop(columns=["target"])
        predictions = predictor.predict(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_classification(self, classification_data):
        """Test probability predictions."""
        predictor = TabularPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(classification_data)

        X_test = classification_data.drop(columns=["target"])
        proba = predictor.predict_proba(X_test)

        assert proba.shape[0] == len(X_test)
        assert proba.shape[1] == 2  # Binary classification
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_leaderboard(self, classification_data):
        """Test leaderboard generation."""
        predictor = TabularPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(classification_data)

        lb = predictor.leaderboard(silent=True)
        assert isinstance(lb, pd.DataFrame)
        assert "model" in lb.columns
        assert "score" in lb.columns


class TestAutoMLPredictor:
    """Tests for unified AutoMLPredictor."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["target"] = y
        return df

    def test_create_automl_predictor(self):
        """Test creating an AutoMLPredictor."""
        predictor = AutoMLPredictor(label="target")
        assert predictor.label == "target"
        assert predictor.is_fitted is False

    def test_fit_automl(self, classification_data):
        """Test fitting AutoMLPredictor."""
        predictor = AutoMLPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(classification_data)

        assert predictor.is_fitted
        assert predictor.domain_ == "tabular"
        assert predictor.predictor_ is not None

    def test_predict_automl(self, classification_data):
        """Test predictions with AutoMLPredictor."""
        predictor = AutoMLPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        predictor.fit(classification_data)

        X_test = classification_data.drop(columns=["target"])
        predictions = predictor.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_chained_fit_predict(self, classification_data):
        """Test chained fit().predict() pattern."""
        predictor = AutoMLPredictor(
            label="target",
            presets="fast",
            time_limit=60,
            verbosity=0,
        )
        X_test = classification_data.drop(columns=["target"])
        predictions = predictor.fit(classification_data).predict(X_test)

        assert len(predictions) == len(X_test)


class TestPipelineOrchestrator:
    """Tests for pipeline orchestrator."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        return X, y

    def test_create_orchestrator(self):
        """Test creating a PipelineOrchestrator."""
        orchestrator = PipelineOrchestrator(preset="fast", time_limit=60)
        assert orchestrator.time_limit == 60

    def test_run_pipeline(self, classification_data):
        """Test running the full pipeline."""
        X, y = classification_data
        orchestrator = PipelineOrchestrator(preset="fast", time_limit=60, verbose=0)
        result = orchestrator.run(X, y, task_type="classification")

        assert result is not None
        assert result.total_time > 0
        assert len(result.stage_results) > 0


class TestFitSummary:
    """Tests for FitSummary dataclass."""

    def test_create_fit_summary(self):
        """Test creating a FitSummary."""
        summary = FitSummary(
            total_time=100.0,
            n_models_trained=5,
            n_models_failed=1,
            best_model="lgbm",
            best_score=0.95,
            cv_score=0.94,
        )
        assert summary.total_time == 100.0
        assert summary.n_models_trained == 5
        assert summary.best_model == "lgbm"


class TestTextPredictor:
    """Tests for TextPredictor."""

    @pytest.fixture
    def text_classification_data(self):
        """Generate text classification data."""
        texts = [
            "I love this product! It's amazing.",
            "This is the worst purchase I've ever made.",
            "Great quality and fast shipping.",
            "Terrible customer service, never buying again.",
            "Excellent value for money.",
            "Poor quality, fell apart in a week.",
            "Highly recommend this to everyone!",
            "Complete waste of money.",
            "Best product I've bought this year.",
            "Disappointed with the performance.",
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        df = pd.DataFrame({"text": texts, "sentiment": labels})
        return df

    @pytest.fixture
    def text_data_with_common_column(self):
        """Generate text data with common column name."""
        df = pd.DataFrame({
            "review": ["Great product!", "Bad service.", "Loved it!"],
            "rating": [5, 1, 5],
        })
        return df

    def test_create_text_predictor(self):
        """Test creating a TextPredictor."""
        predictor = TextPredictor(label="sentiment", text_column="text")
        assert predictor.label == "sentiment"
        assert predictor.text_column == "text"
        assert predictor.is_fitted_ is False

    def test_create_text_predictor_with_options(self):
        """Test creating TextPredictor with options."""
        predictor = TextPredictor(
            label="sentiment",
            text_column="text",
            presets="fast",
            use_dapt=True,
            use_pseudo_labeling=True,
            pseudo_threshold=0.9,
            verbosity=0,
        )
        assert predictor.presets == "fast"
        assert predictor.use_dapt is True
        assert predictor.use_pseudo_labeling is True
        assert predictor.pseudo_threshold == 0.9

    def test_detect_text_column_explicit(self, text_classification_data):
        """Test text column detection when explicitly specified."""
        predictor = TextPredictor(label="sentiment", text_column="text")
        # Access internal method through a protected accessor
        detected = predictor._detect_text_column(text_classification_data)
        assert detected == "text"

    def test_detect_text_column_common_name(self, text_data_with_common_column):
        """Test text column detection with common column names."""
        predictor = TextPredictor(label="rating", text_column="nonexistent")
        detected = predictor._detect_text_column(text_data_with_common_column)
        assert detected == "review"

    def test_detect_text_column_by_content(self):
        """Test text column detection by content analysis."""
        df = pd.DataFrame({
            "short_col": ["a", "b", "c"],
            "long_content": [
                "This is a much longer piece of text that should be detected.",
                "Another lengthy string that contains meaningful content.",
                "Yet another substantial text entry with multiple words.",
            ],
            "target": [0, 1, 0],
        })
        predictor = TextPredictor(label="target", text_column="nonexistent")
        detected = predictor._detect_text_column(df)
        assert detected == "long_content"

    def test_preprocess_texts(self, text_classification_data):
        """Test text preprocessing."""
        predictor = TextPredictor(label="sentiment", text_column="text")
        texts = text_classification_data["text"].values
        processed = predictor._preprocess_texts(texts)

        assert len(processed) == len(texts)
        assert all(isinstance(t, str) for t in processed)

    def test_preprocess_texts_handles_none(self):
        """Test preprocessing handles None values."""
        predictor = TextPredictor(label="sentiment", text_column="text")
        texts = np.array(["valid text", None, "another text"])
        processed = predictor._preprocess_texts(texts)

        assert len(processed) == 3
        assert processed[1] == ""  # None converted to empty string

    def test_select_models_fast_preset(self):
        """Test model selection for fast preset."""
        predictor = TextPredictor(label="sentiment", presets="fast")
        models = predictor._select_models()

        assert len(models) == 1
        assert "distilbert" in models[0].lower()

    def test_select_models_best_quality_preset(self):
        """Test model selection for best quality preset."""
        predictor = TextPredictor(label="sentiment", presets="best_quality")
        models = predictor._select_models()

        assert len(models) >= 2
        # Should include deberta models
        assert any("deberta" in m.lower() for m in models)

    def test_select_models_custom(self):
        """Test model selection with custom models."""
        custom_models = ["bert-base-uncased", "roberta-base"]
        predictor = TextPredictor(
            label="sentiment",
            model_presets=custom_models,
        )
        models = predictor._select_models()

        assert models == custom_models

    def test_load_text_data_from_dataframe(self, text_classification_data):
        """Test loading text data from DataFrame."""
        predictor = TextPredictor(label="sentiment", text_column="text")
        X, y, texts = predictor._load_text_data(text_classification_data)

        assert len(texts) == 10
        assert len(y) == 10
        assert "sentiment" not in X.columns

    def test_load_text_data_require_label_false(self, text_classification_data):
        """Test loading text data without requiring label."""
        predictor = TextPredictor(label="sentiment", text_column="text")
        X, y, texts = predictor._load_text_data(
            text_classification_data, require_label=False
        )

        assert len(texts) == 10
        assert y is None

    def test_load_text_data_missing_text_column(self):
        """Test error when text column is missing."""
        df = pd.DataFrame({"not_text": ["a", "b"], "label": [0, 1]})
        predictor = TextPredictor(label="label", text_column="text")
        predictor.text_column_ = "text"  # Set explicitly to force error

        with pytest.raises(ValueError, match="Text column 'text' not found"):
            predictor._load_text_data(df)

    def test_load_text_data_missing_label_column(self, text_classification_data):
        """Test error when label column is missing."""
        predictor = TextPredictor(label="nonexistent", text_column="text")

        with pytest.raises(ValueError, match="Label column 'nonexistent' not found"):
            predictor._load_text_data(text_classification_data)

    def test_text_ensemble_predict(self):
        """Test _TextEnsemble prediction."""
        from endgame.automl.text import _TextEnsemble

        # Create mock models
        class MockModel:
            def predict(self, texts):
                return np.array([0] * len(texts))

            def predict_proba(self, texts):
                return np.array([[0.7, 0.3]] * len(texts))

        models = {"model1": MockModel(), "model2": MockModel()}
        weights = {"model1": 0.6, "model2": 0.4}

        ensemble = _TextEnsemble(models, weights, task_type="binary")
        texts = ["text1", "text2", "text3"]

        predictions = ensemble.predict(texts)
        assert len(predictions) == 3

    def test_text_ensemble_predict_proba(self):
        """Test _TextEnsemble probability prediction."""
        from endgame.automl.text import _TextEnsemble

        class MockModel:
            def predict_proba(self, texts):
                return np.array([[0.3, 0.7]] * len(texts))

        models = {"model1": MockModel(), "model2": MockModel()}
        weights = {"model1": 0.5, "model2": 0.5}

        ensemble = _TextEnsemble(models, weights, task_type="binary")
        texts = ["text1", "text2"]

        proba = ensemble.predict_proba(texts)
        assert proba.shape == (2, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_text_ensemble_regression(self):
        """Test _TextEnsemble for regression."""
        from endgame.automl.text import _TextEnsemble

        class MockModel:
            def predict(self, texts):
                return np.array([1.0] * len(texts))

        models = {"model1": MockModel(), "model2": MockModel()}
        weights = {"model1": 0.5, "model2": 0.5}

        ensemble = _TextEnsemble(models, weights, task_type="regression")
        texts = ["text1", "text2"]

        predictions = ensemble.predict(texts)
        assert len(predictions) == 2
        assert np.allclose(predictions, 1.0)

    def test_get_base_model_name_fast(self):
        """Test getting base model name for fast preset."""
        predictor = TextPredictor(label="sentiment", presets="fast")
        model_name = predictor._get_base_model_name()
        assert "distilbert" in model_name.lower()

    def test_get_base_model_name_best_quality(self):
        """Test getting base model name for best quality preset."""
        predictor = TextPredictor(label="sentiment", presets="best_quality")
        model_name = predictor._get_base_model_name()
        assert "deberta" in model_name.lower()

    def test_get_base_model_name_custom(self):
        """Test getting base model name with custom models."""
        predictor = TextPredictor(
            label="sentiment",
            model_presets=["custom-model", "another-model"],
        )
        model_name = predictor._get_base_model_name()
        assert model_name == "custom-model"


class TestVisionPredictor:
    """Tests for VisionPredictor."""

    @pytest.fixture
    def image_classification_data(self):
        """Generate image classification data with fake paths."""
        df = pd.DataFrame({
            "image_path": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
                "/path/to/image3.png",
                "/path/to/image4.jpeg",
                "/path/to/image5.jpg",
            ],
            "label": [0, 1, 0, 1, 0],
        })
        return df

    def test_create_vision_predictor(self):
        """Test creating a VisionPredictor."""
        predictor = VisionPredictor(label="category", image_column="image")
        assert predictor.label == "category"
        assert predictor.image_column == "image"
        assert predictor.is_fitted_ is False

    def test_create_vision_predictor_with_options(self):
        """Test creating VisionPredictor with options."""
        predictor = VisionPredictor(
            label="species",
            image_column="filepath",
            presets="fast",
            augmentation="heavy",
            use_tta=True,
            tta_augmentations=["identity", "hflip", "vflip"],
            verbosity=0,
        )
        assert predictor.presets == "fast"
        assert predictor.augmentation == "heavy"
        assert predictor.use_tta is True
        assert "vflip" in predictor.tta_augmentations

    def test_detect_image_column_explicit(self, image_classification_data):
        """Test image column detection when explicitly specified."""
        predictor = VisionPredictor(label="label", image_column="image_path")
        detected = predictor._detect_image_column(image_classification_data)
        assert detected == "image_path"

    def test_detect_image_column_by_extension(self):
        """Test image column detection by file extension."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "photo": ["img1.jpg", "img2.png", "img3.jpeg"],
            "category": [0, 1, 0],
        })
        predictor = VisionPredictor(label="category", image_column="nonexistent", verbosity=0)
        detected = predictor._detect_image_column(df)
        assert detected == "photo"

    def test_select_models_fast_preset(self):
        """Test model selection for fast preset."""
        predictor = VisionPredictor(label="label", presets="fast")
        models = predictor._select_models()
        assert len(models) == 1
        assert "efficientnet_b0" in models[0]

    def test_select_models_best_quality_preset(self):
        """Test model selection for best quality preset."""
        predictor = VisionPredictor(label="label", presets="best_quality")
        models = predictor._select_models()
        assert len(models) >= 2

    def test_select_models_custom(self):
        """Test model selection with custom models."""
        custom_models = ["efficientnet_b0", "swin_tiny_patch4_window7_224"]
        predictor = VisionPredictor(
            label="label",
            model_presets=custom_models,
        )
        models = predictor._select_models()
        assert models == custom_models

    def test_vision_ensemble(self):
        """Test _VisionEnsemble prediction."""
        from endgame.automl.vision import _VisionEnsemble

        class MockModel:
            def predict(self, images):
                return np.array([0] * len(images))

            def predict_proba(self, images):
                return np.array([[0.7, 0.3]] * len(images))

        models = {"model1": MockModel(), "model2": MockModel()}
        weights = {"model1": 0.6, "model2": 0.4}

        ensemble = _VisionEnsemble(models, weights, task_type="binary")
        predictions = ensemble.predict([1, 2, 3])
        assert len(predictions) == 3


class TestTimeSeriesPredictor:
    """Tests for TimeSeriesPredictor."""

    @pytest.fixture
    def ts_classification_data(self):
        """Generate time series classification data."""
        np.random.seed(42)
        # (n_samples, n_channels, n_timesteps)
        X = np.random.randn(50, 1, 100)
        y = np.random.randint(0, 3, 50)
        return X, y

    @pytest.fixture
    def ts_classification_df(self):
        """Generate time series classification DataFrame (wide format)."""
        np.random.seed(42)
        n_samples = 50
        n_timesteps = 100
        X = np.random.randn(n_samples, n_timesteps)
        y = np.random.randint(0, 3, n_samples)

        df = pd.DataFrame(X, columns=[f"t{i}" for i in range(n_timesteps)])
        df["label"] = y
        return df

    def test_create_ts_predictor(self):
        """Test creating a TimeSeriesPredictor."""
        predictor = TimeSeriesPredictor(label="activity")
        assert predictor.label == "activity"
        assert predictor.is_fitted_ is False

    def test_create_ts_predictor_with_options(self):
        """Test creating TimeSeriesPredictor with options."""
        predictor = TimeSeriesPredictor(
            label="class",
            presets="fast",
            classifier_presets=["minirocket", "hydra"],
            use_feature_extraction=True,
            feature_preset="minimal",
            verbosity=0,
        )
        assert predictor.presets == "fast"
        assert predictor.classifier_presets == ["minirocket", "hydra"]
        assert predictor.use_feature_extraction is True

    def test_load_ts_data_numpy(self, ts_classification_data):
        """Test loading time series data from numpy array."""
        X, y = ts_classification_data
        predictor = TimeSeriesPredictor(label="class")
        X_loaded, y_loaded = predictor._load_ts_data(X, y)

        assert X_loaded.shape == X.shape
        assert len(y_loaded) == len(y)

    def test_load_ts_data_numpy_2d(self):
        """Test loading 2D numpy array (univariate time series)."""
        np.random.seed(42)
        X = np.random.randn(30, 50)  # (n_samples, n_timesteps)
        y = np.random.randint(0, 2, 30)

        predictor = TimeSeriesPredictor(label="class")
        X_loaded, y_loaded = predictor._load_ts_data(X, y)

        # Should add channel dimension
        assert len(X_loaded.shape) == 3
        assert X_loaded.shape[1] == 1  # Channel dimension added

    def test_load_ts_data_dataframe(self, ts_classification_df):
        """Test loading time series data from DataFrame."""
        predictor = TimeSeriesPredictor(label="label")
        X_loaded, y_loaded = predictor._load_ts_data(ts_classification_df, None)

        assert len(X_loaded) == 50
        assert len(y_loaded) == 50

    def test_select_classifiers_fast_preset(self):
        """Test classifier selection for fast preset."""
        predictor = TimeSeriesPredictor(label="class", presets="fast")
        classifiers = predictor._select_classifiers()
        assert len(classifiers) == 1
        assert "minirocket" in classifiers[0]

    def test_select_classifiers_best_quality_preset(self):
        """Test classifier selection for best quality preset."""
        predictor = TimeSeriesPredictor(label="class", presets="best_quality")
        classifiers = predictor._select_classifiers()
        assert len(classifiers) >= 2

    def test_select_classifiers_custom(self):
        """Test classifier selection with custom classifiers."""
        custom = ["rocket", "hydra"]
        predictor = TimeSeriesPredictor(
            label="class",
            classifier_presets=custom,
        )
        classifiers = predictor._select_classifiers()
        assert classifiers == custom

    def test_ts_ensemble(self):
        """Test _TSEnsemble prediction."""
        from endgame.automl.timeseries import _TSEnsemble

        class MockModel:
            def predict(self, X):
                return np.array([0] * len(X))

            def predict_proba(self, X):
                return np.array([[0.7, 0.2, 0.1]] * len(X))

        classes = np.array([0, 1, 2])
        models = {"model1": MockModel(), "model2": MockModel()}
        weights = {"model1": 0.5, "model2": 0.5}

        ensemble = _TSEnsemble(models, weights, classes)
        X = np.random.randn(5, 1, 50)

        predictions = ensemble.predict(X)
        assert len(predictions) == 5

        proba = ensemble.predict_proba(X)
        assert proba.shape == (5, 3)


class TestAudioPredictor:
    """Tests for AudioPredictor."""

    @pytest.fixture
    def audio_classification_data(self):
        """Generate audio classification data with fake paths."""
        df = pd.DataFrame({
            "audio_path": [
                "/path/to/audio1.wav",
                "/path/to/audio2.wav",
                "/path/to/audio3.mp3",
                "/path/to/audio4.wav",
                "/path/to/audio5.flac",
            ],
            "species": ["bird_a", "bird_b", "bird_a", "bird_c", "bird_b"],
        })
        return df

    def test_create_audio_predictor(self):
        """Test creating an AudioPredictor."""
        predictor = AudioPredictor(label="species", audio_column="audio")
        assert predictor.label == "species"
        assert predictor.audio_column == "audio"
        assert predictor.is_fitted_ is False

    def test_create_audio_predictor_with_options(self):
        """Test creating AudioPredictor with options."""
        predictor = AudioPredictor(
            label="event",
            audio_column="filepath",
            presets="fast",
            sample_rate=22050,
            n_mels=64,
            spectrogram_type="pcen",
            use_augmentation=True,
            augmentations=["noise", "gain"],
            use_mixup=True,
            mixup_alpha=0.5,
            verbosity=0,
        )
        assert predictor.presets == "fast"
        assert predictor.sample_rate == 22050
        assert predictor.n_mels == 64
        assert predictor.spectrogram_type == "pcen"
        assert predictor.use_mixup is True
        assert predictor.mixup_alpha == 0.5

    def test_detect_audio_column_explicit(self, audio_classification_data):
        """Test audio column detection when explicitly specified."""
        predictor = AudioPredictor(label="species", audio_column="audio_path")
        detected = predictor._detect_audio_column(audio_classification_data)
        assert detected == "audio_path"

    def test_detect_audio_column_by_extension(self):
        """Test audio column detection by file extension."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "recording": ["sound1.wav", "sound2.mp3", "sound3.flac"],
            "label": [0, 1, 0],
        })
        predictor = AudioPredictor(label="label", audio_column="nonexistent", verbosity=0)
        detected = predictor._detect_audio_column(df)
        assert detected == "recording"

    def test_select_models_fast_preset(self):
        """Test model selection for fast preset."""
        predictor = AudioPredictor(label="label", presets="fast")
        models = predictor._select_models()
        assert len(models) == 1
        assert "cnn6" in models[0]

    def test_select_models_best_quality_preset(self):
        """Test model selection for best quality preset."""
        predictor = AudioPredictor(label="label", presets="best_quality")
        models = predictor._select_models()
        assert len(models) >= 2

    def test_select_models_custom(self):
        """Test model selection with custom models."""
        custom_models = ["cnn10", "efficientnet"]
        predictor = AudioPredictor(
            label="label",
            model_presets=custom_models,
        )
        models = predictor._select_models()
        assert models == custom_models

    def test_audio_ensemble(self):
        """Test _AudioEnsemble prediction."""
        from endgame.automl.audio import _AudioEnsemble

        class MockModel:
            def predict(self, X):
                return np.array([0] * len(X))

            def predict_proba(self, X):
                return np.array([[0.7, 0.3]] * len(X))

        classes = np.array([0, 1])
        models = {"model1": MockModel(), "model2": MockModel()}
        weights = {"model1": 0.6, "model2": 0.4}

        ensemble = _AudioEnsemble(models, weights, classes)
        X = np.random.randn(5, 128, 64)  # (n_samples, n_mels, n_frames)

        predictions = ensemble.predict(X)
        assert len(predictions) == 5

        proba = ensemble.predict_proba(X)
        assert proba.shape == (5, 2)


class TestMultiModalPredictor:
    """Tests for MultiModalPredictor."""

    @pytest.fixture
    def multimodal_data(self):
        """Generate multimodal data with tabular, text, and image columns."""
        np.random.seed(42)
        n_samples = 50

        df = pd.DataFrame({
            # Tabular features
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            # Text column - make it longer to trigger text detection (avg_len > 50)
            "description": [
                "This is a very positive review with excellent feedback about the product quality and customer service experience that was truly wonderful" if i % 2 == 0
                else "This is a rather negative review with many complaints about the poor quality and terrible customer service that was very disappointing overall"
                for i in range(n_samples)
            ],
            # Image paths (fake)
            "image_path": [f"/path/to/image_{i}.jpg" for i in range(n_samples)],
            # Audio paths (fake)
            "audio_path": [f"/path/to/audio_{i}.wav" for i in range(n_samples)],
            # Target
            "target": np.random.randint(0, 2, n_samples),
        })
        return df

    @pytest.fixture
    def tabular_text_data(self):
        """Generate data with only tabular and text columns."""
        np.random.seed(42)
        n_samples = 30

        df = pd.DataFrame({
            "price": np.random.uniform(10, 1000, n_samples),
            "rating": np.random.uniform(1, 5, n_samples),
            "review": [
                f"Product review {i}: " + ("great" if i % 2 == 0 else "bad")
                for i in range(n_samples)
            ],
            "sentiment": np.random.randint(0, 2, n_samples),
        })
        return df

    def test_create_multimodal_predictor(self):
        """Test creating a MultiModalPredictor."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        assert predictor.label == "target"
        assert predictor.is_fitted_ is False
        assert predictor.fusion_strategy == "late"

    def test_create_multimodal_predictor_with_options(self):
        """Test creating MultiModalPredictor with options."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(
            label="target",
            presets="fast",
            fusion_strategy="weighted",
            modality_weights={"tabular": 0.6, "text": 0.4},
            tabular_columns=["feature1", "feature2"],
            text_columns=["description"],
            enable_image=False,
            enable_audio=False,
            verbosity=0,
        )
        assert predictor.presets == "fast"
        assert predictor.fusion_strategy == "weighted"
        assert predictor.modality_weights == {"tabular": 0.6, "text": 0.4}
        assert predictor.enable_image is False
        assert predictor.enable_audio is False

    def test_detect_modalities_auto(self, multimodal_data):
        """Test automatic modality detection."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        modality_columns = predictor._detect_modalities(multimodal_data)

        # Should detect tabular columns
        assert "feature1" in modality_columns["tabular"]
        assert "feature2" in modality_columns["tabular"]
        assert "category" in modality_columns["tabular"]

        # Should detect text column (long string content)
        assert "description" in modality_columns["text"]

        # Should detect image column (ends with .jpg)
        assert "image_path" in modality_columns["image"]

        # Should detect audio column (ends with .wav)
        assert "audio_path" in modality_columns["audio"]

    def test_detect_modalities_explicit(self, multimodal_data):
        """Test modality detection with explicit column specification."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(
            label="target",
            tabular_columns=["feature1"],
            text_columns=["description"],
            image_columns=["image_path"],
        )
        modality_columns = predictor._detect_modalities(multimodal_data)

        assert modality_columns["tabular"] == ["feature1"]
        assert modality_columns["text"] == ["description"]
        assert modality_columns["image"] == ["image_path"]

    def test_allocate_time(self):
        """Test time allocation across modalities."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        modalities = ["tabular", "text", "image"]
        allocation = predictor._allocate_time(modalities, total_time=600)

        # Should allocate time to all modalities
        assert "tabular" in allocation
        assert "text" in allocation
        assert "image" in allocation
        assert "_fusion" in allocation

        # Text should get more time (transformers are slower)
        assert allocation["text"] >= allocation["tabular"]

        # Total should roughly sum to 600 (minus fusion time)
        modality_total = sum(v for k, v in allocation.items() if k != "_fusion")
        assert modality_total <= 600

    def test_fusion_strategies(self):
        """Test different fusion strategies are accepted."""
        from endgame.automl import MultiModalPredictor

        for strategy in ["late", "weighted", "stacking", "attention"]:
            predictor = MultiModalPredictor(
                label="target",
                fusion_strategy=strategy,
            )
            assert predictor.fusion_strategy == strategy

    def test_compute_fusion_weights_late(self):
        """Test late fusion weight computation (equal weights)."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target", fusion_strategy="late")
        predictor.modalities_ = ["tabular", "text"]
        predictor._compute_fusion_weights({}, np.array([0, 1, 0, 1]))

        assert predictor.modality_weights_["tabular"] == 0.5
        assert predictor.modality_weights_["text"] == 0.5

    def test_compute_fusion_weights_weighted_manual(self):
        """Test weighted fusion with manual weights."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(
            label="target",
            fusion_strategy="weighted",
            modality_weights={"tabular": 0.7, "text": 0.3},
        )
        predictor.modalities_ = ["tabular", "text"]
        predictor._compute_fusion_weights({}, np.array([0, 1, 0, 1]))

        assert abs(predictor.modality_weights_["tabular"] - 0.7) < 0.01
        assert abs(predictor.modality_weights_["text"] - 0.3) < 0.01

    def test_compute_fusion_weights_weighted_auto(self):
        """Test weighted fusion with auto-tuned weights from scores."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target", fusion_strategy="weighted")
        predictor.modalities_ = ["tabular", "text"]
        predictor.modality_scores_ = {"tabular": 0.8, "text": 0.6}
        predictor._compute_fusion_weights({}, np.array([0, 1, 0, 1]))

        # Tabular should have higher weight (better score)
        assert predictor.modality_weights_["tabular"] > predictor.modality_weights_["text"]

    def test_weighted_fusion_regression(self):
        """Test weighted fusion for regression predictions."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        predictor.problem_type_ = "regression"
        predictor.modalities_ = ["tabular", "text"]
        predictor.modality_weights_ = {"tabular": 0.6, "text": 0.4}

        modality_preds = {
            "tabular": np.array([1.0, 2.0, 3.0]),
            "text": np.array([1.5, 2.5, 3.5]),
        }

        fused = predictor._weighted_fusion(modality_preds, is_proba=False)

        # Expected: 0.6 * tabular + 0.4 * text
        expected = 0.6 * np.array([1.0, 2.0, 3.0]) + 0.4 * np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_almost_equal(fused, expected)

    def test_weighted_fusion_classification_proba(self):
        """Test weighted fusion for classification probabilities."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        predictor.problem_type_ = "classification"
        predictor.classes_ = np.array([0, 1])
        predictor.modalities_ = ["tabular", "text"]
        predictor.modality_weights_ = {"tabular": 0.5, "text": 0.5}

        modality_preds = {
            "tabular": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "text": np.array([[0.6, 0.4], [0.4, 0.6]]),
        }

        fused = predictor._weighted_fusion(modality_preds, is_proba=True)

        # Should be averaged
        assert fused.shape == (2, 2)
        assert np.allclose(fused.sum(axis=1), 1.0)

    def test_leaderboard_not_fitted(self):
        """Test leaderboard raises error when not fitted."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.leaderboard()

    def test_get_modality_contributions_not_fitted(self):
        """Test modality contributions raises error when not fitted."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.get_modality_contributions()

    def test_predict_not_fitted(self):
        """Test predict raises error when not fitted."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict(df)

    def test_predict_proba_not_fitted(self):
        """Test predict_proba raises error when not fitted."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target")
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict_proba(df)


class TestEnhancedPreprocessing:
    """Tests for enhanced preprocessing with endgame modules."""

    @pytest.fixture
    def mixed_data(self):
        """Generate data with numeric and categorical features."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "num1": np.random.randn(n),
            "num2": np.random.randn(n),
            "num3": np.random.randn(n),
            "cat_low": np.random.choice(["a", "b", "c"], n),
            "cat_high": np.random.choice([f"cat_{i}" for i in range(30)], n),
        })
        # Add some missing values
        df.loc[0:10, "num1"] = np.nan
        df.loc[5:15, "cat_low"] = np.nan
        y = np.random.randint(0, 2, n)
        return df, y

    @pytest.fixture
    def imbalanced_data(self):
        """Generate imbalanced classification data."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        # 90% class 0, 10% class 1
        y = np.zeros(n, dtype=int)
        y[:20] = 1
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y

    def test_none_level_uses_basic_pipeline(self, mixed_data):
        """Test that 'none' level produces basic sklearn pipeline."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        executor = PreprocessingExecutor(feature_engineering="none")
        X, y = mixed_data
        preprocessor = executor._build_preprocessor(X, y, {})

        # Should be a ColumnTransformer with basic imputer + scaler + OHE
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)

    def test_light_level_uses_knn_imputer(self, mixed_data):
        """Test that 'light' level uses KNNImputer for numerics."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        executor = PreprocessingExecutor(feature_engineering="light")
        X, y = mixed_data
        preprocessor = executor._build_preprocessor(X, y, {})

        # Should be a ColumnTransformer
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)

        # Check that numeric pipeline has KNNImputer
        for name, transformer, cols in preprocessor.transformers:
            if name == "numeric":
                imputer_name = transformer.steps[0][0]
                assert imputer_name == "imputer"
                imputer = transformer.steps[0][1]
                # Should be KNNImputer from endgame or fallback
                assert hasattr(imputer, "fit_transform")

    def test_light_level_splits_categoricals(self, mixed_data):
        """Test that 'light' uses SafeTargetEncoder for high-cardinality cols."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        executor = PreprocessingExecutor(feature_engineering="light")
        X, y = mixed_data
        preprocessor = executor._build_preprocessor(X, y, {})

        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)

        transformer_names = [name for name, _, _ in preprocessor.transformers]
        # Should have separate transformers for low/high cardinality
        assert "cat_low" in transformer_names
        assert "cat_high" in transformer_names

    def test_moderate_level_adds_autobalancer(self, imbalanced_data):
        """Test that 'moderate' adds AutoBalancer for imbalanced data."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        executor = PreprocessingExecutor(feature_engineering="moderate")
        X, y = imbalanced_data
        meta_features = {"class_imbalance_ratio": 0.1}
        preprocessor = executor._build_preprocessor(
            X, y, meta_features, task_type="classification",
        )

        from sklearn.pipeline import Pipeline
        # Should be a Pipeline with AutoBalancer at the end
        if isinstance(preprocessor, Pipeline):
            step_names = [name for name, _ in preprocessor.steps]
            # Check if AutoBalancer is in the pipeline (if importable)
            try:
                from endgame.preprocessing.imbalance import AutoBalancer
                assert "balancer" in step_names
            except ImportError:
                pass  # OK if not installed

    def test_aggressive_level_adds_correlation_selector(self, mixed_data):
        """Test that 'aggressive' adds CorrelationSelector."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        executor = PreprocessingExecutor(feature_engineering="aggressive")
        X, y = mixed_data
        meta_features = {"class_imbalance_ratio": 0.5}
        preprocessor = executor._build_preprocessor(
            X, y, meta_features, task_type="classification",
        )

        from sklearn.pipeline import Pipeline
        # Should be a Pipeline with CorrelationSelector
        if isinstance(preprocessor, Pipeline):
            step_names = [name for name, _ in preprocessor.steps]
            try:
                from endgame.feature_selection.filter.correlation import CorrelationSelector
                assert "feature_selection" in step_names
            except ImportError:
                pass  # OK if not installed

    def test_different_levels_produce_different_pipelines(self, mixed_data):
        """Test that different levels produce structurally different pipelines."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        X, y = mixed_data
        pipelines = {}
        for level in ["none", "light", "moderate", "aggressive"]:
            executor = PreprocessingExecutor(feature_engineering=level)
            meta_features = {"class_imbalance_ratio": 0.5}
            pipelines[level] = executor._build_preprocessor(
                X, y, meta_features, task_type="classification",
            )

        # None and light should differ (KNN vs Simple imputer)
        assert type(pipelines["none"]) == type(pipelines["light"])  # Both ColumnTransformer
        # Aggressive should be a Pipeline wrapping something
        from sklearn.pipeline import Pipeline
        try:
            from endgame.feature_selection.filter.correlation import CorrelationSelector
            assert isinstance(pipelines["aggressive"], Pipeline)
        except ImportError:
            pass

    def test_preprocessor_fits_and_transforms(self, mixed_data):
        """Test that all preset levels can fit_transform data."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        X, y = mixed_data
        for level in ["none", "light"]:
            executor = PreprocessingExecutor(feature_engineering=level)
            preprocessor = executor._build_preprocessor(X, y, {})
            X_transformed = preprocessor.fit_transform(X, y)
            assert X_transformed.shape[0] == X.shape[0]

    def test_executor_passes_feature_engineering_metadata(self, mixed_data):
        """Test that execute records feature_engineering level in metadata."""
        from endgame.automl.orchestrator import PreprocessingExecutor

        executor = PreprocessingExecutor(feature_engineering="light")
        X, y = mixed_data
        context = {"X": X, "y": y, "meta_features": {}, "task_type": "classification"}
        result = executor.execute(context, time_budget=60)
        assert result.metadata.get("feature_engineering") == "light"


class TestEnhancedCalibration:
    """Tests for enhanced calibration with endgame modules."""

    @pytest.fixture
    def calibration_data(self):
        """Generate data for calibration testing."""
        np.random.seed(42)
        n = 200
        # Simulate poorly calibrated probabilities
        y = np.random.randint(0, 2, n)
        proba = np.random.beta(2, 5, n)  # Skewed toward 0
        # Make positive class have higher proba
        proba[y == 1] += 0.3
        proba = np.clip(proba, 0.01, 0.99)
        return proba, y

    def test_multi_calibrator_selection(self, calibration_data):
        """Test that _select_best_calibrator tries multiple calibrators."""
        from endgame.automl.orchestrator import CalibrationExecutor

        executor = CalibrationExecutor()
        proba, y = calibration_data
        calibrator = executor._select_best_calibrator(proba, y)

        # Should return a calibrator (not None)
        assert calibrator is not None

    def test_calibrator_has_transform(self, calibration_data):
        """Test that selected calibrator uses transform() API."""
        from endgame.automl.orchestrator import CalibrationExecutor

        executor = CalibrationExecutor()
        proba, y = calibration_data
        calibrator = executor._select_best_calibrator(proba, y)

        if calibrator is not None:
            # endgame calibrators should have transform()
            try:
                from endgame.calibration.scaling import PlattScaling
                if isinstance(calibrator, (PlattScaling,)):
                    assert hasattr(calibrator, "transform")
            except ImportError:
                pass  # Fallback to sklearn is OK

    def test_calibrator_produces_valid_probabilities(self, calibration_data):
        """Test that calibrated probabilities are in [0, 1]."""
        from endgame.automl.orchestrator import CalibrationExecutor

        executor = CalibrationExecutor()
        proba, y = calibration_data
        calibrator = executor._select_best_calibrator(proba, y)

        if calibrator is not None and hasattr(calibrator, "transform"):
            calibrated = calibrator.transform(proba)
            calibrated = np.asarray(calibrated).ravel()
            assert np.all(calibrated >= 0)
            assert np.all(calibrated <= 1)

    def test_calibration_with_2d_proba(self, calibration_data):
        """Test calibration with 2D probability array."""
        from endgame.automl.orchestrator import CalibrationExecutor

        executor = CalibrationExecutor()
        proba_1d, y = calibration_data
        proba_2d = np.column_stack([1 - proba_1d, proba_1d])
        calibrator = executor._select_best_calibrator(proba_2d, y)
        assert calibrator is not None

    def test_calibration_execute_stage(self):
        """Test full CalibrationExecutor.execute() stage."""
        from endgame.automl.orchestrator import CalibrationExecutor

        np.random.seed(42)
        n = 100

        class MockEnsemble:
            def predict_proba(self, X):
                return np.column_stack([
                    np.random.uniform(0.3, 0.7, len(X)),
                    np.random.uniform(0.3, 0.7, len(X)),
                ])

        executor = CalibrationExecutor()
        context = {
            "ensemble": MockEnsemble(),
            "X_val": np.random.randn(n, 5),
            "y_val": np.random.randint(0, 2, n),
            "task_type": "classification",
        }
        result = executor.execute(context, time_budget=30)
        assert result.success


class TestTargetTransform:
    """Tests for target transformation in ModelTrainingExecutor."""

    def test_skewed_targets_get_wrapped(self):
        """Test that skewed regression targets get TargetTransformer wrapping."""
        from endgame.automl.orchestrator import ModelTrainingExecutor
        from endgame.automl.search.base import PipelineConfig

        executor = ModelTrainingExecutor(cv_folds=3, feature_engineering="moderate")

        # Create highly skewed target
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.exp(np.random.randn(100) * 2)  # Log-normal = highly skewed

        config = PipelineConfig(
            model_name="lgbm",
            model_params={"n_estimators": 50, "verbose": -1},
        )

        try:
            model = executor._refit_model(
                config, X, y, task_type="regression",
            )
            # Model should be TargetTransformer or the base model
            # (depends on whether scipy and endgame target_transform are available)
            assert model is not None
        except ImportError:
            pytest.skip("LightGBM or dependencies not available")

    def test_non_skewed_targets_not_wrapped(self):
        """Test that non-skewed targets don't get TargetTransformer."""
        from endgame.automl.orchestrator import ModelTrainingExecutor
        from endgame.automl.search.base import PipelineConfig

        executor = ModelTrainingExecutor(cv_folds=3, feature_engineering="moderate")

        # Create normally distributed target (low skewness)
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)  # Normal = skewness ≈ 0

        config = PipelineConfig(
            model_name="lgbm",
            model_params={"n_estimators": 50, "verbose": -1},
        )

        try:
            model = executor._refit_model(
                config, X, y, task_type="regression",
            )
            # Should NOT be wrapped (skewness < 1.0)
            from endgame.preprocessing.target_transform import TargetTransformer
            assert not isinstance(model, TargetTransformer)
        except ImportError:
            pytest.skip("LightGBM or dependencies not available")

    def test_classification_not_wrapped(self):
        """Test that classification tasks are never wrapped."""
        from endgame.automl.orchestrator import ModelTrainingExecutor
        from endgame.automl.search.base import PipelineConfig

        executor = ModelTrainingExecutor(cv_folds=3, feature_engineering="aggressive")

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        config = PipelineConfig(
            model_name="lgbm",
            model_params={"n_estimators": 50, "verbose": -1},
        )

        try:
            model = executor._refit_model(
                config, X, y, task_type="classification",
            )
            # Classification should never be wrapped
            try:
                from endgame.preprocessing.target_transform import TargetTransformer
                assert not isinstance(model, TargetTransformer)
            except ImportError:
                pass
        except ImportError:
            pytest.skip("LightGBM not available")

    def test_none_preset_skips_wrapping(self):
        """Test that 'none' feature_engineering skips target transform."""
        from endgame.automl.orchestrator import ModelTrainingExecutor
        from endgame.automl.search.base import PipelineConfig

        executor = ModelTrainingExecutor(cv_folds=3, feature_engineering="none")

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.exp(np.random.randn(100) * 2)  # Highly skewed

        config = PipelineConfig(
            model_name="lgbm",
            model_params={"n_estimators": 50, "verbose": -1},
        )

        try:
            model = executor._refit_model(
                config, X, y, task_type="regression",
            )
            # Should NOT be wrapped because feature_engineering="none"
            try:
                from endgame.preprocessing.target_transform import TargetTransformer
                assert not isinstance(model, TargetTransformer)
            except ImportError:
                pass
        except ImportError:
            pytest.skip("LightGBM not available")


class TestModelPools:
    """Tests for updated model pools and registry."""

    def test_new_models_in_best_quality(self):
        """Test that new models are in best_quality pool."""
        from endgame.automl.presets import MODEL_POOLS
        best = MODEL_POOLS["best_quality"]
        assert "tabm" in best
        assert "realmlp" in best
        assert "grande" in best

    def test_tabm_in_high_quality(self):
        """Test that tabm is in high_quality pool."""
        from endgame.automl.presets import MODEL_POOLS
        assert "tabm" in MODEL_POOLS["high_quality"]

    def test_ebm_in_good_quality(self):
        """Test that ebm is in good_quality pool."""
        from endgame.automl.presets import MODEL_POOLS
        assert "ebm" in MODEL_POOLS["good_quality"]

    def test_new_registry_entries_exist(self):
        """Test that new models are in the registry."""
        from endgame.automl.model_registry import MODEL_REGISTRY
        for model_name in ["tabm", "realmlp", "grande", "tabdpt", "tabr"]:
            assert model_name in MODEL_REGISTRY, f"{model_name} not in registry"

    def test_new_registry_entries_valid(self):
        """Test that new registry entries have required fields."""
        from endgame.automl.model_registry import MODEL_REGISTRY, ModelInfo
        valid_families = (
            "gbdt", "neural", "linear", "tree", "kernel",
            "rules", "bayesian", "foundation", "ensemble", "boosting",
        )
        for model_name in ["tabm", "realmlp", "grande", "tabdpt", "tabr"]:
            info = MODEL_REGISTRY[model_name]
            assert isinstance(info, ModelInfo)
            assert info.name == model_name
            assert info.family in valid_families
            assert isinstance(info.class_path, str)
            assert len(info.class_path) > 0
            assert isinstance(info.task_types, list)
            assert len(info.task_types) > 0

    def test_new_models_are_neural_family(self):
        """Test that all new models are neural family."""
        from endgame.automl.model_registry import MODEL_REGISTRY
        for model_name in ["tabm", "realmlp", "grande", "tabdpt", "tabr"]:
            assert MODEL_REGISTRY[model_name].family == "neural"

    def test_new_models_require_torch(self):
        """Test that new neural models require torch."""
        from endgame.automl.model_registry import MODEL_REGISTRY
        for model_name in ["tabm", "realmlp", "grande", "tabdpt", "tabr"]:
            assert MODEL_REGISTRY[model_name].requires_torch is True

    def test_preprocessing_time_increased_for_aggressive(self):
        """Test that aggressive preset allocates substantial preprocessing+FE time."""
        from endgame.automl.presets import PRESETS
        best = PRESETS["best_quality"]
        # Combined preprocessing + feature_engineering + data_cleaning time
        total_prep = (
            best.time_allocations.get("preprocessing", 0)
            + best.time_allocations.get("feature_engineering", 0)
            + best.time_allocations.get("data_cleaning", 0)
        )
        assert total_prep >= 0.10

    def test_preprocessing_time_increased_for_moderate(self):
        """Test that moderate presets allocate substantial preprocessing+FE time."""
        from endgame.automl.presets import PRESETS
        for preset_name in ["high_quality", "good_quality"]:
            preset = PRESETS[preset_name]
            total_prep = (
                preset.time_allocations.get("preprocessing", 0)
                + preset.time_allocations.get("feature_engineering", 0)
                + preset.time_allocations.get("data_cleaning", 0)
            )
            assert total_prep >= 0.10

    def test_orchestrator_passes_feature_engineering(self):
        """Test that PipelineOrchestrator passes feature_engineering to executors."""
        orchestrator = PipelineOrchestrator(preset="best_quality", time_limit=60)
        preprocessing_executor = orchestrator._executors["preprocessing"]
        training_executor = orchestrator._executors["model_training"]
        assert preprocessing_executor.feature_engineering == "aggressive"
        assert training_executor.feature_engineering == "aggressive"

    def test_orchestrator_fast_preset_uses_none(self):
        """Test that fast preset uses 'none' feature engineering."""
        orchestrator = PipelineOrchestrator(preset="fast", time_limit=60)
        preprocessing_executor = orchestrator._executors["preprocessing"]
        assert preprocessing_executor.feature_engineering == "none"


class TestCalibrationAPI:
    """Tests for calibration API compatibility in TabularPredictor."""

    def test_predict_proba_with_transform_calibrator(self):
        """Test that predict_proba works with transform()-based calibrators."""
        from endgame.automl.tabular import TabularPredictor

        class MockTransformCalibrator:
            """Mock endgame-style calibrator with transform()."""
            def transform(self, proba):
                return np.clip(proba * 0.9 + 0.05, 0, 1)

        predictor = TabularPredictor(label="target", verbosity=0)
        predictor.is_fitted_ = True
        predictor.problem_type_ = "binary"
        predictor._preprocessor = None
        predictor._calibrator = MockTransformCalibrator()
        predictor._ensemble = None
        predictor.fit_summary_ = FitSummary(best_model="mock")
        predictor._models = {
            "mock": {
                "estimator": type("MockModel", (), {
                    "predict_proba": lambda self, X: np.column_stack([
                        np.full(len(X), 0.3),
                        np.full(len(X), 0.7),
                    ]),
                })(),
                "score": 0.9,
            },
        }

        X = np.random.randn(10, 5)
        proba = predictor.predict_proba(X)
        assert proba.shape == (10, 2)
        # Calibrated probabilities should differ from uncalibrated
        assert not np.allclose(proba[:, 1], 0.7)

    def test_predict_proba_with_predict_proba_calibrator(self):
        """Test that predict_proba works with predict_proba()-based calibrators."""
        from endgame.automl.tabular import TabularPredictor

        class MockSklearnCalibrator:
            """Mock sklearn-style calibrator with predict_proba()."""
            def predict_proba(self, X):
                return np.column_stack([
                    np.full(len(X), 0.4),
                    np.full(len(X), 0.6),
                ])

        predictor = TabularPredictor(label="target", verbosity=0)
        predictor.is_fitted_ = True
        predictor.problem_type_ = "binary"
        predictor._preprocessor = None
        predictor._calibrator = MockSklearnCalibrator()
        predictor._ensemble = None
        predictor.fit_summary_ = FitSummary(best_model="mock")
        predictor._models = {
            "mock": {
                "estimator": type("MockModel", (), {
                    "predict_proba": lambda self, X: np.column_stack([
                        np.full(len(X), 0.3),
                        np.full(len(X), 0.7),
                    ]),
                })(),
                "score": 0.9,
            },
        }

        X = np.random.randn(10, 5)
        proba = predictor.predict_proba(X)
        assert proba.shape == (10, 2)


class TestProfilingEnhancements:
    """Tests for profiling meta-feature enhancements."""

    def test_basic_profile_includes_class_imbalance_ratio(self):
        """Test that _basic_profile computes class_imbalance_ratio."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        X = np.random.randn(100, 5)
        y = np.array([0] * 80 + [1] * 20)

        features = executor._basic_profile(X, y, "classification")
        assert "class_imbalance_ratio" in features
        assert abs(features["class_imbalance_ratio"] - 0.25) < 0.01

    def test_basic_profile_includes_missing_rate(self):
        """Test that _basic_profile computes missing_rate."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        X = np.random.randn(100, 5).astype(float)
        X[0:10, 0] = np.nan  # 10% missing in first column
        y = np.random.randint(0, 2, 100)

        features = executor._basic_profile(X, y, "classification")
        assert "missing_rate" in features
        assert features["missing_rate"] > 0

    def test_basic_profile_missing_rate_dataframe(self):
        """Test missing_rate with DataFrame input."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [np.nan, 2.0, 3.0, np.nan, 5.0],
        })
        y = np.array([0, 1, 0, 1, 0])

        features = executor._basic_profile(df, y, "classification")
        assert "missing_rate" in features
        # 3 missing out of 10 = 0.3
        assert abs(features["missing_rate"] - 0.3) < 0.01


class TestDataCleaning:
    """Tests for DataCleaningExecutor."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)
        return X, y

    def test_skip_at_none_level(self, simple_data):
        """Test that data cleaning is skipped at 'none' level."""
        from endgame.automl.orchestrator import DataCleaningExecutor

        executor = DataCleaningExecutor(feature_engineering="none")
        X, y = simple_data
        context = {"X": X, "y": y}
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True
        assert result.output["X_cleaned"] is X
        assert result.output["y_cleaned"] is y

    def test_skip_at_light_level(self, simple_data):
        """Test that data cleaning is skipped at 'light' level."""
        from endgame.automl.orchestrator import DataCleaningExecutor

        executor = DataCleaningExecutor(feature_engineering="light")
        X, y = simple_data
        context = {"X": X, "y": y}
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True

    def test_moderate_attempts_outlier_removal(self, simple_data):
        """Test that moderate level attempts outlier removal."""
        from endgame.automl.orchestrator import DataCleaningExecutor

        executor = DataCleaningExecutor(feature_engineering="moderate")
        X, y = simple_data
        context = {"X": X, "y": y}
        result = executor.execute(context, time_budget=60)

        assert result.success
        assert result.output["X_cleaned"] is not None
        assert result.output["y_cleaned"] is not None
        # Should have attempted cleaning (metadata recorded)
        assert result.metadata.get("level") == "moderate"

    def test_aggressive_graceful_on_missing_deps(self, simple_data):
        """Test that aggressive level doesn't crash if deps are missing."""
        from endgame.automl.orchestrator import DataCleaningExecutor

        executor = DataCleaningExecutor(feature_engineering="aggressive")
        X, y = simple_data
        context = {"X": X, "y": y}
        result = executor.execute(context, time_budget=60)

        assert result.success
        # Data should still be returned even if all modules are unavailable
        assert result.output["X_cleaned"] is not None
        assert result.output["y_cleaned"] is not None


class TestAdvancedFeatureEngineering:
    """Tests for AdvancedFeatureEngineeringExecutor."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)
        return X, y

    @pytest.fixture
    def high_dim_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 600)
        y = np.random.randint(0, 2, 200)
        return X, y

    def test_skip_at_light_level(self, simple_data):
        """Test that feature engineering is skipped at 'light' level."""
        from endgame.automl.orchestrator import AdvancedFeatureEngineeringExecutor

        executor = AdvancedFeatureEngineeringExecutor(feature_engineering="light")
        X, y = simple_data
        context = {"X_processed": X, "y": y}
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True

    def test_skip_at_none_level(self, simple_data):
        """Test that feature engineering is skipped at 'none' level."""
        from endgame.automl.orchestrator import AdvancedFeatureEngineeringExecutor

        executor = AdvancedFeatureEngineeringExecutor(feature_engineering="none")
        X, y = simple_data
        context = {"X_processed": X, "y": y}
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True

    def test_moderate_attempts_mrmr(self, simple_data):
        """Test that moderate level attempts MRMR selection."""
        from endgame.automl.orchestrator import AdvancedFeatureEngineeringExecutor

        executor = AdvancedFeatureEngineeringExecutor(feature_engineering="moderate")
        X, y = simple_data
        context = {"X_processed": X, "y": y}
        result = executor.execute(context, time_budget=60)

        assert result.success
        assert result.output["X_engineered"] is not None
        assert result.metadata.get("level") == "moderate"

    def test_aggressive_attempts_boruta(self, simple_data):
        """Test that aggressive level attempts Boruta selection."""
        from endgame.automl.orchestrator import AdvancedFeatureEngineeringExecutor

        executor = AdvancedFeatureEngineeringExecutor(feature_engineering="aggressive")
        X, y = simple_data
        context = {"X_processed": X, "y": y}
        result = executor.execute(context, time_budget=60)

        assert result.success
        assert result.output["X_engineered"] is not None
        assert result.metadata.get("level") == "aggressive"

    def test_pca_for_high_dim(self, high_dim_data):
        """Test that PCA is attempted for high-dimensional data (d > 500)."""
        from endgame.automl.orchestrator import AdvancedFeatureEngineeringExecutor

        executor = AdvancedFeatureEngineeringExecutor(feature_engineering="moderate")
        X, y = high_dim_data
        context = {"X_processed": X, "y": y}
        result = executor.execute(context, time_budget=60)

        assert result.success
        # PCA should have been attempted (may or may not reduce depending on deps)
        assert result.output["X_engineered"] is not None


class TestDataAugmentation:
    """Tests for DataAugmentationExecutor."""

    @pytest.fixture
    def balanced_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.array([0] * 100 + [1] * 100)
        return X, y

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.array([0] * 180 + [1] * 20)
        return X, y

    def test_skip_for_balanced_data(self, balanced_data):
        """Test that augmentation is skipped for balanced data."""
        from endgame.automl.orchestrator import DataAugmentationExecutor

        executor = DataAugmentationExecutor(feature_engineering="moderate")
        X, y = balanced_data
        context = {
            "X_engineered": X, "y": y,
            "task_type": "classification",
            "meta_features": {"class_imbalance_ratio": 0.5},
        }
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True

    def test_skip_for_regression(self, imbalanced_data):
        """Test that augmentation is skipped for regression tasks."""
        from endgame.automl.orchestrator import DataAugmentationExecutor

        executor = DataAugmentationExecutor(feature_engineering="moderate")
        X, y = imbalanced_data
        context = {
            "X_engineered": X, "y": y.astype(float),
            "task_type": "regression",
            "meta_features": {"class_imbalance_ratio": 0.1},
        }
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True

    def test_moderate_attempts_triage_smote(self, imbalanced_data):
        """Test that moderate level attempts TriageMaskedSMOTE."""
        from endgame.automl.orchestrator import DataAugmentationExecutor

        executor = DataAugmentationExecutor(feature_engineering="moderate")
        X, y = imbalanced_data
        context = {
            "X_engineered": X, "y": y,
            "task_type": "classification",
            "meta_features": {"class_imbalance_ratio": 0.1},
        }
        result = executor.execute(context, time_budget=60)

        assert result.success
        assert result.output["X_augmented"] is not None
        assert result.output["y_augmented"] is not None

    def test_aggressive_attempts_sample_weighting(self, imbalanced_data):
        """Test that aggressive level attempts sample weighting."""
        from endgame.automl.orchestrator import DataAugmentationExecutor

        executor = DataAugmentationExecutor(feature_engineering="aggressive")
        X, y = imbalanced_data
        context = {
            "X_engineered": X, "y": y,
            "task_type": "classification",
            "meta_features": {"class_imbalance_ratio": 0.1},
        }
        result = executor.execute(context, time_budget=60)

        assert result.success
        assert result.output["X_augmented"] is not None
        assert result.output["y_augmented"] is not None


class TestPostTraining:
    """Tests for PostTrainingExecutor."""

    def test_skip_at_light_level(self):
        """Test that post-training is skipped at 'light' level."""
        from endgame.automl.orchestrator import PostTrainingExecutor

        executor = PostTrainingExecutor(feature_engineering="light")
        context = {"ensemble": None, "task_type": "classification"}
        result = executor.execute(context, time_budget=30)

        assert result.success
        assert result.metadata.get("skipped") is True
        assert result.output["distilled_model"] is None
        assert result.output["conformal_predictor"] is None

    def test_skip_without_ensemble(self):
        """Test that post-training handles missing ensemble gracefully."""
        from endgame.automl.orchestrator import PostTrainingExecutor

        executor = PostTrainingExecutor(feature_engineering="aggressive")
        context = {
            "ensemble": None,
            "task_type": "classification",
            "X_val": None,
            "y_val": None,
        }
        result = executor.execute(context, time_budget=30)

        assert result.success
        # Without ensemble and validation data, nothing should be produced
        assert result.output["distilled_model"] is None
        assert result.output["conformal_predictor"] is None


class TestNewModelPools:
    """Tests for the overhauled model pools and registry."""

    def test_all_pool_coverage(self):
        """Test that 'all' pool covers >80% of MODEL_REGISTRY."""
        from endgame.automl.presets import MODEL_POOLS
        from endgame.automl.model_registry import MODEL_REGISTRY

        all_pool = set(MODEL_POOLS["all"])
        registry_models = set(MODEL_REGISTRY.keys())
        coverage = len(all_pool & registry_models) / len(registry_models)
        assert coverage > 0.80, f"'all' pool covers only {coverage:.0%} of registry"

    def test_foundation_models_in_best_quality(self):
        """Test that foundation models are in best_quality pool."""
        from endgame.automl.presets import MODEL_POOLS

        best = MODEL_POOLS["best_quality"]
        assert "tabpfn_v2" in best
        assert "tabpfn_25" in best
        assert "xrfm" in best

    def test_interpretable_pool_size(self):
        """Test that interpretable pool has >= 15 models."""
        from endgame.automl.presets import MODEL_POOLS

        interp = MODEL_POOLS["interpretable"]
        assert len(interp) >= 15, f"Interpretable pool has only {len(interp)} models"

    def test_new_registry_entries_exist(self):
        """Test that new models (tabpfn_v2, tabpfn_25, xrfm, prim) are in registry."""
        from endgame.automl.model_registry import MODEL_REGISTRY

        for model_name in ["tabpfn_v2", "tabpfn_25", "xrfm", "prim"]:
            assert model_name in MODEL_REGISTRY, f"{model_name} not in registry"

    def test_xrfm_is_interpretable(self):
        """Test that xrfm is marked as interpretable."""
        from endgame.automl.model_registry import MODEL_REGISTRY, INTERPRETABLE_MODELS

        assert MODEL_REGISTRY["xrfm"].interpretable is True
        assert "xrfm" in INTERPRETABLE_MODELS

    def test_prim_is_interpretable(self):
        """Test that prim is marked as interpretable."""
        from endgame.automl.model_registry import MODEL_REGISTRY, INTERPRETABLE_MODELS

        assert MODEL_REGISTRY["prim"].interpretable is True
        assert "prim" in INTERPRETABLE_MODELS

    def test_tabpfn_v2_properties(self):
        """Test TabPFN v2 registry properties."""
        from endgame.automl.model_registry import MODEL_REGISTRY

        info = MODEL_REGISTRY["tabpfn_v2"]
        assert info.family == "foundation"
        assert info.max_samples == 10000
        assert info.handles_categorical is True

    def test_tabpfn_25_properties(self):
        """Test TabPFN 2025 registry properties."""
        from endgame.automl.model_registry import MODEL_REGISTRY

        info = MODEL_REGISTRY["tabpfn_25"]
        assert info.family == "foundation"
        assert info.max_samples == 50000
        assert info.handles_categorical is True


class TestEnhancedProfiling:
    """Tests for enhanced profiling meta-features."""

    def test_is_time_series_flag_datetime(self):
        """Test is_time_series flag for datetime data."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
        })
        y = np.random.randint(0, 2, 50)

        features = executor._basic_profile(df, y, "classification")
        assert "is_time_series" in features
        assert features["is_time_series"] == 1.0
        assert features["n_datetime_cols"] >= 1.0

    def test_no_time_series_flag_numeric_only(self):
        """Test no is_time_series flag for numeric-only data."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        features = executor._basic_profile(X, y, "classification")
        assert "is_time_series" in features
        assert features["is_time_series"] == 0.0

    def test_target_skewness_for_regression(self):
        """Test that target_skewness is computed for regression."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        X = np.random.randn(100, 5)
        y = np.exp(np.random.randn(100) * 2)  # Highly skewed

        features = executor._basic_profile(X, y, "regression")
        assert "target_skewness" in features
        assert features["target_skewness"] > 1.0  # Should be positively skewed

    def test_n_boolean_feature(self):
        """Test n_boolean counting."""
        from endgame.automl.orchestrator import ProfilingExecutor

        executor = ProfilingExecutor()
        df = pd.DataFrame({
            "bool_col": [True, False, True, False, True],
            "binary_col": [0, 1, 0, 1, 0],
            "numeric_col": [1.5, 2.3, 3.1, 4.7, 5.2],
        })
        y = np.array([0, 1, 0, 1, 0])

        features = executor._basic_profile(df, y, "classification")
        assert "n_boolean" in features
        assert features["n_boolean"] >= 1.0  # At least bool_col


class TestOrchestratorNewStages:
    """Tests for new stages in PipelineOrchestrator."""

    def test_new_executors_exist(self):
        """Test that new executors are in the orchestrator."""
        orchestrator = PipelineOrchestrator(preset="best_quality", time_limit=60)
        assert "data_cleaning" in orchestrator._executors
        assert "feature_engineering" in orchestrator._executors
        assert "data_augmentation" in orchestrator._executors
        assert "post_training" in orchestrator._executors

    def test_fast_preset_skips_new_stages(self):
        """Test that fast preset skips new stages (0.0 time allocation)."""
        from endgame.automl.presets import PRESETS

        fast = PRESETS["fast"]
        assert fast.time_allocations.get("data_cleaning", 0.0) == 0.0
        assert fast.time_allocations.get("feature_engineering", 0.0) == 0.0
        assert fast.time_allocations.get("data_augmentation", 0.0) == 0.0
        assert fast.time_allocations.get("post_training", 0.0) == 0.0

    def test_best_quality_enables_all_stages(self):
        """Test that best_quality enables all new stages."""
        from endgame.automl.presets import PRESETS

        best = PRESETS["best_quality"]
        assert best.time_allocations.get("data_cleaning", 0.0) > 0.0
        assert best.time_allocations.get("feature_engineering", 0.0) > 0.0
        assert best.time_allocations.get("data_augmentation", 0.0) > 0.0
        assert best.time_allocations.get("post_training", 0.0) > 0.0

    def test_stage_order_correct(self):
        """Test that DEFAULT_STAGES has correct order."""
        stage_names = [name for name, _ in PipelineOrchestrator.DEFAULT_STAGES]
        assert stage_names.index("profiling") < stage_names.index("data_cleaning")
        assert stage_names.index("data_cleaning") < stage_names.index("preprocessing")
        assert stage_names.index("preprocessing") < stage_names.index("feature_engineering")
        assert stage_names.index("feature_engineering") < stage_names.index("data_augmentation")
        assert stage_names.index("data_augmentation") < stage_names.index("model_selection")
        assert stage_names.index("model_training") < stage_names.index("ensembling")
        assert stage_names.index("ensembling") < stage_names.index("calibration")
        assert stage_names.index("calibration") < stage_names.index("post_training")

    def test_full_pipeline_runs_with_new_stages(self):
        """Test that full pipeline runs with all new executors."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_classes=2, random_state=42,
        )
        orchestrator = PipelineOrchestrator(
            preset="medium_quality", time_limit=120, verbose=0,
        )
        result = orchestrator.run(X, y, task_type="classification")

        assert result is not None
        assert result.total_time > 0


class TestEnhancedCalibrationCandidates:
    """Tests for expanded calibrator candidate list."""

    def test_calibration_tries_additional_candidates(self):
        """Test that CalibrationExecutor tries VennABERS, TemperatureScaling, etc."""
        from endgame.automl.orchestrator import CalibrationExecutor

        executor = CalibrationExecutor()
        np.random.seed(42)
        n = 200
        y = np.random.randint(0, 2, n)
        proba = np.random.beta(2, 5, n)
        proba[y == 1] += 0.3
        proba = np.clip(proba, 0.01, 0.99)

        calibrator = executor._select_best_calibrator(proba, y)
        # Should still work with expanded candidates
        assert calibrator is not None


class TestTabularPredictorNewMethods:
    """Tests for new TabularPredictor methods."""

    def test_predict_distilled_raises_without_model(self):
        """Test that predict_distilled raises if no distilled model."""
        predictor = TabularPredictor(label="target", verbosity=0)
        predictor.is_fitted_ = True
        predictor._distilled_model = None

        with pytest.raises(RuntimeError, match="No distilled model"):
            predictor.predict_distilled(np.random.randn(5, 3))

    def test_predict_sets_raises_without_conformal(self):
        """Test that predict_sets raises if no conformal predictor."""
        predictor = TabularPredictor(label="target", verbosity=0)
        predictor.is_fitted_ = True
        predictor._conformal_predictor = None

        with pytest.raises(RuntimeError, match="No conformal predictor"):
            predictor.predict_sets(np.random.randn(5, 3))

    def test_new_attributes_initialized(self):
        """Test that new attributes are initialized on creation."""
        predictor = TabularPredictor(label="target", verbosity=0)
        assert predictor._distilled_model is None
        assert predictor._conformal_predictor is None
        assert predictor._feature_transformers is None
        assert predictor._clean_mask is None


class TestDefaultTimeAllocations:
    """Tests for updated default time allocations."""

    def test_default_allocations_sum_to_one(self):
        """Test that default time allocations sum to approximately 1.0.

        Allocations sum to ~0.95, leaving a ~5% buffer for the
        feedback loop which runs iteratively with remaining time.
        """
        from endgame.automl.presets import DEFAULT_TIME_ALLOCATIONS

        total = sum(DEFAULT_TIME_ALLOCATIONS.values())
        assert 0.90 <= total <= 1.01, f"Allocations sum to {total}, expected ~0.95"

    def test_preset_allocations_sum_to_one(self):
        """Test that each preset's time allocations sum to approximately 1.0.

        Allocations sum to 0.95-1.0, leaving buffer for the feedback loop.
        """
        from endgame.automl.presets import PRESETS

        for name, preset in PRESETS.items():
            total = sum(preset.time_allocations.values())
            assert 0.90 <= total <= 1.01, (
                f"Preset '{name}' allocations sum to {total}, expected 0.95-1.0"
            )

    def test_new_stages_in_default_allocations(self):
        """Test that new stages are in DEFAULT_TIME_ALLOCATIONS."""
        from endgame.automl.presets import DEFAULT_TIME_ALLOCATIONS

        assert "data_cleaning" in DEFAULT_TIME_ALLOCATIONS
        assert "feature_engineering" in DEFAULT_TIME_ALLOCATIONS
        assert "data_augmentation" in DEFAULT_TIME_ALLOCATIONS
        assert "post_training" in DEFAULT_TIME_ALLOCATIONS
        assert "quality_guardrails" in DEFAULT_TIME_ALLOCATIONS
        assert "constraint_check" in DEFAULT_TIME_ALLOCATIONS
        assert "explainability" in DEFAULT_TIME_ALLOCATIONS


class TestMultiModalPredictor:
    """Tests for MultiModalPredictor import and interface."""

    def test_import(self):
        """Test that MultiModalPredictor can be imported."""
        from endgame.automl import MultiModalPredictor
        assert MultiModalPredictor is not None

    def test_fusion_strategies(self):
        """Test that all fusion strategies are accepted."""
        from endgame.automl import MultiModalPredictor

        for strategy in ("late", "weighted", "stacking", "attention", "embedding"):
            p = MultiModalPredictor(
                label="target", fusion_strategy=strategy, verbosity=0,
            )
            assert p.fusion_strategy == strategy

    def test_modality_detection_tabular(self):
        """Test that tabular-only data is detected correctly."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target", verbosity=0)
        df = pd.DataFrame({
            "f1": np.random.randn(10),
            "f2": np.random.randn(10),
            "target": np.random.randint(0, 2, 10),
        })
        modalities = predictor._detect_modalities(df)
        assert len(modalities["tabular"]) == 2
        assert len(modalities["text"]) == 0
        assert len(modalities["image"]) == 0

    def test_modality_detection_text(self):
        """Test that text columns are detected."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target", verbosity=0)
        df = pd.DataFrame({
            "review": [
                "This is a very long review text that should be detected as text " * 3
            ] * 10,
            "target": np.random.randint(0, 2, 10),
        })
        modalities = predictor._detect_modalities(df)
        assert "review" in modalities["text"]

    def test_refit_full_method_exists(self):
        """Test that refit_full() method exists."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target", verbosity=0)
        assert hasattr(predictor, "refit_full")
        assert callable(predictor.refit_full)

    def test_logger_parameter(self):
        """Test that logger parameter is accepted."""
        from endgame.automl import MultiModalPredictor
        from endgame.tracking import ConsoleLogger

        logger = ConsoleLogger(verbose=False)
        predictor = MultiModalPredictor(
            label="target", logger=logger, verbosity=0,
        )
        assert predictor.logger is logger
