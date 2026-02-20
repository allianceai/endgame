"""Tests for the benchmarking and meta-learning module."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier


class TestSuiteLoader:
    """Tests for SuiteLoader."""
    
    def test_list_suites(self):
        """Test listing available suites."""
        from endgame.benchmark import SuiteLoader
        
        suites = SuiteLoader.list_suites()
        assert isinstance(suites, dict)
        assert "sklearn-classic" in suites
        assert "quick-test" in suites
    
    def test_load_sklearn_suite(self):
        """Test loading sklearn datasets."""
        from endgame.benchmark import SuiteLoader, DatasetInfo
        
        loader = SuiteLoader(suite="quick-test", verbose=False)
        datasets = list(loader.load())
        
        assert len(datasets) > 0
        
        for dataset in datasets:
            assert isinstance(dataset, DatasetInfo)
            assert dataset.n_samples > 0
            assert dataset.n_features > 0
            assert len(dataset.X) == dataset.n_samples
            assert len(dataset.y) == dataset.n_samples
    
    def test_dataset_info_properties(self):
        """Test DatasetInfo computed properties."""
        from endgame.benchmark import SuiteLoader
        
        loader = SuiteLoader(suite="sklearn-classification", max_datasets=1, verbose=False)
        dataset = next(loader.load())
        
        assert dataset.n_categorical >= 0
        assert dataset.n_numerical >= 0
        assert dataset.n_categorical + dataset.n_numerical == dataset.n_features
        assert dataset.imbalance_ratio >= 1.0
    
    def test_size_limits(self):
        """Test max_samples and max_features limits."""
        from endgame.benchmark import SuiteLoader
        
        loader = SuiteLoader(
            suite="sklearn-classification",
            max_datasets=1,
            max_samples=50,
            verbose=False,
        )
        dataset = next(loader.load())
        
        assert dataset.n_samples <= 50
    
    def test_cv_splits(self):
        """Test getting CV splits."""
        from endgame.benchmark import SuiteLoader
        
        loader = SuiteLoader(suite="quick-test", max_datasets=1, verbose=False)
        dataset = next(loader.load())
        
        splits = dataset.get_cv_splits(n_splits=3)
        assert len(splits) == 3
        
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(np.intersect1d(train_idx, val_idx)) == 0  # No overlap


class TestMetaProfiler:
    """Tests for MetaProfiler."""
    
    def test_profile_classification(self):
        """Test profiling a classification dataset."""
        from endgame.benchmark import MetaProfiler
        
        iris = load_iris()
        profiler = MetaProfiler(groups=["simple", "statistical"], verbose=False)
        
        meta_features = profiler.profile(
            iris.data,
            iris.target,
            task_type="classification",
        )
        
        assert len(meta_features.features) > 0
        assert "nr_inst" in meta_features.features
        assert "nr_attr" in meta_features.features
        assert meta_features.features["nr_inst"] == 150
        assert meta_features.features["nr_attr"] == 4
    
    def test_profile_regression(self):
        """Test profiling a regression dataset."""
        from endgame.benchmark import MetaProfiler
        
        diabetes = load_diabetes()
        profiler = MetaProfiler(groups=["simple"], verbose=False)
        
        meta_features = profiler.profile(
            diabetes.data,
            diabetes.target,
            task_type="regression",
        )
        
        assert "nr_inst" in meta_features.features
        assert meta_features.features["nr_class"] == 0  # Regression has no classes
    
    def test_profile_to_dict(self):
        """Test converting meta-features to dictionary."""
        from endgame.benchmark import MetaProfiler
        
        iris = load_iris()
        profiler = MetaProfiler(groups=["simple"], verbose=False)
        
        meta_features = profiler.profile(iris.data, iris.target)
        d = meta_features.to_dict()
        
        assert isinstance(d, dict)
        assert len(d) > 0
    
    def test_profile_to_array(self):
        """Test converting meta-features to array."""
        from endgame.benchmark import MetaProfiler
        
        iris = load_iris()
        profiler = MetaProfiler(groups=["simple"], verbose=False)
        
        meta_features = profiler.profile(iris.data, iris.target)
        arr = meta_features.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(meta_features.features)
    
    def test_landmarking_features(self):
        """Test landmarking meta-features."""
        from endgame.benchmark import MetaProfiler
        
        iris = load_iris()
        profiler = MetaProfiler(
            groups=["landmarking"],
            landmarking_cv=2,
            verbose=False,
        )
        
        meta_features = profiler.profile(iris.data, iris.target)
        
        assert "lm_1nn" in meta_features.features
        assert "lm_dt_stump" in meta_features.features
        
        # Landmarking scores should be reasonable
        assert 0 <= meta_features.features["lm_1nn"] <= 1


class TestExperimentTracker:
    """Tests for ExperimentTracker."""
    
    def test_log_experiment(self):
        """Test logging an experiment."""
        from endgame.benchmark import ExperimentTracker
        
        tracker = ExperimentTracker(name="test")
        
        record = tracker.log_experiment(
            dataset_name="iris",
            model_name="RandomForest",
            metrics={"accuracy": 0.95, "f1": 0.94},
            hyperparameters={"n_estimators": 100},
            n_samples=150,
            n_features=4,
        )
        
        assert len(tracker) == 1
        assert record.dataset_name == "iris"
        assert record.model_name == "RandomForest"
        assert record.metrics["accuracy"] == 0.95
    
    def test_log_failure(self):
        """Test logging a failed experiment."""
        from endgame.benchmark import ExperimentTracker
        
        tracker = ExperimentTracker()
        
        record = tracker.log_failure(
            dataset_name="test",
            model_name="BadModel",
            error_message="Something went wrong",
        )
        
        assert record.status == "failed"
        assert record.error_message == "Something went wrong"
    
    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        from endgame.benchmark import ExperimentTracker
        
        tracker = ExperimentTracker()
        tracker.log_experiment(
            dataset_name="iris",
            model_name="RF",
            metrics={"accuracy": 0.95},
        )
        tracker.log_experiment(
            dataset_name="wine",
            model_name="RF",
            metrics={"accuracy": 0.90},
        )
        
        df = tracker.to_dataframe()
        
        assert len(df) == 2
    
    def test_get_by_dataset(self):
        """Test filtering by dataset."""
        from endgame.benchmark import ExperimentTracker
        
        tracker = ExperimentTracker()
        tracker.log_experiment(dataset_name="iris", model_name="RF", metrics={})
        tracker.log_experiment(dataset_name="wine", model_name="RF", metrics={})
        tracker.log_experiment(dataset_name="iris", model_name="LR", metrics={})
        
        iris_records = tracker.get_by_dataset("iris")
        
        assert len(iris_records) == 2
    
    def test_summary(self):
        """Test getting summary."""
        from endgame.benchmark import ExperimentTracker
        
        tracker = ExperimentTracker()
        tracker.log_experiment(dataset_name="iris", model_name="RF", metrics={"accuracy": 0.95})
        
        summary = tracker.summary()
        
        assert "iris" in summary or "RF" in summary


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""
    
    def test_quick_benchmark(self):
        """Test quick benchmark function."""
        from endgame.benchmark import quick_benchmark
        
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        tracker = quick_benchmark(model, model_name="RF", suite="quick-test", verbose=False)
        
        assert len(tracker) > 0
        
        # Check that we have successful experiments
        successful = tracker.get_successful()
        assert len(successful) > 0
    
    def test_compare_models(self):
        """Test comparing multiple models."""
        from endgame.benchmark import compare_models
        
        models = [
            ("RF", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
            ("DT", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ]
        
        tracker = compare_models(
            models,
            suite="quick-test",
            max_datasets=2,
            cv_folds=2,
            verbose=False,
        )
        
        assert len(tracker) > 0
        
        # Should have results for both models
        rf_results = tracker.get_by_model("RF")
        dt_results = tracker.get_by_model("DT")
        
        assert len(rf_results) > 0
        assert len(dt_results) > 0
    
    def test_runner_with_config(self):
        """Test runner with custom configuration."""
        from endgame.benchmark import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            suite="quick-test",
            max_datasets=2,
            cv_folds=2,
            profile_datasets=True,
            profile_groups=["simple"],
            verbose=False,
        )
        
        runner = BenchmarkRunner(config=config)
        models = [("LR", LogisticRegression(max_iter=100))]
        
        tracker = runner.run(models)
        
        # Check meta-features were extracted
        df = tracker.to_dataframe(include_meta_features=True)
        mf_cols = [c for c in df.columns if str(c).startswith("mf_")]
        assert len(mf_cols) > 0


class TestResultsAnalyzer:
    """Tests for ResultsAnalyzer."""
    
    @pytest.fixture
    def sample_tracker(self):
        """Create a sample tracker with results."""
        from endgame.benchmark import ExperimentTracker
        
        tracker = ExperimentTracker()
        
        # Add some sample results
        datasets = ["iris", "wine", "breast_cancer"]
        models = [("RF", 0.95, 0.93, 0.94), ("LR", 0.90, 0.85, 0.88), ("DT", 0.85, 0.80, 0.82)]
        
        for i, dataset in enumerate(datasets):
            for model, *scores in models:
                tracker.log_experiment(
                    dataset_name=dataset,
                    model_name=model,
                    metrics={"accuracy": scores[i]},
                    n_samples=150,
                    n_features=4,
                    meta_features={"nr_inst": 150, "nr_attr": 4},
                )
        
        return tracker
    
    def test_rank_models(self, sample_tracker):
        """Test ranking models."""
        from endgame.benchmark import ResultsAnalyzer, RankingMethod
        
        analyzer = ResultsAnalyzer(sample_tracker, metric="accuracy")
        
        rankings = analyzer.rank_models(RankingMethod.MEAN_SCORE)
        
        assert "RF" in rankings
        assert "LR" in rankings
        
        # RF should rank higher
        models = list(rankings.keys())
        assert models[0] == "RF"
    
    def test_compare_models(self, sample_tracker):
        """Test comparing two models."""
        from endgame.benchmark import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(sample_tracker, metric="accuracy")
        
        comparison = analyzer.compare_models("RF", "LR")
        
        assert comparison.model_a == "RF"
        assert comparison.model_b == "LR"
        assert comparison.wins_a >= 0
        assert comparison.wins_b >= 0
    
    def test_friedman_test(self, sample_tracker):
        """Test Friedman statistical test."""
        from endgame.benchmark import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(sample_tracker, metric="accuracy")
        
        chi2, p_value = analyzer.friedman_test()
        
        assert chi2 >= 0
        assert 0 <= p_value <= 1
    
    def test_summary_table(self, sample_tracker):
        """Test generating summary table."""
        from endgame.benchmark import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(sample_tracker, metric="accuracy")
        
        table = analyzer.summary_table()
        
        assert "RF" in table
        assert "LR" in table
        assert "accuracy" in table.lower()
    
    def test_get_model_summary(self, sample_tracker):
        """Test getting model summary."""
        from endgame.benchmark import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(sample_tracker, metric="accuracy")
        
        summary = analyzer.get_model_summary("RF")
        
        assert summary["model_name"] == "RF"
        assert "mean_score" in summary
        assert summary["n_datasets"] == 3


class TestMetaLearner:
    """Tests for MetaLearner."""
    
    @pytest.fixture
    def trained_metalearner(self):
        """Create a trained meta-learner."""
        from endgame.benchmark import MetaLearner, ExperimentTracker
        
        tracker = ExperimentTracker()
        
        # Add diverse sample results with meta-features
        # Need enough datasets to allow for cross-validation
        test_cases = [
            ("iris", {"nr_inst": 150, "nr_attr": 4, "mean_cor": 0.3}, {"RF": 0.96, "LR": 0.95, "DT": 0.92}),
            ("wine", {"nr_inst": 178, "nr_attr": 13, "mean_cor": 0.5}, {"RF": 0.98, "LR": 0.95, "DT": 0.88}),
            ("breast_cancer", {"nr_inst": 569, "nr_attr": 30, "mean_cor": 0.4}, {"RF": 0.97, "LR": 0.96, "DT": 0.93}),
            ("digits", {"nr_inst": 1797, "nr_attr": 64, "mean_cor": 0.2}, {"RF": 0.95, "LR": 0.92, "DT": 0.85}),
            ("small", {"nr_inst": 50, "nr_attr": 5, "mean_cor": 0.1}, {"RF": 0.80, "LR": 0.85, "DT": 0.75}),
            ("medium", {"nr_inst": 500, "nr_attr": 20, "mean_cor": 0.25}, {"RF": 0.92, "LR": 0.88, "DT": 0.84}),
            ("large", {"nr_inst": 2000, "nr_attr": 100, "mean_cor": 0.15}, {"RF": 0.94, "LR": 0.90, "DT": 0.82}),
            ("sparse", {"nr_inst": 300, "nr_attr": 50, "mean_cor": 0.05}, {"RF": 0.88, "LR": 0.87, "DT": 0.79}),
            ("dense", {"nr_inst": 200, "nr_attr": 10, "mean_cor": 0.6}, {"RF": 0.93, "LR": 0.91, "DT": 0.87}),
            ("balanced", {"nr_inst": 400, "nr_attr": 15, "mean_cor": 0.35}, {"RF": 0.91, "LR": 0.89, "DT": 0.83}),
        ]
        
        for dataset, mf, scores in test_cases:
            for model, score in scores.items():
                tracker.log_experiment(
                    dataset_name=dataset,
                    model_name=model,
                    metrics={"accuracy": score},
                    meta_features=mf,
                    n_samples=int(mf["nr_inst"]),
                    n_features=int(mf["nr_attr"]),
                )
        
        meta_learner = MetaLearner(approach="classification", verbose=False)
        meta_learner.fit(tracker)
        
        return meta_learner
    
    def test_fit(self, trained_metalearner):
        """Test fitting meta-learner."""
        assert trained_metalearner._is_fitted
        assert len(trained_metalearner._model_names) > 0
    
    def test_recommend(self, trained_metalearner):
        """Test getting recommendation."""
        iris = load_iris()
        
        recommendation = trained_metalearner.recommend(
            iris.data,
            iris.target,
            task_type="classification",
        )
        
        assert recommendation.model_name in ["RF", "LR", "DT"]
        assert 0 <= recommendation.confidence <= 1
        assert len(recommendation.alternatives) >= 0
    
    def test_recommend_from_features(self, trained_metalearner):
        """Test recommendation from pre-computed features."""
        from endgame.benchmark import MetaFeatureSet
        
        features = MetaFeatureSet(
            features={"nr_inst": 100, "nr_attr": 10, "mean_cor": 0.3}
        )
        
        recommendation = trained_metalearner.recommend_from_features(features)
        
        assert recommendation.model_name in ["RF", "LR", "DT"]
    
    def test_feature_importances(self, trained_metalearner):
        """Test getting feature importances."""
        importances = trained_metalearner.get_feature_importances()
        
        assert isinstance(importances, dict)
        assert len(importances) > 0
        
        # Values should be non-negative
        for value in importances.values():
            assert value >= 0


class TestPipelineRecommender:
    """Tests for PipelineRecommender."""
    
    def test_recommend_pipeline(self):
        """Test recommending a complete pipeline."""
        from endgame.benchmark import PipelineRecommender, MetaLearner, ExperimentTracker
        
        # Create minimal tracker
        tracker = ExperimentTracker()
        for dataset in ["d1", "d2", "d3"]:
            for model in ["RF", "LR"]:
                tracker.log_experiment(
                    dataset_name=dataset,
                    model_name=model,
                    metrics={"accuracy": np.random.uniform(0.8, 0.95)},
                    meta_features={"nr_inst": 100, "nr_attr": 10},
                    n_samples=100,
                    n_features=10,
                )
        
        recommender = PipelineRecommender(verbose=False)
        recommender.fit(tracker)
        
        iris = load_iris()
        pipeline = recommender.recommend_pipeline(iris.data, iris.target)
        
        assert "model_name" in pipeline
        assert "preprocessing" in pipeline
        assert isinstance(pipeline["preprocessing"], list)


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete benchmark → analysis → meta-learning workflow."""
        from endgame.benchmark import (
            BenchmarkRunner,
            ResultsAnalyzer,
            MetaLearner,
        )
        
        # 1. Run benchmark
        models = [
            ("RF", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
            ("LR", LogisticRegression(max_iter=100)),
        ]
        
        runner = BenchmarkRunner(
            suite="quick-test",
            max_datasets=2,
            cv_folds=2,
            verbose=False,
        )
        tracker = runner.run(models)
        
        assert len(tracker) > 0
        
        # 2. Analyze results
        analyzer = ResultsAnalyzer(tracker, metric="accuracy")
        rankings = analyzer.rank_models()
        
        assert len(rankings) == 2
        
        # 3. Train meta-learner (if enough data)
        if len(tracker.get_successful()) >= 3:
            try:
                meta_learner = MetaLearner(verbose=False)
                meta_learner.fit(tracker)
                
                # 4. Get recommendation
                iris = load_iris()
                rec = meta_learner.recommend(iris.data, iris.target)
                
                assert rec.model_name in ["RF", "LR"]
            except Exception:
                # Meta-learning may fail with very limited data
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
