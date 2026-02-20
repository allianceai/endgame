#!/usr/bin/env python
"""
Comprehensive Benchmark Example for Endgame
============================================

This script demonstrates the full benchmarking and meta-learning workflow:

1. Load benchmark datasets (sklearn or OpenML)
2. Profile datasets with meta-features
3. Run cross-validation across multiple models
4. Analyze and compare results
5. Train a meta-learner for model recommendation

Usage:
    python scripts/run_benchmark.py --suite sklearn-classic --output results.parquet
    python scripts/run_benchmark.py --suite OpenML-CC18 --max-datasets 10
    python scripts/run_benchmark.py --quick  # Quick test run
"""

import argparse
import warnings
from datetime import datetime

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def get_models():
    """Define models to benchmark."""
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    models = [
        # Tree-based
        ("RandomForest", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )),
        ("ExtraTrees", ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )),
        ("GradientBoosting", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )),
        ("DecisionTree", DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
        )),
        ("AdaBoost", AdaBoostClassifier(
            n_estimators=50,
            random_state=42,
        )),
        
        # Linear models
        ("LogisticRegression", LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )),
        ("RidgeClassifier", RidgeClassifier(random_state=42)),
        
        # Instance-based
        ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        
        # Probabilistic
        ("NaiveBayes", GaussianNB()),
        
        # Neural network
        ("MLP", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
        )),
    ]
    
    # Try to add GBDT wrappers if available
    try:
        import lightgbm as lgb
        models.append(("LightGBM", lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )))
    except ImportError:
        pass
    
    try:
        import xgboost as xgb
        models.append(("XGBoost", xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )))
    except ImportError:
        pass
    
    try:
        from catboost import CatBoostClassifier
        models.append(("CatBoost", CatBoostClassifier(
            iterations=100,
            depth=7,
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )))
    except ImportError:
        pass
    
    return models


def get_simple_models():
    """Get a smaller set of models for quick testing."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    return [
        ("RandomForest", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=500, random_state=42)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ]


def run_benchmark(args):
    """Run the complete benchmark workflow."""
    from endgame.benchmark import (
        BenchmarkRunner,
        BenchmarkConfig,
        ResultsAnalyzer,
        MetaLearner,
        SuiteLoader,
    )
    
    print("=" * 70)
    print("ENDGAME BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Get models
    if args.quick:
        models = get_simple_models()
        print(f"\nQuick mode: Using {len(models)} simple models")
    else:
        models = get_models()
        print(f"\nUsing {len(models)} models")
    
    print(f"Models: {[name for name, _ in models]}")
    
    # Show available suites
    print("\nAvailable benchmark suites:")
    for name, desc in SuiteLoader.list_suites().items():
        print(f"  - {name}: {desc}")
    
    # Configure runner
    config = BenchmarkConfig(
        suite=args.suite,
        max_datasets=args.max_datasets,
        max_samples=args.max_samples,
        cv_folds=args.cv_folds,
        profile_datasets=True,
        profile_groups=["simple", "statistical"],
        verbose=True,
    )
    
    print(f"\n{'=' * 70}")
    print(f"Suite: {args.suite}")
    print(f"Max datasets: {args.max_datasets or 'all'}")
    print(f"CV folds: {args.cv_folds}")
    print(f"{'=' * 70}")
    
    # Run benchmark
    runner = BenchmarkRunner(config=config)
    tracker = runner.run(models, output_file=args.output)
    
    # Print results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(tracker.summary())
    
    # Analyze results
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    analyzer = ResultsAnalyzer(tracker, metric="accuracy")
    
    # Print ranking table
    print("\n" + analyzer.summary_table())
    
    # Statistical significance
    print("\n\nPairwise Comparisons (Wilcoxon test):")
    print("-" * 60)
    
    rankings = analyzer.rank_models()
    top_models = list(rankings.keys())[:min(5, len(rankings))]
    
    for i, model_a in enumerate(top_models):
        for model_b in top_models[i+1:]:
            try:
                comparison = analyzer.compare_models(model_a, model_b)
                sig = "*" if comparison.is_significant else ""
                print(f"{model_a} vs {model_b}: wins {comparison.wins_a}-{comparison.wins_b}, "
                      f"p={comparison.p_value:.4f}{sig}")
            except Exception as e:
                pass
    
    # Meta-feature correlations
    print("\n\nMeta-feature correlations with performance:")
    print("-" * 60)
    
    correlations = analyzer.meta_feature_correlation()
    for i, (feature, corr) in enumerate(list(correlations.items())[:10]):
        print(f"  {feature}: {corr:+.3f}")
    
    # Train meta-learner
    if args.train_metalearner and len(tracker) >= 5:
        print("\n" + "=" * 70)
        print("META-LEARNING")
        print("=" * 70)
        
        try:
            meta_learner = MetaLearner(
                approach="classification",
                metric="accuracy",
                verbose=True,
            )
            meta_learner.fit(tracker)
            
            # Show feature importances
            print("\nMeta-feature importances for model selection:")
            for feature, importance in list(meta_learner.get_feature_importances().items())[:10]:
                print(f"  {feature}: {importance:.4f}")
            
            # Test recommendation on a sample dataset
            print("\nTesting recommendation on Iris dataset...")
            from sklearn.datasets import load_iris
            iris = load_iris()
            recommendation = meta_learner.recommend(iris.data, iris.target)
            print(f"  Recommended: {recommendation.model_name}")
            print(f"  Confidence: {recommendation.confidence:.2f}")
            print(f"  Alternatives: {recommendation.alternatives}")
            print(f"  Similar datasets: {recommendation.similar_datasets}")
            
        except Exception as e:
            print(f"Meta-learning failed: {e}")
    
    # Save results
    if args.output:
        print(f"\n\nResults saved to: {args.output}")
    
    # Additional output formats
    if args.output:
        base_path = args.output.rsplit(".", 1)[0]
        
        # Save as CSV for easy viewing
        csv_path = f"{base_path}.csv"
        tracker.save(csv_path)
        print(f"CSV saved to: {csv_path}")
        
        # Save summary as text
        txt_path = f"{base_path}_summary.txt"
        with open(txt_path, "w") as f:
            f.write(f"Benchmark Summary\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Suite: {args.suite}\n\n")
            f.write(tracker.summary())
            f.write("\n\n")
            f.write(analyzer.summary_table())
        print(f"Summary saved to: {txt_path}")
    
    print("\n" + "=" * 70)
    print(f"Benchmark complete! {len(tracker)} experiments recorded.")
    print("=" * 70)
    
    return tracker


def main():
    parser = argparse.ArgumentParser(
        description="Run Endgame benchmark across multiple datasets and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sklearn datasets
  python run_benchmark.py --quick
  
  # Full sklearn classic suite
  python run_benchmark.py --suite sklearn-classic --output results.parquet
  
  # OpenML-CC18 (72 datasets)
  python run_benchmark.py --suite OpenML-CC18 --max-datasets 10
  
  # Custom configuration
  python run_benchmark.py --suite uci-popular --cv-folds 10 --max-samples 5000
        """
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        default="sklearn-classic",
        help="Benchmark suite name (default: sklearn-classic)",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Maximum number of datasets to run",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.parquet",
        help="Output file path (default: benchmark_results.parquet)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with minimal settings",
    )
    parser.add_argument(
        "--train-metalearner",
        action="store_true",
        default=True,
        help="Train meta-learner after benchmarking",
    )
    parser.add_argument(
        "--no-metalearner",
        action="store_false",
        dest="train_metalearner",
        help="Skip meta-learner training",
    )
    
    args = parser.parse_args()
    
    # Apply quick mode settings
    if args.quick:
        args.suite = "quick-test"
        args.max_datasets = 3
        args.cv_folds = 3
        args.output = "quick_benchmark_results.parquet"
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
