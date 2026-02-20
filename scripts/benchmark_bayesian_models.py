#!/usr/bin/env python
"""
Benchmark Bayesian Network Classifiers
======================================

This script benchmarks Endgame's Bayesian Network Classifiers against
sklearn Naive Bayes variants and XGBoost on OpenML datasets.

Models tested:
- Endgame Bayesian: TANClassifier, EBMCClassifier, ESKDBClassifier, KDBClassifier, NeuralKDBClassifier
- Sklearn Naive Bayes: GaussianNB, MultinomialNB, CategoricalNB, ComplementNB
- Baselines: XGBoost, RandomForest, LogisticRegression

Key considerations:
- Bayesian classifiers require discrete input, so we use BayesianDiscretizer
- We compare both with and without discretization for fair comparison

Usage:
    python scripts/benchmark_bayesian_models.py
    python scripts/benchmark_bayesian_models.py --quick  # Faster run with fewer datasets
    python scripts/benchmark_bayesian_models.py --openml OpenML-CC18  # Use specific suite
"""

import argparse
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_covtype, make_classification
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Sklearn baseline models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB

warnings.filterwarnings('ignore')


# =============================================================================
# Helper Classes
# =============================================================================


class DiscretizedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that discretizes data before fitting a classifier.
    
    This allows Bayesian classifiers to be used in sklearn cross-validation.
    """
    
    def __init__(
        self,
        classifier,
        discretizer_strategy: str = 'mdlp',
        max_bins: int = 10,
    ):
        self.classifier = classifier
        self.discretizer_strategy = discretizer_strategy
        self.max_bins = max_bins
        self.discretizer_ = None
    
    def fit(self, X, y):
        from endgame.preprocessing import BayesianDiscretizer
        
        self.classes_ = np.unique(y)
        
        # Fit discretizer
        self.discretizer_ = BayesianDiscretizer(
            strategy=self.discretizer_strategy,
            max_bins=self.max_bins,
        )
        X_disc = self.discretizer_.fit_transform(X, y)
        
        # Fit classifier
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X_disc, y)
        
        return self
    
    def predict(self, X):
        X_disc = self.discretizer_.transform(X)
        return self.classifier_.predict(X_disc)
    
    def predict_proba(self, X):
        X_disc = self.discretizer_.transform(X)
        return self.classifier_.predict_proba(X_disc)
    
    def get_params(self, deep=True):
        return {
            'classifier': self.classifier,
            'discretizer_strategy': self.discretizer_strategy,
            'max_bins': self.max_bins,
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SklearnDiscretizedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that uses sklearn KBinsDiscretizer for non-negative data."""
    
    def __init__(self, classifier, n_bins: int = 10, strategy: str = 'quantile'):
        self.classifier = classifier
        self.n_bins = n_bins
        self.strategy = strategy
        self.discretizer_ = None
        self.scaler_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # Scale to positive range for MultinomialNB
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        X_shifted = X_scaled - X_scaled.min(axis=0) + 0.001
        
        # Discretize
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy=self.strategy,
        )
        X_disc = self.discretizer_.fit_transform(X_shifted)
        
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X_disc, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        X_shifted = X_scaled - self.scaler_.transform(X).min(axis=0) + 0.001
        X_shifted = np.clip(X_shifted, 0.001, None)
        X_disc = self.discretizer_.transform(X_shifted)
        return self.classifier_.predict(X_disc)
    
    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        X_shifted = X_scaled - self.scaler_.transform(X).min(axis=0) + 0.001
        X_shifted = np.clip(X_shifted, 0.001, None)
        X_disc = self.discretizer_.transform(X_shifted)
        return self.classifier_.predict_proba(X_disc)
    
    def get_params(self, deep=True):
        return {
            'classifier': self.classifier,
            'n_bins': self.n_bins,
            'strategy': self.strategy,
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# =============================================================================
# Model Definitions
# =============================================================================


def get_bayesian_models() -> List[Tuple[str, Any]]:
    """Get Endgame Bayesian Network Classifiers."""
    models = []
    
    try:
        from endgame.models.bayesian import TANClassifier
        models.append(("eg.TAN", TANClassifier(
            smoothing=1.0, random_state=42
        )))
    except ImportError as e:
        print(f"  Skipping TANClassifier: {e}")
    
    try:
        from endgame.models.bayesian import KDBClassifier
        models.append(("eg.KDB-1", KDBClassifier(
            k=1, smoothing='laplace', random_state=42
        )))
        models.append(("eg.KDB-2", KDBClassifier(
            k=2, smoothing='laplace', random_state=42
        )))
    except ImportError as e:
        print(f"  Skipping KDBClassifier: {e}")
    
    try:
        from endgame.models.bayesian import EBMCClassifier
        models.append(("eg.EBMC", EBMCClassifier(
            score='bdeu', max_parents=2, random_state=42
        )))
    except ImportError as e:
        print(f"  Skipping EBMCClassifier: {e}")
    
    try:
        from endgame.models.bayesian import ESKDBClassifier
        models.append(("eg.ESKDB-10", ESKDBClassifier(
            n_estimators=10, k=2, diversity_method='sao', random_state=42
        )))
        models.append(("eg.ESKDB-30", ESKDBClassifier(
            n_estimators=30, k=2, diversity_method='sao', random_state=42
        )))
    except ImportError as e:
        print(f"  Skipping ESKDBClassifier: {e}")
    
    return models


def get_neural_bayesian_models() -> List[Tuple[str, Any]]:
    """Get Neural Bayesian models (requires PyTorch)."""
    models = []
    
    try:
        from endgame.models.bayesian import NeuralKDBClassifier
        models.append(("eg.NeuralKDB-k1", NeuralKDBClassifier(
            k=1, epochs=20, batch_size=64, learning_rate=0.01,
            random_state=42, verbose=False
        )))
        models.append(("eg.NeuralKDB-k2", NeuralKDBClassifier(
            k=2, epochs=20, batch_size=64, learning_rate=0.01,
            random_state=42, verbose=False
        )))
    except ImportError as e:
        print(f"  Skipping NeuralKDBClassifier: {e}")
    
    return models


def get_sklearn_naive_bayes() -> List[Tuple[str, Any]]:
    """Get sklearn Naive Bayes models."""
    return [
        ("sklearn.GaussianNB", GaussianNB()),
        ("sklearn.MultinomialNB-disc", SklearnDiscretizedClassifier(
            MultinomialNB(alpha=1.0), n_bins=10, strategy='quantile'
        )),
        ("sklearn.ComplementNB-disc", SklearnDiscretizedClassifier(
            ComplementNB(alpha=1.0), n_bins=10, strategy='quantile'
        )),
    ]


def get_baseline_models() -> List[Tuple[str, Any]]:
    """Get baseline comparison models."""
    models = [
        ("sklearn.RandomForest", RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )),
        ("sklearn.LogisticRegression", LogisticRegression(
            max_iter=1000, random_state=42
        )),
    ]
    
    # XGBoost
    try:
        import xgboost as xgb
        models.append(("xgb.XGBClassifier", xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1
        )))
    except ImportError:
        print("  Skipping XGBoost: not installed")
    
    return models


def load_openml_datasets(max_datasets: int = 20, max_samples: int = 2000) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Load datasets directly from OpenML."""
    datasets = []
    
    try:
        import openml
        
        # Popular classification datasets from OpenML
        dataset_ids = [
            (61, "iris"),           # iris
            (54, "vehicle"),        # vehicle
            (1462, "banknote"),     # banknote authentication
            (1464, "blood"),        # blood transfusion
            (1510, "wdbc"),         # breast cancer wisconsin
            (40975, "car"),         # car evaluation
            (40984, "segment"),     # image segmentation
            (1063, "kc2"),          # software defect
            (1467, "climate"),      # climate model simulation
            (40536, "SpeedDating"), # speed dating
            (40668, "connect-4"),   # connect-4 (large)
            (1461, "bank"),         # bank marketing
            (23, "cmc"),            # contraceptive method
            (40499, "texture"),     # texture
            (1466, "cardio"),       # cardiotocography
            (40496, "LED"),         # LED display
            (40979, "mfeat-factors"),# mfeat-factors
            (40994, "climate-simulation"), # climate
            (1468, "cnae-9"),       # cnae-9
            (40701, "churn"),       # churn
        ]
        
        for did, name in dataset_ids[:max_datasets]:
            try:
                dataset = openml.datasets.get_dataset(did, download_data=True)
                X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
                
                # Convert to numpy
                if hasattr(X, 'values'):
                    X = X.values
                if hasattr(y, 'values'):
                    y = y.values
                
                # Handle missing values
                X = np.nan_to_num(X.astype(float), nan=0.0)
                
                # Encode labels
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                
                # Subsample if too large
                if len(X) > max_samples:
                    idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
                    X, y = X[idx], y[idx]
                
                datasets.append((name, X, y))
                print(f"    Loaded: {name} ({X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes)")
                
            except Exception as e:
                print(f"    Failed to load {name} (id={did}): {e}")
                continue
    
    except ImportError:
        print("  OpenML not available")
    
    return datasets


def load_sklearn_extended_datasets(max_samples: int = 2000) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Load extended sklearn datasets."""
    datasets = []
    
    # Standard sklearn datasets
    datasets.append(("iris", load_iris().data, load_iris().target))
    datasets.append(("wine", load_wine().data, load_wine().target))
    datasets.append(("breast_cancer", load_breast_cancer().data, load_breast_cancer().target))
    datasets.append(("digits", load_digits().data, load_digits().target))
    
    # Synthetic datasets with varying characteristics
    # Easy separable
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    datasets.append(("synth_easy", X, y))
    
    # Harder with more noise
    X, y = make_classification(
        n_samples=1000, n_features=30, n_informative=8, n_redundant=10,
        n_classes=4, flip_y=0.1, random_state=42
    )
    datasets.append(("synth_noisy", X, y))
    
    # High dimensional
    X, y = make_classification(
        n_samples=500, n_features=100, n_informative=20, n_redundant=30,
        n_classes=2, random_state=42
    )
    datasets.append(("synth_highdim", X, y))
    
    # Imbalanced
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=2, weights=[0.9, 0.1], random_state=42
    )
    datasets.append(("synth_imbalanced", X, y))
    
    # Try covtype (forest cover type) - large real dataset
    try:
        covtype = fetch_covtype()
        X, y = covtype.data, covtype.target
        # Subsample
        if len(X) > max_samples:
            idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
        datasets.append(("covtype", X, y))
    except Exception as e:
        print(f"  Could not load covtype: {e}")
    
    return datasets


# =============================================================================
# Benchmark Functions
# =============================================================================


def run_benchmark(
    datasets: List[Tuple[str, np.ndarray, np.ndarray]],
    models: List[Tuple[str, Any]],
    cv_folds: int = 5,
    verbose: bool = True,
    use_discretizer: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run benchmark on multiple datasets and models.
    
    Parameters
    ----------
    datasets : list of (name, X, y) tuples
    models : list of (name, model) tuples
    cv_folds : int
        Number of CV folds
    verbose : bool
        Print progress
    use_discretizer : bool
        Wrap Bayesian models with discretizer
        
    Returns
    -------
    results : dict
        {dataset_name: {model_name: {'accuracy': mean, 'std': std, 'time': fit_time}}}
    """
    from endgame.preprocessing import BayesianDiscretizer
    
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    total_experiments = len(datasets) * len(models)
    completed = 0
    
    for dataset_name, X, y in datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            print('='*60)
        
        results[dataset_name] = {}
        
        # Preprocess: handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(np.unique(y_encoded))
        
        if verbose:
            print(f"  Classes: {n_classes}")
        
        for model_name, model in models:
            completed += 1
            
            try:
                start_time = time.time()
                
                # Determine if this is a Bayesian model that needs discretization
                is_bayesian = model_name.startswith('eg.')
                
                if is_bayesian and use_discretizer:
                    # Wrap with discretizer
                    wrapped_model = DiscretizedClassifier(
                        classifier=model,
                        discretizer_strategy='equal_freq',  # MDLP can be slow
                        max_bins=10,
                    )
                else:
                    # For other models, scale the data
                    wrapped_model = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('model', clone(model)),
                    ])
                
                # Cross-validation
                scores = cross_val_score(
                    wrapped_model, X, y_encoded,
                    cv=cv, scoring='accuracy', n_jobs=1
                )
                
                fit_time = time.time() - start_time
                
                results[dataset_name][model_name] = {
                    'accuracy': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'time': fit_time,
                }
                
                if verbose:
                    print(f"  [{completed}/{total_experiments}] {model_name}: "
                          f"{np.mean(scores):.4f} (+/- {np.std(scores):.4f}), "
                          f"time: {fit_time:.2f}s")
                
            except Exception as e:
                if verbose:
                    print(f"  [{completed}/{total_experiments}] {model_name}: FAILED - {e}")
                results[dataset_name][model_name] = {
                    'accuracy': 0.0,
                    'std': 0.0,
                    'time': 0.0,
                    'error': str(e),
                }
    
    return results


def print_results_table(results: Dict, title: str = "Results"):
    """Print results as a formatted table."""
    print(f"\n{'='*80}")
    print(title)
    print('='*80)
    
    # Collect all model names
    all_models = set()
    for dataset_results in results.values():
        all_models.update(dataset_results.keys())
    all_models = sorted(all_models)
    
    # Print header
    print(f"\n{'Model':<25}", end="")
    for dataset in results.keys():
        print(f"{dataset:<15}", end="")
    print(f"{'Mean':<10}")
    print("-" * (25 + 15 * len(results) + 10))
    
    # Print each model's results
    model_means = {}
    for model in all_models:
        print(f"{model:<25}", end="")
        accs = []
        for dataset in results.keys():
            if model in results[dataset] and 'error' not in results[dataset][model]:
                acc = results[dataset][model]['accuracy']
                accs.append(acc)
                print(f"{acc:.4f}         ", end="")
            else:
                print(f"{'FAIL':<15}", end="")
        
        if accs:
            mean_acc = np.mean(accs)
            model_means[model] = mean_acc
            print(f"{mean_acc:.4f}")
        else:
            print("N/A")
    
    # Print best models
    print("\n" + "-"*80)
    print("RANKINGS (by mean accuracy):")
    print("-"*80)
    
    sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, acc) in enumerate(sorted_models, 1):
        print(f"  {rank:2d}. {model:<25} {acc:.4f}")
    
    return sorted_models


def print_comparison_summary(results: Dict, bayesian_models: List[str], baseline_models: List[str]):
    """Print comparison between Bayesian and baseline models."""
    print("\n" + "="*80)
    print("BAYESIAN vs BASELINE COMPARISON")
    print("="*80)
    
    bayesian_accs = []
    baseline_accs = []
    
    for dataset, dataset_results in results.items():
        for model, metrics in dataset_results.items():
            if 'error' not in metrics:
                if any(model.startswith(prefix) for prefix in ['eg.', 'sklearn.GaussianNB', 'sklearn.MultinomialNB', 'sklearn.ComplementNB']):
                    bayesian_accs.append(metrics['accuracy'])
                else:
                    baseline_accs.append(metrics['accuracy'])
    
    if bayesian_accs:
        print(f"\nBayesian models mean accuracy: {np.mean(bayesian_accs):.4f} (+/- {np.std(bayesian_accs):.4f})")
    if baseline_accs:
        print(f"Baseline models mean accuracy: {np.mean(baseline_accs):.4f} (+/- {np.std(baseline_accs):.4f})")
    
    # Per-dataset comparison
    print("\nPer-dataset winners:")
    for dataset, dataset_results in results.items():
        valid = {k: v for k, v in dataset_results.items() if 'error' not in v}
        if valid:
            best_model = max(valid.items(), key=lambda x: x[1]['accuracy'])
            print(f"  {dataset}: {best_model[0]} ({best_model[1]['accuracy']:.4f})")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Bayesian Network Classifiers"
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with fewer models and datasets')
    parser.add_argument('--openml', action='store_true',
                       help='Include OpenML datasets')
    parser.add_argument('--max-datasets', type=int, default=15,
                       help='Maximum number of OpenML datasets')
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Maximum samples per dataset')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output', type=str, default='bayesian_benchmark.csv',
                       help='Output file')
    parser.add_argument('--no-neural', action='store_true',
                       help='Skip NeuralKDB models')
    args = parser.parse_args()
    
    print("="*80)
    print("BAYESIAN NETWORK CLASSIFIERS BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    
    # Load datasets
    print("\n[1/3] Loading datasets...")
    
    if args.quick:
        datasets = [
            ("iris", load_iris().data, load_iris().target),
            ("wine", load_wine().data, load_wine().target),
            ("breast_cancer", load_breast_cancer().data, load_breast_cancer().target),
        ]
    else:
        # Load extended sklearn datasets
        print("  Loading sklearn datasets...")
        datasets = load_sklearn_extended_datasets(max_samples=args.max_samples)
        
        # Optionally add OpenML datasets
        if args.openml:
            print("  Loading OpenML datasets...")
            openml_datasets = load_openml_datasets(
                max_datasets=args.max_datasets,
                max_samples=args.max_samples
            )
            datasets.extend(openml_datasets)
    
    print(f"  Total: {len(datasets)} datasets")
    
    # Collect models
    print("\n[2/3] Collecting models...")
    
    all_models = []
    
    # Bayesian models
    print("  Adding Endgame Bayesian models...")
    bayesian = get_bayesian_models()
    all_models.extend(bayesian)
    print(f"    Added {len(bayesian)} Bayesian models")
    
    # Neural Bayesian models (included by default now)
    if not args.no_neural:
        print("  Adding Neural Bayesian models...")
        neural = get_neural_bayesian_models()
        all_models.extend(neural)
        print(f"    Added {len(neural)} Neural models")
    
    # Sklearn Naive Bayes
    print("  Adding sklearn Naive Bayes...")
    nb_models = get_sklearn_naive_bayes()
    all_models.extend(nb_models)
    print(f"    Added {len(nb_models)} Naive Bayes models")
    
    # Baselines
    print("  Adding baseline models...")
    baselines = get_baseline_models()
    all_models.extend(baselines)
    print(f"    Added {len(baselines)} baseline models")
    
    if args.quick:
        # Reduce models for quick run
        quick_models = ['eg.TAN', 'eg.KDB-1', 'eg.NeuralKDB-k1', 'sklearn.GaussianNB', 
                       'sklearn.RandomForest', 'xgb.XGBClassifier']
        all_models = [(n, m) for n, m in all_models if n in quick_models]
    
    print(f"\n  Total models: {len(all_models)}")
    
    # Run benchmark
    print("\n[3/3] Running benchmark...")
    results = run_benchmark(
        datasets=datasets,
        models=all_models,
        cv_folds=args.cv_folds,
        verbose=True,
    )
    
    # Print results
    rankings = print_results_table(results, "BENCHMARK RESULTS")
    
    # Print comparison
    bayesian_names = [n for n, _ in bayesian]
    baseline_names = [n for n, _ in baselines]
    print_comparison_summary(results, bayesian_names, baseline_names)
    
    # Timing analysis
    print("\n" + "="*80)
    print("TIMING ANALYSIS")
    print("="*80)
    
    for model_name in sorted(set(m for d in results.values() for m in d.keys())):
        times = []
        for dataset_results in results.values():
            if model_name in dataset_results and 'error' not in dataset_results[model_name]:
                times.append(dataset_results[model_name]['time'])
        if times:
            print(f"  {model_name:<25}: {np.mean(times):.2f}s avg")
    
    # Save results
    print("\n" + "="*80)
    try:
        import pandas as pd
        
        rows = []
        for dataset, dataset_results in results.items():
            for model, metrics in dataset_results.items():
                rows.append({
                    'dataset': dataset,
                    'model': model,
                    'accuracy': metrics.get('accuracy', 0),
                    'std': metrics.get('std', 0),
                    'time': metrics.get('time', 0),
                    'error': metrics.get('error', ''),
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"Could not save results: {e}")
    
    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
