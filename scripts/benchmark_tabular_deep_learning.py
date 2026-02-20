#!/usr/bin/env python
"""
Benchmark Modern Tabular Deep Learning Models
==============================================

This script benchmarks the newly implemented modern tabular deep learning models
against XGBoost on OpenML classification datasets.

Models benchmarked:
- XGBoost (baseline)
- FT-Transformer (Feature Tokenizer Transformer)
- SAINT (Self-Attention and Intersample Attention Transformer)
- NODE (Neural Oblivious Decision Ensembles)
- ModernNCA (Modern Neighborhood Component Analysis)
- TabPFN (Tabular Prior-Data Fitted Network) - for small datasets only

Results are saved to the master meta-learning database at:
    ~/.endgame/meta_learning_db.parquet

This accumulates results across runs, avoiding duplicates, to build a 
comprehensive meta-learning dataset for model/pipeline recommendation.

Usage:
    python scripts/benchmark_tabular_deep_learning.py
    python scripts/benchmark_tabular_deep_learning.py --max-datasets 10
    python scripts/benchmark_tabular_deep_learning.py --quick  # Fast run
    python scripts/benchmark_tabular_deep_learning.py --suite OpenML-CC18
"""

import argparse
import gc
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone

warnings.filterwarnings('ignore')

# Endgame imports
from endgame.benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    ResultsAnalyzer,
    SuiteLoader,
    ExperimentTracker,
    MetaProfiler,
)


def get_xgboost_model():
    """Get XGBoost baseline model."""
    try:
        from endgame.models import XGBWrapper
        return ("XGBoost", XGBWrapper(
            preset='custom',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=False,
            use_gpu=False,
            verbosity=0,
        ))
    except ImportError:
        try:
            from xgboost import XGBClassifier
            # Note: eval_metric='mlogloss' works for both binary and multiclass
            return ("XGBoost", XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0,
                eval_metric='mlogloss',  # Works for both binary and multiclass
            ))
        except ImportError:
            print("Warning: XGBoost not available")
            return None


def get_ft_transformer_model(n_features: int, n_classes: int):
    """Get FT-Transformer model with reasonable defaults for benchmarking."""
    try:
        from endgame.models.tabular import FTTransformerClassifier
        return ("FTTransformer", FTTransformerClassifier(
            n_blocks=2,
            d_token=64,
            n_heads=4,
            attention_dropout=0.2,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            n_epochs=50,
            batch_size=256,
            learning_rate=1e-4,
            early_stopping=10,
            random_state=42,
            verbose=False,
        ))
    except Exception as e:
        print(f"Warning: FT-Transformer not available: {e}")
        return None


def get_saint_model(n_features: int, n_classes: int):
    """Get SAINT model with optimized defaults for benchmarking."""
    try:
        from endgame.models.tabular import SAINTClassifier
        # Optimized settings:
        # - Higher learning rate (1e-3) for faster convergence
        # - validation_fraction for internal early stopping
        return ("SAINT", SAINTClassifier(
            n_layers=3,
            d_model=32,
            n_heads=4,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            use_intersample=True,
            n_epochs=100,
            batch_size=256,
            learning_rate=1e-3,  # Higher learning rate
            early_stopping=15,
            validation_fraction=0.1,  # Internal validation for early stopping
            random_state=42,
            verbose=False,
        ))
    except Exception as e:
        print(f"Warning: SAINT not available: {e}")
        return None


def get_node_model(n_features: int, n_classes: int):
    """Get NODE model with optimized defaults for benchmarking."""
    try:
        from endgame.models.tabular import NODEClassifier
        # Optimized settings based on tuning:
        # - Higher learning rate (0.01) for faster convergence
        # - Smaller batch size (128) for more gradient updates
        # - Single layer often sufficient
        # - validation_fraction creates internal val split for early stopping
        return ("NODE", NODEClassifier(
            n_layers=1,
            n_trees=128,
            tree_depth=4,
            choice_function='softmax',
            bin_function='sigmoid',
            n_epochs=100,
            batch_size=128,
            learning_rate=0.01,  # Higher learning rate works better
            early_stopping=15,
            max_grad_norm=1.0,
            validation_fraction=0.1,  # Internal validation for early stopping
            random_state=42,
            verbose=False,
        ))
    except Exception as e:
        print(f"Warning: NODE not available: {e}")
        return None


def get_modern_nca_model(n_features: int, n_classes: int):
    """Get ModernNCA model with reasonable defaults for benchmarking."""
    try:
        from endgame.models.tabular import ModernNCAClassifier
        return ("ModernNCA", ModernNCAClassifier(
            n_neighbors=32,
            embedding_dim=64,
            temperature=0.1,
            n_epochs=50,
            batch_size=256,
            learning_rate=1e-3,
            early_stopping=10,
            random_state=42,
            verbose=False,
        ))
    except Exception as e:
        print(f"Warning: ModernNCA not available: {e}")
        return None


def get_tabpfn_model():
    """Get TabPFN model (only for small datasets)."""
    try:
        from endgame.models.tabular import TabPFNClassifier
        return ("TabPFN", TabPFNClassifier(
            n_ensemble_configurations=16,
            device='cpu',  # Use CPU for consistent benchmarking
        ))
    except Exception as e:
        print(f"Warning: TabPFN not available: {e}")
        return None


def can_use_tabpfn(n_samples: int, n_features: int, n_classes: int) -> Tuple[bool, str]:
    """Check if TabPFN can be used for this dataset.
    
    TabPFN has hard limits:
    - Max 10,000 training samples
    - Max 100 features
    - Max 10 classes
    
    Returns
    -------
    Tuple of (can_use: bool, reason: str)
    """
    issues = []
    if n_samples > 10000:
        issues.append(f"samples={n_samples}>10000")
    if n_features > 100:
        issues.append(f"features={n_features}>100")
    if n_classes > 10:
        issues.append(f"classes={n_classes}>10")
    
    if issues:
        return False, ", ".join(issues)
    return True, ""


def run_single_experiment(
    model_name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run a single model on a single dataset.
    
    Returns
    -------
    Dict with keys: accuracy, accuracy_std, f1, f1_std, fit_time, status, error
    """
    try:
        # Clone model for fresh fit
        try:
            model_clone = clone(model)
        except Exception:
            model_clone = model
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        start_time = time.time()
        
        # Cross-validation
        scores = cross_val_score(
            model_clone, X, y,
            cv=cv, scoring='accuracy', n_jobs=1
        )
        
        fit_time = time.time() - start_time
        
        # Try F1 score
        try:
            f1_scores = cross_val_score(
                clone(model) if hasattr(model, 'get_params') else model,
                X, y,
                cv=cv, scoring='f1_weighted', n_jobs=1
            )
            f1_mean = float(np.mean(f1_scores))
            f1_std = float(np.std(f1_scores))
        except Exception:
            f1_mean = 0.0
            f1_std = 0.0
        
        return {
            'accuracy': float(np.mean(scores)),
            'accuracy_std': float(np.std(scores)),
            'f1': f1_mean,
            'f1_std': f1_std,
            'fit_time': fit_time,
            'cv_scores': scores.tolist(),
            'status': 'success',
            'error': None,
        }
        
    except Exception as e:
        return {
            'accuracy': 0.0,
            'accuracy_std': 0.0,
            'f1': 0.0,
            'f1_std': 0.0,
            'fit_time': 0.0,
            'cv_scores': [],
            'status': 'failed',
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark modern tabular deep learning models"
    )
    parser.add_argument('--suite', type=str, default='uci-popular',
                       help='OpenML suite to use (default: uci-popular)')
    parser.add_argument('--max-datasets', type=int, default=20,
                       help='Maximum number of datasets (default: 20)')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum samples per dataset (default: 10000)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with fewer epochs and datasets')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip experiments already in master database (default: True)')
    parser.add_argument('--output', type=str, default=None,
                       help='Additional output file (results also saved to master DB)')
    parser.add_argument('--no-save-master', action='store_true',
                       help='Do not save to master database')
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.max_datasets = 5
        args.max_samples = 2000
        args.cv_folds = 3
    
    print("=" * 70)
    print("MODERN TABULAR DEEP LEARNING BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Create tracker
    tracker = ExperimentTracker(name=f"tabular_dl_benchmark_{args.suite}")
    
    # Load existing master database to check for existing experiments
    existing_hashes = set()
    if args.skip_existing:
        try:
            master = ExperimentTracker.load_master()
            existing_hashes = master.get_config_hashes()
            print(f"\nMaster database has {len(master)} existing experiments")
            print(f"  Path: {ExperimentTracker.get_master_db_path()}")
        except Exception as e:
            print(f"\nNote: No existing master database found ({e})")
    
    # Load datasets
    print(f"\n[1/4] Loading datasets from suite: {args.suite}")
    loader = SuiteLoader(
        suite=args.suite,
        max_datasets=args.max_datasets,
        max_samples=args.max_samples,
        verbose=True,
    )
    
    datasets = list(loader.load())
    print(f"  Loaded {len(datasets)} datasets")
    
    # Initialize profiler for meta-features
    profiler = MetaProfiler(groups=['simple', 'statistical'], verbose=False)
    
    # Track results
    all_results = []
    
    for ds_idx, dataset in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset.name}")
        print(f"  Samples: {dataset.n_samples}, Features: {dataset.n_features}, Classes: {dataset.n_classes}")
        print('='*60)
        
        # Prepare data
        X = dataset.X.copy()
        y = dataset.y.copy()
        
        # Handle NaN and encode labels
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Extract meta-features
        try:
            meta_features = profiler.profile(X, y, task_type='classification')
            mf_dict = meta_features.to_dict()
        except Exception as e:
            print(f"  Warning: Could not extract meta-features: {e}")
            mf_dict = {}
        
        # Collect models
        models = []
        
        # XGBoost baseline
        xgb = get_xgboost_model()
        if xgb:
            models.append(xgb)
        
        # FT-Transformer
        ft = get_ft_transformer_model(dataset.n_features, dataset.n_classes)
        if ft:
            models.append(ft)
        
        # SAINT
        saint = get_saint_model(dataset.n_features, dataset.n_classes)
        if saint:
            models.append(saint)
        
        # NODE
        node = get_node_model(dataset.n_features, dataset.n_classes)
        if node:
            models.append(node)
        
        # ModernNCA
        nca = get_modern_nca_model(dataset.n_features, dataset.n_classes)
        if nca:
            models.append(nca)
        
        # TabPFN (only for small datasets)
        tabpfn_ok, tabpfn_reason = can_use_tabpfn(dataset.n_samples, dataset.n_features, dataset.n_classes)
        if tabpfn_ok:
            tabpfn = get_tabpfn_model()
            if tabpfn:
                models.append(tabpfn)
        else:
            print(f"  Note: TabPFN skipped ({tabpfn_reason})")
        
        print(f"  Models to test: {[m[0] for m in models]}")
        
        # Run each model
        for model_name, model in models:
            # Check if already in master database
            from endgame.benchmark.tracker import get_experiment_hash
            config_hash = get_experiment_hash(
                dataset.name,
                model_name,
                model.get_params() if hasattr(model, 'get_params') else {},
                'classification',
            )
            
            if args.skip_existing and config_hash in existing_hashes:
                print(f"    {model_name}: SKIPPED (already in master DB)")
                continue
            
            print(f"    Running {model_name}...", end=" ", flush=True)
            
            # Use scaled data for neural models, raw for XGBoost
            X_use = X if model_name == 'XGBoost' else X_scaled
            
            result = run_single_experiment(
                model_name=model_name,
                model=model,
                X=X_use,
                y=y,
                cv_folds=args.cv_folds,
            )
            
            if result['status'] == 'success':
                print(f"accuracy={result['accuracy']:.4f} (+/- {result['accuracy_std']:.4f}), "
                      f"time={result['fit_time']:.2f}s")
            else:
                print(f"FAILED: {result['error'][:50]}...")
            
            # Log to tracker
            tracker.log_experiment(
                dataset_name=dataset.name,
                dataset_id=str(dataset.openml_id) if dataset.openml_id else None,
                model_name=model_name,
                metrics={
                    'accuracy': result['accuracy'],
                    'accuracy_std': result['accuracy_std'],
                    'f1': result['f1'],
                    'f1_std': result['f1_std'],
                },
                hyperparameters=model.get_params() if hasattr(model, 'get_params') else {},
                meta_features=mf_dict,
                cv_scores=result['cv_scores'],
                fit_time=result['fit_time'],
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                task_type='classification',
                status=result['status'],
                error_message=result['error'],
            )
            
            all_results.append({
                'dataset': dataset.name,
                'model': model_name,
                **result,
            })
            
            # Clean up memory
            gc.collect()
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save to master database
    if not args.no_save_master:
        n_added = tracker.save_to_master()
        print(f"\nSaved {n_added} new experiments to master database")
        print(f"  Path: {ExperimentTracker.get_master_db_path()}")
        
        # Show total count
        master = ExperimentTracker.load_master()
        print(f"  Total experiments in master DB: {len(master)}")
    
    # Save to additional output file if specified
    if args.output:
        tracker.save(args.output)
        print(f"\nAlso saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n" + tracker.summary())
    
    # Analyze results
    if len(tracker.get_successful()) > 0:
        analyzer = ResultsAnalyzer(tracker, metric='accuracy')
        print("\n" + analyzer.summary_table())
        
        # Per-dataset best models
        print("\n" + "-" * 50)
        print("BEST MODEL PER DATASET")
        print("-" * 50)
        
        for dataset in datasets:
            records = tracker.get_by_dataset(dataset.name)
            successful = [r for r in records if r.status == 'success']
            
            if successful:
                best = max(successful, key=lambda r: r.metrics.get('accuracy', 0))
                print(f"  {dataset.name}: {best.model_name} ({best.metrics.get('accuracy', 0):.4f})")
        
        # Model rankings
        print("\n" + "-" * 50)
        print("MODEL AVERAGE RANKINGS")
        print("-" * 50)
        
        model_scores = {}
        for record in tracker.get_successful():
            if record.model_name not in model_scores:
                model_scores[record.model_name] = []
            model_scores[record.model_name].append(record.metrics.get('accuracy', 0))
        
        rankings = []
        for model_name, scores in model_scores.items():
            rankings.append((model_name, np.mean(scores), len(scores)))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, avg_score, n_datasets) in enumerate(rankings, 1):
            print(f"  {i}. {model_name}: {avg_score:.4f} (across {n_datasets} datasets)")
    
    print("\n" + "=" * 70)
    print(f"Benchmark complete! {len(tracker)} experiments recorded.")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
