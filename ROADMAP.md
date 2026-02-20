# Endgame Development Roadmap

**Last Updated:** 2026-02-22
**Version:** 1.0.0
**Status:** Stable Release

This document tracks all implemented features and planned additions for the Endgame library.

---

## Current Implementation Status

### Core (`eg.core`) ✅

- [x] `EndgameEstimator` - Base class for all estimators
- [x] `PolarsTransformer` - Base class for Polars-based transformers
- [x] `EndgameClassifierMixin` / `EndgameRegressorMixin` - Mixins
- [x] Preset configurations (LGBM/XGB/CatBoost endgame & fast defaults)
- [x] Type definitions (`OOFResult`, `OptimizationResult`, etc.)
- [x] Polars operations (`to_lazyframe`, `from_lazyframe`, type inference)

---

### Validation (`eg.validation`) ✅

#### Adversarial Validation
- [x] `AdversarialValidator` - Detect train/test drift via classifier AUC

#### CV Splitters
- [x] `PurgedTimeSeriesSplit` - Time series CV with purging & embargo (financial competitions)
- [x] `StratifiedGroupKFold` - Stratified splits respecting groups
- [x] `MultilabelStratifiedKFold` - Stratified CV for multi-label problems
- [x] `AdversarialKFold` - CV based on adversarial validation similarity
- [x] `RepeatedStratifiedGroupKFold` - Repeated version of StratifiedGroupKFold
- [x] `CombinatorialPurgedKFold` - CPCV from de Prado's "Advances in Financial ML" for backtesting

#### CV Utilities
- [x] `cross_validate_oof` - Generate out-of-fold predictions
- [x] `check_cv_lb_correlation` - Analyze CV/LB correlation

---

### Preprocessing (`eg.preprocessing`) ✅

#### Encoding
- [x] `SafeTargetEncoder` - M-estimate smoothing with inner-fold encoding (prevents leakage)
- [x] `LeaveOneOutEncoder` - LOO target encoding
- [x] `CatBoostEncoder` - Ordered target statistics encoding
- [x] `FrequencyEncoder` - Frequency-based encoding

#### Feature Generation
- [x] `AutoAggregator` - "Magic feature" group aggregations (mean, std, skew, rank)
- [x] `InteractionFeatures` - Automatic feature interactions
- [x] `RankFeatures` - Rank-based feature transformations

#### Temporal Features
- [x] `TemporalFeatures` - Date/time feature extraction
- [x] `LagFeatures` - Lag feature generation
- [x] `RollingFeatures` - Rolling window statistics

#### Feature Selection (legacy; see `eg.feature_selection` for comprehensive framework)
- [x] `AdversarialFeatureSelector` - Select features that differ train/test
- [x] `PermutationImportanceSelector` - Permutation-based selection
- [x] `NullImportanceSelector` - Null importance feature selection

#### Discretization
- [x] `BayesianDiscretizer` - MDLP, equal-width, equal-freq, k-means discretization

#### Deep Learning Preprocessing
- [x] `DenoisingAutoEncoder` - DAE for representation learning (PyTorch)

#### Class Balancing (25+ samplers)

**Over-sampling:**
- [x] `SMOTEResampler` - Original SMOTE
- [x] `BorderlineSMOTEResampler` - Focus on borderline instances
- [x] `ADASYNResampler` - Adaptive synthetic sampling
- [x] `SVMSMOTEResampler` - SVM-based borderline detection
- [x] `KMeansSMOTEResampler` - Cluster-based oversampling
- [x] `RandomOverSampler` - Simple random duplication

**Under-sampling:**
- [x] `EditedNearestNeighbours` - Remove misclassified instances via k-NN
- [x] `AllKNNUnderSampler` - Multi-pass ENN with increasing k
- [x] `TomekLinksUnderSampler` - Remove borderline instance pairs
- [x] `RandomUnderSampler` - Simple random removal
- [x] `NearMissUnderSampler` - Distance-based selection (v1, v2, v3)
- [x] `CondensedNearestNeighbour` - Find minimal consistent subset
- [x] `OneSidedSelectionUnderSampler` - CNN + Tomek links
- [x] `NeighbourhoodCleaningRule` - ENN + majority cleaning
- [x] `InstanceHardnessThresholdSampler` - Classifier-based noise detection
- [x] `ClusterCentroidsUnderSampler` - Replace with cluster centroids

**Combined:**
- [x] `SMOTEENNResampler` - SMOTE + Edited Nearest Neighbors cleanup
- [x] `SMOTETomekResampler` - SMOTE + Tomek links removal

**Geometric (modern):**
- [x] `MultivariateGaussianSMOTE` - Multivariate Gaussian synthesis
- [x] `SimplicialSMOTE` - Simplicial complex-based synthesis
- [x] `CVSMOTEResampler` - Cross-validated SMOTE
- [x] `OverlapRegionDetector` - Detect class overlap regions

**Generative (optional: PyTorch/ctgan):**
- [x] `CTGANResampler` - Conditional GAN for tabular data
- [x] `ForestFlowResampler` - Density flow-based generation
- [x] `TabDDPMResampler` - Diffusion model for tabular data
- [x] `TabSynResampler` - Tabular synthesis via VAE+diffusion

**LLM-based (optional: transformers):**
- [x] `GReaTResampler` - LLM-based tabular data generation

**Auto-balancer:**
- [x] `AutoBalancer` - Automatic strategy selection based on imbalance ratio
- [x] Utilities: `get_imbalance_ratio()`, `get_class_distribution()`
- [x] Sampler registries: `OVER_SAMPLERS`, `UNDER_SAMPLERS`, `COMBINED_SAMPLERS`, `GEOMETRIC_SAMPLERS`, `GENERATIVE_SAMPLERS`, `LLM_SAMPLERS`, `ALL_SAMPLERS`

---

### Models (`eg.models`) ✅

#### GBDT Wrappers
- [x] `GBDTWrapper` - Unified interface for all GBDT libraries
- [x] `LGBMWrapper` - LightGBM with endgame presets
- [x] `XGBWrapper` - XGBoost with endgame presets
- [x] `CatBoostWrapper` - CatBoost with endgame presets

#### Probabilistic Models
- [x] `NGBoostRegressor` - Natural Gradient Boosting for probabilistic regression
- [x] `NGBoostClassifier` - Natural Gradient Boosting for probabilistic classification

#### Custom Tree Models
- [x] `C50Classifier` - Quinlan's C5.0 algorithm
- [x] `C50Ensemble` - Boosted C5.0 ensemble
- [x] `CubistRegressor` - Rule-based regression
- [x] `RotationForestClassifier` / `RotationForestRegressor` - PCA rotation forests
- [x] `ObliqueRandomForestClassifier` / `ObliqueRandomForestRegressor` - Oblique splits (ridge, PCA, LDA, SVM, Householder)
- [x] `ObliqueDecisionTreeClassifier` / `ObliqueDecisionTreeRegressor`
- [x] `AlternatingDecisionTreeClassifier` - ADTree for classification
- [x] `AlternatingModelTreeRegressor` - Model trees with linear models at leaves
- [x] `QuantileRegressorForest` - Quantile regression for prediction intervals & uncertainty
- [x] `EvolutionaryTreeClassifier` / `EvolutionaryTreeRegressor` - Genetic algorithm optimized trees

#### Interpretable Models
- [x] `EBMClassifier` / `EBMRegressor` - Explainable Boosting Machines
- [x] `MARSClassifier` / `MARSRegressor` - Multivariate Adaptive Regression Splines
- [x] `RuleFitClassifier` / `RuleFitRegressor` - Rule-based ensemble
- [x] `FURIAClassifier` - Fuzzy Unordered Rule Induction Algorithm (fuzzy rules)
- [x] `CORELSClassifier` - Certifiably Optimal Rule Lists
- [x] `GOSDTClassifier` - Globally Optimal Sparse Decision Trees
- [x] `NodeGAMClassifier` / `NodeGAMRegressor` - Neural Oblivious Decision Ensembles for GAMs
- [x] `GAMINetClassifier` / `GAMINetRegressor` - GAMs with Structured Interactions
- [x] `SLIMClassifier` / `FasterRiskClassifier` - Sparse Integer Linear Models for risk scores
- [x] `GAMClassifier` / `GAMRegressor` - pyGAM Generalized Additive Models

#### Bayesian Network Classifiers
- [x] `TANClassifier` - Tree Augmented Naive Bayes
- [x] `EBMCClassifier` - Efficient Bayesian Multivariate Classifier
- [x] `ESKDBClassifier` - Ensemble of Selective K-Dependence Bayes
- [x] `KDBClassifier` - K-Dependence Bayes
- [x] `NeuralKDBClassifier` - Neural KDB (PyTorch)
- [x] `AutoSLE` - Automatic structure learning ensemble
- [x] Structure learning utilities: `bdeu_score`, `bic_score`, `k2_score`, `chow_liu_tree`, `greedy_hill_climbing`

#### Neural Networks (PyTorch)
- [x] `MLPClassifier` / `MLPRegressor` - Standard MLP
- [x] `EmbeddingMLPClassifier` / `EmbeddingMLPRegressor` - MLP with categorical embeddings
- [x] `TabNetClassifier` / `TabNetRegressor` - Attention-based tabular model
- [x] `NAMClassifier` / `NAMRegressor` - Neural Additive Models (interpretable)

#### Modern Deep Tabular (PyTorch)
- [x] `TabPFNClassifier` - Prior-fitted networks (v1)
- [x] `TabTransformerClassifier` / `TabTransformerRegressor` - Contextual embeddings for categoricals
- [x] `FTTransformerClassifier` / `FTTransformerRegressor` - Feature Tokenizer Transformer
- [x] `SAINTClassifier` / `SAINTRegressor` - Self-Attention and Intersample Attention
- [x] `NODEClassifier` / `NODERegressor` - Neural Oblivious Decision Ensembles
- [x] `ModernNCAClassifier` - Modern Neighborhood Component Analysis
- [x] `GANDALFClassifier` / `GANDALFRegressor` - Gated Adaptive Network (GFLU-based)
- [x] `TabularResNetClassifier` / `TabularResNetRegressor` - ResNet for tabular data
- [x] `GRANDEClassifier` / `GRANDERegressor` - Gradient-based Neural Decision Ensembles
- [x] `TabRClassifier` / `TabRRegressor` - Retrieval-Augmented Tabular Deep Learning
- [x] `TabMClassifier` / `TabMRegressor` - Parameter-Efficient MLP Ensembling (BatchEnsemble)
- [x] `RealMLPClassifier` / `RealMLPRegressor` - Meta-Tuned MLP with Robust Preprocessing
- [x] `xRFMClassifier` / `xRFMRegressor` - Tree-Structured Recursive Feature Machines (kernel + AGOP)

#### Foundation Models / In-Context Learning
- [x] `TabPFNv2Classifier` / `TabPFNv2Regressor` - TabPFN v2 Foundation Model (Nature 2025)
- [x] `TabPFN25Classifier` / `TabPFN25Regressor` - RealTabPFN-2.5 (50K samples, 2K features)
- [x] `TabDPTClassifier` / `TabDPTRegressor` - Tabular Discriminative Pre-trained Transformer

#### Kernel Methods
- [x] `GPClassifier` / `GPRegressor` - Gaussian Process with competition defaults
- [x] `SVMClassifier` / `SVMRegressor` - Support Vector Machine wrapper

#### Random Projection Methods
- [x] `ELMClassifier` / `ELMRegressor` - Extreme Learning Machine (no backprop)

#### Simple Baselines
- [x] `NaiveBayesClassifier` - Auto-selecting Naive Bayes (Gaussian/Bernoulli/Multinomial)
- [x] `LDAClassifier` / `QDAClassifier` / `RDAClassifier` - Discriminant Analysis suite
- [x] `KNNClassifier` / `KNNRegressor` - K-Nearest Neighbors with auto-scaling
- [x] `LinearClassifier` / `LinearRegressor` - Linear models (L1/L2/ElasticNet)

#### Subgroup Discovery
- [x] `PRIMClassifier` / `PRIMRegressor` - Patient Rule Induction Method (bump hunting)
- [x] `Box` - Rectangular region representation
- [x] `PRIMResult` - Result container with trajectory and rules

#### Ordinal Regression
- [x] `OrdinalClassifier` - Auto-selecting wrapper
- [x] `LogisticAT` / `LogisticIT` / `LogisticSE` - Threshold-based ordinal models
- [x] `OrdinalRidge` / `LAD` - Regression-based ordinal models

#### Probabilistic / Bayesian
- [x] `BARTClassifier` / `BARTRegressor` - Bayesian Additive Regression Trees (PyMC)

#### Symbolic Regression
- [x] `SymbolicRegressor` / `SymbolicClassifier` - PySR equation discovery (Julia backend)

---

### Ensemble (`eg.ensemble`) ✅

- [x] `HillClimbingEnsemble` - Forward selection ensemble optimization
- [x] `StackingEnsemble` - Meta-learner stacking
- [x] `BlendingEnsemble` - Simple blending with holdout
- [x] `OptimizedBlender` - Optimized weight blending (scipy minimize)
- [x] `RankAverageBlender` - Rank-based averaging
- [x] `ThresholdOptimizer` - Classification threshold optimization

---

### Calibration (`eg.calibration`) ✅

#### Probability Calibration
- [x] `TemperatureScaling` - Neural network calibration
- [x] `PlattScaling` - Logistic calibration
- [x] `BetaCalibration` - Beta distribution calibration
- [x] `IsotonicCalibration` - Isotonic regression calibration
- [x] `HistogramBinning` - Histogram-based calibration
- [x] `VennABERS` - Venn-ABERS predictors (distribution-free)

#### Conformal Prediction
- [x] `ConformalClassifier` - Prediction sets with coverage guarantees
- [x] `ConformalRegressor` - Prediction intervals with coverage guarantees
- [x] `ConformizedQuantileRegressor` - CQR for adaptive intervals with coverage guarantees

#### Analysis
- [x] `CalibrationAnalyzer` - Comprehensive calibration analysis
- [x] `CalibrationReport` - Detailed calibration reporting
- [x] `expected_calibration_error` - ECE computation
- [x] `maximum_calibration_error` - MCE computation
- [x] `brier_score_decomposition` - Reliability/resolution/uncertainty decomposition

---

### Tuning (`eg.tune`) ✅

- [x] `OptunaOptimizer` - Optuna-based hyperparameter optimization
- [x] `get_lgbm_space` / `get_xgb_space` / `get_catboost_space` - Pre-defined search spaces
- [x] `get_space` - Generic space retrieval

---

### Quick API (`eg.quick`) ✅

- [x] `classify` - One-line classification with presets
- [x] `regress` - One-line regression with presets
- [x] `compare` - Quick model comparison with leaderboard
- [x] `PRESETS` - Preset configurations (fast, default, competition, interpretable)
- [x] `QuickResult` / `ComparisonResult` - Result containers

---

### Utils (`eg.utils`) ✅

#### Metrics
- [x] `quadratic_weighted_kappa` - QWK for ordinal classification
- [x] `map_at_k` - Mean Average Precision @ K
- [x] `ndcg_at_k` - Normalized Discounted Cumulative Gain
- [x] `competition_metric` - Generic competition metric wrapper

#### Sharpe Ratio & Multiple Testing
- [x] `sharpe_ratio` - Annualized Sharpe ratio calculation
- [x] `probabilistic_sharpe_ratio` - PSR accounting for non-normality
- [x] `deflated_sharpe_ratio` - DSR correcting for multiple testing (de Prado)
- [x] `expected_max_sharpe` - Expected max SR under null hypothesis
- [x] `analyze_sharpe` - Comprehensive Sharpe analysis with DSR
- [x] `minimum_track_record_length` - MinTRL for significance
- [x] `haircut_sharpe_ratio` - Adjusted SR for data mining
- [x] `multiple_testing_summary` - Summary report for multiple strategies

#### Submission
- [x] `SubmissionHelper` - Competition submission utilities

#### Reproducibility
- [x] `SeedEverything` / `seed_everything` - Global seed management

---

### Anomaly Detection (`eg.anomaly`) ✅

- [x] `IsolationForestDetector` - With competition-tuned defaults
- [x] `ExtendedIsolationForest` - Random hyperplane splits
- [x] `LocalOutlierFactorDetector` - Density-based anomaly detection
- [x] `GritBotDetector` - Rule-based anomaly detection with interpretable context
- [x] `PyODDetector` - Universal wrapper for 39+ PyOD algorithms
- [x] `PYOD_ALGORITHMS` - Algorithm registry
- [x] `create_detector_ensemble` - Quick ensemble of diverse detectors

---

### Semi-Supervised Learning (`eg.semi_supervised`) ✅

- [x] `SelfTrainingClassifier` - Iterative pseudo-labeling wrapper for any classifier
- [x] `SelfTrainingRegressor` - Self-training for regression with uncertainty estimation

---

### Clustering (`eg.clustering`) ✅ NEW

16 clustering algorithms with automatic method selection.

#### Centroid-Based
- [x] `KMeansClusterer` - Standard K-Means
- [x] `MiniBatchKMeansClusterer` - Scalable Mini-Batch K-Means
- [x] `KStarMeansClusterer` - K*-Means variant

#### Density-Based
- [x] `DBSCANClusterer` - DBSCAN density clustering
- [x] `HDBSCANClusterer` - Hierarchical DBSCAN
- [x] `OPTICSClusterer` - Ordering Points To Identify Clustering Structure
- [x] `DensityPeaksClusterer` - Density Peaks clustering

#### Hierarchical
- [x] `AgglomerativeClusterer` - Agglomerative hierarchical clustering
- [x] `GenieClusterer` - Genie clustering (optional: genieclust)

#### Distribution-Based
- [x] `GaussianMixtureClusterer` - Gaussian Mixture Models
- [x] `FuzzyCMeansClusterer` - Fuzzy C-Means soft clustering

#### Graph/Spectral
- [x] `SpectralClusterer` - Spectral clustering
- [x] `AffinityPropagationClusterer` - Message-passing clustering

#### Scalable
- [x] `BIRCHClusterer` - Balanced Iterative Reducing and Clustering using Hierarchies
- [x] `MeanShiftClusterer` - Mean-shift mode seeking
- [x] `FINCHClusterer` - First Integer Neighbor Clustering Hierarchy (optional: finch-clust)

#### Auto-Selection
- [x] `AutoCluster` - Automatic method selection with noise detection

---

### Feature Selection (`eg.feature_selection`) ✅

Comprehensive feature selection framework with 16+ methods across 4 categories.

#### Filter Methods
- [x] `UnivariateSelector` - Statistical tests (f_classif, f_regression, chi2, mutual_info)
- [x] `MutualInfoSelector` / `FTestSelector` / `Chi2Selector` - Specialized univariate selectors
- [x] `CorrelationSelector` - Remove highly correlated features (pearson, spearman, kendall)
- [x] `MRMRSelector` - Minimum Redundancy Maximum Relevance (MI-based)
- [x] `ReliefFSelector` - ReliefF and MultiSURF algorithms

#### Wrapper Methods
- [x] `RFESelector` - Recursive Feature Elimination (any estimator, with CV)
- [x] `SequentialSelector` - Forward/backward stepwise selection (any estimator)
- [x] `BorutaSelector` - Boruta all-relevant feature selection (shadow features)
- [x] `GeneticSelector` - Evolutionary feature selection via genetic algorithms

#### Importance-Based Methods
- [x] `PermutationSelector` - Model-agnostic permutation importance
- [x] `TreeImportanceSelector` - Tree-based feature importance (MDI, split-based)
- [x] `SHAPSelector` - SHAP-based importance selection (TreeSHAP, KernelSHAP)

#### Advanced Methods
- [x] `StabilitySelector` - Bootstrap stability selection (FDR control)
- [x] `KnockoffSelector` - Knockoff filter for FDR-controlled selection
- [x] `AdversarialFeatureSelector` - Select features that differ train/test
- [x] `NullImportanceSelector` - Null importance testing

---

### Dimensionality Reduction (`eg.dimensionality_reduction`) ✅

Comprehensive dimensionality reduction framework with 11 methods.

#### Linear Methods
- [x] `PCAReducer` - Principal Component Analysis (with automatic selection)
- [x] `RandomizedPCA` - Randomized PCA for large datasets
- [x] `KernelPCAReducer` - Kernel PCA (rbf, poly, sigmoid, cosine)
- [x] `TruncatedSVDReducer` - Sparse-friendly SVD (LSA)
- [x] `ICAReducer` - Independent Component Analysis

#### Manifold Methods
- [x] `UMAPReducer` - UMAP for embedding (local+global structure)
- [x] `ParametricUMAP` - Neural network-based UMAP
- [x] `TriMAPReducer` - TriMAP (preserves global structure better)
- [x] `PHATEReducer` - PHATE for trajectory visualization
- [x] `PaCMAPReducer` - Pairwise Controlled Manifold Approximation

#### Deep Learning Methods
- [x] `VAEReducer` - Variational AutoEncoder for learned representations (PyTorch)

---

### Signal Processing (`eg.signal`) ✅

#### Base Classes
- [x] `BaseSignalTransformer` - sklearn-compatible transformer base class
- [x] `BaseFeatureExtractor` - Feature extractor base class
- [x] `SignalMixin` - Common signal processing functionality

#### Filtering
- [x] `ButterworthFilter` - IIR Butterworth filter (lowpass, highpass, bandpass, bandstop)
- [x] `FIRFilter` - Finite Impulse Response filter with window method
- [x] `SavgolFilter` - Savitzky-Golay smoothing/derivative filter
- [x] `NotchFilter` - Powerline interference removal (with harmonics)
- [x] `MedianFilter` - Impulse noise removal
- [x] `FilterBank` - Parallel bandpass filters for band decomposition

#### Spectral Analysis
- [x] `FFTTransformer` - Fast Fourier Transform (complex, magnitude, power, phase, dB)
- [x] `WelchPSD` - Welch's method for power spectral density estimation
- [x] `MultitaperPSD` - Multitaper PSD with DPSS/Slepian tapers (adaptive weighting)
- [x] `BandPowerExtractor` - Absolute/relative band power extraction (EEG bands default)
- [x] `SpectralFeatureExtractor` - Spectral features (centroid, spread, entropy, flatness, etc.)

#### Time-Domain Features
- [x] `TimeDomainFeatures` - Comprehensive time-domain feature set
- [x] `StatisticalFeatures` - Statistical measures (mean, std, skew, kurtosis, RMS, etc.)
- [x] `HjorthParameters` - Activity, mobility, complexity
- [x] `ZeroCrossingFeatures` - Zero/threshold crossing statistics
- [x] `PeakFeatures` - Peak detection and statistics

#### Wavelet Transforms (optional: PyWavelets)
- [x] `CWTTransformer` - Continuous Wavelet Transform (time-frequency)
- [x] `DWTTransformer` - Discrete Wavelet Transform (multi-resolution)
- [x] `WaveletPacketTransformer` - Wavelet packet decomposition
- [x] `WaveletFeatureExtractor` - Wavelet-based feature extraction

#### Entropy Measures
- [x] `PermutationEntropy` - Ordinal pattern entropy
- [x] `SampleEntropy` - Template matching complexity
- [x] `ApproximateEntropy` - Regularity measure
- [x] `SpectralEntropy` - Spectral flatness entropy
- [x] `SVDEntropy` - Singular value decomposition entropy
- [x] `EntropyFeatureExtractor` - Comprehensive entropy extraction

#### Complexity / Fractal Measures
- [x] `HiguchiFD` - Higuchi fractal dimension
- [x] `PetrosianFD` - Petrosian fractal dimension
- [x] `KatzFD` - Katz fractal dimension
- [x] `HurstExponent` - Long-term memory measure (R/S analysis)
- [x] `DFA` - Detrended Fluctuation Analysis
- [x] `LempelZivComplexity` - Symbolic sequence complexity
- [x] `ComplexityFeatureExtractor` - Comprehensive complexity extraction

#### Spatial Filtering (Riemannian Geometry)
- [x] `CovarianceEstimator` - SPD covariance matrix estimation (OAS, LW, empirical)
- [x] `CSP` - Common Spatial Patterns for discriminative filtering
- [x] `FilterBankCSP` - Multi-band CSP
- [x] `TangentSpace` - Riemannian tangent space projection

#### Connectivity Features
- [x] `CoherenceFeatureExtractor` - Cross-spectral coherence
- [x] `PLVFeatureExtractor` - Phase-locking value
- [x] `BurstSuppressionFeatures` - Burst/suppression detection (EEG)
- [x] `SpikeFeatures` - Spike detection features
- [x] `ConnectivityFeatureExtractor` - Comprehensive connectivity

---

### Time Series (`eg.timeseries`) ✅

#### Base Classes
- [x] `BaseForecaster` - sklearn-compatible forecaster base class
- [x] `ForecasterMixin` - Common forecaster functionality
- [x] `UnivariateForecasterMixin` / `MultivariateForecasterMixin` - Series type mixins

#### Simple Baselines (no dependencies)
- [x] `NaiveForecaster` - Last/mean/median strategies
- [x] `SeasonalNaiveForecaster` - Seasonal naive forecasting
- [x] `MovingAverageForecaster` - Simple/weighted moving average
- [x] `ExponentialSmoothingForecaster` - SES and Holt's method
- [x] `DriftForecaster` - Random walk with drift
- [x] `ThetaForecaster` - Theta method (M3 competition winner)

#### Statistical Models (statsforecast)
- [x] `AutoARIMAForecaster` - Automatic ARIMA selection
- [x] `AutoETSForecaster` - Automatic ETS selection
- [x] `AutoThetaForecaster` - Automatic Theta method
- [x] `MSTLForecaster` - Multiple Seasonal-Trend decomposition
- [x] `CESForecaster` - Complex Exponential Smoothing

#### Neural Models (Darts)
- [x] `NBEATSForecaster` - Neural Basis Expansion Analysis
- [x] `NHITSForecaster` - Neural Hierarchical Interpolation
- [x] `TFTForecaster` - Temporal Fusion Transformer
- [x] `PatchTSTForecaster` - Patch Time Series Transformer
- [x] `DLinearForecaster` - Linear decomposition model
- [x] `TimesNetForecaster` - Temporal 2D-Variation modeling

#### Feature Extraction
- [x] `TSFreshFeatureExtractor` - tsfresh wrapper (794+ features)
- [x] `TimeSeriesFeatureExtractor` - Fast native extractor

#### Validation
- [x] `ExpandingWindowCV` - Expanding window cross-validation
- [x] `SlidingWindowCV` - Sliding window cross-validation
- [x] `BlockedTimeSeriesSplit` - Blocked time series with gaps

#### Metrics
- [x] `mase` - Mean Absolute Scaled Error
- [x] `smape` - Symmetric Mean Absolute Percentage Error
- [x] `mape` - Mean Absolute Percentage Error
- [x] `rmsse` - Root Mean Squared Scaled Error (M5 metric)
- [x] `wape` - Weighted Absolute Percentage Error
- [x] `coverage` - Prediction interval coverage
- [x] `interval_width` - Average interval width
- [x] `winkler_score` - Winkler score for intervals

#### ROCKET Family - Time Series Classification (sktime)
- [x] `RocketTransformer` - ROCKET transform (10K random kernels)
- [x] `MiniRocketTransformer` - MiniROCKET (75x faster, almost deterministic)
- [x] `MultiRocketTransformer` - MultiROCKET (multivariate + extra pooling)
- [x] `HydraTransformer` - HYDRA (dictionary + ROCKET hybrid)
- [x] `RocketClassifier` - ROCKET + RidgeClassifierCV
- [x] `MiniRocketClassifier` - MiniROCKET + RidgeClassifierCV (recommended default)
- [x] `MultiRocketClassifier` - MultiROCKET classifier
- [x] `HydraClassifier` - HYDRA classifier
- [x] `HydraMiniRocketClassifier` - Combined HYDRA + MiniROCKET (best accuracy)

---

### AutoML (`eg.automl`) ✅

Intelligent AutoML framework matching AutoGluon's 3-line simplicity while leveraging Endgame's 100+ models.

#### Core Infrastructure
- [x] `AutoMLPredictor` - Unified entry point for all modalities
- [x] `TabularPredictor` - Tabular-specific AutoML
- [x] `BasePredictor` - Base class with common interfaces
- [x] `PipelineOrchestrator` - Stage-based pipeline coordination
- [x] `TimeBudgetManager` - Time allocation across stages

#### Domain-Specific Predictors (lazy loaded)
- [x] `TextPredictor` - NLP tasks
- [x] `VisionPredictor` - Computer vision tasks
- [x] `TimeSeriesPredictor` - Forecasting tasks
- [x] `AudioPredictor` - Audio classification
- [x] `MultiModalPredictor` - Multi-modal fusion (late, weighted, stacking, attention, embedding)

#### Model Registry
- [x] `MODEL_REGISTRY` - Centralized registry with 50+ models
- [x] `ModelInfo` - Dataclass for model metadata
- [x] `register_model()` / `unregister_model()` - Dynamic registration
- [x] `get_model_info()` / `get_model_class()` / `instantiate_model()`
- [x] `list_models()` - Query models by criteria
- [x] `get_default_portfolio()` - Smart portfolio selection

#### Search Strategies
- [x] `BaseSearchStrategy` - Abstract search interface
- [x] `PortfolioSearch` - Default: diverse parallel model training
- [x] `HeuristicSearch` - Data-driven rule-based selection
- [x] `RandomSearch` - Random valid pipeline sampling
- [x] `BayesianSearch` - Optuna-wrapped Bayesian optimization
- [x] `GeneticSearch` - Evolutionary pipeline optimization

#### Presets
- [x] `best_quality` - Maximum performance (no time limit)
- [x] `high_quality` - High performance with 4-hour budget
- [x] `good_quality` - Good performance with 1-hour budget
- [x] `medium_quality` - Balanced (default, 15 min)
- [x] `fast` - Quick single-model baseline (5 min)
- [x] `interpretable` - Glass-box models only

#### Preprocessing
- [x] `AutoPreprocessor` - Automatic preprocessing pipeline selection

---

### Vision (`eg.vision`) ✅

- [x] `VisionBackbone` - timm-based backbone wrapper
- [x] `TestTimeAugmentation` - TTA pipeline (flips, rotations, scales)
- [x] `WeightedBoxesFusion` - WBF for object detection
- [x] `soft_nms` - Soft Non-Maximum Suppression
- [x] `AugmentationPipeline` - Albumentations wrapper with presets
- [x] `SegmentationModel` - segmentation-models-pytorch wrapper

---

### NLP (`eg.nlp`) ✅

#### Classification/Regression
- [x] `TransformerClassifier` / `TransformerRegressor` - HuggingFace transformers wrapper

#### Translation
- [x] `TransformerTranslator` - Sequence-to-sequence translation
- [x] `TranslationConfig` - Translation configuration
- [x] `BackTranslator` - Back-translation for data augmentation

#### Pre-training
- [x] `DomainAdaptivePretrainer` - Continue MLM on domain corpus
- [x] `PseudoLabelTrainer` - Pseudo-labeling for semi-supervised learning

#### Tokenization
- [x] `TokenizerOptimizer` - Tokenizer analysis and optimization
- [x] `TokenizationReport` - Tokenization quality report

#### LLM Utilities
- [x] `LLMWrapper` - Quantized LLM inference (4-bit, 8-bit)
- [x] `ChainOfThoughtPrompt` - CoT prompting utilities
- [x] `SyntheticDataGenerator` - Generate synthetic training data

#### Metrics
- [x] `bleu_score`, `chrf_score`, `comet_score`, `bert_score`
- [x] `rouge_score`, `meteor_score`
- [x] `translation_metrics`, `nlp_metric`

#### Preprocessing
- [x] `TextPreprocessor` - Comprehensive text cleaning
- [x] Unicode normalization, whitespace handling, HTML/URL cleaning

#### Post-processing
- [x] `LLMPostEditor` - Clean and fix LLM outputs
- [x] `TranslationPostProcessor` - Translation output cleanup

---

### Audio (`eg.audio`) ✅

- [x] `SpectrogramTransformer` - Mel/STFT spectrogram conversion
- [x] `PCENTransformer` - Per-Channel Energy Normalization
- [x] `AudioAugmentation` - Audio augmentation pipeline (SpecAugment, mixup)
- [x] `SEDModel` - Sound Event Detection model wrapper
- [x] `AudioFeatureExtractor` - Feature extraction pipeline
- [x] `PretrainedAudioClassifier` - Fine-tunable pretrained audio models

---

### Benchmark (`eg.benchmark`) ✅

#### Data Loading
- [x] `SuiteLoader` - Load benchmark suites (OpenML-CC18, etc.)
- [x] `BUILTIN_SUITES` - Pre-defined benchmark suites

#### Meta-Feature Extraction
- [x] `MetaProfiler` - Extract dataset meta-features
- [x] `MetaFeatureSet` - Feature set definitions
- [x] Simple, statistical, info-theory, landmarking meta-features
- [x] **Meta-feature caching** - Disk-based caching to avoid re-profiling datasets

#### Experiment Tracking
- [x] `ExperimentTracker` - Track experiments across runs
- [x] `ExperimentRecord` - Individual experiment records
- [x] **Incremental saving** - Save results after each trial for resumability

#### Benchmarking
- [x] `BenchmarkRunner` - Run systematic benchmarks
- [x] `BenchmarkConfig` - Benchmark configuration
- [x] `quick_benchmark` / `compare_models` - Quick comparison utilities

#### Metrics (Classification)
- [x] `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`
- [x] `roc_auc`, `roc_auc_ovr_weighted`, `log_loss`, `brier_score`, `mcc`

#### Metrics (Regression)
- [x] `r2`, `neg_mean_squared_error`, `neg_mean_absolute_error`

#### Analysis
- [x] `ResultsAnalyzer` - Analyze benchmark results
- [x] `RankingMethod` - Different ranking methodologies

#### Meta-Learning
- [x] `MetaLearner` - Learn to predict optimal models
- [x] `PipelineRecommender` - Recommend pipelines for new datasets

#### Reporting & Learning Curves
- [x] `BenchmarkReportGenerator` - Generate benchmark reports
- [x] `LearningCurveExperiment` - Learning curve experiments
- [x] `LearningCurveConfig` / `LearningCurveResults` - LC configuration and results

#### Synthetic Data
- [x] `make_rotated_blobs`, `make_hidden_structure`, `make_xor_rotated` - Controlled synthetic datasets
- [x] `get_synthetic_suite`, `get_control_dataset` - Suite utilities

---

### Kaggle Integration (`eg.kaggle`) ✅

- [x] `KaggleClient` - API client (kagglehub + legacy kaggle)
- [x] `Competition` - High-level competition interface
- [x] `CompetitionProject` - Project structure scaffolding
- [x] `CompetitionInfo`, `SubmissionInfo`, `SubmissionResult`, `DatasetInfo`

---

### Explainability (`eg.explain`) ✅ NEW

- [x] `explain()` - One-line model explanation (auto-selects best method)
- [x] `Explanation` - Dataclass with `values`, `base_value`, `feature_names`, `plot()`, `to_dataframe()`, `top_features()`
- [x] `BaseExplainer` - ABC for all explainers
- [x] `SHAPExplainer` - Auto-detection: tree/linear/kernel/deep SHAP
- [x] `LIMEExplainer` - LIME explanations for any model
- [x] `PartialDependence` - 1D and 2D partial dependence
- [x] `FeatureInteraction` - H-statistic interaction detection
- [x] `CounterfactualExplainer` - DiCE counterfactual generation

---

### Fairness (`eg.fairness`) ✅ NEW

#### Metrics
- [x] `demographic_parity()` - Selection rate by group
- [x] `equalized_odds()` - TPR/FPR parity across groups
- [x] `disparate_impact()` - Four-fifths rule ratio
- [x] `calibration_by_group()` - Calibration metrics per group

#### Mitigation
- [x] `ReweighingPreprocessor` - Sample weight adjustment (sklearn TransformerMixin)
- [x] `ExponentiatedGradient` - Fairness-constrained training (fairlearn wrapper)
- [x] `CalibratedEqOdds` - Post-processing threshold adjustment per group

#### Reporting
- [x] `FairnessReport` - Standalone HTML fairness report

---

### Persistence & Deployment (`eg.persistence`) ✅ NEW

- [x] `save()` / `load()` - Model serialization with metadata
- [x] `export_onnx()` - ONNX export with auto-backend detection (skl2onnx, hummingbird, torch.onnx)
- [x] `ModelServer` - ONNX Runtime inference server (`predict()`, `predict_proba()`, `predict_raw()`)

---

### Experiment Tracking (`eg.tracking`) ✅ NEW

- [x] `ExperimentLogger` - Abstract base class for experiment logging backends
- [x] `MLflowLogger` - MLflow-backed logger (lazy import, param flattening, model flavor auto-detection)
- [x] `ConsoleLogger` - Zero-dependency console/file logger (JSON lines output)
- [x] `get_logger()` - Factory function (`"console"` or `"mlflow"` backend)
- [x] AutoML integration: optional `logger` parameter on `BasePredictor`, `TabularPredictor`, `MultiModalPredictor`
- [x] Quick API integration: optional `logger` parameter on `classify()`, `regress()`, `compare()`

---

## Planned Additions: Priority Roadmap

The items below are organized by strategic priority for making Endgame a truly SOTA, comprehensive ML framework. Priorities reflect the current landscape where tabular foundation models, data-centric AI, and deployment tooling are the key differentiators.

### Tier 0: Critical Gaps (Essential for SOTA Competitiveness)

These features represent the biggest gaps relative to AutoGluon and the current SOTA landscape.

#### Tabular Foundation Models
- [x] **TabPFN v2 Integration** - Tabular foundation model (Nature 2025) ✅
  - Reference: Hollmann et al., 2025 - "Accurate predictions on small data with a tabular foundation model" (Nature)

- [x] **TabR** - Retrieval-Augmented Tabular Model ✅
  - Reference: Gorishniy et al., 2024 - "TabR: Tabular Deep Learning Meets Nearest Neighbors"

#### Data Quality & Preprocessing
- [x] **Missing Data Module** (`eg.preprocessing.imputation`) ✅
  - Implemented: `MICEImputer`, `MissForestImputer`, `KNNImputer`, `SimpleImputer`, `IndicatorImputer`, `AutoImputer`

- [x] **Target Transformation** - Automatic target normalization ✅
  - Implemented: `TargetTransformer`, `TargetQuantileTransformer` (Box-Cox, Yeo-Johnson, log, quantile, auto)

- [x] **Label Noise Detection** - Confident Learning / data cleaning ✅
  - Implemented: `ConfidentLearningFilter`, `ConsensusFilter`, `CrossValNoiseDetector`
  - Reference: Northcutt et al., 2021 - "Confident Learning: Estimating Uncertainty in Dataset Labels"

#### Deployment & Production
- [x] **Knowledge Distillation** - Compress ensemble to single model ✅
  - Implemented: `KnowledgeDistiller` with soft/hard label distillation, temperature scaling

- [x] **ONNX Export** - Portable model format ✅
  - Implemented: `export_onnx()` with auto-backend detection (skl2onnx for sklearn, hummingbird for trees, torch.onnx for neural). `ModelServer` for ONNX Runtime inference.

- [x] **Refit on Full Data** - Collapse bagging for deployment ✅
  - Implemented: `refit_full()` on `BasePredictor`, `TabularPredictor`, `MultiModalPredictor`. Retrains all models + ensemble on full train+val data for deployment.

#### Validation & Evaluation
- [x] **Nested CV** - Proper nested cross-validation utility ✅
  - Implemented: `NestedCV` with `NestedCVResult` dataclass

#### Multi-Target Learning
- [x] **Multi-Output Wrapper** - Native multi-target support ✅
  - Implemented: `MultiOutputClassifier`, `MultiOutputRegressor`, `ClassifierChain`, `RegressorChain`

#### Visualization
- [x] **Interactive Decision Tree Visualization** - HTML/JS tree rendering ✅
  - Implemented: `TreeVisualizer` with interactive D3.js-based HTML rendering
- [x] **Comprehensive Visualization Suite** - 42 interactive chart types ✅
  - Shared infrastructure: `BaseVisualizer` ABC, palettes (_palettes.py), HTML template (_html_template.py)
  - **Tier 1 (Critical):** `BarChartVisualizer`, `HeatmapVisualizer`, `ConfusionMatrixVisualizer`, `HistogramVisualizer`, `LineChartVisualizer`, `ScatterplotVisualizer`
  - **Tier 2 (Important):** `BoxPlotVisualizer`, `ViolinPlotVisualizer`, `ErrorBarsVisualizer`, `ParallelCoordinatesVisualizer`, `RadarChartVisualizer`, `TreemapVisualizer`, `SunburstVisualizer`
  - **Tier 3 (Nice-to-have):** `SankeyVisualizer`, `DotMatrixVisualizer`, `VennDiagramVisualizer`, `WordCloudVisualizer`
  - **Tier 4 (Extended):** `ArcDiagramVisualizer`, `ChordDiagramVisualizer`, `DonutChartVisualizer`, `FlowChartVisualizer`, `NetworkDiagramVisualizer` (Bayesian Network support), `NightingaleRoseVisualizer`, `RadialBarVisualizer`, `SpiralPlotVisualizer`, `StreamGraphVisualizer`
  - **Tier A (ML Evaluation):** `ROCCurveVisualizer`, `PRCurveVisualizer`, `CalibrationPlotVisualizer`, `LiftChartVisualizer`, `PDPVisualizer`/`PDP2DVisualizer`, `WaterfallVisualizer`
  - **Tier B (General-Purpose):** `RidgelinePlotVisualizer`, `BumpChartVisualizer`, `LollipopChartVisualizer`, `DumbbellChartVisualizer`, `FunnelChartVisualizer`, `GaugeChartVisualizer`
  - All charts: self-contained HTML (no CDN), dark/light themes, Jupyter display, classmethod constructors for ML patterns

#### Documentation
- [x] **Documentation Directory Setup** - Sphinx/MkDocs scaffolding ✅
  - Implemented: `docs/` with configuration, API reference, guides, and module index

---

### Tier 1: SOTA Deep Tabular Models

These are the latest high-performing tabular models from the research literature.

- [x] **GRANDE** - Gradient-based Decision Tree Ensembles ✅
  - Reference: Marton et al., 2024 - "GRANDE: Gradient-Based Decision Tree Ensembles"

- [x] **RealMLP** - Regularized MLP Baseline ✅
  - Reference: Holzmüller et al., 2024 - "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data"

- [x] **TabM** - Parameter-Efficient MLP Ensembling ✅
  - Reference: Gorishniy et al., 2025 - "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"

- [x] **TabDPT** - Tabular Discriminative Pre-trained Transformer ✅
  - Reference: Ma et al., 2024 - "TabDPT: Scaling Tabular Foundation Models"

- [x] **TabPFN 2.5** - RealTabPFN-2.5 (50K samples, 2K features) ✅
  - Reference: PriorLabs, 2025

- [x] **xRFM** - Tree-Structured Recursive Feature Machines ✅
  - Reference: Beaglehole & Holzmüller, 2025 - "xRFM: Accurate, scalable, and interpretable feature learning models for tabular data"

- [ ] **ExcelFormer** - Enhanced Tabular Transformer
  - Priority: 🟡 Medium
  - Complexity: High
  - Notes: Semi-permeable attention module addressing rotational invariance. Beats FT-Transformer. Includes tailored data augmentation and attentive feedforward network.
  - Reference: Chen et al., 2023 - "ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"

- [ ] **TabICL** - In-Context Learning for Tabular Data
  - Priority: 🟡 Medium
  - Complexity: High
  - Notes: Transformer-based in-context learning competitive with TabPFN on larger datasets. Strong on numeric features.
  - Reference: Ye et al., 2025 - "TabICL: A Tabular Foundation Model for In-Context Learning"

- [ ] **Trompt** - Tabular Prompt Learning
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: Prompt-based tabular learning for efficient few-shot transfer. Novel approach using column-specific prompts.
  - Reference: Chen et al., 2023 - "Trompt: Towards a Better Deep Neural Network for Tabular Data"

- [ ] **DANet** - Deep Abstract Networks
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: Explicit feature abstraction layers for tabular data.

- [ ] **AutoInt** - Automatic Feature Interaction Learning
  - Priority: 🟡 Medium
  - Complexity: Medium
  - Notes: Self-attention for automatic feature interactions. Good complement to tree-based models.

---

### Tier 2: Data-Centric AI & Explainability

These features address the growing importance of data quality, fairness, and model interpretability.

#### Explainability Module (`eg.explain`)
- [x] **Unified Explainability Framework** ✅
  - Implemented: `explain()` convenience function, `SHAPExplainer` (auto: tree/linear/kernel/deep), `LIMEExplainer`, `CounterfactualExplainer` (DiCE), `PartialDependence` (1D + 2D), `FeatureInteraction` (H-statistic)
  - Unified API: `explain(model, X)` → `Explanation` dataclass with `plot()`, `to_dataframe()`, `top_features()`

- [x] **Feature Interaction Detection** ✅
  - Implemented: `FeatureInteraction` using H-statistic. Automatic detection and ranking of feature interactions.

#### Fairness Module (`eg.fairness`)
- [x] **Fairness Metrics** ✅
  - Implemented: `demographic_parity()`, `equalized_odds()`, `disparate_impact()`, `calibration_by_group()`. All with signature `f(y_true, y_pred, sensitive_attr) -> dict`.

- [x] **Bias Mitigation** ✅
  - Implemented: `ReweighingPreprocessor` (sample weight adjustment), `ExponentiatedGradient` (fairlearn wrapper), `CalibratedEqOdds` (post-processing threshold adjustment per group), `FairnessReport` (HTML generation).

#### Data Quality Module (`eg.data_quality`)
- [ ] **Data Profiling**
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: `DataProfiler` generating automatic reports: type inference, missing patterns, cardinality, distribution, outlier summary, feature correlations. Like pandas-profiling but integrated.

- [ ] **Duplicate Detection**
  - Priority: 🟢 Low
  - Complexity: Low
  - Notes: Fuzzy duplicate detection, near-duplicate removal. Important for data cleaning.

- [ ] **Data Drift Detection**
  - Priority: 🟡 Medium
  - Complexity: Medium
  - Notes: `DriftDetector` with PSI, KS-test, MMD, adversarial detection. Extends existing `AdversarialValidator` concept to production monitoring.

---

### Tier 3: Advanced Learning Paradigms

#### Active Learning (`eg.active_learning`)
- [ ] **Query Strategies**
  - Priority: 🟡 Medium
  - Complexity: Medium
  - Notes: `UncertaintySampling`, `QueryByCommittee`, `ExpectedModelChange`, `DensityWeighted`, `BatchBALD`. Unified `ActiveLearner` wrapper with pool-based and stream-based modes.
  - Reference: Settles, 2009 - "Active Learning Literature Survey"

#### Online / Streaming Learning
- [ ] **River Integration** - Online learning wrapper
  - Priority: 🟢 Low
  - Complexity: Low
  - Notes: Wraps River library for concept drift adaptation, online GBDTs, Hoeffding trees.

- [ ] **Mondrian Forest** - Online random forest
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: Streaming data, incremental updates without full retraining.

#### Survival Analysis (`eg.survival`)
- [ ] **Random Survival Forest**
  - Priority: 🟡 Medium
  - Complexity: Medium
  - Notes: scikit-survival integration. Important for healthcare, churn prediction.

- [ ] **Cox Proportional Hazards**
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: Cox PH wrapper with regularization. Foundation of survival analysis.

- [ ] **DeepSurv** - Neural Cox model
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Deep learning for survival analysis.

#### Causal Inference (`eg.causal`)
- [ ] **Treatment Effect Estimation**
  - Priority: 🟡 Medium
  - Complexity: High
  - Notes: `CausalForest`, `DoublyRobustEstimator`, `InversePropensityWeighting`. Growing importance in industry. DoWhy/EconML integration.
  - Reference: Athey & Imbens, 2019 - "Generalized Random Forests"

#### Semi-Supervised Extensions
- [ ] **Label Propagation** - Graph-based semi-supervised
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: Useful for clustered data. sklearn wrapper with competition defaults.

- [ ] **FixMatch / MixMatch Trainer** - Consistency regularization
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: Modern semi-supervised for neural models.

---

### Tier 4: Additional Models & Methods

#### Rule Induction
- [ ] **RIPPER (JRip)** - Repeated Incremental Pruning
  - Priority: 🟡 Medium
  - Complexity: Medium
  - Notes: Efficient rule learner, basis for FURIA. Interpretable ordered rules.
  - Reference: Cohen, 1995 - "Fast Effective Rule Induction"

- [ ] **CN2 Rule Learner** - Covering algorithm
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Classic rule learner producing unordered rule sets.

#### High-Dimensional Methods
- [ ] **Sparse LDA** - L1-penalized discriminant analysis
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: Feature selection built into LDA. Good for p >> n.

- [ ] **High-Dimensional DA (hdda)** - For p >> n problems
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Intrinsic dimensionality estimation.

#### Distance-Based Methods
- [ ] **Relevance Vector Machine (RVM)** - Sparse Bayesian SVM
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Automatic relevance determination; uncertainty quantification.

- [ ] **Distance Weighted Discrimination (DWD)** - SVM alternative
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Better for high-dimensional data; avoids SVM's data piling.

#### Fuzzy Systems
- [ ] **ANFIS** - Adaptive Neuro-Fuzzy Inference System
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: Hybrid neural network + fuzzy logic.

#### Training Utilities
- [ ] **Stochastic Weight Averaging (SWA)** - Training enhancement
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: Improves generalization for any neural model. Easy to implement.

- [ ] **Mixup / CutMix for Tabular** - Data augmentation
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: Regularization via interpolation for neural tabular models.

- [ ] **Snapshot Ensembles** - Cyclic LR ensembling
  - Priority: 🟢 Low
  - Complexity: Low
  - Notes: Free ensemble from single training run.

#### Time Series Additions
- [ ] **Chronos Integration** - Foundation model for time series
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: Amazon's zero-shot forecasting model. Chronos-2 recently released.

#### Signal Processing Additions
- [ ] EMD (Empirical Mode Decomposition)
- [ ] ICA (Independent Component Analysis for signal)
- [ ] Catch22 feature set
- [ ] HRV (Heart Rate Variability) features

---

### Tier 5: Infrastructure & Production

#### Pipeline & Workflow
- [ ] **Pipeline Templates** - Pre-built competition pipelines
  - Priority: 🟡 Medium
  - Complexity: Low
  - Notes: `eg.templates.TabularClassification()`, `eg.templates.TimeSeriesForecasting()`. One-stop solutions.

- [ ] **Feature Hashing** - High-cardinality handling
  - Priority: 🟢 Low
  - Complexity: Low
  - Notes: Hashing trick for memory efficiency.

- [ ] **Binning Strategies** - Optimal binning (weight of evidence)
  - Priority: 🟡 Medium
  - Complexity: Medium
  - Notes: optbinning integration.

#### Model Compression
- [ ] **Model Pruning Utilities** - Neural network pruning
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Reduce model size post-training.

- [ ] **TensorRT Optimization** - GPU inference optimization
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: `optimize_for_inference()` method.

#### Meta-Learning
- [ ] **Zero-Shot HPO** - TabRepo-style learned portfolios
  - Priority: 🟡 Medium
  - Complexity: High
  - Notes: Pre-computed optimal configs from 200+ datasets.

#### Multi-Target Extensions
- [ ] **Classifier Chains** - Structured multi-label
  - Priority: 🟢 Low
  - Complexity: Low
  - Notes: Exploit label dependencies.

- [ ] **Multi-Task Neural Network** - Shared backbone multi-task
  - Priority: 🟢 Low
  - Complexity: Medium
  - Notes: Joint learning with task-specific heads.

#### Validation Additions
- [ ] **Blocking Time Series Split** - Block bootstrap CV
  - Priority: 🟢 Low
  - Complexity: Low
  - Notes: Alternative to purged TS split.

#### Ensemble Additions
- [ ] **Negative Correlation Learning** - Diversity-promoting ensemble
  - Priority: 🟢 Low
  - Complexity: High
  - Notes: Train diverse ensemble members jointly.

---

## Implementation Notes

### Priority Legend
- 🔴 **Critical/High**: Essential for competitive SOTA library — implement ASAP
- 🟡 **Medium**: Valuable addition, implement when time permits
- 🟢 **Low**: Nice-to-have, consider for future versions

### Complexity Guide
- **Low**: < 200 lines, straightforward wrapper or adaptation
- **Medium**: 200-500 lines, requires careful design
- **High**: 500+ lines, significant research/implementation effort

### Design Principles (All New Code)
1. Sklearn-compatible API (`fit`, `predict`, `predict_proba`)
2. Support `sample_weight` where applicable
3. Implement `feature_importances_` property where meaningful
4. Include preset hyperparameters (`preset='competition'`)
5. Add comprehensive docstrings with examples
6. Write unit tests covering edge cases
7. Polars-first for data transformations

---

## Version Targets

### v0.6.0
- [x] Time Series Module (`eg.timeseries`) ✅
- [x] Signal Processing Module (`eg.signal`) ✅
- [x] Feature Selection Module (`eg.feature_selection`) ✅
- [x] Dimensionality Reduction Module (`eg.dimensionality_reduction`) ✅
- [x] Clustering Module (`eg.clustering`) ✅
- [x] AutoML Module (`eg.automl`) - Full framework ✅
- [x] Interpretable Models (`eg.models.interpretable`) - CORELS, GOSDT, NODE-GAM, GAMI-Net, SLIM, GAM ✅
- [x] Geometric/Generative/LLM Imbalance Methods ✅
- [x] Symbolic Regression ✅

### v0.7.0 ✅
- [x] Interactive Decision Tree Visualization (HTML/JS, collapsible, zoomable) ✅
- [x] Documentation Directory Setup (Sphinx/MkDocs scaffolding) ✅
- [x] TabPFN v2 Integration (Nature 2025 foundation model) ✅
- [x] TabPFN 2.5 / RealTabPFN (50K samples, 2K features) ✅
- [x] TabR (Retrieval-augmented tabular) ✅
- [x] GRANDE (Gradient decision tree ensembles) ✅
- [x] TabM (Parameter-efficient MLP ensembling) ✅
- [x] TabDPT (Discriminative pre-trained transformer) ✅
- [x] RealMLP (Meta-tuned MLP with robust preprocessing) ✅
- [x] xRFM (Tree-structured Recursive Feature Machines) ✅
- [x] Missing Data Module (MICE, MissForest, KNN imputation) ✅
- [x] Target Transformation (Box-Cox, Yeo-Johnson, auto-selection) ✅
- [x] Label Noise Detection (Confident Learning) ✅
- [x] Multi-Output Wrapper (classifier/regressor chains) ✅
- [x] Nested CV ✅
- [x] Knowledge Distillation ✅

### v0.8.0
- [x] Comprehensive Visualization Suite (42 chart types, 6 tiers, shared infrastructure) ✅
- [x] Explainability Module (SHAP, LIME, PDP, interactions, counterfactuals) ✅
- [x] Fairness Module (metrics + mitigation + reports) ✅
- [x] ONNX Export (auto-backend detection + ModelServer) ✅
- [x] Experiment Tracking (MLflow + console logger + AutoML integration) ✅
- [x] Refit on Full Data (`refit_full()` for deployment) ✅
- [x] MultiModal Enhancement (embedding + attention fusion) ✅

### v1.0.0 (Current) ✅
- [x] Version bump and stable release
- [x] All public modules accessible via `eg.<name>` lazy loading
- [x] API stability guarantee
- [x] Comprehensive documentation site
- [x] Tutorial notebooks for each module
- [x] Neuroevolution models (NEAT, TensorNEAT)

### v1.1.0 (Next)
- [ ] Data Quality Module (profiling, drift detection)
- [ ] ExcelFormer, TabICL (remaining SOTA tabular models)
- [ ] Active Learning framework
- [ ] SWA Training Utility
- [ ] Pipeline Templates
- [ ] Survival Analysis (RSF, Cox, DeepSurv)
- [ ] Causal Inference (CausalForest, DoublyRobust)
- [ ] Online/Streaming Learning (River integration)
- [ ] Zero-Shot HPO
- [ ] Published performance benchmarks on TabArena
---

## Test Coverage

| Module | Test File | Status |
|--------|-----------|--------|
| core | test_core.py | ✅ |
| validation | test_validation.py, test_cpcv.py | ✅ |
| preprocessing | test_preprocessing.py | ✅ |
| preprocessing.imbalance | test_imbalance.py | ✅ |
| preprocessing.imbalance_geometric | test_imbalance_geometric.py | ✅ |
| preprocessing.imbalance_generative | test_imbalance_generative.py | ✅ |
| preprocessing.imbalance_llm | test_imbalance_llm.py | ✅ |
| models | test_models.py | ✅ |
| models.trees | test_c50.py, test_oblique_forest.py, test_adtree.py, test_quantile_forest.py | ✅ |
| models.rules | test_furia.py | ✅ |
| models.bayesian | test_bayesian.py | ✅ |
| models.ngboost | test_ngboost.py | ✅ |
| models.tabular.gandalf | test_gandalf.py | ✅ |
| models.tabular.nam | (in test_models.py) | ✅ |
| models.kernel | (in test_models.py) | ✅ |
| models.baselines | (in test_models.py) | ✅ |
| models.ordinal | (in test_models.py) | ✅ |
| models.probabilistic | (in test_models.py) | ✅ |
| models.symbolic | test_symbolic.py | ✅ |
| models.interpretable | test_interpretable.py | ✅ |
| ensemble | test_ensemble.py | ✅ |
| calibration | test_cqr.py | ✅ |
| tune | test_tune.py | ✅ |
| utils | test_utils.py, test_sharpe.py | ✅ |
| vision | test_vision.py | ✅ |
| nlp | test_nlp.py, test_nlp_translation.py | ✅ |
| audio | test_audio.py | ✅ |
| benchmark | test_benchmark.py | ✅ |
| kaggle | test_kaggle.py | ✅ |
| semi_supervised | test_semi_supervised.py | ✅ |
| anomaly | test_anomaly.py | ✅ |
| clustering | test_clustering.py | ✅ |
| feature_selection | test_feature_selection.py | ✅ |
| dimensionality_reduction | test_dimensionality_reduction.py | ✅ |
| automl | test_automl.py | ✅ |
| models.tabular.tabm | test_tabm.py | ✅ |
| models.tabular.tabdpt | test_tabdpt.py | ✅ |
| models.tabular.realmlp | test_realmlp.py | ✅ |
| models.tabular.tabpfn_v2 | test_tabpfn_v2.py | ✅ |
| models.tabular.tabpfn_25 | test_tabpfn_25.py | ✅ |
| models.tabular.xrfm | test_xrfm.py | ✅ |
| models.tabular.grande | test_grande.py | ✅ |
| models.tabular.tabr | test_tabr.py | ✅ |
| preprocessing.imputation | test_imputation.py | ✅ |
| preprocessing.target_transform | test_target_transform.py | ✅ |
| preprocessing.noise_detection | test_noise_detection.py | ✅ |
| ensemble.multi_output | test_multi_output.py | ✅ |
| ensemble.distillation | test_distillation.py | ✅ |
| validation.nested_cv | test_nested_cv.py | ✅ |
| visualization | test_tree_visualization.py | ✅ |
| visualization (suite) | test_visualizations.py | ✅ |
| explain | test_explain.py | ✅ |
| fairness | test_fairness.py | ✅ |
| persistence | test_persistence.py | ✅ |
| tracking | test_tracking.py | ✅ |
| quick | test_quick.py | 🔄 |

---

## Contributing

When implementing a new feature:

1. Create a new file in the appropriate subdirectory
2. Follow the existing code style (see `CLAUDE.md`)
3. Add tests in `tests/test_<module>.py`
4. Update `__init__.py` exports
5. Mark item complete in this roadmap
6. Update `README.md` if user-facing

---

## References

### Key Papers (Implemented)
- NGBoost: Duan et al., 2020 - "NGBoost: Natural Gradient Boosting for Probabilistic Prediction" ✅
- QRF: Meinshausen, 2006 - "Quantile Regression Forests" (JMLR) ✅
- CQR: Romano et al., 2019 - "Conformalized Quantile Regression" (NeurIPS) ✅
- NAM: Agarwal et al., 2021 - "Neural Additive Models" ✅
- GANDALF: Joseph & Raj, 2022 - "GANDALF: Gated Adaptive Network for Deep Automated Learning" ✅
- BART: Chipman et al., 2010 - "BART: Bayesian Additive Regression Trees" ✅
- evtree: Grubinger et al., 2014 - "evtree: Evolutionary Learning of Globally Optimal Trees" ✅
- PySR: Cranmer, 2023 - "Interpretable Machine Learning for Science with PySR" ✅
- FURIA: Hühn & Hüllermeier, 2009 - "FURIA: An Algorithm for Unordered Fuzzy Rule Induction" ✅
- CORELS: Angelino et al., 2017 - "Learning Certifiably Optimal Rule Lists" ✅
- GOSDT: Lin et al., 2020 - "Generalized and Scalable Optimal Sparse Decision Trees" ✅
- PRIM: Friedman & Fisher, 1999 - "Bump Hunting in High-Dimensional Data" ✅
- CPCV: de Prado, 2018 - "Advances in Financial Machine Learning" (Chapter 12) ✅
- DSR: Bailey & de Prado, 2014 - "The Deflated Sharpe Ratio" ✅
- TabPFN v2: Hollmann et al., 2025 - "Accurate predictions on small data with a tabular foundation model" (Nature) ✅
- TabR: Gorishniy et al., 2024 - "TabR: Tabular Deep Learning Meets Nearest Neighbors" ✅
- GRANDE: Marton et al., 2024 - "GRANDE: Gradient-Based Decision Tree Ensembles" ✅
- RealMLP: Holzmüller et al., 2024 - "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees" ✅
- TabM: Gorishniy et al., 2025 - "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling" ✅
- TabDPT: Ma et al., 2024 - "TabDPT: Scaling Tabular Foundation Models" ✅
- xRFM: Beaglehole & Holzmüller, 2025 - "xRFM: Accurate, scalable, and interpretable feature learning models for tabular data" ✅
- Confident Learning: Northcutt et al., 2021 - "Confident Learning: Estimating Uncertainty in Dataset Labels" ✅

### Papers to Implement (Priority Order)
- ExcelFormer: Chen et al., 2023 - "ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
- TabICL: Ye et al., 2025 - "TabICL: A Tabular Foundation Model for In-Context Learning"
- CausalForest: Athey & Imbens, 2019 - "Generalized Random Forests"
- RIPPER: Cohen, 1995 - "Fast Effective Rule Induction"
- RVM: Tipping, 2001 - "Sparse Bayesian Learning and the Relevance Vector Machine"

### Libraries for Reference
- `tabpfn` - TabPFN v2 (PriorLabs)
- `interpret` - EBM, glass-box models
- `cleanlab` - Confident learning, data quality
- `aif360` / `fairlearn` - Fairness metrics and mitigation
- `shap` / `lime` - Model explainability
- `river` - Online learning
- `scikit-survival` - Survival analysis
- `econml` / `dowhy` - Causal inference
- `pytorch-tabular` - Deep tabular models
- `imbalanced-learn` - Resampling methods
- `pyod` - Python Outlier Detection
- `hummingbird-ml` - ONNX conversion for tree models
- `KEEL` (Java) - Instance selection, noise filters, fuzzy systems
