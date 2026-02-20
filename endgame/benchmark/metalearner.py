"""Meta-learning for automatic model selection.

Uses benchmark results and meta-features to predict optimal models/pipelines
for new datasets.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from endgame.benchmark.profiler import MetaFeatureSet, MetaProfiler
from endgame.benchmark.tracker import ExperimentTracker

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@dataclass
class ModelRecommendation:
    """Recommendation for a model/pipeline.

    Attributes
    ----------
    model_name : str
        Recommended model name.
    confidence : float
        Confidence score (0-1).
    predicted_score : float
        Predicted performance score.
    reasoning : str
        Explanation for the recommendation.
    alternatives : List[Tuple[str, float]]
        Alternative models with their scores.
    similar_datasets : List[str]
        Most similar datasets from training.
    """
    model_name: str
    confidence: float = 0.0
    predicted_score: float = 0.0
    reasoning: str = ""
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    similar_datasets: list[str] = field(default_factory=list)


class MetaLearner:
    """Learn to predict optimal models from dataset meta-features.

    Trains a meta-model that predicts which model will perform best
    on a new dataset based on its meta-features.

    Parameters
    ----------
    approach : str, default="ranking"
        Meta-learning approach:
        - "ranking": Predict model rankings
        - "classification": Predict best model (classification)
        - "regression": Predict model scores (regression)
    base_estimator : BaseEstimator, optional
        Base model for meta-learning. If None, uses RandomForest.
    metric : str, default="accuracy"
        Target metric to optimize.
    n_top_models : int, default=3
        Number of top models to consider for recommendations.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Train meta-learner from benchmark results
    >>> meta_learner = MetaLearner()
    >>> meta_learner.fit(tracker)
    >>>
    >>> # Get recommendation for new dataset
    >>> recommendation = meta_learner.recommend(X_new, y_new)
    >>> print(f"Best model: {recommendation.model_name}")
    """

    def __init__(
        self,
        approach: str = "ranking",
        base_estimator: BaseEstimator | None = None,
        metric: str = "accuracy",
        n_top_models: int = 3,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.approach = approach
        self.base_estimator = base_estimator
        self.metric = metric
        self.n_top_models = n_top_models
        self.random_state = random_state
        self.verbose = verbose

        self._meta_model = None
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._feature_names: list[str] = []
        self._model_names: list[str] = []
        self._profiler = MetaProfiler(
            groups=["simple", "statistical"],
            random_state=random_state,
            verbose=False,
        )

        # Training data storage
        self._X_meta: np.ndarray | None = None
        self._y_meta: np.ndarray | None = None
        self._dataset_features: dict[str, np.ndarray] = {}
        self._is_fitted = False

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[MetaLearner] {message}")

    def fit(
        self,
        tracker: ExperimentTracker,
        metric: str | None = None,
    ) -> "MetaLearner":
        """Fit meta-learner from benchmark results.

        Parameters
        ----------
        tracker : ExperimentTracker
            Tracker containing benchmark results.
        metric : str, optional
            Override target metric.

        Returns
        -------
        self
        """
        metric = metric or self.metric
        metric_col = f"metric_{metric}"

        df = tracker.to_dataframe()

        # Get successful experiments
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            df = df.filter(pl.col("status") == "success")

            # Get unique models and datasets
            model_names = df["model_name"].unique().to_list()
            dataset_names = df["dataset_name"].unique().to_list()

            # Get meta-feature columns
            mf_cols = sorted([c for c in df.columns if c.startswith("mf_")])
        else:
            df = df[df["status"] == "success"]

            model_names = df["model_name"].unique().tolist()
            dataset_names = df["dataset_name"].unique().tolist()

            mf_cols = sorted([c for c in df.columns if c.startswith("mf_")])

        if not mf_cols:
            raise ValueError("No meta-features found in tracker. Enable profile_datasets=True in BenchmarkRunner.")

        self._model_names = model_names
        self._feature_names = [c[3:] for c in mf_cols]  # Remove "mf_" prefix

        self._log(f"Building meta-learning dataset from {len(dataset_names)} datasets, {len(model_names)} models")
        self._log(f"Meta-features: {len(mf_cols)}")

        # Build training data based on approach
        if self.approach == "classification":
            X_meta, y_meta = self._build_classification_data(df, mf_cols, metric_col)
        elif self.approach == "regression":
            X_meta, y_meta = self._build_regression_data(df, mf_cols, metric_col)
        else:  # ranking
            X_meta, y_meta = self._build_ranking_data(df, mf_cols, metric_col)

        if len(X_meta) == 0:
            raise ValueError("No valid training samples")

        # Scale features
        X_meta = np.nan_to_num(X_meta, nan=0.0)
        X_scaled = self._scaler.fit_transform(X_meta)

        self._X_meta = X_scaled
        self._y_meta = y_meta

        # Store dataset features for similarity computation
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            for dataset_name in dataset_names:
                ds_row = df.filter(pl.col("dataset_name") == dataset_name).head(1)
                if len(ds_row) > 0:
                    features = np.array([
                        float(ds_row[col][0]) if ds_row[col][0] is not None else 0.0
                        for col in mf_cols
                    ])
                    self._dataset_features[dataset_name] = features
        else:
            for dataset_name in dataset_names:
                ds_row = df[df["dataset_name"] == dataset_name].iloc[0]
                features = np.array([
                    float(ds_row[col]) if ds_row[col] is not None and not np.isnan(ds_row[col]) else 0.0
                    for col in mf_cols
                ])
                self._dataset_features[dataset_name] = features

        # Create and train meta-model
        if self.base_estimator is not None:
            self._meta_model = self.base_estimator
        else:
            if self.approach == "classification":
                self._meta_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                self._meta_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

        self._log(f"Training meta-model ({type(self._meta_model).__name__})...")
        self._meta_model.fit(X_scaled, y_meta)

        # Compute CV score of meta-model
        if len(X_scaled) >= 5:
            cv_scores = cross_val_score(
                self._meta_model,
                X_scaled,
                y_meta,
                cv=min(5, len(X_scaled)),
            )
            self._log(f"Meta-model CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        self._is_fitted = True
        return self

    def _build_classification_data(
        self,
        df,
        mf_cols: list[str],
        metric_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build data for classification approach (predict best model)."""
        X_list = []
        y_list = []

        if HAS_POLARS and isinstance(df, pl.DataFrame):
            dataset_names = df["dataset_name"].unique().to_list()

            for dataset_name in dataset_names:
                ds_df = df.filter(pl.col("dataset_name") == dataset_name)

                if len(ds_df) == 0:
                    continue

                # Get meta-features
                features = np.array([
                    float(ds_df[col][0]) if ds_df[col][0] is not None else 0.0
                    for col in mf_cols
                ])

                # Find best model
                scores = ds_df[metric_col].to_numpy()
                models = ds_df["model_name"].to_list()

                valid_mask = ~np.isnan(scores)
                if not np.any(valid_mask):
                    continue

                best_idx = np.nanargmax(scores)
                best_model = models[best_idx]

                X_list.append(features)
                y_list.append(best_model)
        else:
            dataset_names = df["dataset_name"].unique().tolist()

            for dataset_name in dataset_names:
                ds_df = df[df["dataset_name"] == dataset_name]

                if len(ds_df) == 0:
                    continue

                # Get meta-features
                features = np.array([
                    float(ds_df[col].iloc[0]) if not np.isnan(ds_df[col].iloc[0]) else 0.0
                    for col in mf_cols
                ])

                # Find best model
                scores = ds_df[metric_col].values
                models = ds_df["model_name"].tolist()

                valid_mask = ~np.isnan(scores)
                if not np.any(valid_mask):
                    continue

                best_idx = np.nanargmax(scores)
                best_model = models[best_idx]

                X_list.append(features)
                y_list.append(best_model)

        X = np.array(X_list)
        y = self._label_encoder.fit_transform(y_list)

        return X, y

    def _build_regression_data(
        self,
        df,
        mf_cols: list[str],
        metric_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build data for regression approach (predict score per model)."""
        X_list = []
        y_list = []

        # For each (dataset, model) pair, predict the score
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            for row in df.iter_rows(named=True):
                features = np.array([
                    float(row[col]) if row[col] is not None else 0.0
                    for col in mf_cols
                ])

                # Add model indicator (one-hot encoded)
                model_idx = self._model_names.index(row["model_name"])
                model_indicator = np.zeros(len(self._model_names))
                model_indicator[model_idx] = 1

                X_list.append(np.concatenate([features, model_indicator]))

                score = row[metric_col]
                y_list.append(score if score is not None else 0.0)
        else:
            for _, row in df.iterrows():
                features = np.array([
                    float(row[col]) if not np.isnan(row[col]) else 0.0
                    for col in mf_cols
                ])

                model_idx = self._model_names.index(row["model_name"])
                model_indicator = np.zeros(len(self._model_names))
                model_indicator[model_idx] = 1

                X_list.append(np.concatenate([features, model_indicator]))

                score = row[metric_col]
                y_list.append(score if not np.isnan(score) else 0.0)

        return np.array(X_list), np.array(y_list)

    def _build_ranking_data(
        self,
        df,
        mf_cols: list[str],
        metric_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build data for ranking approach (predict relative ranks)."""
        # Same as classification but with rank as target
        X_list = []
        y_list = []

        if HAS_POLARS and isinstance(df, pl.DataFrame):
            dataset_names = df["dataset_name"].unique().to_list()

            for dataset_name in dataset_names:
                ds_df = df.filter(pl.col("dataset_name") == dataset_name)

                if len(ds_df) == 0:
                    continue

                features = np.array([
                    float(ds_df[col][0]) if ds_df[col][0] is not None else 0.0
                    for col in mf_cols
                ])

                scores = ds_df[metric_col].to_numpy()
                models = ds_df["model_name"].to_list()

                # Compute ranks (1 = best)
                valid_mask = ~np.isnan(scores)
                if not np.any(valid_mask):
                    continue

                # Get best model (rank 1)
                best_idx = np.nanargmax(scores)
                best_model = models[best_idx]

                X_list.append(features)
                y_list.append(best_model)
        else:
            dataset_names = df["dataset_name"].unique().tolist()

            for dataset_name in dataset_names:
                ds_df = df[df["dataset_name"] == dataset_name]

                if len(ds_df) == 0:
                    continue

                features = np.array([
                    float(ds_df[col].iloc[0]) if not np.isnan(ds_df[col].iloc[0]) else 0.0
                    for col in mf_cols
                ])

                scores = ds_df[metric_col].values
                models = ds_df["model_name"].tolist()

                valid_mask = ~np.isnan(scores)
                if not np.any(valid_mask):
                    continue

                best_idx = np.nanargmax(scores)
                best_model = models[best_idx]

                X_list.append(features)
                y_list.append(best_model)

        X = np.array(X_list)
        y = self._label_encoder.fit_transform(y_list)

        return X, y

    def recommend(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indicator: list[bool] | None = None,
        task_type: str = "classification",
    ) -> ModelRecommendation:
        """Get model recommendation for a new dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target variable.
        categorical_indicator : List[bool], optional
            Boolean mask for categorical features.
        task_type : str, default="classification"
            Task type: "classification" or "regression".

        Returns
        -------
        ModelRecommendation
            Recommended model with confidence and alternatives.
        """
        if not self._is_fitted:
            raise RuntimeError("MetaLearner must be fitted before calling recommend()")

        # Extract meta-features
        meta_features = self._profiler.profile(
            X, y,
            categorical_indicator=categorical_indicator,
            task_type=task_type,
        )

        return self.recommend_from_features(meta_features)

    def recommend_from_features(
        self,
        meta_features: MetaFeatureSet | dict[str, float],
    ) -> ModelRecommendation:
        """Get recommendation from pre-computed meta-features.

        Parameters
        ----------
        meta_features : MetaFeatureSet or Dict
            Pre-computed meta-features.

        Returns
        -------
        ModelRecommendation
            Recommended model.
        """
        if not self._is_fitted:
            raise RuntimeError("MetaLearner must be fitted before calling recommend_from_features()")

        if isinstance(meta_features, MetaFeatureSet):
            features_dict = meta_features.to_dict()
        else:
            features_dict = meta_features

        # Build feature vector
        X = np.array([features_dict.get(f, 0.0) for f in self._feature_names])
        X = np.nan_to_num(X, nan=0.0).reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        # Get prediction
        if self.approach == "classification" or self.approach == "ranking":
            # Predict probabilities for each model
            if hasattr(self._meta_model, "predict_proba"):
                probs = self._meta_model.predict_proba(X_scaled)[0]
            else:
                # Use decision function or default
                pred = self._meta_model.predict(X_scaled)[0]
                probs = np.zeros(len(self._label_encoder.classes_))
                # Convert predicted class to index
                pred_idx = int(pred) if isinstance(pred, (int, np.integer)) else 0
                if pred_idx < len(probs):
                    probs[pred_idx] = 1.0

            # Get top models
            top_indices = np.argsort(probs)[::-1][:self.n_top_models]

            best_idx = top_indices[0]
            best_model = self._label_encoder.inverse_transform([best_idx])[0]
            confidence = float(probs[best_idx])

            alternatives = [
                (self._label_encoder.inverse_transform([idx])[0], float(probs[idx]))
                for idx in top_indices[1:]
            ]
        else:
            # Regression approach: predict score for each model
            predictions = []

            for i, model_name in enumerate(self._model_names):
                model_indicator = np.zeros(len(self._model_names))
                model_indicator[i] = 1
                X_with_model = np.concatenate([X_scaled[0], model_indicator]).reshape(1, -1)
                pred = self._meta_model.predict(X_with_model)[0]
                predictions.append((model_name, pred))

            # Sort by predicted score
            predictions.sort(key=lambda x: x[1], reverse=True)

            best_model = predictions[0][0]
            confidence = 1.0 / (1 + len(self._model_names))  # Lower confidence for regression

            alternatives = predictions[1:self.n_top_models]

        # Find similar datasets
        similar = self._find_similar_datasets(X[0], n=3)

        # Build reasoning
        reasoning = self._build_reasoning(
            best_model,
            confidence,
            features_dict,
            similar,
        )

        return ModelRecommendation(
            model_name=best_model,
            confidence=confidence,
            predicted_score=confidence,  # Approximate
            reasoning=reasoning,
            alternatives=alternatives,
            similar_datasets=[s[0] for s in similar],
        )

    def _find_similar_datasets(
        self,
        features: np.ndarray,
        n: int = 3,
    ) -> list[tuple[str, float]]:
        """Find most similar datasets from training data."""
        features = np.nan_to_num(features, nan=0.0)
        features_scaled = self._scaler.transform(features.reshape(1, -1))[0]

        similarities = []

        for dataset_name, ds_features in self._dataset_features.items():
            ds_features_scaled = self._scaler.transform(ds_features.reshape(1, -1))[0]

            # Cosine similarity
            norm1 = np.linalg.norm(features_scaled)
            norm2 = np.linalg.norm(ds_features_scaled)

            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(features_scaled, ds_features_scaled) / (norm1 * norm2)
            else:
                similarity = 0.0

            similarities.append((dataset_name, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:n]

    def _build_reasoning(
        self,
        model_name: str,
        confidence: float,
        features: dict[str, float],
        similar: list[tuple[str, float]],
    ) -> str:
        """Build human-readable reasoning for recommendation."""
        lines = [f"Recommended: {model_name} (confidence: {confidence:.2f})"]

        # Dataset characteristics
        n_samples = features.get("nr_inst", 0)
        n_features = features.get("nr_attr", 0)

        lines.append(f"Dataset: {int(n_samples)} samples, {int(n_features)} features")

        # Similar datasets
        if similar:
            similar_str = ", ".join([f"{name} ({sim:.2f})" for name, sim in similar[:2]])
            lines.append(f"Similar to: {similar_str}")

        return " | ".join(lines)

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importances from meta-model.

        Returns
        -------
        Dict[str, float]
            Feature name to importance mapping.
        """
        if not self._is_fitted:
            raise RuntimeError("MetaLearner must be fitted first")

        if hasattr(self._meta_model, "feature_importances_"):
            importances = self._meta_model.feature_importances_

            # Handle regression approach (includes model indicators)
            if len(importances) > len(self._feature_names):
                importances = importances[:len(self._feature_names)]

            return dict(sorted(
                zip(self._feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            ))

        return {}


class PipelineRecommender:
    """Recommend complete pipelines (preprocessing + model) for new datasets.

    Extends MetaLearner to recommend full preprocessing pipelines
    in addition to models.

    Parameters
    ----------
    meta_learner : MetaLearner, optional
        Pre-trained meta-learner.
    preprocessing_options : List[str], default=["none", "scaling", "imputation"]
        Available preprocessing options.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> recommender = PipelineRecommender()
    >>> recommender.fit(tracker)
    >>> pipeline = recommender.recommend_pipeline(X, y)
    >>> print(pipeline)
    """

    def __init__(
        self,
        meta_learner: MetaLearner | None = None,
        preprocessing_options: list[str] | None = None,
        verbose: bool = False,
    ):
        self.meta_learner = meta_learner or MetaLearner(verbose=verbose)
        self.preprocessing_options = preprocessing_options or [
            "none",
            "standard_scaling",
            "robust_scaling",
            "imputation",
            "imputation+scaling",
        ]
        self.verbose = verbose

    def fit(
        self,
        tracker: ExperimentTracker,
        **kwargs,
    ) -> "PipelineRecommender":
        """Fit recommender from benchmark results."""
        self.meta_learner.fit(tracker, **kwargs)
        return self

    def recommend_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indicator: list[bool] | None = None,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """Recommend a complete pipeline.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target variable.
        categorical_indicator : List[bool], optional
            Boolean mask for categorical features.
        task_type : str
            Task type.

        Returns
        -------
        Dict[str, Any]
            Pipeline recommendation with model and preprocessing.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler, StandardScaler

        # Get model recommendation
        model_rec = self.meta_learner.recommend(
            X, y,
            categorical_indicator=categorical_indicator,
            task_type=task_type,
        )

        # Analyze data characteristics
        has_missing = np.any(np.isnan(X))
        has_outliers = self._detect_outliers(X)

        # Choose preprocessing
        preprocessing_steps = []

        if has_missing:
            preprocessing_steps.append(("imputer", SimpleImputer(strategy="median")))

        if has_outliers:
            preprocessing_steps.append(("scaler", RobustScaler()))
        else:
            preprocessing_steps.append(("scaler", StandardScaler()))

        return {
            "model_name": model_rec.model_name,
            "model_confidence": model_rec.confidence,
            "preprocessing": preprocessing_steps,
            "alternatives": model_rec.alternatives,
            "reasoning": model_rec.reasoning,
            "has_missing": has_missing,
            "has_outliers": has_outliers,
        }

    def _detect_outliers(self, X: np.ndarray, threshold: float = 0.1) -> bool:
        """Detect if dataset has significant outliers."""
        X_clean = np.nan_to_num(X, nan=0.0)

        outlier_count = 0
        total = X_clean.shape[0] * X_clean.shape[1]

        for col in range(X_clean.shape[1]):
            q1, q3 = np.percentile(X_clean[:, col], [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count += np.sum((X_clean[:, col] < lower) | (X_clean[:, col] > upper))

        return (outlier_count / total) > threshold
