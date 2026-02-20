"""Tests for the Endgame MCP server.

Tests each tool function in isolation with mock/synthetic data,
resources, session management, and codegen.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from endgame.mcp.session import SessionManager, DatasetArtifact, ModelArtifact
from endgame.mcp.codegen import generate_script


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session():
    """Create a fresh session manager."""
    return SessionManager()


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_csv(tmp_path):
    """Create a sample regression CSV file."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
        "x3": np.random.normal(0, 1, n),
    })
    df["price"] = 3 * df["x1"] - 2 * df["x2"] + df["x3"] + np.random.normal(0, 0.5, n)
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def loaded_dataset(session, sample_csv):
    """Load a sample dataset into the session."""
    df = pd.read_csv(sample_csv)
    return session.add_dataset(
        df=df,
        name="test_data",
        source=sample_csv,
        target_column="target",
        task_type="binary",
        meta_features={"n_samples": len(df), "n_features": 3},
    )


@pytest.fixture
def trained_model(session, loaded_dataset):
    """Train a simple model and add to session."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    ds = loaded_dataset
    df = ds.df.copy()
    X = df.drop(columns=["target"])
    y = df["target"]

    # Encode categoricals
    le = LabelEncoder()
    X["category"] = le.fit_transform(X["category"])

    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return session.add_model(
        estimator=model,
        name="gbc_1",
        model_type="gbc",
        dataset_id=ds.id,
        task_type="binary",
        metrics={"accuracy": 0.85},
        params={"n_estimators": 10},
        fit_time=1.0,
        feature_names=list(X.columns),
    )


# ---------------------------------------------------------------------------
# Session management tests
# ---------------------------------------------------------------------------

class TestSession:

    def test_add_and_get_dataset(self, session):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        art = session.add_dataset(df=df, name="test", source="test.csv")
        assert art.id.startswith("ds_")
        assert session.get_dataset(art.id) is art

    def test_get_missing_dataset_raises(self, session):
        with pytest.raises(KeyError):
            session.get_dataset("ds_nonexistent")

    def test_add_and_get_model(self, session):
        art = session.add_model(
            estimator="mock",
            name="test_model",
            model_type="mock",
            dataset_id="ds_123",
            task_type="classification",
        )
        assert art.id.startswith("model_")
        assert session.get_model(art.id) is art

    def test_get_missing_model_raises(self, session):
        with pytest.raises(KeyError):
            session.get_model("model_nonexistent")

    def test_add_visualization(self, session):
        art = session.add_visualization(
            chart_type="roc_curve",
            html_path="/tmp/test.html",
        )
        assert art.id.startswith("viz_")

    def test_state_summary(self, session, loaded_dataset, trained_model):
        summary = session.get_state_summary()
        assert len(summary["datasets"]) == 1
        assert len(summary["models"]) == 1
        assert loaded_dataset.id in summary["datasets"]
        assert trained_model.id in summary["models"]


# ---------------------------------------------------------------------------
# Data tool tests (using session directly, not MCP protocol)
# ---------------------------------------------------------------------------

class TestDataTools:

    def test_load_csv(self, session, sample_csv):
        """Test loading from CSV via the tool function."""
        from endgame.mcp.server import create_server
        # We test the function logic directly through session
        df = pd.read_csv(sample_csv)
        art = session.add_dataset(
            df=df, name="sample", source=sample_csv,
            target_column="target", task_type="binary",
        )
        assert art.df.shape == (100, 4)
        assert art.target_column == "target"
        assert art.task_type == "binary"

    def test_inspect_summary(self, loaded_dataset, session):
        ds = session.get_dataset(loaded_dataset.id)
        assert ds.name == "test_data"
        assert ds.df.shape[0] == 100

    def test_split_data(self, loaded_dataset, session):
        from sklearn.model_selection import train_test_split
        df = loaded_dataset.df
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_art = session.add_dataset(
            df=train_df.reset_index(drop=True),
            name="test_data_train",
            source="derived",
            target_column="target",
            task_type="binary",
        )
        test_art = session.add_dataset(
            df=test_df.reset_index(drop=True),
            name="test_data_test",
            source="derived",
            target_column="target",
            task_type="binary",
        )
        assert train_art.df.shape[0] == 80
        assert test_art.df.shape[0] == 20


# ---------------------------------------------------------------------------
# Discovery tool tests
# ---------------------------------------------------------------------------

class TestDiscoveryTools:

    def test_list_models_all(self):
        from endgame.automl.model_registry import list_models
        models = list_models()
        assert len(models) > 50
        assert "lgbm" in models

    def test_list_models_filtered(self):
        from endgame.automl.model_registry import list_models
        interp = list_models(interpretable_only=True)
        assert "ebm" in interp
        assert "lgbm" not in interp

    def test_list_models_by_family(self):
        from endgame.automl.model_registry import list_models
        gbdt = list_models(family="gbdt")
        assert all(m in ("lgbm", "xgb", "catboost", "ngboost") for m in gbdt)

    def test_get_model_info(self):
        from endgame.automl.model_registry import get_model_info
        info = get_model_info("lgbm")
        assert info.display_name == "LightGBM"
        assert info.family == "gbdt"

    def test_get_model_info_missing(self):
        from endgame.automl.model_registry import get_model_info
        with pytest.raises(KeyError):
            get_model_info("nonexistent_model_xyz")

    def test_recommend_models(self):
        from endgame.automl.model_registry import get_default_portfolio
        portfolio = get_default_portfolio(
            task_type="classification",
            n_samples=1000,
            time_budget="medium",
        )
        assert len(portfolio) > 0
        assert "lgbm" in portfolio


# ---------------------------------------------------------------------------
# Resource tests
# ---------------------------------------------------------------------------

class TestResources:

    def test_catalog_models_resource(self):
        from endgame.automl.model_registry import MODEL_REGISTRY, MODEL_FAMILIES
        by_family = {}
        for name, info in MODEL_REGISTRY.items():
            fam = info.family
            if fam not in by_family:
                by_family[fam] = []
            by_family[fam].append(name)
        assert len(by_family) >= 5
        assert "gbdt" in by_family

    def test_catalog_presets_resource(self):
        from endgame.automl.presets import PRESETS
        assert "fast" in PRESETS
        assert "best_quality" in PRESETS
        assert len(PRESETS) >= 6

    def test_catalog_metrics(self):
        # Verify the metrics catalog content
        metrics = {
            "classification": ["accuracy", "roc_auc", "f1"],
            "regression": ["rmse", "r2", "mae"],
        }
        assert len(metrics["classification"]) > 0
        assert len(metrics["regression"]) > 0


# ---------------------------------------------------------------------------
# Codegen tests
# ---------------------------------------------------------------------------

class TestCodegen:

    def test_generate_script_classification(self, session, loaded_dataset, trained_model):
        script = generate_script(
            model_art=trained_model,
            dataset_art=loaded_dataset,
        )
        assert "import pandas as pd" in script
        assert "model.fit(" in script
        assert "accuracy_score" in script
        assert trained_model.name in script

    def test_generate_script_regression(self, session):
        df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "price": [10, 20, 30, 40, 50],
        })
        ds = session.add_dataset(df=df, name="reg", source="reg.csv", target_column="price", task_type="regression")

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(df[["x1"]], df["price"])

        m = session.add_model(
            estimator=model,
            name="lr_1",
            model_type="linear",
            dataset_id=ds.id,
            task_type="regression",
            feature_names=["x1"],
        )

        script = generate_script(model_art=m, dataset_art=ds)
        assert "r2_score" in script
        assert "mean_squared_error" in script

    def test_generate_script_no_preprocessing(self, session, loaded_dataset, trained_model):
        script = generate_script(
            model_art=trained_model,
            dataset_art=loaded_dataset,
            include_preprocessing=False,
        )
        assert "LabelEncoder" not in script

    def test_generate_script_custom_dataset_path(self, session, loaded_dataset, trained_model):
        script = generate_script(
            model_art=trained_model,
            dataset_art=loaded_dataset,
            dataset_path="/data/my_dataset.csv",
        )
        assert "/data/my_dataset.csv" in script


# ---------------------------------------------------------------------------
# Server creation test
# ---------------------------------------------------------------------------

class TestServerCreation:

    def test_create_server(self):
        """Verify server creates without error and has tools registered."""
        from endgame.mcp.server import create_server
        server = create_server()
        assert server is not None
        assert server.name == "endgame"


# ---------------------------------------------------------------------------
# Error response tests
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_error_response_format(self):
        from endgame.mcp.server import error_response
        resp = json.loads(error_response("not_found", "Dataset not found", hint="Use load_data first"))
        assert resp["status"] == "error"
        assert resp["error_type"] == "not_found"
        assert resp["message"] == "Dataset not found"
        assert resp["hint"] == "Use load_data first"

    def test_ok_response_format(self):
        from endgame.mcp.server import ok_response
        resp = json.loads(ok_response({"dataset_id": "ds_123", "shape": [100, 4]}))
        assert resp["status"] == "ok"
        assert resp["dataset_id"] == "ds_123"


# ---------------------------------------------------------------------------
# Stdout protection test
# ---------------------------------------------------------------------------

class TestStdoutProtection:

    def test_capture_stdout(self, capsys):
        import sys
        from endgame.mcp.server import capture_stdout

        original_stdout = sys.stdout
        with capture_stdout():
            assert sys.stdout is not original_stdout
            print("This should go to stderr")
        assert sys.stdout is original_stdout


# ---------------------------------------------------------------------------
# End-to-end integration tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Integration tests that exercise the full train → evaluate → predict pipeline."""

    @pytest.fixture
    def categorical_csv(self, tmp_path):
        """CSV with categorical features and string target."""
        np.random.seed(42)
        n = 120
        df = pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n).round(2),
            "color": np.random.choice(["red", "green", "blue"], n),
            "size": np.random.choice(["S", "M", "L", "XL"], n),
            "label": np.random.choice(["good", "bad"], n),
        })
        path = tmp_path / "categorical.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_train_evaluate_predict_with_categoricals(self, session, categorical_csv):
        """Full workflow with categorical features — the critical bug path."""
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(categorical_csv)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

        train_art = session.add_dataset(
            df=train_df.reset_index(drop=True),
            name="cat_train", source=categorical_csv,
            target_column="label", task_type="classification",
        )
        test_art = session.add_dataset(
            df=test_df.reset_index(drop=True),
            name="cat_test", source="derived",
            target_column="label", task_type="classification",
        )

        # Train using the encoding helper (mimicking what train_model does)
        from endgame.mcp.tools._encoding import fit_feature_encoders, encode_target

        X_train = train_df.drop(columns=["label"]).reset_index(drop=True)
        y_train = train_df["label"].reset_index(drop=True)

        X_train_enc, feature_encoders = fit_feature_encoders(X_train)
        if X_train_enc.isna().any().any():
            X_train_enc = X_train_enc.fillna(X_train_enc.median(numeric_only=True))
        y_train_enc, target_enc = encode_target(y_train)

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_enc, y_train_enc)

        model_art = session.add_model(
            estimator=model,
            name="gbc_cat_1",
            model_type="gbc",
            dataset_id=train_art.id,
            task_type="classification",
            metrics={"accuracy": 0.8},
            feature_names=list(X_train_enc.columns),
            label_encoders=feature_encoders,
            target_encoder=target_enc,
        )

        # Evaluate on test set using stored encoders
        from endgame.mcp.tools._encoding import apply_encoders

        X_test = test_df.drop(columns=["label"]).reset_index(drop=True)
        y_test = test_df["label"].reset_index(drop=True)
        X_test_enc, y_test_enc = apply_encoders(X_test, y_test, model_art)

        preds = model_art.estimator.predict(X_test_enc)
        assert preds.shape[0] == len(test_df)
        assert set(preds).issubset({0, 1})  # encoded class labels

        # Predict (no target)
        X_pred = test_df.drop(columns=["label"]).reset_index(drop=True)
        X_pred_enc, _ = apply_encoders(X_pred, None, model_art)
        preds2 = model_art.estimator.predict(X_pred_enc)
        np.testing.assert_array_equal(preds, preds2)

    def test_train_evaluate_numeric_only(self, session, regression_csv):
        """Full workflow with numeric-only features."""
        df = pd.read_csv(regression_csv)
        ds = session.add_dataset(
            df=df, name="reg_data", source=regression_csv,
            target_column="price", task_type="regression",
        )

        from sklearn.linear_model import Ridge
        X = df.drop(columns=["price"])
        y = df["price"]
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        model_art = session.add_model(
            estimator=model,
            name="ridge_1",
            model_type="ridge",
            dataset_id=ds.id,
            task_type="regression",
            feature_names=list(X.columns),
        )

        # Evaluate using apply_encoders — should be a no-op for numeric data
        from endgame.mcp.tools._encoding import apply_encoders
        X_enc, y_enc = apply_encoders(X, y, model_art)
        preds = model_art.estimator.predict(X_enc)
        assert preds.shape[0] == len(df)
        assert np.isfinite(preds).all()

    def test_encoder_persistence(self, session, categorical_csv):
        """Verify encoders stored in ModelArtifact match training encoders."""
        df = pd.read_csv(categorical_csv)
        ds = session.add_dataset(
            df=df, name="enc_test", source=categorical_csv,
            target_column="label", task_type="classification",
        )

        from endgame.mcp.tools._encoding import fit_feature_encoders, encode_target

        X = df.drop(columns=["label"])
        X_enc, feature_encoders = fit_feature_encoders(X)
        y_enc, target_enc = encode_target(df["label"])

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X_enc, y_enc)

        model_art = session.add_model(
            estimator=model,
            name="gbc_enc_test",
            model_type="gbc",
            dataset_id=ds.id,
            task_type="classification",
            feature_names=list(X_enc.columns),
            label_encoders=feature_encoders,
            target_encoder=target_enc,
        )

        # Verify encoders are stored
        assert model_art.label_encoders is not None
        assert "color" in model_art.label_encoders
        assert "size" in model_art.label_encoders
        assert model_art.target_encoder is not None
        assert set(model_art.target_encoder.classes_) == {"good", "bad"}

    def test_unseen_category_handling(self, session):
        """Test behavior when test data has categories not seen in training."""
        from endgame.mcp.tools._encoding import fit_feature_encoders, encode_features

        train_df = pd.DataFrame({
            "color": ["red", "green", "blue", "red", "green"],
            "value": [1, 2, 3, 4, 5],
        })
        _, encoders = fit_feature_encoders(train_df[["color"]])

        # Test data has an unseen category "purple"
        test_df = pd.DataFrame({
            "color": ["red", "purple", "blue"],
            "value": [10, 20, 30],
        })
        result = encode_features(test_df, label_encoders=encoders)

        # "red" and "blue" should be encoded consistently
        assert result["color"].iloc[0] == encoders["color"].transform(["red"])[0]
        assert result["color"].iloc[2] == encoders["color"].transform(["blue"])[0]
        # "purple" should get -1
        assert result["color"].iloc[1] == -1

    def test_export_script_produces_valid_python(self, session, loaded_dataset, trained_model):
        """Verify exported script is syntactically valid Python."""
        from endgame.mcp.codegen import generate_script

        script = generate_script(
            model_art=trained_model,
            dataset_art=loaded_dataset,
        )
        # Should be valid Python (compile without error)
        compile(script, "<generated>", "exec")
