"""Tests for the Kaggle integration module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# Test data classes and basic functionality without API calls
from endgame.kaggle.client import (
    CompetitionInfo,
    SubmissionInfo,
    SubmissionResult,
    DatasetInfo,
)
from endgame.kaggle.competition import ValidationResult


class TestDataClasses:
    """Test data class instantiation and methods."""
    
    def test_competition_info_creation(self):
        """Test CompetitionInfo dataclass."""
        info = CompetitionInfo(
            slug="titanic",
            title="Titanic - Machine Learning from Disaster",
            category="Getting Started",
            evaluation_metric="Accuracy",
            team_count=15000,
        )
        
        assert info.slug == "titanic"
        assert info.title == "Titanic - Machine Learning from Disaster"
        assert info.team_count == 15000
        assert "titanic" in str(info).lower()
    
    def test_submission_info_creation(self):
        """Test SubmissionInfo dataclass."""
        from datetime import datetime
        
        info = SubmissionInfo(
            submission_id=12345,
            date=datetime.now(),
            description="Test submission",
            status="complete",
            public_score=0.78,
        )
        
        assert info.submission_id == 12345
        assert info.status == "complete"
        assert info.public_score == 0.78
        assert "12345" in str(info)
    
    def test_submission_result_success(self):
        """Test SubmissionResult for successful submission."""
        result = SubmissionResult(
            success=True,
            message="Submission successful",
            submission_id=123,
            public_score=0.85,
        )
        
        assert result.success
        assert result.public_score == 0.85
        assert "successful" in str(result).lower()
    
    def test_submission_result_failure(self):
        """Test SubmissionResult for failed submission."""
        result = SubmissionResult(
            success=False,
            error="File format invalid",
        )
        
        assert not result.success
        assert "failed" in str(result).lower()
        assert "invalid" in str(result).lower()
    
    def test_dataset_info_creation(self):
        """Test DatasetInfo dataclass."""
        info = DatasetInfo(
            slug="user/my-dataset",
            title="My Dataset",
            size=1024 * 1024,  # 1 MB
            download_count=500,
        )
        
        assert info.slug == "user/my-dataset"
        assert info.size == 1024 * 1024
    
    def test_validation_result_valid(self):
        """Test ValidationResult for valid submission."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=["Contains potential outliers"],
            n_rows=1000,
            expected_rows=1000,
        )
        
        assert result.valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert "Valid" in str(result)
    
    def test_validation_result_invalid(self):
        """Test ValidationResult for invalid submission."""
        result = ValidationResult(
            valid=False,
            errors=["Row count mismatch", "Missing columns"],
            warnings=[],
            n_rows=500,
            expected_rows=1000,
        )
        
        assert not result.valid
        assert len(result.errors) == 2
        assert "Invalid" in str(result)


class TestCompetitionWithoutAPI:
    """Test Competition class functionality without API calls."""
    
    def test_competition_initialization(self):
        """Test Competition initialization."""
        from endgame.kaggle.competition import Competition
        
        comp = Competition("titanic")
        
        assert comp.slug == "titanic"
        assert "titanic" in str(comp.data_dir)
    
    def test_competition_custom_data_dir(self):
        """Test Competition with custom data directory."""
        from endgame.kaggle.competition import Competition
        
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = Competition("titanic", data_dir=tmpdir)
            
            assert comp.data_dir == Path(tmpdir)
            assert comp.raw_dir == Path(tmpdir) / "raw"
    
    def test_competition_repr(self):
        """Test Competition string representation."""
        from endgame.kaggle.competition import Competition
        
        comp = Competition("titanic")
        assert "titanic" in repr(comp)


class TestCompetitionProject:
    """Test CompetitionProject scaffolding."""
    
    def test_project_structure_creation(self):
        """Test that project creates correct folder structure."""
        from endgame.kaggle.project import CompetitionProject
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project without downloading data
            with patch.object(CompetitionProject, '_create_structure') as mock_create:
                project = CompetitionProject("titanic", Path(tmpdir) / "titanic")
                
                assert project.slug == "titanic"
                assert project.root == Path(tmpdir) / "titanic"
                assert project.data_dir == Path(tmpdir) / "titanic" / "data"
                assert project.raw_dir == Path(tmpdir) / "titanic" / "data" / "raw"
                assert project.models_dir == Path(tmpdir) / "titanic" / "models"
    
    def test_project_directory_creation(self):
        """Test actual directory creation."""
        from endgame.kaggle.project import CompetitionProject
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = CompetitionProject("test-comp", Path(tmpdir) / "test-comp")
            project._create_structure()
            
            # Check directories were created
            assert project.data_dir.exists()
            assert project.raw_dir.exists()
            assert project.processed_dir.exists()
            assert project.external_dir.exists()
            assert project.models_dir.exists()
            assert project.submissions_dir.exists()
            assert project.notebooks_dir.exists()
            assert project.src_dir.exists()
            
            # Check files were created
            assert (project.src_dir / "config.py").exists()
            assert (project.root / "README.md").exists()
            assert (project.root / ".gitignore").exists()
    
    def test_config_file_content(self):
        """Test config file has correct content."""
        from endgame.kaggle.project import CompetitionProject
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = CompetitionProject("my-competition", Path(tmpdir) / "my-competition")
            project._create_structure()
            
            config_path = project.src_dir / "config.py"
            with open(config_path) as f:
                content = f.read()
            
            assert 'COMPETITION = "my-competition"' in content
            assert "RANDOM_SEED = 42" in content
            assert "N_FOLDS = 5" in content
    
    def test_notebook_creation(self):
        """Test notebook template creation."""
        from endgame.kaggle.project import CompetitionProject
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = CompetitionProject("titanic", Path(tmpdir) / "titanic")
            project._create_structure()
            
            # Create a blank notebook
            nb_path = project.create_notebook("01_eda", template="eda")
            
            assert nb_path.exists()
            assert nb_path.suffix == ".ipynb"
            
            # Check it's valid JSON
            with open(nb_path) as f:
                nb_content = json.load(f)
            
            assert "cells" in nb_content
            assert nb_content["nbformat"] == 4
    
    def test_project_repr(self):
        """Test project string representation."""
        from endgame.kaggle.project import CompetitionProject
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = CompetitionProject("titanic", Path(tmpdir) / "titanic")
            
            assert "titanic" in repr(project)


class TestValidationResult:
    """Test submission validation functionality."""
    
    def test_validation_columns_default(self):
        """Test ValidationResult columns default to empty list."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
        )
        
        assert result.columns == []
        assert result.expected_columns == []


class TestModuleImports:
    """Test that module imports work correctly."""
    
    def test_import_from_kaggle_module(self):
        """Test imports from endgame.kaggle."""
        from endgame.kaggle import (
            KaggleClient,
            Competition,
            CompetitionProject,
            CompetitionInfo,
            SubmissionInfo,
            SubmissionResult,
            DatasetInfo,
            ValidationResult,
        )
        
        # Check classes exist
        assert KaggleClient is not None
        assert Competition is not None
        assert CompetitionProject is not None
    
    def test_lazy_import_from_endgame(self):
        """Test lazy import of kaggle from endgame."""
        import endgame as eg
        
        # Access kaggle module (lazy import)
        kaggle = eg.kaggle
        
        assert hasattr(kaggle, 'Competition')
        assert hasattr(kaggle, 'KaggleClient')
        assert hasattr(kaggle, 'CompetitionProject')


class TestSubmissionCreation:
    """Test submission file creation without API."""
    
    def test_create_submission_with_mock_sample(self):
        """Test creating a submission file."""
        from endgame.kaggle.competition import Competition
        
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = Competition("test", data_dir=tmpdir)
            comp.raw_dir.mkdir(parents=True, exist_ok=True)
            comp.submissions_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a mock sample_submission.csv
            try:
                import pandas as pd
                sample = pd.DataFrame({
                    'Id': [1, 2, 3, 4, 5],
                    'Prediction': [0.0, 0.0, 0.0, 0.0, 0.0]
                })
                sample.to_csv(comp.raw_dir / "sample_submission.csv", index=False)
                
                # Create predictions
                predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                
                # Create submission
                submission_path = comp.create_submission(predictions, validate=False)
                
                assert submission_path.exists()
                
                # Check content
                sub = pd.read_csv(submission_path)
                assert list(sub.columns) == ['Id', 'Prediction']
                assert len(sub) == 5
                np.testing.assert_array_almost_equal(
                    sub['Prediction'].values,
                    predictions,
                    decimal=5
                )
            except ImportError:
                pytest.skip("pandas not available")
    
    def test_validate_submission_correct(self):
        """Test validation of a correct submission."""
        from endgame.kaggle.competition import Competition
        
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = Competition("test", data_dir=tmpdir)
            comp.raw_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                import pandas as pd
                
                # Create sample submission
                sample = pd.DataFrame({
                    'Id': [1, 2, 3],
                    'Target': [0.0, 0.0, 0.0]
                })
                sample.to_csv(comp.raw_dir / "sample_submission.csv", index=False)
                
                # Create a valid submission
                submission = pd.DataFrame({
                    'Id': [1, 2, 3],
                    'Target': [0.5, 0.6, 0.7]
                })
                sub_path = comp.raw_dir / "test_submission.csv"
                submission.to_csv(sub_path, index=False)
                
                # Validate
                result = comp.validate_submission(sub_path)
                
                assert result.valid
                assert len(result.errors) == 0
                
            except ImportError:
                pytest.skip("pandas not available")
    
    def test_validate_submission_wrong_rows(self):
        """Test validation catches wrong row count."""
        from endgame.kaggle.competition import Competition
        
        with tempfile.TemporaryDirectory() as tmpdir:
            comp = Competition("test", data_dir=tmpdir)
            comp.raw_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                import pandas as pd
                
                # Create sample submission
                sample = pd.DataFrame({
                    'Id': [1, 2, 3],
                    'Target': [0.0, 0.0, 0.0]
                })
                sample.to_csv(comp.raw_dir / "sample_submission.csv", index=False)
                
                # Create invalid submission (wrong row count)
                submission = pd.DataFrame({
                    'Id': [1, 2],
                    'Target': [0.5, 0.6]
                })
                sub_path = comp.raw_dir / "test_submission.csv"
                submission.to_csv(sub_path, index=False)
                
                # Validate
                result = comp.validate_submission(sub_path)
                
                assert not result.valid
                assert any("row" in e.lower() for e in result.errors)
                
            except ImportError:
                pytest.skip("pandas not available")


class TestKaggleClientWithoutAuth:
    """Test KaggleClient functionality that doesn't require authentication."""
    
    def test_client_creation_without_kaggle_package(self):
        """Test client raises helpful error without kaggle packages."""
        from endgame.kaggle import client
        
        # Save original values
        orig_hub = client.HAS_KAGGLEHUB
        orig_legacy = client.HAS_KAGGLE_LEGACY
        
        try:
            # Simulate no packages installed
            client.HAS_KAGGLEHUB = False
            client.HAS_KAGGLE_LEGACY = False
            
            with pytest.raises(ImportError) as exc_info:
                client._ensure_kaggle()
            
            assert "kagglehub" in str(exc_info.value).lower()
            
        finally:
            # Restore
            client.HAS_KAGGLEHUB = orig_hub
            client.HAS_KAGGLE_LEGACY = orig_legacy
