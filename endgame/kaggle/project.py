"""Competition project scaffolding and organization.

Creates standardized folder structures for Kaggle competitions.
"""

from datetime import datetime
from pathlib import Path

from endgame.kaggle.client import KaggleClient
from endgame.kaggle.competition import Competition


class CompetitionProject:
    """Create and manage a competition project structure.

    Generates a standardized folder layout for Kaggle competitions with
    organized directories for data, models, submissions, and code.

    Parameters
    ----------
    competition : str
        Competition slug (e.g., 'titanic').
    root_dir : str or Path
        Root directory for the project.
    client : KaggleClient, optional
        Kaggle client instance.

    Attributes
    ----------
    competition : Competition
        Competition instance for data management.
    root : Path
        Root project directory.
    data_dir : Path
        Directory for data files.
    raw_dir : Path
        Directory for raw competition data.
    processed_dir : Path
        Directory for processed/feature-engineered data.
    external_dir : Path
        Directory for external datasets.
    models_dir : Path
        Directory for saved models.
    submissions_dir : Path
        Directory for submission files.
    notebooks_dir : Path
        Directory for Jupyter notebooks.
    src_dir : Path
        Directory for Python source code.

    Examples
    --------
    >>> # Create a new project
    >>> project = CompetitionProject.create("titanic")
    >>> print(project.root)
    titanic/

    >>> # Load training data
    >>> train = project.competition.load_train()

    >>> # Create submission
    >>> project.competition.create_submission(predictions)

    >>> # Access project paths
    >>> print(project.models_dir)
    titanic/models/
    """

    def __init__(
        self,
        competition: str,
        root_dir: str | Path,
        client: KaggleClient | None = None,
    ):
        self.slug = competition
        self.root = Path(root_dir)
        self._client = client

        # Set up directory structure
        self.data_dir = self.root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        self.models_dir = self.root / "models"
        self.submissions_dir = self.root / "submissions"
        self.notebooks_dir = self.root / "notebooks"
        self.src_dir = self.root / "src"

        # Competition instance uses raw_dir for data
        self._competition: Competition | None = None

    @property
    def client(self) -> KaggleClient:
        """Get or create Kaggle client."""
        if self._client is None:
            self._client = KaggleClient()
        return self._client

    @property
    def competition(self) -> Competition:
        """Get Competition instance for data management."""
        if self._competition is None:
            self._competition = Competition(
                slug=self.slug,
                data_dir=self.data_dir,
                client=self.client,
            )
            # Override raw_dir to use our project structure
            self._competition.raw_dir = self.raw_dir
            self._competition.submissions_dir = self.submissions_dir
        return self._competition

    @classmethod
    def create(
        cls,
        competition: str,
        root_dir: str | Path = ".",
        download_data: bool = True,
        client: KaggleClient | None = None,
    ) -> "CompetitionProject":
        """Create a new competition project with full folder structure.

        Parameters
        ----------
        competition : str
            Competition slug (e.g., 'titanic', 'house-prices-advanced-regression-techniques').
        root_dir : str or Path, default='.'
            Parent directory for the project. Project folder will be created inside.
        download_data : bool, default=True
            Download competition data after creating structure.
        client : KaggleClient, optional
            Kaggle client instance.

        Returns
        -------
        CompetitionProject
            Initialized project instance.

        Examples
        --------
        >>> project = CompetitionProject.create("titanic")
        >>> # Creates:
        >>> # titanic/
        >>> # ├── data/
        >>> # │   ├── raw/           # Competition files downloaded here
        >>> # │   ├── processed/     # Your feature-engineered data
        >>> # │   └── external/      # External datasets
        >>> # ├── models/            # Saved model files
        >>> # ├── submissions/       # Submission CSVs
        >>> # ├── notebooks/         # Jupyter notebooks
        >>> # ├── src/
        >>> # │   └── config.py      # Competition config
        >>> # └── README.md
        """
        root_dir = Path(root_dir)
        project_dir = root_dir / competition

        project = cls(competition, project_dir, client)
        project._create_structure()

        if download_data:
            try:
                project.competition.download_data()
            except Exception as e:
                print(f"Warning: Could not download data: {e}")
                print("You may need to accept the competition rules first.")

        return project

    @classmethod
    def load(
        cls,
        project_dir: str | Path,
        client: KaggleClient | None = None,
    ) -> "CompetitionProject":
        """Load an existing competition project.

        Parameters
        ----------
        project_dir : str or Path
            Path to existing project directory.
        client : KaggleClient, optional
            Kaggle client instance.

        Returns
        -------
        CompetitionProject
            Loaded project instance.
        """
        project_dir = Path(project_dir)

        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")

        # Try to get competition slug from directory name or config
        slug = project_dir.name
        config_file = project_dir / "src" / "config.py"

        if config_file.exists():
            # Try to read slug from config
            try:
                with open(config_file) as f:
                    content = f.read()
                    import re
                    match = re.search(r'COMPETITION\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        slug = match.group(1)
            except Exception:
                pass

        return cls(slug, project_dir, client)

    def _create_structure(self) -> None:
        """Create the project directory structure."""
        # Create all directories
        directories = [
            self.root,
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.external_dir,
            self.models_dir,
            self.submissions_dir,
            self.notebooks_dir,
            self.src_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create config file
        self._create_config()

        # Create README
        self._create_readme()

        # Create .gitignore
        self._create_gitignore()

        # Create __init__.py in src
        (self.src_dir / "__init__.py").touch()

        print(f"Created project structure at: {self.root}")

    def _create_config(self) -> None:
        """Create competition config file."""
        config_content = f'''"""Competition configuration for {self.slug}."""

# Competition settings
COMPETITION = "{self.slug}"
COMPETITION_URL = "https://www.kaggle.com/c/{self.slug}"

# Paths (relative to project root)
DATA_DIR = "data"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
EXTERNAL_DIR = "data/external"
MODELS_DIR = "models"
SUBMISSIONS_DIR = "submissions"

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation settings
N_FOLDS = 5
STRATIFIED = True

# Model settings (customize as needed)
MODEL_PARAMS = {{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}}

# Feature settings
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = []
TARGET_COLUMN = "target"  # Update this
ID_COLUMN = "id"  # Update this
'''

        config_path = self.src_dir / "config.py"
        with open(config_path, 'w') as f:
            f.write(config_content)

    def _create_readme(self) -> None:
        """Create README file with competition info."""
        # Try to get competition info
        try:
            info = self.client.get_competition(self.slug)
            title = info.title
            description = info.description
            metric = info.evaluation_metric
            deadline = info.deadline.strftime("%Y-%m-%d") if info.deadline else "N/A"
            url = info.url
        except Exception:
            title = self.slug.replace("-", " ").title()
            description = ""
            metric = ""
            deadline = "N/A"
            url = f"https://www.kaggle.com/c/{self.slug}"

        readme_content = f'''# {title}

Kaggle Competition: [{self.slug}]({url})

## Overview

{description}

- **Evaluation Metric**: {metric}
- **Deadline**: {deadline}

## Project Structure

```
{self.slug}/
├── data/
│   ├── raw/           # Original competition data
│   ├── processed/     # Feature-engineered data
│   └── external/      # External datasets
├── models/            # Saved model files
├── submissions/       # Submission CSV files
├── notebooks/         # Jupyter notebooks
├── src/
│   ├── __init__.py
│   └── config.py      # Competition configuration
└── README.md
```

## Getting Started

```python
import endgame as eg

# Load the competition
comp = eg.kaggle.Competition("{self.slug}", data_dir="data")

# Load data
train = comp.load_train()
test = comp.load_test()

# Train your model...
# predictions = model.predict(test)

# Create and submit
submission = comp.create_submission(predictions)
result = comp.submit(submission, "My submission message")
print(f"Score: {{result.public_score}}")
```

## Notebooks

- `01_eda.ipynb` - Exploratory Data Analysis
- `02_baseline.ipynb` - Baseline model
- `03_feature_engineering.ipynb` - Feature engineering experiments
- `04_modeling.ipynb` - Model training and tuning
- `05_ensemble.ipynb` - Ensemble methods

## Submission Log

| Date | Score | Description |
|------|-------|-------------|
| | | |

---
Created with [endgame](https://github.com/username/endgame) on {datetime.now().strftime("%Y-%m-%d")}
'''

        readme_path = self.root / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

    def _create_gitignore(self) -> None:
        """Create .gitignore file."""
        gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Environment
.env
.venv
env/
venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data files (large files, download from Kaggle)
data/raw/*.csv
data/raw/*.zip
data/raw/*.parquet
data/external/

# Model files (large, regenerate from training)
models/*.pkl
models/*.joblib
models/*.pt
models/*.h5
models/*.onnx

# Kaggle
kaggle.json

# OS
.DS_Store
Thumbs.db

# Logs
*.log
'''

        gitignore_path = self.root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)

    def create_notebook(
        self,
        name: str,
        template: str = "blank",
    ) -> Path:
        """Create a new Jupyter notebook from template.

        Parameters
        ----------
        name : str
            Notebook name (without .ipynb extension).
        template : str, default='blank'
            Template type: 'blank', 'eda', 'modeling', 'ensemble'.

        Returns
        -------
        Path
            Path to created notebook.
        """
        import json

        templates = {
            "blank": self._blank_notebook_template(),
            "eda": self._eda_notebook_template(),
            "modeling": self._modeling_notebook_template(),
            "ensemble": self._ensemble_notebook_template(),
        }

        if template not in templates:
            template = "blank"

        notebook_content = templates[template]

        notebook_path = self.notebooks_dir / f"{name}.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)

        return notebook_path

    def _notebook_base(self) -> dict:
        """Base notebook structure."""
        return {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

    def _code_cell(self, source: str) -> dict:
        """Create a code cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().split("\n")
        }

    def _markdown_cell(self, source: str) -> dict:
        """Create a markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().split("\n")
        }

    def _blank_notebook_template(self) -> dict:
        """Blank notebook template."""
        nb = self._notebook_base()
        nb["cells"] = [
            self._markdown_cell(f"# {self.slug.replace('-', ' ').title()}"),
            self._code_cell("""import numpy as np
import pandas as pd
import endgame as eg

# Load competition
comp = eg.kaggle.Competition("{slug}", data_dir="../data")

# Load data
train = comp.load_train()
test = comp.load_test()

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")""".replace("{slug}", self.slug)),
        ]
        return nb

    def _eda_notebook_template(self) -> dict:
        """EDA notebook template."""
        nb = self._notebook_base()
        nb["cells"] = [
            self._markdown_cell(f"# EDA: {self.slug.replace('-', ' ').title()}"),
            self._code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import endgame as eg

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')

# Load data
comp = eg.kaggle.Competition("{slug}", data_dir="../data")
train = comp.load_train()
test = comp.load_test()""".replace("{slug}", self.slug)),
            self._markdown_cell("## Data Overview"),
            self._code_cell("""print("Train shape:", train.shape)
print("Test shape:", test.shape)
print()
print("Columns:", train.columns.tolist())"""),
            self._code_cell("train.head()"),
            self._code_cell("train.info()"),
            self._code_cell("train.describe()"),
            self._markdown_cell("## Missing Values"),
            self._code_cell("""missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values:")
print(missing)"""),
            self._markdown_cell("## Target Distribution"),
            self._code_cell("""# Update 'target' with your actual target column
# target_col = 'target'
# train[target_col].value_counts().plot(kind='bar')
# plt.title('Target Distribution')
# plt.show()"""),
            self._markdown_cell("## Feature Distributions"),
            self._code_cell("""# Numeric features
numeric_cols = train.select_dtypes(include=[np.number]).columns[:10]
train[numeric_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()"""),
            self._markdown_cell("## Correlations"),
            self._code_cell("""numeric_train = train.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_train.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.show()"""),
        ]
        return nb

    def _modeling_notebook_template(self) -> dict:
        """Modeling notebook template."""
        nb = self._notebook_base()
        nb["cells"] = [
            self._markdown_cell(f"# Modeling: {self.slug.replace('-', ' ').title()}"),
            self._code_cell("""import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import endgame as eg

# Load data
comp = eg.kaggle.Competition("{slug}", data_dir="../data")
train = comp.load_train()
test = comp.load_test()

# TODO: Update these columns
TARGET = 'target'
ID_COL = 'id'

X = train.drop(columns=[TARGET, ID_COL], errors='ignore')
y = train[TARGET]
X_test = test.drop(columns=[ID_COL], errors='ignore')""".replace("{slug}", self.slug)),
            self._markdown_cell("## Preprocessing"),
            self._code_cell("""# Simple preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])"""),
            self._markdown_cell("## Model Training"),
            self._code_cell("""# LightGBM model
model = eg.models.LGBMWrapper(preset='endgame')

# Cross-validation
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")"""),
            self._markdown_cell("## Generate Submission"),
            self._code_cell("""# Train on full data
model.fit(X, y)

# Predict
predictions = model.predict_proba(X_test)[:, 1]

# Create submission
submission_path = comp.create_submission(predictions)
print(f"Submission saved to: {submission_path}")"""),
        ]
        return nb

    def _ensemble_notebook_template(self) -> dict:
        """Ensemble notebook template."""
        nb = self._notebook_base()
        nb["cells"] = [
            self._markdown_cell(f"# Ensemble: {self.slug.replace('-', ' ').title()}"),
            self._code_cell("""import numpy as np
import pandas as pd
import endgame as eg

# Load data
comp = eg.kaggle.Competition("{slug}", data_dir="../data")
train = comp.load_train()
test = comp.load_test()

# TODO: Load your OOF predictions
# oof_lgbm = np.load('../models/oof_lgbm.npy')
# oof_xgb = np.load('../models/oof_xgb.npy')
# oof_catboost = np.load('../models/oof_catboost.npy')""".replace("{slug}", self.slug)),
            self._markdown_cell("## Hill Climbing Ensemble"),
            self._code_cell("""# Example: Hill climbing to find optimal weights
# from endgame.ensemble import HillClimbingEnsemble

# oof_preds = np.column_stack([oof_lgbm, oof_xgb, oof_catboost])
# y = train['target'].values

# ensemble = HillClimbingEnsemble(metric='roc_auc')
# ensemble.fit(oof_preds, y)

# print("Weights:", ensemble.weights_)
# print("Best score:", ensemble.best_score_)"""),
            self._markdown_cell("## Generate Ensemble Submission"),
            self._code_cell("""# Load test predictions
# test_lgbm = np.load('../models/test_lgbm.npy')
# test_xgb = np.load('../models/test_xgb.npy')
# test_catboost = np.load('../models/test_catboost.npy')

# test_preds = np.column_stack([test_lgbm, test_xgb, test_catboost])
# final_preds = ensemble.predict(test_preds)

# submission_path = comp.create_submission(final_preds)
# print(f"Ensemble submission saved to: {submission_path}")"""),
        ]
        return nb

    def __repr__(self) -> str:
        return f"CompetitionProject('{self.slug}', root='{self.root}')"
