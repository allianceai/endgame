"""Sphinx configuration for Endgame documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Endgame"
copyright = "2026, Cameron Hamilton"
author = "Cameron Hamilton"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings (Google-style docstrings) -----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = False

# -- Mock optional dependencies for RTD builds -------------------------------
autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "xgboost", "lightgbm", "catboost",
    "interpret", "ngboost",
    "transformers", "tokenizers", "datasets", "bitsandbytes", "peft", "evaluate",
    "timm", "albumentations", "segmentation_models_pytorch",
    "librosa", "soundfile",
    "shap", "lime", "dice_ml",
    "fairlearn",
    "mlflow",
    "onnx", "onnxruntime", "skl2onnx", "hummingbird",
    "statsforecast", "darts", "tsfresh", "sktime",
    "openml", "pymfe",
    "pyod",
    "pgmpy", "causal_learn",
    "pywt",
    "plotly", "matplotlib", "seaborn",
    "mcp",
    "pytorch_tabnet", "pytorch_tabular",
    "pygam", "mord",
    "ctgan",
    "tabpfn", "tabdpt",
    "neat", "tensorneat",
    "pandas",
    "treeple",
    "numba",
    "imbalanced_learn", "imblearn",
    "fasterrisk",
    "sacrebleu",
]

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "polars": ("https://docs.pola.rs/py-polars/html", None),
}

# -- MyST-Parser settings (for .md files) ------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "Endgame"
html_static_path = []
html_css_files = []

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#4e79a7",
        "color-brand-content": "#4e79a7",
    },
    "dark_css_variables": {
        "color-brand-primary": "#76b7b2",
        "color-brand-content": "#76b7b2",
    },
}
