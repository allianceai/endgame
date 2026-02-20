"""Kaggle integration module for competition management.

Provides utilities for downloading competition data, creating submissions,
and managing competition project structures.

Uses kagglehub (modern Kaggle library) for downloads and the legacy kaggle
package for submission functionality.

Installation
------------
pip install kagglehub  # For downloads (recommended)
pip install kaggle     # For submissions

Authentication
--------------
Option 1: Run kagglehub.login() to authenticate interactively
Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
Option 3: Place kaggle.json in ~/.kaggle/

Get your API token from https://www.kaggle.com/settings

Examples
--------
>>> import endgame as eg

>>> # Quick start with a competition
>>> comp = eg.kaggle.Competition("titanic")
>>> comp.download_data()

>>> train = comp.load_train()
>>> test = comp.load_test()

>>> # Make predictions...
>>> predictions = model.predict(test)

>>> # Create and submit
>>> submission = comp.create_submission(predictions)
>>> result = comp.submit(submission, "My first submission")
>>> print(f"Score: {result.public_score}")

>>> # Create full project structure
>>> project = eg.kaggle.CompetitionProject.create("house-prices-advanced-regression-techniques")
>>> # Creates organized folder structure with data downloaded
"""

from endgame.kaggle.client import (
    CompetitionInfo,
    DatasetInfo,
    KaggleClient,
    SubmissionInfo,
    SubmissionResult,
)
from endgame.kaggle.competition import (
    Competition,
    ValidationResult,
)
from endgame.kaggle.project import CompetitionProject

__all__ = [
    # Client
    "KaggleClient",
    # Data classes
    "CompetitionInfo",
    "SubmissionInfo",
    "SubmissionResult",
    "DatasetInfo",
    "ValidationResult",
    # High-level interfaces
    "Competition",
    "CompetitionProject",
]
