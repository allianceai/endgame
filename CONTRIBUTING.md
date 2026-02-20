# Contributing to Endgame

Thank you for your interest in contributing to Endgame. This guide covers what you need to know to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/allianceai/endgame.git
cd endgame

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, ruff, and mypy. For domain-specific work, install additional extras as needed (e.g., `pip install -e ".[dev,tabular]"`).

## Code Style

We use **ruff** for linting and formatting:

```bash
# Check for lint errors
ruff check endgame/

# Auto-fix what can be fixed
ruff check endgame/ --fix

# Format code
ruff format endgame/
```

Key rules:
- **Line length**: 100 characters
- **Imports**: sorted, one per line for multi-line blocks
- **Docstrings**: Google style with examples
- **Type hints**: encouraged on all public APIs

## Sklearn API Requirements

Every estimator in Endgame must follow the scikit-learn estimator interface. This is non-negotiable.

**Classifiers** must implement:
- `fit(X, y, sample_weight=None)` -- returns `self`
- `predict(X)` -- returns array of predictions
- `predict_proba(X)` -- returns array of shape `(n_samples, n_classes)`
- `score(X, y)` -- returns accuracy (or override with a relevant metric)

**Regressors** must implement:
- `fit(X, y, sample_weight=None)` -- returns `self`
- `predict(X)` -- returns array of predictions

**Transformers** must implement:
- `fit(X, y=None)` -- returns `self`
- `transform(X)` -- returns transformed data
- `fit_transform(X, y=None)` -- inherited from `TransformerMixin`

Additional conventions:
- Inherit from `sklearn.base.BaseEstimator` and the appropriate mixin (`ClassifierMixin`, `RegressorMixin`, `TransformerMixin`)
- Store all constructor parameters as attributes with the same name
- Implement `feature_importances_` property where meaningful
- Support `preset='competition'` for competition-tuned defaults where applicable
- Use `sklearn.utils.validation.check_is_fitted` before predict/transform

## Adding a New Model

1. **Choose the right submodule** under `endgame/models/` (e.g., `trees/`, `bayesian/`, `kernel/`). Create a new submodule if the model does not fit an existing category.

2. **Create the implementation file** (e.g., `endgame/models/trees/my_model.py`):

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class MyModelClassifier(BaseEstimator, ClassifierMixin):
    """One-line description.

    Longer description of the algorithm and when to use it.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of estimators.

    Example
    -------
    >>> from endgame.models.trees import MyModelClassifier
    >>> clf = MyModelClassifier(n_estimators=50)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """

    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # ... fitting logic ...
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        # ... prediction logic ...
        return predictions

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        # ... probability logic ...
        return probabilities
```

3. **Export from `__init__.py`**: Add the class to the appropriate `__init__.py` so it is importable from the submodule.

4. **Write tests** in `tests/test_models.py` (or a new test file if adding a submodule):

```python
def test_my_model_basic():
    from endgame.models.trees import MyModelClassifier
    clf = MyModelClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert hasattr(clf, 'classes_')
```

5. **Handle optional dependencies** gracefully. If the model wraps a library that is not a core dependency, use a lazy import pattern:

```python
def fit(self, X, y, sample_weight=None):
    try:
        import some_optional_lib
    except ImportError:
        raise ImportError(
            "some_optional_lib is required for MyModelClassifier. "
            "Install it with: pip install some_optional_lib"
        )
    # ...
```

## Adding a New Module

1. Create the directory under `endgame/` with an `__init__.py`.
2. Add lazy loading in `endgame/__init__.py` if the module has heavy dependencies.
3. Add the module to the documentation table in `README.md`.
4. Create a corresponding test file `tests/test_<module>.py`.

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py -v

# Run a specific test
pytest tests/test_models.py::test_lgbm_wrapper -v

# Run with coverage
pytest tests/ --cov=endgame
```

All new code must have tests. Aim for coverage of:
- Normal operation with typical inputs
- Edge cases (empty input, single sample, single feature)
- Error handling (invalid parameters, missing dependencies)
- Sklearn compatibility (passes `sklearn.utils.estimator_checks.check_estimator` where feasible)

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main` (e.g., `feat/my-new-model` or `fix/calibration-edge-case`).
2. **Keep PRs focused**: One feature or fix per PR. If a PR touches multiple unrelated areas, split it up.
3. **Write a clear description**: Explain what the PR does and why. Link to any relevant issues.
4. **Ensure CI passes**: All tests must pass and ruff must report no errors.
5. **Respond to review feedback**: Maintainers may request changes. Push follow-up commits to the same branch.

### PR Checklist

Before submitting, verify:
- [ ] Code follows the sklearn API conventions described above
- [ ] Tests are included and pass locally (`pytest tests/ -v`)
- [ ] Linting passes (`ruff check endgame/`)
- [ ] New public classes are exported from `__init__.py`
- [ ] Docstrings are present with parameter descriptions and examples

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests. When reporting bugs, include:
- Python version and OS
- Endgame version (`import endgame; print(endgame.__version__)`)
- Minimal reproduction code
- Full traceback

## License

By contributing to Endgame, you agree that your contributions will be licensed under the Apache License 2.0.
