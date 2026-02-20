"""Tests for Nested Cross-Validation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from endgame.validation import NestedCV, NestedCVResult


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return X, y


class TestNestedCV:
    def test_basic_classifier(self, clf_data):
        X, y = clf_data
        ncv = NestedCV(
            estimator=DecisionTreeClassifier(random_state=42),
            outer_cv=3,
        )
        result = ncv.evaluate(X, y)
        assert isinstance(result, NestedCVResult)
        assert len(result.outer_scores) == 3
        assert 0.0 <= result.mean_score <= 1.0
        assert result.std_score >= 0

    def test_basic_regressor(self, reg_data):
        X, y = reg_data
        ncv = NestedCV(
            estimator=DecisionTreeRegressor(random_state=42),
            outer_cv=3,
            scoring='r2',
        )
        result = ncv.evaluate(X, y)
        assert len(result.outer_scores) == 3

    def test_with_search(self, clf_data):
        X, y = clf_data
        search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid={'max_depth': [2, 4, 6]},
            cv=3, scoring='accuracy', refit=True,
        )
        ncv = NestedCV(search=search, outer_cv=3)
        result = ncv.evaluate(X, y)
        assert len(result.best_params) == 3
        assert all('max_depth' in p for p in result.best_params)
        assert len(result.inner_scores) == 3

    def test_oof_predictions(self, clf_data):
        X, y = clf_data
        ncv = NestedCV(
            estimator=DecisionTreeClassifier(random_state=42),
            outer_cv=3,
            return_oof=True,
        )
        result = ncv.evaluate(X, y)
        assert result.oof_predictions is not None
        assert len(result.oof_predictions) == len(y)

    def test_auto_scoring(self, clf_data, reg_data):
        # Classifier should use accuracy
        X, y = clf_data
        ncv = NestedCV(estimator=DecisionTreeClassifier(random_state=42), outer_cv=3)
        result = ncv.evaluate(X, y)
        assert result.scoring == 'accuracy'

        # Regressor should use r2
        X, y = reg_data
        ncv = NestedCV(estimator=DecisionTreeRegressor(random_state=42), outer_cv=3)
        result = ncv.evaluate(X, y)
        assert result.scoring == 'r2'

    def test_custom_scoring(self, clf_data):
        X, y = clf_data
        ncv = NestedCV(
            estimator=DecisionTreeClassifier(random_state=42),
            outer_cv=3,
            scoring='balanced_accuracy',
        )
        result = ncv.evaluate(X, y)
        assert result.scoring == 'balanced_accuracy'

    def test_requires_estimator_or_search(self):
        with pytest.raises(ValueError, match="Either"):
            NestedCV()

    def test_result_repr(self, clf_data):
        X, y = clf_data
        ncv = NestedCV(estimator=DecisionTreeClassifier(random_state=42), outer_cv=3)
        result = ncv.evaluate(X, y)
        r = repr(result)
        assert 'NestedCVResult' in r
        assert 'n_folds=3' in r
