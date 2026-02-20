"""Tests for Knowledge Distillation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from endgame.ensemble.distillation import KnowledgeDistiller


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return X, y


class TestKnowledgeDistiller:
    def test_basic_classification(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=DecisionTreeClassifier(max_depth=3, random_state=42),
        )
        distiller.fit(X, y)
        preds = distiller.predict(X)
        assert preds.shape == (len(X),)
        assert distiller.student_score_ > 0

    def test_basic_regression(self, reg_data):
        X, y = reg_data
        teacher = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=LinearRegression(),
        )
        distiller.fit(X, y)
        preds = distiller.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=LogisticRegression(max_iter=500),
        )
        distiller.fit(X, y)
        proba = distiller.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_augmentation(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=DecisionTreeClassifier(max_depth=4, random_state=42),
            augment=True,
            augment_ratio=0.5,
            random_state=42,
        )
        distiller.fit(X, y)
        assert distiller.student_score_ > 0

    def test_temperature(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        # Different temperatures
        for temp in [1.0, 3.0, 10.0]:
            distiller = KnowledgeDistiller(
                teacher=teacher,
                student=DecisionTreeClassifier(max_depth=3, random_state=42),
                temperature=temp,
            )
            distiller.fit(X, y)
            assert distiller.student_score_ > 0

    def test_alpha(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        for alpha in [0.0, 0.5, 1.0]:
            distiller = KnowledgeDistiller(
                teacher=teacher,
                student=DecisionTreeClassifier(max_depth=3, random_state=42),
                alpha=alpha,
                random_state=42,
            )
            distiller.fit(X, y)
            assert distiller.student_score_ > 0

    def test_compression_report(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=DecisionTreeClassifier(max_depth=3, random_state=42),
        )
        distiller.fit(X, y)
        report = distiller.compression_report()
        assert 'teacher_score' in report
        assert 'student_score' in report
        assert 'score_retention_pct' in report
        assert report['metric'] == 'accuracy'

    def test_regression_report(self, reg_data):
        X, y = reg_data
        teacher = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=LinearRegression(),
        )
        distiller.fit(X, y)
        report = distiller.compression_report()
        assert report['metric'] == 'r2'

    def test_feature_importances(self, clf_data):
        X, y = clf_data
        teacher = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=DecisionTreeClassifier(max_depth=3, random_state=42),
        )
        distiller.fit(X, y)
        assert distiller.feature_importances_ is not None
        assert len(distiller.feature_importances_) == X.shape[1]
