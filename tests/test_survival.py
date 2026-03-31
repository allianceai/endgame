"""Tests for the endgame.survival module.

Covers base utilities, metrics, non-parametric estimators, parametric models,
Cox models, datasets, validation, and ensembles. Tree/GBDT tests are skipped
when scikit-survival or xgboost are not available.
"""

from __future__ import annotations

import pytest
import numpy as np
from sklearn.base import clone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def survival_data():
    """Moderately-sized dataset with correlated features and survival times."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    # True model: time depends on first 2 features
    true_risk = np.exp(0.5 * X[:, 0] - 0.3 * X[:, 1])
    time = rng.exponential(1.0 / true_risk)
    # Random censoring
    censor_time = rng.exponential(3.0, n)
    observed_time = np.minimum(time, censor_time)
    event = time <= censor_time
    from endgame.survival.base import make_survival_y

    y = make_survival_y(observed_time, event)
    return X, y


@pytest.fixture
def simple_survival():
    """Small dataset for exact checks."""
    from endgame.survival.base import make_survival_y

    time = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    event = np.array([True, True, False, True, True, False, True, False])
    return make_survival_y(time, event)


# ---------------------------------------------------------------------------
# 1. test_make_survival_y
# ---------------------------------------------------------------------------


class TestMakeSurvivalY:
    def test_correct_dtype_and_fields(self):
        from endgame.survival.base import make_survival_y, SURVIVAL_DTYPE

        y = make_survival_y([1.0, 2.0, 3.0], [True, False, True])
        assert y.dtype == SURVIVAL_DTYPE
        assert y.dtype.names == ("event", "time")

    def test_correct_shape(self):
        from endgame.survival.base import make_survival_y

        y = make_survival_y([1, 2, 3, 4], [True, True, False, True])
        assert y.shape == (4,)

    def test_correct_values(self):
        from endgame.survival.base import make_survival_y

        time = [5.0, 3.0, 8.0]
        event = [True, False, True]
        y = make_survival_y(time, event)
        np.testing.assert_array_equal(y["time"], [5.0, 3.0, 8.0])
        np.testing.assert_array_equal(y["event"], [True, False, True])

    def test_mismatched_shapes_raises(self):
        from endgame.survival.base import make_survival_y

        with pytest.raises(ValueError, match="same shape"):
            make_survival_y([1, 2, 3], [True, False])

    def test_int_event_cast_to_bool(self):
        from endgame.survival.base import make_survival_y

        y = make_survival_y([1.0, 2.0], [1, 0])
        assert y["event"].dtype == bool
        assert y["event"][0] is np.bool_(True)
        assert y["event"][1] is np.bool_(False)


# ---------------------------------------------------------------------------
# 2-4. Concordance index tests
# ---------------------------------------------------------------------------


class TestConcordanceIndex:
    def test_perfect_ordering(self):
        """Perfect risk ordering gives C-index = 1.0."""
        from endgame.survival.base import make_survival_y
        from endgame.survival.metrics import concordance_index

        time = np.array([1, 2, 3, 4, 5], dtype=float)
        event = np.array([True, True, True, True, True])
        y = make_survival_y(time, event)
        # Higher risk for shorter time
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        c = concordance_index(y, risk)
        assert c == pytest.approx(1.0)

    def test_inverse_ordering(self):
        """Inverse risk ordering gives C-index = 0.0."""
        from endgame.survival.base import make_survival_y
        from endgame.survival.metrics import concordance_index

        time = np.array([1, 2, 3, 4, 5], dtype=float)
        event = np.array([True, True, True, True, True])
        y = make_survival_y(time, event)
        # Lower risk for shorter time (completely wrong)
        risk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = concordance_index(y, risk)
        assert c == pytest.approx(0.0)

    def test_random_around_half(self):
        """Random risk scores should give C-index near 0.5."""
        from endgame.survival.base import make_survival_y
        from endgame.survival.metrics import concordance_index

        rng = np.random.RandomState(123)
        n = 500
        time = rng.exponential(5.0, n)
        event = rng.binomial(1, 0.7, n).astype(bool)
        y = make_survival_y(time, event)
        risk = rng.randn(n)
        c = concordance_index(y, risk)
        assert 0.4 < c < 0.6

    def test_concordance_index_censored(self):
        """Test the alternate concordance_index_censored interface."""
        from endgame.survival.metrics import concordance_index_censored

        time = np.array([1, 2, 3, 4, 5], dtype=float)
        event = np.array([True, True, True, True, True])
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        c = concordance_index_censored(event, time, risk)
        assert c == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5-7. Kaplan-Meier tests
# ---------------------------------------------------------------------------


class TestKaplanMeier:
    def test_fit_attributes(self, simple_survival):
        from endgame.survival.nonparametric import KaplanMeierEstimator

        km = KaplanMeierEstimator()
        km.fit(simple_survival)
        assert hasattr(km, "survival_function_")
        assert hasattr(km, "timeline_")
        assert hasattr(km, "median_survival_time_")
        assert hasattr(km, "confidence_intervals_")
        assert km.n_events_ > 0
        assert km.n_censored_ >= 0

    def test_survival_monotone_non_increasing(self, simple_survival):
        from endgame.survival.nonparametric import KaplanMeierEstimator

        km = KaplanMeierEstimator().fit(simple_survival)
        sf = km.survival_function_
        # S(t) should be monotonically non-increasing
        diffs = np.diff(sf)
        assert np.all(diffs <= 1e-15)

    def test_boundaries(self):
        from endgame.survival.base import make_survival_y
        from endgame.survival.nonparametric import KaplanMeierEstimator

        # All events, no censoring -> S should drop to 0
        time = np.array([1, 2, 3, 4, 5], dtype=float)
        event = np.array([True, True, True, True, True])
        y = make_survival_y(time, event)
        km = KaplanMeierEstimator().fit(y)

        # S(0) = 1.0 (before any events)
        s_before = km.predict_survival_function(np.array([0.0]))
        assert s_before[0] == pytest.approx(1.0)

        # S(t) at last event time with all events should eventually be 0
        assert km.survival_function_[-1] == pytest.approx(0.0, abs=1e-10)

    def test_predict_survival_function_at_query_times(self, simple_survival):
        from endgame.survival.nonparametric import KaplanMeierEstimator

        km = KaplanMeierEstimator().fit(simple_survival)
        query = np.array([0.5, 1.5, 5.0, 100.0])
        sf = km.predict_survival_function(query)
        assert sf.shape == (4,)
        # Before first event, S = 1.0
        assert sf[0] == pytest.approx(1.0)
        # All values in [0, 1]
        assert np.all((sf >= 0) & (sf <= 1))

    def test_log_rank_test(self):
        from endgame.survival.base import make_survival_y
        from endgame.survival.nonparametric import KaplanMeierEstimator

        y1 = make_survival_y([1, 2, 3], [True, True, True])
        y2 = make_survival_y([10, 20, 30], [True, True, True])
        result = KaplanMeierEstimator.log_rank_test(y1, y2)
        assert "chi2" in result
        assert "p_value" in result
        assert result["chi2"] > 0


# ---------------------------------------------------------------------------
# 8. Nelson-Aalen tests
# ---------------------------------------------------------------------------


class TestNelsonAalen:
    def test_fit_attributes(self, simple_survival):
        from endgame.survival.nonparametric import NelsonAalenEstimator

        na = NelsonAalenEstimator().fit(simple_survival)
        assert hasattr(na, "cumulative_hazard_")
        assert hasattr(na, "timeline_")
        assert na.n_events_ > 0

    def test_cumulative_hazard_non_decreasing(self, simple_survival):
        from endgame.survival.nonparametric import NelsonAalenEstimator

        na = NelsonAalenEstimator().fit(simple_survival)
        ch = na.cumulative_hazard_
        diffs = np.diff(ch)
        assert np.all(diffs >= -1e-15)

    def test_predict_cumulative_hazard(self, simple_survival):
        from endgame.survival.nonparametric import NelsonAalenEstimator

        na = NelsonAalenEstimator().fit(simple_survival)
        query = np.array([0.5, 2.0, 10.0])
        ch = na.predict_cumulative_hazard(query)
        assert ch.shape == (3,)
        # Before first event, H = 0
        assert ch[0] == pytest.approx(0.0)

    def test_summary(self, simple_survival):
        from endgame.survival.nonparametric import NelsonAalenEstimator

        na = NelsonAalenEstimator().fit(simple_survival)
        s = na.summary()
        assert "cumulative_hazard" in s
        assert "timeline" in s
        assert "n_events" in s


# ---------------------------------------------------------------------------
# 9. Exponential regressor
# ---------------------------------------------------------------------------


class TestExponentialRegressor:
    def test_fit_predict(self, survival_data):
        from endgame.survival.parametric import ExponentialRegressor
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        model = ExponentialRegressor()
        model.fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

        c = concordance_index(y, risk)
        assert c > 0.5, f"C-index {c} should be > 0.5 for correlated data"

    def test_has_attributes(self, survival_data):
        from endgame.survival.parametric import ExponentialRegressor

        X, y = survival_data
        model = ExponentialRegressor().fit(X, y)
        assert hasattr(model, "coefficients_")
        assert hasattr(model, "intercept_")
        assert hasattr(model, "AIC_")
        assert hasattr(model, "BIC_")


# ---------------------------------------------------------------------------
# 10. Weibull AFT
# ---------------------------------------------------------------------------


class TestWeibullAFT:
    def test_fit_predict(self, survival_data):
        from endgame.survival.parametric import WeibullAFTRegressor

        X, y = survival_data
        model = WeibullAFTRegressor().fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

    def test_attributes(self, survival_data):
        from endgame.survival.parametric import WeibullAFTRegressor

        X, y = survival_data
        model = WeibullAFTRegressor().fit(X, y)
        assert hasattr(model, "coefficients_")
        assert hasattr(model, "AIC_")
        assert hasattr(model, "shape_")

    def test_median_survival_time(self, survival_data):
        from endgame.survival.parametric import WeibullAFTRegressor

        X, y = survival_data
        model = WeibullAFTRegressor().fit(X, y)
        med = model.predict_median_survival_time(X)
        assert med.shape == (len(X),)
        assert np.all(med > 0)

    def test_summary(self, survival_data):
        from endgame.survival.parametric import WeibullAFTRegressor

        X, y = survival_data
        model = WeibullAFTRegressor().fit(X, y)
        s = model.summary()
        assert "coefficients" in s
        assert "intercept" in s
        assert "shape" in s
        assert "AIC" in s
        assert "BIC" in s


# ---------------------------------------------------------------------------
# 11. Log-Normal AFT
# ---------------------------------------------------------------------------


class TestLogNormalAFT:
    def test_fit_predict(self, survival_data):
        from endgame.survival.parametric import LogNormalAFTRegressor

        X, y = survival_data
        model = LogNormalAFTRegressor().fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

    def test_survival_function(self, survival_data):
        from endgame.survival.parametric import LogNormalAFTRegressor

        X, y = survival_data
        model = LogNormalAFTRegressor().fit(X, y)
        times = np.array([0.5, 1.0, 2.0, 5.0])
        sf = model.predict_survival_function(X[:5], times)
        assert sf.shape == (5, 4)
        assert np.all((sf >= 0) & (sf <= 1))


# ---------------------------------------------------------------------------
# 12. Log-Logistic AFT
# ---------------------------------------------------------------------------


class TestLogLogisticAFT:
    def test_fit_predict(self, survival_data):
        from endgame.survival.parametric import LogLogisticAFTRegressor

        X, y = survival_data
        model = LogLogisticAFTRegressor().fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

    def test_survival_function_values(self, survival_data):
        from endgame.survival.parametric import LogLogisticAFTRegressor

        X, y = survival_data
        model = LogLogisticAFTRegressor().fit(X, y)
        times = np.array([0.1, 1.0, 10.0])
        sf = model.predict_survival_function(X[:3], times)
        assert sf.shape == (3, 3)
        assert np.all((sf >= 0) & (sf <= 1))


# ---------------------------------------------------------------------------
# 13. Cox PH
# ---------------------------------------------------------------------------


class TestCoxPH:
    def test_fit_predict(self, survival_data):
        from endgame.survival.cox import CoxPHRegressor
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        model = CoxPHRegressor(penalizer=0.01)
        model.fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

        c = concordance_index(y, risk)
        assert c > 0.55, f"C-index {c} should be > 0.55 on correlated data"

    def test_attributes(self, survival_data):
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model = CoxPHRegressor().fit(X, y)
        assert hasattr(model, "coefficients_")
        assert hasattr(model, "hazard_ratios_")
        assert hasattr(model, "concordance_index_")
        assert hasattr(model, "log_likelihood_")
        np.testing.assert_array_almost_equal(
            model.hazard_ratios_, np.exp(model.coefficients_)
        )

    # 14. Cox PH summary
    def test_summary(self, survival_data):
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model = CoxPHRegressor(penalizer=0.01).fit(X, y)
        s = model.summary()
        assert "coef" in s
        assert "exp(coef)" in s
        assert "se" in s
        assert "z" in s
        assert "p" in s

    # 15. Cox PH survival function
    def test_survival_function(self, survival_data):
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model = CoxPHRegressor(penalizer=0.01).fit(X, y)
        times = np.array([0.5, 1.0, 2.0, 5.0])
        sf = model.predict_survival_function(X[:10], times)
        assert sf.shape == (10, 4)
        assert np.all((sf >= -1e-10) & (sf <= 1.0 + 1e-10))

    # 16. Cox PH regularization
    def test_regularization_shrinks_coefficients(self, survival_data):
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model_low = CoxPHRegressor(penalizer=0.01).fit(X, y)
        model_high = CoxPHRegressor(penalizer=10.0).fit(X, y)

        norm_low = np.linalg.norm(model_low.coefficients_)
        norm_high = np.linalg.norm(model_high.coefficients_)
        assert norm_high < norm_low, (
            "Higher penalizer should shrink coefficient magnitudes"
        )


# ---------------------------------------------------------------------------
# 17-18. Sklearn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompatibility:
    def test_clone_cox(self):
        from endgame.survival.cox import CoxPHRegressor

        model = CoxPHRegressor(penalizer=0.5, l1_ratio=0.2)
        cloned = clone(model)
        assert cloned.penalizer == 0.5
        assert cloned.l1_ratio == 0.2
        assert cloned is not model

    def test_clone_weibull(self):
        from endgame.survival.parametric import WeibullAFTRegressor

        model = WeibullAFTRegressor(alpha=0.1)
        cloned = clone(model)
        assert cloned.alpha == 0.1
        assert cloned is not model

    def test_get_set_params_cox(self):
        from endgame.survival.cox import CoxPHRegressor

        model = CoxPHRegressor(penalizer=0.1, max_iter=200)
        params = model.get_params()
        assert params["penalizer"] == 0.1
        assert params["max_iter"] == 200

        model.set_params(penalizer=0.5)
        assert model.penalizer == 0.5
        # Round-trip
        params2 = model.get_params()
        assert params2["penalizer"] == 0.5

    def test_get_set_params_weibull(self):
        from endgame.survival.parametric import WeibullAFTRegressor

        model = WeibullAFTRegressor(alpha=0.3)
        params = model.get_params()
        assert params["alpha"] == 0.3

        model.set_params(alpha=1.0)
        assert model.alpha == 1.0

    def test_score_method(self, survival_data):
        """BaseSurvivalEstimator.score returns concordance index."""
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model = CoxPHRegressor(penalizer=0.01).fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 19. Synthetic dataset
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_make_synthetic_survival(self):
        from endgame.survival.datasets import make_synthetic_survival

        X, y, true_coef = make_synthetic_survival(
            n_samples=100, n_features=8, random_state=42
        )
        assert X.shape == (100, 8)
        assert y.shape == (100,)
        assert y.dtype.names == ("event", "time")
        assert true_coef.shape == (8,)
        # First 5 coefficients should be non-zero
        assert np.any(true_coef[:5] != 0)
        # Remaining should be zero
        assert np.all(true_coef[5:] == 0)

    # 20. Veterans dataset
    def test_load_veterans(self):
        from endgame.survival.datasets import load_veterans

        X, y, feature_names = load_veterans()
        assert X.shape == (137, 6)
        assert y.shape == (137,)
        assert len(feature_names) == 6
        assert y.dtype.names == ("event", "time")
        assert np.all(y["time"] > 0)

    def test_load_rossi(self):
        from endgame.survival.datasets import load_rossi

        X, y, feature_names = load_rossi()
        assert X.shape == (432, 7)
        assert y.shape == (432,)
        assert len(feature_names) == 7

    def test_load_gbsg2(self):
        from endgame.survival.datasets import load_gbsg2

        X, y, feature_names = load_gbsg2()
        assert X.ndim == 2
        assert y.shape[0] == X.shape[0]
        assert y.dtype.names is not None

    def test_load_whas500(self):
        from endgame.survival.datasets import load_whas500

        X, y, feature_names = load_whas500()
        assert X.ndim == 2
        assert y.shape[0] == X.shape[0]
        assert len(feature_names) > 0


# ---------------------------------------------------------------------------
# 21. Survival stratified K-fold
# ---------------------------------------------------------------------------


class TestSurvivalStratifiedKFold:
    def test_event_rate_preserved(self, survival_data):
        from endgame.survival.validation import SurvivalStratifiedKFold
        from endgame.survival.base import _get_time_event

        X, y = survival_data
        overall_rate = y["event"].mean()

        cv = SurvivalStratifiedKFold(n_splits=5, random_state=42)
        for train_idx, test_idx in cv.split(X, y):
            fold_rate = y[test_idx]["event"].mean()
            # Event rate per fold should be within 15% of overall
            assert abs(fold_rate - overall_rate) < 0.15, (
                f"Fold event rate {fold_rate:.2f} too far from "
                f"overall {overall_rate:.2f}"
            )

    def test_correct_number_of_folds(self, survival_data):
        from endgame.survival.validation import SurvivalStratifiedKFold

        X, y = survival_data
        cv = SurvivalStratifiedKFold(n_splits=3, random_state=0)
        folds = list(cv.split(X, y))
        assert len(folds) == 3


# ---------------------------------------------------------------------------
# 22. evaluate_survival
# ---------------------------------------------------------------------------


class TestEvaluateSurvival:
    def test_returns_concordance_index(self, survival_data):
        from endgame.survival.validation import evaluate_survival
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model = CoxPHRegressor(penalizer=0.1)
        results = evaluate_survival(model, X, y, cv=3)
        assert "concordance_index" in results
        ci = results["concordance_index"]
        assert "mean" in ci
        assert "std" in ci
        assert "folds" in ci
        assert len(ci["folds"]) == 3
        assert 0.0 < ci["mean"] < 1.0


# ---------------------------------------------------------------------------
# 23. survival_train_test_split
# ---------------------------------------------------------------------------


class TestSurvivalTrainTestSplit:
    def test_preserves_event_rate(self, survival_data):
        from endgame.survival.validation import survival_train_test_split

        X, y = survival_data
        X_train, X_test, y_train, y_test = survival_train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        overall_rate = y["event"].mean()
        train_rate = y_train["event"].mean()
        test_rate = y_test["event"].mean()

        assert abs(train_rate - overall_rate) < 0.1
        assert abs(test_rate - overall_rate) < 0.1

    def test_correct_sizes(self, survival_data):
        from endgame.survival.validation import survival_train_test_split

        X, y = survival_data
        X_train, X_test, y_train, y_test = survival_train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_test) == pytest.approx(len(X) * 0.2, abs=2)


# ---------------------------------------------------------------------------
# 24. Voting ensemble
# ---------------------------------------------------------------------------


class TestSurvivalVotingEnsemble:
    def test_combines_models(self, survival_data):
        from endgame.survival.ensemble import SurvivalVotingEnsemble
        from endgame.survival.parametric import ExponentialRegressor
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        voting = SurvivalVotingEnsemble(
            estimators=[
                ("exp", ExponentialRegressor()),
                ("cox", CoxPHRegressor(penalizer=0.1)),
            ],
            weights=[0.4, 0.6],
        )
        voting.fit(X, y)
        risk = voting.predict(X)
        assert risk.shape == (len(X),)
        assert not np.any(np.isnan(risk))

    def test_uniform_weights(self, survival_data):
        from endgame.survival.ensemble import SurvivalVotingEnsemble
        from endgame.survival.parametric import ExponentialRegressor

        X, y = survival_data
        voting = SurvivalVotingEnsemble(
            estimators=[
                ("exp1", ExponentialRegressor()),
                ("exp2", ExponentialRegressor(alpha=0.1)),
            ],
        )
        voting.fit(X, y)
        np.testing.assert_array_almost_equal(
            voting.weights_, [0.5, 0.5]
        )


# ---------------------------------------------------------------------------
# 25. Stacking ensemble
# ---------------------------------------------------------------------------


class TestSurvivalStackingEnsemble:
    def test_stacking_basic(self, survival_data):
        from endgame.survival.ensemble import SurvivalStackingEnsemble
        from endgame.survival.parametric import ExponentialRegressor
        from endgame.survival.cox import CoxPHRegressor
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        stack = SurvivalStackingEnsemble(
            base_estimators=[
                ExponentialRegressor(),
                CoxPHRegressor(penalizer=0.1),
            ],
            cv=3,
            random_state=42,
        )
        stack.fit(X, y)
        risk = stack.predict(X)
        assert risk.shape == (len(X),)

        c = concordance_index(y, risk)
        # Should be at least reasonable
        assert c > 0.45, f"Stacking C-index {c} unexpectedly low"

    def test_cv_scores_available(self, survival_data):
        from endgame.survival.ensemble import SurvivalStackingEnsemble
        from endgame.survival.parametric import ExponentialRegressor

        X, y = survival_data
        stack = SurvivalStackingEnsemble(
            base_estimators=[ExponentialRegressor()],
            cv=3,
        )
        stack.fit(X, y)
        assert hasattr(stack, "cv_scores_")
        assert len(stack.cv_scores_) == 1  # one base estimator


# ---------------------------------------------------------------------------
# 26. Random Survival Forest (requires sksurv)
# ---------------------------------------------------------------------------


class TestRandomSurvivalForest:
    @pytest.fixture(autouse=True)
    def _skip_without_sksurv(self):
        pytest.importorskip("sksurv")

    def test_fit_predict(self, survival_data):
        from endgame.survival.trees import RandomSurvivalForestRegressor
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        model = RandomSurvivalForestRegressor(
            n_estimators=20, random_state=42
        )
        model.fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

        c = concordance_index(y, risk)
        assert c > 0.5, f"RSF C-index {c} should be > 0.5"

    def test_survival_function(self, survival_data):
        from endgame.survival.trees import RandomSurvivalForestRegressor

        X, y = survival_data
        model = RandomSurvivalForestRegressor(
            n_estimators=10, random_state=42
        ).fit(X, y)
        sf = model.predict_survival_function(X[:5])
        assert sf.ndim == 2
        assert sf.shape[0] == 5


# ---------------------------------------------------------------------------
# 27. Gradient Boosted Survival (requires sksurv)
# ---------------------------------------------------------------------------


class TestGradientBoostedSurvival:
    @pytest.fixture(autouse=True)
    def _skip_without_sksurv(self):
        pytest.importorskip("sksurv")

    def test_fit_predict(self, survival_data):
        from endgame.survival.trees import GradientBoostedSurvivalRegressor
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        model = GradientBoostedSurvivalRegressor(
            n_estimators=20, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

        c = concordance_index(y, risk)
        assert c > 0.5


# ---------------------------------------------------------------------------
# 28. SurvivalGBDTWrapper with XGBoost
# ---------------------------------------------------------------------------


class TestSurvivalGBDTXGBoost:
    @pytest.fixture(autouse=True)
    def _skip_without_xgboost(self):
        pytest.importorskip("xgboost")

    def test_fit_predict(self, survival_data):
        from endgame.survival.gbdt import SurvivalGBDTWrapper
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        model = SurvivalGBDTWrapper(
            backend="xgboost",
            n_estimators=20,
            max_depth=3,
            learning_rate=0.1,
            preset="fast",
            random_state=42,
        )
        model.fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)

        c = concordance_index(y, risk)
        assert c > 0.5, f"XGBoost survival C-index {c} should be > 0.5"


# ---------------------------------------------------------------------------
# 29. DeepSurv (requires torch) -- skipped since neural.py does not exist
# ---------------------------------------------------------------------------


class TestDeepSurv:
    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")
        # Also skip if the neural module does not exist
        try:
            from endgame.survival import neural  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pytest.skip("endgame.survival.neural not available")

    def test_fit_predict(self, survival_data):
        from endgame.survival.neural import DeepSurvRegressor
        from endgame.survival.metrics import concordance_index

        X, y = survival_data
        model = DeepSurvRegressor(random_state=42)
        model.fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)
        c = concordance_index(y, risk)
        assert c > 0.5


# ---------------------------------------------------------------------------
# 30. Aalen-Johansen (competing risks) -- skipped if module missing
# ---------------------------------------------------------------------------


class TestAalenJohansen:
    @pytest.fixture(autouse=True)
    def _skip_if_not_available(self):
        try:
            from endgame.survival.competing_risks import (  # noqa: F401
                AalenJohansenEstimator,
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("endgame.survival.competing_risks not available")

    def test_aalen_johansen(self):
        from endgame.survival.competing_risks import AalenJohansenEstimator
        from endgame.survival.base import make_survival_y

        rng = np.random.RandomState(42)
        n = 100
        time = rng.exponential(5.0, n)
        # event_type: 0 = censored, 1 = cause 1, 2 = cause 2
        event_type = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        event = event_type > 0

        y = make_survival_y(time, event)
        estimator = AalenJohansenEstimator()
        estimator.fit(y, event_type=event_type)

        # CIF values should be in [0, 1]
        if hasattr(estimator, "cif_"):
            for cause_cif in estimator.cif_.values():
                assert np.all(cause_cif >= 0)
                assert np.all(cause_cif <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Additional parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ModelClass",
    [
        "ExponentialRegressor",
        "WeibullAFTRegressor",
        "LogNormalAFTRegressor",
        "LogLogisticAFTRegressor",
    ],
)
class TestParametricModels:
    def test_fit_predict_survival_function(self, ModelClass, survival_data):
        from endgame.survival import parametric

        Cls = getattr(parametric, ModelClass)
        X, y = survival_data
        model = Cls().fit(X, y)
        times = np.array([0.5, 1.0, 3.0])
        sf = model.predict_survival_function(X[:5], times)
        assert sf.shape == (5, 3)
        assert np.all(sf >= -1e-10)
        assert np.all(sf <= 1.0 + 1e-10)

    def test_predict_shape(self, ModelClass, survival_data):
        from endgame.survival import parametric

        Cls = getattr(parametric, ModelClass)
        X, y = survival_data
        model = Cls().fit(X, y)
        risk = model.predict(X)
        assert risk.shape == (len(X),)
        assert not np.any(np.isnan(risk))


# ---------------------------------------------------------------------------
# _check_survival_y edge cases
# ---------------------------------------------------------------------------


class TestCheckSurvivalY:
    def test_tuple_input(self):
        from endgame.survival.base import _check_survival_y

        time = np.array([1.0, 2.0, 3.0])
        event = np.array([True, False, True])
        y = _check_survival_y((time, event))
        assert y.dtype.names == ("event", "time")
        np.testing.assert_array_equal(y["time"], time)

    def test_invalid_input_raises(self):
        from endgame.survival.base import _check_survival_y

        with pytest.raises(ValueError, match="must be a structured array"):
            _check_survival_y([1, 2, 3])

    def test_already_structured(self):
        from endgame.survival.base import _check_survival_y, make_survival_y

        y_orig = make_survival_y([1.0, 2.0], [True, False])
        y = _check_survival_y(y_orig)
        np.testing.assert_array_equal(y["time"], y_orig["time"])
        np.testing.assert_array_equal(y["event"], y_orig["event"])


# ---------------------------------------------------------------------------
# SurvivalPrediction container
# ---------------------------------------------------------------------------


class TestSurvivalPrediction:
    def test_predict_full(self, survival_data):
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        model = CoxPHRegressor(penalizer=0.01).fit(X, y)
        pred = model.predict_full(X[:5])
        assert pred.risk_scores.shape == (5,)
        assert pred.survival_function is not None
        assert pred.survival_function.ndim == 2


# ---------------------------------------------------------------------------
# SurvivalTimeSeriesSplit
# ---------------------------------------------------------------------------


class TestSurvivalTimeSeriesSplit:
    def test_splits(self, survival_data):
        from endgame.survival.validation import SurvivalTimeSeriesSplit

        X, y = survival_data
        cv = SurvivalTimeSeriesSplit(n_splits=3)
        folds = list(cv.split(X, y))
        assert len(folds) >= 1
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0


# ---------------------------------------------------------------------------
# Hill climbing ensemble
# ---------------------------------------------------------------------------


class TestSurvivalHillClimbingEnsemble:
    def test_basic(self, survival_data):
        from endgame.survival.ensemble import SurvivalHillClimbingEnsemble
        from endgame.survival.parametric import ExponentialRegressor
        from endgame.survival.cox import CoxPHRegressor

        X, y = survival_data
        # Generate OOF predictions from two models
        m1 = ExponentialRegressor().fit(X, y)
        m2 = CoxPHRegressor(penalizer=0.1).fit(X, y)
        oof = np.column_stack([m1.predict(X), m2.predict(X)])

        hc = SurvivalHillClimbingEnsemble(
            n_iterations=10, patience=5
        )
        hc.fit(oof, y)
        assert hasattr(hc, "weights_")
        assert hasattr(hc, "best_score_")

        combined = hc.predict(base_predictions=oof)
        assert combined.shape == (len(X),)
