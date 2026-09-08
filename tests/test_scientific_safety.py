"""Scientific contracts and regressions from the b5c41dc imaging review."""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from endgame.models import PLSDAClassifier
from endgame.models.block_stacking import BlockStackingClassifier
from endgame.preprocessing import ComBatHarmonizer, NormativeDeviation
from endgame.preprocessing.blocks import BlockwiseHarmonizer, MissingBlockIndicator
from endgame.utils.diagnostics import batch_leakage_check
from endgame.utils.metrics import bootstrap_ci, decision_curve, delong_test, paired_bootstrap_diff
from endgame.validation import PurgedPanelSplit


def sites(n=160, p=6, seed=42):
    rng = np.random.default_rng(seed)
    batch = np.repeat(["A", "B"], n//2)
    X = pd.DataFrame(rng.normal(size=(n, p)) + (batch == "B")[:, None]*np.arange(1, p+1),
                     columns=[f"f{i}" for i in range(p)])
    X["site"] = batch
    X.index = pd.Index([f"patient_{i}" for i in range(n)], name="patient")
    return X


def test_combat_one_active_feature_has_explicit_finite_fallback():
    X = sites(p=1)
    with pytest.raises(ValueError, match="at least two active"):
        ComBatHarmonizer("site").fit(X)
    plain = ComBatHarmonizer("site", eb=False).fit_transform(X)
    with pytest.warns(UserWarning, match="unshrunk"):
        fallback = ComBatHarmonizer("site", eb_fallback="no_eb").fit_transform(X)
    np.testing.assert_allclose(plain, fallback)
    assert np.isfinite(fallback).all().all()


@pytest.mark.parametrize("options", [{}, {"mean_only": True}, {"eb": False}])
def test_combat_excluded_constants_cannot_change_active_features(options):
    X = sites()
    base = ComBatHarmonizer("site", **options).fit_transform(X)
    padded = pd.concat([X, pd.DataFrame(0., index=X.index, columns=[f"zero{i}" for i in range(40)])], axis=1)
    h = ComBatHarmonizer("site", **options).fit(padded)
    actual = h.transform(padded)
    np.testing.assert_allclose(actual[base.columns], base, atol=1e-9)
    assert h.constant_mask_.sum() == 40
    assert actual[[f"zero{i}" for i in range(40)]].eq(0).all().all()


def test_combat_degenerate_priors_and_iteration_are_explicit():
    X = sites()
    with pytest.raises(ValueError, match="did not converge"):
        ComBatHarmonizer("site", max_iter=1, tol=1e-15).fit(X)
    with pytest.warns(UserWarning, match="unshrunk"):
        h = ComBatHarmonizer("site", max_iter=1, tol=1e-15, eb_fallback="no_eb").fit(X)
    assert np.isfinite(h.transform(X)).all().all()
    # Exactly equal within-batch variances yield an undefined inverse-gamma prior.
    z = pd.DataFrame({"f0": [-1., 1., 2., 4.], "f1": [-1., 1., 2., 4.], "site": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="variance prior"):
        ComBatHarmonizer("site").fit(z)


def test_combat_confounded_design_and_scanner_identifiers_fail():
    X = sites()
    X["age"] = np.where(X.site == "A", 60., 70.)
    with pytest.raises(ValueError, match="rank deficient"):
        ComBatHarmonizer("site", covariates=["age"]).fit(X)
    X = sites()
    X["scanner_code"] = (X.site == "B").astype(float)
    with pytest.raises(ValueError, match="possible batch identifiers"):
        ComBatHarmonizer("site").fit(X)
    with pytest.warns(UserWarning, match="passed through"):
        h = ComBatHarmonizer("site", degenerate_features="passthrough").fit(X)
    assert h.degenerate_mask_[-1] and not h.constant_mask_[-1]


@pytest.mark.parametrize("options", [{}, {"mean_only": True}, {"eb": False}, {"mean_only": True, "eb": False}])
def test_combat_reference_parity_across_supported_modes(options):
    neurocombat = pytest.importorskip("neuroCombat")
    X = sites()
    rng = np.random.default_rng(21)
    X["age"] = rng.uniform(40, 80, len(X))
    X["sex"] = np.tile(["F", "M"], len(X)//2)
    h = ComBatHarmonizer("site", covariates=["age"], categorical=["sex"], **options)
    ours = h.fit_transform(X)[h.features_].to_numpy()
    ref = neurocombat.neuroCombat(X[h.features_].to_numpy().T, X[["site", "age", "sex"]],
                                 batch_col="site", categorical_cols=["sex"], continuous_cols=["age"], **options)["data"].T
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-3)


def test_combat_schema_serialization_chunk_invariance_and_unknown_status():
    X = sites()
    X["age"] = np.arange(len(X)) % 13
    h = ComBatHarmonizer("site", covariates=["age"], unknown_batch="passthrough").fit(X)
    before = pickle.dumps(h)
    chunks = pd.concat([h.transform(X.iloc[i:i+17]) for i in range(0, len(X), 17)])
    np.testing.assert_allclose(chunks, h.transform(X))
    assert pickle.dumps(h) == before
    assert list(h.get_feature_names_out()) == list(h.transform(X))
    assert list(clone(h).fit(X).set_output(transform="pandas").transform(X)) == list(h.transform(X))
    new = X.iloc[:4].assign(site="new_scanner")
    with pytest.warns(UserWarning, match="raw passthrough"):
        actual = h.transform(new)
    np.testing.assert_allclose(actual[h.features_], new[h.features_])
    assert h.adjustment_report(new).status.eq("unseen_batch_passthrough").all()


def test_norm_marker_free_transform_and_reference_integrity():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"age": rng.uniform(40, 80, 120), "f": rng.normal(size=120), "control": np.tile([0, 1], 60)})
    h = NormativeDeviation(["age"], reference="control", extrapolation="ignore").fit(X)
    np.testing.assert_allclose(h.transform(X), h.transform(X.drop(columns="control")))
    np.testing.assert_allclose(h.transform(X), h.transform(X.assign(control=1-X.control)))
    assert list(h.get_feature_names_out()) == list(h.transform(X))
    assert list(h.set_output(transform="pandas").transform(X.drop(columns="control"))) == ["age", "f"]
    for bad in (np.nan, "False", 2, "0"):
        invalid = X.copy()
        invalid["control"] = invalid.control.astype(object)
        invalid.loc[0, "control"] = bad
        with pytest.raises(ValueError, match="membership"):
            NormativeDeviation(["age"], reference="control").fit(invalid)


def test_normative_held_out_controls_and_known_deficit():
    rng = np.random.default_rng(42)
    def sample(n, deficit=0.):
        age = rng.uniform(40, 80, n)
        return pd.DataFrame({"age": age, "f": 8-.05*age+rng.normal(0, .5, n)-deficit})
    h = NormativeDeviation(["age"], extrapolation="ignore").fit(sample(500))
    controls = h.transform(sample(800)).f
    patients = h.transform(sample(800, deficit=1.)).f
    assert abs(controls.mean()) < .15
    assert .85 < controls.std() < 1.15
    assert -2.3 < patients.mean() < -1.7


def test_norm_rejects_saturated_collinear_and_zero_residual_models():
    X = pd.DataFrame(np.eye(20), columns=[f"c{i}" for i in range(20)])
    X["f"] = np.arange(20)
    with pytest.raises(ValueError, match="rank deficient|degrees of freedom"):
        NormativeDeviation(list(X.columns[:-1])).fit(X)
    X = pd.DataFrame({"age": np.arange(40), "f": 2*np.arange(40)+1})
    with pytest.raises(ValueError, match="residual scale"):
        NormativeDeviation(["age"]).fit(X)


def stack_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    return pd.DataFrame({"age": x, "img": rng.normal(size=200)}, index=np.arange(1000, 1200)), (x+rng.normal(0, .5, 200) > 0).astype(int)


def test_stacking_single_row_and_batch_predictions_agree_with_missing_passthrough():
    X, y = stack_data()
    h = BlockStackingClassifier({"demo": ["age"], "img": ["img"]}, passthrough=["demo"], cv=3).fit(X, y)
    test = pd.DataFrame({"age": [np.nan, -5., 5.], "img": [0., 0., 0.]}, index=[42, 81, 93])
    state = pickle.dumps(h)
    solo = h.predict_proba(test.iloc[[0]])[0]
    np.testing.assert_allclose(solo, h.predict_proba(test.iloc[[0, 1]])[0])
    np.testing.assert_allclose(solo, h.predict_proba(test.iloc[[0, 2]])[0])
    assert pickle.dumps(h) == state
    assert h.block_scores(test).index.equals(test.index)
    np.testing.assert_allclose(pickle.loads(state).predict_proba(test), h.predict_proba(test))


def test_stacking_single_class_or_empty_modality_has_fitted_fallback():
    y = np.tile([0, 1], 40)
    X = pd.DataFrame({"a": np.where(y == 0, np.arange(80), np.nan), "empty": np.nan})
    with pytest.warns(UserWarning, match="training prior"):
        h = BlockStackingClassifier({"a": ["a"], "empty": ["empty"]}, passthrough=["empty"], cv=3).fit(X, y)
    assert h.block_models_["a"] is None
    assert np.isfinite(h.predict_proba(X)).all()
    assert np.isfinite(h.predict_proba(X.drop(columns="empty"))).all()


def test_stacking_grouped_oof_blocks_patient_memorization():
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(60), 4)
    y = np.repeat(np.r_[np.zeros(30, int), np.ones(30, int)], 4)
    X = pd.DataFrame(np.repeat(rng.normal(size=(60, 5)), 4, axis=0), columns=list("abcde"))
    h = BlockStackingClassifier({"m": list(X)}, base_estimator=KNeighborsClassifier(1), cv=GroupKFold(3)).fit(X, y, groups=groups)
    assert .3 < roc_auc_score(y, h.oof_predictions_[:, 0]) < .7
    assert all(not set(groups[tr]) & set(groups[va]) for tr, va in h.folds_)
    bad = [(np.arange(0, 160), np.arange(159, 240))]
    with pytest.raises(ValueError, match="overlap"):
        BlockStackingClassifier({"m": list(X)}, cv=bad).fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="patient groups"):
        BlockStackingClassifier({"m": list(X)}, cv=[(np.arange(0, 161), np.arange(161, 240))]).fit(X, y, groups=groups)


def test_stacking_chronological_coverage_and_future_perturbations():
    times = np.repeat(np.arange(20), 8)
    y = np.tile([0, 1], 80)
    X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=160) + y})
    kwargs = dict(times=times, label_end_times=times+1)
    cv = PurgedPanelSplit(n_splits=2, train_fraction=.6)
    h = BlockStackingClassifier({"m": ["a"]}, cv=cv).fit(X, y, **kwargs)
    assert np.isnan(h.oof_predictions_[~h.oof_coverage_]).all()
    np.testing.assert_array_equal(h.meta_training_rows_, np.flatnonzero(h.oof_coverage_))
    y2 = y.copy()
    y2[times >= 16] = 1-y2[times >= 16]
    other = clone(h).fit(X, y2, **kwargs)
    first_valid = h.folds_[0][1]
    np.testing.assert_allclose(h.oof_predictions_[first_valid], other.oof_predictions_[first_valid])
    with pytest.raises(ValueError, match="chronological splitter"):
        BlockStackingClassifier({"m": ["a"]}, cv=3).fit(X, y, **kwargs)
    with pytest.raises(ValueError, match="future decisions|overlapping label"):
        BlockStackingClassifier({"m": ["a"]}, cv=[(np.arange(80), np.arange(80, 120))]).fit(X, y, times=times, label_end_times=times+20)


class FitSpy(TransformerMixin, BaseEstimator):
    rows = []
    def fit(self, X, y=None):
        type(self).rows.append(set(X.index))
        return self
    def transform(self, X):
        return X


def test_stacking_preprocessing_is_inside_inner_fold_with_named_metadata():
    X = sites(n=240)
    y = np.tile([0, 1], 120)
    feats = [f"f{i}" for i in range(6)]
    FitSpy.rows = []
    base = make_pipeline(FitSpy(), ComBatHarmonizer("site", features=feats),
                         ColumnTransformer([("features", "passthrough", feats)]), LogisticRegression())
    h = BlockStackingClassifier({"m": feats}, base_estimator=base, block_metadata={"m": ["site"]}, cv=3).fit(X, y)
    assert len(FitSpy.rows) == 4  # three folds and one final refit
    for fitted_rows, (tr, va) in zip(FitSpy.rows[:3], h.folds_):
        assert fitted_rows == set(X.index[tr])
        assert not fitted_rows & set(X.index[va])
    assert h.predict_proba(X.iloc[:2]).shape == (2, 2)


def test_blockwise_partial_schema_rare_and_unknown_status():
    X = sites()
    spec = {"m": {"features": ["f0", "f1"], "batch": "site"}}
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="partially missing"):
        BlockwiseHarmonizer(spec).fit(X)
    h = BlockwiseHarmonizer(spec, partial_missing="passthrough").fit(X)
    assert h.is_fitted
    assert h.adjustment_report(X).iloc[0, 0] == "partial_missing_passthrough"
    assert np.isnan(h.transform(X).iloc[0, 0])
    missing = X.drop(columns=["f0", "f1"])
    assert h.adjustment_report(missing).m.eq("missing_modality").all()
    with pytest.raises(ValueError, match="schema"):
        h.transform(X.drop(columns="f0"))
    new = X.iloc[1:3].assign(site="new")
    with pytest.warns(UserWarning, match="raw passthrough"):
        out = h.transform(new)
    np.testing.assert_allclose(out[["f0", "f1"]], new[["f0", "f1"]])
    assert h.adjustment_report(new).m.eq("unseen_batch_passthrough").all()
    assert list(h.get_feature_names_out()) == list(h.transform(X))


def test_missing_indicator_empty_feature_fitted_state_and_schema():
    X = pd.DataFrame({"a": [np.nan]*20, "other": np.arange(20)})
    h = MissingBlockIndicator({"m": ["a"]}).fit(X)
    assert h.is_fitted and h.empty_features_["m"] == ["a"]
    assert h.transform(X).a.eq(0).all()
    assert h.transform(X.drop(columns="a")).m_missing.eq(1).all()
    assert list(h.get_feature_names_out(np.asarray(X.columns))) == list(h.transform(X))
    assert list(h.set_output(transform="pandas").transform(X)) == ["a", "other", "m_missing"]


def test_continuous_bootstrap_has_sampling_variation_and_rejects_class_strata():
    rng = np.random.default_rng(10)
    y = rng.normal(size=100)
    p = y + rng.normal(size=100)
    point, lo, hi = bootstrap_ci(mean_squared_error, y, p, n_boot=150)
    assert lo < point < hi and hi-lo > .1
    with pytest.raises(ValueError, match="continuous outcomes"):
        bootstrap_ci(mean_squared_error, y, p, stratified=True)


def test_cluster_bootstrap_resamples_whole_patients():
    # With constant per-patient prediction errors, duplicating visits cannot
    # change uncertainty when the independent resampling unit is the patient.
    y = np.arange(20.)
    p = y + np.random.default_rng(3).normal(size=20)
    one = bootstrap_ci(mean_squared_error, y, p, n_boot=150, groups=np.arange(20))
    repeated = bootstrap_ci(mean_squared_error, np.repeat(y, 4), np.repeat(p, 4),
                            n_boot=150, groups=np.repeat(np.arange(20), 4))
    np.testing.assert_allclose(one, repeated)


def test_paired_permutation_identical_scores_and_known_alternative():
    y = np.tile([0, 1], 100)
    rng = np.random.default_rng(22)
    good = y+rng.normal(0, .2, len(y))
    bad = rng.normal(size=len(y))
    same = paired_bootstrap_diff(roc_auc_score, y, good, good, n_boot=99, stratified=True)
    assert same["p_value"] == 1. and same["diff"] == 0.
    result = paired_bootstrap_diff(roc_auc_score, y, good, bad, n_boot=99, stratified=True)
    assert result["p_value"] <= .05 and result["ci_lo"] > 0
    assert result["p_value_method"] == "paired_permutation"


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"n_boot": 1}, {"n_boot": 2.5}, {"ci": 1}, {"ci": np.nan}])
def test_bootstrap_invalid_options_fail(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci(mean_squared_error, [0., 1., 2.], [1., 3., 2.], **kwargs)


@pytest.mark.parametrize("p", [np.array([[.1], [.2], [.8], [.9]]), [.1, .2], [.1, .2, np.nan, .9], [-.1, .2, .8, .9]])
def test_decision_curve_rejects_ambiguous_or_invalid_probabilities(p):
    with pytest.raises(ValueError):
        decision_curve([0, 0, 1, 1], p)


@pytest.mark.parametrize("thresholds", [[0], [1], [-.1], [np.nan], []])
def test_decision_curve_rejects_invalid_thresholds(thresholds):
    with pytest.raises(ValueError):
        decision_curve([0, 0, 1, 1], [.1, .2, .8, .9], thresholds)


def test_decision_curve_hand_computation_and_target_prevalence():
    y, p = [0, 0, 1, 1], [.1, .2, .8, .9]
    assert decision_curve(y, p, [.5]).net_benefit_model.iloc[0] == .5
    adjusted = decision_curve(y, p, [.5], prevalence=.1)
    assert adjusted.net_benefit_model.iloc[0] == .1
    assert adjusted.net_benefit_all.iloc[0] == pytest.approx(-.8)
    with pytest.raises(ValueError, match="binary"):
        decision_curve([-1, -1, 1, 1], p)


def test_delong_matches_hand_auc_with_ties_and_validates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    a, b = np.array([0, 1, 1, 1, 2, 3]), np.array([0, 2, 1, 1, 1, 2])
    r = delong_test(y, a, b)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, a))
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, b))
    assert delong_test(y, a, a)["p_value"] == 1
    with pytest.raises(ValueError, match="two subjects"):
        delong_test([0, 1, 1], [0, 1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="binary"):
        delong_test(y*2, a, b)


class NumericFitSpy(ClassifierMixin, BaseEstimator):
    matrices = []
    def fit(self, X, y):
        type(self).matrices.append(np.asarray(X).copy())
        self.classes_ = np.array([0, 1])
        return self
    def predict_proba(self, X):
        return np.tile([.5, .5], (len(X), 1))


def test_diagnostic_imputation_and_preprocessing_fit_only_training_rows():
    X = pd.DataFrame({"f": [np.nan, 2., 4., 6., 1000., 2000., 3000., 4000.]})
    batch = np.tile([0, 1], 4)
    folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
    NumericFitSpy.matrices, FitSpy.rows = [], []
    result = batch_leakage_check(X, batch, cv=folds, min_n=2,
                                 model=NumericFitSpy(), preprocessor=FitSpy())
    assert NumericFitSpy.matrices[0][0, 0] == 4.  # mean of 2,4,6, not the test batch
    assert FitSpy.rows == [set(range(4)), set(range(4, 8))]
    assert result.attrs["covered"].all()


def test_diagnostic_detects_nonlinear_site_information_with_multiple_probes():
    X = np.random.default_rng(4).normal(size=(600, 2))
    batch = (X[:, 0]*X[:, 1] > 0).astype(int)
    probes = {"linear": LogisticRegression(), "forest": RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0)}
    result = batch_leakage_check(X, batch, cv=3, models=probes)
    macro = result[result.batch == "MACRO"].set_index("probe").auroc
    assert macro["forest"] > .97 and macro["linear"] < .65


def test_pls_string_class_weights_schema_multiclass_and_serialization():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = np.where(X.a > .4, "PD", np.where(X.a < -.4, "HC", "other"))
    h = PLSDAClassifier(n_components=3, class_weight={"PD": 2., "HC": 1., "other": 1.}).fit(X, y)
    assert h.predict_proba(X).shape == (120, 3)
    np.testing.assert_allclose(h.predict_proba(X), h.predict_proba(X[list("fedcba")]))
    np.testing.assert_allclose(pickle.loads(pickle.dumps(h)).predict_proba(X), h.predict_proba(X))
    with pytest.raises(ValueError, match="schema"):
        h.predict_proba(X.rename(columns={"a": "typo"}))
    with pytest.raises(ValueError, match="n_components"):
        PLSDAClassifier(n_components=0).fit(X, y)
    with pytest.raises(ValueError, match="nonconstant"):
        PLSDAClassifier().fit(np.zeros((120, 6)), y)


def test_unseen_batch_rejection_is_reported_and_enforced():
    X = sites()
    new = X.iloc[:5].assign(site="new")
    h = ComBatHarmonizer("site").fit(X)
    assert h.adjustment_report(new).status.eq("unseen_batch_rejected").all()
    with pytest.raises(ValueError, match="Unseen batch"):
        h.transform(new)
    block = BlockwiseHarmonizer({"m": {"features": ["f0", "f1"], "batch": "site"}},
                                unknown_batch="raise").fit(X)
    assert block.adjustment_report(new).m.eq("unseen_batch_rejected").all()
    with pytest.raises(ValueError, match="Unseen batch"):
        block.transform(new)


def test_paired_randomization_matches_exact_small_sample_and_clusters():
    from itertools import product

    y = np.zeros(5)
    a, b = np.array([1., 2., 3., 4., 5.]), np.array([4., 3., 2., 1., 1.])
    observed = mean_squared_error(y, a) - mean_squared_error(y, b)
    differences = []
    for swap in product([False, True], repeat=5):
        differences.append(mean_squared_error(y, np.where(swap, b, a))
                           - mean_squared_error(y, np.where(swap, a, b)))
    exact_p = np.mean(np.abs(differences) >= abs(observed) - 1e-12)
    result = paired_bootstrap_diff(mean_squared_error, y, a, b, n_boot=999)
    assert result["p_value"] == pytest.approx(exact_p, abs=.05)
    # Repeated visits must swap together as well as being bootstrapped together.
    repeated = paired_bootstrap_diff(mean_squared_error, np.repeat(y, 3), np.repeat(a, 3),
                                     np.repeat(b, 3), n_boot=999, groups=np.repeat(np.arange(5), 3))
    for key in ("p_value", "ci_lo", "ci_hi", "diff"):
        assert repeated[key] == pytest.approx(result[key])
