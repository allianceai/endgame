"""Small deterministic offline fixtures; never a financial backtest or search."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from endgame.ranking import GroupedRanker, group_percentiles, group_rank_average, group_rank_diagnostics
from endgame.validation import PurgedPanelSplit, purged_panel_oof


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def forbidden(*a, **kw):
        raise AssertionError('offline unit tests cannot connect to the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def panel():
    groups = np.repeat(np.arange(20), 10)
    x = np.tile(np.linspace(-1, 1, 10), 20)
    X = pd.DataFrame({'signal': x, 'noise': np.sin(np.arange(len(x))), 'missing': np.nan})
    return X, x + groups / 10, groups


def test_panel_split_purges_label_ends_and_keeps_dates_whole():
    X, y, t = panel()
    order = np.random.default_rng(42).permutation(len(X))
    t, end = t[order], (t + 2)[order]
    seen = set()
    cv = PurgedPanelSplit(n_splits=3, train_fraction=.7)
    folds = list(cv.split(X.iloc[order], groups=t, label_end_times=end))
    assert len(folds) == cv.get_n_splits()
    for tr, va in folds:
        assert end[tr].max() < t[va].min()
        assert not set(t[tr]) & set(t[va])
        for group in set(t[tr]):
            assert set(np.flatnonzero(t == group)) <= set(tr)
        for group in set(t[va]):
            assert set(np.flatnonzero(t == group)) <= set(va)
        assert not seen & set(va)
        seen.update(va)


def test_long_label_purges_entire_date_and_rolling_window():
    X, y, t = panel()
    ends = t + 1
    ends[10] = 100
    folds = list(PurgedPanelSplit(n_splits=2, train_fraction=.8, max_train_groups=4).split(X, groups=t, label_end_times=ends))
    for tr, _ in folds:
        assert 1 not in t[tr] and len(set(t[tr])) == 4


def test_datetime_panel_split_and_missing_dates():
    X, y, t = panel()
    dates = np.datetime64('2020-01-01') + t.astype('timedelta64[D]')
    for tr, va in PurgedPanelSplit().split(X, groups=dates, label_end_times=dates + np.timedelta64(2, 'D')):
        assert dates[tr].max() + np.timedelta64(2, 'D') < dates[va].min()
    dates[0] = np.datetime64('NaT')
    with pytest.raises(ValueError, match='missing'):
        list(PurgedPanelSplit().split(X, groups=dates, label_end_times=dates))


@pytest.mark.parametrize('options', [{'n_splits': 0}, {'n_splits': 1.5}, {'train_fraction': 1}, {'n_splits': 100}, {'max_train_groups': 0}])
def test_bad_splits_fail(options):
    X, y, t = panel()
    with pytest.raises(ValueError):
        list(PurgedPanelSplit(**options).split(X, groups=t, label_end_times=t + 1))


def test_missing_end_times_and_empty_purge_fail():
    X, y, t = panel()
    with pytest.raises(ValueError, match='explicit'):
        list(PurgedPanelSplit().split(X, groups=t))
    with pytest.raises(ValueError, match='empty fold'):
        list(PurgedPanelSplit().split(X, groups=t, label_end_times=t + 100))


def test_oof_cold_start_is_nan_and_future_labels_do_not_change_predictions():
    X, y, t = panel()
    model = GroupedRanker(alpha=1.)
    a = purged_panel_oof(model, X, y, t, t + 2, fit_groups=True)
    changed = y.copy()
    changed[t == t.max()] = -1000
    b = purged_panel_oof(model, X, changed, t, t + 2, fit_groups=True)
    assert a.covered.sum() == 50
    assert np.isnan(a.predictions[~a.covered]).all()
    assert np.all(a.fold_id[~a.covered] == -1)
    np.testing.assert_allclose(a.predictions, b.predictions, equal_nan=True)
    assert not hasattr(model, 'model_')


def test_oof_rejects_shuffled_cv():
    from sklearn.model_selection import KFold
    X, y, t = panel()
    with pytest.raises(ValueError, match='PurgedPanelSplit'):
        purged_panel_oof(Ridge(), X, y, t, t, cv=KFold(3))


def test_group_percentiles_ties_constant_and_alignment():
    scores = group_percentiles([5, 5, 1, 4, 3, 9], ['a', 'a', 'b', 'b', 'b', 'c'])
    np.testing.assert_allclose(scores, [.5, .5, 0., 1., .5, .5])
    with pytest.raises(ValueError):
        group_percentiles([np.nan], ['a'])
    with pytest.raises(ValueError):
        group_percentiles([1], [None])


def test_ridge_learns_order_not_between_group_market_level():
    X, y, g = panel()
    model = GroupedRanker(alpha=1.).fit(X, y, groups=g)
    assert model.score(X, y, groups=g) > .95
    shifted = clone(model).fit(X, y + g * 1000, groups=g)
    np.testing.assert_allclose(model.predict(X), shifted.predict(X), atol=1e-10)
    assert model.imputer_.statistics_[-1] == 0.
    assert model.get_params()['n_jobs'] == 1


def test_group_weighting_and_prediction_schema_are_explicit():
    X, y, g = panel()
    keep = (g != 0) | (np.arange(len(g)) < 3)
    X, y, g = X.loc[keep], y[keep], g[keep]
    model = GroupedRanker().fit(X, y, groups=g)
    masses = [model.training_weight_[g == q].sum() for q in np.unique(g)]
    np.testing.assert_allclose(masses, masses[0])
    np.testing.assert_allclose(model.predict(X), model.predict(X[['missing', 'noise', 'signal']]))
    with pytest.raises(ValueError, match='schema'):
        model.predict(X.drop(columns='noise'))


@pytest.mark.parametrize('options', [{'backend': 'unknown'}, {'focus': 'bottom'}, {'n_jobs': -1}, {'n_bins': 1}, {'alpha': 0}, {'colsample_bytree': 0}])
def test_invalid_rank_options_fail(options):
    X, y, g = panel()
    with pytest.raises(ValueError):
        GroupedRanker(**options).fit(X, y, groups=g)


@pytest.mark.parametrize('focus', ['top', 'bottom'])
def test_tiny_lambdamart_groups_and_output_direction(focus):
    pytest.importorskip('lightgbm')
    X, y, g = panel()
    model = GroupedRanker(backend='lambdamart', focus=focus, n_estimators=8, num_leaves=4,
                          min_child_samples=3, colsample_bytree=1., n_jobs=1).fit(X, y, groups=g)
    assert model.score(X, y, groups=g) > .8
    assert model.training_relevance_.dtype.kind in 'iu'
    assert model.training_relevance_.min() >= 0 and model.training_relevance_.max() < model.n_bins
    assert model.group_sizes_.sum() == len(X)
    assert model.model_.get_params()['label_gain'] == list(range(5))
    if focus == 'bottom':
        assert model.training_relevance_[0] > model.training_relevance_[9]
        assert model.predict(X)[0] < model.predict(X)[9]


def test_ranker_serializes_and_clones(tmp_path):
    import joblib
    X, y, g = panel()
    model = GroupedRanker().fit(X, y, groups=g)
    path = tmp_path / 'ranker.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(X), model.predict(X))
    assert not hasattr(clone(model), 'model_')


def test_lambdamart_reorders_interleaved_queries_features_labels_and_weights(monkeypatch):
    lightgbm = pytest.importorskip('lightgbm')
    captured = {}
    class QuerySpy:
        def __init__(self, **kw):
            pass
        def fit(self, X, y, **kw):
            captured.update(X=X, y=y, **kw)
            return self
    monkeypatch.setattr(lightgbm, 'LGBMRanker', QuerySpy)
    groups = np.array(['b', 'a', 'b', 'a', 'b'])
    y = np.array([2., 9., 4., 3., 6.])
    X = np.arange(10.).reshape(5, 2)
    model = GroupedRanker(backend='lambdamart').fit(X, y, groups)
    order = [1, 3, 0, 2, 4]
    assert captured['group'] == [2, 3]
    np.testing.assert_array_equal(captured['X'], X[order])
    np.testing.assert_array_equal(captured['y'], model.training_relevance_[order])
    np.testing.assert_array_equal(captured['sample_weight'], model.training_weight_[order])


def test_rank_average_is_grouped_fixed_and_missing_scores_fail():
    preds = np.array([[1, 10], [2, 5], [100, 1000], [90, 2000.]])
    actual = group_rank_average(preds, ['a', 'a', 'b', 'b'], [.75, .25])
    np.testing.assert_allclose(actual, [.25, .75, .75, .25])
    preds[0, 0] = np.nan
    with pytest.raises(ValueError):
        group_rank_average(preds, ['a', 'a', 'b', 'b'])


def test_tail_diagnostics_do_not_manufacture_spread_on_ties():
    rows = group_rank_diagnostics([1, 2, 3, 4], [0, 0, 0, 0], [1, 1, 1, 1])
    assert rows[0]['rank_ic'] is None
    assert rows[0]['top_minus_bottom'] == 0
    assert rows[0]['top_n'] == rows[0]['bottom_n'] == 4
