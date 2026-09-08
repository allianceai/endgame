"""Tests for ComBatHarmonizer."""

import numpy as np
import pandas as pd
import pytest

from endgame.preprocessing.harmonization import ComBatHarmonizer


def _make_sites(n_per_site=(40, 30, 25), n_features=6, seed=0):
    """Features with a covariate effect plus additive/multiplicative site effects."""
    rng = np.random.RandomState(seed)
    rows = []
    for s, n in enumerate(n_per_site):
        age = rng.uniform(50, 80, n)
        sex = rng.choice(["F", "M"], n)
        base = rng.randn(n, n_features) + 0.05 * age[:, None] + 0.3 * (sex == "M")[:, None]
        shifted = base * (1 + 0.4 * s) + 2.0 * s  # scale + location site effect
        df = pd.DataFrame(shifted, columns=[f"f{i}" for i in range(n_features)])
        df["site"], df["age"], df["sex"] = f"site{s}", age, sex
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_removes_site_effect_and_keeps_covariates():
    df = _make_sites()
    h = ComBatHarmonizer(batch="site", covariates=["age"], categorical=["sex"])
    out = h.fit_transform(df)
    assert "site" not in out.columns and list(out[["age", "sex"]].columns) == ["age", "sex"]
    # site means of the harmonized residuals (after covariate effect) should be close
    site_means = out.groupby(df["site"])[h.features_].mean()
    assert site_means.std(axis=0).max() < 0.35  # raw data has ~2-4 unit site gaps
    raw_site_means = df.groupby("site")[h.features_].mean()
    assert raw_site_means.std(axis=0).min() > 1.0
    # age effect still present
    assert np.corrcoef(out["age"], out["f0"])[0, 1] > 0.2


def test_transform_matches_fit_transform_and_unknown_batch():
    df = _make_sites()
    h = ComBatHarmonizer(batch="site", covariates=["age"]).fit(df)
    np.testing.assert_allclose(h.transform(df)[h.features_], h.fit_transform(df)[h.features_])
    new = df.iloc[:5].copy()
    new["site"] = "site99"
    with pytest.raises(ValueError):
        h.transform(new)
    h2 = ComBatHarmonizer(batch="site", covariates=["age"], unknown_batch="passthrough").fit(df)
    out = h2.transform(new)
    np.testing.assert_allclose(out[h2.features_].to_numpy(), new[h2.features_].to_numpy())


def test_matches_neurocombat_reference():
    neuroCombat = pytest.importorskip("neuroCombat")
    df = _make_sites()
    h = ComBatHarmonizer(batch="site", covariates=["age"], categorical=["sex"])
    ours = h.fit_transform(df)[h.features_].to_numpy()
    ref = neuroCombat.neuroCombat(
        dat=df[h.features_].to_numpy().T,
        covars=df[["site", "age", "sex"]],
        batch_col="site",
        categorical_cols=["sex"],
        continuous_cols=["age"],
    )["data"].T
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-3)


def test_mean_only_and_no_eb_run():
    df = _make_sites()
    for kw in ({"mean_only": True}, {"eb": False}, {"eb": False, "mean_only": True}):
        out = ComBatHarmonizer(batch="site", **kw).fit_transform(df)
        assert np.isfinite(out[[c for c in out.columns if c.startswith("f")]].to_numpy()).all()


def test_input_validation():
    df = _make_sites()
    with pytest.raises(ValueError):
        ComBatHarmonizer(batch="nope").fit(df)
    bad = df.copy()
    bad.loc[0, "f0"] = np.nan
    with pytest.raises(ValueError):
        ComBatHarmonizer(batch="site").fit(bad)
    tiny = pd.concat([df, df.iloc[[0]].assign(site="lonely")], ignore_index=True)
    with pytest.raises(ValueError):
        ComBatHarmonizer(batch="site").fit(tiny)
    nan_batch = df.copy()
    nan_batch.loc[0, "site"] = None
    with pytest.raises(ValueError, match="missing"):
        ComBatHarmonizer(batch="site").fit(nan_batch)


def test_constant_feature_passes_through_without_leaking_batch():
    df = _make_sites()
    df["zeros"] = 0.0
    df["almost"] = 1e-9 * np.arange(len(df)) % 3
    h = ComBatHarmonizer(batch="site", covariates=["age"]).fit(df)
    out = h.fit_transform(df)
    assert h.constant_mask_[h.features_.index("zeros")]
    np.testing.assert_array_equal(out["zeros"].to_numpy(), df["zeros"].to_numpy())
    assert np.isfinite(out[h.features_].to_numpy()).all()
    # the real features are still harmonized
    assert out.groupby(df["site"])["f0"].mean().std() < df.groupby("site")["f0"].mean().std()
