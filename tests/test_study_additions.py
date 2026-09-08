import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from endgame.preprocessing.blocks import BlockwiseHarmonizer, MissingBlockIndicator
from endgame.utils.diagnostics import batch_leakage_check
from endgame.utils.metrics import decision_curve, delong_test, paired_bootstrap_diff


def _scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.8, n)          # AUROC ~0.8
    weak = y + rng.normal(0, 2.5, n)          # AUROC ~0.6
    return y, good, weak


def test_paired_bootstrap_and_delong_agree_on_direction():
    y, good, weak = _scores()
    pb = paired_bootstrap_diff(roc_auc_score, y, good, weak, n_boot=300)
    assert pb["diff"] > 0.1 and pb["ci_lo"] > 0 and pb["p_value"] < 0.05
    dl = delong_test(y, good, weak)
    assert abs(dl["auc_a"] - roc_auc_score(y, good)) < 1e-9 and abs(dl["auc_b"] - roc_auc_score(y, weak)) < 1e-9
    assert dl["diff"] > 0.1 and dl["p_value"] < 0.01 and dl["ci_lo"] > 0
    same = delong_test(y, good, good)
    assert abs(same["diff"]) < 1e-12 and same["p_value"] > 0.99


def test_decision_curve_shape_and_limits():
    y, good, _ = _scores()
    p = 1 / (1 + np.exp(-good))
    dc = decision_curve(y, p, thresholds=[0.1, 0.3, 0.5, 0.7])
    assert list(dc.columns) == ["threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none", "fraction_flagged"]
    assert (dc["net_benefit_none"] == 0).all() and dc["fraction_flagged"].is_monotonic_decreasing
    assert abs(dc.loc[dc.threshold == 0.5, "net_benefit_all"].item() - (y.mean() - (1 - y.mean()))) < 1e-9
    assert dc["net_benefit_model"].iloc[2] > dc["net_benefit_all"].iloc[2]      # a useful model beats treat-all at prevalence


def _blocks_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age": rng.normal(60, 8, n), "sex": rng.integers(0, 2, n)})
    df["scanner_batch"] = rng.choice(["A", "B", "C"], n)
    df["dwi_batch"] = rng.choice(["s1", "s2"], n)
    for i in range(3):
        df[f"vol_{i}"] = rng.normal(0, 1, n) + df["scanner_batch"].map({"A": 0.0, "B": 1.0, "C": -1.0}) + 0.01 * df["age"]
    for i in range(2):
        df[f"dwi_{i}"] = rng.normal(0, 1, n) + df["dwi_batch"].map({"s1": 0.0, "s2": 2.0})
    df.loc[:49, ["dwi_0", "dwi_1", "dwi_batch"]] = np.nan          # 50 subjects without diffusion
    return df


def test_blockwise_harmonizer_removes_each_blocks_batch_effect_and_passes_missing_rows():
    df = _blocks_data()
    blocks = {"t1": {"features": ["vol_0", "vol_1", "vol_2"], "batch": "scanner_batch"}, "dwi": {"features": ["dwi_0", "dwi_1"], "batch": "dwi_batch"}}
    h = BlockwiseHarmonizer(blocks, covariates=["age"]).fit(df.iloc[:200])
    out = h.transform(df)
    assert list(out.columns) == list(df.columns)
    # diffusion batch effect gone on harmonised rows, structural too
    has = out["dwi_batch"].notna()
    gap = out.loc[has].groupby("dwi_batch")["dwi_0"].mean()
    assert abs(gap["s1"] - gap["s2"]) < 0.4
    gap_t1 = out.groupby("scanner_batch")["vol_0"].mean()
    assert gap_t1.max() - gap_t1.min() < 0.4
    assert out.loc[~has, "dwi_0"].isna().all()                                   # missing block untouched
    before = df.groupby("dwi_batch")["dwi_0"].mean()
    assert abs(before["s1"] - before["s2"]) > 1.5


def test_missing_block_indicator():
    df = _blocks_data()
    m = MissingBlockIndicator({"dwi": ["dwi_0", "dwi_1"], "t1": ["vol_0"]}).fit(df.iloc[100:])
    out = m.transform(df)
    assert out["dwi_missing"].iloc[:50].eq(1).all() and out["dwi_missing"].iloc[50:].eq(0).all() and out["t1_missing"].eq(0).all()
    assert out["dwi_0"].notna().all() and abs(out["dwi_0"].iloc[0] - df["dwi_0"].iloc[100:].mean()) < 1e-9


def test_batch_leakage_check_detects_and_clears():
    df = _blocks_data()
    raw = batch_leakage_check(df[["vol_0", "vol_1", "vol_2"]], df["scanner_batch"], cv=3)
    assert raw.loc[raw.batch == "MACRO", "auroc"].item() > 0.75
    h = BlockwiseHarmonizer({"t1": {"features": ["vol_0", "vol_1", "vol_2"], "batch": "scanner_batch"}}).fit(df)
    harm = batch_leakage_check(h.transform(df)[["vol_0", "vol_1", "vol_2"]], df["scanner_batch"], cv=3)
    assert harm.loc[harm.batch == "MACRO", "auroc"].item() < raw.loc[raw.batch == "MACRO", "auroc"].item() - 0.15
