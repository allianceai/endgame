"""Tests for Sharpe Ratio utilities and Deflated Sharpe Ratio."""

import pytest
import numpy as np
from scipy import stats

from endgame.utils.sharpe import (
    sharpe_ratio,
    sharpe_ratio_std,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    analyze_sharpe,
    minimum_track_record_length,
    haircut_sharpe_ratio,
    estimate_n_independent_trials,
    multiple_testing_summary,
    SharpeAnalysis,
    EULER_MASCHERONI,
)


class TestSharpeRatio:
    """Tests for basic Sharpe ratio calculation."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio for positive returns."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001  # Positive mean
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio for negative returns."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 - 0.001  # Negative mean
        sr = sharpe_ratio(returns)
        assert sr < 0

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.ones(100) * 0.001
        sr = sharpe_ratio(returns)
        # With zero volatility, SR is 0 (handled by code)
        assert sr == 0.0 or not np.isfinite(sr)

    def test_sharpe_ratio_annualization(self):
        """Test that annualization factor works correctly."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005

        sr_daily = sharpe_ratio(returns, annualization_factor=1.0)
        sr_annual = sharpe_ratio(returns, annualization_factor=252.0)

        # Annual SR should be sqrt(252) times daily SR
        assert np.isclose(sr_annual, sr_daily * np.sqrt(252), rtol=0.01)

    def test_sharpe_ratio_risk_free_rate(self):
        """Test risk-free rate adjustment."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        sr_no_rf = sharpe_ratio(returns, risk_free_rate=0.0)
        sr_with_rf = sharpe_ratio(returns, risk_free_rate=0.0005)

        # SR should be lower with positive risk-free rate
        assert sr_with_rf < sr_no_rf


class TestSharpeRatioStd:
    """Tests for Sharpe ratio standard error."""

    def test_std_decreases_with_n(self):
        """Test that SE decreases with more observations."""
        sr = 1.0
        std_100 = sharpe_ratio_std(sr, n_obs=100)
        std_400 = sharpe_ratio_std(sr, n_obs=400)

        # SE should be ~half with 4x observations
        assert std_400 < std_100
        assert np.isclose(std_400, std_100 / 2, rtol=0.1)

    def test_std_increases_with_kurtosis(self):
        """Test that SE increases with fat tails."""
        sr = 1.0
        n = 252
        std_normal = sharpe_ratio_std(sr, n, kurtosis=3.0)  # Normal
        std_fat = sharpe_ratio_std(sr, n, kurtosis=6.0)  # Fat tails

        assert std_fat > std_normal

    def test_std_non_negative(self):
        """Test that SE is non-negative."""
        for sr in [-1.0, 0.0, 1.0, 2.0]:
            for n in [50, 100, 252]:
                std = sharpe_ratio_std(sr, n)
                assert std >= 0


class TestProbabilisticSharpeRatio:
    """Tests for Probabilistic Sharpe Ratio."""

    def test_psr_high_sharpe(self):
        """Test PSR for high Sharpe ratio."""
        psr = probabilistic_sharpe_ratio(
            sharpe=2.0, benchmark_sharpe=0.0, n_obs=252
        )
        # High SR should have PSR near 1
        assert psr > 0.99

    def test_psr_low_sharpe(self):
        """Test PSR for low Sharpe ratio."""
        psr = probabilistic_sharpe_ratio(
            sharpe=0.1, benchmark_sharpe=0.0, n_obs=50
        )
        # Low SR with short track record should have moderate PSR
        assert 0.3 < psr < 0.8

    def test_psr_negative_sharpe(self):
        """Test PSR for negative Sharpe ratio."""
        psr = probabilistic_sharpe_ratio(
            sharpe=-1.0, benchmark_sharpe=0.0, n_obs=252
        )
        # Negative SR should have PSR near 0
        assert psr < 0.01

    def test_psr_equals_benchmark(self):
        """Test PSR when SR equals benchmark."""
        psr = probabilistic_sharpe_ratio(
            sharpe=1.0, benchmark_sharpe=1.0, n_obs=252
        )
        # Should be ~0.5 when equal to benchmark
        assert np.isclose(psr, 0.5, atol=0.05)

    def test_psr_increases_with_n(self):
        """Test that PSR increases with track record length."""
        psr_short = probabilistic_sharpe_ratio(
            sharpe=0.5, benchmark_sharpe=0.0, n_obs=50
        )
        psr_long = probabilistic_sharpe_ratio(
            sharpe=0.5, benchmark_sharpe=0.0, n_obs=500
        )
        assert psr_long > psr_short


class TestExpectedMaxSharpe:
    """Tests for expected maximum Sharpe ratio."""

    def test_e_max_increases_with_trials(self):
        """Test that E[max] increases with number of trials."""
        e_max_10 = expected_max_sharpe(n_trials=10, sharpe_std=0.5)
        e_max_100 = expected_max_sharpe(n_trials=100, sharpe_std=0.5)
        e_max_1000 = expected_max_sharpe(n_trials=1000, sharpe_std=0.5)

        assert e_max_1000 > e_max_100 > e_max_10

    def test_e_max_increases_with_std(self):
        """Test that E[max] increases with SR variance."""
        e_max_low = expected_max_sharpe(n_trials=100, sharpe_std=0.3)
        e_max_high = expected_max_sharpe(n_trials=100, sharpe_std=0.7)

        assert e_max_high > e_max_low

    def test_e_max_single_trial(self):
        """Test E[max] with single trial."""
        e_max = expected_max_sharpe(n_trials=1, sharpe_std=0.5, mean_sharpe=0.0)
        assert e_max == 0.0

    def test_e_max_with_nonzero_mean(self):
        """Test E[max] with non-zero mean."""
        e_max_zero = expected_max_sharpe(n_trials=100, sharpe_std=0.5, mean_sharpe=0.0)
        e_max_positive = expected_max_sharpe(n_trials=100, sharpe_std=0.5, mean_sharpe=0.5)

        assert e_max_positive > e_max_zero
        assert np.isclose(e_max_positive - e_max_zero, 0.5, rtol=0.01)

    def test_e_max_known_values(self):
        """Test E[max] against known approximate values."""
        # For N=100, std=0.5, E[max] ≈ 1.27
        e_max = expected_max_sharpe(n_trials=100, sharpe_std=0.5)
        assert 1.1 < e_max < 1.4


class TestDeflatedSharpeRatio:
    """Tests for Deflated Sharpe Ratio."""

    def test_dsr_high_sharpe_few_trials(self):
        """Test DSR for high SR with few trials."""
        dsr = deflated_sharpe_ratio(
            sharpe=2.0,
            n_trials=5,
            sharpe_std_trials=0.5,
            n_obs=252,
        )
        # High SR with few trials should still be significant
        assert dsr > 0.9

    def test_dsr_moderate_sharpe_many_trials(self):
        """Test DSR for moderate SR with many trials."""
        dsr = deflated_sharpe_ratio(
            sharpe=1.0,
            n_trials=100,
            sharpe_std_trials=0.5,
            n_obs=252,
        )
        # Moderate SR after many trials may not be significant
        assert dsr < 0.5

    def test_dsr_decreases_with_trials(self):
        """Test that DSR decreases with more trials."""
        dsr_10 = deflated_sharpe_ratio(
            sharpe=1.5, n_trials=10, sharpe_std_trials=0.5, n_obs=252
        )
        dsr_100 = deflated_sharpe_ratio(
            sharpe=1.5, n_trials=100, sharpe_std_trials=0.5, n_obs=252
        )
        assert dsr_10 > dsr_100

    def test_dsr_single_trial_equals_psr(self):
        """Test that DSR with 1 trial equals PSR (SR > 0)."""
        sr = 1.0
        n_obs = 252

        dsr = deflated_sharpe_ratio(
            sharpe=sr, n_trials=1, sharpe_std_trials=0.5, n_obs=n_obs
        )
        psr = probabilistic_sharpe_ratio(
            sharpe=sr, benchmark_sharpe=0.0, n_obs=n_obs
        )
        # Should be close (E[max] with 1 trial = mean = 0)
        assert np.isclose(dsr, psr, rtol=0.01)

    def test_dsr_range(self):
        """Test that DSR is in [0, 1]."""
        for sr in [-1.0, 0.0, 1.0, 2.0, 3.0]:
            for n_trials in [1, 10, 100]:
                dsr = deflated_sharpe_ratio(
                    sharpe=sr, n_trials=n_trials, sharpe_std_trials=0.5, n_obs=252
                )
                assert 0 <= dsr <= 1


class TestAnalyzeSharpe:
    """Tests for comprehensive Sharpe analysis."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return np.random.randn(252) * 0.01 + 0.0005

    def test_analyze_returns_dataclass(self, sample_returns):
        """Test that analyze_sharpe returns correct type."""
        analysis = analyze_sharpe(sample_returns)
        assert isinstance(analysis, SharpeAnalysis)

    def test_analyze_all_fields_present(self, sample_returns):
        """Test that all fields are populated."""
        analysis = analyze_sharpe(sample_returns, n_trials=10)

        assert analysis.sharpe_ratio is not None
        assert analysis.probabilistic_sharpe is not None
        assert analysis.deflated_sharpe is not None
        assert analysis.expected_max_sharpe is not None
        assert analysis.p_value is not None
        assert analysis.is_significant is not None
        assert analysis.n_trials == 10
        assert analysis.skewness is not None
        assert analysis.kurtosis is not None
        assert analysis.track_record_length == 252

    def test_analyze_significance(self, sample_returns):
        """Test significance determination."""
        # With few trials, moderate SR should be significant
        analysis_few = analyze_sharpe(
            sample_returns, n_trials=1, significance_level=0.05
        )

        # With many trials, it may not be
        analysis_many = analyze_sharpe(
            sample_returns, n_trials=1000, significance_level=0.05
        )

        # P-value should be higher with more trials
        assert analysis_many.p_value >= analysis_few.p_value

    def test_analyze_with_all_sharpes(self, sample_returns):
        """Test analysis with all Sharpe ratios provided."""
        all_sharpes = np.random.randn(50) * 0.5

        analysis = analyze_sharpe(
            sample_returns,
            n_trials=50,
            all_sharpes=all_sharpes,
        )

        assert analysis.n_trials == 50


class TestMinimumTrackRecordLength:
    """Tests for minimum track record length."""

    def test_mtrl_high_sharpe(self):
        """Test MinTRL for high Sharpe ratio."""
        n_min = minimum_track_record_length(sharpe=2.0)
        # High SR needs few observations
        assert n_min < 20

    def test_mtrl_low_sharpe(self):
        """Test MinTRL for low Sharpe ratio."""
        n_min = minimum_track_record_length(sharpe=0.25)
        # Low SR needs more observations than high SR
        n_min_high = minimum_track_record_length(sharpe=2.0)
        assert n_min > n_min_high

    def test_mtrl_decreases_with_sharpe(self):
        """Test that MinTRL decreases with higher SR."""
        mtrl_05 = minimum_track_record_length(sharpe=0.5)
        mtrl_10 = minimum_track_record_length(sharpe=1.0)
        mtrl_20 = minimum_track_record_length(sharpe=2.0)

        assert mtrl_05 > mtrl_10 > mtrl_20

    def test_mtrl_increases_with_confidence(self):
        """Test that MinTRL increases with higher confidence."""
        mtrl_90 = minimum_track_record_length(sharpe=1.0, confidence=0.90)
        mtrl_95 = minimum_track_record_length(sharpe=1.0, confidence=0.95)
        mtrl_99 = minimum_track_record_length(sharpe=1.0, confidence=0.99)

        assert mtrl_99 > mtrl_95 > mtrl_90

    def test_mtrl_sr_below_benchmark(self):
        """Test MinTRL when SR below benchmark."""
        n_min = minimum_track_record_length(sharpe=0.5, benchmark_sharpe=1.0)
        assert n_min == np.inf


class TestHaircutSharpeRatio:
    """Tests for Sharpe ratio haircut."""

    def test_haircut_positive_adjustment(self):
        """Test haircut makes adjustment."""
        adj_sr, haircut = haircut_sharpe_ratio(sharpe=2.0, n_trials=100)

        assert adj_sr < 2.0  # Adjusted SR is lower
        assert 0 < haircut < 1  # Haircut is between 0 and 100%

    def test_haircut_increases_with_trials(self):
        """Test haircut increases with more trials."""
        _, haircut_10 = haircut_sharpe_ratio(sharpe=2.0, n_trials=10)
        _, haircut_100 = haircut_sharpe_ratio(sharpe=2.0, n_trials=100)

        assert haircut_100 > haircut_10

    def test_haircut_can_be_negative(self):
        """Test adjusted SR can be negative."""
        adj_sr, haircut = haircut_sharpe_ratio(sharpe=0.5, n_trials=100)

        # If observed SR < E[max], adjusted SR is negative
        assert adj_sr < 0


class TestEstimateNIndependentTrials:
    """Tests for estimating effective number of trials."""

    def test_count_method(self):
        """Test count method returns raw count."""
        sharpes = np.random.randn(100)
        n_eff = estimate_n_independent_trials(sharpes, method="count")
        assert n_eff == 100

    def test_variance_method_correlated(self):
        """Test variance method with correlated strategies."""
        # Correlated strategies have low variance
        sharpes = np.random.randn(100) * 0.1  # Low variance

        n_eff = estimate_n_independent_trials(sharpes, method="variance")
        assert n_eff < 100

    def test_variance_method_independent(self):
        """Test variance method with independent strategies."""
        # Independent strategies have expected variance
        sharpes = np.random.randn(100) * 0.5  # Normal variance

        n_eff = estimate_n_independent_trials(sharpes, method="variance")
        assert n_eff >= 50  # Should be reasonable fraction

    def test_invalid_method(self):
        """Test invalid method raises error."""
        sharpes = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_n_independent_trials(sharpes, method="invalid")


class TestMultipleTestingSummary:
    """Tests for multiple testing summary."""

    def test_summary_has_expected_keys(self):
        """Test summary contains all expected keys."""
        sharpes = np.random.randn(50)
        summary = multiple_testing_summary(sharpes)

        expected_keys = [
            "n_trials", "n_effective", "sharpe_mean", "sharpe_std",
            "best_sharpe", "expected_max_sharpe", "best_dsr",
            "haircut_sharpe", "haircut_percent", "n_significant",
            "significance_level",
        ]
        for key in expected_keys:
            assert key in summary

    def test_summary_n_trials(self):
        """Test n_trials is correct."""
        sharpes = np.random.randn(100)
        summary = multiple_testing_summary(sharpes)
        assert summary["n_trials"] == 100

    def test_summary_best_sharpe(self):
        """Test best_sharpe is maximum."""
        sharpes = np.random.randn(50)
        summary = multiple_testing_summary(sharpes)
        assert np.isclose(summary["best_sharpe"], np.max(sharpes))

    def test_summary_significance_level(self):
        """Test significance level is configurable."""
        sharpes = np.random.randn(50)

        summary_05 = multiple_testing_summary(sharpes, significance_level=0.05)
        summary_01 = multiple_testing_summary(sharpes, significance_level=0.01)

        assert summary_05["significance_level"] == 0.05
        assert summary_01["significance_level"] == 0.01


class TestEdgeCases:
    """Edge case tests."""

    def test_euler_mascheroni_constant(self):
        """Test Euler-Mascheroni constant value."""
        assert np.isclose(EULER_MASCHERONI, 0.5772156649015329, rtol=1e-10)

    def test_empty_returns(self):
        """Test with empty returns array."""
        # Empty returns should return NaN (not crash)
        sr = sharpe_ratio(np.array([]))
        assert np.isnan(sr) or not np.isfinite(sr)

    def test_single_return(self):
        """Test with single return."""
        # Should not crash
        sr = sharpe_ratio(np.array([0.01]))
        # Result may be 0 or nan/inf

    def test_large_n_trials(self):
        """Test with very large number of trials."""
        e_max = expected_max_sharpe(n_trials=1000000, sharpe_std=0.5)
        # Should be finite and positive
        assert np.isfinite(e_max)
        assert e_max > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
