from __future__ import annotations

"""Sharpe Ratio utilities for backtesting and multiple testing correction.

This module implements the Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe
Ratio (PSR) from Bailey & López de Prado's work on addressing the multiple
testing problem in quantitative finance.

References
----------
- Bailey, D.H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  The Journal of Portfolio Management, 40(5), 94-107.
- López de Prado, M. (2018). "Advances in Financial Machine Learning."
  John Wiley & Sons, Chapter 14.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Euler-Mascheroni constant
EULER_MASCHERONI = 0.5772156649015329


@dataclass
class SharpeAnalysis:
    """Results from Sharpe ratio analysis.

    Attributes
    ----------
    sharpe_ratio : float
        The estimated Sharpe ratio.
    probabilistic_sharpe : float
        PSR - probability that true SR > benchmark.
    deflated_sharpe : float
        DSR - PSR adjusted for multiple testing.
    expected_max_sharpe : float
        Expected maximum SR under null hypothesis.
    p_value : float
        P-value for the null hypothesis that true SR = 0.
    is_significant : bool
        Whether DSR exceeds significance threshold.
    n_trials : int
        Number of trials considered.
    skewness : float
        Skewness of returns.
    kurtosis : float
        Excess kurtosis of returns.
    track_record_length : int
        Number of observations.
    """
    sharpe_ratio: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    is_significant: bool
    n_trials: int
    skewness: float
    kurtosis: float
    track_record_length: int


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Calculate the annualized Sharpe ratio.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    risk_free_rate : float, default=0.0
        Risk-free rate (same period as returns).
    annualization_factor : float, default=252.0
        Factor to annualize (252 for daily, 12 for monthly, 52 for weekly).

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Examples
    --------
    >>> returns = np.random.randn(252) * 0.01 + 0.0005  # Daily returns
    >>> sr = sharpe_ratio(returns)
    """
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0 or np.isclose(std_excess, 0, atol=1e-10):
        return 0.0

    # Annualize
    sr = (mean_excess / std_excess) * np.sqrt(annualization_factor)
    return sr


def sharpe_ratio_std(
    sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Calculate the standard error of the Sharpe ratio estimate.

    Uses the Lo (2002) / Mertens (2002) correction for non-normality.

    Parameters
    ----------
    sharpe : float
        Estimated Sharpe ratio.
    n_obs : int
        Number of observations.
    skewness : float, default=0.0
        Skewness of returns.
    kurtosis : float, default=3.0
        Kurtosis of returns (not excess kurtosis).

    Returns
    -------
    float
        Standard error of the Sharpe ratio.

    Notes
    -----
    The formula accounts for:
    - Sampling variability
    - Non-normal returns (skewness and fat tails)

    References
    ----------
    Lo, A. (2002). "The Statistics of Sharpe Ratios."
    Financial Analysts Journal, 58(4), 36-52.
    """
    # Excess kurtosis (kurtosis - 3 for normal)
    excess_kurt = kurtosis - 3.0

    # Variance of SR estimator under non-normality
    # Var(SR) = (1 - skew*SR + ((kurtosis-1)/4)*SR^2) / (n-1)
    variance = (
        1
        - skewness * sharpe
        + ((excess_kurt + 2) / 4) * sharpe ** 2
    ) / (n_obs - 1)

    return np.sqrt(max(variance, 0))


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Calculate the Probabilistic Sharpe Ratio (PSR).

    PSR is the probability that the true Sharpe ratio exceeds the benchmark,
    accounting for non-normality of returns.

    Parameters
    ----------
    sharpe : float
        Estimated Sharpe ratio.
    benchmark_sharpe : float
        Benchmark Sharpe ratio to compare against.
    n_obs : int
        Number of observations.
    skewness : float, default=0.0
        Skewness of returns.
    kurtosis : float, default=3.0
        Kurtosis of returns (not excess kurtosis).

    Returns
    -------
    float
        Probability in [0, 1] that true SR > benchmark SR.

    Examples
    --------
    >>> # Test if strategy beats SR = 0
    >>> psr = probabilistic_sharpe_ratio(sharpe=1.5, benchmark_sharpe=0,
    ...                                   n_obs=252, skewness=-0.2, kurtosis=4.0)
    >>> print(f"Probability true SR > 0: {psr:.2%}")

    Notes
    -----
    PSR corrects for:
    - Sample length (finite track record)
    - Non-normal returns (skewness and fat tails)

    It does NOT correct for multiple testing - use DSR for that.

    References
    ----------
    Bailey, D.H. and López de Prado, M. (2012). "The Sharpe Ratio
    Efficient Frontier." Journal of Risk, 15(2), 3-44.
    """
    # Standard error of SR
    sr_std = sharpe_ratio_std(sharpe, n_obs, skewness, kurtosis)

    if sr_std == 0:
        return 1.0 if sharpe > benchmark_sharpe else 0.0

    # Z-score
    z = (sharpe - benchmark_sharpe) / sr_std

    # PSR = Phi(z)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(
    n_trials: int,
    sharpe_std: float,
    mean_sharpe: float = 0.0,
) -> float:
    """Calculate expected maximum Sharpe ratio under null hypothesis.

    This is the expected maximum SR when all strategies have true SR = mean_sharpe,
    but we observe inflated values due to multiple testing.

    Parameters
    ----------
    n_trials : int
        Number of independent trials/strategies tested.
    sharpe_std : float
        Standard deviation of Sharpe ratio estimates across trials.
    mean_sharpe : float, default=0.0
        Mean Sharpe ratio under null (typically 0).

    Returns
    -------
    float
        Expected maximum Sharpe ratio E[max{SR_i}].

    Notes
    -----
    Uses the approximation from Bailey & López de Prado (2014):

    E[max{SR}] ≈ μ + σ * [(1-γ)*Φ^(-1)(1-1/N) + γ*Φ^(-1)(1-1/(N*e))]

    where γ is the Euler-Mascheroni constant.

    Examples
    --------
    >>> # After 100 trials, what SR do we expect by chance?
    >>> e_max = expected_max_sharpe(n_trials=100, sharpe_std=0.5)
    >>> print(f"Expected max SR: {e_max:.2f}")
    """
    if n_trials <= 1:
        return mean_sharpe

    gamma = EULER_MASCHERONI

    # Quantiles
    # Φ^(-1)(1 - 1/N)
    q1 = stats.norm.ppf(1 - 1 / n_trials)
    # Φ^(-1)(1 - 1/(N*e))
    q2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))

    # Expected maximum
    e_max = mean_sharpe + sharpe_std * ((1 - gamma) * q1 + gamma * q2)

    return e_max


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    sharpe_std_trials: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    mean_sharpe_null: float = 0.0,
) -> float:
    """Calculate the Deflated Sharpe Ratio (DSR).

    DSR corrects for multiple testing by computing the probability that the
    observed Sharpe ratio exceeds the expected maximum SR under the null
    hypothesis that all strategies have zero true SR.

    Parameters
    ----------
    sharpe : float
        Estimated Sharpe ratio of the selected strategy.
    n_trials : int
        Number of independent trials/strategies tested.
    sharpe_std_trials : float
        Standard deviation of Sharpe ratios across all trials.
    n_obs : int
        Number of observations (track record length).
    skewness : float, default=0.0
        Skewness of returns.
    kurtosis : float, default=3.0
        Kurtosis of returns (not excess kurtosis).
    mean_sharpe_null : float, default=0.0
        Mean Sharpe ratio under null hypothesis.

    Returns
    -------
    float
        Deflated Sharpe Ratio in [0, 1].

    Examples
    --------
    >>> # Tested 100 strategies, best has SR = 2.0
    >>> dsr = deflated_sharpe_ratio(
    ...     sharpe=2.0,
    ...     n_trials=100,
    ...     sharpe_std_trials=0.5,
    ...     n_obs=252,
    ...     skewness=-0.3,
    ...     kurtosis=4.5,
    ... )
    >>> print(f"DSR: {dsr:.2%}")
    >>> # If DSR < 0.95, the strategy may be a statistical fluke

    Notes
    -----
    DSR answers: "What is the probability that this strategy would have
    beaten random chance, given that we tested N strategies?"

    A DSR of 0.95 means there's a 95% probability that the strategy's
    performance is real and not due to overfitting from multiple testing.

    References
    ----------
    Bailey, D.H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    The Journal of Portfolio Management, 40(5), 94-107.
    """
    # Calculate expected max SR under null
    sr0 = expected_max_sharpe(n_trials, sharpe_std_trials, mean_sharpe_null)

    # DSR is just PSR with benchmark = expected max SR
    dsr = probabilistic_sharpe_ratio(
        sharpe=sharpe,
        benchmark_sharpe=sr0,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    return dsr


def analyze_sharpe(
    returns: np.ndarray,
    n_trials: int = 1,
    sharpe_std_trials: float | None = None,
    all_sharpes: np.ndarray | None = None,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
    significance_level: float = 0.05,
) -> SharpeAnalysis:
    """Comprehensive Sharpe ratio analysis with multiple testing correction.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns for the selected strategy.
    n_trials : int, default=1
        Number of independent trials/strategies tested.
    sharpe_std_trials : float, optional
        Standard deviation of Sharpe ratios across all trials.
        If not provided and all_sharpes is given, computed from all_sharpes.
        If neither provided, estimated as 1/sqrt(n_obs).
    all_sharpes : np.ndarray, optional
        Sharpe ratios of all tested strategies (for computing variance).
    risk_free_rate : float, default=0.0
        Risk-free rate (same period as returns).
    annualization_factor : float, default=252.0
        Factor to annualize Sharpe ratio.
    significance_level : float, default=0.05
        Significance level for hypothesis testing.

    Returns
    -------
    SharpeAnalysis
        Comprehensive analysis results.

    Examples
    --------
    >>> # Single strategy analysis
    >>> returns = np.random.randn(252) * 0.01 + 0.0005
    >>> analysis = analyze_sharpe(returns)
    >>> print(f"SR: {analysis.sharpe_ratio:.2f}")
    >>> print(f"PSR (SR > 0): {analysis.probabilistic_sharpe:.2%}")

    >>> # Multiple testing scenario
    >>> all_sharpes = np.random.randn(100) * 0.5  # 100 strategies tested
    >>> best_idx = np.argmax(all_sharpes)
    >>> analysis = analyze_sharpe(
    ...     returns=best_returns,
    ...     n_trials=100,
    ...     all_sharpes=all_sharpes,
    ... )
    >>> print(f"DSR: {analysis.deflated_sharpe:.2%}")
    >>> print(f"Significant: {analysis.is_significant}")
    """
    returns = np.asarray(returns)
    n_obs = len(returns)

    # Compute basic statistics
    excess_returns = returns - risk_free_rate
    skewness = float(stats.skew(excess_returns))
    kurtosis = float(stats.kurtosis(excess_returns, fisher=False))  # Not excess

    # Compute Sharpe ratio
    sr = sharpe_ratio(returns, risk_free_rate, annualization_factor)

    # Estimate SR standard deviation across trials
    if sharpe_std_trials is None:
        if all_sharpes is not None:
            sharpe_std_trials = float(np.std(all_sharpes, ddof=1))
        else:
            # Conservative estimate: assume SR ~ N(0, 1/sqrt(T))
            sharpe_std_trials = 1.0 / np.sqrt(n_obs) * np.sqrt(annualization_factor)

    # Compute PSR (probability true SR > 0)
    psr = probabilistic_sharpe_ratio(
        sharpe=sr,
        benchmark_sharpe=0.0,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Compute expected max SR under null
    e_max_sr = expected_max_sharpe(n_trials, sharpe_std_trials, 0.0)

    # Compute DSR
    dsr = deflated_sharpe_ratio(
        sharpe=sr,
        n_trials=n_trials,
        sharpe_std_trials=sharpe_std_trials,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # P-value (1 - DSR when testing against expected max)
    p_value = 1.0 - dsr

    return SharpeAnalysis(
        sharpe_ratio=sr,
        probabilistic_sharpe=psr,
        deflated_sharpe=dsr,
        expected_max_sharpe=e_max_sr,
        p_value=p_value,
        is_significant=dsr >= (1 - significance_level),
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis,
        track_record_length=n_obs,
    )


def minimum_track_record_length(
    sharpe: float,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """Calculate minimum track record length needed for statistical significance.

    Answers: "How many observations do we need to be confident that the
    strategy's Sharpe ratio is real?"

    Parameters
    ----------
    sharpe : float
        Target Sharpe ratio.
    benchmark_sharpe : float, default=0.0
        Benchmark to beat.
    confidence : float, default=0.95
        Required confidence level.
    skewness : float, default=0.0
        Expected skewness of returns.
    kurtosis : float, default=3.0
        Expected kurtosis of returns.

    Returns
    -------
    int
        Minimum number of observations needed.

    Examples
    --------
    >>> # How long to verify SR = 1.0 strategy?
    >>> n_min = minimum_track_record_length(sharpe=1.0)
    >>> print(f"Need at least {n_min} observations")

    Notes
    -----
    This is the "MinTRL" from Bailey & López de Prado (2012).

    A strategy with SR = 2.0 and normal returns needs only ~16 observations.
    A strategy with SR = 0.5 needs ~256 observations!
    """
    if sharpe <= benchmark_sharpe:
        return np.inf

    # z-score for desired confidence
    z = stats.norm.ppf(confidence)

    # Excess kurtosis
    excess_kurt = kurtosis - 3.0

    # Solve for n in the PSR formula
    # We need: (SR - SR_0) * sqrt(n-1) / sqrt(1 - skew*SR + ...) >= z
    # Rearranging: n >= 1 + (z / (SR - SR_0))^2 * (1 - skew*SR + ...)

    sr_diff = sharpe - benchmark_sharpe
    variance_factor = 1 - skewness * sharpe + ((excess_kurt + 2) / 4) * sharpe ** 2

    n_min = 1 + (z / sr_diff) ** 2 * variance_factor

    return int(np.ceil(n_min))


def haircut_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    sharpe_std_trials: float = 0.5,
) -> tuple[float, float]:
    """Apply haircut to Sharpe ratio for multiple testing.

    Returns an adjusted Sharpe ratio that accounts for data mining.

    Parameters
    ----------
    sharpe : float
        Observed Sharpe ratio.
    n_trials : int
        Number of strategies tested.
    sharpe_std_trials : float, default=0.5
        Standard deviation of SR estimates across trials.

    Returns
    -------
    Tuple[float, float]
        (haircut_sharpe, haircut_percent)
        - haircut_sharpe: Adjusted Sharpe ratio
        - haircut_percent: Percentage reduction applied

    Examples
    --------
    >>> sr_adj, haircut = haircut_sharpe_ratio(sharpe=2.0, n_trials=100)
    >>> print(f"Adjusted SR: {sr_adj:.2f} (haircut: {haircut:.1%})")

    Notes
    -----
    The haircut is the expected maximum SR under null hypothesis.
    The adjusted SR is: SR_adjusted = SR_observed - E[max{SR}|null]
    """
    e_max = expected_max_sharpe(n_trials, sharpe_std_trials, 0.0)

    haircut_sr = sharpe - e_max
    haircut_pct = e_max / sharpe if sharpe != 0 else 0.0

    return haircut_sr, haircut_pct


def estimate_n_independent_trials(
    sharpe_ratios: np.ndarray,
    method: str = "variance",
) -> int:
    """Estimate effective number of independent trials from correlated strategies.

    When strategies are correlated, the effective number of independent trials
    is less than the total number tested.

    Parameters
    ----------
    sharpe_ratios : np.ndarray
        Array of Sharpe ratios from all tested strategies.
    method : str, default="variance"
        Method to estimate N:
        - "variance": Use variance ratio (conservative)
        - "count": Just use the raw count (anti-conservative)

    Returns
    -------
    int
        Estimated number of independent trials.

    Notes
    -----
    López de Prado (2018) recommends using clustering (ONC algorithm) for
    more accurate estimation. This function provides simpler heuristics.
    """
    n_total = len(sharpe_ratios)

    if method == "count":
        return n_total
    elif method == "variance":
        # If SRs are highly correlated, their variance will be lower
        # than expected for independent trials
        sr_std = np.std(sharpe_ratios, ddof=1)

        # Expected std for independent trials with mean=0
        # is roughly 1/sqrt(T) where T is track record length
        # Here we use a heuristic based on observed variance

        # If variance is low, strategies are correlated
        # Conservative estimate: N_eff = N * (observed_var / expected_var)
        # We cap at n_total
        expected_var = 0.5 ** 2  # Assume SR ~ N(0, 0.5) for independent strategies
        observed_var = sr_std ** 2

        n_eff = int(np.ceil(n_total * min(observed_var / expected_var, 1.0)))
        return max(1, n_eff)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'variance' or 'count'.")


def multiple_testing_summary(
    sharpe_ratios: np.ndarray,
    returns_list: list[np.ndarray] | None = None,
    n_obs: int = 252,
    significance_level: float = 0.05,
) -> dict:
    """Generate a summary report for multiple testing analysis.

    Parameters
    ----------
    sharpe_ratios : np.ndarray
        Sharpe ratios of all tested strategies.
    returns_list : List[np.ndarray], optional
        List of return arrays for each strategy (for detailed stats).
    n_obs : int, default=252
        Number of observations per strategy.
    significance_level : float, default=0.05
        Significance level for testing.

    Returns
    -------
    dict
        Summary statistics including:
        - n_trials: Total strategies tested
        - n_effective: Estimated independent trials
        - best_sharpe: Highest observed SR
        - expected_max: Expected max SR under null
        - best_dsr: DSR of best strategy
        - haircut: Haircut percentage
        - n_significant: Number passing DSR threshold
    """
    n_trials = len(sharpe_ratios)
    sharpe_std = np.std(sharpe_ratios, ddof=1)
    best_idx = np.argmax(sharpe_ratios)
    best_sharpe = sharpe_ratios[best_idx]

    # Estimate effective number of trials
    n_eff = estimate_n_independent_trials(sharpe_ratios)

    # Expected max under null
    e_max = expected_max_sharpe(n_trials, sharpe_std, 0.0)

    # Compute DSR for best strategy
    if returns_list is not None and len(returns_list) > best_idx:
        best_returns = returns_list[best_idx]
        skewness = float(stats.skew(best_returns))
        kurtosis = float(stats.kurtosis(best_returns, fisher=False))
        n_obs = len(best_returns)
    else:
        skewness = 0.0
        kurtosis = 3.0

    best_dsr = deflated_sharpe_ratio(
        sharpe=best_sharpe,
        n_trials=n_trials,
        sharpe_std_trials=sharpe_std,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Count significant strategies
    threshold = 1 - significance_level
    n_significant = 0
    for sr in sharpe_ratios:
        dsr = deflated_sharpe_ratio(
            sharpe=sr,
            n_trials=n_trials,
            sharpe_std_trials=sharpe_std,
            n_obs=n_obs,
            skewness=skewness,
            kurtosis=kurtosis,
        )
        if dsr >= threshold:
            n_significant += 1

    # Haircut
    haircut_sr, haircut_pct = haircut_sharpe_ratio(best_sharpe, n_trials, sharpe_std)

    return {
        "n_trials": n_trials,
        "n_effective": n_eff,
        "sharpe_mean": float(np.mean(sharpe_ratios)),
        "sharpe_std": sharpe_std,
        "best_sharpe": best_sharpe,
        "expected_max_sharpe": e_max,
        "best_dsr": best_dsr,
        "haircut_sharpe": haircut_sr,
        "haircut_percent": haircut_pct,
        "n_significant": n_significant,
        "significance_level": significance_level,
    }
