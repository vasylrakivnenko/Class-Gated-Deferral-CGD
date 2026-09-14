"""
evalstats.stats -- defensible statistics for cost-vs-accuracy model comparison.

Design contract
---------------
* Every number on the chart comes from the HELD-OUT TEST split, which the
  optimizer (GEPA train + val) never saw.
* Single-model uncertainty  -> Wilson score interval (wilson_ci).
* Two models, same test set  -> PAIRED tests only (mcnemar_test, paired_bootstrap,
  non_inferiority_test / tango_score_ci). Never compare two independent CIs.
* Many models vs one baseline -> holm_bonferroni / benjamini_hochberg.

Dependencies: numpy, scipy, statsmodels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps
from scipy.optimize import brentq
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.multitest import multipletests

__all__ = [
    "wilson_ci", "clopper_pearson_ci",
    "mcnemar_test", "paired_bootstrap", "tango_score_ci", "non_inferiority_test",
    "adjust_pvalues", "seed_summary", "label_noise_report", "decide",
    "sample_size_ni", "min_detectable_delta",
]

# --------------------------------------------------------------------------
# 1. Single-proportion intervals
# --------------------------------------------------------------------------

@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    method: str
    n: int

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (+/-{self.half_width:.3f}, {self.method}, n={self.n})"


def wilson_ci(k: int, n: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

        center = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/(4n^2) )

    Recommended default for accuracy on n = 100-1000 test items
    (Brown, Cai & DasGupta 2001). Never degenerate at p = 0 or 1,
    always inside [0, 1], and far better coverage than the Wald interval.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= k <= n):
        raise ValueError("k must be in [0, n]")
    p = k / n
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, center - half), min(1.0, center + half), "wilson", n)


def clopper_pearson_ci(k: int, n: int, conf: float = 0.95) -> Interval:
    """Exact (Clopper-Pearson) interval. Guaranteed >= nominal coverage,
    therefore conservative (wider). Use only when you must not overstate
    precision -- e.g. a contractual accuracy guarantee."""
    a = 1 - conf
    lo = 0.0 if k == 0 else sps.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - a / 2, k + 1, n - k)
    return Interval(k / n, float(lo), float(hi), "clopper-pearson", n)


# --------------------------------------------------------------------------
# 2. Paired comparison of two models on the SAME test set
# --------------------------------------------------------------------------

def _discordant(correct_a, correct_b):
    a = np.asarray(correct_a).astype(bool).ravel()
    b = np.asarray(correct_b).astype(bool).ravel()
    if a.shape != b.shape:
        raise ValueError("correct_a and correct_b must be the same length (same test items, same order)")
    n = a.size
    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))   # A wrong, B right
    n11 = int(np.sum(a & b))
    n00 = int(np.sum(~a & ~b))
    return n, n11, n01, n10, n00


@dataclass
class McNemarResult:
    n: int
    n_discordant: int
    b_only_a_right: int
    c_only_b_right: int
    delta: float          # acc_b - acc_a
    p_exact: float
    p_midp: float
    p_chi2_cc: float
    p_used: float
    method_used: str
    agreement: float


def mcnemar_test(correct_a, correct_b, method: str = "auto") -> McNemarResult:
    """McNemar's test on two 0/1 correctness vectors over the SAME items.

    Only the discordant pairs carry information; items both models get right
    (or both wrong) contribute nothing. That is exactly why this is far more
    powerful than comparing two independent Wilson intervals.

    method:
      'exact'  -- binomial test on min(b, c), statsmodels mcnemar(exact=True).
                  Conditionally exact => CONSERVATIVE (real alpha < nominal).
      'midp'   -- exact minus half the point mass at the observed value.
                  Fagerland, Lydersen & Laake (2013) show mid-p beats both the
                  exact conditional test and the continuity-corrected asymptotic
                  test on type-I error and power. DEFAULT for n_discordant < 25.
      'chi2'   -- asymptotic chi-square with continuity correction; use only
                  when n_discordant >= 25.
      'auto'   -- 'midp' if n_discordant < 25 else 'chi2'.
    """
    n, n11, b, c, n00 = _discordant(correct_a, correct_b)
    nd = b + c
    delta = (c - b) / n
    agreement = (n11 + n00) / n

    if nd == 0:
        p_exact = p_midp = p_chi2 = 1.0
    else:
        k = min(b, c)
        p_exact = float(min(1.0, 2 * sps.binom.cdf(k, nd, 0.5)))
        p_midp = float(min(1.0, 2 * sps.binom.cdf(k, nd, 0.5) - sps.binom.pmf(k, nd, 0.5)))
        p_chi2 = float(_sm_mcnemar(np.array([[n11, b], [c, n00]]),
                                   exact=False, correction=True).pvalue)

    if method == "auto":
        method = "midp" if nd < 25 else "chi2"
    p_used = {"exact": p_exact, "midp": p_midp, "chi2": p_chi2}[method]
    return McNemarResult(n, nd, b, c, delta, p_exact, p_midp, p_chi2,
                         float(p_used), method, agreement)


@dataclass
class BootstrapResult:
    delta: float
    lo: float
    hi: float
    conf: float
    n_resamples: int
    method: str
    se: float


def paired_bootstrap(correct_a, correct_b, conf: float = 0.95,
                     n_resamples: int = 10000, method: str = "bca",
                     statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
                     seed: int | None = 0) -> BootstrapResult:
    """Bootstrap CI for (metric_b - metric_a), resampling ITEMS, not models.

    The same resampled index vector is applied to both models, so the shared
    "which items are hard" variance cancels -- the standard error of the paired
    difference is much smaller than the difference of two independent SEs
    (Miller 2024, recommendation 4).

    Use this instead of McNemar when the metric is not plain accuracy
    (macro-F1, cost-weighted score, per-item partial credit): McNemar only
    works on binary correctness.

    method='bca' (bias-corrected & accelerated) is the default -- the plain
    percentile interval is noticeably anti-conservative at n < ~200.
    """
    a = np.asarray(correct_a, dtype=float).ravel()
    b = np.asarray(correct_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("vectors must be same length / same item order")

    if statistic is None:
        def _stat(x, y, axis=-1):
            return np.mean(y, axis=axis) - np.mean(x, axis=axis)
        vectorized = True
    else:
        def _stat(x, y):
            return statistic(x, y)
        vectorized = False

    res = sps.bootstrap(
        (a, b), _stat, paired=True, vectorized=vectorized, axis=-1,
        n_resamples=n_resamples, confidence_level=conf,
        method={"bca": "BCa", "percentile": "percentile", "basic": "basic"}[method],
        rng=np.random.default_rng(seed),
    )
    return BootstrapResult(
        delta=float(np.mean(b) - np.mean(a)),
        lo=float(res.confidence_interval.low),
        hi=float(res.confidence_interval.high),
        conf=conf, n_resamples=n_resamples, method=method,
        se=float(res.standard_error),
    )


def tango_score_ci(b: int, c: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Tango (1998) score confidence interval for the PAIRED difference of
    proportions, delta = p_b - p_a, where
        b = #(A right, B wrong), c = #(A wrong, B right), n = #items.

    Score statistic at a hypothesised delta = d:
        pa   = 2n
        pb   = -b - c + (2n - c + b) * d
        pc   = -b * d * (1 - d)
        q21  = (sqrt(pb^2 - 4*pa*pc) - pb) / (2*pa)          # constrained MLE
        Z(d) = (c - b - n*d) / sqrt(n * (2*q21 + d*(1 - d)))
    The interval is {d : |Z(d)| < z_{1-alpha/2}}, found by bisection.

    Ported from PropCIs::scoreci.mp (CRAN, GPL). This is the closed-form,
    deterministic alternative to the bootstrap and has better small-sample
    coverage than a Wald interval on the paired difference.
    """
    z = sps.norm.ppf(1 - (1 - conf) / 2)

    def score(d: float) -> float:
        pa = 2 * n
        pbb = -b - c + (2 * n - c + b) * d
        pcc = -b * d * (1 - d)
        disc = pbb * pbb - 4 * pa * pcc
        q21 = (np.sqrt(max(disc, 0.0)) - pbb) / (2 * pa)
        var = n * (2 * q21 + d * (1 - d))
        if var <= 0:
            return np.inf
        return (c - b - n * d) / np.sqrt(var)

    point = (c - b) / n
    eps = 1e-9

    def solve(target: float, lo_b: float, hi_b: float) -> float:
        # score(d) is monotone decreasing in d; find d with score(d) == target
        f = lambda d: score(d) - target
        a, bb = lo_b + eps, hi_b - eps
        if f(a) * f(bb) > 0:                       # root outside bracket
            return hi_b if abs(f(bb)) < abs(f(a)) else lo_b
        return float(brentq(f, a, bb, xtol=1e-10, rtol=1e-12))

    hi = 1.0 if c == n else solve(-z, point, 1.0)
    lo = -1.0 if b == n else solve(+z, -1.0, point)
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# 3. Non-inferiority (the claim the product actually makes)
# --------------------------------------------------------------------------

@dataclass
class NonInferiorityResult:
    delta: float                # acc_cheap - acc_reference (negative = worse)
    margin: float               # the tolerated loss, positive number
    lower_bound: float          # one-sided (1-alpha) lower confidence bound on delta
    p_value: float              # one-sided p for H0: delta <= -margin
    non_inferior: bool
    also_superior: bool
    method: str
    n: int
    n_discordant: int
    agreement: float
    powered: bool               # False => "inconclusive", not "worse"
    note: str = ""


def non_inferiority_test(correct_ref, correct_cheap, margin: float = 0.03,
                         alpha: float = 0.05, method: str = "tango",
                         n_resamples: int = 10000, seed: int | None = 0
                         ) -> NonInferiorityResult:
    """One-sided paired non-inferiority test -- the correct test for
    "this cheap model is not worse than your expensive one by more than `margin`".

        H0 (what we must reject):  acc_cheap - acc_ref <= -margin
        H1 (what we want to show): acc_cheap - acc_ref >  -margin

    Decision rule: declare NON-INFERIOR iff the one-sided (1 - alpha) LOWER
    confidence bound on delta lies strictly above -margin. This is exactly the
    lower half of a TOST; the upper half is not needed, because being BETTER
    than the reference is never a problem for us.

    method:
      'tango'     -- Tango (1998) score interval, closed form, deterministic.
                     Default: reproducible, no RNG in the published number.
      'bootstrap' -- paired BCa bootstrap; use for non-accuracy metrics.

    IMPORTANT: a failure to reject is NOT evidence of inferiority. `powered`
    reports whether the test could have succeeded at all: if the achievable
    lower bound at delta = 0 would still sit below -margin, the test set is
    too small and the honest label is "inconclusive".
    """
    if margin <= 0:
        raise ValueError("margin must be a positive number, e.g. 0.03 for 3 accuracy points")
    n, n11, b, c, n00 = _discordant(correct_ref, correct_cheap)
    nd = b + c
    delta = (c - b) / n
    agreement = (n11 + n00) / n

    if method == "tango":
        lo, _ = tango_score_ci(b, c, n, conf=1 - 2 * alpha)  # two-sided (1-2a) => one-sided (1-a) lower
        # one-sided p: smallest alpha' at which -margin is excluded
        pa2, pbb = 2 * n, -b - c + (2 * n - c + b) * (-margin)
        pcc = -b * (-margin) * (1 + margin)
        q21 = (np.sqrt(max(pbb * pbb - 4 * pa2 * pcc, 0.0)) - pbb) / (2 * pa2)
        var = n * (2 * q21 + (-margin) * (1 + margin))
        zstat = (c - b - n * (-margin)) / np.sqrt(var) if var > 0 else np.inf
        p = float(sps.norm.sf(zstat))
    elif method == "bootstrap":
        bs = paired_bootstrap(correct_ref, correct_cheap, conf=1 - 2 * alpha,
                              n_resamples=n_resamples, method="bca", seed=seed)
        lo = bs.lo
        p = float("nan")
    else:
        raise ValueError("method must be 'tango' or 'bootstrap'")

    # Best case achievable: same discordant total, but all favouring the cheap model.
    best_lo, _ = tango_score_ci(0, nd, n, conf=1 - 2 * alpha) if nd else (0.0, 0.0)
    powered = bool(best_lo > -margin)

    sup = mcnemar_test(correct_ref, correct_cheap, method="auto")
    return NonInferiorityResult(
        delta=delta, margin=margin, lower_bound=float(lo), p_value=p,
        non_inferior=bool(lo > -margin),
        also_superior=bool(delta > 0 and sup.p_used < alpha),
        method=method, n=n, n_discordant=nd, agreement=agreement,
        powered=powered,
        note="" if powered else
             f"Test set too small: even a perfect discordant split could not clear a {margin:.3f} margin. Report INCONCLUSIVE.",
    )


# --------------------------------------------------------------------------
# 4. Multiple comparisons
# --------------------------------------------------------------------------

def adjust_pvalues(pvals: Sequence[float], alpha: float = 0.05, method: str = "holm"):
    """method='holm'  -> Holm-Bonferroni, controls FWER. Use for the SHIPPED
                          winner claim ("model X clears your bar").
       method='fdr_bh' -> Benjamini-Hochberg, controls FDR. Use for the
                          exploratory shortlist column of the table."""
    reject, p_adj, _, _ = multipletests(np.asarray(pvals, float), alpha=alpha, method=method)
    return np.asarray(reject), np.asarray(p_adj)


# --------------------------------------------------------------------------
# 5. Run-to-run variance
# --------------------------------------------------------------------------

@dataclass
class SeedSummary:
    mean: float
    sd: float
    se: float
    lo: float
    hi: float
    runs: int
    min: float
    max: float
    flip_rate: float | None = None


def seed_summary(accuracies: Sequence[float], conf: float = 0.95,
                 per_item_correct: np.ndarray | None = None) -> SeedSummary:
    """Summarise R repeated runs of the SAME model on the SAME test set.

    Reports mean +/- sd across seeds, and a t-based CI on the mean
    (t, not z: R is small). `per_item_correct` is an R x n 0/1 matrix; if given,
    `flip_rate` = fraction of test items whose correctness was not identical
    across all R runs -- the honest, dataset-level measure of nondeterminism.
    """
    x = np.asarray(accuracies, float).ravel()
    R = x.size
    if R < 2:
        raise ValueError("need >= 2 runs")
    m, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / np.sqrt(R)
    t = sps.t.ppf(1 - (1 - conf) / 2, R - 1)
    flip = None
    if per_item_correct is not None:
        M = np.asarray(per_item_correct).astype(int)
        flip = float(np.mean(M.min(axis=0) != M.max(axis=0)))
    return SeedSummary(m, sd, float(se), m - t * se, m + t * se, R,
                       float(x.min()), float(x.max()), flip)


# --------------------------------------------------------------------------
# 6. Label noise / teacher-agreement trap
# --------------------------------------------------------------------------

@dataclass
class LabelNoiseReport:
    n: int
    n_flagged: int
    est_noise_rate: float
    ceiling: float
    ceiling_lo: float
    ceiling_hi: float
    verdict: str


def label_noise_report(est_noise_rate: float, n: int, best_observed_acc: float,
                       n_flagged: int | None = None) -> LabelNoiseReport:
    """Turn an estimated label-error rate e into the number the user needs:
    the CEILING = 1 - e. Any model already inside the ceiling's CI is
    measuring label noise, not capability, and must not be ranked by accuracy.

    Get `est_noise_rate` from (a) a re-labelled audit subsample -- the honest
    way -- or (b) cleanlab.filter.find_label_issues on cross-validated
    out-of-sample probabilities, then hand-check the flagged items.
    """
    ceiling = 1.0 - est_noise_rate
    k = int(round(ceiling * n))
    ci = wilson_ci(k, n)
    if best_observed_acc >= ci.lo:
        verdict = ("AT THE LABEL CEILING: the best model is statistically "
                   "indistinguishable from perfect agreement with clean labels. "
                   "Rank by cost, not accuracy, and fix labels before re-running.")
    elif ceiling - best_observed_acc < 0.05:
        verdict = "Within 5 points of the label ceiling -- accuracy differences here are mostly noise."
    else:
        verdict = "Headroom remains: accuracy differences are meaningful."
    return LabelNoiseReport(n, n_flagged or -1, est_noise_rate,
                            ceiling, ci.lo, ci.hi, verdict)


# --------------------------------------------------------------------------
# 7. The single decision the product ships
# --------------------------------------------------------------------------

def decide(correct_ref, correct_cheap, bar: float | None = None,
           margin: float = 0.03, alpha: float = 0.05, p_adj: float | None = None) -> dict:
    """One row of the final table. `p_adj` is the Holm-adjusted p from the
    family of all candidate models; pass it so the shipped verdict is
    multiplicity-corrected."""
    ni = non_inferiority_test(correct_ref, correct_cheap, margin=margin, alpha=alpha)
    acc = wilson_ci(int(np.sum(np.asarray(correct_cheap).astype(bool))), len(correct_cheap))
    p = p_adj if p_adj is not None else ni.p_value
    if not ni.powered:
        verdict = "INCONCLUSIVE (test set too small)"
    elif ni.lower_bound > -margin and p < alpha:
        verdict = "MATCHES BASELINE"
    elif acc.lo >= (bar if bar is not None else -np.inf):
        verdict = "CLEARS YOUR BAR (but not proven equal to baseline)"
    else:
        verdict = "WORSE"
    out = asdict(ni)
    out.update(accuracy=acc.point, acc_lo=acc.lo, acc_hi=acc.hi,
               p_adjusted=p, verdict=verdict)
    return out


# --------------------------------------------------------------------------
# 8. Planning: how big must the test set be? (run this BEFORE you spend money)
# --------------------------------------------------------------------------

def sample_size_ni(disagreement: float, margin: float, alpha: float = 0.05,
                   power: float = 0.80, true_delta: float = 0.0) -> int:
    """Test-set size needed to *prove* non-inferiority at `margin`.

        n = (z_alpha + z_power)^2 * (p01 + p10 - delta^2) / (margin + delta)^2

    `disagreement` = p01 + p10 = expected fraction of test items where the two
    models differ. Estimate it from a 50-item pilot -- it is the dominant term.

    At true_delta = 0 this reduces to n = (z_a+z_b)^2 * disagreement / margin^2,
    i.e. HALVING the margin QUADRUPLES the test set.

    Reference values (alpha=.05 one-sided, power=.80, true_delta=0):
        disagreement 5%,  margin 0.05 -> n = 124
        disagreement 5%,  margin 0.03 -> n = 343
        disagreement 10%, margin 0.05 -> n = 247
        disagreement 10%, margin 0.03 -> n = 687
    """
    za, zb = sps.norm.ppf(1 - alpha), sps.norm.ppf(power)
    var = disagreement - true_delta ** 2
    return int(np.ceil((za + zb) ** 2 * var / (margin + true_delta) ** 2))


def min_detectable_delta(n: int, disagreement: float, alpha: float = 0.05,
                         power: float = 0.80) -> float:
    """Smallest true accuracy gap a paired (McNemar) comparison on `n` items can
    detect with the given power. Anything smaller than this on your chart is
    a coin flip -- render it as a tie, not as a ranking."""
    za, zb = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)
    return float((za + zb) * np.sqrt(disagreement / n))
