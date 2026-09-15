"""Statistics for honest model comparison on a small held-out test set.

The whole product rests on one claim: "this cheaper model clears your accuracy
bar." That claim is only worth anything if it survives the fact that the test
set is small. Everything here exists to keep us from overclaiming.

Three ideas do the work:

1. A point accuracy of 0.85 on n=150 is not 0.85. It is 0.85 with a Wilson
   interval of roughly +/-0.06. Any chart that draws bars without that interval
   is lying about its own resolution.

2. Two models scored on the *same* items produce paired data. Comparing their
   independent intervals is the wrong test and is badly underpowered -- the
   models agree on most items, and only the disagreements carry information.
   That is exactly what McNemar's test looks at.

3. Our actual claim is non-inferiority, not superiority. We never need to show
   the cheap model is *better*. We need to show it is not worse by more than a
   margin the user chose. That is a one-sided test against -delta, and it is a
   different (and more honest) question than "is there a significant difference".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps


# ── point estimates with intervals ──────────────────────────────────────────

@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    def to_dict(self) -> dict:
        return {**asdict(self), "half_width": self.half_width}


def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation, which is embarrassingly wrong near
    0 and 1 (it happily returns intervals extending past 100% accuracy) and
    undercovers at the sample sizes we care about. Preferred over
    Clopper-Pearson, which is exact but conservative -- it guarantees *at least*
    95% coverage by being wider than it needs to be, which would make our chart
    look less decisive than the data warrants.

    Wilson is the standard recommendation for n in the 100-200 range.
    """
    if n_total <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    z = sps.norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), n_total)


def accuracy_ci(correct: np.ndarray, confidence: float = 0.95) -> Interval:
    """Wilson interval from a boolean per-item correctness vector."""
    c = np.asarray(correct, dtype=bool)
    return wilson_ci(int(c.sum()), int(c.size), confidence)


# ── paired comparison of two models on the same items ───────────────────────

@dataclass(frozen=True)
class PairedComparison:
    acc_a: float
    acc_b: float
    diff: float               # acc_a - acc_b
    n_both_right: int
    n_a_only: int             # a right, b wrong   (McNemar's b)
    n_b_only: int             # b right, a wrong   (McNemar's c)
    n_both_wrong: int
    n_discordant: int
    p_value: float
    test_used: str
    diff_ci_lo: float
    diff_ci_hi: float

    def to_dict(self) -> dict:
        return asdict(self)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray,
                 confidence: float = 0.95, n_boot: int = 10_000,
                 seed: int = 0, method: str = "mid-p") -> PairedComparison:
    """Paired test for 'do these two models differ on this test set?'

    Only the discordant pairs carry information: items where both models agree
    say nothing about which is better. With n=250 and two models agreeing on 220
    items, the effective sample size for the comparison is 30, not 250 -- which
    is exactly why comparing two independent confidence intervals misleads here.

    Default is the **mid-p** McNemar test. This is deliberate. The two variants
    most code reaches for -- the exact conditional test and the asymptotic test
    with a continuity correction -- are both explicitly recommended against by
    Fagerland, Lydersen & Laake (2013), who conclude: "We do not recommend use
    of the McNemar exact conditional test nor the asymptotic test with CC in any
    situation." Both are conservative: they under-reject, which for us means
    calling a real difference "not significant" and telling a user their cheap
    model is fine when it is not. Mid-p keeps the exact test's validity at small
    discordant counts while recovering most of the lost power.

    Reference: Fagerland MW, Lydersen S, Laake P. "The McNemar test for binary
    matched-pairs data: mid-p and asymptotic are better than exact conditional."
    BMC Medical Research Methodology 2013;13:91.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must align: {a.shape} vs {b.shape}")

    n_a_only = int(np.sum(a & ~b))
    n_b_only = int(np.sum(~a & b))
    n_disc = n_a_only + n_b_only

    if n_disc == 0:
        p_value, test_used = 1.0, "identical (no discordant pairs)"
    elif method == "mid-p":
        # Exact two-sided binomial p, less half the point probability at the
        # observed count -- the mid-p correction for discreteness.
        exact = float(sps.binomtest(n_a_only, n_disc, 0.5).pvalue)
        point = float(sps.binom.pmf(n_a_only, n_disc, 0.5))
        p_value = min(1.0, max(0.0, exact - point))
        test_used = "McNemar mid-p"
    elif method == "exact":
        p_value = float(sps.binomtest(n_a_only, n_disc, 0.5).pvalue)
        test_used = "McNemar exact conditional (conservative; not recommended)"
    elif method == "asymptotic":
        # Without the continuity correction, per the same reference.
        chi2 = (n_a_only - n_b_only) ** 2 / n_disc
        p_value = float(sps.chi2.sf(chi2, df=1))
        test_used = "McNemar asymptotic (no continuity correction)"
    else:
        raise ValueError(f"unknown method {method!r}")

    lo, hi = paired_bootstrap_diff(a, b, confidence=confidence, n_boot=n_boot, seed=seed)

    return PairedComparison(
        acc_a=float(a.mean()), acc_b=float(b.mean()), diff=float(a.mean() - b.mean()),
        n_both_right=int(np.sum(a & b)), n_a_only=n_a_only, n_b_only=n_b_only,
        n_both_wrong=int(np.sum(~a & ~b)), n_discordant=n_disc,
        p_value=p_value, test_used=test_used, diff_ci_lo=lo, diff_ci_hi=hi,
    )


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray,
                          confidence: float = 0.95, n_boot: int = 10_000,
                          seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for (acc_a - acc_b), resampling *items*.

    Resampling item indices rather than each model's scores independently is the
    point: it preserves the correlation between the two models, which is what
    makes the paired comparison tighter than the unpaired one.
    """
    diffs = _bootstrap_diffs(correct_a, correct_b, n_boot=n_boot, seed=seed)
    if diffs is None:
        return float("nan"), float("nan")
    alpha = 1 - confidence
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def _bootstrap_diffs(correct_a: np.ndarray, correct_b: np.ndarray,
                     n_boot: int = 10_000, seed: int = 0):
    """One paired resample of (acc_a - acc_b), or None for an empty input.

    Factored out so that a caller needing both a CI and a p-value derives them
    from the SAME resample. `non_inferiority_test` used to draw two, seeded
    `seed` and `seed + 1`, which made its own `passes` flag and its own
    `p_value` two answers to one question computed on different samples.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    n = a.size
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    return a[idx].mean(axis=1) - b[idx].mean(axis=1)


# ── the test our product actually needs ─────────────────────────────────────

@dataclass(frozen=True)
class NonInferiority:
    candidate_acc: float
    baseline_acc: float
    diff: float
    margin: float
    diff_ci_lo: float
    diff_ci_hi: float
    p_value: float
    passes: bool
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def non_inferiority_test(correct_candidate: np.ndarray, correct_baseline: np.ndarray,
                         margin: float = 0.02, confidence: float = 0.95,
                         n_boot: int = 10_000, seed: int = 0) -> NonInferiority:
    """Is the cheap model no worse than the expensive one by more than `margin`?

    This is the question the product sells an answer to, and it is deliberately
    not "is the cheap model better" and not "is there no significant difference".

    "No significant difference" is the trap: failing to reject the null is not
    evidence of equivalence, it is usually just evidence of a small test set.
    A useless test set would pass that bar every time. Non-inferiority inverts
    the burden of proof -- the null is that the candidate IS worse by at least
    `margin`, and we only get to claim a pass by rejecting it.

    Decision rule: pass iff the lower bound of the one-sided CI on
    (candidate - baseline) lies above -margin. That is the standard
    CI-based equivalence rule and it is what we draw on the chart.

    A margin of 0.02 says "I will accept up to 2 points of accuracy loss to save
    the money." It is the user's business decision, not a statistical one, and
    the product must make them set it deliberately.
    """
    cand = np.asarray(correct_candidate, dtype=bool)
    base = np.asarray(correct_baseline, dtype=bool)
    if cand.shape != base.shape:
        raise ValueError(f"paired vectors must align: {cand.shape} vs {base.shape}")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    diff = float(cand.mean() - base.mean())

    # ONE resample, used for both the interval and the p-value. Drawing two
    # (the CI on `seed`, the p-value on `seed + 1`) let the decision rule and
    # the number printed beside it disagree near the boundary: at n=100 with 11
    # candidate-only and 6 baseline-only correct, seed 0 gave lo=-0.01 (passes)
    # while seed 1 gave p=0.0531 (not significant at 0.05), and experiment.py
    # ANDs those two together. Sharing the resample makes them consistent by
    # construction: `lo` is the 5th percentile of `boot`, so lo > -margin and
    # p < 0.05 are the same statement about the same sample.
    two_sided = 1 - 2 * (1 - confidence)
    boot = _bootstrap_diffs(cand, base, n_boot=n_boot, seed=seed)
    alpha = 1 - two_sided
    if boot is None:
        lo = hi = p_value = float("nan")
    else:
        lo = float(np.quantile(boot, alpha / 2))
        hi = float(np.quantile(boot, 1 - alpha / 2))
        # H0: diff <= -margin. Algebraically what the two-sample form computed
        # (`boot - diff <= -margin - diff` cancels to this), now on the sample
        # the interval came from.
        p_value = float(np.mean(boot <= -margin))

    passes = bool(lo > -margin)
    if passes and diff >= 0:
        verdict = "non-inferior (and not worse on the point estimate)"
    elif passes:
        verdict = f"non-inferior within the {margin:.0%} margin"
    elif hi < -margin:
        verdict = f"clearly worse by more than the {margin:.0%} margin"
    else:
        verdict = f"inconclusive -- test set too small to rule out a {margin:.0%} loss"

    return NonInferiority(
        candidate_acc=float(cand.mean()), baseline_acc=float(base.mean()), diff=diff,
        margin=margin, diff_ci_lo=lo, diff_ci_hi=hi, p_value=p_value,
        passes=passes, verdict=verdict,
    )


# ── multiple comparisons ────────────────────────────────────────────────────

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns per-hypothesis reject flags.

    We test ~6-10 candidate models against one baseline, so roughly one in three
    runs would throw a spurious "winner" at an uncorrected alpha of 0.05. Holm
    controls the family-wise error rate, is uniformly more powerful than plain
    Bonferroni, and needs no independence assumption -- which matters, because
    our candidates are scored on the same items and are anything but independent.

    Preferred over Benjamini-Hochberg here: BH controls the false discovery rate,
    which suits screening a large candidate pool. We report a single winner that
    someone will act on, so controlling the chance of *any* false claim is the
    right guarantee.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break  # step-down: first failure stops all subsequent rejections
    return reject


# ── label quality ───────────────────────────────────────────────────────────

def bootstrap_metric_ci(values: np.ndarray, confidence: float = 0.95,
                        n_boot: int = 10_000, seed: int = 0) -> Interval:
    """Percentile bootstrap CI for the mean of any per-item metric.

    For scores that are not 0/1 (macro-F1, cost per item, latency), where the
    binomial machinery behind Wilson does not apply.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    alpha = 1 - confidence
    return Interval(float(v.mean()), float(np.quantile(means, alpha / 2)),
                    float(np.quantile(means, 1 - alpha / 2)), int(v.size))


def resolution_warning(n_test: int, expected_acc: float = 0.85) -> str:
    """Plain-language statement of what a test set of this size can and cannot see.

    Surfaced in the report because "we cannot tell these two apart" is a real,
    useful answer, and a product that hides it is selling noise.
    """
    ci = wilson_ci(int(round(expected_acc * n_test)), n_test)
    hw = ci.half_width
    return (
        f"At n={n_test} and ~{expected_acc:.0%} accuracy, the 95% Wilson interval is "
        f"+/-{hw:.1%}. Two models closer than roughly {2*hw:.1%} apart cannot be "
        f"separated by unpaired comparison on this set. Paired tests (McNemar) do "
        f"better because they use only the items where the models disagree."
    )
