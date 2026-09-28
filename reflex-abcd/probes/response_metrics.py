"""The metric for each response-selection head, and the argument for it.

THE RULE THIS MODULE ENFORCES (DECISIONS D5)
--------------------------------------------
Every headline metric is reported beside what a LABEL-BLIND CONSTANT scores, and
a metric a constant wins is DROPPED, not caveated. The constant is fit on the
FIT split's label prior and applied to the EVAL split -- a constant chosen by
looking at the eval labels is an oracle, not a baseline.

H5 -- SKELETON: top-1 accuracy. recall@3 as a secondary. NEVER recall@10.
--------------------------------------------------------------------------
570 classes, but the prior is brutally concentrated: on the train labels of bank
``c270df0e249444bb`` the four single-act skeletons ASK / ACK / OFFER / CLOSE
cover 59.6% of turns. Computed from that label file, the FIT-split PRIOR (the
cumulative train share of the most frequent skeletons; 18181/71133 for S0000,
``outputs/compile/compile_summary.md``) is:

    recall@1 0.2556 | recall@3 0.5174 | recall@5 0.6474 | recall@10 0.8229

That row is the prior, NOT the constant's score. APPLIED to test_seen -- the
quantity the rule above defines -- the label-blind constant SCORES
(``outputs/probes/response/select.json`` ``headline_test_seen.h5.constant``):

    recall@1 0.2630 | recall@3 0.5269 | recall@5 0.6543 | recall@10 0.8336

D5's bar is the applied row, never the prior row,
so **recall@10 is 83 points for a predictor that has not looked at the input**.
Reporting it would be exactly the ``nextstep`` trap D5 names. top-1 is the
headline because it has the most room (a 74-point gap to a perfect score);
recall@3 is reported because the gate consumes a conformal SET, and the
single-act/two-act boundary is where the set has to widen. Anything at k >= 5 is
recorded in the artifact but must not be quoted.

Near-miss tiers, reported as diagnostics and never as the headline:
``act_multiset`` (right acts, wrong order) and ``arity`` (right number of
sentences). They say what SHAPE the error has -- an arity error changes how many
templates H7 is then asked for, so it is the error that cascades.

H7 -- TEMPLATE: per-act top-1, plus recall@5/@20, plus three nested tiers,
on TWO denominators, always as a pair.
--------------------------------------------------------------------------
Plain top-1 is the wrong instrument ALONE, for three separate reasons, and the
brief's framing of "a ~4,505-way choice" is itself the first of them:

1. **It is not a 4,489-way choice.** ``select._score_templates`` ranks each
   position only within ``templates_by_act[act]``. The real label space is 208
   (INSTRUCT) to 1,066 (ASK) candidates. A flat-bank metric measures a head that
   does not exist.
2. **The near-duplicate problem is real but it is NOT what D17 left behind.**
   The field-set merge guard drove field divergence to 0.0%, so within a
   surviving template every surface form requests the same fields. What remains
   is CROSS-template near-duplication: two rows the compiler kept apart. Whether
   picking the neighbour is a "miss" depends on what the mistake COSTS, which is
   not a single question, so this module answers it with three nested predicates
   rather than one:

   * ``exact``       -- same ``template_id``. The strict reading.
   * ``paraphrase``  -- ``cosine(canonical_pred, canonical_gold) >=
     compile.dedup_threshold`` under ``compile.dedup_embed_model``. This is
     literally the relation ``compile.dedup_templates`` used, so a pair above it
     is one the compiler WOULD have merged but for the field guard or for
     single-linkage not chaining them. It is also the instrument DEFECTS_OPEN
     D-7 asks for and calls ``strict_template_coverage``.
   * ``field_safe``  -- ``compile._requested_fields(pred) ==
     _requested_fields(gold)``. Its complement is the **wrong-field rate**,
     D16's number: the fast path asking the customer for identifiers no human
     asked for. On bank ``c270df0e249444bb`` only 710 of 4,489 templates request
     a named field (9,568 of 58,263 occurrences), so this predicate is narrow --
     but it is the one a customer feels, and D16 is explicit that it is the
     number that bites.

   EACH TIER GETS ITS OWN CONSTANT BAR. A loosened predicate also helps the
   constant, sometimes more than it helps the model, and assuming otherwise is
   how a metric a constant wins gets shipped.
3. **The gate consumes a SET, not an argmax.** ``gate.evaluate_gate`` reads a
   conformal prediction set; ``select.prediction_set`` builds it from H7's
   probabilities. recall@5 and recall@20 are the label-blind-k stand-ins for
   "could the right template have been in the set at all". Their constant bars
   on the current bank are 0.3391 and 0.5687 micro -- high, but far enough below
   1 to leave the metric usable, unlike H5's recall@10.

**The two denominators, which are the part most likely to be got wrong.**
35.0% of sentence positions have NO gold template, because
``compile.min_template_count: 2`` drops every phrasing no human used twice.
So:

  * ``conditional``   -- positions with a gold. This is what the HEAD can do.
    It is optimistic by construction: the surviving positions are exactly the
    frequent, repetitive ones.
  * ``unconditional`` -- all positions; no gold counts as a miss. This is what
    the SYSTEM delivers.

Quoting one without the other is the D6 failure mode verbatim -- a baseline
without its configuration attached. This module refuses to emit one alone.

COMPOSED (turn level) -- the metric that actually describes the reflex.
----------------------------------------------------------------------
``compose@1``: the predicted skeleton is exact AND every position's top-1
template is exact, i.e. the turn's whole gold template tuple is recovered. That
is what ``Selection.exact_template_match`` reports and what "reflex fidelity"
means. Its constant bar is the lowest of any metric here -- 0.0446 over fully
covered turns, 0.0270 over all retrieve turns -- so it has the most headroom and
is the cleanest headline for the response path as a whole.

H4 -- VALUES: top-1 in INDEX space, with the constant computed in index space.
-------------------------------------------------------------------------------
The label is a column in a 226-wide space, not a string (see
:mod:`probes.response_labels`). Two consequences drive the metric:

* **The constant must be the modal COLUMN, not the modal string.** A copy head's
  columns are positions in a token window, and gold values cluster at similar
  positions, so the modal column can be far stronger than the modal string
  (whose share is ~4.4% on the train labels). Computing the bar in string space
  would understate it and could let through a metric a constant wins. This
  module computes both and reports the LARGER as the bar.
* **String-level accuracy is reported alongside**, because that is the predicate
  ``evaluate.ast_metrics`` compares and two columns can decode to the same
  string. They are different questions and both are quoted.

Denominator: rows where ``value_target_index >= 0``. Rows at -1 are unscorable
(the official processor drops them), and because the copy tier is built from the
context window, **the scorable set changes with the window**. Cross-window
comparisons are taken on the INTERSECTION; :func:`align_h4_denominator` does it.

CONFIDENCE INTERVALS: CLUSTERED BY CONVERSATION, NOT BY ROW.
------------------------------------------------------------
Turns inside one conversation are not independent, and H7 positions inside one
turn are barely independent at all -- they share a context string exactly. The
CI recorded in D7 ("half-width 0.68 points at 13,284 rows") is what a row-level
binomial gives at p~0.83, so it assumes independence this data does not have and
is therefore OPTIMISTIC. :func:`bootstrap_ci` resamples CONVERSATIONS with
replacement. Expect wider intervals than D7's, and treat that as the correction
it is rather than as a discrepancy.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Optional, Sequence


# --------------------------------------------------------------------------- #
# Constant baselines -- fit on the FIT split, applied to the EVAL split
# --------------------------------------------------------------------------- #


def constant_ranking(fit_golds: Sequence[str]) -> list:
    """The label-blind ranking: classes by descending frequency in the FIT split."""
    return [cls for cls, _ in Counter(fit_golds).most_common()]


def constant_recall_at_k(
    fit_golds: Sequence[str],
    eval_golds: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict:
    """``recall@k`` of the constant predictor, fit on one split, scored on another.

    Note the asymmetry that makes this a real baseline: the ranking comes from
    ``fit_golds`` and is then frozen. Ranking on ``eval_golds`` would be an
    oracle and would inflate the bar, which -- perversely -- would make the
    model look better by making the D5 comparison unfair in the model's favour
    only when the model loses. Both directions are wrong; this one is the
    baseline.
    """
    ranking = constant_ranking(fit_golds)
    counts = Counter(eval_golds)
    total = sum(counts.values())
    if not total:
        return {f"recall@{k}": None for k in ks}
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(counts.get(cls, 0) for cls in ranking[:k]) / total
    return out


def constant_recall_at_k_grouped(
    fit_pairs: Sequence[tuple],
    eval_pairs: Sequence[tuple],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict:
    """Per-group constant, micro- and macro-averaged. ``pairs`` are ``(group, gold)``.

    For H7 the group is the ACT, because that is the candidate pool
    ``select._score_templates`` ranks within. A single global constant would be a
    strawman: it would ignore that the act is known before the template is
    scored.
    """
    fit_by_group = defaultdict(list)
    for group, gold in fit_pairs:
        fit_by_group[group].append(gold)
    eval_by_group = defaultdict(list)
    for group, gold in eval_pairs:
        eval_by_group[group].append(gold)

    per_group: dict = {}
    micro_hits = {k: 0 for k in ks}
    micro_n = 0
    for group, golds in eval_by_group.items():
        ranking = constant_ranking(fit_by_group.get(group, []))
        counts = Counter(golds)
        n = sum(counts.values())
        row = {}
        for k in ks:
            hits = sum(counts.get(cls, 0) for cls in ranking[:k])
            row[f"recall@{k}"] = hits / n if n else None
            micro_hits[k] += hits
        row["n"] = n
        per_group[group] = row
        micro_n += n
    micro = {f"recall@{k}": (micro_hits[k] / micro_n if micro_n else None) for k in ks}
    macro = {
        f"recall@{k}": (
            sum(r[f"recall@{k}"] for r in per_group.values()) / len(per_group)
            if per_group else None
        )
        for k in ks
    }
    return {"micro": micro, "macro": macro, "per_group": per_group, "n": micro_n}


# --------------------------------------------------------------------------- #
# Scoring from a ranked candidate list
# --------------------------------------------------------------------------- #


def hits_at_k(ranked: Sequence[str], gold: str, ks: Sequence[int]) -> dict:
    """``{k: bool}`` -- is ``gold`` in the top k of ``ranked``."""
    try:
        rank = list(ranked).index(gold)
    except ValueError:
        rank = None
    return {k: (rank is not None and rank < k) for k in ks}


def accuracy(flags: Sequence[bool]) -> Optional[float]:
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


# --------------------------------------------------------------------------- #
# H7 near-duplicate tiers
# --------------------------------------------------------------------------- #


class TemplateEquivalence:
    """The three nested "is this miss really a miss" predicates for H7.

    Built ONCE per bank. The paraphrase relation embeds the 4,489 canonical
    ``text_delex`` strings with ``compile.dedup_embed_model`` -- the same
    sentence-trained encoder ``compile.dedup_templates`` uses, and deliberately
    NOT ``model.encoder``: D12 measured raw ModernBERT-base putting 59% of
    arbitrary ABCD sentence pairs at cosine >= 0.92, which would make every pair
    "a paraphrase" and the tier meaningless.
    """

    def __init__(self, bank: Any, cfg: dict) -> None:
        from reflex.compile import _requested_fields
        from reflex.config import get_dotted

        self.threshold = float(get_dotted(cfg, "compile.dedup_threshold"))
        terms = get_dotted(cfg, "compile.merge_field_terms")
        self.ids = [t.template_id for t in bank.templates]
        self.index = {tid: i for i, tid in enumerate(self.ids)}
        self.act = {t.template_id: t.act for t in bank.templates}
        self.text = {t.template_id: t.text_delex for t in bank.templates}
        self.fields = {t.template_id: _requested_fields(t.text_delex, terms) for t in bank.templates}
        self._vectors = None
        self._cfg = cfg
        self._bank = bank

    def _embed(self):
        if self._vectors is None:
            from reflex.compile import _embed_texts
            self._vectors = _embed_texts(
                [self.text[tid] for tid in self.ids], self._cfg, "compile.dedup_embed_model"
            )
        return self._vectors

    def exact(self, pred: str, gold: str) -> bool:
        return bool(pred) and pred == gold

    def field_safe(self, pred: str, gold: str) -> bool:
        """Same requested field-set. Complement = the D16 wrong-field event."""
        if not pred or not gold:
            return False
        return self.fields.get(pred, frozenset()) == self.fields.get(gold, frozenset())

    def paraphrase(self, pred: str, gold: str) -> bool:
        """Cosine >= ``compile.dedup_threshold`` between the two canonicals."""
        if not pred or not gold:
            return False
        if pred == gold:
            return True
        vectors = self._embed()
        i, j = self.index.get(pred), self.index.get(gold)
        if i is None or j is None:
            return False
        return float(vectors[i] @ vectors[j]) >= self.threshold

    def tier_flags(self, pred: str, gold: str) -> dict:
        """All three, nested: exact => paraphrase is also checked independently."""
        return {
            "exact": self.exact(pred, gold),
            "paraphrase": self.paraphrase(pred, gold),
            "field_safe": self.field_safe(pred, gold),
        }

    def paraphrase_density(self) -> dict:
        """How LOOSE the paraphrase tier actually is on this bank, per act pool.

        Report this beside the tier or the tier will be misread. Measured on real
        sentences here, ``compile.dedup_threshold`` (0.92) is STRICT: two forms
        that plainly mean the same thing ("can i have your account id?" /
        "may i have your account id?") score 0.880 under
        ``compile.dedup_embed_model`` and do NOT clear it. DEFECTS_OPEN D-7 says
        the same thing from the other side -- 80.6% of merged forms sit below
        threshold against their own canonical, because single-linkage CHAINS.

        So the ``paraphrase`` tier is close to ``exact`` by design, and a large
        gap between them would be the surprise, not the expectation. What it
        buys is D-7's ``strict_template_coverage``: the share of "misses" that
        are a genuinely interchangeable phrasing rather than a different thing
        to say.
        """
        import numpy as np

        vectors = self._embed()
        out: dict = {}
        by_act: dict = defaultdict(list)
        for i, tid in enumerate(self.ids):
            by_act[self.act[tid]].append(i)
        for act, idx in sorted(by_act.items()):
            if len(idx) < 2:
                out[act] = {"n": len(idx), "pair_rate_above_threshold": None}
                continue
            block = vectors[np.asarray(idx)] @ vectors[np.asarray(idx)].T
            n = len(idx)
            above = int((block >= self.threshold).sum()) - n     # drop the diagonal
            out[act] = {
                "n": n,
                "pair_rate_above_threshold": above / max(1, n * (n - 1)),
                "mean_neighbours_above_threshold": above / n,
            }
        return {
            "threshold": self.threshold,
            "embed_model_key": "compile.dedup_embed_model",
            "per_act": out,
            "note": (
                "a high pair rate would mean the tier is loose and its numbers are not "
                "evidence of a near-miss. On this bank it is expected to be LOW -- "
                "dedup_threshold is strict and single-linkage chained rather than made "
                "dense clusters (DEFECTS_OPEN D-7)."
            ),
        }

    def bank_facts(self) -> dict:
        """How discriminative each predicate is on THIS bank -- report it with the tiers.

        A predicate that holds almost everywhere is not evidence of a near-miss,
        it is evidence that the predicate is loose.
        """
        field_bearing = [tid for tid, f in self.fields.items() if f]
        return {
            "n_templates": len(self.ids),
            "n_requesting_a_named_field": len(field_bearing),
            "field_safe_is_vacuous_for": len(self.ids) - len(field_bearing),
            "note": (
                "field_safe is trivially true for any pair of templates that request no "
                "named field. Quote the wrong-field RATE over field-requesting gold rows, "
                "not over all rows, or the denominator hides the event D16 is about."
            ),
        }


# --------------------------------------------------------------------------- #
# Composed (turn-level) metric
# --------------------------------------------------------------------------- #


def compose_at_1(
    turn_rows: dict,
    skeleton_correct: dict,
    template_correct: dict,
    all_turns: Sequence[str],
) -> dict:
    """Share of turns where the skeleton AND every template position are exact.

    Args:
        turn_rows: ``turn_id -> [position, ...]`` positions belonging to the turn.
        skeleton_correct: ``turn_id -> bool``.
        template_correct: ``(turn_id, position) -> bool``.
        all_turns: every retrieve turn id, INCLUDING turns with no gold anywhere.
            The unconditional denominator uses this; the conditional one uses the
            fully-covered subset.

    Returns both denominators, because this is the number most likely to be
    quoted on its own.
    """
    covered = [t for t in turn_rows if all(template_correct.get((t, p)) is not None
                                           for p in turn_rows[t])]
    def _ok(turn_id):
        return bool(skeleton_correct.get(turn_id)) and all(
            template_correct.get((turn_id, p), False) for p in turn_rows[turn_id]
        )
    cond_hits = sum(1 for t in covered if _ok(t))
    uncond_hits = sum(1 for t in all_turns if _ok(t))
    return {
        "conditional": {"n": len(covered), "compose@1": (cond_hits / len(covered)) if covered else None},
        "unconditional": {
            "n": len(all_turns),
            "compose@1": (uncond_hits / len(all_turns)) if all_turns else None,
        },
    }


# --------------------------------------------------------------------------- #
# H4 denominator alignment across windows
# --------------------------------------------------------------------------- #


def align_h4_denominator(arms: dict) -> dict:
    """Intersect the scorable row sets of several H4 arms.

    ``arms`` is ``arm_label -> set(turn_id)`` of rows with ``gold_column >= 0``.
    The copy tier reads the context window, so a narrower K resolves fewer golds;
    comparing arms on their own denominators compares two different questions.
    Returns the intersection plus, for each arm, how many rows it loses to it --
    which is itself a finding about how much of H4 is copy-bound.
    """
    if not arms:
        return {"intersection": set(), "per_arm_loss": {}}
    shared = set.intersection(*[set(v) for v in arms.values()])
    return {
        "intersection_size": len(shared),
        "intersection": shared,
        "per_arm": {
            label: {"scorable": len(rows), "lost_to_intersection": len(set(rows) - shared)}
            for label, rows in arms.items()
        },
        "note": (
            "report H4 across windows on the intersection. An arm's own scorable count is "
            "still reported, because a window that resolves FEWER golds is a real cost of "
            "that window and must not be hidden by the alignment."
        ),
    }


# --------------------------------------------------------------------------- #
# Conversation-clustered bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    per_row: Sequence[float],
    cluster_ids: Sequence[Any],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile CI, resampling CONVERSATIONS with replacement.

    Args:
        per_row: 1/0 (or any per-row score) aligned with ``cluster_ids``.
        cluster_ids: the conversation id of each row. Rows sharing one are
            resampled together.

    Why clustered: H7 positions inside a turn share their context string exactly,
    and turns inside a conversation share the scenario, the customer and the
    intent. A row-level bootstrap (or a binomial CI) treats them as independent
    and reports an interval that is too narrow. The CI quoted in D7 (+-0.68 pts
    at n=13,284) is the row-level binomial value; expect this one to be wider.
    """
    import numpy as np

    values = np.asarray(list(per_row), dtype=float)
    if values.size == 0:
        return {"point": None, "lo": None, "hi": None, "n_rows": 0, "n_clusters": 0}
    keys = list(cluster_ids)
    groups: dict = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    cluster_keys = list(groups)
    index_lists = [np.asarray(groups[k], dtype=np.int64) for k in cluster_keys]
    sums = np.asarray([values[idx].sum() for idx in index_lists])
    sizes = np.asarray([idx.size for idx in index_lists], dtype=float)

    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_keys)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot_sums = sums[draws].sum(axis=1)
    boot_sizes = sizes[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.where(boot_sizes > 0, boot_sums / boot_sizes, np.nan)
    stats = stats[~np.isnan(stats)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(values.mean())
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "half_width_points": float((hi - lo) / 2 * 100),
        "n_rows": int(values.size),
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "row_level_binomial_half_width_points": float(
            1.96 * math.sqrt(max(point * (1 - point), 0.0) / values.size) * 100
        ),
        "note": (
            "clustered by conversation. row_level_binomial_half_width_points is what an "
            "independence assumption would have given -- reported so the difference is "
            "visible, never as the interval."
        ),
    }


def paired_bootstrap_delta(
    a_per_row: Sequence[float],
    b_per_row: Sequence[float],
    cluster_ids: Sequence[Any],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI on ``mean(a) - mean(b)`` over the SAME rows, clustered by conversation.

    Two arms measured on the same rows are correlated; an unpaired comparison of
    two independent CIs is the wrong test and will call real differences
    insignificant. Every arm-vs-arm claim in this harness goes through here.
    """
    import numpy as np

    a = np.asarray(list(a_per_row), dtype=float)
    b = np.asarray(list(b_per_row), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs aligned rows, got {a.shape} and {b.shape}")
    diff = a - b
    stats = bootstrap_ci(diff, cluster_ids, n_boot=n_boot, alpha=alpha, seed=seed)
    stats["delta_points"] = (stats["point"] or 0.0) * 100
    stats["significant"] = (
        stats["lo"] is not None and (stats["lo"] > 0 or stats["hi"] < 0)
    )
    return stats


# --------------------------------------------------------------------------- #
# The guard that stops a vacuous control being reported as evidence
# --------------------------------------------------------------------------- #


def order_sensitivity_headroom(
    vectorizer: Any,
    natural_texts: Sequence[str],
    shuffled_texts: Sequence[str],
    min_headroom: float = 0.01,
) -> dict:
    """How much of the representation CAN respond to a line permutation at all.

    D7's first two order controls were incapable of failing and both were nearly
    reported as evidence: mean pooling is permutation-invariant by construction,
    and plain word unigrams are EXACTLY 0.00% order-sensitive because permuting
    lines cannot change a multiset of tokens. A null from a control like that is
    arithmetic, not a finding.

    This measures the share of TF-IDF mass that MOVES between the natural and the
    shuffled rendering of the same rows.

    **``shuffled_texts`` must come from the FEATURIZER, rendered with
    ``shuffle="within_window"`` -- not from permuting the natural strings here.**
    That distinction is the whole measurement: under recency tagging the bucket
    must be recomputed from the token's NEW position, and permuting an
    already-rendered string carries each line's old tag along with it, which
    makes a tagged representation look exactly as order-insensitive as a plain
    one. Measured on a synthetic sample, the string-permuting version reported
    0.087 tagged vs 0.089 plain -- i.e. it showed tagging buying nothing, which
    is the opposite of D7's finding and is an artefact of the shortcut.

    Returns ``headroom`` in [0, 1]. A control run below
    ``probe.min_order_headroom`` must be reported as VACUOUS, not as a null:
    the caller passes that config value as ``min_headroom`` (the default mirrors
    probes/probe.yaml) and the threshold used is echoed in the result.
    Expect roughly D7's 5.5% for plain uni+bigrams (only the bigrams straddling
    turn junctions can move) and much more under tagging.
    """
    natural = list(natural_texts)
    shuffled = list(shuffled_texts)
    if not natural or len(natural) != len(shuffled):
        raise ValueError(
            f"need aligned natural/shuffled renderings, got {len(natural)} and {len(shuffled)}"
        )

    X_nat = vectorizer.transform(natural)
    X_shuf = vectorizer.transform(shuffled)
    total = float(abs(X_nat).sum())
    moved = float(abs(X_nat - X_shuf).sum())
    headroom = (moved / total) if total else 0.0
    identical = sum(1 for a, b in zip(natural, shuffled) if a == b)
    return {
        "headroom": headroom,
        "n_sampled": len(natural),
        "n_rows_shuffle_was_a_no_op": identical,
        "vacuous": headroom < float(min_headroom),
        "min_headroom": float(min_headroom),
        "note": (
            "share of TF-IDF mass that moves between the natural and the featurizer's "
            "shuffled rendering. 0 means the representation is permutation-invariant and "
            "the order control CANNOT fail; a null from it would be arithmetic, not a "
            "finding (DECISIONS D7). Rows whose window holds 0 or 1 turn cannot shuffle "
            "and are counted separately -- a sample dominated by them deflates the number."
        ),
    }


# --------------------------------------------------------------------------- #
# D5 verdict
# --------------------------------------------------------------------------- #


def d5_verdict(measured: Optional[float], constant: Optional[float],
               ci_lo: Optional[float] = None) -> dict:
    """Does this metric survive the constant-predictor guard?

    ``DROP`` when the constant wins or ties within the interval. A metric that
    lands here is removed from the report, per D5 -- not reported with a
    footnote.
    """
    if measured is None or constant is None:
        return {"verdict": "UNMEASURED", "margin_points": None}
    margin = (measured - constant) * 100
    if measured <= constant:
        verdict = "DROP -- a label-blind constant wins"
    elif ci_lo is not None and ci_lo <= constant:
        verdict = "DROP -- the constant is inside the CI"
    else:
        verdict = "KEEP"
    return {
        "verdict": verdict,
        "measured": measured,
        "constant": constant,
        "margin_points": margin,
        "ci_lo": ci_lo,
    }
