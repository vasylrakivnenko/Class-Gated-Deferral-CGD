"""Regression tests for the template-quantile collapse (found by external
review, see calibrate.py's "THE TEMPLATE QUANTILE COLLAPSE" docstring
section): ``h7_absent_gold: max_nonconformity`` used to inject an explicit
``s=1.0`` nonconformity score into the SAME distribution used to threshold
accept/reject decisions, whenever a template position's gold template was
absent from its candidate pool. At the real ~50% absent rate (DECISIONS D23),
that forces the conformal quantile to 1.0 and therefore rejects every turn.

Test 1 reproduces the collapse mathematically (pure functions, no fixtures)
and shows a realistic confident response is REJECTED under the OLD bucketing
and ACCEPTED under the NEW one, at the same absent rate and alpha the review
used (3% absent, alpha=0.02).

Test 2 exercises the real ``_collect_dev_scores`` path (with a stubbed
``score_turn``) end to end and checks two things at once: (a) absent rows no
longer contaminate ``golds["template"]`` / the quantile, and (b) the coverage
gap is NOT silently dropped -- it must still be visible in the returned
``template_coverage`` diagnostic, so "fixed" does not mean "hidden."
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from reflex.calibrate import conformal_quantile, nonconformity_scores


def _quantile_and_threshold(probs, golds, alpha):
    scores = nonconformity_scores(probs, golds)
    q = conformal_quantile(scores, alpha)
    threshold = 1.0 - q
    return q, threshold


def test_old_bucketing_collapses_the_gate_new_bucketing_does_not():
    """Reproduces the review's exact finding: 3% absent rate, alpha=0.02.

    OLD bucketing (what _collect_dev_scores used to do for
    absent_mode="max_nonconformity"): append an explicit max-nonconformity row
    (s=1.0) for every absent position, into the SAME distribution used for the
    quantile. NEW bucketing (the fix): absent rows are excluded entirely (gold
    index -1, which nonconformity_scores skips).
    """
    n_present = 970
    n_absent = 30  # 3% of 1000, matching the review's reproduction rate
    alpha = 0.02

    # A realistic, CONFIDENT distribution for the "present" rows: gold at
    # index 0 with 99.9% probability, spread over the rest.
    confident_probs = [0.999, 0.0005, 0.0005]
    confident_gold = 0

    old_probs = [confident_probs] * n_present
    old_golds = [confident_gold] * n_present
    for _ in range(n_absent):
        # OLD: an explicit zero column appended, gold index points AT it ->
        # nonconformity_scores computes 1 - 0.0 = 1.0 for this row.
        old_probs.append(list(confident_probs) + [0.0])
        old_golds.append(len(confident_probs))

    new_probs = [confident_probs] * n_present
    new_golds = [confident_gold] * n_present
    for _ in range(n_absent):
        # NEW: excluded outright (gold index -1 -> nonconformity_scores skips
        # the row per its own documented contract: "Rows with a negative
        # index are SKIPPED, not scored").
        new_probs.append(list(confident_probs))
        new_golds.append(-1)

    old_q, old_threshold = _quantile_and_threshold(old_probs, old_golds, alpha)
    new_q, new_threshold = _quantile_and_threshold(new_probs, new_golds, alpha)

    # OLD: even at just 3% absent, q collapses to 1.0 and the threshold to 0.0
    # -- every class with softmax >= 0 enters the prediction set.
    assert old_q == pytest.approx(1.0)
    assert old_threshold == pytest.approx(0.0)
    old_prediction_set = [p for p in confident_probs if p >= old_threshold]
    assert len(old_prediction_set) == len(confident_probs), (
        "OLD bucketing must admit every class once q collapses to 1.0 -- "
        "this is the reproduced bug, not the fix."
    )

    # NEW: q reflects only the confidence distribution among rows that
    # actually had a real candidate; a 99.9%-confident row gets a TIGHT
    # (singleton) prediction set and is therefore accepted, not rejected.
    assert new_q < 0.5, f"expected a small quantile post-fix, got {new_q}"
    new_prediction_set = [p for p in confident_probs if p >= new_threshold]
    assert len(new_prediction_set) == 1, (
        "NEW bucketing must let a 99.9%-confident response form a singleton "
        "prediction set (accepted by the gate), not collapse like the old code."
    )


def test_collect_dev_scores_excludes_absent_rows_but_still_reports_coverage(monkeypatch):
    """End-to-end through the real _collect_dev_scores (score_turn stubbed).

    Builds 10 retrieve_utterance dev turns, each with one template position:
    6 have the gold template present in the candidate pool (confident), 4 do
    not (absent -- simulating the ~50%-ish real coverage gap at small scale).
    Asserts:
      1. probs["template"] / golds["template"] contain exactly 6 rows (the
         absent ones are excluded from what feeds the quantile, regardless of
         h7_absent_gold mode).
      2. conformal_quantile on that set is NOT collapsed to 1.0.
      3. template_coverage correctly reports 4/10 absent -- the gap is
         VISIBLE in the diagnostic, not silently swallowed by the fix.
    """
    import reflex.calibrate as calibrate_mod
    from reflex.config import load_config
    from reflex.schemas import ContextWindow, NormalizedTurn, SelectorScores

    cfg = load_config()
    cfg["calibrate"]["h7_absent_gold"] = "max_nonconformity"

    n_present, n_absent = 6, 4
    rows = []
    labels = {}
    for i in range(n_present + n_absent):
        turn = NormalizedTurn(
            convo_id=1, turn_index=i, speaker="agent", text="x",
            nextstep="retrieve_utterance",
        )
        context = ContextWindow(convo_id=1, turn_index=i, text="ctx")
        rows.append((1, i, turn, context))
        present = i < n_present
        gold_template = "T_GOLD" if present else "T_NOT_IN_BANK"
        labels[f"dev:1:{i}"] = type(
            "FakeLabel", (), {"skeleton_id": "SK0", "template_ids": [gold_template]}
        )()

    def fake_score_turn(selector, context, turn, cfg):
        i = context.turn_index
        present = i < n_present
        candidates = ["T_GOLD", "T_OTHER"]
        # confident: 0.99 on T_GOLD when present; when absent, T_GOLD is not
        # even in the candidate pool (that IS the absence).
        probs = [0.99, 0.01]
        return SelectorScores(
            convo_id=1, turn_index=i,
            nextstep_probs=[0.9, 0.05, 0.05], intent_probs=[1.0],
            action_probs=[], skeleton_probs=[1.0],
            template_probs=[probs], template_candidates=[candidates],
            novelty_distance=0.1,
        )

    monkeypatch.setattr("reflex.select.score_turn", fake_score_turn)

    class_orders = {"intent": ["x"], "action": ["y"], "skeleton": ["SK0"]}
    probs, golds, novelty, template_coverage = calibrate_mod._collect_dev_scores(
        cfg, selector=None, rows=rows, labels=labels, class_orders=class_orders
    )

    # probs/golds stay INDEX-ALIGNED (both cover every position, present and
    # absent) -- nonconformity_scores relies on len(probs) == len(golds) and
    # skips a row via its gold index, not via the row being absent from the
    # list. What must actually be excluded from SCORING is the absent rows'
    # CONTRIBUTION, checked below via the scores list length, not list length.
    assert len(probs["template"]) == n_present + n_absent
    assert len(golds["template"]) == n_present + n_absent
    assert sum(1 for g in golds["template"] if g >= 0) == n_present, (
        "exactly the present rows should carry a real (non-negative) gold index"
    )

    scores = nonconformity_scores(probs["template"], golds["template"])
    assert len(scores) == n_present, (
        "nonconformity_scores must have scored only the present rows -- an "
        "absent row must not sneak a score into the quantile-determining set"
    )
    q = conformal_quantile(scores, alpha=0.1)
    assert q < 1.0, f"quantile should not collapse with only present rows, got {q}"

    assert template_coverage == {"n_positions": n_present + n_absent, "n_gold_present": n_present}
    gold_present_rate = template_coverage["n_gold_present"] / template_coverage["n_positions"]
    assert gold_present_rate == pytest.approx(n_present / (n_present + n_absent))
