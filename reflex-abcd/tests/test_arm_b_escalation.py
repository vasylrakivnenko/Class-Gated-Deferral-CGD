"""Regression tests for the Arm B escalation-wiring fix.

BEFORE this fix, ``_run_arm_b`` (src/reflex/run.py) never called
``reflex.llm_agent.llm_decide`` on escalation: with ``llm.enabled`` true it
still unconditionally raised ``LLMDisabledError`` (the ``if not forced_reflex``
branch had no ``llm_enabled`` check at all), and ``_build_decision`` had no
shape that could carry an LLM's actual answer even if one had been obtained --
its only two branches were "fast path answered" and "unanswered, everything
withheld". So an enabled LLM was a dead end: constructible, individually
callable (``llm_decide`` was already unit-tested in isolation, see
test_no_paid_calls.py), but never reached by the orchestrator.

WHY THIS FILE TESTS ``_build_decision`` DIRECTLY, NOT ``_run_arm_b`` END TO END:
there is no trained checkpoint on this machine (see test_run_report.py's own
top-of-file note) -- ``build_selector`` requires one and none exists in this
repo checkout, so a live ``_run_arm_b`` run is not exercisable here regardless
of this fix (that gap is tracked separately: the certified TF-IDF+logreg
selector was never wired to this runtime path at all). ``_build_decision`` is
where the actual bug lived -- the decision-assembly logic that turns a
(selection, gate_output, llm_decision) triple into the Decision that gets
scored -- and it is a pure function, so it is tested directly and rigorously
here without needing the corpus or a checkpoint.
"""

from __future__ import annotations

from reflex.run import _build_decision
from reflex.schemas import GateOutput, LLMDecision, Selection


def _selection(**overrides) -> Selection:
    base = dict(
        nextstep="retrieve_utterance",
        intent="status_delivery_time",
        action=None,
        values=None,
        skeleton_id="SK001",
        template_ids=["T000123"],
        composed_text_delex="your order is on the way.",
        candidate_utt_id=42,
        candidate_rank=7,
        exact_template_match=None,
    )
    base.update(overrides)
    return Selection(**base)


def _gate_output(**overrides) -> GateOutput:
    base = dict(
        route="escalated",
        reason="low_confidence",
        set_sizes={"nextstep": 1, "intent": 2, "action": 0, "skeleton": 3, "templates": [2]},
        novelty_distance=0.31,
        missing_slots=[],
    )
    base.update(overrides)
    return GateOutput(**base)


def _llm_decision(**overrides) -> LLMDecision:
    base = dict(
        nextstep="retrieve_utterance",
        intent="status_delivery_time",
        action=None,
        values=None,
        candidate_index=3,
        utterance_text="your package should arrive by friday.",
        tokens_in=812,
        tokens_out=47,
        latency_ms=642.5,
        parse_failed=False,
        cache_hit=False,
        model_id="test-model-id",
        prompt_hash="deadbeef01234567",
    )
    base.update(overrides)
    return LLMDecision(**base)


def test_answered_escalation_uses_the_llm_decision_not_the_rejected_selection() -> None:
    """The core fix: an LLM answer, when present, must populate the Decision.

    Every field a reader would use to judge Arm B's ANSWERED-escalation
    quality (nextstep/intent/action/values/utterance/candidate_rank/tokens/
    latency) must come from the LLM's own parsed response -- never from the
    fast-path ``selection`` the gate just rejected.
    """
    selection = _selection(nextstep="take_action", intent="manage_dispute", action="offer-refund",
                            values=["17.50"])
    gate_output = _gate_output()
    llm = _llm_decision()

    decision = _build_decision(
        convo_id=1001,
        turn_index=4,
        selection=selection,
        gate_output=gate_output,
        utterance_text=None,
        elapsed_ms=12.3,
        escalated=True,
        llm_decision=llm,
    )

    assert decision.route == "escalated"
    # Every one of these must trace to `llm`, NOT to `selection` (which says
    # take_action/manage_dispute/offer-refund -- if any of these read that
    # instead, the fix regressed to showing the rejected fast-path guess).
    assert decision.nextstep == "retrieve_utterance"
    assert decision.intent == "status_delivery_time"
    assert decision.action is None
    assert decision.values is None
    assert decision.utterance_text == "your package should arrive by friday."
    assert decision.candidate_rank == 3
    assert decision.llm_tokens_in == 812
    assert decision.llm_tokens_out == 47
    assert decision.latency_ms_llm == 642.5
    assert decision.latency_ms_fastpath == 12.3
    assert decision.cache_hit is False
    # The LLM answers freely; it does not compose from the bank.
    assert decision.skeleton_id is None
    assert decision.template_ids is None


def test_answered_escalation_take_action_carries_action_and_values() -> None:
    """A take_action escalation must carry the LLM's action/values through."""
    llm = _llm_decision(
        nextstep="take_action", intent="manage_dispute", action="offer-refund",
        values=["17.50"], candidate_index=-1, utterance_text=None,
    )
    decision = _build_decision(
        convo_id=2, turn_index=1, selection=_selection(), gate_output=_gate_output(),
        utterance_text=None, elapsed_ms=5.0, escalated=True, llm_decision=llm,
    )
    assert decision.action == "offer-refund"
    assert decision.values == ["17.50"]
    assert decision.candidate_rank == -1


def test_unanswered_escalation_still_withholds_everything_but_h1_h2() -> None:
    """Regression guard: forced-reflex's unanswered shape must not have moved.

    ``llm_decision=None`` (the forced-reflex / kill-switch-on path) must
    produce EXACTLY the old withheld-everything Decision -- this fix must not
    have changed behavior for the case it was not touching.
    """
    selection = _selection(nextstep="take_action", intent="manage_dispute")
    decision = _build_decision(
        convo_id=3, turn_index=2, selection=selection, gate_output=_gate_output(),
        utterance_text=None, elapsed_ms=9.9, escalated=True, llm_decision=None,
    )
    assert decision.route == "escalated"
    assert decision.nextstep == "take_action"
    assert decision.intent == "manage_dispute"
    assert decision.action is None
    assert decision.values is None
    assert decision.skeleton_id is None
    assert decision.template_ids is None
    assert decision.utterance_text is None
    assert decision.candidate_utt_id is None
    assert decision.llm_tokens_in == 0
    assert decision.llm_tokens_out == 0
    assert decision.latency_ms_llm == 0.0
    assert decision.candidate_rank == -1
    assert decision.cache_hit is False


def test_fast_path_decision_is_unaffected_by_the_llm_decision_parameter() -> None:
    """Regression guard: a non-escalated (reflex-route) turn must be unchanged."""
    selection = _selection()
    decision_without = _build_decision(
        convo_id=4, turn_index=0, selection=selection, gate_output=_gate_output(),
        utterance_text="your order is on the way.", elapsed_ms=1.5, escalated=False,
    )
    decision_with_ignored = _build_decision(
        convo_id=4, turn_index=0, selection=selection, gate_output=_gate_output(),
        utterance_text="your order is on the way.", elapsed_ms=1.5, escalated=False,
        llm_decision=_llm_decision(),
    )
    # `llm_decision` must be irrelevant on the fast path -- it is only
    # consulted when `escalated` is True.
    assert decision_without == decision_with_ignored
    assert decision_without.route == "reflex"
    assert decision_without.candidate_rank == 7
    assert decision_without.skeleton_id == "SK001"
