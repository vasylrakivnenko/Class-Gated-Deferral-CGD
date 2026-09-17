"""Spec 4.6 tests. The gate is the ONLY decision point, so every one of these is a
routing invariant somebody could break without any metric noticing.

What is pinned here:
  * precedence ``novel > unseen_action > unavailable_slot > low_confidence``
    when several signals fail at once -- the case a single-signal test misses;
  * an EMPTY conformal set escalates and is never "fixed" by inserting the
    argmax (that would silently break coverage, spec 6.6);
  * only the heads the predicted nextstep requires are consulted, and the rest
    report 0 / [] rather than a missing key;
  * the gate never modifies the scores it routes on;
  * the two uses of confidence stay separable: each signal can be ablated on its
    own and is still REPORTED when it is no longer routed on;
  * an alpha the calibration never computed raises instead of gating at the
    wrong quantile.
"""

from __future__ import annotations

import copy
import json

import pytest

from reflex.config import apply_overrides, get_dotted, load_config
from reflex.contracts import ContractViolation
from reflex.gate import evaluate_gate
from reflex.schemas import Calibration, Selection, SelectorScores

CONFIG = "configs/default.yaml"

#: alpha 0.02 -> q 0.02 -> a class is in the set at p >= 0.98.
_Q = 0.02


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


def _calibration(alpha: float = 0.02, novelty_threshold: float = 0.5, **quantiles) -> Calibration:
    heads = {"nextstep": _Q, "intent": _Q, "action": _Q, "skeleton": _Q, "template": _Q}
    heads.update(quantiles)
    return Calibration(
        alpha=alpha,
        quantiles=heads,
        novelty_threshold=novelty_threshold,
        n_dev_turns={head: 1000 for head in heads},
        checkpoint_path="outputs/checkpoints/fake.pt",
        seed=1,
    )


def _confident(width: int, index: int = 0) -> list[float]:
    """A distribution with one class at 0.99 -- set size 1 at q=0.02."""
    rest = (1.0 - 0.99) / (width - 1)
    probs = [rest] * width
    probs[index] = 0.99
    return probs


def _diffuse(width: int) -> list[float]:
    """A flat distribution -- EMPTY set at q=0.02, which must escalate."""
    return [1.0 / width] * width


def _scores(nextstep: str, **overrides) -> SelectorScores:
    base = dict(
        convo_id=1,
        turn_index=4,
        nextstep_probs=_confident(3),
        intent_probs=_confident(55),
        novelty_distance=0.1,
        context_hash="abc123",
    )
    if nextstep == "take_action":
        base["action_probs"] = _confident(30)
    if nextstep == "retrieve_utterance":
        base["skeleton_probs"] = _confident(570)
        base["template_probs"] = [_confident(12)]
        base["template_candidates"] = [["T000001"] * 12]
    base.update(overrides)
    return SelectorScores(**base)


def _selection(nextstep: str, **overrides) -> Selection:
    base: dict = dict(nextstep=nextstep, intent="return_size")
    if nextstep == "take_action":
        base["action"] = "verify-identity"
    if nextstep == "retrieve_utterance":
        base["skeleton_id"] = "S0000"
        base["template_ids"] = ["T000001"]
    base.update(overrides)
    return Selection(**base)


# --------------------------------------------------------------------------- #
# The happy path and the head-applicability table
# --------------------------------------------------------------------------- #


def test_all_signals_pass_routes_to_reflex(cfg: dict) -> None:
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False, _calibration(), cfg
    )
    assert out.route == "reflex"
    assert out.reason == "ok"
    assert out.missing_slots == []
    assert out.set_sizes == {
        "nextstep": 1, "intent": 1, "action": 0, "skeleton": 1, "templates": [1],
    }


@pytest.mark.parametrize(
    "nextstep,expected",
    [
        ("retrieve_utterance", {"nextstep": 1, "intent": 1, "action": 0, "skeleton": 1, "templates": [1]}),
        ("take_action", {"nextstep": 1, "intent": 1, "action": 1, "skeleton": 0, "templates": []}),
        ("end_conversation", {"nextstep": 1, "intent": 1, "action": 0, "skeleton": 0, "templates": []}),
    ],
)
def test_only_the_required_heads_are_consulted(cfg: dict, nextstep: str, expected: dict) -> None:
    """A head that does not apply reports 0 (or []), never a missing key."""
    out = evaluate_gate(_scores(nextstep), _selection(nextstep), [], False, _calibration(), cfg)
    assert out.set_sizes == expected
    assert out.route == "reflex"


def test_an_inapplicable_head_cannot_escalate(cfg: dict) -> None:
    """A diffuse skeleton head on a take_action turn is not consulted at all."""
    scores = _scores("take_action", skeleton_probs=_diffuse(570))
    out = evaluate_gate(scores, _selection("take_action"), [], False, _calibration(), cfg)
    assert out.route == "reflex"
    assert out.set_sizes["skeleton"] == 0


# --------------------------------------------------------------------------- #
# Signal 1: conformal confidence
# --------------------------------------------------------------------------- #


def test_empty_prediction_set_escalates_and_is_not_patched_with_the_argmax(cfg: dict) -> None:
    """Spec 6.6: an empty set is |set| != 1. Inserting the argmax breaks coverage."""
    scores = _scores("take_action", intent_probs=_diffuse(55))
    out = evaluate_gate(scores, _selection("take_action"), [], False, _calibration(), cfg)
    assert out.set_sizes["intent"] == 0
    assert (out.route, out.reason) == ("escalated", "low_confidence")


def test_set_larger_than_one_escalates(cfg: dict) -> None:
    probs = [0.0] * 55
    probs[3] = probs[7] = 0.5
    out = evaluate_gate(
        _scores("take_action", intent_probs=probs),
        _selection("take_action"),
        [], False,
        _calibration(intent=0.6),  # q=0.6 -> in the set at p >= 0.4
        cfg,
    )
    assert out.set_sizes["intent"] == 2
    assert out.reason == "low_confidence"


def test_one_bad_act_position_escalates_the_whole_turn(cfg: dict) -> None:
    scores = _scores("retrieve_utterance", template_probs=[_confident(12), _diffuse(12)])
    selection = _selection("retrieve_utterance", template_ids=["T000001", "T000002"])
    out = evaluate_gate(scores, selection, [], False, _calibration(), cfg)
    assert out.set_sizes["templates"] == [1, 0]
    assert out.reason == "low_confidence"


def test_an_act_position_with_no_candidate_escalates_rather_than_raising(cfg: dict) -> None:
    """No template for that act is a COVERAGE fact (61.01% ceiling), not a bug."""
    scores = _scores("retrieve_utterance", template_probs=[_confident(12), []])
    selection = _selection("retrieve_utterance", template_ids=["T000001", "T000002"])
    out = evaluate_gate(scores, selection, [], False, _calibration(), cfg)
    assert out.set_sizes["templates"] == [1, 0]
    assert out.reason == "low_confidence"


# --------------------------------------------------------------------------- #
# Signal 2: novelty
# --------------------------------------------------------------------------- #


def test_novelty_is_strictly_greater_than_the_threshold(cfg: dict) -> None:
    calibration = _calibration(novelty_threshold=0.5)
    at = evaluate_gate(
        _scores("take_action", novelty_distance=0.5), _selection("take_action"), [], False, calibration, cfg
    )
    over = evaluate_gate(
        _scores("take_action", novelty_distance=0.5000001), _selection("take_action"), [], False, calibration, cfg
    )
    assert at.reason == "ok"
    assert over.reason == "novel"
    assert over.route == "escalated"


# --------------------------------------------------------------------------- #
# Signal 3: availability (computed in fill / by the caller, consumed here)
# --------------------------------------------------------------------------- #


def test_missing_slots_escalate_as_unavailable_slot(cfg: dict) -> None:
    out = evaluate_gate(
        _scores("take_action"), _selection("take_action"), ["account_id"], False, _calibration(), cfg
    )
    assert (out.reason, out.missing_slots) == ("unavailable_slot", ["account_id"])


def test_unseen_action_escalates(cfg: dict) -> None:
    out = evaluate_gate(
        _scores("take_action"), _selection("take_action"), [], True, _calibration(), cfg
    )
    assert out.reason == "unseen_action"


# --------------------------------------------------------------------------- #
# Precedence -- the case a per-signal test cannot see
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "distance,unseen,missing,expected",
    [
        (0.9, True, ["account_id"], "novel"),
        (0.1, True, ["account_id"], "unseen_action"),
        (0.1, False, ["account_id"], "unavailable_slot"),
        (0.1, False, [], "low_confidence"),
    ],
)
def test_precedence_when_several_signals_fail(
    cfg: dict, distance: float, unseen: bool, missing: list, expected: str
) -> None:
    """novel > unseen_action > unavailable_slot > low_confidence (spec 6.6)."""
    scores = _scores("take_action", intent_probs=_diffuse(55), novelty_distance=distance)
    out = evaluate_gate(scores, _selection("take_action"), missing, unseen, _calibration(), cfg)
    assert out.reason == expected
    assert out.route == "escalated"
    # Every signal is still reported, whichever one won the precedence race.
    assert out.set_sizes["intent"] == 0
    assert out.novelty_distance == distance
    assert out.missing_slots == missing


# --------------------------------------------------------------------------- #
# The gate never modifies scores, and is deterministic
# --------------------------------------------------------------------------- #


def test_gate_does_not_modify_scores_or_selection(cfg: dict) -> None:
    scores = _scores("retrieve_utterance")
    selection = _selection("retrieve_utterance")
    before = (copy.deepcopy(scores.to_dict()), copy.deepcopy(selection.to_dict()))
    evaluate_gate(scores, selection, ["account_id"], True, _calibration(), cfg)
    assert (scores.to_dict(), selection.to_dict()) == before


def test_same_input_same_decision(cfg: dict) -> None:
    """Spec 10 determinism: same context -> same Decision, always."""
    args = (_scores("retrieve_utterance"), _selection("retrieve_utterance"), ["x"], False, _calibration(), cfg)
    assert evaluate_gate(*args).to_dict() == evaluate_gate(*args).to_dict()


# --------------------------------------------------------------------------- #
# E3b ablations: one signal at a time, and a disabled signal is still MEASURED
# --------------------------------------------------------------------------- #


def test_no_novelty_ablation_still_reports_the_distance(cfg: dict) -> None:
    ablated = apply_overrides(cfg, ["gate.use_novelty=false"])
    scores = _scores("take_action", novelty_distance=0.9)
    out = evaluate_gate(scores, _selection("take_action"), [], False, _calibration(), ablated)
    assert out.route == "reflex"
    assert out.novelty_distance == 0.9  # measured, just not routed on


def test_no_availability_ablation_still_reports_missing_slots(cfg: dict) -> None:
    ablated = apply_overrides(cfg, ["gate.use_availability=false"])
    out = evaluate_gate(
        _scores("take_action"), _selection("take_action"), ["account_id"], True, _calibration(), ablated
    )
    assert out.route == "reflex"
    assert out.missing_slots == ["account_id"]


def test_raw_softmax_threshold_ablation_replaces_the_conformal_set(cfg: dict) -> None:
    ablated = apply_overrides(
        cfg, ["gate.confidence_mode=softmax_threshold", "gate.softmax_threshold_ablation=0.995"]
    )
    # p_max = 0.99 clears the calibrated q=0.02 bar but not the raw 0.995 one.
    conformal = evaluate_gate(
        _scores("take_action"), _selection("take_action"), [], False, _calibration(), cfg
    )
    raw = evaluate_gate(
        _scores("take_action"), _selection("take_action"), [], False, _calibration(), ablated
    )
    assert conformal.route == "reflex"
    assert (raw.route, raw.reason) == ("escalated", "low_confidence")
    assert raw.set_sizes["nextstep"] == 0


def test_unknown_confidence_mode_raises(cfg: dict) -> None:
    broken = apply_overrides(cfg, ["gate.confidence_mode=whatever"])
    with pytest.raises(ValueError, match="gate.confidence_mode"):
        evaluate_gate(_scores("take_action"), _selection("take_action"), [], False, _calibration(), broken)


# --------------------------------------------------------------------------- #
# Loud failures: the silent-wrongness cases
# --------------------------------------------------------------------------- #


def test_affect_signal_must_be_off(cfg: dict) -> None:
    on = apply_overrides(cfg, ["gate.affect_signal=\"on\""])
    with pytest.raises(ValueError, match="affect_signal"):
        evaluate_gate(_scores("take_action"), _selection("take_action"), [], False, _calibration(), on)


def test_logits_are_refused(cfg: dict) -> None:
    """SelectorScores: "Do not pass logits". On logits the conformal rule is noise."""
    with pytest.raises(ContractViolation, match="probabilit"):
        evaluate_gate(
            _scores("take_action", nextstep_probs=[2.5, -1.0, 0.3]),
            _selection("take_action"), [], False, _calibration(), cfg,
        )


def test_applicable_head_without_scores_raises(cfg: dict) -> None:
    scores = _scores("retrieve_utterance", skeleton_probs=[])
    with pytest.raises(ContractViolation, match="skeleton"):
        evaluate_gate(scores, _selection("retrieve_utterance"), [], False, _calibration(), cfg)


def test_missing_quantile_raises(cfg: dict) -> None:
    calibration = Calibration(alpha=0.02, quantiles={"nextstep": _Q}, novelty_threshold=0.5)
    with pytest.raises(ContractViolation, match="intent"):
        evaluate_gate(_scores("take_action"), _selection("take_action"), [], False, calibration, cfg)


def test_unknown_nextstep_raises(cfg: dict) -> None:
    bogus = Selection(nextstep="chit_chat", intent="return_size")
    with pytest.raises(ContractViolation, match="nextstep"):
        evaluate_gate(_scores("take_action"), bogus, [], False, _calibration(), cfg)


# --------------------------------------------------------------------------- #
# E5: sweeping gate.alpha must not silently reuse the calibrated quantiles
# --------------------------------------------------------------------------- #


def test_alpha_sweep_quantiles_are_used_when_the_run_alpha_differs(cfg: dict) -> None:
    calibration = Calibration(
        alpha=0.02,
        quantiles={h: _Q for h in ("nextstep", "intent", "action", "skeleton", "template")},
        alpha_sweep_quantiles={
            repr(0.05): {h: 0.6 for h in ("nextstep", "intent", "action", "skeleton", "template")}
        },
        novelty_threshold=0.5,
    )
    swept = apply_overrides(cfg, ["gate.alpha=0.05"])
    probs = [0.0] * 55
    probs[3] = probs[7] = 0.5
    out = evaluate_gate(
        _scores("take_action", intent_probs=probs), _selection("take_action"), [], False, calibration, swept
    )
    # q=0.6 (the swept value), not q=0.02: both 0.5-mass classes are in the set.
    assert out.set_sizes["intent"] == 2


def test_alpha_without_calibrated_quantiles_raises(cfg: dict) -> None:
    swept = apply_overrides(cfg, ["gate.alpha=0.123"])
    with pytest.raises(ContractViolation, match="alpha"):
        evaluate_gate(_scores("take_action"), _selection("take_action"), [], False, _calibration(), swept)


# --------------------------------------------------------------------------- #
# Cross-module consistency: the selector and the gate must agree on the turn
# --------------------------------------------------------------------------- #


def test_template_position_count_must_match_the_selection(cfg: dict) -> None:
    """Two distributions for a one-position skeleton is a select bug, not a turn."""
    scores = _scores("retrieve_utterance", template_probs=[_confident(12), _confident(12)])
    with pytest.raises(ContractViolation, match="template_ids"):
        evaluate_gate(scores, _selection("retrieve_utterance"), [], False, _calibration(), cfg)


# --------------------------------------------------------------------------- #
# DECISIONS D16 backstop: a template whose merge cluster is internally inconsistent.
# Conformal confidence CANNOT catch this -- the model is confident and the cluster
# is what is wrong -- so it is an availability-class signal, not a confidence one.
# --------------------------------------------------------------------------- #


def test_no_sidecar_configured_means_no_check(cfg: dict) -> None:
    """The shipped default: the BUILD-time merge guard is the defence."""
    assert get_dotted(cfg, "gate.template_consistency_path") is None
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False, _calibration(), cfg
    )
    assert out.route == "reflex"


def test_flagged_template_escalates_as_availability(cfg: dict, tmp_path) -> None:
    sidecar = tmp_path / "inconsistent.json"
    sidecar.write_text(json.dumps(["T000001"]), encoding="utf-8")
    flagged = apply_overrides(cfg, [f"gate.template_consistency_path={sidecar}"])
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False, _calibration(), flagged
    )
    assert (out.route, out.reason) == ("escalated", "unavailable_slot")
    assert out.set_sizes["templates"] == [1]  # every head was perfectly confident


def test_unflagged_template_still_routes(cfg: dict, tmp_path) -> None:
    sidecar = tmp_path / "inconsistent.json"
    sidecar.write_text(json.dumps({"T999999": True, "T000001": False}), encoding="utf-8")
    flagged = apply_overrides(cfg, [f"gate.template_consistency_path={sidecar}"])
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False, _calibration(), flagged
    )
    assert out.route == "reflex"


def test_a_configured_sidecar_that_does_not_exist_raises(cfg: dict, tmp_path) -> None:
    """A guard that cannot fire is worse than no guard."""
    broken = apply_overrides(cfg, [f"gate.template_consistency_path={tmp_path / 'nope.json'}"])
    with pytest.raises(ContractViolation, match="does not exist"):
        evaluate_gate(
            _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False,
            _calibration(), broken,
        )


# --------------------------------------------------------------------------- #
# The YAML boolean-coercion trap that broke every route once already
# --------------------------------------------------------------------------- #


def test_affect_signal_must_be_a_string_not_a_yaml_boolean(cfg: dict) -> None:
    """Bare `off` in YAML 1.1 is the BOOLEAN False. The error must name the trap."""
    coerced = apply_overrides(cfg, ["gate.affect_signal=off"])
    assert get_dotted(coerced, "gate.affect_signal") is False
    with pytest.raises(ValueError, match="YAML"):
        evaluate_gate(_scores("take_action"), _selection("take_action"), [], False, _calibration(), coerced)


def test_shipped_config_routes(cfg: dict) -> None:
    """Regression for the defect that shipped: every call raised on the real config."""
    out = evaluate_gate(_scores("take_action"), _selection("take_action"), [], False, _calibration(), cfg)
    assert (out.route, out.reason) == ("reflex", "ok")


# --------------------------------------------------------------------------- #
# DECISIONS D16 backstop: a template flagged as internally inconsistent
#
# Conformal confidence CANNOT catch this -- the model can be perfectly confident
# in a template whose merge cluster asks for the wrong identifiers -- so the
# backstop needs its own regression net.
# --------------------------------------------------------------------------- #


def test_flagged_template_escalates_and_stays_distinguishable(cfg: dict, tmp_path) -> None:
    sidecar = tmp_path / "inconsistent.json"
    sidecar.write_text('["T000001"]', encoding="utf-8")
    guarded = apply_overrides(cfg, [f"gate.template_consistency_path={sidecar}"])
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False,
        _calibration(), guarded,
    )
    assert (out.route, out.reason) == ("escalated", "unavailable_slot")
    # The two causes of `unavailable_slot` stay separable in the metrics: a real
    # slot failure names slots, an inconsistent template names none.
    assert out.missing_slots == []


def test_an_unflagged_template_still_routes(cfg: dict, tmp_path) -> None:
    sidecar = tmp_path / "other.json"
    sidecar.write_text('{"T999999": true, "T000001": false}', encoding="utf-8")
    guarded = apply_overrides(cfg, [f"gate.template_consistency_path={sidecar}"])
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False,
        _calibration(), guarded,
    )
    assert out.route == "reflex"


def test_a_configured_but_absent_sidecar_raises(cfg: dict, tmp_path) -> None:
    """A guard that cannot fire is worse than no guard."""
    guarded = apply_overrides(cfg, [f"gate.template_consistency_path={tmp_path / 'nope.json'}"])
    with pytest.raises(ContractViolation, match="does not exist"):
        evaluate_gate(
            _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False,
            _calibration(), guarded,
        )


def test_the_d16_backstop_has_its_own_switch(cfg: dict, tmp_path) -> None:
    """An E3b arm that removes two mechanisms at once measures neither.

    gate.use_availability=false must switch off slot availability and
    unseen_action ONLY; the template-consistency backstop keeps firing until its
    own flag is cleared.
    """
    sidecar = tmp_path / "inconsistent.json"
    sidecar.write_text(json.dumps(["T000001"]), encoding="utf-8")
    base = [f"gate.template_consistency_path={sidecar}"]

    no_availability = apply_overrides(cfg, base + ["gate.use_availability=false"])
    out = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), ["account_id"], True,
        _calibration(), no_availability,
    )
    assert (out.route, out.reason) == ("escalated", "unavailable_slot")
    assert out.missing_slots == ["account_id"]  # reported, not routed on

    both_off = apply_overrides(cfg, base + [
        "gate.use_availability=false", "gate.use_template_consistency=false",
    ])
    assert evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), ["account_id"], True,
        _calibration(), both_off,
    ).route == "reflex"


def test_the_two_unavailable_slot_meanings_are_discriminated_by_missing_slots(
    cfg: dict, tmp_path
) -> None:
    """Report these as separate rows: missing_slots is EMPTY on the D16 kind."""
    sidecar = tmp_path / "inconsistent.json"
    sidecar.write_text(json.dumps(["T000001"]), encoding="utf-8")
    flagged = apply_overrides(cfg, [f"gate.template_consistency_path={sidecar}"])
    from_template = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), [], False, _calibration(), flagged
    )
    from_slot = evaluate_gate(
        _scores("retrieve_utterance"), _selection("retrieve_utterance"), ["email"], False,
        _calibration(), cfg,
    )
    assert from_template.reason == from_slot.reason == "unavailable_slot"
    assert from_template.missing_slots == []
    assert from_slot.missing_slots == ["email"]
