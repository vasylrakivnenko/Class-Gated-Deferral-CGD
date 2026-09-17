"""Tests for reflex.arm_b0 -- the cheap baseline the encoder must beat.

SYNTHETIC FIXTURES ONLY. Nothing here parses the ABCD corpus.

NO ACCURACY NUMBER IS ASSERTED, and none should be. These fixtures are twenty
toy rows; an accuracy measured on them says nothing about ABCD, and
DECISIONS D6 exists precisely because a number was once quoted away from the
configuration that produced it. What is asserted is STRUCTURE: that each head
runs at its measured winning config, that the three configs differ in the two
ways the record says they must, that a score can never leave `evaluate` without
its config and its label-blind constant, and that the plan in code and the plan
in PROPOSED_CONFIG_B0.yaml cannot drift apart.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import yaml

from reflex.arm_b0 import (
    CONFIG_KEYS,
    NO_ACTION,
    ArmB0,
    B0_HEADS,
    ClassifierSpec,
    HeadSpec,
    RowSpec,
    agent_rows,
    default_plan,
    head_label,
)
from reflex.config import load_config, repo_root
from reflex.contracts import ContractViolation
from reflex.featurize import FeaturizeConfig, WindowSpec
from reflex.schemas import NEXT_STEPS, NormalizedTurn

CONFIG = os.environ.get("REFLEX_TEST_CONFIG", "configs/default.yaml")
PROPOSED = os.path.join(repo_root(), "PROPOSED_CONFIG_B0.yaml")


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

# Two "intents" with disjoint vocabulary, and a local cue in the MOST RECENT
# turn that decides the action. A plain bag over the whole thread sees the cue
# wherever it sits; only the recency-tagged view can tell it apart from the same
# word said eight turns ago. That is the structure the fixtures are built to
# have -- not to demonstrate an accuracy, but so the three heads have something
# genuinely different to learn.

_FILLER = [
    "customer|i have been waiting for a while now",
    "agent|thank you for your patience today",
    "customer|no problem at all thanks",
    "agent|let me check the system for you",
]


def _thread(intent: str, cue: str, depth: int) -> list[str]:
    opener = {
        "timing_4": ["customer|when will my shipment arrive", "agent|let me look up the delivery window"],
        "refund_2": ["customer|i want my money back for this order", "agent|i can look at a refund for you"],
    }[intent]
    lines = list(opener)
    for i in range(depth):
        lines.append(_FILLER[i % len(_FILLER)])
    lines.append(f"customer|{cue}")
    return lines


def _fixture_rows() -> tuple[list[list[str]], list[NormalizedTurn]]:
    """Synthetic rows: (context lines, the turn being predicted).

    The no-action rows outnumber the action rows, mirroring ABCD's shape (where
    the label-blind constant for both nextstep and action is 0.7227) so that the
    constant-predictor guard has something real to guard against.
    """
    contexts: list[list[str]] = []
    turns: list[NormalizedTurn] = []
    cues = {
        "pull-up-account": ("here is my account id kwzrzrwfye", (0, 2)),
        "validate-purchase": ("the purchase receipt number is 3348917502", (0, 2)),
        None: ("that all sounds good to me", (0, 1, 2, 3, 4, 5)),
    }
    convo_id = 0
    for intent in ("timing_4", "refund_2"):
        for action, (cue, depths) in cues.items():
            for depth in depths:
                convo_id += 1
                contexts.append(_thread(intent, cue, depth))
                turns.append(
                    NormalizedTurn(
                        convo_id=convo_id,
                        turn_index=len(contexts[-1]),
                        speaker="action" if action else "agent",
                        text=action or "i can help with that",
                        nextstep="take_action" if action else "retrieve_utterance",
                        intent=intent,
                        action=action,
                    )
                )
    return contexts, turns


CONTEXTS, TURNS = _fixture_rows()


@pytest.fixture()
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture()
def proposed() -> dict:
    """PROPOSED_CONFIG_B0.yaml, loaded with a duplicate-key-rejecting loader."""
    with open(PROPOSED, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_NoDuplicateLoader)


class _NoDuplicateLoader(yaml.SafeLoader):
    """A loader that refuses duplicate keys at any depth.

    D18: PyYAML silently keeps only the LAST of two duplicate keys and raises
    nothing, so two agents each added a `select:` block and one agent's entire
    configuration was dead weight while both believed theirs was live.
    """


def _no_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def _flatten(node: Any, prefix: str = "") -> list[str]:
    """Every leaf of a nested mapping as a dotted key."""
    if not isinstance(node, dict):
        return [prefix]
    out: list[str] = []
    for key, value in node.items():
        out.extend(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


# --------------------------------------------------------------------------- #
# The proposed config block
# --------------------------------------------------------------------------- #


def test_proposed_config_parses_and_has_no_duplicate_keys(proposed) -> None:
    """D18's three silent YAML traps, guarded on the file this agent ships."""
    assert set(proposed) == {"baseline_b0"}, "one new top-level block, nothing else"


def test_no_bare_on_off_yes_no_values() -> None:
    """D18: a bare `off` is the BOOLEAN False, not the string 'off'."""
    with open(PROPOSED, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            body = line.split("#", 1)[0]
            if ":" not in body:
                continue
            value = body.split(":", 1)[1].strip()
            assert value.lower() not in {"on", "off", "yes", "no", "y", "n"}, (
                f"{PROPOSED}:{lineno} has a bare {value!r}; quote it or YAML will "
                "coerce it to a bool (D18)"
            )


def test_every_key_the_code_reads_exists_in_the_proposed_block(proposed) -> None:
    """D18: 'a test that every key the code READS actually EXISTS -- twenty such
    keys were missing here, each a latent runtime KeyError'."""
    present = set(_flatten(proposed["baseline_b0"]))
    assert set(CONFIG_KEYS) <= present, f"missing from the YAML: {sorted(set(CONFIG_KEYS) - present)}"


def test_the_proposed_block_has_no_dead_keys(proposed) -> None:
    """The other direction: a key nothing reads is dead weight, and dead weight
    in a config file is how a block silently stops being live."""
    present = set(_flatten(proposed["baseline_b0"]))
    assert present <= set(CONFIG_KEYS), f"nothing reads: {sorted(present - set(CONFIG_KEYS))}"


def test_code_defaults_and_proposed_yaml_agree(proposed) -> None:
    """The library defaults in default_plan() and the shipped config must be the
    same plan, or a run and a test would measure different things."""
    cfg = {"baseline_b0": proposed["baseline_b0"]}
    arm = ArmB0.from_cfg(cfg)
    defaults = default_plan()
    assert set(arm.plan) == set(defaults) == set(B0_HEADS)
    for head, spec in defaults.items():
        assert arm.plan[head].featurize.describe() == spec.featurize.describe(), head
    assert arm.rows == RowSpec()
    assert arm.plan["nextstep"].classifier == ClassifierSpec()


def test_from_cfg_raises_a_useful_error_before_the_block_is_merged(cfg) -> None:
    """configs/default.yaml is coordinator-owned and this agent must not edit it,
    so the un-merged state must fail loudly and say what to do."""
    with pytest.raises(ContractViolation) as excinfo:
        ArmB0.from_cfg(cfg)
    assert "PROPOSED_CONFIG_B0.yaml" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The per-head plan
# --------------------------------------------------------------------------- #


def test_each_head_runs_at_its_own_measured_window() -> None:
    plan = default_plan()
    assert plan["nextstep"].featurize.window == WindowSpec.last_k(6)
    assert plan["intent"].featurize.window == WindowSpec.full()
    assert plan["action"].featurize.window == WindowSpec.full()


def test_intent_is_plain_and_the_other_two_are_tagged() -> None:
    """D7's local/global split, in one assertion: tagging helps the local heads
    and HURTS intent (0.7981 vs 0.8078), which is order-invariant."""
    plan = default_plan()
    assert plan["intent"].featurize.recency.enabled is False
    assert plan["nextstep"].featurize.recency.enabled is True
    assert plan["action"].featurize.recency.enabled is True


def test_every_tagged_head_defaults_to_maxb3() -> None:
    """D7: granularity saturates at 3-4 buckets; past 4 it is flat and costs
    vocabulary (56k -> 148k). Default r0..r3; do not search it hard."""
    for head in ("nextstep", "action"):
        assert default_plan()[head].featurize.recency.max_bucket == 3


def test_intent_documents_carry_no_tag_while_nextstep_documents_do() -> None:
    plan = default_plan()
    intent_doc = _doc(plan["intent"], CONTEXTS[0])
    nextstep_doc = _doc(plan["nextstep"], CONTEXTS[0])
    assert "|" not in intent_doc
    assert any(tok.startswith("r0|") for tok in nextstep_doc.split())


def _doc(spec: HeadSpec, lines: list[str]) -> str:
    from reflex.featurize import build_document

    return build_document(lines, None, spec.featurize)


def test_skeleton_and_template_may_not_be_given_a_plan() -> None:
    """D2b and D7: H5 and H7 are unmeasured on every axis. Measure before
    choosing; do not inherit a default."""
    plan = dict(default_plan())
    plan["skeleton"] = HeadSpec(head="skeleton", featurize=FeaturizeConfig())
    with pytest.raises(ContractViolation) as excinfo:
        ArmB0(plan)
    assert "skeleton" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Rows and labels
# --------------------------------------------------------------------------- #


def test_action_none_becomes_an_explicit_null_class() -> None:
    turn = TURNS[8]
    assert turn.action is None
    assert head_label(turn, "action") == NO_ACTION
    assert head_label(turn, "nextstep") == "retrieve_utterance"
    assert head_label(turn, "intent") == "timing_4"


def test_unknown_head_is_rejected() -> None:
    with pytest.raises(ContractViolation):
        head_label(TURNS[0], "skeleton")


def test_synthetic_end_turns_are_excluded_by_default() -> None:
    real = NormalizedTurn(
        convo_id=1, turn_index=0, speaker="agent", text="hi", nextstep="retrieve_utterance"
    )
    synthetic = NormalizedTurn(
        convo_id=1,
        turn_index=1,
        speaker="agent",
        text="",
        nextstep="end_conversation",
        is_synthetic_end=True,
    )
    partition = {1: [real, synthetic]}
    assert [t.turn_index for _, _, t in agent_rows(partition)] == [0]
    assert [t.turn_index for _, _, t in agent_rows(partition, include_synthetic_end=True)] == [0, 1]


# --------------------------------------------------------------------------- #
# Fit / predict / evaluate
# --------------------------------------------------------------------------- #


def test_arm_fits_all_three_heads_and_predicts_one_label_per_row() -> None:
    arm = ArmB0().fit_rows(CONTEXTS, TURNS)
    predictions = arm.predict(CONTEXTS)
    assert set(predictions) == set(B0_HEADS)
    for head, preds in predictions.items():
        assert len(preds) == len(CONTEXTS), head
        assert all(isinstance(p, str) for p in preds), head


def test_evaluate_never_returns_a_bare_number() -> None:
    """D5's constant-predictor guard and D6's configuration rule, structurally:
    an accuracy cannot leave evaluate() without the constant for the same rows
    and the full config that produced it."""
    arm = ArmB0().fit_rows(CONTEXTS, TURNS)
    result = arm.evaluate(CONTEXTS, arm.labels_from_turns(TURNS))
    for head in B0_HEADS:
        row = result["heads"][head]
        assert {"accuracy", "constant_accuracy", "constant_label", "config", "n_scored"} <= set(row)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["constant_accuracy"] <= 1.0
        assert row["config"]["featurize"]["window"]["label"]
        assert row["config"]["converged"] in (True, False)


def test_evaluate_labels_external_probe_bars_as_external(cfg) -> None:
    """The bars live in the coordinator-owned report.probe_* keys; arm_b0 reads
    them rather than keeping a second copy that could drift, and marks them as
    belonging to the frozen probe harness and not to this run."""
    arm = ArmB0().fit_rows(CONTEXTS, TURNS)
    result = arm.evaluate(CONTEXTS, arm.labels_from_turns(TURNS), cfg=cfg)
    assert "EXTERNAL" in result["external_bars_source"]
    assert result["heads"]["nextstep"]["external_probe_bar"] == cfg["report"]["probe_cheap_bars"]["nextstep"]
    assert result["heads"]["intent"]["external_probe_constant"] == cfg["report"]["probe_constant_bars"]["intent"]


def test_the_label_blind_constant_is_the_training_majority() -> None:
    arm = ArmB0().fit_rows(CONTEXTS, TURNS)
    # 12 of 20 fixture rows carry no action, so the null class is the majority --
    # the same shape as ABCD, where the action head's constant is 0.7227.
    assert arm.heads["action"].constant_label == NO_ACTION


def test_fitting_twice_gives_identical_predictions() -> None:
    a = ArmB0().fit_rows(CONTEXTS, TURNS)
    b = ArmB0().fit_rows(CONTEXTS, TURNS)
    assert a.predict(CONTEXTS) == b.predict(CONTEXTS)
    assert a.predict_proba(CONTEXTS) == b.predict_proba(CONTEXTS)


def test_probabilities_align_to_a_canonical_class_order() -> None:
    """SelectorScores requires each vector to sum to 1 over the head's full
    class list in canonical order, with unseen classes at 0.0."""
    arm = ArmB0().fit_rows(CONTEXTS, TURNS)
    rows = arm.heads["nextstep"].probabilities_in_order(CONTEXTS[:3], NEXT_STEPS)
    assert all(len(r) == len(NEXT_STEPS) for r in rows)
    assert all(abs(sum(r) - 1.0) < 1e-9 for r in rows)
    end_index = NEXT_STEPS.index("end_conversation")
    assert all(r[end_index] == 0.0 for r in rows), "a class never trained on must be 0.0, not dropped"


def test_a_class_outside_the_canonical_order_raises() -> None:
    arm = ArmB0().fit_rows(CONTEXTS, TURNS)
    with pytest.raises(ContractViolation):
        arm.heads["intent"].probabilities_in_order(CONTEXTS[:2], ["timing_4"])


def test_fit_rejects_a_missing_head_label() -> None:
    arm = ArmB0()
    labels = arm.labels_from_turns(TURNS)
    labels.pop("action")
    with pytest.raises(ContractViolation):
        arm.fit(CONTEXTS, labels)


def test_arm_fits_from_real_context_windows(cfg) -> None:
    """End-to-end through data.build_context, on a synthetic conversation."""
    from reflex.data import build_context

    scenario = {"personal": {"customer_name": "crystal minh"}, "order": {"order_id": "3348917502"}}
    turns = [
        NormalizedTurn(
            convo_id=7,
            turn_index=i,
            speaker=("agent" if i % 2 else "customer"),
            text=line,
            nextstep=("retrieve_utterance" if i % 2 else None),
            intent="timing_4",
        )
        for i, line in enumerate(
            [
                "when will my shipment arrive",
                "let me look up the delivery window",
                "the order id is 3348917502",
                "thanks i can see it now",
                "great how long will it take",
                "about three business days",
            ]
        )
    ]
    contexts = [build_context(turns, i, scenario, cfg) for i in (1, 3, 5)]
    arm = ArmB0()
    arm.fit(contexts, {head: ["retrieve_utterance"] * 3 if head == "nextstep" else ["timing_4"] * 3
                       if head == "intent" else [NO_ACTION] * 3 for head in B0_HEADS})
    assert all(len(p) == 3 for p in arm.predict(contexts).values())


def test_describe_ships_the_caveat_with_the_arm() -> None:
    """D7: THE CAVEAT ships at the same prominence as the result."""
    described = ArmB0().describe()
    assert "not been fairly tested" in described["caveat"]
    assert "inside the 0.68-point CI" in described["structured_features"]
