"""Spec 4.1 tests. The centre of gravity is the LEAKAGE BOUNDARY.

This project once shipped a feature leak that read as 99-100% accuracy, so the
rule ``build_context`` enforces -- a scenario field reaches the model only after
the CUSTOMER has stated it -- is tested twice over: once on hand-built
conversations where the expected answer is obvious, and once against real ABCD
dev conversations, where every disclosed field must be backed by evidence in an
earlier customer turn.

The real-data tests are skipped (not failed) when the ABCD corpus is not present
at ``data.abcd_dir``, so the suite still runs on a machine without the 116 MB
download.
"""

from __future__ import annotations

import os

import pytest

from reflex.config import load_config, resolve_path
from reflex.contracts import ContractViolation
from reflex.data import (
    build_context,
    build_partitions,
    iter_agent_turns,
    learning_curve_subset,
    load_ontology,
    load_raw_abcd,
    load_utterances,
    normalize_conversation,
    select_novel_subflows,
    subflow_list,
    turn_key,
)
from reflex.schemas import NormalizedTurn

CONFIG = "configs/default.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


def _have_corpus(cfg: dict) -> bool:
    return os.path.exists(
        os.path.join(resolve_path(cfg, "data.abcd_dir"), "data", f"abcd_v{cfg['data']['version']}.json")
    )


# --------------------------------------------------------------------------- #
# Hand-built fixtures: the expected answer is visible by eye
# --------------------------------------------------------------------------- #

SCENARIO = {
    "personal": {
        "customer_name": "crystal minh",
        "email": "cminh730@email.com",
        "member_level": "bronze",
        "phone": "(977) 625-2661",
        "username": "cminh730",
    },
    "order": {
        "street_address": "6821 1st ave",
        "city": "san mateo",
        "order_id": "3348917502",
        "zip_code": "75227",
        "products": "[{'brand': 'michael_kors', 'product_type': 'jeans'}]",
    },
    "product": {"names": ["michael_kors jeans"], "amounts": [94]},
    "flow": "product_defect",
    "subflow": "return_size",
}


def _turn(index: int, speaker: str, text: str, **kw) -> NormalizedTurn:
    nextstep = {"agent": "retrieve_utterance", "action": "take_action", "customer": None}[speaker]
    return NormalizedTurn(
        convo_id=1,
        turn_index=index,
        speaker=speaker,
        text=text,
        nextstep=nextstep,
        intent="return_size",
        turn_count=index + 1,
        candidates=[7, 8, 9] if speaker == "agent" else None,
        utt_rank=0 if speaker == "agent" else -1,
        utt_id=7 if speaker == "agent" else None,
        **kw,
    )


@pytest.fixture()
def conversation() -> list[NormalizedTurn]:
    """A conversation that discloses three fields, in three different ways."""
    return [
        _turn(0, "agent", "hi! how can i help you today?"),
        _turn(1, "customer", "i need to return an item, can you help?"),
        _turn(2, "agent", "sure, may i have your name please?"),
        _turn(3, "customer", "crystal minh"),  # discloses customer_name by value
        _turn(4, "action", "account has been pulled up for crystal minh.",
              action="pull-up-account", values=["crystal minh"]),
        _turn(5, "agent", "and your order id?"),
        _turn(6, "customer", "order id: <order_id>"),  # discloses order_id by ABCD marker
        _turn(7, "agent", "thanks, one moment."),
    ]


# --------------------------------------------------------------------------- #
# THE LEAKAGE BOUNDARY
# --------------------------------------------------------------------------- #


def test_undisclosed_scenario_fields_never_reach_the_context(cfg, conversation) -> None:
    """The one test this module exists for.

    Everything in ``SCENARIO`` that the customer has not stated must be absent
    from ``disclosed`` AND absent from the rendered ``text``.
    """
    context = build_context(conversation, 7, SCENARIO, cfg)

    assert set(context.disclosed) == {"personal.customer_name", "order.order_id"}
    undisclosed = {
        "cminh730@email.com",     # personal.email
        "(977) 625-2661",         # personal.phone
        "cminh730",               # personal.username
        "bronze",                 # personal.member_level
        "6821 1st ave",           # order.street_address
        "san mateo",              # order.city
        "75227",                  # order.zip_code
        "michael_kors jeans",     # product.names
    }
    for value in undisclosed:
        assert value not in context.text, f"undisclosed scenario value leaked into the context: {value!r}"


def test_gold_intent_can_never_be_disclosed(cfg, conversation) -> None:
    """``flow``/``subflow`` are the label. No customer turn may put them in the state."""
    turns = list(conversation)
    turns[1] = _turn(1, "customer", "this is a return_size product_defect issue")
    context = build_context(turns, 7, SCENARIO, cfg)
    assert "subflow" not in context.disclosed
    assert "flow" not in context.disclosed
    assert not any(field.endswith(("flow", "subflow")) for field in context.disclosed)
    state_line = context.text.split("\n")[-1]
    assert "return_size" not in state_line and "product_defect" not in state_line


def test_disclosure_is_not_retroactive(cfg, conversation) -> None:
    """A value stated at turn 6 must not appear in the context built for turn 5."""
    early = build_context(conversation, 5, SCENARIO, cfg)
    later = build_context(conversation, 7, SCENARIO, cfg)
    assert "order.order_id" not in early.disclosed
    assert "order.order_id" in later.disclosed
    assert SCENARIO["order"]["order_id"] not in early.text


def test_only_customer_turns_disclose(cfg) -> None:
    """The agent saying a value does not make it known to the model."""
    turns = [
        _turn(0, "agent", "is your email cminh730@email.com and zip 75227?"),
        _turn(1, "customer", "i would rather not say"),
        _turn(2, "agent", "no problem."),
    ]
    context = build_context(turns, 2, SCENARIO, cfg)
    assert context.disclosed == {}


def test_abcd_marker_counts_as_disclosure(cfg) -> None:
    """``<phone>`` is ABCD's mask for a value the customer really typed."""
    turns = [
        _turn(0, "agent", "what is your phone number?"),
        _turn(1, "customer", "<phone>"),
        _turn(2, "agent", "thank you."),
    ]
    context = build_context(turns, 2, SCENARIO, cfg)
    assert context.disclosed == {"personal.phone": SCENARIO["personal"]["phone"]}


def test_short_values_do_not_match(cfg) -> None:
    """``data.disclosure_min_value_chars`` keeps 'ny'/'yes'/'1' out of the state."""
    scenario = {"order": {"state": "ny", "packaging": "yes", "num_products": "1", "city": "san mateo"}}
    turns = [
        _turn(0, "agent", "hello"),
        _turn(1, "customer", "yes, 1 item, i live in ny near san mateo"),
        _turn(2, "agent", "ok"),
    ]
    context = build_context(turns, 2, scenario, cfg)
    assert set(context.disclosed) == {"order.city"}


def test_value_match_respects_token_boundaries(cfg) -> None:
    """``75227`` inside ``175227890`` is not a disclosure of the zip code."""
    turns = [
        _turn(0, "agent", "hello"),
        _turn(1, "customer", "the tracking number is 175227890"),
        _turn(2, "agent", "ok"),
    ]
    context = build_context(turns, 2, SCENARIO, cfg)
    assert "order.zip_code" not in context.disclosed


# --------------------------------------------------------------------------- #
# Context shape, K, determinism
# --------------------------------------------------------------------------- #


def test_context_is_deterministic(cfg, conversation) -> None:
    a = build_context(conversation, 7, SCENARIO, cfg)
    b = build_context(conversation, 7, SCENARIO, cfg)
    assert a.text == b.text and a.context_hash == b.context_hash
    assert len(a.context_hash) == 16


def test_context_excludes_the_predicted_turn(cfg, conversation) -> None:
    context = build_context(conversation, 5, SCENARIO, cfg)
    assert len(context.turns) == 5
    assert context.turns[-1] == "action|account has been pulled up for crystal minh."
    assert conversation[5].text not in context.text


def test_context_turns_K_accepts_full_and_int(cfg, conversation) -> None:
    """Decision A: the default is the FULL thread; an int still windows the turns."""
    full = build_context(conversation, 7, SCENARIO, load_config(CONFIG, ["data.context_turns_K=full"]))
    k2 = build_context(conversation, 7, SCENARIO, load_config(CONFIG, ["data.context_turns_K=2"]))
    assert len(full.turns) == 7
    assert len(k2.turns) == 2
    # K truncates the turn list only -- the state still summarizes the whole prefix.
    assert k2.disclosed == full.disclosed
    assert k2.actions_so_far == full.actions_so_far


def test_context_records_actions_with_values(cfg, conversation) -> None:
    context = build_context(conversation, 7, SCENARIO, cfg)
    assert context.actions_so_far == [{"action": "pull-up-account", "values": ["crystal minh"]}]
    assert context.text.split("\n")[-1].startswith("state|")
    assert "pull-up-account(crystal minh)" in context.text


def test_context_rejects_out_of_range_turn_index(cfg, conversation) -> None:
    with pytest.raises(ContractViolation):
        build_context(conversation, len(conversation), SCENARIO, cfg)


# --------------------------------------------------------------------------- #
# Normalization: the rank/id split and the synthetic end turn
# --------------------------------------------------------------------------- #


RAW_CONVO = {
    "convo_id": 42,
    "scenario": SCENARIO,
    "original": [["agent", "Hi!"], ["customer", "Hello"], ["action", "Account pulled up"]],
    "delexed": [
        {"speaker": "agent", "text": "hi!", "turn_count": 1,
         "targets": ["return_size", "retrieve_utterance", None, [], 2], "candidates": [10, 20, 30]},
        {"speaker": "customer", "text": "hello", "turn_count": 2,
         "targets": ["return_size", None, None, [], -1], "candidates": []},
        {"speaker": "action", "text": "account pulled up", "turn_count": 4,
         "targets": ["return_size", "take_action", "pull-up-account", ["crystal minh"], -1], "candidates": []},
    ],
}
UTTERANCES = [f"u{i}" for i in range(50)]


def test_utt_rank_resolves_through_candidates(cfg) -> None:
    turns = normalize_conversation(RAW_CONVO, UTTERANCES, cfg)
    agent = turns[0]
    assert agent.utt_rank == 2, "utt_rank is raw targets[4]"
    assert agent.utt_id == 30, "utt_id is candidates[utt_rank]"
    assert agent.candidates == [10, 20, 30]
    assert turns[1].candidates is None and turns[1].utt_id is None


def test_end_conversation_turn_is_synthesized(cfg) -> None:
    turns = normalize_conversation(RAW_CONVO, UTTERANCES, cfg)
    assert len(turns) == 4
    end = turns[-1]
    assert end.is_synthetic_end and end.nextstep == "end_conversation"
    assert end.intent == "return_size", "inherits the last turn's targets[0]"
    assert end.turn_count == 4, "reuses the last turn's turn_count"
    assert end.utt_rank == -1 and end.utt_id is None
    # utils/process.py copies action/values but its CDS processor then emits -1 for
    # both unless nextstep == take_action; carrying them would inflate the official
    # action denominator (sum(bslot_label >= 0)) by one turn per conversation.
    assert end.action is None and end.values is None
    assert [t.turn_index for t in turns] == [0, 1, 2, 3]


def test_end_conversation_synthesis_is_switchable() -> None:
    off = load_config(CONFIG, ["data.synthesize_end_conversation=false"])
    assert len(normalize_conversation(RAW_CONVO, UTTERANCES, off)) == 3


def test_speaker_nextstep_disagreement_is_a_violation(cfg) -> None:
    broken = {**RAW_CONVO, "delexed": [dict(RAW_CONVO["delexed"][0], speaker="customer")],
              "original": [["customer", "Hi!"]]}
    with pytest.raises(ContractViolation):
        normalize_conversation(broken, UTTERANCES, cfg)


def test_out_of_range_rank_is_a_violation(cfg) -> None:
    bad = dict(RAW_CONVO["delexed"][0])
    bad["targets"] = ["return_size", "retrieve_utterance", None, [], 7]
    broken = {**RAW_CONVO, "delexed": [bad], "original": [["agent", "Hi!"]]}
    with pytest.raises(ContractViolation):
        normalize_conversation(broken, UTTERANCES, cfg)


def test_iter_agent_turns_skips_customer_turns(cfg) -> None:
    partition = {42: normalize_conversation(RAW_CONVO, UTTERANCES, cfg)}
    rows = list(iter_agent_turns(partition))
    assert [t.nextstep for _, _, t in rows] == ["retrieve_utterance", "take_action", "end_conversation"]


def test_turn_key_format() -> None:
    assert turn_key("train", 1234, 12) == "train:1234:12"


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def test_learning_curve_subsets_are_nested_and_ordered() -> None:
    ids = list(range(1000))
    subsets = {f: learning_curve_subset(ids, f, 13) for f in (0.10, 0.25, 0.50, 1.00)}
    assert [len(subsets[f]) for f in (0.10, 0.25, 0.50, 1.00)] == [100, 250, 500, 1000]
    for smaller, larger in ((0.10, 0.25), (0.25, 0.50), (0.50, 1.00)):
        assert set(subsets[smaller]) <= set(subsets[larger])
    assert subsets[0.25] == sorted(subsets[0.25]), "original relative order is preserved"
    assert learning_curve_subset(ids, 0.10, 13) == subsets[0.10], "deterministic"


# --------------------------------------------------------------------------- #
# Real corpus (skipped without the download)
# --------------------------------------------------------------------------- #


def test_real_splits_and_counts(cfg) -> None:
    if not _have_corpus(cfg):
        pytest.skip("ABCD corpus not present")
    raw = load_raw_abcd(cfg)
    assert [len(raw[s]) for s in ("train", "dev", "test")] == [8034, 1004, 1004]
    utterances = load_utterances(cfg)
    assert len(utterances) == 95288
    counts = {"agent": 0, "customer": 0, "action": 0}
    for convo in raw["train"]:
        for turn in convo["delexed"]:
            counts[turn["speaker"]] += 1
    assert counts == {"agent": 75985, "customer": 71259, "action": 29190}


def test_novel_split_removes_novel_subflows_from_train_and_dev(cfg) -> None:
    if not _have_corpus(cfg):
        pytest.skip("ABCD corpus not present")
    partitions = build_partitions(cfg)
    novel = set(partitions.novel_subflows)
    assert len(novel) == cfg["data"]["novel_subflows_count"]
    assert novel <= set(subflow_list(load_ontology(cfg)))
    for name in ("train", "dev"):
        for turns in getattr(partitions, name).values():
            assert turns[0].intent not in novel
    assert {turns[0].intent for turns in partitions.test_novel.values()} <= novel
    assert not (novel & {turns[0].intent for turns in partitions.test_seen.values()})
    assert len(partitions.test_seen) + len(partitions.test_novel) == 1004
    assert select_novel_subflows(load_ontology(cfg), 5, 13) == partitions.novel_subflows


def test_every_disclosure_on_real_data_has_customer_evidence(cfg) -> None:
    """The inverse direction: nothing enters the state without a customer turn saying it.

    Over real dev conversations, each disclosed field must be backed by an
    earlier CUSTOMER turn containing either its surface value or ABCD's marker
    for it -- and the state line must never carry the gold subflow.
    """
    if not _have_corpus(cfg):
        pytest.skip("ABCD corpus not present")
    raw = load_raw_abcd(cfg)
    partitions = build_partitions(cfg)
    scenarios = {c["convo_id"]: c["scenario"] for cs in raw.values() for c in cs}

    checked = 0
    for convo_id in list(partitions.dev)[:200]:
        turns = partitions.dev[convo_id]
        scenario = scenarios[convo_id]
        for turn in turns:
            if turn.nextstep is None:
                continue
            context = build_context(turns, turn.turn_index, scenario, cfg)
            checked += 1
            earlier = " ".join(
                t.text.lower() for t in turns[: turn.turn_index] if t.speaker == "customer"
            )
            for field, value in context.disclosed.items():
                group, _, leaf = field.partition(".")
                marker_hit = f"<{leaf}>" in earlier or f"<{leaf.rstrip('s')}>" in earlier
                value_hit = any(part.strip().lower() in earlier for part in value.split(", "))
                assert marker_hit or value_hit, (
                    f"convo {convo_id} turn {turn.turn_index}: disclosed {field}={value!r} "
                    "with no earlier customer turn stating it"
                )
            assert scenario["subflow"] not in context.disclosed.values()
    assert checked > 1000, f"only {checked} contexts checked"
