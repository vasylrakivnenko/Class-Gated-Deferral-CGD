"""Mechanically enforce the frozen contract.

Nine agents implement twelve modules in parallel without talking to each other.
These tests are the only thing that notices when two of them drift apart, so they
must keep passing at every stage -- while a module is still a stub, and after it
is implemented.

What is checked:
  * every module defines exactly the public names it OWNS (MECE, spec Section 4);
  * every implemented signature matches the frozen one byte for byte;
  * no name is owned by two modules;
  * the spec-5 schemas round-trip through ``to_dict`` / ``from_dict``;
  * the normative constants (nextstep order, gate precedence) are intact.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from reflex import contracts
from reflex.contracts import MODULE_FUNCTIONS
from reflex.schemas import (
    ACT_INVENTORY,
    GATE_REASON_PRECEDENCE,
    NEXT_STEPS,
    SPEC5_SCHEMAS,
    ActionPattern,
    Decision,
    EvalRecord,
    GateOutput,
    NormalizedTurn,
    Skeleton,
    Template,
)

ALL_OWNED = [(mod, fn) for mod, fns in MODULE_FUNCTIONS.items() for fn in fns]


# --------------------------------------------------------------------------- #
# Ownership is MECE
# --------------------------------------------------------------------------- #


def test_ownership_is_mutually_exclusive() -> None:
    """No public function may be owned by two modules (spec Section 4 is MECE)."""
    seen: dict[str, str] = {}
    for mod, fn in ALL_OWNED:
        assert fn not in seen, f"{fn!r} is owned by both {seen[fn]!r} and {mod!r}"
        seen[fn] = mod


def test_ownership_is_collectively_exhaustive() -> None:
    """Every callable the contract exports must be assigned to exactly one module."""
    owned = {fn for _, fn in ALL_OWNED}
    exported = {
        name
        for name in contracts.__all__
        if inspect.isfunction(getattr(contracts, name))
    }
    unassigned = exported - owned
    assert not unassigned, f"contract functions with no owning module: {sorted(unassigned)}"
    phantom = owned - exported
    assert not phantom, f"modules claim functions the contract does not declare: {sorted(phantom)}"


def test_module_function_table_covers_spec_section_4() -> None:
    """All twelve spec Section 4 modules plus the cross-cutting config module."""
    expected = {
        "config", "data", "compile", "models", "train", "calibrate",
        "select", "gate", "fill", "llm_agent", "evaluate", "report", "run",
    }
    assert set(MODULE_FUNCTIONS) == expected


# --------------------------------------------------------------------------- #
# Each module exposes exactly what it owns, with the frozen signature
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module_name", sorted(MODULE_FUNCTIONS))
def test_module_all_matches_ownership(module_name: str) -> None:
    """``__all__`` must equal the owned set -- no extras, no omissions."""
    mod = importlib.import_module(f"reflex.{module_name}")
    declared = set(getattr(mod, "__all__", ()))
    owned = set(MODULE_FUNCTIONS[module_name])
    # config.py legitimately exports two extra helpers of its own.
    allowed_extras = {"repo_root", "get_dotted"} if module_name == "config" else set()
    assert owned <= declared, (
        f"reflex.{module_name}.__all__ is missing owned names: {sorted(owned - declared)}"
    )
    assert declared - owned <= allowed_extras, (
        f"reflex.{module_name}.__all__ exports names it does not own: "
        f"{sorted(declared - owned - allowed_extras)}"
    )


@pytest.mark.parametrize("module_name,func_name", ALL_OWNED, ids=[f"{m}.{f}" for m, f in ALL_OWNED])
def test_signature_matches_contract(module_name: str, func_name: str) -> None:
    """An implemented function must match the frozen signature exactly.

    Same parameter names, same order, same defaults, same annotations, same
    return annotation. This is the check that stops module B from calling
    module A with the argument order module A silently changed.
    """
    mod = importlib.import_module(f"reflex.{module_name}")
    assert hasattr(mod, func_name), f"reflex.{module_name} does not define {func_name!r}"
    impl = getattr(mod, func_name)
    frozen = getattr(contracts, func_name)
    if impl is frozen:
        pytest.skip(f"{module_name}.{func_name} is still the contract stub")
    assert inspect.signature(impl) == inspect.signature(frozen), (
        f"reflex.{module_name}.{func_name} signature drifted from the contract.\n"
        f"  contract: {inspect.signature(frozen)}\n"
        f"  impl    : {inspect.signature(impl)}"
    )


@pytest.mark.parametrize("module_name,func_name", ALL_OWNED, ids=[f"{m}.{f}" for m, f in ALL_OWNED])
def test_contract_functions_are_documented(module_name: str, func_name: str) -> None:
    """Every contract function carries a docstring; implementations keep one too."""
    del module_name
    doc = getattr(contracts, func_name).__doc__
    assert doc and len(doc.strip()) > 40, f"{func_name} has no substantive contract docstring"


# --------------------------------------------------------------------------- #
# Normative constants
# --------------------------------------------------------------------------- #


def test_next_steps_order_is_normative() -> None:
    """``cds_report`` branches on nextstep_label == 0/1/2 with exactly this meaning."""
    assert NEXT_STEPS == ("retrieve_utterance", "take_action", "end_conversation")
    assert NEXT_STEPS.index("retrieve_utterance") == 0
    assert NEXT_STEPS.index("take_action") == 1
    assert NEXT_STEPS.index("end_conversation") == 2


def test_gate_reason_precedence_matches_spec_6_6() -> None:
    """Spec 6.6: novel > unseen_action > unavailable_slot > low_confidence."""
    assert GATE_REASON_PRECEDENCE[:4] == (
        "novel", "unseen_action", "unavailable_slot", "low_confidence",
    )
    assert GATE_REASON_PRECEDENCE[-1] == "ok"


def test_act_inventory_is_nine_acts() -> None:
    """Spec 2 / 12 fix the inventory at nine acts, with OTHER as the fallback."""
    assert len(ACT_INVENTORY) == 9
    assert ACT_INVENTORY[-1] == "OTHER"


# --------------------------------------------------------------------------- #
# Schema round-trips
# --------------------------------------------------------------------------- #


def _sample(schema: type) -> object:
    gate = GateOutput(
        route="escalated",
        reason="low_confidence",
        set_sizes={"nextstep": 1, "intent": 3, "action": 0, "skeleton": 1, "templates": [2]},
        novelty_distance=0.11,
        missing_slots=[],
    )
    decision = Decision(
        convo_id=42, turn_index=7, arm="B", route="escalated",
        nextstep="retrieve_utterance", intent="return_size", gate=gate,
        candidate_utt_id=8416, candidate_rank=84,
    )
    return {
        NormalizedTurn: NormalizedTurn(
            convo_id=42, turn_index=7, speaker="agent", text="hi!",
            nextstep="retrieve_utterance", intent="return_size",
            candidates=[1, 2, 3], utt_id=2, utt_rank=1, turn_count=8,
        ),
        Template: Template(
            template_id="T000123", act="INFORM",
            text_delex="Your order {order_id} will arrive on {arrival_date}.",
            slots=["order_id", "arrival_date"], count=9,
            surface_forms=["Your order {order_id} will arrive on {arrival_date}."],
            example_context_ids=["train:42:7"],
        ),
        Skeleton: Skeleton(skeleton_id="S0042", acts=["ACK", "INFORM", "OFFER"], count=5),
        ActionPattern: ActionPattern(action="verify-identity", required_slots=["account_id"], count=3),
        GateOutput: gate,
        Decision: decision,
        EvalRecord: EvalRecord(
            decision=decision,
            gold={"nextstep": "retrieve_utterance", "intent": "return_size",
                  "action": None, "values": None, "utt_id": 2},
            correct={"nextstep": True, "intent": True, "action": False,
                     "values": False, "utterance": True},
            seed=1, run_id="run-x",
        ),
    }[schema]


@pytest.mark.parametrize("schema", SPEC5_SCHEMAS, ids=[s.__name__ for s in SPEC5_SCHEMAS])
def test_spec5_schema_round_trips(schema: type) -> None:
    """``from_dict(to_dict(x)) == x`` for every normative schema."""
    obj = _sample(schema)
    payload = obj.to_dict()  # type: ignore[attr-defined]
    assert schema.from_dict(payload) == obj  # type: ignore[attr-defined]


@pytest.mark.parametrize("schema", SPEC5_SCHEMAS, ids=[s.__name__ for s in SPEC5_SCHEMAS])
def test_spec5_schemas_are_frozen(schema: type) -> None:
    """Spec 10 determinism: fields cannot be rebound after construction."""
    obj = _sample(schema)
    field_name = next(iter(obj.to_dict()))  # type: ignore[attr-defined]
    with pytest.raises(Exception):
        setattr(obj, field_name, None)


def test_from_dict_rejects_unknown_keys() -> None:
    """A silently-dropped field is how three agents end up with three schemas."""
    payload = Skeleton(skeleton_id="S0001", acts=["ACK"], count=1).to_dict()
    payload["acts_v2"] = ["ACK"]
    with pytest.raises(KeyError):
        Skeleton.from_dict(payload)


def test_eval_record_is_flat_on_disk() -> None:
    """Spec 5.7's on-disk shape is "Decision fields + gold/correct/seed/run_id"."""
    record = _sample(EvalRecord)
    payload = record.to_dict()  # type: ignore[attr-defined]
    assert "decision" not in payload, "EvalRecord must flatten the Decision on disk"
    for key in ("convo_id", "turn_index", "arm", "route", "nextstep", "intent"):
        assert key in payload
    for key in ("gold", "correct", "seed", "run_id"):
        assert key in payload
    assert EvalRecord.from_dict(payload) == record


def test_normalized_turn_separates_utt_id_from_utt_rank() -> None:
    """The rank/id distinction is the one data fact the spec gets wrong."""
    turn = NormalizedTurn(
        convo_id=1, turn_index=0, speaker="agent", text="hi!",
        nextstep="retrieve_utterance", candidates=[10, 20, 30],
        utt_rank=1, utt_id=20,
    )
    assert turn.utt_rank == 1, "utt_rank is the raw targets[4]"
    assert turn.utt_id == 20, "utt_id is candidates[utt_rank]"
    assert turn.candidates is not None and turn.candidates[turn.utt_rank] == turn.utt_id
