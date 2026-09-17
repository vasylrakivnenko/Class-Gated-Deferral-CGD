"""Spec 4.2 regression tests. Both cover defects that were SHIPPED and measured.

Defect A -- ``compile_bank`` typed action values with no evidence
    ``_type_literal(value, [], {})`` was called with an empty category list and
    an empty enumerable index, so the shape patterns fired unconstrained and
    4,699 of 21,335 stored ``TurnLabel.slot_values`` entries on train (22.0%)
    were wrong: ``shipping-status ['in transit']`` became
    ``customer_name='in transit'``. That mapping is what spec 6.2 step 2b hands
    to the filler.

Defect B -- ``extract_action_patterns`` voted only over TYPABLE values
    A position whose values are agent choices (``reason_slotval``,
    ``order_slotval``, ``company_team``) contributed whatever rare stray could be
    typed, so ``notify-team`` required ``customer_name`` on 24 of 454 values
    (5.3%) and ``update-order`` required ``amount`` on 105 of 1,288 (8.2%).
    ``required_slots`` feeds :func:`reflex.fill.check_availability`, the gate's
    ``unavailable_slot`` signal, so 7,211 train action turns (26.6%) would have
    escalated for a value they never carried.

The synthetic tests always run. The corpus-backed ones are skipped, not failed,
when the ABCD data or a compiled bank is absent.
"""

from __future__ import annotations

import json
import os

import pytest

from reflex.compile import (
    _type_literal,
    build_slot_registry,
    extract_action_patterns,
    load_bank,
)
from reflex.config import load_config, resolve_path
from reflex.data import load_ontology, load_raw_abcd
from reflex.schemas import NormalizedTurn

CONFIG = "configs/default.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def ontology(cfg: dict) -> dict:
    if not _have_corpus(cfg):
        pytest.skip("ABCD corpus not present")
    return load_ontology(cfg)


@pytest.fixture(scope="module")
def registry(ontology: dict, cfg: dict):
    return build_slot_registry(ontology, cfg)


def _have_corpus(cfg: dict) -> bool:
    return os.path.exists(
        os.path.join(resolve_path(cfg, "data.abcd_dir"), "data", f"abcd_v{cfg['data']['version']}.json")
    )


def _have_bank(cfg: dict) -> bool:
    return os.path.exists(os.path.join(resolve_path(cfg, "paths.bank_dir"), "templates.jsonl"))


def _categories(ontology: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _section, buttons in ontology["actions"].items():
        for button, cats in buttons.items():
            out[button] = list(cats)
    return out


def _enumerable(ontology: dict) -> dict[str, list[str]]:
    return {
        category: [str(v).lower() for v in values]
        for category, values in ontology["values"]["enumerable"].items()
    }


def _action_turn(action: str, values: list[str], turn_index: int) -> NormalizedTurn:
    return NormalizedTurn(
        convo_id=1,
        turn_index=turn_index,
        speaker="action",
        text=action,
        nextstep="take_action",
        intent="return_size",
        action=action,
        values=values,
    )


# --------------------------------------------------------------------------- #
# Defect A: a value is typed against the button that took it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "button,value,expected",
    [
        # Each of these was mistyped by the no-evidence call, with the wrong
        # answer in the comment.
        ("shipping-status", "in transit", "shipping_status"),   # was customer_name
        ("update-order", "change order", None),                 # was customer_name
        ("record-reason", "5", None),                           # was amount
        ("notify-team", "website team", None),                  # was customer_name
        ("enter-details", "2481 kennedy st monterey, wa 97444", None),  # was street_address
        # ... and these must keep working.
        ("pull-up-account", "crystal minh", "customer_name"),
        ("membership", "gold", "membership_level"),
        ("offer-refund", "64", "amount"),
    ],
)
def test_value_typing_uses_the_buttons_declared_categories(
    ontology: dict, button: str, value: str, expected
) -> None:
    """An action value means what the ontology says that BUTTON's values mean.

    Typing by shape alone is what produced ``customer_name='in transit'``: two
    lowercase words look exactly like a person's name.
    """
    typed = _type_literal(value, _categories(ontology)[button], _enumerable(ontology))
    assert typed == expected


def test_slot_values_on_action_turns_agree_with_ontology_typing(cfg: dict, ontology: dict) -> None:
    """Every stored ``slot_values`` entry must match a typing done WITH evidence.

    This is the end-to-end form of the defect: the unit test above can pass
    while ``compile_bank`` still calls the typer with no categories.
    """
    labels_path = os.path.join(resolve_path(cfg, "paths.labels_dir"), "train.jsonl")
    if not os.path.exists(labels_path):
        pytest.skip("labels/train.jsonl not compiled")
    categories, enumerable = _categories(ontology), _enumerable(ontology)
    checked = 0
    with open(labels_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["nextstep"] != "take_action":
                continue
            expected: dict[str, str] = {}
            for value in row.get("values") or []:
                slot = _type_literal(str(value), categories.get(row["action"], []), enumerable)
                if slot:
                    expected.setdefault(slot, str(value))
            for slot, surface in (row.get("slot_values") or {}).items():
                assert expected.get(slot) == surface, (
                    f"{row['turn_id']} ({row['action']}, values={row.get('values')}) stored "
                    f"{slot}={surface!r}, which typing with the button's categories does not support"
                )
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# Defect B: a position requires a slot only when its values actually carry one
# --------------------------------------------------------------------------- #


def test_position_dominated_by_agent_choices_requires_nothing(registry, cfg: dict) -> None:
    """``notify-team``'s value is a team the AGENT picks; no source can supply it."""
    turns = [_action_turn("notify-team", ["purchasing department"], i) for i in range(90)]
    # The strays that used to win the vote: a name typed into the box.
    turns += [_action_turn("notify-team", ["chloe zhang"], 90 + i) for i in range(10)]
    pattern = {p.action: p for p in extract_action_patterns(turns, registry, cfg)}["notify-team"]
    assert pattern.count == 100
    assert pattern.required_slots == [], (
        "a position whose values are agent choices must require no slot: requiring one makes "
        "the gate escalate every turn of that button on unavailable_slot"
    )


def test_majority_supported_slot_is_still_required(registry, cfg: dict) -> None:
    """The fix must not throw away the slots that ARE real."""
    turns = [_action_turn("pull-up-account", ["chloe zhang"], i) for i in range(95)]
    turns += [_action_turn("pull-up-account", ["n/a"], 95 + i) for i in range(5)]
    pattern = {p.action: p for p in extract_action_patterns(turns, registry, cfg)}["pull-up-account"]
    assert pattern.required_slots == ["customer_name"]


def test_every_required_slot_is_the_modal_typing_of_its_position(cfg: dict, ontology: dict) -> None:
    """On real train data, no required slot may rest on a minority of its position.

    Stated as "the slot must be its position's modal typing, counting the values
    that map to nothing" rather than as a hard-coded support percentage, so the
    test states the rule instead of a number (spec 10).
    """
    if not _have_corpus(cfg) or not _have_bank(cfg):
        pytest.skip("corpus or compiled bank not present")
    bank = load_bank(cfg)
    categories, enumerable = _categories(ontology), _enumerable(ontology)
    raw = load_raw_abcd(cfg)
    per_position: dict[str, dict[int, list]] = {}
    for convo in raw["train"]:
        for turn in convo["delexed"]:
            if turn["speaker"] != "action":
                continue
            button = turn["targets"][2]
            for position, value in enumerate(turn["targets"][3] or []):
                slot = _type_literal(str(value), categories.get(button, []), enumerable)
                per_position.setdefault(button, {}).setdefault(position, []).append(slot)

    for pattern in bank.actions:
        if not pattern.required_slots:
            continue
        modals = set()
        for position, typings in sorted(per_position.get(pattern.action, {}).items()):
            counts: dict = {}
            for slot in typings:
                counts[slot] = counts.get(slot, 0) + 1
            modal = max(sorted(counts, key=lambda s: (s is None, str(s))), key=lambda s: counts[s])
            if modal is not None:
                modals.add(modal)
        for slot in pattern.required_slots:
            assert slot in modals, (
                f"{pattern.action} requires {slot!r}, which is not the modal typing of any of its "
                f"value positions; that slot would fire the gate's unavailable_slot signal on "
                f"turns that never carried it"
            )
