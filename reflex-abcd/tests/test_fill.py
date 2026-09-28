"""Spec 4.7 tests. The filler's one rule is that it NEVER guesses, so most of these
are tests that it refused to.

What is pinned here:
  * a slot with no available value returns ``(None, missing)`` -- no default, no
    empty substitution, no half-filled sentence, and never a ``{placeholder}``
    left in customer-facing text;
  * the strict source priority customer > prior action > disclosed scenario;
  * the LEAKAGE GUARD: an undisclosed scenario field is not a source, and only
    turns before the one being answered are read;
  * ``compose_utterance`` reports the UNION of missing slots, not the first one;
  * the DEFECTS D-4 support guard fires when it is switched on (and the honest
    reason it is off by default).

The synthetic tests always run. The corpus-backed ones skip, rather than fail,
when ABCD is absent.
"""

from __future__ import annotations

import json
import os

import pytest

from reflex.config import apply_overrides, load_config, resolve_path
from reflex.contracts import ContractViolation
from reflex.data import load_ontology
from reflex.fill import check_availability, collect_slot_sources, compose_utterance, fill_template
from reflex.schemas import NormalizedTurn, SlotRegistry, SlotSources, SlotSpec, Template

CONFIG = "configs/default.yaml"

#: One real ABCD scenario (convo 1 of train), copied verbatim.
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
        "full_address": "6821 1st ave  san mateo, ny 75227",
        "city": "san mateo",
        "num_products": "1",
        "order_id": "3348917502",
        "packaging": "yes",
        "payment_method": "credit card",
        "purchase_date": "2019-11-06",
        "state": "ny",
        "zip_code": "75227",
    },
    "product": {"names": ["michael_kors jeans"], "amounts": [94]},
    "flow": "product_defect",
    "subflow": "return_size",
}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def registry() -> SlotRegistry:
    """A hand-built registry: these tests must not need the corpus."""
    specs = [
        SlotSpec(name="order_id", type="id", source="customer_utterance", abcd_marker="<order_id>"),
        SlotSpec(name="email", type="email", source="customer_utterance", abcd_marker="<email>"),
        SlotSpec(name="customer_name", type="name", source="customer_utterance"),
        SlotSpec(name="amount", type="money", source="scenario_field", abcd_marker="<amount>"),
    ]
    return SlotRegistry(
        slots={spec.name: spec for spec in specs},
        marker_to_slot={spec.abcd_marker: spec.name for spec in specs if spec.abcd_marker},
    )


def _template(template_id: str, text: str, *slots: str) -> Template:
    return Template(
        template_id=template_id, act="ASK", text_delex=text, slots=list(slots), count=9
    )


def _have_corpus(cfg: dict) -> bool:
    return os.path.exists(os.path.join(resolve_path(cfg, "data.abcd_dir"), "data", "ontology.json"))


def _conversation() -> list[NormalizedTurn]:
    """Agent opener, customer disclosing two fields, an action, then the turn to answer."""
    return [
        NormalizedTurn(convo_id=1, turn_index=0, speaker="agent", text="hello! how may i help you?",
                       nextstep="retrieve_utterance", intent="return_size"),
        NormalizedTurn(convo_id=1, turn_index=1, speaker="customer",
                       text="hi, this is crystal minh, my order id is 3348917502"),
        NormalizedTurn(convo_id=1, turn_index=2, speaker="action", text="pull-up-account",
                       nextstep="take_action", action="pull-up-account", values=["crystal minh"]),
        NormalizedTurn(convo_id=1, turn_index=3, speaker="agent", text="thanks, one moment.",
                       nextstep="retrieve_utterance", intent="return_size"),
        NormalizedTurn(convo_id=1, turn_index=4, speaker="customer",
                       text="my email is cminh730@email.com"),
        NormalizedTurn(convo_id=1, turn_index=5, speaker="agent", text="got it.",
                       nextstep="retrieve_utterance", intent="return_size"),
    ]


# --------------------------------------------------------------------------- #
# THE ONE RULE: a missing value is a gate failure, never a guess
# --------------------------------------------------------------------------- #


def test_missing_slot_returns_none_and_never_guesses(cfg: dict, registry: SlotRegistry) -> None:
    template = _template("T000001", "can i have your order id {order_id}?", "order_id")
    text, missing = fill_template(template, SlotSources(), registry, cfg)
    assert text is None, "a filler that returns text here has guessed a value"
    assert missing == ["order_id"]


def test_no_placeholder_ever_survives_into_text(cfg: dict, registry: SlotRegistry) -> None:
    """The failure mode this guards: a half-filled sentence reaching a customer."""
    template = _template("T000002", "order {order_id} to {email}", "order_id", "email")
    sources = SlotSources(from_customer={"order_id": "3348917502"})
    text, missing = fill_template(template, sources, registry, cfg)
    assert text is None
    assert missing == ["email"]  # the one that was missing, not the one that was not


def test_filled_text_is_exact_substitution(cfg: dict, registry: SlotRegistry) -> None:
    template = _template("T000003", "your order {order_id} is on its way.", "order_id")
    sources = SlotSources(from_scenario={"order_id": "3348917502"})
    text, missing = fill_template(template, sources, registry, cfg)
    assert text == "your order 3348917502 is on its way."
    assert missing == []
    assert "{" not in text


def test_repeated_placeholder_is_filled_everywhere(cfg: dict, registry: SlotRegistry) -> None:
    template = _template("T000004", "{order_id}? yes, {order_id}.", "order_id")
    text, _ = fill_template(template, SlotSources(from_customer={"order_id": "42"}), registry, cfg)
    assert text == "42? yes, 42."


def test_unknown_placeholder_raises(cfg: dict, registry: SlotRegistry) -> None:
    """Spec 6.3: the registry is the only source of slot names."""
    template = _template("T000005", "your {promo_code} is ready", "promo_code")
    with pytest.raises(ContractViolation, match="promo_code"):
        fill_template(template, SlotSources(), registry, cfg)


# --------------------------------------------------------------------------- #
# Source priority (spec 6.7) -- customer, then prior action, then scenario
# --------------------------------------------------------------------------- #


def test_customer_value_wins_over_action_and_scenario(cfg: dict, registry: SlotRegistry) -> None:
    sources = SlotSources(
        from_customer={"order_id": "111"},
        from_actions={"order_id": "222"},
        from_scenario={"order_id": "333"},
    )
    text, _ = fill_template(_template("T1", "{order_id}", "order_id"), sources, registry, cfg)
    assert text == "111"


def test_action_value_wins_over_scenario(cfg: dict, registry: SlotRegistry) -> None:
    sources = SlotSources(from_actions={"order_id": "222"}, from_scenario={"order_id": "333"})
    text, _ = fill_template(_template("T1", "{order_id}", "order_id"), sources, registry, cfg)
    assert text == "222"


# --------------------------------------------------------------------------- #
# compose_utterance
# --------------------------------------------------------------------------- #


def test_compose_joins_with_a_single_space(cfg: dict, registry: SlotRegistry) -> None:
    templates = [_template("T1", "thanks!"), _template("T2", "your order {order_id} shipped.", "order_id")]
    text, missing = compose_utterance(
        templates, SlotSources(from_customer={"order_id": "7"}), registry, cfg
    )
    assert text == "thanks! your order 7 shipped."
    assert missing == []


def test_compose_reports_the_union_of_missing_slots(cfg: dict, registry: SlotRegistry) -> None:
    """Not just the first offender: the gate reports what the whole turn needs."""
    templates = [
        _template("T1", "hi {customer_name}", "customer_name"),
        _template("T2", "order {order_id} to {email}", "order_id", "email"),
    ]
    text, missing = compose_utterance(
        templates, SlotSources(from_scenario={"order_id": "7"}), registry, cfg
    )
    assert text is None
    assert missing == ["customer_name", "email"]  # first-seen order, deduplicated


def test_compose_of_no_templates_is_empty_not_a_slot_failure(cfg: dict, registry: SlotRegistry) -> None:
    """The contract's iff: text is None IFF something was unfillable. Nothing was."""
    assert compose_utterance([], SlotSources(), registry, cfg) == ("", [])


# --------------------------------------------------------------------------- #
# check_availability
# --------------------------------------------------------------------------- #


def test_check_availability_reports_missing_in_required_order(cfg: dict) -> None:
    sources = SlotSources(from_actions={"order_id": "7"})
    assert check_availability(["customer_name", "order_id", "email"], sources, cfg) == [
        "customer_name", "email",
    ]


def test_check_availability_empty_when_everything_is_sourced(cfg: dict) -> None:
    sources = SlotSources(from_customer={"email": "a@b.com"}, from_scenario={"order_id": "7"})
    assert check_availability(["email", "order_id"], sources, cfg) == []


def test_a_null_value_is_not_a_value(cfg: dict) -> None:
    """ABCD writes "n/a" into value positions; filling it would ship it to a customer."""
    sources = SlotSources(from_actions={"order_id": "n/a"}, from_scenario={"email": "   "})
    assert check_availability(["order_id", "email"], sources, cfg) == ["order_id", "email"]


def test_duplicate_requirements_are_reported_once(cfg: dict) -> None:
    assert check_availability(["email", "email"], SlotSources(), cfg) == ["email"]


# --------------------------------------------------------------------------- #
# DEFECTS D-4: the required-slot support guard
# --------------------------------------------------------------------------- #


def test_support_guard_is_off_by_default(cfg: dict) -> None:
    """Honest default: nothing writes the sidecar yet, so the guard cannot assert."""
    assert check_availability(["order_id"], SlotSources(from_customer={"order_id": "7"}), cfg) == []


def test_support_guard_refuses_a_missing_sidecar(cfg: dict, tmp_path) -> None:
    on = apply_overrides(
        cfg,
        [
            "fill.require_slot_support_metadata=true",
            f"fill.slot_support_path={tmp_path / 'absent.json'}",
        ],
    )
    with pytest.raises(ContractViolation, match="D-4"):
        check_availability(["order_id"], SlotSources(), on)


def test_support_guard_refuses_a_weakly_supported_slot(cfg: dict, tmp_path) -> None:
    """The pre-fix bank required customer_name on notify-team at 5.3% support."""
    sidecar = tmp_path / "support.json"
    sidecar.write_text(json.dumps({"notify-team": {"customer_name": 0.053}}), encoding="utf-8")
    on = apply_overrides(
        cfg,
        ["fill.require_slot_support_metadata=true", f"fill.slot_support_path={sidecar}"],
    )
    with pytest.raises(ContractViolation, match="support"):
        check_availability(["customer_name"], SlotSources(), on)


def test_support_guard_passes_a_well_supported_slot(cfg: dict, tmp_path) -> None:
    sidecar = tmp_path / "support.json"
    sidecar.write_text(json.dumps({"verify-identity": {"account_id": 0.58}}), encoding="utf-8")
    on = apply_overrides(
        cfg,
        ["fill.require_slot_support_metadata=true", f"fill.slot_support_path={sidecar}"],
    )
    assert check_availability(["account_id"], SlotSources(), on) == ["account_id"]


# --------------------------------------------------------------------------- #
# collect_slot_sources -- the leakage guard and the delex mapping
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpus_registry(cfg: dict) -> SlotRegistry:
    if not _have_corpus(cfg):
        pytest.skip("ABCD ontology not present")
    from reflex.compile import build_slot_registry

    return build_slot_registry(load_ontology(cfg), cfg)


def test_nothing_is_available_before_the_customer_speaks(cfg: dict, corpus_registry) -> None:
    """THE LEAKAGE GUARD. The scenario holds every answer; disclosure is what gates it."""
    sources = collect_slot_sources(_conversation(), 0, SCENARIO, corpus_registry, cfg)
    assert sources.from_customer == {}
    assert sources.from_actions == {}
    assert sources.from_scenario == {}


def test_a_disclosed_field_becomes_available(cfg: dict, corpus_registry) -> None:
    sources = collect_slot_sources(_conversation(), 3, SCENARIO, corpus_registry, cfg)
    assert sources.from_scenario.get("order_id") == "3348917502"
    assert sources.from_scenario.get("customer_name") == "crystal minh"


def test_an_undisclosed_field_stays_unavailable(cfg: dict, corpus_registry) -> None:
    """The customer states the email at turn 4; at turn 3 it does not exist yet."""
    at_3 = collect_slot_sources(_conversation(), 3, SCENARIO, corpus_registry, cfg)
    at_5 = collect_slot_sources(_conversation(), 5, SCENARIO, corpus_registry, cfg)
    assert "email" not in at_3.from_scenario
    assert "email" not in at_3.from_customer
    assert check_availability(["email"], at_3, cfg) == ["email"]
    assert "email" in at_5.from_customer or "email" in at_5.from_scenario


def test_prior_action_values_are_typed_not_shape_guessed(cfg: dict, corpus_registry) -> None:
    """DEFECTS D-3: with no ontology evidence, any two-word phrase types as a name."""
    sources = collect_slot_sources(_conversation(), 3, SCENARIO, corpus_registry, cfg)
    assert sources.from_actions.get("customer_name") == "crystal minh"


def test_a_shipping_status_value_does_not_become_a_customer_name(cfg: dict, corpus_registry) -> None:
    """The exact D-3 mistyping: shipping-status "in transit" -> customer_name."""
    turns = [
        NormalizedTurn(convo_id=2, turn_index=0, speaker="customer", text="where is my order?"),
        NormalizedTurn(convo_id=2, turn_index=1, speaker="action", text="shipping-status",
                       nextstep="take_action", action="shipping-status", values=["in transit"]),
        NormalizedTurn(convo_id=2, turn_index=2, speaker="agent", text="it is on its way.",
                       nextstep="retrieve_utterance", intent="status_delivery_time"),
    ]
    sources = collect_slot_sources(turns, 2, {}, corpus_registry, cfg)
    assert sources.from_actions.get("customer_name") != "in transit"


def test_only_turns_before_the_answered_one_are_read(cfg: dict, corpus_registry) -> None:
    """Reading the turn being answered would hand the filler the answer."""
    turns = _conversation()
    early = collect_slot_sources(turns, 1, SCENARIO, corpus_registry, cfg)
    assert early.from_scenario == {}  # the customer's turn 1 is not yet in the past


def test_collect_slot_sources_is_deterministic(cfg: dict, corpus_registry) -> None:
    turns = _conversation()
    first = collect_slot_sources(turns, 5, SCENARIO, corpus_registry, cfg)
    second = collect_slot_sources(turns, 5, SCENARIO, corpus_registry, cfg)
    assert first.to_dict() == second.to_dict()


def test_turn_index_outside_the_conversation_raises(cfg: dict, corpus_registry) -> None:
    with pytest.raises(ContractViolation, match="turn_index"):
        collect_slot_sources(_conversation(), 99, SCENARIO, corpus_registry, cfg)


# --------------------------------------------------------------------------- #
# The REAL bank, and SlotSpec.format
#
# Added alongside the synthetic cases above: a filler that passes hand-built
# templates can still fail on the 48 slot-bearing templates the compiler
# actually produced, and the denominator is itself a finding worth pinning.
# --------------------------------------------------------------------------- #


def _have_bank(cfg: dict) -> bool:
    return os.path.exists(os.path.join(resolve_path(cfg, "paths.bank_dir"), "templates.jsonl"))


def test_every_slot_bearing_bank_template_renders_or_reports(cfg: dict, corpus_registry) -> None:
    """With values: no brace survives. Without: nothing renders and every slot is named.

    The denominator is asserted deliberately. Only ~1% of templates bear a slot,
    so any "slot filling works" claim measured over the whole bank would be the
    DEFECTS D-2 failure mode -- a metric that scores well because the thing it
    measures barely happens.
    """
    if not _have_bank(cfg):
        pytest.skip("compiled bank not present")
    path = os.path.join(resolve_path(cfg, "paths.bank_dir"), "templates.jsonl")
    everything = SlotSources(from_customer={name: f"value-of-{name}" for name in corpus_registry.slots})
    total = 0
    slot_bearing = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            template = Template.from_dict(json.loads(line))
            total += 1
            if not template.slots:
                continue
            slot_bearing += 1
            filled, missing = fill_template(template, everything, corpus_registry, cfg)
            assert missing == []
            assert filled is not None and "{" not in filled and "}" not in filled
            empty_text, empty_missing = fill_template(template, SlotSources(), corpus_registry, cfg)
            assert empty_text is None, "a template filled from nothing has guessed"
            assert empty_missing == list(dict.fromkeys(template.slots))
    assert total > 0 and slot_bearing > 0
    assert slot_bearing / total < 0.02, (
        f"{slot_bearing}/{total} bank templates bear a slot -- if this rises, the "
        f"denominator caveat in the report must be re-measured, not copied forward"
    )


def test_a_regex_tier_value_no_other_source_holds_is_still_found(cfg: dict, corpus_registry) -> None:
    """compile.delex_regex_slots (email, phone) is the only tier that can DISCOVER a value."""
    turns = [
        NormalizedTurn(convo_id=7, turn_index=0, speaker="agent", text="what is your email?",
                       nextstep="retrieve_utterance"),
        NormalizedTurn(convo_id=7, turn_index=1, speaker="customer",
                       text="it is someone.else@example.com"),
        NormalizedTurn(convo_id=7, turn_index=2, speaker="agent", text="thanks.",
                       nextstep="retrieve_utterance"),
    ]
    sources = collect_slot_sources(turns, 2, SCENARIO, corpus_registry, cfg)
    assert sources.from_customer.get("email") == "someone.else@example.com"


def test_format_is_applied_when_a_slot_declares_one(cfg: dict) -> None:
    """Unexercised by the shipped registry (every slot has format ""), so pinned here."""
    registry = SlotRegistry(
        slots={"amount": SlotSpec(name="amount", type="money", source="scenario_field", format="${}")},
        marker_to_slot={"<amount>": "amount"},
    )
    text, missing = fill_template(
        _template("T900001", "your refund of {amount} is approved.", "amount"),
        SlotSources(from_actions={"amount": "94"}),
        registry,
        cfg,
    )
    assert (text, missing) == ("your refund of $94 is approved.", [])


def test_a_format_that_cannot_apply_raises_rather_than_half_rendering(cfg: dict) -> None:
    registry = SlotRegistry(
        slots={"amount": SlotSpec(name="amount", type="money", source="scenario_field",
                                  format="{missing_key}")},
        marker_to_slot={},
    )
    with pytest.raises(ContractViolation, match="format"):
        fill_template(
            _template("T900002", "your refund of {amount}.", "amount"),
            SlotSources(from_actions={"amount": "94"}),
            registry,
            cfg,
        )
