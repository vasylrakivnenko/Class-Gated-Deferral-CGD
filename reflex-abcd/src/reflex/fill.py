"""Spec 4.7 -- Slot filler: turning a chosen template into text with values, and reporting
unavailable slots. No selection. Never guesses a missing value.

THE ONE RULE (spec 6.7)
-----------------------
**A slot with no available value is a GATE FAILURE, never a guess.** Every
function here returns ``None`` plus the missing slot names rather than inventing,
defaulting, dropping the sentence, or leaving a ``{placeholder}`` in customer-
facing text. :mod:`reflex.gate` turns that list into ``unavailable_slot`` and
escalates. This module never routes and never constructs a
:class:`reflex.schemas.GateOutput` (boundary ruling 2 in ``contracts.py``).

SOURCE PRIORITY, STRICT (spec 6.7)
-----------------------------------
1. values the CUSTOMER stated,
2. values entered in PRIOR ACTIONS,
3. scenario fields the customer has ALREADY DISCLOSED.

Priority 3 carries the same leakage guard as
:func:`reflex.data.build_context`: an undisclosed scenario field is not
available, however convenient it would be. This module does not re-implement
that rule -- it calls ``build_context`` and reads its ``disclosed`` map, so the
filler and the model can never disagree about what the customer has said. The
COMPILER deliberately does the opposite (whole conversation, no disclosure
filter) because at compile time an unmasked value is a leak into the bank; see
the "WHY COMPILE-TIME SLOT SOURCES IGNORE THE DISCLOSURE GUARD" note in
:mod:`reflex.compile`.

WHAT THE BANK MAKES OF THIS, MEASURED
--------------------------------------
Only **48 of 4,505 templates (1.07%) bear any slot at all**, 126 occurrences, in
6 distinct slots: customer_name 21, amount 13, name 8, account_id 2,
street_address 2, email 2. (Re-measured on the field-set-merge-guard bank,
``bank_hash 654d796ecc51cea1``; the pre-guard bank read 49 of 4,417 = 1.11%, so
the guard did not change this picture.) Template-driven ``unavailable_slot``
escalations are therefore rare by construction, and the availability signal is
carried almost entirely by action ``required_slots``.
Two consequences that belong in any report of these numbers:

* A "slot fill accuracy" figure computed over all templates is the delex_check
  failure mode in a new costume (DEFECTS D-2: 99.93% looked clean because the
  thing being measured barely happens). Report the denominator: 48 templates,
  not 4,505.
* ``compose_utterance`` is mostly a join. That is the honest description of the
  fast path, not an understatement of it.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Sequence

from reflex import compile as _compile
from reflex import data as _data
from reflex.config import get_dotted, resolve_path
from reflex.contracts import ContractViolation
from reflex.schemas import NormalizedTurn, SlotRegistry, SlotSources, SlotSpec, Template

__all__ = [
    "collect_slot_sources",
    "check_availability",
    "fill_template",
    "compose_utterance",
]


# ---------------------------------------------------------------------------
# Conventions borrowed from the compiler ON PURPOSE.
#
# These three are the compiler's private vocabulary, imported rather than
# copied. A filler with its own idea of what a placeholder looks like, of what
# counts as a non-value, or of how a literal types, is a filler that renders
# text the bank never contained -- and the mismatch would be silent. If a rename
# upstream breaks this import, that is the correct outcome: a loud failure
# beats two quietly diverging conventions.
# ---------------------------------------------------------------------------

#: ``{slot}`` in a template. Templates use braces; ABCD's own markers are
#: ``<slot>`` and never survive compilation (``Template.text_delex``).
_PLACEHOLDER_RE: re.Pattern[str] = _compile._PLACEHOLDER_RE

#: Strings ABCD writes into a value position that are not values ("n/a", ...).
_NULL_VALUES: frozenset[str] = _compile._NULL_VALUES

#: ``(value, declared_categories, enumerable) -> registry slot | None``.
_type_literal = _compile._type_literal

#: ``value -> (scenario surface, transcript surface)``. ABCD writes product
#: names as ``michael_kors jeans`` in the scenario and ``michael kors jeans`` in
#: the transcript; the compiler already treats the two as the same surface.
_surface_variants = _compile._surface_variants

#: Turn speaker whose ``values`` are prior-action values (spec 5.1: the action
#: turn's text IS the button name and ``nextstep`` is ``take_action``).
_ACTION_SPEAKER: str = "action"
_CUSTOMER_SPEAKER: str = "customer"

#: Spec 6.7's join. Frozen by the contract ("a single space between filled
#: sentences"), so it is not a config knob: spec 13 forbids any rewriting on the
#: fast path, and a configurable joiner is the first step toward one.
_JOIN: str = " "


# ---------------------------------------------------------------------------
# Ontology-aware typing of prior action values (DEFECTS D-3)
# ---------------------------------------------------------------------------

#: ``abcd_dir -> (button -> declared value categories, category -> lowered values)``.
#: A memo, not a guess: ``ontology.json`` is small, immutable during a run, and
#: re-read per turn would otherwise cost one file parse per predicted turn.
_TYPING_CACHE: dict[str, tuple[dict[str, list[str]], dict[str, list[str]]]] = {}
_TYPING_CACHE_SIZE: int = 4


def _typing_evidence(cfg: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Evidence :func:`_type_literal` needs: declared categories and enumerable values.

    DEFECTS D-3 is the reason this is not ``_type_literal(value, [], {})``: with
    no categories the shape patterns fire unconstrained, and ``customer_name``'s
    ``^[a-z]+(?:[ '-][a-z]+)+$`` then swallows any two-word phrase -- measured,
    that mistyped 4,699 of 21,335 stored values (22.0%), sending
    shipping-status "in transit" and update-order "change order" into
    ``customer_name``. A mistyped action value here would put a wrong value in a
    customer-facing sentence, which is worse than escalating.
    """
    key = str(get_dotted(cfg, "data.abcd_dir"))
    cached = _TYPING_CACHE.get(key)
    if cached is not None:
        return cached
    ontology = _data.load_ontology(cfg)
    declared: dict[str, list[str]] = {}
    for _category, buttons in (ontology.get("actions") or {}).items():
        if isinstance(buttons, dict):
            for button, categories in buttons.items():
                declared[str(button)] = [str(c) for c in (categories or [])]
    enumerable: dict[str, list[str]] = {
        str(category): [str(v).lower() for v in (values or [])]
        for category, values in (ontology.get("values", {}).get("enumerable") or {}).items()
    }
    if len(_TYPING_CACHE) >= _TYPING_CACHE_SIZE:
        _TYPING_CACHE.clear()
    _TYPING_CACHE[key] = (declared, enumerable)
    return declared, enumerable


# ---------------------------------------------------------------------------
# Scenario field -> registry slot
# ---------------------------------------------------------------------------


def _scenario_slot(field: str, registry: SlotRegistry, cfg: dict[str, Any]) -> Optional[str]:
    """Registry slot for a dotted scenario field, or ``None`` if it names no entity.

    Derived from the registry rather than from a second hand-written table:
    ABCD's scenario leaf names ARE the registry names for all but three fields.
    The exceptions are config (``fill.scenario_field_slot_aliases``) and the two
    pluralized product lists, which ``build_slot_registry`` registers under
    ABCD's own singular markers ``<name>`` / ``<amount>`` (compile M2: ``<name>``
    is the PRODUCT name, not a person).
    """
    leaf = field.rsplit(".", 1)[-1]
    aliases = {
        str(k): str(v)
        for k, v in (get_dotted(cfg, "fill.scenario_field_slot_aliases") or {}).items()
    }
    for candidate in (aliases.get(field), aliases.get(leaf), leaf):
        if candidate and candidate in registry.slots:
            return candidate
    if leaf.endswith("s") and not leaf.endswith("ss"):
        singular = leaf[:-1]
        if singular in registry.slots:
            return singular
    return None


# ---------------------------------------------------------------------------
# Source lookup, in the spec's strict priority order
# ---------------------------------------------------------------------------


def _is_value(value: Any) -> bool:
    return bool(str(value).strip()) and str(value).strip().lower() not in _NULL_VALUES


def _transcript_surface(value: str) -> str:
    """How a SCENARIO value is spelled in customer-facing text.

    ``_surface_variants`` returns the scenario spelling first and the transcript
    spelling last, which for ``michael_kors shirt`` is ``michael kors shirt``.
    Splicing the scenario spelling into an agent sentence puts an underscore in
    front of the customer. Applied to scenario-sourced values ONLY, never to a
    value the customer typed: measured over all 10,042 ABCD conversations,
    ``product.names`` is the only disclosable scenario leaf whose values ever
    contain ``_`` (6,973 occurrences), so no email, username or id is touched.
    """
    variants = _surface_variants(value)
    return variants[-1] if variants else str(value).strip()


def disclosed_slot_values(
    disclosed: dict[str, str],
    registry: SlotRegistry,
    cfg: dict[str, Any],
    leaf_values: Optional[dict[str, list[str]]] = None,
) -> dict[str, str]:
    """``registry slot -> the ONE value the customer has disclosed for it``.

    The single reading of :attr:`reflex.schemas.ContextWindow.disclosed` as slot
    values. :func:`collect_slot_sources` (the filler) and
    ``reflex.select._disclosed_lexicon`` (H4's copy tier) both call it, so the
    two cannot hold different ideas of what the customer said.

    THE ONE RULE. A leaf that disclosed SEVERAL of its values at once
    (``product.names``, ``product.amounts`` -- :func:`reflex.data._disclosure_timeline`
    joins them with ``", "``) names no single one, and "which of the three
    products" is exactly the guess this module refuses: the slot is left out,
    so it stays unavailable and the gate escalates.

    How a compound is recognised depends on what the caller may see:

    * ``leaf_values`` given (the filler, which holds the scenario): cardinality
      is read off the leaf, never off a comma count -- a street address ("0069
      kennedy st newark, tx 78018") legitimately contains ``", "``.
    * ``leaf_values`` omitted (the selector, which by the leakage boundary never
      holds the scenario): a LIST leaf is recognised structurally, as one that
      reached its slot through :func:`_scenario_slot`'s singular-strip branch
      (``names -> name``, ``amounts -> amount``), and only for such a leaf does
      ``", "`` mean a join. Measured over all 10,042 conversations of
      ``abcd_v1.1.json``: ``product.names`` and ``product.amounts`` are the only
      list-valued scenario leaves and none of their values contains ``", "``,
      so on ABCD the two readings agree exactly.

    Values are returned in their TRANSCRIPT spelling (:func:`_transcript_surface`).
    """
    out: dict[str, str] = {}
    for field, value in (disclosed or {}).items():
        slot = _scenario_slot(str(field), registry, cfg)
        if not slot or not _is_value(value):
            continue
        text = str(value).strip()
        if leaf_values is not None:
            options = leaf_values.get(str(field))
            compound = options is not None and len(options) > 1 and text not in options
        else:
            leaf = str(field).rsplit(".", 1)[-1]
            compound = leaf != slot and leaf[:-1] == slot and ", " in text
        if compound:
            continue
        out.setdefault(slot, _transcript_surface(text))
    return out


def _lookup(sources: SlotSources, slot: str) -> Optional[str]:
    """The value for ``slot``, or ``None``. Priority: customer, then action, then scenario.

    The field order of :class:`SlotSources` IS the priority order (spec 6.7), so
    this reads them in declaration order and never re-ranks.
    """
    for bucket in (sources.from_customer, sources.from_actions, sources.from_scenario):
        value = (bucket or {}).get(slot)
        if value is not None and _is_value(value):
            return str(value).strip()
    return None


# ---------------------------------------------------------------------------
# 4.7 collect_slot_sources
# ---------------------------------------------------------------------------


def collect_slot_sources(
    turns: Sequence[NormalizedTurn],
    turn_index: int,
    scenario: dict[str, Any],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> SlotSources:
    """Gather available slot values for the turn at ``turn_index`` (spec 6.7).

    See :func:`reflex.contracts.collect_slot_sources` for the frozen contract.

    Only turns STRICTLY BEFORE ``turn_index`` are read -- reading the turn being
    predicted would hand the filler the answer. Within each origin the FIRST
    statement wins (``setdefault``), matching
    :func:`reflex.data.build_context`'s "first turn that disclosed it" and the
    compiler's own convention. A customer who corrects a value later in the
    conversation therefore keeps the earlier one; that is a known, deliberate
    limitation rather than an oversight, and it is invisible for any value that
    also appears in the scenario (the common case), because both readings
    resolve to the same scenario string.

    Determinism (spec 10): a pure function of ``turns[:turn_index]``, the
    scenario and the config. No RNG, no clock, no model.
    """
    if not turns:
        raise ContractViolation("cannot collect slot sources for an empty conversation")
    if turn_index < 0 or turn_index >= len(turns):
        raise ContractViolation(
            f"turn_index {turn_index} is outside the conversation ({len(turns)} turns)"
        )
    prior = list(turns[:turn_index])

    # --- 3. scenario fields the customer has ALREADY DISCLOSED -------------- #
    # build_context owns the disclosure boundary; re-deriving it here is how the
    # filler and the model end up disagreeing about what the customer said.
    disclosed = _data.build_context(turns, turn_index, scenario or {}, cfg).disclosed
    leaf_values = dict(_data._flatten_scenario(scenario or {}, cfg))
    from_scenario = disclosed_slot_values(disclosed, registry, cfg, leaf_values)

    # --- 2. values entered in PRIOR ACTIONS ---------------------------------- #
    declared, enumerable = _typing_evidence(cfg)
    from_actions: dict[str, str] = {}
    for turn in prior:
        if turn.speaker != _ACTION_SPEAKER or not turn.values:
            continue
        categories = declared.get(str(turn.action or ""), [])
        for value in turn.values:
            if not _is_value(value):
                continue
            slot = _type_literal(str(value), categories, enumerable)
            if slot and slot in registry.slots:
                from_actions.setdefault(slot, str(value).strip())

    # --- 1. values the CUSTOMER stated --------------------------------------- #
    # "From the delex mapping of customer turns" (spec 6.7): the compiler's
    # delexicalizer IS that mapping, so it is called rather than re-implemented.
    # It reports slot -> surface for ABCD's own <slot> markers, for exact matches
    # against the values already gathered above, and for the shape-only classes
    # in compile.delex_regex_slots (email, phone) -- which is the only tier that
    # can discover a value no other source has. A marker whose value no source
    # holds yields NO mapping entry: the slot stays unavailable and the gate
    # escalates, which is the point.
    partial = SlotSources(from_actions=dict(from_actions), from_scenario=dict(from_scenario))
    from_customer: dict[str, str] = {}
    for turn in prior:
        if turn.speaker != _CUSTOMER_SPEAKER or not turn.text:
            continue
        # The whole turn, not sentence by sentence: every tier of the
        # delexicalizer is sentence-independent for the MAPPING it returns, and
        # running pysbd over every customer turn of every predicted turn would
        # buy nothing but latency.
        _text_delex, mapping = _compile.delexicalize_sentence(turn.text, partial, registry, cfg)
        for slot, surface in mapping.items():
            if slot in registry.slots and _is_value(surface):
                from_customer.setdefault(slot, str(surface).strip())

    return SlotSources(
        from_customer=from_customer,
        from_actions=from_actions,
        from_scenario=from_scenario,
    )


# ---------------------------------------------------------------------------
# 4.7 check_availability  (+ the DEFECTS D-4 support guard)
# ---------------------------------------------------------------------------


def _check_required_slot_support(required_slots: Sequence[str], cfg: dict[str, Any]) -> None:
    """Refuse required slots whose per-position support nobody measured (DEFECTS D-4).

    D-4 is the reason this guard exists. ``ActionPattern.required_slots`` once
    counted only TYPED occurrences in its modal vote, so a position whose modal
    value had no registry slot still emitted a minority slot -- notify-team
    required ``customer_name`` on 24 of 454 values (5.3%). Every turn taking that
    button then escalated as ``unavailable_slot`` for a value it never carried,
    and nothing in the metrics said why: a bad required-slot list is a silent,
    total escalation of a whole button. After the fix, 7 of 30 buttons keep
    required slots and the minimum measured support is 58%.

    **This guard is OFF by default, and that is a DELIBERATE DEFERRAL, recorded
    with its reasons** (maintainer decision, 2026-09-16).

    1. It cannot be the assertion the D-4 lesson asks for.
       :class:`reflex.schemas.ActionPattern` is frozen at three fields
       (``action``, ``required_slots``, ``count``) and carries no support
       metadata, ``bank/actions.jsonl`` therefore has none, and
       ``check_availability`` receives bare slot NAMES with no button attached.
       "Assert that the patterns you load carry support metadata" cannot be
       done at the point of use, because the information is not there.
    2. It is not load-bearing. **D-4 was fixed AT SOURCE** -- the modal vote now
       includes ``None`` and emits nothing when ``None`` wins -- and that fix
       carries a regression test proven to fail on the pre-fix module. This
       would be a second line of defence behind a first line that is already
       tested, at the price of a compile-side artifact, a frozen-schema
       amendment and more config surface.

    What is built and tested is the switch, both ways, so the intent is
    documented for whoever picks it up: point ``fill.slot_support_path`` at a
    sidecar ``{slot: support}`` or ``{button: {slot: support}}`` and set
    ``fill.require_slot_support_metadata: true``, and any required slot missing
    from it, or below ``fill.min_required_slot_support``, raises instead of
    escalating a whole button's turns invisibly.
    """
    if not bool(get_dotted(cfg, "fill.require_slot_support_metadata")):
        return
    path = resolve_path(cfg, "fill.slot_support_path")
    if not os.path.exists(path):
        raise ContractViolation(
            f"fill.require_slot_support_metadata is on but no support sidecar exists at "
            f"{path}. required_slots drives the gate's unavailable_slot signal, and a "
            f"slot with no measured support silently escalates every turn of its button "
            f"(DEFECTS D-4). Write the sidecar in the compiler or turn the guard off "
            f"deliberately."
        )
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    support: dict[str, float] = {}
    for key, value in (raw or {}).items():
        if isinstance(value, dict):  # {button: {slot: support}}
            for slot, fraction in value.items():
                support[str(slot)] = max(support.get(str(slot), 0.0), float(fraction))
        else:  # {slot: support}
            support[str(key)] = float(value)
    minimum = float(get_dotted(cfg, "fill.min_required_slot_support"))
    for slot in required_slots:
        name = str(slot)
        if name not in support:
            raise ContractViolation(
                f"required slot {name!r} has no support metadata in {path} (DEFECTS D-4): "
                f"refusing to escalate turns on a requirement nobody measured."
            )
        if support[name] < minimum:
            raise ContractViolation(
                f"required slot {name!r} has measured support {support[name]:.3f}, below "
                f"fill.min_required_slot_support ({minimum}). D-4: a weakly-supported "
                f"required slot escalates a whole button for a value its turns never "
                f"carried."
            )


def check_availability(
    required_slots: Sequence[str],
    sources: SlotSources,
    cfg: dict[str, Any],
) -> list[str]:
    """Return the required slots that have NO value in any source (spec 6.6 signal 3).

    See :func:`reflex.contracts.check_availability` for the frozen contract.
    This is a report, not a route: :mod:`reflex.gate` decides what a non-empty
    return means.
    """
    _check_required_slot_support(required_slots, cfg)
    missing: list[str] = []
    for slot in required_slots:
        name = str(slot).strip()
        if not name:
            raise ContractViolation(
                "required_slots contains an empty slot name; a nameless requirement "
                "can never be satisfied and would escalate every turn silently"
            )
        if name in missing:
            continue
        if _lookup(sources, name) is None:
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# 4.7 fill_template / compose_utterance
# ---------------------------------------------------------------------------


def _render(value: str, spec: SlotSpec) -> str:
    """Apply ``SlotSpec.format``; empty format means verbatim substitution.

    UNEXERCISED IN THIS BUILD, stated rather than hidden: all 16 registry slots
    carry ``format: ""``, so every substitution today is verbatim. The branch is
    implemented so a future dated or currency slot has somewhere to live, and it
    raises rather than falling back to the raw value if the format string does
    not apply -- a half-applied format in customer-facing text is worse than a
    crash in a test run.
    """
    template = (spec.format or "").strip()
    if not template:
        return value
    try:
        return template.format(value, value=value, slot=spec.name)
    except (IndexError, KeyError, ValueError) as exc:
        raise ContractViolation(
            f"slot {spec.name!r} has format {spec.format!r}, which does not apply to "
            f"value {value!r}: {exc}"
        ) from exc


def fill_template(
    template: Template,
    sources: SlotSources,
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> tuple[Optional[str], list[str]]:
    """Substitute values into one template's ``{slot}`` placeholders (spec 6.7).

    See :func:`reflex.contracts.fill_template` for the frozen contract.

    **A slot with no available value is never guessed.** There is no default, no
    empty-string substitution and no dropping of the offending sentence: the
    return is ``(None, missing)`` and the gate escalates the turn.

    The TEXT is authoritative, not ``Template.slots``: the placeholders actually
    present in ``text_delex`` are what must be filled for the rendered sentence
    to be free of braces. (They agree on all 4,417 compiled templates -- checked,
    not assumed.)
    """
    del cfg  # every knob this function could read is frozen by the contract
    text = template.text_delex or ""
    placeholders = list(dict.fromkeys(_PLACEHOLDER_RE.findall(text)))

    missing: list[str] = []
    values: dict[str, str] = {}
    for slot in placeholders:
        if slot not in registry.slots:
            raise ContractViolation(
                f"template {template.template_id} contains placeholder {{{slot}}}, which "
                f"is not in the slot registry. Spec 6.3 makes the registry the only "
                f"source of slot names and spec 14 acceptance item 5 checks it."
            )
        value = _lookup(sources, slot)
        if value is None:
            missing.append(slot)
        else:
            values[slot] = _render(value, registry.slots[slot])
    if missing:
        return None, missing
    filled = _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], text)
    return filled, []


def compose_utterance(
    templates: Sequence[Template],
    sources: SlotSources,
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> tuple[Optional[str], list[str]]:
    """Fill and join one template per act position into the final utterance.

    See :func:`reflex.contracts.compose_utterance` for the frozen contract.

    Join only -- spec 13 forbids any rewriting, paraphrasing or generation on the
    fast path, so there is no casing fix, no punctuation repair and no connective
    inserted between sentences. EVERY template is filled even after the first
    failure, so ``missing_slots`` is the complete union the gate reports rather
    than only the first offender.

    Edge case, stated because the contract's "iff" decides it: an EMPTY template
    list returns ``("", [])``, not ``(None, [])`` -- nothing was unfillable. A
    caller that reaches this with no templates has a selection bug, and an empty
    string makes that visible downstream instead of masking it as a slot failure.
    """
    filled: list[str] = []
    missing: list[str] = []
    for template in templates:
        text, template_missing = fill_template(template, sources, registry, cfg)
        for slot in template_missing:
            if slot not in missing:
                missing.append(slot)
        if text is not None:
            filled.append(text)
    if missing:
        return None, missing
    return _JOIN.join(filled), []
