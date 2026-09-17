"""Spec 4.2 + 6.3 -- Compiler: sentence splitting, delexicalization, act labeling, template
deduplication, skeleton extraction, action-pattern extraction, slot registry.
No training of the selector.

WHAT THIS MODULE DOES (spec 6.2, TRAIN SPLIT ONLY)
--------------------------------------------------
1. Split every train agent utterance into sentences (``pysbd``).
2. Delexicalize each sentence against the slot registry (6.3).
3. Label each sentence with one of the nine acts, using the FREE LOCAL labeler
   (``compile.act_labeler: rules_plus_embed``): high-precision cue rules first,
   frozen-embedding nearest-centroid for whatever no rule fires on.
4. Deduplicate templates (exact, then single-linkage cosine within one act).
5. Extract skeletons (the ordered act tuple of an utterance); keep ALL of them.
6. Extract one action pattern per button.
7. Persist ``bank/{templates,skeletons,actions}.jsonl``,
   ``bank/slot_registry.json`` and ``labels/train.jsonl``.

FIVE MEASURED FACTS THAT SHAPED THE IMPLEMENTATION
--------------------------------------------------
M1  **Agent utterances are nearly slot-free.** Train carries 17,073 ABCD
    ``<slot>`` markers, but only 343 of them sit in an AGENT turn (2.0%); the
    rest are in customer and action turns. The bank is therefore mostly literal
    text, and template coverage -- not slot recall -- is the binding constraint
    on the reflex rate.
M2  **``<name>`` is the PRODUCT name, not a person.** Recovered by aligning
    ``delexed`` against ``original``: the 16 distinct surfaces behind ``<name>``
    are ``guess shirt``, ``michael_kors boots``, ... So ``customer_name`` is NOT
    a parallel name for an entity ABCD already marks; it is a genuine gap, and
    it is the single most common action value (``pull-up-account``, 5,653 turns).
M3  **``model.encoder`` cannot be used for dedup similarity.** Raw
    ModernBERT-base is not sentence-trained: over a 15-sentence ABCD probe,
    59% of ALL pairs score cosine >= 0.92 (CLS) and it puts
    "what is your membership level?" and "the shipping fee is $4.99 per item."
    at 0.961. Single-linkage at ``compile.dedup_threshold`` would collapse the
    bank into one cluster. ``compile.dedup_embed_model`` (a sentence-trained
    MiniLM) puts 0 of those pairs over 0.92 and the true paraphrase pairs at
    0.80-0.82. See "spec deviations" in the report.
M4  **Most money in an agent turn is a POLICY PRICE, not a fillable value.**
    1,275 ``$`` amounts appear in train agent turns; they are dominated by
    "gift wrapping costs $4.99 per item" and "tailoring is $15 per pair". A bare
    money regex would mask them into ``{amount}``, and the filler -- which never
    guesses -- would then have to escalate every one of those turns. So money
    and dates are delexicalized ONLY when they match a value this conversation
    actually has; ``compile.delex_regex_slots`` lists the shape-only classes
    (email, phone) that are safe without corroboration.
M5  **Category-valued slots must not be masked in text.** "are you a gold or
    silver member?" would become "are you a {membership_level} or
    {membership_level} member?", which no filler can reconstruct.
    ``compile.delex_match_types`` keeps such slots in the registry (action
    patterns need them) while barring them from text masking: they are typed
    ``other``, and only the entity types in that list may be masked.

WHY COMPILE-TIME SLOT SOURCES IGNORE THE DISCLOSURE GUARD
---------------------------------------------------------
:func:`reflex.data.build_context` and :func:`reflex.fill.collect_slot_sources`
hide undisclosed scenario fields, because showing them to a model leaks the
answer. The compiler faces the OPPOSITE risk: a value it fails to mask is
written verbatim into the bank, where it becomes a literal another
conversation's customer would receive. Masking more is the safe direction here,
so the private source builder below uses the whole conversation and the whole
scenario. Nothing it produces reaches a model: only ``{slot}`` names survive
into a template.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional, Sequence

from reflex import data as _data
from reflex.config import get_dotted, resolve_path
from reflex.contracts import ActLabeler, ContractViolation, ReflexError
from reflex.schemas import (
    ACT_INVENTORY,
    ActionPattern,
    Bank,
    NormalizedTurn,
    Skeleton,
    SlotRegistry,
    SlotSources,
    SlotSpec,
    Template,
    TurnLabel,
)

__all__ = [
    "build_slot_registry",
    "split_sentences",
    "delexicalize_sentence",
    "get_act_labeler",
    "label_acts",
    "extract_templates",
    "dedup_templates",
    "extract_skeletons",
    "extract_action_patterns",
    "compile_bank",
    "write_bank",
    "load_bank",
    "write_turn_labels",
    "load_turn_labels",
    "write_delex_check",
    "write_act_check",
]


# --------------------------------------------------------------------------- #
# Module constants.
#
# Spec 10 bans thresholds, model ids, paths and prices from code -- every one of
# those lives in ``configs/default.yaml``. What follows is VOCABULARY: the file
# names spec 6.2 step 7 fixes, the slot typing of ABCD's own marker set, and the
# cue patterns of the free local act labeler (``compile.act_rules_path`` can
# replace them wholesale).
# --------------------------------------------------------------------------- #

#: Spec 6.2 step 7 file names, relative to ``paths.bank_dir`` / ``paths.labels_dir``.
_F_TEMPLATES = "templates.jsonl"
_F_SKELETONS = "skeletons.jsonl"
_F_ACTIONS = "actions.jsonl"
_F_REGISTRY = "slot_registry.json"
#: Sidecar (not one of the four): ``Bank.source_fraction`` and ``bank_hash`` have
#: nowhere to live in the four spec files, and recomputing them on load would
#: silently invent a fraction. See "spec deviations".
_F_META = "meta.json"

#: The only supported ``compile.slot_registry_backbone``.
_BACKBONE = "ontology_non_enumerable"

#: ABCD's own marker vocabulary, typed. Keys are exactly
#: ``ontology["values"]["non_enumerable"]`` flattened (11 names).
_ABCD_SLOT_TYPES: dict[str, str] = {
    "amount": "money",
    "name": "name",  # M2: the PRODUCT name ("guess shirt"), not a person
    "account_id": "id",
    "email": "email",
    "phone": "phone",
    "pin_number": "id",
    "username": "id",
    "full_address": "address",
    "order_id": "id",
    "street_address": "address",
    "zip_code": "address",
}

#: Where :mod:`reflex.fill` should look FIRST for each ABCD slot (spec 6.7's
#: priority order is customer utterance > prior action value > scenario field).
_ABCD_SLOT_SOURCES: dict[str, str] = {
    "amount": "scenario_field",
    "name": "scenario_field",
    "account_id": "customer_utterance",
    "email": "customer_utterance",
    "phone": "customer_utterance",
    "pin_number": "customer_utterance",
    "username": "customer_utterance",
    "full_address": "customer_utterance",
    "order_id": "customer_utterance",
    "street_address": "customer_utterance",
    "zip_code": "customer_utterance",
}

#: REFLEX-added slots, for entities ABCD leaves literal. ``(name, type, source)``.
#: Every name is SPECIFIC (spec 6.3 / acceptance item 5) and every one is
#: sourceable by the filler -- a slot no source can ever fill would only teach
#: the gate to escalate. ``customer_name`` is a gap, not a parallel name (M2).
_EXTENSION_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("customer_name", "name", "customer_utterance"),
    ("membership_level", "other", "customer_utterance"),
    ("payment_method", "other", "scenario_field"),
    ("shipping_status", "other", "prior_action_value"),
    ("purchase_date", "date", "scenario_field"),
)

#: Names too generic to be a REFLEX-added slot (spec 6.3). ``name`` is absent
#: because ABCD itself defines ``<name>``; the guard below still forbids a
#: REFLEX-added slot from re-using any ABCD marker name.
_GENERIC_SLOT_NAMES: frozenset[str] = frozenset(
    {"date", "value", "id", "number", "text", "item", "info", "data", "field",
     "thing", "code", "address", "time", "price", "type", "status"}
)

#: ``scenario`` field -> registry slot. Fields absent here are not entities the
#: filler can place in text (``num_products``, ``packaging``, ``flow``, ...).
_SCENARIO_FIELD_TO_SLOT: dict[tuple[str, str], str] = {
    ("personal", "customer_name"): "customer_name",
    ("personal", "email"): "email",
    ("personal", "member_level"): "membership_level",
    ("personal", "phone"): "phone",
    ("personal", "username"): "username",
    ("personal", "account_id"): "account_id",
    ("personal", "pin_number"): "pin_number",
    ("order", "street_address"): "street_address",
    ("order", "full_address"): "full_address",
    ("order", "order_id"): "order_id",
    ("order", "zip_code"): "zip_code",
    ("order", "payment_method"): "payment_method",
    ("order", "purchase_date"): "purchase_date",
    ("order", "shipping_status"): "shipping_status",
}

#: ``ontology["actions"][category][button]`` names value CATEGORIES. These are
#: the ones that name a sourceable entity; the rest (``reason_slotval``,
#: ``change_option``, ``single_item_query``, ...) are choices the H4 values head
#: predicts, not values the filler can look up, so they contribute no required
#: slot. ``product`` maps onto ABCD's own ``<name>`` (M2) rather than a parallel
#: ``product_name``.
_VALUE_CATEGORY_TO_SLOT: dict[str, str] = {
    "customer_name": "customer_name",
    "membership_level": "membership_level",
    "payment_method": "payment_method",
    "shipping_option": "shipping_status",
    "product": "name",
}

#: Shape tests used to type a literal (an action value, or a regex-tier span).
#: Order is significant: the first match wins.
_SHAPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"^[\w.+-]+@[\w-]+\.[a-z]{2,}$")),
    ("phone", re.compile(r"^\(?\d{3}\)?[ -]?\d{3}-\d{4}$")),
    ("purchase_date", re.compile(r"^(?:20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})$")),
    ("amount", re.compile(r"^\$?\d{1,4}(?:\.\d{2})?$")),
    ("order_id", re.compile(r"^\d{10}$")),
    ("zip_code", re.compile(r"^\d{5}$")),
    ("pin_number", re.compile(r"^\d{4,6}$")),
    ("account_id", re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]{10}$")),
    ("street_address", re.compile(r"^\d+\s+[a-z].*$")),
    ("customer_name", re.compile(r"^[a-z]+(?:[ '-][a-z]+)+$")),
)

#: In-text regexes for ``compile.delex_regex_slots``. Shape alone is enough
#: evidence for these two; M4 explains why money and dates are not here.
_INTEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"),
    "phone": re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b"),
    "order_id": re.compile(r"\b\d{10}\b"),
    "zip_code": re.compile(r"\b\d{5}\b"),
}

#: ABCD marker, e.g. ``<order_id>``.
_MARKER_RE = re.compile(r"<([a-z_]+)>")
#: REFLEXIVE placeholder, e.g. ``{order_id}``.
_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
#: Protected span used while delexicalizing and while splitting sentences. Pure
#: alphanumerics, so ``pysbd`` never finds a sentence boundary inside one and no
#: later pass matches one.
_SENTINEL = "zqx{}xqz"
_SENTINEL_RE = re.compile(r"zqx(\d+)xqz")

#: Values that are not values (ABCD writes these into action slots).
_NULL_VALUES: frozenset[str] = frozenset({"", "n/a", "na", "none", "null", "unknown", "-"})

#: Ballot cast by an action value that maps to NO sourceable slot. It is a
#: candidate in the per-position vote of :func:`extract_action_patterns`, not a
#: slot name: a position most of whose values are agent choices must require
#: nothing. Never written to a Bank field.
_NO_SLOT = "\x00no_slot"

#: Cue rules for ``rules_plus_embed``, in priority order: the first pattern that
#: matches a sentence decides its act. Everything that matches nothing goes to
#: the embedding tier. Definitions are the ones in ``prompts/act_labeling.txt``,
#: so the two labelers mean the same thing by a label.
#:
#: Two conventions worth stating once, because they are arbitrary but must be
#: consistent (skeletons are act TUPLES, so an inconsistent convention is worse
#: than a debatable one):
#:   * "is there anything else I can help you with?" is OFFER (a remedy the
#:     customer may decline), while "how can I help you?" is ASK (it requests
#:     information the agent does not have).
#:   * A bare greeting is ACK, but a greeting welded to a question
#:     ("hello, how can I help you today?") is scored on the question.
_BUILTIN_ACT_RULES: tuple[tuple[str, str], ...] = (
    ("CLOSE", r"\b(good\s?bye|bye now|bye[.!]?$|farewell)\b"),
    ("CLOSE", r"\bhave a (great|good|nice|wonderful|lovely|fantastic|happy|blessed|safe|pleasant)\b[^?]*\b(day|night|evening|afternoon|morning|weekend|one|rest)\b"),
    ("CLOSE", r"\b(have a good one|take care|enjoy (your|the) (day|evening|weekend|night)|stay (safe|well)|be safe|until next time)\b"),
    ("VERIFY", r"\bverif(y|ying|ication)\b"),
    ("VERIFY", r"\b(can|could|may|would|will) you (please )?(confirm|verify|repeat|spell)\b"),
    ("VERIFY", r"\b(just to|let me) (confirm|verify|make sure|double.?check|be sure)\b"),
    ("VERIFY", r"\bis (that|this|it) (correct|right|accurate)\b|\b(correct|right)\?\s*$"),
    ("VERIFY", r"\b(did|do) you (say|mean)\b|\bare you sure\b|\bfor security purposes\b"),
    ("OFFER", r"\b(anything|something|any ?thing) (else|more|further)\b|\banything else\b"),
    ("OFFER", r"\bis there (anything|something|anything more)\b|\bdo you need (anything|any) (else|more|other)\b"),
    ("OFFER", r"\bwould you (like|want|prefer|be interested)\b|\bdo you want me to\b|\bshall i\b"),
    ("OFFER", r"\bi (can|could|will be able to) (help|assist|offer|check|look into|take care of|do that)\b"),
    ("OFFER", r"\bi'?(d| would| am| 'm) (be )?(happy|glad|able|more than happy) to\b|\blet me know if\b"),
    ("CONFIRM", r"\b(has|have|had) (now )?been (updated|processed|cancell?ed|completed|added|entered|refunded|sent|issued|applied|removed|changed|placed|submitted|pulled up|verified|approved|set up|taken care of)\b"),
    ("CONFIRM", r"\bi (have|'ve|just|already) (updated|processed|cancell?ed|completed|added|entered|refunded|sent|issued|applied|removed|changed|submitted|verified|placed|gone ahead)\b"),
    ("CONFIRM", r"\b(you'?re|you are|we are|it is|that'?s|everything is) all set\b|\byou'?re good to go\b"),
    ("CONFIRM", r"\b(is|are) now (complete|completed|done|updated|processed|cancell?ed)\b|\ball done\b"),
    ("INSTRUCT", r"^(please )?(go ahead|click|press|select|choose|enter|type|send|ship|print|visit|log ?(in|out)|try|fill|return|bring|drop|use|wait|hold|allow|follow|head|navigate)\b"),
    ("INSTRUCT", r"\byou (will need|'ll need|need|have) to (go|click|send|ship|print|visit|log|try|fill|return|wait|contact|bring|drop|use)\b"),
    ("INSTRUCT", r"\bplease (go|click|send|ship|print|visit|log|try|fill|return|wait|hold|allow|follow|check back|be patient)\b"),
    ("ACK", r"^(thanks|thank you|thankyou|thx|ty)\b"),
    ("ACK", r"^(ok|okay|k|alright|alrighty|sure|great|perfect|awesome|excellent|wonderful|fantastic|good|nice|got it|i see|understood|no problem|np|of course|certainly|absolutely|indeed|yes|yep|yeah|yup|no|nope)\b[.!,]*$"),
    ("ACK", r"^(hi|hello|hey|greetings|good (morning|afternoon|evening))\b[.!,]*$"),
    ("ACK", r"\byou'?(re| are) welcome\b|\bmy (pleasure|apologies)\b|\bno problem\b"),
    ("ACK", r"\b(one|a|just a|give me a) (moment|second|minute|sec)\b|\bplease hold\b|\bbear with me\b"),
    ("ACK", r"\b(i'?m |i am )?(so |very |really |terribly )?sorry\b|\bi apologi[sz]e\b|\bunfortunately\b.*\bsorry\b"),
    ("ACK", r"\blet me (check|look|see|take a look|pull up|have a look)\b|\bi understand\b|\bi see\b"),
    ("ACK", r"\bglad (to help|i could help|that worked|to hear)\b|\bwelcome to\b|\bthank you for (contacting|shopping|chatting|choosing)\b"),
    ("ASK", r"\b(may|can|could|would) i (please )?(have|get|ask|know)\b|\bi (need|'ll need|will need|would need) (your|the)\b"),
    ("ASK", r"^(please )?(provide|give me|tell me|share)\b"),
    ("ASK", r"\?\s*$"),
)

#: Seed sentences for the embedding tier. Frozen, hand-written from real ABCD
#: agent text; the centroid of each act's seeds is what an unmatched sentence is
#: compared against. Seeds are deliberately short and prototypical.
_ACT_SEEDS: dict[str, tuple[str, ...]] = {
    "ACK": (
        "thank you.", "okay.", "great.", "no problem.", "you're welcome.",
        "i see.", "got it.", "one moment please.", "i am sorry to hear that.",
        "i apologize for the inconvenience.", "hello.", "sure thing.",
        "let me check that for you.", "i understand your frustration.",
    ),
    "VERIFY": (
        "i will need to verify your identity first.",
        "can you confirm your username for me?",
        "just to confirm, your order id is {order_id}?",
        "is that the correct email address?",
        "let me make sure i have that right.",
        "could you repeat your account id please?",
        "for security purposes i need to check a few details.",
        "did you say {zip_code} was your zip code?",
    ),
    "ASK": (
        "what is your membership level?",
        "may i have your full name please?",
        "when did you make the purchase?",
        "do you have the order id handy?",
        "what is the reason for the return?",
        "which item would you like to return?",
        "how can i help you today?",
        "could i have your zip code, phone number, and email address?",
        "are you still there?",
        "what is the name of the product?",
    ),
    "INFORM": (
        "the shipping fee is $4.99 per item.",
        "your order is currently in transit.",
        "we accept returns within 90 days of purchase.",
        "gold members get free shipping on every order.",
        "the jacket is made of 100% cotton.",
        "that product is currently out of stock.",
        "our stores are open from 8am to 11pm.",
        "there is a fee for expedited shipping.",
        "the annual sale started in january.",
        "i am unable to change the price of an order.",
    ),
    "INSTRUCT": (
        "please ship the item back to us.",
        "click the link i just sent you.",
        "go ahead and log out and log back in.",
        "print the return label and attach it to the package.",
        "you will need to visit the website to complete this.",
        "try again in a few minutes.",
        "please hold while i transfer you.",
        "enter your new address on the account page.",
    ),
    "CONFIRM": (
        "your order has been cancelled.",
        "i have updated your account.",
        "the refund has been processed.",
        "your address has been changed to {street_address}.",
        "i went ahead and applied the discount.",
        "everything is all set on my end.",
        "the subscription has been extended.",
        "your details have been entered.",
    ),
    "OFFER": (
        "is there anything else i can help you with?",
        "would you like me to cancel the order?",
        "i can offer you a 20% discount code.",
        "i would be happy to escalate this to my manager.",
        "do you want me to check the status for you?",
        "i can help you with that.",
        "shall i place the order for you?",
        "let me know if you would like to proceed.",
    ),
    "CLOSE": (
        "have a great day!", "goodbye!", "thank you and have a nice day.",
        "take care.", "have a good one!", "enjoy the rest of your day.",
        "thanks for chatting with us today, bye.", "have a wonderful evening!",
    ),
    "OTHER": (
        "asdf", "...", "hmm", "brb", "test test",
    ),
}


# --------------------------------------------------------------------------- #
# Small private helpers
# --------------------------------------------------------------------------- #


def _sha16(payload: bytes) -> str:
    """16-hex-char sha256 prefix -- the same rule ``data`` and ``llm_agent`` use."""
    return hashlib.sha256(payload).hexdigest()[:16]


def _act_inventory(cfg: dict[str, Any]) -> tuple[str, ...]:
    """Return the act inventory, asserting config and schema agree.

    ``ACT_INVENTORY`` is the class order of head H6 everywhere; a config that
    disagreed would silently relabel every template.
    """
    configured = tuple(get_dotted(cfg, "compile.act_inventory"))
    if configured != ACT_INVENTORY:
        raise ContractViolation(
            f"compile.act_inventory {configured} disagrees with schemas.ACT_INVENTORY "
            f"{ACT_INVENTORY}; the tuple index is head H6's class id, so the two must match"
        )
    return ACT_INVENTORY


def _mask_spans(text: str, patterns: Iterable[re.Pattern[str]]) -> tuple[str, list[str]]:
    """Replace every match of ``patterns`` with an inert sentinel token."""
    kept: list[str] = []

    def _swap(match: re.Match[str]) -> str:
        kept.append(match.group(0))
        return _SENTINEL.format(len(kept) - 1)

    for pattern in patterns:
        text = pattern.sub(_swap, text)
    return text, kept


def _unmask_spans(text: str, kept: Sequence[str]) -> str:
    """Undo :func:`_mask_spans`."""
    if not kept:
        return text
    return _SENTINEL_RE.sub(lambda m: kept[int(m.group(1))], text)


#: One ``pysbd`` segmenter per process. Building one costs ~100 ms and it holds
#: no per-call state.
_SEGMENTER: Any = None


def _segmenter() -> Any:
    global _SEGMENTER
    if _SEGMENTER is None:
        import pysbd

        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return _SEGMENTER


def _slots_in(text: str) -> list[str]:
    """Slot names in ``text``, in order of first appearance, deduplicated."""
    out: list[str] = []
    for name in _PLACEHOLDER_RE.findall(text):
        if name not in out:
            out.append(name)
    return out


def _normalize_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation spacing (spec 6.2 step 4).

    A trailing ``.``/``!`` run is dropped so "thank you." and "thank you" merge;
    ``?`` is kept, because a question and a statement are different acts of
    speech even when they share every other character.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    return re.sub(r"[.!\s]+$", "", text) or text.strip()


def _type_literal(value: str, candidates: Sequence[str], enumerable: dict[str, list[str]]) -> Optional[str]:
    """Return the registry slot a raw literal belongs to, or ``None``.

    ``candidates`` are the value categories ``ontology["actions"]`` declares for
    the button, which is the same evidence ``utils/process.py::value_to_id``
    uses: an enumerable category is settled by membership, a non-enumerable one
    by shape.
    """
    low = value.strip().lower()
    if low in _NULL_VALUES:
        return None
    for category in candidates:
        options = enumerable.get(category)
        if options is not None and low in options:
            return _VALUE_CATEGORY_TO_SLOT.get(category)
    allowed = {c for c in candidates if c not in enumerable}
    for slot, pattern in _SHAPE_PATTERNS:
        if pattern.match(low) and (not allowed or slot in allowed or slot in _VALUE_CATEGORY_TO_SLOT.values()):
            if not allowed or slot in allowed:
                return slot
    return None


# --------------------------------------------------------------------------- #
# 6.3 slot registry
# --------------------------------------------------------------------------- #


def build_slot_registry(ontology: dict[str, Any], cfg: dict[str, Any]) -> SlotRegistry:
    """Build the slot registry (spec 6.3).

    See :func:`reflex.contracts.build_slot_registry` for the frozen contract.
    ABCD's own 11-name ``non_enumerable`` vocabulary is the backbone and is
    registered with its ``<marker>``; the five REFLEX-added names cover entities
    ABCD leaves literal (M2) and are refused if they are generic or would shadow
    a marker.
    """
    backbone = get_dotted(cfg, "compile.slot_registry_backbone")
    if backbone != _BACKBONE:
        raise ValueError(
            f"unknown compile.slot_registry_backbone {backbone!r}; expected {_BACKBONE!r}"
        )
    non_enumerable = ontology.get("values", {}).get("non_enumerable")
    if not isinstance(non_enumerable, dict) or not non_enumerable:
        raise ContractViolation(
            "ontology['values']['non_enumerable'] is missing or empty; it is the slot "
            "registry backbone (spec 6.3) and the marker set utils/load.py tokenizes"
        )

    slots: dict[str, SlotSpec] = {}
    marker_to_slot: dict[str, str] = {}
    for _category, names in non_enumerable.items():
        for name in names:
            marker = f"<{name}>"
            slots[name] = SlotSpec(
                name=name,
                type=_ABCD_SLOT_TYPES.get(name, "other"),  # type: ignore[arg-type]
                source=_ABCD_SLOT_SOURCES.get(name, "customer_utterance"),  # type: ignore[arg-type]
                format="",
                abcd_marker=marker,
            )
            marker_to_slot[marker] = name

    for name, slot_type, source in _EXTENSION_SLOTS:
        if name in slots:
            raise ContractViolation(
                f"REFLEX-added slot {name!r} shadows ABCD's own marker <{name}>; spec 6.3 "
                f"forbids a parallel name for an entity ABCD already marks"
            )
        if name in _GENERIC_SLOT_NAMES:
            raise ContractViolation(
                f"REFLEX-added slot {name!r} is generic; spec 6.3 and acceptance item 5 "
                f"require specific names (arrival_date, not date)"
            )
        slots[name] = SlotSpec(
            name=name,
            type=slot_type,  # type: ignore[arg-type]
            source=source,  # type: ignore[arg-type]
            format="",
            abcd_marker="",
        )
    return SlotRegistry(slots=slots, marker_to_slot=marker_to_slot)


# --------------------------------------------------------------------------- #
# 6.2 step 1 -- sentence splitting
# --------------------------------------------------------------------------- #


def split_sentences(text: str, cfg: dict[str, Any]) -> list[str]:
    """Split one agent utterance into sentences (spec 6.2 step 1).

    See :func:`reflex.contracts.split_sentences` for the frozen contract.
    Markers and placeholders are masked with an alphanumeric sentinel before the
    splitter runs, so no boundary can ever fall inside one.
    """
    splitter = get_dotted(cfg, "compile.sentence_splitter")
    if splitter != "pysbd":
        raise ValueError(
            f"unknown compile.sentence_splitter {splitter!r}; spec 6.2 step 1 is "
            f"rule-based and this build supports 'pysbd' only"
        )
    if not text or not text.strip():
        return []
    masked, kept = _mask_spans(text, (_MARKER_RE, _PLACEHOLDER_RE))
    out: list[str] = []
    for piece in _segmenter().segment(masked):
        restored = _unmask_spans(piece, kept).strip()
        if restored:
            out.append(restored)
    return out


# --------------------------------------------------------------------------- #
# 6.2 step 2 -- delexicalization
# --------------------------------------------------------------------------- #


def _source_items(sources: SlotSources) -> list[tuple[str, str]]:
    """``(slot, value)`` pairs in spec 6.7 priority order, longest value first.

    Longest-first matters: ``full_address`` contains ``street_address``, and
    masking the shorter one first would leave half an address in the template.
    """
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for bucket in (sources.from_customer, sources.from_actions, sources.from_scenario):
        for slot, value in bucket.items():
            key = (slot, str(value).lower())
            if key in seen:
                continue
            seen.add(key)
            ordered.append((slot, str(value)))
    ordered.sort(key=lambda item: (-len(item[1]), item[0], item[1]))
    return ordered


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is not glued to surrounding alphanumerics."""
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or after.isalnum())


def _derived_username_stems(sources: "SlotSources") -> "list[str]":
    """Username stems this conversation's agent is likely to invent.

    MEASURED (train split, 296 agent-stated usernames, 25 distinct): **zero**
    equal ``scenario.personal.username``, and **288 = 97.3%** equal
    first-initial + surname of ``customer_name`` followed by digits --
    "sanya afzal" -> ``safzal1``, "crystal minh" -> ``cminh1``.

    In the make-password flow the agent GENERATES a handle the scenario does not
    contain, so the exact-match tier has nothing to match and 22 templates /
    248 occurrences of literal usernames reached the bank. A fast path built on
    that bank tells one customer another person's username, which is why this
    tier exists and why the bank assertion that follows it FAILS the compile
    rather than warning.
    """
    stems: list[str] = []
    for slot, value in _source_items(sources):
        if slot != "customer_name":
            continue
        parts = [w for w in str(value).lower().replace("-", " ").split() if w.isalpha()]
        if len(parts) >= 2:
            stems.append(parts[0][0] + parts[-1])
            stems.append(parts[0] + parts[-1][0])
    return list(dict.fromkeys(stems))


def delexicalize_sentence(
    sentence: str,
    sources: SlotSources,
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Delexicalize one sentence and report what was replaced (spec 6.2 step 2).

    See :func:`reflex.contracts.delexicalize_sentence` for the frozen contract.

    Three tiers run in the contract's order: ABCD's own markers are CONVERTED
    (never re-derived), then conversation values are matched exactly, then the
    shape-only regex classes named by ``compile.delex_regex_slots`` fire. The
    NER tier is controlled by ``compile.delex_ner_model`` and is off: a local
    NER pass would buy 9 masked person-names per 28,758 train agent turns
    (measured) and would risk mangling product names, and a hosted one is
    forbidden outright.

    GLUED MARKERS, AND THE ONE PLACE THE CONTRACT ASKS FOR TWO THINGS AT ONCE.
    The contract says a glued marker must stay "literal surface text" AND that
    ``text_delex`` must never contain ``<``. Both cannot hold literally, so the
    angle brackets are dropped and the marker's bare word is left in the text
    (``"$1<amount>"`` -> ``"$1amount"``), which keeps the span out of the slot
    vocabulary -- the point of the guard -- while honouring the ``<`` rule. Such
    spans are rare (300 of 17,073 train markers, 9 of them in an agent turn) and
    the templates carrying them are nearly always singletons dropped by
    ``compile.min_template_count``.
    """
    guard_glued = bool(get_dotted(cfg, "compile.guard_glued_markers"))
    min_chars = int(get_dotted(cfg, "compile.min_literal_match_chars"))
    regex_slots = tuple(get_dotted(cfg, "compile.delex_regex_slots") or ())
    match_types = frozenset(get_dotted(cfg, "compile.delex_match_types") or ())
    ner_model = get_dotted(cfg, "compile.delex_ner_model")

    mapping: dict[str, str] = {}
    kept: list[str] = []

    def _emit(slot: str) -> str:
        if slot not in registry.slots:
            raise ContractViolation(
                f"delexicalization would emit slot {slot!r}, which is not in the slot "
                f"registry; spec 6.3 makes the registry the only source of slot names"
            )
        kept.append("{" + slot + "}")
        return _SENTINEL.format(len(kept) - 1)

    # --- tier 1: convert ABCD's own markers -------------------------------- #
    pieces: list[str] = []
    cursor = 0
    for match in _MARKER_RE.finditer(sentence):
        pieces.append(sentence[cursor:match.start()])
        cursor = match.end()
        slot = registry.marker_to_slot.get(match.group(0))
        glued = not _boundary_ok(sentence, match.start(), match.end())
        if slot is None or (guard_glued and glued):
            pieces.append(match.group(1))  # literal word, no slot, no '<'
            continue
        pieces.append(_emit(slot))
        for candidate_slot, value in _source_items(sources):
            if candidate_slot == slot and str(value).strip().lower() not in _NULL_VALUES:
                mapping.setdefault(slot, str(value))
                break
    pieces.append(sentence[cursor:])
    text = "".join(pieces)

    # --- tier 2: exact match against this conversation's values ------------ #
    for slot, value in _source_items(sources):
        spec = registry.slots.get(slot)
        if spec is None or spec.type not in match_types:
            # M5: `other`-typed slots (membership_level, payment_method,
            # shipping_status) stay in the registry for action patterns but are
            # never masked in text -- "are you a gold or silver member?" must not
            # become "are you a {membership_level} or {membership_level} member?".
            continue
        for surface in _surface_variants(value):
            if len(surface) < min_chars:
                continue
            lowered = text.lower()
            needle = surface.lower()
            start = lowered.find(needle)
            while start != -1:
                end = start + len(needle)
                if _boundary_ok(text, start, end):
                    mapping.setdefault(slot, text[start:end])
                    text = text[:start] + _emit(slot) + text[end:]
                    lowered = text.lower()
                    start = lowered.find(needle)
                else:
                    start = lowered.find(needle, start + 1)

    # --- tier 2.5: usernames the agent DERIVES from the customer's name ----- #
    # These are invented in-flow and are absent from the scenario, so tier 2
    # cannot see them. Without this the bank leaks real handles (D-1).
    if bool(get_dotted(cfg, "compile.derive_username_from_name")) and "username" in registry.slots:
        for stem in _derived_username_stems(sources):
            pattern = re.compile(r"\b" + re.escape(stem) + r"\d+\b", re.IGNORECASE)
            while True:
                match = pattern.search(text)
                if match is None:
                    break
                mapping.setdefault("username", match.group(0))
                text = text[:match.start()] + _emit("username") + text[match.end():]

    # --- tier 3: shape-only regexes ---------------------------------------- #
    for slot in regex_slots:
        pattern = _INTEXT_PATTERNS.get(slot)
        if pattern is None:
            raise ValueError(
                f"compile.delex_regex_slots names {slot!r}, which has no in-text pattern; "
                f"known: {sorted(_INTEXT_PATTERNS)}"
            )
        if slot not in registry.slots:
            raise ContractViolation(
                f"compile.delex_regex_slots names {slot!r}, which is not in the slot registry"
            )
        while True:
            match = pattern.search(text)
            if match is None:
                break
            mapping.setdefault(slot, match.group(0))
            text = text[:match.start()] + _emit(slot) + text[match.end():]

    # --- tier 4: local NER, deliberately off ------------------------------- #
    if ner_model:
        raise ValueError(
            f"compile.delex_ner_model is set to {ner_model!r} but no local NER tier is "
            f"implemented in this build; set it to null. A hosted NER model is forbidden "
            f"(ZERO PAID API CALLS)."
        )

    text = _unmask_spans(text, kept)
    if "<" in text or ">" in text:
        text = text.replace("<", "").replace(">", "")
    return re.sub(r"\s+", " ", text).strip(), mapping


def _surface_variants(value: str) -> tuple[str, ...]:
    """Surfaces worth searching for one scenario value.

    ABCD writes product names as ``michael_kors jeans`` in the scenario and
    ``michael kors jeans`` in the transcript, so the underscore variant is
    always tried too.
    """
    value = str(value).strip()
    if not value:
        return ()
    variants = [value]
    if "_" in value:
        variants.append(value.replace("_", " "))
    return tuple(dict.fromkeys(variants))


# --------------------------------------------------------------------------- #
# 6.2 step 3 -- act labeling
# --------------------------------------------------------------------------- #


class _RulesPlusEmbedLabeler:
    """The FREE LOCAL act labeler (``compile.act_labeler: rules_plus_embed``).

    Cue rules decide the easy acts; everything else is assigned by cosine to the
    nearest act centroid, where a centroid is the mean of that act's frozen seed
    embeddings from ``compile.act_labeler_embed_model``. Deterministic by
    construction: the rules are ordered, the seeds are frozen, the model runs in
    eval mode under ``torch.no_grad`` and every batch is padded to a constant
    width so a sentence's embedding cannot depend on its batch-mates.
    """

    name = "rules_plus_embed"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._inventory = _act_inventory(cfg)
        self._rules = _load_act_rules(cfg)
        self._floor = float(get_dotted(cfg, "compile.act_embed_min_similarity"))
        self._centroids: Any = None
        self._centroid_acts: tuple[str, ...] = ()

    # -- public interface (the ActLabeler protocol) ------------------------- #

    def label(self, sentences: Sequence[str]) -> list[str]:
        """Return one act per input sentence, same length and order as input."""
        return [act for act, _ in self.label_with_provenance(sentences)]

    # -- used by write_act_check, which reports how each label was reached --- #

    def label_with_provenance(self, sentences: Sequence[str]) -> list[tuple[str, str]]:
        """Return ``(act, provenance)`` per sentence; provenance is a rule or 'embed:<cos>'."""
        out: list[Optional[tuple[str, str]]] = [None] * len(sentences)
        unresolved: list[int] = []
        for i, sentence in enumerate(sentences):
            hit = self._rule_label(sentence)
            if hit is None:
                unresolved.append(i)
            else:
                act, index = hit
                out[i] = (act, f"rule:{index}")
        if unresolved:
            texts = [sentences[i] for i in unresolved]
            acts, scores = self._nearest_centroid(texts)
            for i, act, score in zip(unresolved, acts, scores):
                out[i] = (act, f"embed:{score:.3f}")
        return [item if item is not None else ("OTHER", "fallback") for item in out]

    # -- internals ----------------------------------------------------------- #

    def _rule_label(self, sentence: str) -> Optional[tuple[str, int]]:
        text = sentence.strip().lower()
        if not text:
            return ("OTHER", -1)
        for index, (act, pattern) in enumerate(self._rules):
            if pattern.search(text):
                return (act, index)
        return None

    def _ensure_centroids(self) -> None:
        if self._centroids is not None:
            return
        import numpy as np

        acts = tuple(act for act in self._inventory if _ACT_SEEDS.get(act))
        seeds: list[str] = []
        spans: list[tuple[int, int]] = []
        for act in acts:
            start = len(seeds)
            seeds.extend(_ACT_SEEDS[act])
            spans.append((start, len(seeds)))
        vectors = _embed_texts(seeds, self._cfg, "compile.act_labeler_embed_model")
        centroids = np.stack([vectors[a:b].mean(axis=0) for a, b in spans])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self._centroids = centroids / np.clip(norms, 1e-12, None)
        self._centroid_acts = acts

    def _nearest_centroid(self, texts: Sequence[str]) -> tuple[list[str], list[float]]:
        self._ensure_centroids()
        import numpy as np

        # Embed each DISTINCT sentence once, in sorted order: identical text must
        # get an identical label no matter where it appeared.
        unique = sorted(set(texts))
        vectors = _embed_texts(unique, self._cfg, "compile.act_labeler_embed_model")
        scores = vectors @ self._centroids.T
        best = np.argmax(scores, axis=1)
        table = {
            text: (
                self._centroid_acts[int(best[i])] if float(scores[i, best[i]]) >= self._floor else "OTHER",
                float(scores[i, best[i]]),
            )
            for i, text in enumerate(unique)
        }
        acts = [table[t][0] for t in texts]
        sims = [table[t][1] for t in texts]
        return acts, sims


class _LLMActLabeler:
    """Spec 6.2 step 3a's original LLM labeler -- implemented, and off.

    Constructing it while ``llm.enabled`` is false raises
    :class:`reflex.contracts.LLMDisabledError` from
    :func:`reflex.llm_agent.build_llm_agent`, which is the paid-call kill switch.
    """

    name = "llm"

    def __init__(self, cfg: dict[str, Any]) -> None:
        from reflex.llm_agent import build_llm_agent, load_prompt

        self._cfg = cfg
        self._inventory = _act_inventory(cfg)
        self._agent = build_llm_agent(cfg, str(get_dotted(cfg, "compile.act_label_model")))
        self._prompt, self._prompt_hash = load_prompt(cfg, "act_labeling")
        self._batch = int(get_dotted(cfg, "compile.act_label_batch_size"))
        self._max_tokens = int(get_dotted(cfg, "compile.act_label_max_tokens"))

    def label(self, sentences: Sequence[str]) -> list[str]:
        """Label in batches of ``compile.act_label_batch_size`` at temperature 0."""
        system, user_template = self._prompt.split("[USER]", 1)
        system = system.replace("[SYSTEM]", "").strip()
        backend = getattr(self._agent, "backend", None)
        if backend is None:  # pragma: no cover - unreachable while the switch is off
            raise ReflexError("llm act labeler: the agent exposes no backend to call")
        out: list[str] = []
        for start in range(0, len(sentences), self._batch):
            chunk = sentences[start:start + self._batch]
            numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(chunk))
            user = user_template.replace("{numbered_sentences}", numbered).strip()
            text, _, _ = backend.complete(
                system, user, float(get_dotted(self._cfg, "llm.temperature")), self._max_tokens
            )
            out.extend(self._parse(text, len(chunk)))
        return out

    def _parse(self, raw: str, expected: int) -> list[str]:
        labels = ["OTHER"] * expected
        for line in raw.splitlines():
            match = re.match(r"\s*(\d+)\s*[.)]\s*([A-Z]+)", line.strip())
            if not match:
                continue
            index = int(match.group(1)) - 1
            act = match.group(2)
            if 0 <= index < expected and act in self._inventory:
                labels[index] = act
        return labels


def _load_act_rules(cfg: dict[str, Any]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile the act cue rules: ``compile.act_rules_path`` if set, else built-in."""
    path = get_dotted(cfg, "compile.act_rules_path")
    if not path:
        raw = _BUILTIN_ACT_RULES
    else:
        import yaml

        resolved = resolve_path(cfg, "compile.act_rules_path")
        with open(resolved, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or []
        raw = tuple((str(item["act"]), str(item["pattern"])) for item in loaded)
    inventory = _act_inventory(cfg)
    rules: list[tuple[str, re.Pattern[str]]] = []
    for act, pattern in raw:
        if act not in inventory:
            raise ContractViolation(f"act rule names {act!r}, which is not in the act inventory")
        rules.append((act, re.compile(pattern)))
    return tuple(rules)


#: Encoder cache, keyed by model id. Loading MiniLM costs ~1 s.
_EMBEDDERS: dict[str, tuple[Any, Any]] = {}


def _embedder(cfg: dict[str, Any], model_id: str) -> tuple[Any, Any]:
    if model_id in _EMBEDDERS:
        return _EMBEDDERS[model_id]
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(int(get_dotted(cfg, "runtime.torch_threads")))
    local_only = bool(get_dotted(cfg, "model.local_files_only"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        model = AutoModel.from_pretrained(model_id, local_files_only=local_only)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ReflexError(
            f"could not load the frozen embedding model {model_id!r} "
            f"(local_files_only={local_only}): {exc}. The compiler will not silently "
            f"degrade to labelling everything OTHER; fix the model id or the cache."
        ) from exc
    model.eval()
    _EMBEDDERS[model_id] = (tokenizer, model)
    return _EMBEDDERS[model_id]


def _embed_texts(texts: Sequence[str], cfg: dict[str, Any], model_key: str) -> Any:
    """L2-normalized mean-pooled embeddings, one row per text, order preserved.

    ``model_key`` is the dotted config key naming the frozen encoder
    (``compile.act_labeler_embed_model`` or ``compile.dedup_embed_model``).
    Every batch is padded to ``compile.embed_max_len`` so a row is independent of
    what else shares its batch -- determinism (spec 10) beats the throughput that
    dynamic padding would buy.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype="float32")
    import torch

    model_id = str(get_dotted(cfg, model_key))
    tokenizer, model = _embedder(cfg, model_id)
    max_len = int(get_dotted(cfg, "compile.embed_max_len"))
    batch_size = int(get_dotted(cfg, "compile.embed_batch_size"))
    chunks: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            chunks.append(pooled.cpu().numpy().astype("float32"))
    return np.concatenate(chunks, axis=0)


def get_act_labeler(cfg: dict[str, Any]) -> ActLabeler:
    """Return the configured act labeler (spec 6.2 step 3a).

    See :func:`reflex.contracts.get_act_labeler` for the frozen contract.
    """
    name = str(get_dotted(cfg, "compile.act_labeler"))
    if name == "rules_plus_embed":
        return _RulesPlusEmbedLabeler(cfg)
    if name == "llm":
        return _LLMActLabeler(cfg)
    raise ValueError(
        f"unknown compile.act_labeler {name!r}; expected 'rules_plus_embed' (free, local, "
        f"the default) or 'llm' (requires llm.enabled, which is false)"
    )


def label_acts(sentences: Sequence[str], cfg: dict[str, Any]) -> list[str]:
    """Label every train sentence with an act (spec 6.2 step 3a).

    See :func:`reflex.contracts.label_acts` for the frozen contract.
    """
    if not sentences:
        return []
    return get_act_labeler(cfg).label(sentences)


# --------------------------------------------------------------------------- #
# 6.2 step 4 -- templates
# --------------------------------------------------------------------------- #


def _assign_template_ids(rows: Sequence[Template]) -> list[Template]:
    """Re-id templates by descending count then text, so ids are reproducible."""
    ordered = sorted(rows, key=lambda t: (-t.count, t.text_delex))
    return [
        Template(
            template_id=f"T{i:06d}",
            act=row.act,
            text_delex=row.text_delex,
            slots=row.slots,
            count=row.count,
            surface_forms=row.surface_forms,
            example_context_ids=row.example_context_ids,
        )
        for i, row in enumerate(ordered)
    ]


def extract_templates(
    labeled_sentences: Sequence[tuple[str, str, str]],
    cfg: dict[str, Any],
) -> list[Template]:
    """Build raw :class:`Template` objects, one per distinct ``(act, text_delex)``.

    See :func:`reflex.contracts.extract_templates` for the frozen contract.
    """
    inventory = _act_inventory(cfg)
    max_examples = int(get_dotted(cfg, "compile.max_example_context_ids"))
    counts: dict[tuple[str, str], int] = defaultdict(int)
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for turn_id, act, text in labeled_sentences:
        if act not in inventory:
            raise ContractViolation(
                f"act {act!r} is not in the inventory {inventory}; index into that tuple is "
                f"head H6's class id"
            )
        key = (act, text)
        counts[key] += 1
        bucket = examples[key]
        if len(bucket) < max_examples and turn_id not in bucket:
            bucket.append(turn_id)
    rows = [
        Template(
            template_id="",
            act=act,  # type: ignore[arg-type]
            text_delex=text,
            slots=_slots_in(text),
            count=count,
            surface_forms=[text],
            example_context_ids=list(examples[(act, text)]),
        )
        for (act, text), count in counts.items()
    ]
    return _assign_template_ids(rows)


def _requested_fields(text: str, terms: "dict[str, str] | None") -> frozenset:
    """The set of named fields an utterance asks the customer for.

    Used to BLOCK a dedup merge between two forms that request different things
    (spec 6.2 step 4, as amended by DECISIONS D16). Longest surface phrase wins,
    so "account number" is not also counted as "account".

    Returns an empty set for text that requests no field; such templates are
    unconstrained and merge on cosine alone, as before.
    """
    if not terms:
        return frozenset()
    # punctuation -> space, so "order id?" and "order id," still match
    cleaned = "".join(c if (c.isalnum() or c == "#") else " " for c in text.lower())
    low = " " + " ".join(cleaned.split()) + " "
    found: set = set()
    for phrase in sorted(terms, key=len, reverse=True):
        norm = "".join(c if (c.isalnum() or c == "#") else " " for c in phrase.lower())
        token = " " + " ".join(norm.split()) + " "
        while token in low:
            found.add(terms[phrase])
            low = low.replace(token, "  ", 1)
    return frozenset(found)


def _union_find_clusters(
    vectors: Any,
    threshold: float,
    chunk: int,
    field_sets: "list[frozenset] | None" = None,
) -> list[int]:
    """Single-linkage cluster labels over L2-normalized rows at cosine >= threshold.

    When ``field_sets`` is given, a pair is only joined if both rows request the
    SAME named fields. Field-set equality is an equivalence relation, so refusing
    mismatched pairwise unions is sufficient to make every resulting cluster
    field-homogeneous -- no post-hoc splitting is needed.
    """
    import numpy as np

    n = int(vectors.shape[0])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for start in range(0, n, chunk):
        block = vectors[start:start + chunk] @ vectors.T
        rows, cols = np.nonzero(block >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            i = start + row
            if col > i:
                if field_sets is not None and field_sets[i] != field_sets[col]:
                    continue
                union(i, col)
    return [find(i) for i in range(n)]


def dedup_templates(
    templates: Sequence[Template],
    cfg: dict[str, Any],
) -> list[Template]:
    """Merge near-duplicate templates and drop rare ones (spec 6.2 step 4).

    See :func:`reflex.contracts.dedup_templates` for the frozen contract.

    ONE DELIBERATE DEVIATION: the contract says the cosine comes from
    ``model.encoder``. It cannot. Raw ModernBERT-base is not sentence-trained and
    scores 59% of arbitrary ABCD sentence pairs at cosine >= 0.92 (M3 in the
    module docstring), so single-linkage at ``compile.dedup_threshold`` would
    fuse the entire bank. ``compile.dedup_embed_model`` names a sentence-trained
    encoder instead; the threshold keeps its intended meaning.
    """
    if not templates:
        return []
    threshold = float(get_dotted(cfg, "compile.dedup_threshold"))
    min_count = int(get_dotted(cfg, "compile.min_template_count"))
    max_examples = int(get_dotted(cfg, "compile.max_example_context_ids"))
    chunk = int(get_dotted(cfg, "compile.embed_batch_size"))

    # -- exact merge on the normalized form, within one act ------------------ #
    exact: dict[tuple[str, str], list[Template]] = defaultdict(list)
    for row in templates:
        exact[(row.act, _normalize_for_dedup(row.text_delex))].append(row)
    reps: list[Template] = [_merge_group(rows, max_examples) for rows in exact.values()]

    # -- single-linkage cosine merge, within one act ------------------------- #
    by_act: dict[str, list[Template]] = defaultdict(list)
    for row in reps:
        by_act[row.act].append(row)

    merged: list[Template] = []
    for act in sorted(by_act):
        rows = sorted(by_act[act], key=lambda t: (-t.count, t.text_delex))
        if len(rows) == 1:
            merged.append(rows[0])
            continue
        vectors = _embed_texts([r.text_delex for r in rows], cfg, "compile.dedup_embed_model")
        field_sets = None
        if bool(get_dotted(cfg, "compile.merge_block_on_field_mismatch")):
            terms = get_dotted(cfg, "compile.merge_field_terms")
            field_sets = [_requested_fields(r.text_delex, terms) for r in rows]
        labels = _union_find_clusters(vectors, threshold, max(chunk, 1), field_sets)
        clusters: dict[int, list[Template]] = defaultdict(list)
        for label, row in zip(labels, rows):
            clusters[label].append(row)
        for label in sorted(clusters):
            merged.append(_merge_group(clusters[label], max_examples))

    kept = [row for row in merged if row.count >= min_count]
    return _assign_template_ids(kept)


def _merge_group(rows: Sequence[Template], max_examples: int) -> Template:
    """Collapse one cluster: most frequent form wins, ties broken alphabetically."""
    ordered = sorted(rows, key=lambda t: (-t.count, t.text_delex))
    canonical = ordered[0]
    forms: list[str] = []
    examples: list[str] = []
    for row in ordered:
        for form in [row.text_delex, *row.surface_forms]:
            if form not in forms:
                forms.append(form)
        for example in row.example_context_ids:
            if example not in examples and len(examples) < max_examples:
                examples.append(example)
    return Template(
        template_id="",
        act=canonical.act,
        text_delex=canonical.text_delex,
        slots=_slots_in(canonical.text_delex),
        count=sum(row.count for row in rows),
        surface_forms=forms,
        example_context_ids=examples,
    )


# --------------------------------------------------------------------------- #
# 6.2 steps 5 and 6 -- skeletons and action patterns
# --------------------------------------------------------------------------- #


def extract_skeletons(
    utterance_acts: Sequence[Sequence[str]],
    cfg: dict[str, Any],
) -> list[Skeleton]:
    """Extract skeletons: the ordered act tuple of each agent utterance (spec 6.2 step 5).

    See :func:`reflex.contracts.extract_skeletons` for the frozen contract. Every
    observed skeleton is kept -- spec 6.2 step 5 sets no minimum count.
    """
    inventory = _act_inventory(cfg)
    counts: Counter[tuple[str, ...]] = Counter()
    for acts in utterance_acts:
        tuple_acts = tuple(acts)
        if not tuple_acts:
            continue
        for act in tuple_acts:
            if act not in inventory:
                raise ContractViolation(f"skeleton act {act!r} is not in the act inventory")
        counts[tuple_acts] += 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        Skeleton(skeleton_id=f"S{i:04d}", acts=list(acts), count=count)
        for i, (acts, count) in enumerate(ordered)
    ]


def extract_action_patterns(
    action_turns: Sequence[NormalizedTurn],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> list[ActionPattern]:
    """Extract one :class:`ActionPattern` per button seen in train (spec 6.2 step 6).

    See :func:`reflex.contracts.extract_action_patterns` for the frozen contract.

    ``required_slots`` holds only the positions whose value the FILLER could
    source: an entity (a name, an id, an address, a membership level). ABCD's
    other value categories -- ``reason_slotval``, ``change_option``,
    ``single_item_query`` and friends -- are choices head H4 predicts, not values
    that exist anywhere to be looked up, and listing them would make the gate
    escalate every such turn on ``unavailable_slot``.
    """
    ontology = _data.load_ontology(cfg)
    enumerable = {
        category: [str(v).lower() for v in values]
        for category, values in ontology.get("values", {}).get("enumerable", {}).items()
    }
    declared: dict[str, list[str]] = {}
    for _category, buttons in ontology.get("actions", {}).items():
        for button, categories in buttons.items():
            declared[button] = list(categories)

    counts: Counter[str] = Counter()
    arities: dict[str, Counter[int]] = defaultdict(Counter)
    position_slots: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for turn in action_turns:
        action = turn.action
        if not action:
            continue
        values = list(turn.values or [])
        counts[action] += 1
        arities[action][len(values)] += 1
        for position, value in enumerate(values):
            slot = _type_literal(str(value), declared.get(action, []), enumerable)
            if slot is not None and slot not in registry.slots:
                slot = None
            # VOTE ON EVERY VALUE, INCLUDING THE UNTYPABLE ONES. Counting only
            # the typed ones let a rare stray win a position whose values are
            # overwhelmingly agent choices: measured on train, that made
            # notify-team require customer_name on 24 of 454 values (5.3%) and
            # update-order require amount on 105 of 1,288 (8.2%). required_slots
            # feeds fill.check_availability, which is the gate's
            # unavailable_slot signal, so those turns would have escalated for a
            # value they never carried.
            position_slots[action][position][slot or _NO_SLOT] += 1

    order = {name: i for i, name in enumerate(_data.action_list(ontology))}
    patterns: list[ActionPattern] = []
    for action in sorted(counts, key=lambda a: (order.get(a, len(order)), a)):
        modal_arity = max(arities[action].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        required: list[str] = []
        for position in range(modal_arity):
            observed = position_slots[action].get(position)
            if not observed:
                continue
            slot = max(sorted(observed.items()), key=lambda kv: kv[1])[0]
            if slot == _NO_SLOT:
                # Most values here are agent choices head H4 predicts
                # (reason_slotval, order_slotval, ...). Nothing to source.
                continue
            if slot not in required:
                required.append(slot)
        patterns.append(
            ActionPattern(action=action, required_slots=required, count=counts[action])
        )
    return patterns


# --------------------------------------------------------------------------- #
# Compile-time slot sources (private -- see the module docstring)
# --------------------------------------------------------------------------- #


def _conversation_sources(
    scenario: dict[str, Any],
    turns: Sequence[NormalizedTurn],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> SlotSources:
    """Every value this conversation is known to contain, bucketed by origin.

    Deliberately NOT disclosure-filtered and deliberately whole-conversation: at
    compile time an unmasked value is a leak into the bank, so the safe error is
    to mask too much. :func:`reflex.fill.collect_slot_sources` is the
    inference-time function and keeps the guard.
    """
    ontology_enumerable: dict[str, list[str]] = {}
    from_scenario: dict[str, str] = {}
    for block_name in ("personal", "order"):
        block = scenario.get(block_name)
        if not isinstance(block, dict):
            continue
        for field, value in block.items():
            slot = _SCENARIO_FIELD_TO_SLOT.get((block_name, field))
            if slot and slot in registry.slots and isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text and text.lower() not in _NULL_VALUES:
                    from_scenario.setdefault(slot, text)
    product = scenario.get("product")
    if isinstance(product, dict):
        for name in product.get("names", []) or []:
            if "name" in registry.slots:
                from_scenario.setdefault("name", str(name))
                break
        for amount in product.get("amounts", []) or []:
            if "amount" in registry.slots:
                from_scenario.setdefault("amount", str(amount))
                break

    from_actions: dict[str, str] = {}
    for turn in turns:
        if turn.nextstep != "take_action" or not turn.values:
            continue
        for value in turn.values:
            slot = _type_literal(str(value), [], ontology_enumerable)
            if slot and slot in registry.slots:
                from_actions.setdefault(slot, str(value))

    from_customer: dict[str, str] = {}
    for turn in turns:
        if turn.speaker != "customer":
            continue
        lowered = turn.text.lower()
        for slot, value in list(from_scenario.items()) + list(from_actions.items()):
            if slot in from_customer:
                continue
            for surface in _surface_variants(value):
                if len(surface) >= int(get_dotted(cfg, "compile.min_literal_match_chars")) and surface.lower() in lowered:
                    from_customer[slot] = value
                    break
        for slot, pattern in _INTEXT_PATTERNS.items():
            if slot in from_customer or slot not in registry.slots:
                continue
            match = pattern.search(turn.text)
            if match:
                from_customer[slot] = match.group(0)
    return SlotSources(
        from_customer=from_customer, from_actions=from_actions, from_scenario=from_scenario
    )


# --------------------------------------------------------------------------- #
# The whole compiler
# --------------------------------------------------------------------------- #


def assert_no_leaked_usernames(templates: "Sequence[Template]", ontology: dict) -> None:
    """FAIL THE BUILD if a persona-derived username survived into the bank (D-1).

    A PII leak must break the compile, not warn. Measured before the fix: 22
    templates / 248 occurrences of real handles (``safzal1`` 32x, ``rdomingo1``
    30x, ``cminh1`` 24x, ...) across ten personas. A fast path serving that bank
    hands one customer another person's username.

    Stems are built from the ontology's own ``customer_name`` list, so the check
    does not depend on any one conversation's sources.
    """
    names = (ontology.get("values", {}).get("enumerable", {}).get("customer_name") or [])
    stems: set = set()
    for raw in names:
        parts = [w for w in str(raw).lower().replace("-", " ").split() if w.isalpha()]
        if len(parts) >= 2:
            stems.add(parts[0][0] + parts[-1])
            stems.add(parts[0] + parts[-1][0])
    if not stems:
        return
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, stems))) + r")\d+\b", re.I)
    offenders = []
    for row in templates:
        for text in [row.text_delex, *getattr(row, "surface_forms", [])]:
            hit = pattern.search(text)
            if hit:
                offenders.append((row.template_id, row.count, hit.group(0), text[:70]))
                break
    if offenders:
        detail = "; ".join(f"{tid} (x{cnt}) {tok!r} in {txt!r}" for tid, cnt, tok, txt in offenders[:5])
        raise ContractViolation(
            f"PII LEAK: {len(offenders)} template(s) contain a persona-derived username. "
            f"A bank carrying real handles must not ship. First: {detail}"
        )


def compile_bank(cfg: dict[str, Any], fraction: float = 1.0) -> Bank:
    """Run the whole compiler (spec 6.2) on the TRAIN split only and return the bank.

    See :func:`reflex.contracts.compile_bank` for the frozen contract.

    It also writes ``labels/train.jsonl`` (spec 6.2 step 7) through
    :func:`write_turn_labels` and a provenance sidecar, because this is the only
    place the per-turn labels exist.
    """
    split = "train"
    ontology = _data.load_ontology(cfg)
    registry = build_slot_registry(ontology, cfg)
    partitions = _data.build_partitions(cfg)
    train = partitions.train
    if fraction != 1.0:
        keep = set(
            _data.learning_curve_subset(
                list(train), fraction, int(get_dotted(cfg, "data.novel_subflows_seed"))
            )
        )
        train = {cid: turns for cid, turns in train.items() if cid in keep}

    raw = _data.load_raw_abcd(cfg)
    scenarios = {int(convo["convo_id"]): convo.get("scenario", {}) for convo in raw[split]}

    # The evidence _type_literal needs to type an action value: the button's
    # declared value categories and the enumerable value lists (same two inputs
    # extract_action_patterns builds).
    enumerable_values = {
        category: [str(v).lower() for v in values]
        for category, values in ontology.get("values", {}).get("enumerable", {}).items()
    }
    action_categories: dict[str, list[str]] = {}
    for _section, buttons in ontology.get("actions", {}).items():
        for button, categories in buttons.items():
            action_categories[button] = list(categories)

    sources_cache: dict[int, SlotSources] = {}
    triples: list[tuple[str, str, str]] = []          # (turn_id, act placeholder, text)
    sentence_owner: list[tuple[str, int]] = []        # (turn_id, sentence position)
    per_turn_sentences: dict[str, list[str]] = {}
    per_turn_slots: dict[str, dict[str, str]] = {}
    action_turns: list[NormalizedTurn] = []
    label_stubs: list[tuple[str, NormalizedTurn]] = []

    for convo_id, turn_index, turn in _data.iter_agent_turns(train):
        turn_id = _data.turn_key(split, convo_id, turn_index)
        label_stubs.append((turn_id, turn))
        if turn.nextstep == "take_action":
            action_turns.append(turn)
            values = {}
            for value in turn.values or []:
                # TYPE AGAINST THE BUTTON'S DECLARED CATEGORIES, exactly as
                # extract_action_patterns does. Typing with no categories and no
                # enumerable index lets the shape patterns fire unconstrained,
                # which mistyped 4,699 of 21,335 stored entries on train
                # (22.0%): shipping-status "in transit" and update-order
                # "change order" both landed as customer_name. This mapping is
                # what spec 6.2 step 2b hands to the filler.
                slot = _type_literal(
                    str(value), action_categories.get(turn.action or "", []), enumerable_values
                )
                if slot and slot in registry.slots:
                    values.setdefault(slot, str(value))
            per_turn_slots[turn_id] = values
            continue
        if turn.nextstep != "retrieve_utterance":
            continue
        if convo_id not in sources_cache:
            sources_cache[convo_id] = _conversation_sources(
                scenarios.get(convo_id, {}), train[convo_id], registry, cfg
            )
        sources = sources_cache[convo_id]
        sentences: list[str] = []
        slot_values: dict[str, str] = {}
        for sentence in split_sentences(turn.text, cfg):
            text_delex, mapping = delexicalize_sentence(sentence, sources, registry, cfg)
            if not text_delex:
                continue
            sentences.append(text_delex)
            for slot, surface in mapping.items():
                slot_values.setdefault(slot, surface)
        per_turn_sentences[turn_id] = sentences
        per_turn_slots[turn_id] = slot_values
        for position, text_delex in enumerate(sentences):
            sentence_owner.append((turn_id, position))
            triples.append((turn_id, "", text_delex))

    acts = label_acts([text for _, _, text in triples], cfg)
    triples = [(turn_id, act, text) for (turn_id, _, text), act in zip(triples, acts)]

    per_turn_acts: dict[str, list[str]] = defaultdict(list)
    for (turn_id, _position), act in zip(sentence_owner, acts):
        per_turn_acts[turn_id].append(act)

    raw_templates = extract_templates(triples, cfg)
    templates = dedup_templates(raw_templates, cfg)
    skeletons = extract_skeletons(
        [per_turn_acts[t] for t in per_turn_sentences if per_turn_acts.get(t)], cfg
    )
    actions = extract_action_patterns(action_turns, registry, cfg)

    templates_by_act: dict[str, list[str]] = defaultdict(list)
    for template in templates:
        templates_by_act[template.act].append(template.template_id)

    # (act, normalized surface form) -> surviving template id. Built from the
    # merged surface forms so a sentence that lost its own row to a merge still
    # resolves to the row that absorbed it.
    lookup: dict[tuple[str, str], str] = {}
    for template in templates:
        for form in template.surface_forms or [template.text_delex]:
            lookup.setdefault((template.act, _normalize_for_dedup(form)), template.template_id)
    skeleton_ids = {tuple(s.acts): s.skeleton_id for s in skeletons}

    labels: list[TurnLabel] = []
    covered_turns = 0
    covered_sentences = 0
    total_sentences = 0
    retrieve_turns = 0
    for turn_id, turn in label_stubs:
        if turn.nextstep == "retrieve_utterance":
            retrieve_turns += 1
            sentences = per_turn_sentences.get(turn_id, [])
            turn_acts = per_turn_acts.get(turn_id, [])
            template_ids = [
                lookup.get((act, _normalize_for_dedup(text)), "")
                for act, text in zip(turn_acts, sentences)
            ]
            total_sentences += len(sentences)
            covered_sentences += sum(1 for tid in template_ids if tid)
            if sentences and all(template_ids):
                covered_turns += 1
            labels.append(
                TurnLabel(
                    turn_id=turn_id,
                    nextstep=str(turn.nextstep),
                    intent=str(turn.intent),
                    skeleton_id=skeleton_ids.get(tuple(turn_acts)),
                    template_ids=template_ids,
                    action=None,
                    values=None,
                    slot_values=per_turn_slots.get(turn_id, {}),
                )
            )
        else:
            labels.append(
                TurnLabel(
                    turn_id=turn_id,
                    nextstep=str(turn.nextstep),
                    intent=str(turn.intent),
                    skeleton_id=None,
                    template_ids=None,
                    action=turn.action,
                    values=list(turn.values) if turn.values else None,
                    slot_values=per_turn_slots.get(turn_id, {}),
                )
            )

    assert_no_leaked_usernames(templates, ontology)

    bank = Bank(
        templates=templates,
        skeletons=skeletons,
        actions=actions,
        slot_registry=registry,
        templates_by_act={act: templates_by_act[act] for act in sorted(templates_by_act)},
        source_fraction=float(fraction),
        bank_hash="",
    )
    bank = Bank(
        templates=bank.templates,
        skeletons=bank.skeletons,
        actions=bank.actions,
        slot_registry=bank.slot_registry,
        templates_by_act=bank.templates_by_act,
        source_fraction=bank.source_fraction,
        bank_hash=_sha16(b"".join(_bank_payloads(bank).values())),
    )
    write_turn_labels(labels, split, cfg)
    _write_compile_stats(
        bank,
        cfg,
        {
            "train_conversations": len(train),
            "agent_side_turns": len(label_stubs),
            "retrieve_utterance_turns": retrieve_turns,
            "take_action_turns": len(action_turns),
            "sentences": total_sentences,
            "raw_templates": len(raw_templates),
            "template_coverage_turns": covered_turns / retrieve_turns if retrieve_turns else 0.0,
            "template_coverage_sentences": (
                covered_sentences / total_sentences if total_sentences else 0.0
            ),
            "novel_subflows": partitions.novel_subflows,
            "dataset_hash": partitions.dataset_hash,
        },
    )
    return bank


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _jsonl(rows: Iterable[Any]) -> bytes:
    return "".join(
        json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _bank_payloads(bank: Bank) -> dict[str, bytes]:
    """The exact bytes of the four spec 6.2 step 7 files, in a fixed order."""
    registry = {
        "slots": {name: spec.to_dict() for name, spec in sorted(bank.slot_registry.slots.items())},
        "marker_to_slot": dict(sorted(bank.slot_registry.marker_to_slot.items())),
    }
    return {
        _F_TEMPLATES: _jsonl(bank.templates),
        _F_SKELETONS: _jsonl(bank.skeletons),
        _F_ACTIONS: _jsonl(bank.actions),
        _F_REGISTRY: (json.dumps(registry, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    }


def write_bank(bank: Bank, cfg: dict[str, Any]) -> dict[str, str]:
    """Persist the bank to the four spec 6.2 step 7 files.

    See :func:`reflex.contracts.write_bank` for the frozen contract. A fifth
    ``meta.json`` sidecar carries ``source_fraction`` and ``bank_hash``: the four
    spec files have nowhere to put them, and :class:`SlotRegistry` rejects
    unknown keys by design, so hiding them there would break the schema.
    """
    bank_dir = resolve_path(cfg, "paths.bank_dir")
    os.makedirs(bank_dir, exist_ok=True)
    written: dict[str, str] = {}
    for filename, payload in _bank_payloads(bank).items():
        path = os.path.join(bank_dir, filename)
        with open(path, "wb") as handle:
            handle.write(payload)
        written[filename.split(".")[0]] = path
    meta_path = os.path.join(bank_dir, _F_META)
    meta = {
        "source_fraction": bank.source_fraction,
        "bank_hash": bank.bank_hash,
        "n_templates": len(bank.templates),
        "n_skeletons": len(bank.skeletons),
        "n_actions": len(bank.actions),
        "n_slots": len(bank.slot_registry.slots),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, sort_keys=True, indent=2)
        handle.write("\n")
    written["meta"] = meta_path
    return written


def load_bank(cfg: dict[str, Any]) -> Bank:
    """Load the bank written by :func:`write_bank`, rebuilding ``templates_by_act``.

    See :func:`reflex.contracts.load_bank` for the frozen contract.
    """
    bank_dir = resolve_path(cfg, "paths.bank_dir")
    paths = {name: os.path.join(bank_dir, name) for name in (_F_TEMPLATES, _F_SKELETONS, _F_ACTIONS, _F_REGISTRY)}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"bank not compiled: missing {missing}. Run `python -m reflex compile` first."
        )

    def _rows(path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    templates = [Template.from_dict(row) for row in _rows(paths[_F_TEMPLATES])]
    skeletons = [Skeleton.from_dict(row) for row in _rows(paths[_F_SKELETONS])]
    actions = [ActionPattern.from_dict(row) for row in _rows(paths[_F_ACTIONS])]
    with open(paths[_F_REGISTRY], "r", encoding="utf-8") as handle:
        registry_raw = json.load(handle)
    registry = SlotRegistry(
        slots={name: SlotSpec.from_dict(spec) for name, spec in registry_raw["slots"].items()},
        marker_to_slot=dict(registry_raw["marker_to_slot"]),
    )

    for template in templates:
        for slot in template.slots:
            if slot not in registry.slots:
                raise ContractViolation(
                    f"template {template.template_id} uses slot {slot!r}, which is not in the "
                    f"slot registry (spec 14 acceptance item 5)"
                )
    for pattern in actions:
        for slot in pattern.required_slots:
            if slot not in registry.slots:
                raise ContractViolation(
                    f"action {pattern.action} requires slot {slot!r}, which is not in the "
                    f"slot registry (spec 14 acceptance item 5)"
                )

    templates_by_act: dict[str, list[str]] = defaultdict(list)
    for template in templates:
        templates_by_act[template.act].append(template.template_id)

    source_fraction = 1.0
    bank_hash = ""
    meta_path = os.path.join(bank_dir, _F_META)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        source_fraction = float(meta.get("source_fraction", 1.0))
        bank_hash = str(meta.get("bank_hash", ""))
    return Bank(
        templates=templates,
        skeletons=skeletons,
        actions=actions,
        slot_registry=registry,
        templates_by_act={act: templates_by_act[act] for act in sorted(templates_by_act)},
        source_fraction=source_fraction,
        bank_hash=bank_hash,
    )


def write_turn_labels(labels: Sequence[TurnLabel], split: str, cfg: dict[str, Any]) -> str:
    """Persist per-turn labels to ``labels/<split>.jsonl`` (spec 6.2 step 7).

    See :func:`reflex.contracts.write_turn_labels` for the frozen contract.
    """
    labels_dir = resolve_path(cfg, "paths.labels_dir")
    os.makedirs(labels_dir, exist_ok=True)
    path = os.path.join(labels_dir, f"{split}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return path


def load_turn_labels(split: str, cfg: dict[str, Any]) -> list[TurnLabel]:
    """Load ``labels/<split>.jsonl``.

    See :func:`reflex.contracts.load_turn_labels` for the frozen contract.
    """
    path = os.path.join(resolve_path(cfg, "paths.labels_dir"), f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"turn labels for split {split!r} not found: {path}. Run `python -m reflex compile`."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return [TurnLabel.from_dict(json.loads(line)) for line in handle if line.strip()]


# --------------------------------------------------------------------------- #
# Review artifacts (spec 6.2 steps 2c and 3b)
# --------------------------------------------------------------------------- #


def _compile_dir(cfg: dict[str, Any]) -> str:
    path = resolve_path(cfg, "paths.compile_dir")
    os.makedirs(path, exist_ok=True)
    return path


def _original_texts(bank: Bank, cfg: dict[str, Any]) -> dict[str, str]:
    """``turn_id -> raw agent text`` for every example id in the bank, if readable.

    A masked sentence cannot be reviewed without the sentence it came from, so
    the check file shows both. If the corpus cannot be read the layout degrades
    to bank-only rather than failing the compile.
    """
    wanted: set[str] = {tid for t in bank.templates for tid in t.example_context_ids}
    if not wanted:
        return {}
    try:
        raw = _data.load_raw_abcd(cfg)
    except Exception:  # pragma: no cover - environment dependent
        return {}
    turn_key = str(get_dotted(cfg, "data.turn_list_key"))
    by_convo = {int(convo["convo_id"]): convo for convo in raw.get("train", [])}
    out: dict[str, str] = {}
    for tid in wanted:
        try:
            _split, convo_id, turn_index = tid.split(":")
            convo = by_convo.get(int(convo_id))
            if convo is None:
                continue
            turns = convo[turn_key]
            index = int(turn_index)
            if 0 <= index < len(turns):
                out[tid] = str(turns[index].get("text", ""))
        except (ValueError, KeyError):  # pragma: no cover - malformed id
            continue
    return out


def _delex_defects(template: Template, registry: SlotRegistry) -> list[str]:
    """Machine-checkable defects of one delexicalized template."""
    defects: list[str] = []
    text = template.text_delex
    if "<" in text or ">" in text:
        defects.append("angle_bracket_survived")
    if text.count("{") != text.count("}"):
        defects.append("unbalanced_braces")
    for slot in template.slots:
        if slot not in registry.slots:
            defects.append(f"unregistered_slot:{slot}")
    for marker_word in _ABCD_SLOT_TYPES:
        if re.search(rf"(?<![a-z_]){marker_word}(?![a-z_])", text) and marker_word not in template.slots:
            # A bare marker word such as "amount" is how a GLUED marker survives.
            if re.search(rf"\d{marker_word}|{marker_word}\d", text):
                defects.append(f"glued_marker_residue:{marker_word}")
    return defects


def write_delex_check(bank: Bank, cfg: dict[str, Any]) -> str:
    """Write the delexicalization quality sample to ``outputs/compile/delex_check.md``.

    See :func:`reflex.contracts.write_delex_check` for the frozen contract. The
    automated tally below is exactly that -- automated. It counts defects a
    machine can see (a surviving ``<``, an unbalanced brace, an unregistered
    slot, a glued-marker residue) and it is NOT the spec's >= 98% correct-masking
    figure, which only a human reading the pairs can produce. The blank tally is
    for that human.
    """
    sample_size = int(get_dotted(cfg, "compile.delex_check_sample"))
    seed = int(get_dotted(cfg, "compile.sample_seed"))
    registry = bank.slot_registry
    rng = random.Random(seed)

    population = list(bank.templates)
    weights = [max(t.count, 1) for t in population]
    sampled: list[Template] = []
    if population:
        indices = rng.choices(range(len(population)), weights=weights, k=min(sample_size, len(population)))
        seen: set[int] = set()
        for index in indices:
            if index not in seen:
                seen.add(index)
                sampled.append(population[index])
    slotted = [t for t in bank.templates if t.slots]
    slot_sample = slotted if len(slotted) <= sample_size else rng.sample(slotted, sample_size)

    bank_defects: Counter[str] = Counter()
    clean_templates = 0
    for template in bank.templates:
        defects = _delex_defects(template, registry)
        if defects:
            for defect in defects:
                bank_defects[defect.split(":")[0]] += 1
        else:
            clean_templates += 1
    sample_defects = {t.template_id: _delex_defects(t, registry) for t in sampled}
    clean_sample = sum(1 for d in sample_defects.values() if not d)

    originals = _original_texts(bank, cfg)
    total = len(bank.templates) or 1
    lines: list[str] = [
        "# Delexicalization check (spec 6.2 step 2c)",
        "",
        f"Sample size: {len(sampled)} templates, drawn with `random.Random({seed})` and",
        "weighted by train frequency, so the sample reads like the corpus rather than",
        "like the tail. A second section lists slot-bearing templates, which are where",
        "masking can actually be wrong.",
        "",
        "## What the machine can check (AUTOMATED, NOT THE SPEC'S 98%)",
        "",
        "The spec's acceptance bar is >= 98% CORRECT masking, and correctness here means",
        "\"the right span was masked with the right slot\" -- only a human comparing the",
        "original and the masked text can say that. What follows is the subset a machine",
        "can decide: a surviving `<`, an unbalanced brace, a slot outside the registry, or",
        "a glued-marker residue. **Do not quote it as the manual score.**",
        "",
        f"- templates in bank: **{len(bank.templates)}**",
        f"- templates with zero machine-visible defects: **{clean_templates}** "
        f"(**{100.0 * clean_templates / total:.2f}%**)",
        f"- sampled templates with zero machine-visible defects: **{clean_sample}/{len(sampled)}**",
        "",
        "| defect | templates |",
        "| --- | ---: |",
    ]
    if bank_defects:
        for defect, count in bank_defects.most_common():
            lines.append(f"| {defect} | {count} |")
    else:
        lines.append("| (none) | 0 |")
    lines += [
        "",
        "## Manual tally -- fill this in by reading the pairs below",
        "",
        "| verdict | count |",
        "| --- | ---: |",
        "| correctly masked | |",
        "| missed a value that should have been masked | |",
        "| masked something that should have stayed literal | |",
        "| right span, wrong slot name | |",
        f"| **total reviewed** | {len(sampled)} |",
        "",
        "correct masking % = correctly masked / total reviewed",
        "",
        "## Sample",
        "",
        "`original` is the raw ABCD agent turn the template's first example came from, so",
        "it may contain several sentences; compare only the relevant one.",
        "",
    ]
    for i, template in enumerate(sampled, start=1):
        example = template.example_context_ids[0] if template.example_context_ids else ""
        original = originals.get(example, "(not available)")
        defects = sample_defects.get(template.template_id) or []
        lines += [
            f"### {i}. {template.template_id} ({template.act}, count={template.count})",
            "",
            f"- delexicalized: `{template.text_delex}`",
            f"- slots: `{template.slots}`",
            f"- original turn ({example}): `{original}`",
            f"- machine defects: `{defects}`",
            "",
        ]
    lines += [
        "## Every slot-bearing template (the masking-sensitive subset)",
        "",
        f"{len(slotted)} of {len(bank.templates)} templates carry at least one slot.",
        "",
        "| template | act | count | slots | text |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for template in sorted(slot_sample, key=lambda t: (-t.count, t.template_id)):
        text = template.text_delex.replace("|", "\\|")
        lines.append(
            f"| {template.template_id} | {template.act} | {template.count} | "
            f"{' '.join(template.slots)} | {text} |"
        )
    lines.append("")

    path = os.path.join(_compile_dir(cfg), "delex_check.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def write_act_check(bank: Bank, cfg: dict[str, Any]) -> str:
    """Write the act-labeling agreement sample to ``outputs/compile/act_check.md``.

    See :func:`reflex.contracts.write_act_check` for the frozen contract. The
    numbers in it are SELF-ASSESSED -- the labeler grading its own rules against
    its own embeddings. That is a consistency measure, not accuracy; only the
    blank tally at the top, filled in by a person, is agreement.
    """
    sample_size = int(get_dotted(cfg, "compile.act_check_sample"))
    seed = int(get_dotted(cfg, "compile.sample_seed"))
    rng = random.Random(seed)
    population = list(bank.templates)
    weights = [max(t.count, 1) for t in population]
    sampled: list[Template] = []
    if population:
        indices = rng.choices(range(len(population)), weights=weights, k=min(sample_size, len(population)))
        seen: set[int] = set()
        for index in indices:
            if index not in seen:
                seen.add(index)
                sampled.append(population[index])

    labeler = get_act_labeler(cfg)
    provenance: list[tuple[str, str]]
    if isinstance(labeler, _RulesPlusEmbedLabeler):
        provenance = labeler.label_with_provenance([t.text_delex for t in sampled])
        centroid_acts, _ = labeler._nearest_centroid([t.text_delex for t in sampled])  # noqa: SLF001
    else:  # pragma: no cover - the llm labeler is off
        provenance = [(a, "labeler") for a in labeler.label([t.text_delex for t in sampled])]
        centroid_acts = [a for a, _ in provenance]

    rule_rows = [i for i, (_, how) in enumerate(provenance) if how.startswith("rule")]
    agree = sum(1 for i in rule_rows if centroid_acts[i] == provenance[i][0])
    by_act = Counter(t.act for t in bank.templates)
    weighted = Counter()
    for template in bank.templates:
        weighted[template.act] += template.count
    total_weighted = sum(weighted.values()) or 1

    lines: list[str] = [
        "# Act-labeling check (spec 6.2 step 3b)",
        "",
        "**SELF-ASSESSED. NOT HUMAN-VERIFIED.** Every number below was produced by the",
        "labeler being checked (`compile.act_labeler: rules_plus_embed`), so it measures",
        "internal consistency, not correctness. The agreement figure the spec asks for",
        "requires a person to fill in the tally.",
        "",
        "## Act conventions used by the rules",
        "",
        "Definitions are the ones in `prompts/act_labeling.txt`. Two conventions are",
        "arbitrary but applied consistently, because a skeleton is an act TUPLE and an",
        "inconsistent convention is worse than a debatable one:",
        "",
        '- "is there anything else I can help you with?" is **OFFER** (a remedy that can',
        '  be declined); "how can I help you?" is **ASK** (it requests information).',
        '- A bare greeting is **ACK**; a greeting welded to a question',
        '  ("hello, how can I help you today?") is scored on the question.',
        "",
        "## Manual tally -- fill this in",
        "",
        "| verdict | count |",
        "| --- | ---: |",
        "| label agrees with mine | |",
        "| label is wrong | |",
        "| sentence is genuinely ambiguous | |",
        f"| **total reviewed** | {len(sampled)} |",
        "",
        "## Bank act distribution",
        "",
        "| act | templates | share of train sentences |",
        "| --- | ---: | ---: |",
    ]
    for act in ACT_INVENTORY:
        lines.append(
            f"| {act} | {by_act.get(act, 0)} | {100.0 * weighted.get(act, 0) / total_weighted:.1f}% |"
        )
    lines += [
        "",
        "## Self-consistency (rules vs. embeddings)",
        "",
        f"- sampled sentences: **{len(sampled)}**",
        f"- decided by a cue rule: **{len(rule_rows)}** "
        f"({100.0 * len(rule_rows) / max(len(sampled), 1):.1f}%)",
        f"- decided by nearest centroid: **{len(sampled) - len(rule_rows)}**",
        f"- rule-labeled sentences whose nearest centroid agrees: **{agree}/{len(rule_rows)}** "
        f"({100.0 * agree / max(len(rule_rows), 1):.1f}%)",
        "",
        "A disagreement is not automatically an error: the rules exist precisely because",
        "the centroids are wrong on the easy cases.",
        "",
        "## Sample",
        "",
        "| # | act | decided by | nearest centroid | count | sentence |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for i, (template, (act, how)) in enumerate(zip(sampled, provenance), start=1):
        text = template.text_delex.replace("|", "\\|")
        lines.append(
            f"| {i} | {act} | {how} | {centroid_acts[i - 1]} | {template.count} | {text} |"
        )
    lines.append("")

    path = os.path.join(_compile_dir(cfg), "act_check.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _write_compile_stats(bank: Bank, cfg: dict[str, Any], stats: dict[str, Any]) -> str:
    """Write the compile provenance sidecar (bank sizes, coverage, top skeletons).

    TEMPLATE COVERAGE is the number to read first: it is the share of train agent
    utterances every sentence of which survives into the bank, and it is the hard
    ceiling on the reflex rate -- the fast path cannot answer a turn the bank
    cannot say.
    """
    coverage_turns = stats["template_coverage_turns"]
    coverage_sentences = stats["template_coverage_sentences"]
    lines = [
        "# Compile summary (spec 6.2)",
        "",
        f"- bank hash: `{bank.bank_hash}`  |  source fraction: {bank.source_fraction}",
        f"- dataset hash: `{stats['dataset_hash']}`",
        f"- NOVEL subflows held out of train: {stats['novel_subflows']}",
        "",
        "## Sizes",
        "",
        "| item | count |",
        "| --- | ---: |",
        f"| train conversations compiled | {stats['train_conversations']} |",
        f"| agent-side turns | {stats['agent_side_turns']} |",
        f"| retrieve_utterance turns | {stats['retrieve_utterance_turns']} |",
        f"| take_action turns | {stats['take_action_turns']} |",
        f"| sentences | {stats['sentences']} |",
        f"| templates before dedup | {stats['raw_templates']} |",
        f"| templates in bank | {len(bank.templates)} |",
        f"| skeletons | {len(bank.skeletons)} |",
        f"| action patterns | {len(bank.actions)} |",
        f"| slot registry entries | {len(bank.slot_registry.slots)} |",
        "",
        "## Template coverage -- the ceiling on the reflex rate",
        "",
        f"- agent utterances fully reconstructible from the bank: **{100.0 * coverage_turns:.2f}%**",
        f"- sentences reconstructible: **{100.0 * coverage_sentences:.2f}%**",
        "",
        "A turn counts only if EVERY one of its sentences survived dedup and the",
        "`compile.min_template_count` filter. No gate, selector or encoder can lift the",
        "fast path above this number.",
        "",
        "## Top 20 skeletons",
        "",
        "| skeleton | acts | count | share |",
        "| --- | --- | ---: | ---: |",
    ]
    total = sum(s.count for s in bank.skeletons) or 1
    for skeleton in bank.skeletons[:20]:
        lines.append(
            f"| {skeleton.skeleton_id} | {' '.join(skeleton.acts)} | {skeleton.count} | "
            f"{100.0 * skeleton.count / total:.1f}% |"
        )
    lines += [
        "",
        "## Action patterns",
        "",
        "| action | count | required slots |",
        "| --- | ---: | --- |",
    ]
    for pattern in bank.actions:
        lines.append(
            f"| {pattern.action} | {pattern.count} | {' '.join(pattern.required_slots) or '(none)'} |"
        )
    lines.append("")
    path = os.path.join(_compile_dir(cfg), "compile_summary.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
