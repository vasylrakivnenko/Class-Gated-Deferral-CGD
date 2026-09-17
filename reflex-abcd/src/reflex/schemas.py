"""Frozen data schemas exchanged between REFLEXIVE modules.

Spec Section 5 is normative: **FIELD NAMES ARE THE CONTRACT.** Every name in the
spec-5 classes below is copied from the spec character-for-character. Do not
rename, do not add, do not reorder the spec fields. `to_dict()` emits exactly the
spec JSON object; `from_dict()` accepts exactly that object.

Two groups of classes live here:

1. **SPEC-5 SCHEMAS** — `NormalizedTurn`, `Template`, `Skeleton`, `ActionPattern`,
   `Decision`, `GateOutput`, `EvalRecord`. Normative. Changing these breaks the
   cross-module contract.

2. **REFLEX-ADDED SCHEMAS** — `ContextWindow`, `SlotSpec`, `SlotRegistry`, `Bank`,
   `TurnLabel`, `SelectorScores`, `Selection`, `Calibration`, `SlotSources`,
   `LLMDecision`, `Partitions`, `RunManifest`. The spec describes these objects in
   prose (Sections 6.1, 6.3, 6.5, 6.8, 6.7, 6.9, 3.4, 10) but gives them no JSON
   schema. They are frozen here anyway so that nine modules agree on them. Each is
   marked `REFLEX-ADDED` in its docstring.

All dataclasses are `frozen=True`. Note what that does and does not buy you:
frozen blocks *attribute rebinding* (``turn.speaker = "x"`` raises). It does NOT
deep-freeze contained ``list``/``dict`` values. By contract, **never mutate a
contained list or dict** — build a new object instead. This matters because spec
Section 10 requires "same context -> same Decision, always".

Lists (not tuples) are used for sequence fields so that ``to_dict`` round-trips
through ``json`` without conversion, matching the on-disk JSONL contract.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Optional, TypeVar

__all__ = [
    # Literal aliases
    "Speaker",
    "NextStep",
    "Act",
    "Arm",
    "Route",
    "GateReason",
    "SlotType",
    "SlotSource",
    # Normative constants
    "NEXT_STEPS",
    "ACT_INVENTORY",
    "GATE_REASON_PRECEDENCE",
    "SPEC5_SCHEMAS",
    # Spec-5 schemas
    "NormalizedTurn",
    "Template",
    "Skeleton",
    "ActionPattern",
    "Decision",
    "GateOutput",
    "EvalRecord",
    # Reflex-added schemas
    "ContextWindow",
    "SlotSpec",
    "SlotRegistry",
    "TurnLabel",
    "Bank",
    "SelectorScores",
    "Selection",
    "Calibration",
    "SlotSources",
    "LLMDecision",
    "Partitions",
    "RunManifest",
    # Helpers
    "DictSchema",
]

# ---------------------------------------------------------------------------
# Literal aliases and normative constants
# ---------------------------------------------------------------------------

Speaker = Literal["agent", "customer", "action"]
NextStep = Literal["retrieve_utterance", "take_action", "end_conversation"]
Act = Literal[
    "ACK", "VERIFY", "ASK", "INFORM", "INSTRUCT", "CONFIRM", "OFFER", "CLOSE", "OTHER"
]
Arm = Literal["A", "B"]
Route = Literal["reflex", "escalated"]
GateReason = Literal[
    "ok", "low_confidence", "novel", "unavailable_slot", "unseen_action"
]
SlotType = Literal[
    "id", "email", "phone", "money", "date", "name", "address", "count", "other"
]
SlotSource = Literal["scenario_field", "prior_action_value", "customer_utterance"]

#: Normative nextstep ordering. This is ``ontology.json["next_steps"]`` verbatim
#: AND the integer encoding that the official ``utils/evaluate.py::cds_report``
#: assumes: it branches on ``nextstep_label == 0 / 1 / 2`` meaning
#: retrieve_utterance / take_action / end_conversation. **Never re-order.**
#: Index into this list is the class id for head H1 everywhere in the codebase.
NEXT_STEPS: tuple[str, ...] = (
    "retrieve_utterance",
    "take_action",
    "end_conversation",
)

#: Spec 2 / 12 act inventory. Index into this tuple is the class id for head H6.
ACT_INVENTORY: tuple[str, ...] = (
    "ACK", "VERIFY", "ASK", "INFORM", "INSTRUCT", "CONFIRM", "OFFER", "CLOSE", "OTHER",
)

#: Spec 6.6: "Precedence for reporting the reason when several fail:
#: novel > unseen_action > unavailable_slot > low_confidence."
#: ``"ok"`` is last because it is only reported when nothing failed.
GATE_REASON_PRECEDENCE: tuple[str, ...] = (
    "novel",
    "unseen_action",
    "unavailable_slot",
    "low_confidence",
    "ok",
)


# ---------------------------------------------------------------------------
# Serialization mixin
# ---------------------------------------------------------------------------

T = TypeVar("T", bound="DictSchema")


class DictSchema:
    """Adds ``to_dict`` / ``from_dict`` to a frozen dataclass.

    ``to_dict`` returns a plain ``dict`` whose keys are the declared field names
    in declaration order, recursing into nested ``DictSchema`` values and into
    lists/dicts of them. ``from_dict`` is the inverse and is *strict about
    unknown keys* -- an unexpected key raises ``KeyError`` rather than being
    silently dropped, because a silently-dropped field is exactly how nine
    parallel agents end up with three different versions of a schema.

    Missing keys are only tolerated when the field has a default.
    """

    #: Set on subclasses that are normative spec-5 schemas.
    spec_section: ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict using the normative field names."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            out[f.name] = _encode(getattr(self, f.name))
        return out

    @classmethod
    def from_dict(cls: type[T], payload: dict[str, Any]) -> T:
        """Build an instance from ``payload``.

        Raises:
            KeyError: if ``payload`` has a key that is not a declared field, or
                omits a field that has no default.
            TypeError: if ``payload`` is not a mapping.
        """
        if not isinstance(payload, dict):
            raise TypeError(f"{cls.__name__}.from_dict expects a dict, got {type(payload).__name__}")
        fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(payload) - set(fields)
        if unknown:
            raise KeyError(f"{cls.__name__}.from_dict: unknown keys {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name, f in fields.items():
            if name in payload:
                kwargs[name] = _decode(f.type, payload[name])
            elif f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                continue
            else:
                raise KeyError(f"{cls.__name__}.from_dict: missing required key {name!r}")
        return cls(**kwargs)


def _encode(value: Any) -> Any:
    if isinstance(value, DictSchema):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def _decode(annotation: Any, value: Any) -> Any:
    """Best-effort reconstruction of nested ``DictSchema`` values.

    Type annotations are strings under ``from __future__ import annotations``, so
    this resolves the few nested-schema cases the contract actually uses by name
    rather than attempting general runtime type reconstruction.
    """
    if value is None:
        return None
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    nested = {
        "GateOutput": GateOutput,
        "Optional[GateOutput]": GateOutput,
        "GateOutput | None": GateOutput,
        "SlotSpec": SlotSpec,
        "Template": Template,
        "Skeleton": Skeleton,
        "ActionPattern": ActionPattern,
        "NormalizedTurn": NormalizedTurn,
        "ContextWindow": ContextWindow,
    }
    for key, klass in nested.items():
        if name == key and isinstance(value, dict):
            return klass.from_dict(value)
    # list[Template] / dict[str, SlotSpec] style annotations
    if isinstance(value, list) and isinstance(name, str):
        for key, klass in nested.items():
            if f"[{key}]" in name and value and isinstance(value[0], dict):
                return [klass.from_dict(v) for v in value]
    if isinstance(value, dict) and isinstance(name, str):
        for key, klass in nested.items():
            if f", {key}]" in name and value and all(isinstance(v, dict) for v in value.values()):
                return {k: klass.from_dict(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# 1. SPEC-5 SCHEMAS (normative)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedTurn(DictSchema):
    """Spec 5.1. One turn of an ABCD conversation after key normalization.

    Produced by ``reflex.data`` (module 4.1) from ``convo["delexed"][i]``.

    ``utt_id`` VS ``utt_rank`` -- READ THIS, THE SPEC CONFLATES THEM
    ---------------------------------------------------------------
    ABCD's raw ``turn["targets"]`` is ``[intent, nextstep, action, values, X]``.
    The official ``utils/process.py`` unpacks ``X`` into a variable it calls
    ``utt_id``, and spec 5.1 copies that name. **The value is not a global
    utterance id.** It is a RANK -- a 0-based index into this turn's own
    ``candidates`` list. Verified over all 95,129 agent turns in all three
    splits:

    * ``utterances[candidates[X]] == original[i][1].lower()`` on 95,127 / 95,129
      (99.998%); the 2 residuals are ABCD pipe-character artifacts in
      ``original`` (see README "Dataset notes").
    * ``utterances[X] == turn["text"]`` on 25 / 95,129 (0.03%) -- i.e. the global
      reading is simply wrong.

    So this schema splits the two concepts:

    * ``utt_rank`` -- the RAW ``targets[4]``. This is the value the official
      metrics want as ``utterance_label`` (``cds_report`` compares it against an
      argpartition over a 100-wide score row, so it must be a rank in
      ``[0, 100)``, or ``-1`` for non-retrieve turns).
    * ``utt_id`` -- the RESOLVED GLOBAL id, ``candidates[utt_rank]``. This is the
      index into ``utterances.json`` and is what you use to fetch gold text.

    Feed ``utt_rank`` to the official evaluator. Use ``utt_id`` for text lookup.
    Never swap them.

    Attributes:
        convo_id: ABCD ``convo_id``.
        turn_index: Position among ALL turns in the conversation (0-based index
            into ``convo["delexed"]``, which is 1:1 index-aligned with
            ``convo["original"]`` for all 10,042 conversations). This is NOT
            ABCD's ``turn_count``; see ``turn_count``.
        speaker: ``agent`` | ``customer`` | ``action``.
        text: Turn text. For ``speaker == "action"`` turns this is the action
            name. NOTE: this is the text from ``convo["delexed"]``, which is
            already partially delexicalized with ``<slot>`` markers (21,297
            markers over 220,983 turns). ``original[turn_index][1]`` holds the
            lexicalized, original-cased text.
        nextstep: Agent-side only; ``None`` for customer turns. In raw ABCD,
            speaker determines nextstep 1:1 (agent->retrieve_utterance,
            action->take_action, customer->None). ``end_conversation`` NEVER
            appears in raw ABCD and is synthesized per conversation; see
            ``is_synthetic_end``.
        intent: Gold subflow (``targets[0]``).
        action: Gold button name for ``take_action`` turns, else ``None``.
        values: Gold values for ``take_action`` turns, else ``None``. Raw
            ``targets[3]`` is always a list; observed lengths on action turns are
            0 (11,327), 1 (19,057) and 3 (6,098).
        utt_id: RESOLVED global utterance id ``candidates[utt_rank]``, or ``None``.
        candidates: The 100 utterance ids for ``retrieve_utterance`` turns, else
            ``None``. Exactly 100 on every agent turn; empty on every other turn.
        utt_rank: REFLEX-ADDED. Raw ``targets[4]``. ``-1`` / ``None`` off-task.
        turn_count: REFLEX-ADDED. Raw ABCD ``turn["turn_count"]``. Required by
            ``cds_report``'s ``ci_and_tc`` argument. Turn counts skip numbers, so
            this is not a dense index -- that is why ``turn_index`` exists too.
        is_synthetic_end: REFLEX-ADDED. ``True`` only for the one extra
            ``end_conversation`` turn appended per conversation to replicate
            ``utils/process.py``. Never ``True`` for a turn read from the file.
    """

    spec_section: ClassVar[str] = "5.1"

    convo_id: int
    turn_index: int
    speaker: Speaker
    text: str
    nextstep: Optional[NextStep] = None
    intent: Optional[str] = None
    action: Optional[str] = None
    values: Optional[list[str]] = None
    utt_id: Optional[int] = None
    candidates: Optional[list[int]] = None
    # --- REFLEX-ADDED beyond spec 5.1 ---
    utt_rank: Optional[int] = None
    turn_count: Optional[int] = None
    is_synthetic_end: bool = False


@dataclass(frozen=True)
class Template(DictSchema):
    """Spec 5.2. One delexicalized agent sentence with typed slots, tagged with one act.

    Attributes:
        template_id: ``"T000123"`` -- literal spec format: ``T`` + 6 zero-padded digits.
        act: One of :data:`ACT_INVENTORY`.
        text_delex: e.g. ``"Your order {order_id} will arrive on {arrival_date}."``
            NOTE the brace style: REFLEXIVE templates use ``{slot}``, while ABCD's
            own raw markers are ``<slot>``. The compiler converts ``<slot>`` ->
            ``{slot}``; ``<`` never survives into ``text_delex``.
        slots: Slot names appearing in ``text_delex``, in order of first
            appearance, e.g. ``["order_id", "arrival_date"]``. Every name MUST be
            a key of the :class:`SlotRegistry` (spec 14 acceptance item).
        count: Occurrences in train (after dedup merging, summed over merged forms).
        surface_forms: Merged near-duplicates. ``surface_forms[0]`` is the
            canonical (most frequent) form and equals ``text_delex``.
        example_context_ids: Up to ``compile.max_example_context_ids`` training
            turn ids, formatted as :func:`turn_key` strings (``"train:1234:12"``).
    """

    spec_section: ClassVar[str] = "5.2"

    template_id: str
    act: Act
    text_delex: str
    slots: list[str]
    count: int
    surface_forms: list[str] = field(default_factory=list)
    example_context_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Skeleton(DictSchema):
    """Spec 5.3. An ordered tuple of acts describing one agent utterance.

    Attributes:
        skeleton_id: ``"S0042"`` -- literal spec format: ``S`` + 4 zero-padded digits.
        acts: e.g. ``["ACK", "INFORM", "OFFER"]``. Order is significant.
        count: Occurrences in train. Spec 6.2 step 5: keep ALL observed
            skeletons, no minimum count.
    """

    spec_section: ClassVar[str] = "5.3"

    skeleton_id: str
    acts: list[str]
    count: int


@dataclass(frozen=True)
class ActionPattern(DictSchema):
    """Spec 5.4. One ABCD button plus the slots its values occupy.

    Attributes:
        action: Button name. Must be a member of the flattened action list from
            ``ontology["actions"]`` (a dict of 3 categories -- kb_query 6,
            interaction 10, faq_policy 14 -- i.e. 30 buttons, which matches the
            "size of 30 long" intent mask in the official ``cds_report``).
        required_slots: Slot registry names whose values must exist to take it.
            Derived from the slot types of the values entered in train.
        count: Occurrences in train.
    """

    spec_section: ClassVar[str] = "5.4"

    action: str
    required_slots: list[str]
    count: int


@dataclass(frozen=True)
class GateOutput(DictSchema):
    """Spec 5.6. The gate's routing verdict for one Arm B turn.

    Only :mod:`reflex.gate` constructs this (spec 4.6: the gate is the only place
    that decides fast-path vs escalate).

    Attributes:
        route: ``reflex`` | ``escalated``.
        reason: ``ok`` when routed to the fast path; otherwise the
            highest-precedence failing signal per :data:`GATE_REASON_PRECEDENCE`.
        set_sizes: Conformal prediction-set size per head. Keys are exactly
            ``"nextstep"``, ``"intent"``, ``"action"``, ``"skeleton"`` (ints) and
            ``"templates"`` (a list of ints, one per act position in the chosen
            skeleton). Heads that do not apply to this turn report ``0`` (and
            ``[]`` for templates), not a missing key.
        novelty_distance: ``1 - max cosine(c, c_train)``.
        missing_slots: Registry slot names with no available value. Empty when
            availability passed.
    """

    spec_section: ClassVar[str] = "5.6"

    route: Route
    reason: GateReason
    set_sizes: dict[str, Any]
    novelty_distance: float
    missing_slots: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Decision(DictSchema):
    """Spec 5.5. The system's output for one turn in one arm.

    Attributes:
        convo_id: ABCD conversation id.
        turn_index: Matches :attr:`NormalizedTurn.turn_index`.
        arm: ``"A"`` (LLM only) or ``"B"`` (fast path + escalation).
        route: ``reflex`` | ``escalated``. Spec 5.5: **Arm A is always
            ``"escalated"`` for accounting purposes.**
        nextstep: Predicted nextstep.
        intent: Predicted subflow.
        action: Predicted button, or ``None``.
        values: Predicted values, or ``None``.
        skeleton_id: Chosen skeleton, or ``None`` (always ``None`` for Arm A and
            for escalated Arm B turns).
        template_ids: One template id per act position, or ``None``.
        utterance_text: Filled text (Arm B reflex) or LLM text (Arm A / escalated).
        candidate_utt_id: Mapped candidate for the official metric (spec 6.5
            step 5). CAUTION: despite the name, the official evaluator needs a
            RANK in ``[0, 100)``, not a global id. Store the global id here (the
            spec's name) and the rank in :attr:`candidate_rank`; the evaluator
            reads ``candidate_rank``.
        gate: :class:`GateOutput`, or ``None`` for Arm A.
        llm_tokens_in: Prompt tokens billed. ``0`` on a reflex turn.
        llm_tokens_out: Completion tokens billed. ``0`` on a reflex turn.
        latency_ms_fastpath: Fast-path wall clock. ``0.0`` for Arm A.
        latency_ms_llm: LLM wall clock. ``0.0`` on a reflex turn.
        candidate_rank: REFLEX-ADDED. Rank of ``candidate_utt_id`` within this
            turn's ``candidates``, or ``-1``. This is what goes to the official
            evaluator as the predicted utterance position.
        exact_template_match: REFLEX-ADDED. Spec 6.5 step 5 requires recording
            "whether the gold utterance's delexicalized form equals the composed
            text". ``None`` when not applicable.
        cache_hit: REFLEX-ADDED. ``True`` if the LLM response came from the
            spec 6.9 step 5 cache (so it was not re-billed).
    """

    spec_section: ClassVar[str] = "5.5"

    convo_id: int
    turn_index: int
    arm: Arm
    route: Route
    nextstep: str
    intent: str
    action: Optional[str] = None
    values: Optional[list[str]] = None
    skeleton_id: Optional[str] = None
    template_ids: Optional[list[str]] = None
    utterance_text: Optional[str] = None
    candidate_utt_id: Optional[int] = None
    gate: Optional[GateOutput] = None
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    latency_ms_fastpath: float = 0.0
    latency_ms_llm: float = 0.0
    # --- REFLEX-ADDED beyond spec 5.5 ---
    candidate_rank: int = -1
    exact_template_match: Optional[bool] = None
    cache_hit: bool = False


@dataclass(frozen=True)
class EvalRecord(DictSchema):
    """Spec 5.7. One row of ``decisions.jsonl``: a Decision plus gold and correctness.

    Spec 5.7 reads "Decision fields + {gold, correct, seed, run_id}". Rather than
    duplicating 17 field names, this schema nests the Decision under
    ``decision``; :meth:`to_dict` FLATTENS it so the on-disk JSONL is literally
    "Decision fields + gold + correct + seed + run_id", exactly as specified.
    :meth:`from_dict` accepts that flat form. Never hand-roll this flattening
    elsewhere -- spec 10 says nothing is computed from logs outside the Evaluator,
    and this is the only reader/writer contract for the log.

    Attributes:
        decision: The :class:`Decision` being scored.
        gold: ``{"nextstep", "intent", "action", "values", "utt_id"}`` -- the
            spec's five keys, verbatim. Add ``"utt_rank"`` alongside if you need
            the rank; the evaluator prefers it when present.
        correct: ``{"nextstep", "intent", "action", "values", "utterance"}`` ->
            bool. Note ``"utterance"``, not ``"utt_id"``, per spec.
        seed: Training seed that produced the decision.
        run_id: The run that produced it.
    """

    spec_section: ClassVar[str] = "5.7"

    decision: Decision
    gold: dict[str, Any]
    correct: dict[str, bool]
    seed: int
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Flatten to the spec 5.7 on-disk shape: Decision fields + 4 extras."""
        out = self.decision.to_dict()
        overlap = {"gold", "correct", "seed", "run_id"} & set(out)
        if overlap:  # pragma: no cover - guards against a future Decision rename
            raise ValueError(f"EvalRecord flattening collides on {sorted(overlap)}")
        out["gold"] = dict(self.gold)
        out["correct"] = dict(self.correct)
        out["seed"] = self.seed
        out["run_id"] = self.run_id
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalRecord":
        """Inverse of :meth:`to_dict`; accepts the flat on-disk shape."""
        payload = dict(payload)
        gold = payload.pop("gold")
        correct = payload.pop("correct")
        seed = payload.pop("seed")
        run_id = payload.pop("run_id")
        return cls(
            decision=Decision.from_dict(payload),
            gold=gold,
            correct=correct,
            seed=seed,
            run_id=run_id,
        )


#: The normative spec-5 schemas, for the contract-parity test in tests/.
SPEC5_SCHEMAS: tuple[type[DictSchema], ...] = (
    NormalizedTurn,
    Template,
    Skeleton,
    ActionPattern,
    Decision,
    GateOutput,
    EvalRecord,
)


# ---------------------------------------------------------------------------
# 2. REFLEX-ADDED SCHEMAS (spec describes these in prose only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextWindow(DictSchema):
    """REFLEX-ADDED (spec 6.1 step 4 prose). The model's view of one agent turn.

    Spec 6.1 step 4: "Context window for any agent turn = the previous K turns
    (config ``data.context_turns_K``) with speaker tags, plus a compact 'state'
    string: scenario fields that the customer has ALREADY DISCLOSED in the
    conversation (not the full scenario -- the agent cannot know undisclosed
    facts), plus actions taken so far with their values."

    The undisclosed-facts rule is a leakage guard, not a nicety. Passing the full
    ``convo["scenario"]`` would hand the model the answer.

    Attributes:
        convo_id: Conversation id.
        turn_index: The agent turn this context precedes.
        text: The single rendered string fed to the encoder AND to the LLM agent
            (spec 6.9 step 1: "the same context string the encoder sees"). This
            is the ONLY field either consumer may read for model input.
        turns: The previous K turns, oldest first, as ``"speaker|text"`` strings
            (matching ``utils/process.py``'s ``f"{speaker}|{text}"``).
        disclosed: Scenario fields the customer has disclosed so far, flat
            ``field -> value``. Never includes undisclosed scenario values.
        actions_so_far: ``[{"action": str, "values": [str]}]`` in order taken.
        context_hash: Stable sha256 hex of ``text``, truncated to 16 chars. Used
            as the LLM cache key component (spec 6.9 step 5) and to assert
            determinism (spec 10).
    """

    convo_id: int
    turn_index: int
    text: str
    turns: list[str] = field(default_factory=list)
    disclosed: dict[str, str] = field(default_factory=dict)
    actions_so_far: list[dict[str, Any]] = field(default_factory=list)
    context_hash: str = ""


@dataclass(frozen=True)
class SlotSpec(DictSchema):
    """REFLEX-ADDED (spec 6.3 prose). One entry of the slot registry.

    Spec 6.3: "slot_name -> {type, source, format}". Slot names must be SPECIFIC
    (``order_id``, ``refund_amount``, ``arrival_date``); generic names like
    ``date`` or ``value`` are a spec 14 acceptance failure.

    Attributes:
        name: The slot name, which is also its registry key.
        type: One of :data:`SlotType`.
        source: Where the filler looks first; one of :data:`SlotSource`.
        format: Format string applied by :mod:`reflex.fill`, or ``""`` for
            verbatim substitution.
        abcd_marker: REFLEX-ADDED. The ABCD ``<slot>`` marker this slot is the
            registry name for (e.g. ``"<order_id>"``), or ``""`` for slots
            REFLEXIVE adds for entities ABCD leaves literal. ABCD's own marker
            vocabulary is exactly ``ontology["values"]["non_enumerable"]``
            flattened, and it is the registry BACKBONE -- do not invent a
            parallel name for anything ABCD already marks.
    """

    name: str
    type: SlotType
    source: SlotSource
    format: str = ""
    abcd_marker: str = ""


@dataclass(frozen=True)
class SlotRegistry(DictSchema):
    """REFLEX-ADDED (spec 6.3 prose). The single source of slot names.

    Persisted as ``bank/slot_registry.json``. Spec 6.3: "Same entity => same slot
    name in every template. Enforced by the registry being the only source of
    slot names in the compiler."

    Attributes:
        slots: ``slot_name -> SlotSpec``. The key must equal ``SlotSpec.name``.
        marker_to_slot: ``"<order_id>" -> "order_id"``. Derived, but persisted so
            readers do not have to re-derive it.
    """

    slots: dict[str, SlotSpec] = field(default_factory=dict)
    marker_to_slot: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnLabel(DictSchema):
    """REFLEX-ADDED (spec 6.2 step 7 prose). One row of ``labels/train.jsonl``.

    Spec 6.2 step 7 names these fields exactly: "{turn_id, skeleton_id,
    template_ids (one per act), action, values, nextstep, intent}".

    Attributes:
        turn_id: ``"{split}:{convo_id}:{turn_index}"``; see :func:`turn_key` in
            :mod:`reflex.contracts`.
        skeleton_id: Chosen skeleton for retrieve_utterance turns, else ``None``.
        template_ids: One template id per act position, else ``None``.
        action: Gold button for take_action turns, else ``None``.
        values: Gold values for take_action turns, else ``None``.
        nextstep: Gold nextstep.
        intent: Gold subflow.
        slot_values: REFLEX-ADDED. The compiler's ``surface value -> slot name``
            mapping for this turn (spec 6.2 step 2b: "Keep a mapping from surface
            value -> slot name per turn for slot filling later"). Stored
            inverted, ``slot_name -> surface value``, because that is the
            direction :mod:`reflex.fill` reads.
    """

    turn_id: str
    nextstep: str
    intent: str
    skeleton_id: Optional[str] = None
    template_ids: Optional[list[str]] = None
    action: Optional[str] = None
    values: Optional[list[str]] = None
    slot_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Bank(DictSchema):
    """REFLEX-ADDED (spec 2 "response bank"). The compiler's whole output.

    Persisted across four files (spec 6.2 step 7): ``bank/templates.jsonl``,
    ``bank/skeletons.jsonl``, ``bank/actions.jsonl``, ``bank/slot_registry.json``.
    This class is the in-memory aggregate the selector consumes.

    Attributes:
        templates: All templates, ordered by ``template_id``.
        skeletons: All observed skeletons, ordered by ``skeleton_id``.
        actions: One :class:`ActionPattern` per observed button.
        slot_registry: The registry.
        templates_by_act: ``act -> [template_id]``. Derived index; the selector
            ranks only within the act of each position (spec 6.4 H7).
        source_fraction: Which learning-curve fraction of train this was compiled
            from (spec 3.4 / E4). ``1.0`` for the headline bank.
        bank_hash: sha256 hex (16 chars) over the persisted bank files, recorded
            in the run manifest.
    """

    templates: list[Template] = field(default_factory=list)
    skeletons: list[Skeleton] = field(default_factory=list)
    actions: list[ActionPattern] = field(default_factory=list)
    slot_registry: SlotRegistry = field(default_factory=SlotRegistry)
    templates_by_act: dict[str, list[str]] = field(default_factory=dict)
    source_fraction: float = 1.0
    bank_hash: str = ""


@dataclass(frozen=True)
class SelectorScores(DictSchema):
    """REFLEX-ADDED (spec 6.5 prose). Raw scores for one turn. No decisions.

    Produced by :mod:`reflex.select` (4.5: "inference producing scores. No
    decision-making"). Consumed by :mod:`reflex.gate`.

    Every probability vector is a plain ``list[float]`` that SUMS TO 1 over the
    head's full class list, indexed by the canonical class order for that head
    (:data:`NEXT_STEPS` for H1; ``data.subflow_list(ontology)`` for H2;
    ``data.action_list(ontology)`` for H3; ``[s.skeleton_id for s in
    bank.skeletons]`` for H5). Do not pass logits.

    Attributes:
        convo_id: Conversation id.
        turn_index: Turn index.
        nextstep_probs: H1, length 3, ordered by :data:`NEXT_STEPS`.
        intent_probs: H2, length 55.
        action_probs: H3, length 30. Empty list when the turn is not take_action.
        skeleton_probs: H5, length ``len(bank.skeletons)``. Empty when not
            retrieve_utterance.
        value_probs: H4, one probability vector per required slot of the
            predicted action, aligned with ``value_candidates``.
        value_candidates: One candidate value list per required slot.
        template_probs: H7, one probability vector per act position of the chosen
            skeleton, each over ``template_candidates`` at the same position.
        template_candidates: One ``[template_id]`` list per act position.
        novelty_distance: ``1 - max cosine(c, c_train)`` from the calibrator's
            FAISS index. OWNERSHIP NOTE: the distance is a SIGNAL, computed here;
            :mod:`reflex.gate` only compares it to the threshold. The index
            itself is built by :mod:`reflex.calibrate`.
        context_hash: Copied from :attr:`ContextWindow.context_hash`, so a
            determinism test can assert score identity per context.
    """

    convo_id: int
    turn_index: int
    nextstep_probs: list[float] = field(default_factory=list)
    intent_probs: list[float] = field(default_factory=list)
    action_probs: list[float] = field(default_factory=list)
    skeleton_probs: list[float] = field(default_factory=list)
    value_probs: list[list[float]] = field(default_factory=list)
    value_candidates: list[list[str]] = field(default_factory=list)
    template_probs: list[list[float]] = field(default_factory=list)
    template_candidates: list[list[str]] = field(default_factory=list)
    novelty_distance: float = 0.0
    context_hash: str = ""


@dataclass(frozen=True)
class Selection(DictSchema):
    """REFLEX-ADDED (spec 6.5 prose). The argmax reading of :class:`SelectorScores`.

    Still not a routing decision -- this is "what the fast path WOULD say".
    :mod:`reflex.gate` decides whether it gets to say it.

    Attributes:
        nextstep: H1 argmax.
        intent: H2 argmax (subflow name).
        action: H3 argmax button name, or ``None`` when nextstep is not take_action.
        values: H4 argmax value per required slot, or ``None``.
        skeleton_id: H5 argmax, or ``None`` when nextstep is not retrieve_utterance.
        template_ids: H7 top-1 per act position, or ``None``.
        composed_text_delex: The chosen templates' ``text_delex`` joined in act
            order -- still slotted, not yet filled. Input to spec 6.5 step 5
            candidate mapping and to :mod:`reflex.fill`.
        candidate_utt_id: Global utterance id chosen from the 100 candidates.
        candidate_rank: Its rank within ``candidates``, or ``-1``.
        exact_template_match: Whether the gold utterance's delexicalized form
            equals ``composed_text_delex``. ``None`` when not applicable.
    """

    nextstep: str
    intent: str
    action: Optional[str] = None
    values: Optional[list[str]] = None
    skeleton_id: Optional[str] = None
    template_ids: Optional[list[str]] = None
    composed_text_delex: Optional[str] = None
    candidate_utt_id: Optional[int] = None
    candidate_rank: int = -1
    exact_template_match: Optional[bool] = None


@dataclass(frozen=True)
class Calibration(DictSchema):
    """REFLEX-ADDED (spec 6.8 prose). Persisted as ``outputs/calibration/gate.json``.

    Attributes:
        alpha: The alpha these quantiles were computed at.
        quantiles: ``head -> q_h``. Keys are exactly ``"nextstep"``, ``"intent"``,
            ``"action"``, ``"skeleton"``, ``"template"`` (one shared q for H7
            across act positions). Spec 6.6: prediction set =
            ``{classes with softmax >= 1 - q_h}``.
        novelty_threshold: The ``gate.novelty_percentile`` percentile of
            ``novelty_distance`` over dev turns.
        alpha_sweep_quantiles: ``str(alpha) -> {head -> q_h}`` for every alpha in
            ``gate.alpha_sweep`` (spec 6.8 step 3, the coverage-error curve
            inputs). Keys are stringified floats because JSON keys are strings.
        n_dev_turns: ``head -> n`` used per head, for the finite-sample quantile.
        novelty_index_path: Where the FAISS index of train context vectors lives.
        checkpoint_path: The checkpoint these quantiles belong to. Quantiles are
            NOT transferable across checkpoints or seeds.
        seed: The training seed of that checkpoint.
    """

    alpha: float
    quantiles: dict[str, float] = field(default_factory=dict)
    novelty_threshold: float = 0.0
    alpha_sweep_quantiles: dict[str, dict[str, float]] = field(default_factory=dict)
    n_dev_turns: dict[str, int] = field(default_factory=dict)
    novelty_index_path: str = ""
    checkpoint_path: str = ""
    seed: int = 0


@dataclass(frozen=True)
class SlotSources(DictSchema):
    """REFLEX-ADDED (spec 6.7 prose). Values available to the filler, by origin.

    Spec 6.7 priority order is STRICT and is the field order here: customer
    utterance, then prior action values, then disclosed scenario fields. "A slot
    with no available value is a gate failure, never a guess."

    Attributes:
        from_customer: ``slot_name -> value``, from the delex mapping of customer
            turns.
        from_actions: ``slot_name -> value``, from values entered in prior actions.
        from_scenario: ``slot_name -> value``, from scenario fields the customer
            has already disclosed.
    """

    from_customer: dict[str, str] = field(default_factory=dict)
    from_actions: dict[str, str] = field(default_factory=dict)
    from_scenario: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMDecision(DictSchema):
    """REFLEX-ADDED (spec 6.9 prose). One parsed LLM response.

    Spec 6.9 step 2: the LLM must output JSON
    ``{nextstep, intent, action, values, candidate_index}``. ``candidate_index``
    is a RANK into the 100 numbered candidates shown in the prompt, matching the
    rank semantics of :attr:`NormalizedTurn.utt_rank`.

    Attributes:
        nextstep: Parsed nextstep.
        intent: Parsed subflow.
        action: Parsed button, or ``None``.
        values: Parsed values, or ``None``.
        candidate_index: Parsed rank in ``[0, 100)``, or ``-1``.
        utterance_text: The candidate text at ``candidate_index``, for logging.
        tokens_in: Billed prompt tokens.
        tokens_out: Billed completion tokens.
        latency_ms: Wall clock for the call. ``0.0`` on a cache hit.
        parse_failed: ``True`` if both the first attempt and the single retry
            failed to parse. Spec 6.9 step 2: the turn is then counted as
            INCORRECT and logged -- it is not retried again and not dropped.
        cache_hit: ``True`` if served from the spec 6.9 step 5 cache.
        model_id: The resolved model id actually called.
        prompt_hash: sha256 hex (16 chars) of the frozen prompt file.
    """

    nextstep: str
    intent: str
    action: Optional[str] = None
    values: Optional[list[str]] = None
    candidate_index: int = -1
    utterance_text: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    parse_failed: bool = False
    cache_hit: bool = False
    model_id: str = ""
    prompt_hash: str = ""


@dataclass(frozen=True)
class Partitions(DictSchema):
    """REFLEX-ADDED (spec 3.4 prose). The five partitions every run refers to by name.

    Spec 3.4: official train/dev/test for headline numbers; NOVEL = 5 subflows
    removed from train and dev entirely, kept as ``test_novel``; the rest of test
    is ``test_seen``.

    ``train`` and ``dev`` here are ALREADY novel-filtered. There is no unfiltered
    accessor on purpose: forgetting to filter is the one mistake that silently
    invalidates spec 8.4.

    Attributes:
        train: convo_id -> turns, novel subflows removed.
        dev: convo_id -> turns, novel subflows removed.
        test_seen: test conversations whose subflow is not novel.
        test_novel: test conversations whose subflow IS novel.
        novel_subflows: The 5 chosen subflow names. Written to the manifest.
        dataset_hash: sha256 hex (16 chars) of ``abcd_v1.1.json``.
    """

    train: dict[int, list[NormalizedTurn]] = field(default_factory=dict)
    dev: dict[int, list[NormalizedTurn]] = field(default_factory=dict)
    test_seen: dict[int, list[NormalizedTurn]] = field(default_factory=dict)
    test_novel: dict[int, list[NormalizedTurn]] = field(default_factory=dict)
    novel_subflows: list[str] = field(default_factory=list)
    dataset_hash: str = ""


@dataclass(frozen=True)
class RunManifest(DictSchema):
    """REFLEX-ADDED (spec 10 prose). ``outputs/runs/<run_id>/manifest.json``.

    Spec 10: "every run has a run_id; config, prompt hashes, dataset hash, git
    commit, seeds and model ids are written to manifest.json" and "document the
    actual machine".

    Attributes:
        run_id: The run identifier (also the output directory name).
        created_utc: ISO-8601 UTC timestamp.
        arm: ``"A"`` or ``"B"``.
        model_key: ``"strong"`` or ``"cheap"``.
        model_id: Resolved model id from config.
        split: Partition name the run scored.
        seed: Training seed.
        alpha: Gate alpha in force.
        config: The fully resolved config, AFTER ``--set`` overrides.
        overrides: The raw ``--set key=value`` strings, for provenance.
        prompt_hashes: ``{"agent_A": hex, "act_labeling": hex}``.
        dataset_hash: From :attr:`Partitions.dataset_hash`.
        bank_hash: From :attr:`Bank.bank_hash`.
        git_commit: HEAD sha, or ``"unknown"`` outside a git work tree.
        novel_subflows: The resolved NOVEL list (spec 3.4 requires it here).
        machine: ``{"platform", "processor", "cpu_count", "python", "torch"}``.
        llm_enabled: The value of ``llm.enabled``. Recorded so a reader can tell
            at a glance whether a run could have billed anything.
        price_list_date: From config, for spec 8.6.
    """

    run_id: str
    created_utc: str
    arm: str
    model_key: str
    model_id: str
    split: str
    seed: int
    alpha: float
    config: dict[str, Any] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    dataset_hash: str = ""
    bank_hash: str = ""
    git_commit: str = "unknown"
    novel_subflows: list[str] = field(default_factory=list)
    machine: dict[str, Any] = field(default_factory=dict)
    llm_enabled: bool = False
    price_list_date: str = ""
