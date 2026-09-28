"""Spec 4.1 -- Loader: reading files, normalizing keys, building splits (spec 3.4), and
building the context window (spec 6.1 step 4). No modeling.

Every number this project reports is downstream of this module, so the four
places ABCD contradicts the spec are handled here, once, loudly:

D1  The processed turn list is ``convo["delexed"]`` (``data.turn_list_key``).
    ``len(original) == len(delexed)`` and the two are index-aligned 1:1.
D2  ``targets[4]`` is a RANK into this turn's own ``candidates``, not a global
    utterance id. :class:`NormalizedTurn` therefore carries BOTH ``utt_rank``
    (raw, what the official evaluator wants) and ``utt_id``
    (``candidates[utt_rank]``, for text lookup).
D3  ``end_conversation`` never appears in raw ABCD. ``utils/process.py``
    synthesizes one extra example per conversation; we replicate it exactly
    (``data.synthesize_end_conversation``).
D4  ``delexed`` text is ALREADY partially delexicalized with ABCD's own
    ``<slot>`` markers. Those markers are read here as *disclosure evidence*:
    a customer turn reading ``"order id: <order_id>"`` discloses the order id
    just as surely as one reading ``"order id: 3348917502"``.

THE LEAKAGE BOUNDARY (spec 6.1 step 4, and the reason this module has a test)
-----------------------------------------------------------------------------
``convo["scenario"]`` holds the gold ``subflow`` (== the intent label) and every
value the customer will ever state. Handing it to the model reads as 99-100%
accuracy and means nothing. :func:`build_context` therefore exposes a scenario
field ONLY when the customer has already said it -- either its surface value or
ABCD's ``<slot>`` marker for it -- in a turn strictly before the one being
predicted. ``flow`` and ``subflow`` are not dict-valued scenario blocks and are
additionally named in ``data.disclosure_exclude_fields``: they can never be
disclosed by any code path. ``tests/test_data.py`` proves it.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code. The module-level constants below are STRUCTURAL FACTS
about ABCD v1.1 (how many splits, which keys, the 5-tuple arity of ``targets``),
not tunables; every tunable is a ``data.*`` config key.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any, Iterator, Optional, Sequence

from reflex.config import get_dotted, resolve_path
from reflex.contracts import ContractViolation
from reflex.schemas import NEXT_STEPS, ContextWindow, NormalizedTurn, Partitions

__all__ = [
    "load_raw_abcd",
    "load_utterances",
    "load_ontology",
    "load_kb",
    "load_guidelines",
    "subflow_list",
    "action_list",
    "nextstep_list",
    "normalize_conversation",
    "normalize_split",
    "select_novel_subflows",
    "build_partitions",
    "learning_curve_subset",
    "build_context",
    "iter_agent_turns",
    "turn_key",
    "dataset_hash",
]


# --------------------------------------------------------------------------- #
# Structural facts about ABCD v1.1 (not tunables -- see the module docstring)
# --------------------------------------------------------------------------- #

#: The three official split keys of ``abcd_v<version>.json``.
_SPLIT_KEYS: tuple[str, ...] = ("train", "dev", "test")

#: The exact key set of one raw conversation.
_CONVO_KEYS: frozenset[str] = frozenset({"convo_id", "scenario", "original", "delexed"})

#: The exact key set of one raw turn.
_TURN_KEYS: frozenset[str] = frozenset({"speaker", "text", "turn_count", "targets", "candidates"})

#: The exact key set of ``ontology.json``.
_ONTOLOGY_KEYS: frozenset[str] = frozenset({"actions", "intents", "next_steps", "values", "vocabulary"})

#: ``targets == [intent, nextstep, action, values, utt_rank]``.
_TARGET_ARITY: int = 5

#: Positions inside ``targets``.
_T_INTENT, _T_NEXTSTEP, _T_ACTION, _T_VALUES, _T_UTT_RANK = 0, 1, 2, 3, 4

#: Speaker determines nextstep 1:1 in raw ABCD (verified over all 220,983 turns).
_SPEAKER_TO_NEXTSTEP: dict[str, Optional[str]] = {
    "agent": "retrieve_utterance",
    "action": "take_action",
    "customer": None,
}

#: ABCD flattens 10 flows into exactly 55 subflows; head H2's class count.
_EXPECTED_SUBFLOWS: int = 55

#: ``targets[4]`` of a synthesized ``end_conversation`` example (utils/process.py).
_NO_UTTERANCE_RANK: int = -1

#: The speaker attributed to the synthetic end turn. ABCD gives it none; ending
#: the conversation is an agent-side decision, and :func:`iter_agent_turns`
#: selects on ``nextstep is not None`` rather than on this value.
_SYNTHETIC_END_SPEAKER: str = "agent"

#: Text of the synthetic end turn. It is never fed to a model: the turn is a
#: prediction target, and its own text is by construction after the context.
_SYNTHETIC_END_TEXT: str = ""

#: Data files hashed by :func:`dataset_hash`, after ``abcd_v<version>.json``.
_HASHED_FILES: tuple[str, ...] = ("utterances.json", "ontology.json", "kb.json", "guidelines.json")

#: I/O buffer for streaming hashes, and the hex prefix length of every id hash.
_HASH_CHUNK_BYTES: int = 1 << 20
_HASH_HEX_LEN: int = 16

#: Rendering of the compact state line. ``"speaker|text"`` matches
#: ``utils/process.py``; ``state|`` is a fourth pseudo-speaker so the encoder can
#: tell the state apart from a real turn.
_STATE_PREFIX: str = "state|"
_STATE_EMPTY: str = "none"

#: ABCD's own ``<slot>`` markers, e.g. ``"<order_id>"``.
_MARKER_RE = re.compile(r"<([a-z_]+)>")

#: Conversations kept in the disclosure-timeline cache. An implementation
#: detail of :func:`build_context` (callers walk one conversation at a time), not
#: a knob that can change any number.
_TIMELINE_CACHE_SIZE: int = 8


# --------------------------------------------------------------------------- #
# Small private helpers
# --------------------------------------------------------------------------- #


def _abcd_dir(cfg: dict[str, Any]) -> str:
    return resolve_path(cfg, "data.abcd_dir")


def _data_path(cfg: dict[str, Any], filename: str) -> str:
    """Absolute path of one file inside ``<data.abcd_dir>/data/``."""
    return os.path.join(_abcd_dir(cfg), "data", filename)


def _abcd_filename(cfg: dict[str, Any]) -> str:
    return f"abcd_v{get_dotted(cfg, 'data.version')}.json"


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ABCD file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha16(text: str) -> str:
    """Stable 16-hex-char sha256 of ``text``; the same rule llm_agent uses."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_HEX_LEN]


#: Single-slot cache for the 116 MB corpus, keyed by (path, mtime_ns, size).
_RAW_CACHE: dict[tuple[str, int, int], dict[str, list[dict[str, Any]]]] = {}


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def load_raw_abcd(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read ``abcd_v<version>.json`` and return it unmodified.

    The file is ~116 MB unzipped; load it once per process and pass it around.
    ``data.cache_raw_in_process`` (default true) makes that automatic: the parsed
    object is memoized on (path, mtime, size), so ``build_partitions`` and a
    caller that also needs ``scenario`` share ONE 0.9 GB copy instead of two.
    The object is shared, so treat it as read-only -- the whole package does.

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir`` and ``data.version``.

    Returns:
        ``{"train": [...], "dev": [...], "test": [...]}`` with 8,034 / 1,004 /
        1,004 conversations. Each conversation has exactly the keys
        ``convo_id``, ``scenario``, ``original``, ``delexed``.

    Raises:
        FileNotFoundError: if the file is absent.
        ContractViolation: if the split keys or conversation keys differ from the
            verified set -- fail loudly rather than half-normalizing.
    """
    path = _data_path(cfg, _abcd_filename(cfg))
    if not os.path.exists(path):
        raise FileNotFoundError(f"ABCD corpus not found: {path}")
    cache_enabled = bool(get_dotted(cfg, "data.cache_raw_in_process"))
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size)
    if cache_enabled and key in _RAW_CACHE:
        return _RAW_CACHE[key]

    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ContractViolation(f"{path}: expected a mapping of splits, got {type(raw).__name__}")
    if tuple(raw.keys()) != _SPLIT_KEYS:
        raise ContractViolation(
            f"{path}: split keys are {list(raw.keys())}, expected {list(_SPLIT_KEYS)}"
        )
    for split, convos in raw.items():
        if not isinstance(convos, list):
            raise ContractViolation(f"{path}: split {split!r} is not a list")
        for position, convo in enumerate(convos):
            if not isinstance(convo, dict) or set(convo.keys()) != _CONVO_KEYS:
                got = sorted(convo.keys()) if isinstance(convo, dict) else type(convo).__name__
                raise ContractViolation(
                    f"{path}: {split}[{position}] has keys {got}, expected {sorted(_CONVO_KEYS)}"
                )
    if cache_enabled:
        _RAW_CACHE.clear()
        _RAW_CACHE[key] = raw
    return raw


def load_utterances(cfg: dict[str, Any]) -> list[str]:
    """Read ``utterances.json``: the flat candidate pool.

    IMPORTANT, AND NOT WHAT THE SPEC IMPLIES: this pool is **LEXICALIZED**. Only
    209 of its 95,288 strings contain a ``<slot>`` marker, whereas
    ``delexed[i]["text"]`` carries 21,297 markers across the corpus. So
    ``utterances[turn.utt_id]`` equals ``original[turn_index][1].lower()``, not
    ``turn.text``, on the 376 turns where ABCD's delexicalizer fired. Anything
    comparing composed template text to candidate text must delexicalize the
    candidates first -- see :func:`delexicalize_candidates`.

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir``.

    Returns:
        95,288 strings. Index == the utterance id used in ``candidates``.
    """
    pool = _read_json(_data_path(cfg, "utterances.json"))
    if not isinstance(pool, list) or not all(isinstance(item, str) for item in pool):
        raise ContractViolation("utterances.json must be a flat list of strings")
    return pool


def load_ontology(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``ontology.json``.

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir``.

    Returns:
        A dict with exactly the keys ``actions``, ``intents``, ``next_steps``,
        ``values``, ``vocabulary``. Note the shapes, which differ from the
        spec's prose: ``intents`` is ``{"flows": [10 names], "subflows": {flow ->
        [names]}}`` -- a DICT of lists, not a flat list of 55. ``actions`` is
        ``{"kb_query": 6, "interaction": 10, "faq_policy": 14}`` worth of names.
        ``values`` is ``{"enumerable", "non_enumerable"}``. Use
        :func:`subflow_list` and :func:`action_list` rather than indexing this
        by hand.
    """
    ontology = _read_json(_data_path(cfg, "ontology.json"))
    if not isinstance(ontology, dict) or set(ontology.keys()) != _ONTOLOGY_KEYS:
        got = sorted(ontology.keys()) if isinstance(ontology, dict) else type(ontology).__name__
        raise ContractViolation(f"ontology.json has keys {got}, expected {sorted(_ONTOLOGY_KEYS)}")
    return ontology


def load_kb(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``kb.json``: 55 entries keyed by subflow name.

    Its key set is exactly :func:`subflow_list`'s output (verified). Used by the
    gate's ``unseen_action`` signal (spec 6.6 signal 3).

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir``.

    Returns:
        ``subflow -> allowed/expected actions``.
    """
    kb = _read_json(_data_path(cfg, "kb.json"))
    if not isinstance(kb, dict):
        raise ContractViolation(f"kb.json must be a mapping, got {type(kb).__name__}")
    return kb


def load_guidelines(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``guidelines.json``: the company policy text rendered into the Arm A prompt.

    NOTE the name collision: the official ``utils/load.py`` also defines
    ``load_guidelines()``, but that one returns ``(kb, ontology)`` read from
    hard-coded relative paths. This is REFLEXIVE's function and returns the
    guidelines document. Do not conflate them.

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir``.

    Returns:
        The parsed guidelines document.
    """
    guidelines = _read_json(_data_path(cfg, "guidelines.json"))
    if not isinstance(guidelines, dict):
        raise ContractViolation(f"guidelines.json must be a mapping, got {type(guidelines).__name__}")
    return guidelines


# --------------------------------------------------------------------------- #
# Canonical class orders
# --------------------------------------------------------------------------- #


def subflow_list(ontology: dict[str, Any]) -> list[str]:
    """Flatten ``ontology["intents"]["subflows"]`` into the canonical 55 subflow names.

    This ordering is the class order for head H2 EVERYWHERE -- training,
    calibration, gating, evaluation and the official metrics' ``intent`` label
    list. It must be a pure function of the ontology file so that three agents
    computing it independently agree. Iterate ``ontology["intents"]["flows"]`` in
    file order, and within each flow keep the subflow list in file order. Do NOT
    sort: sorting would be a second, silently different convention.

    Args:
        ontology: From :func:`load_ontology`.

    Returns:
        Exactly 55 subflow names.

    Raises:
        ContractViolation: if the flattened length is not 55.
    """
    intents = ontology.get("intents")
    if not isinstance(intents, dict) or "flows" not in intents or "subflows" not in intents:
        raise ContractViolation("ontology['intents'] must hold 'flows' and 'subflows'")
    subflows_by_flow = intents["subflows"]
    out: list[str] = []
    for flow in intents["flows"]:
        if flow not in subflows_by_flow:
            raise ContractViolation(f"flow {flow!r} has no entry in ontology['intents']['subflows']")
        out.extend(subflows_by_flow[flow])
    if len(out) != _EXPECTED_SUBFLOWS:
        raise ContractViolation(
            f"flattened {len(out)} subflows, expected {_EXPECTED_SUBFLOWS}; "
            "the ontology file is not ABCD v1.1"
        )
    if len(set(out)) != len(out):
        raise ContractViolation("subflow names are not unique across flows")
    return out


def action_list(ontology: dict[str, Any]) -> list[str]:
    """Flatten ``ontology["actions"]`` into the canonical button list.

    Class order for head H3. Iterate the action categories in file order
    (``kb_query``, ``interaction``, ``faq_policy``) and each category's names in
    file order. The resulting length (30) matches the "intent mask should be size
    of 30 long" comment in the official ``cds_report``.

    Args:
        ontology: From :func:`load_ontology`.

    Returns:
        The button names, deduplicated while preserving first-seen order.
    """
    actions = ontology.get("actions")
    if not isinstance(actions, dict):
        raise ContractViolation("ontology['actions'] must be a mapping of categories")
    out: list[str] = []
    seen: set[str] = set()
    for names in actions.values():
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def nextstep_list(ontology: dict[str, Any]) -> list[str]:
    """Return ``ontology["next_steps"]`` after asserting it matches ``NEXT_STEPS``.

    The order ``[retrieve_utterance, take_action, end_conversation]`` is
    NORMATIVE: the official ``cds_report`` branches on ``nextstep_label == 0/1/2``
    with exactly that meaning. This function exists so that the assertion happens
    somewhere, once.

    Args:
        ontology: From :func:`load_ontology`.

    Returns:
        The three nextstep names in normative order.

    Raises:
        ContractViolation: if the file's order ever differs.
    """
    steps = ontology.get("next_steps")
    if tuple(steps or ()) != NEXT_STEPS:
        raise ContractViolation(
            f"ontology['next_steps'] is {steps!r}, but the official cds_report "
            f"branches on the order {list(NEXT_STEPS)}"
        )
    return list(steps)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def _violation(convo_id: Any, turn_index: int, message: str) -> ContractViolation:
    return ContractViolation(f"convo {convo_id} turn {turn_index}: {message}")


def normalize_conversation(
    convo: dict[str, Any],
    utterances: Sequence[str],
    cfg: dict[str, Any],
) -> list[NormalizedTurn]:
    """Convert one raw conversation into :class:`NormalizedTurn` objects.

    Four things here are easy to get wrong and are therefore spelled out.

    1. **Use ``convo["delexed"]``, not ``convo["original"]``,** as the turn list.
       Spec 3.2 declines to name it. ``len(original) == len(delexed)`` for all
       10,042 conversations and they are index-aligned 1:1, so
       ``original[turn_index][1]`` is the lexicalized twin of
       ``delexed[turn_index]["text"]`` whenever you need original casing.
    2. **``targets[4]`` is a RANK, not a global id.** Set
       ``utt_rank = targets[4]`` and ``utt_id = candidates[targets[4]]``. See
       :class:`NormalizedTurn` for the verification numbers.
    3. **Synthesize the ``end_conversation`` turn.** ``end_conversation`` never
       appears in raw ABCD; ``utils/process.py`` appends ONE extra example per
       conversation after the last turn, with
       ``end_targets = turn["targets"].copy(); end_targets[1] = "end_conversation";
       end_targets[4] = -1`` -- i.e. it inherits the LAST turn's intent, action
       and values, and reuses the last turn's ``turn_count``. Replicate exactly,
       with ``is_synthetic_end=True``, or nextstep accuracy will not be
       comparable to published numbers. Gated on
       ``data.synthesize_end_conversation``.
    4. **Speaker determines nextstep 1:1** in raw ABCD (agent ->
       retrieve_utterance, action -> take_action, customer -> None). Read
       ``targets[1]``, but treat a disagreement with the speaker as a
       :class:`ContractViolation` rather than silently trusting one of them.

    ONE DELIBERATE DEPARTURE FROM A LITERAL READING OF POINT 3: the synthetic end
    turn inherits ``intent`` and ``turn_count`` but NOT ``action`` / ``values``.
    ``utils/process.py`` copies all five target slots, but its own
    ``CDSProcessor.collect_one_example`` then fills ``action_id`` / ``value_id``
    only under ``if nextstep == 'take_action'``, so the inherited button is
    discarded and the example is emitted with ``-1``. The official
    ``cds_report`` counts the action denominator as ``sum(bslot_label >= 0)``.
    Carrying the inherited button here would add 10,042 turns to that
    denominator and make action accuracy incomparable to published numbers, so
    ``action`` and ``values`` are ``None`` -- which is also what
    :class:`NormalizedTurn` documents ("else ``None``").

    Args:
        convo: One raw conversation dict.
        utterances: From :func:`load_utterances`, for resolving ``utt_id``.
        cfg: Resolved config. Uses ``data.turn_list_key`` and
            ``data.synthesize_end_conversation``.

    Returns:
        ``len(delexed)`` turns, plus one synthetic ``end_conversation`` turn when
        enabled. ``turn_index`` is dense and 0-based over the returned list.

    Raises:
        ContractViolation: on any of the structural violations above.
    """
    turn_list_key = str(get_dotted(cfg, "data.turn_list_key"))
    synthesize_end = bool(get_dotted(cfg, "data.synthesize_end_conversation"))

    convo_id = convo.get("convo_id")
    if not isinstance(convo_id, int):
        raise ContractViolation(f"convo_id must be an int, got {convo_id!r}")
    raw_turns = convo.get(turn_list_key)
    if not isinstance(raw_turns, list):
        raise ContractViolation(f"convo {convo_id}: no turn list under data.turn_list_key={turn_list_key!r}")
    original = convo.get("original")
    if isinstance(original, list) and len(original) != len(raw_turns):
        raise ContractViolation(
            f"convo {convo_id}: len(original)={len(original)} != len({turn_list_key})={len(raw_turns)}; "
            "the 1:1 index alignment this project relies on is broken"
        )

    pool_size = len(utterances)
    out: list[NormalizedTurn] = []
    for turn_index, raw in enumerate(raw_turns):
        if not isinstance(raw, dict) or not _TURN_KEYS <= set(raw.keys()):
            got = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
            raise _violation(convo_id, turn_index, f"turn keys are {got}, expected {sorted(_TURN_KEYS)}")

        speaker = raw["speaker"]
        if speaker not in _SPEAKER_TO_NEXTSTEP:
            raise _violation(convo_id, turn_index, f"unknown speaker {speaker!r}")
        targets = raw["targets"]
        if not isinstance(targets, list) or len(targets) != _TARGET_ARITY:
            raise _violation(convo_id, turn_index, f"targets must be a {_TARGET_ARITY}-list, got {targets!r}")

        nextstep = targets[_T_NEXTSTEP]
        expected_nextstep = _SPEAKER_TO_NEXTSTEP[speaker]
        if nextstep != expected_nextstep:
            raise _violation(
                convo_id,
                turn_index,
                f"speaker {speaker!r} implies nextstep {expected_nextstep!r} but targets[1] is {nextstep!r}",
            )

        raw_candidates = raw["candidates"] or []
        utt_rank = targets[_T_UTT_RANK]
        candidates: Optional[list[int]] = None
        utt_id: Optional[int] = None
        if nextstep == "retrieve_utterance":
            if not raw_candidates:
                raise _violation(convo_id, turn_index, "retrieve_utterance turn has no candidates")
            if not isinstance(utt_rank, int) or not 0 <= utt_rank < len(raw_candidates):
                raise _violation(
                    convo_id,
                    turn_index,
                    f"targets[4]={utt_rank!r} is not a rank into this turn's {len(raw_candidates)} candidates "
                    "(it is a RANK, not a global utterance id)",
                )
            candidates = list(raw_candidates)
            utt_id = candidates[utt_rank]
            if not 0 <= utt_id < pool_size:
                raise _violation(
                    convo_id, turn_index, f"resolved utt_id={utt_id} is outside the {pool_size}-utterance pool"
                )
        elif raw_candidates:
            raise _violation(
                convo_id, turn_index, f"{speaker!r} turn carries {len(raw_candidates)} candidates, expected none"
            )

        is_action = nextstep == "take_action"
        raw_values = targets[_T_VALUES]
        if raw_values is not None and not isinstance(raw_values, list):
            raise _violation(convo_id, turn_index, f"targets[3] must be a list, got {raw_values!r}")

        out.append(
            NormalizedTurn(
                convo_id=convo_id,
                turn_index=turn_index,
                speaker=speaker,
                text=raw["text"],
                nextstep=nextstep,
                intent=targets[_T_INTENT],
                action=targets[_T_ACTION] if is_action else None,
                values=[str(v) for v in (raw_values or [])] if is_action else None,
                utt_id=utt_id,
                candidates=candidates,
                utt_rank=utt_rank if isinstance(utt_rank, int) else _NO_UTTERANCE_RANK,
                turn_count=raw["turn_count"],
                is_synthetic_end=False,
            )
        )

    if synthesize_end and raw_turns:
        last = raw_turns[-1]
        end_targets = list(last["targets"])
        end_targets[_T_NEXTSTEP] = "end_conversation"
        end_targets[_T_UTT_RANK] = _NO_UTTERANCE_RANK
        out.append(
            NormalizedTurn(
                convo_id=convo_id,
                turn_index=len(out),
                speaker=_SYNTHETIC_END_SPEAKER,
                text=_SYNTHETIC_END_TEXT,
                nextstep=end_targets[_T_NEXTSTEP],
                intent=end_targets[_T_INTENT],
                action=None,
                values=None,
                utt_id=None,
                candidates=None,
                utt_rank=end_targets[_T_UTT_RANK],
                turn_count=last["turn_count"],
                is_synthetic_end=True,
            )
        )
    return out


def normalize_split(
    conversations: Sequence[dict[str, Any]],
    utterances: Sequence[str],
    cfg: dict[str, Any],
) -> dict[int, list[NormalizedTurn]]:
    """Normalize a whole split.

    Args:
        conversations: Raw conversations for one split.
        utterances: From :func:`load_utterances`.
        cfg: Resolved config.

    Returns:
        ``convo_id -> turns``. Insertion order follows the file, so iteration is
        deterministic.
    """
    out: dict[int, list[NormalizedTurn]] = {}
    for convo in conversations:
        convo_id = int(convo["convo_id"])
        if convo_id in out:
            raise ContractViolation(f"duplicate convo_id {convo_id} within one split")
        out[convo_id] = normalize_conversation(convo, utterances, cfg)
    return out


# --------------------------------------------------------------------------- #
# Partitions (spec 3.4)
# --------------------------------------------------------------------------- #


def select_novel_subflows(ontology: dict[str, Any], count: int, seed: int) -> list[str]:
    """Choose the NOVEL subflows (spec 3.4).

    Must be a pure function of ``(subflow_list(ontology), count, seed)`` -- the
    resulting list goes into every manifest and changing it invalidates every
    spec 8.4 number. Use ``random.Random(seed).sample(sorted(subflows), count)``:
    sorting the population first makes the draw independent of ontology file
    order.

    Args:
        ontology: From :func:`load_ontology`.
        count: ``data.novel_subflows_count`` (default 5).
        seed: ``data.novel_subflows_seed`` (default 13).

    Returns:
        ``count`` subflow names, sorted, for a stable manifest.
    """
    population = sorted(subflow_list(ontology))
    count = int(count)
    if not 0 <= count <= len(population):
        raise ContractViolation(f"novel subflow count {count} is outside [0, {len(population)}]")
    return sorted(random.Random(int(seed)).sample(population, count))


def build_partitions(cfg: dict[str, Any]) -> Partitions:
    """Build the five partitions of spec 3.4 in one call.

    NOVEL conversations are removed from train and dev ENTIRELY (not held out
    within them), and test is cut into ``test_seen`` / ``test_novel`` by whether
    the conversation's ``scenario["subflow"]`` is novel.

    Args:
        cfg: Resolved config.

    Returns:
        A :class:`Partitions` whose ``train`` and ``dev`` are already filtered.
    """
    raw = load_raw_abcd(cfg)
    utterances = load_utterances(cfg)
    ontology = load_ontology(cfg)
    novel = select_novel_subflows(
        ontology,
        count=int(get_dotted(cfg, "data.novel_subflows_count")),
        seed=int(get_dotted(cfg, "data.novel_subflows_seed")),
    )
    novel_set = set(novel)
    known = set(subflow_list(ontology))
    unknown = novel_set - known
    if unknown:
        raise ContractViolation(f"novel subflows are not ontology subflows: {sorted(unknown)}")

    turn_list_key = str(get_dotted(cfg, "data.turn_list_key"))

    def _subflow(convo: dict[str, Any]) -> str:
        """The conversation's subflow, taken from the GOLD INTENT, not the scenario.

        D9, measured here and not in the README's eight: ``scenario["subflow"]``
        takes 96 distinct values, but only 55 of them are ontology subflows.
        2,172 / 10,042 conversations (21.6%) carry a FINER-GRAINED scenario
        string -- ``timing_4``, ``pricing_3``, ``boots_how_1``,
        ``status_questions`` -- whose ontology subflow is ``timing``,
        ``pricing``, ``jeans``, ``status_active``. Matching the novel list
        against that raw string would leave 21.6% of conversations unmatchable
        and quietly leave novel-intent conversations in train, which is exactly
        the mistake spec 3.4 cannot survive. ``targets[0]`` is verified constant
        within every conversation (0 / 10,042 exceptions), always one of the 55,
        and is the label head H2 is trained on, so it is the partition key.
        """
        turns = convo.get(turn_list_key) or []
        intents = {turn["targets"][_T_INTENT] for turn in turns}
        if len(intents) != 1:
            raise ContractViolation(
                f"convo {convo.get('convo_id')}: {len(intents)} distinct targets[0] intents {sorted(intents)}; "
                "a conversation must have exactly one gold subflow"
            )
        intent = str(intents.pop())
        if intent not in known:
            raise ContractViolation(
                f"convo {convo.get('convo_id')}: gold intent {intent!r} is not an ontology subflow"
            )
        scenario_subflow = (convo.get("scenario") or {}).get("subflow")
        if scenario_subflow in known and scenario_subflow != intent:
            raise ContractViolation(
                f"convo {convo.get('convo_id')}: scenario subflow {scenario_subflow!r} and gold intent "
                f"{intent!r} are both ontology subflows but disagree"
            )
        return intent

    seen_train = [c for c in raw["train"] if _subflow(c) not in novel_set]
    seen_dev = [c for c in raw["dev"] if _subflow(c) not in novel_set]
    test_seen = [c for c in raw["test"] if _subflow(c) not in novel_set]
    test_novel = [c for c in raw["test"] if _subflow(c) in novel_set]

    return Partitions(
        train=normalize_split(seen_train, utterances, cfg),
        dev=normalize_split(seen_dev, utterances, cfg),
        test_seen=normalize_split(test_seen, utterances, cfg),
        test_novel=normalize_split(test_novel, utterances, cfg),
        novel_subflows=novel,
        dataset_hash=dataset_hash(cfg),
    )


def learning_curve_subset(convo_ids: Sequence[int], fraction: float, seed: int) -> list[int]:
    """Sample a fraction of conversations for the spec 3.4 / E4 learning curve.

    Sampling is at the CONVERSATION level, never the turn level: turn-level
    sampling would leak context across the subset boundary. Must be nested --
    the 10% subset is a subset of the 25% subset -- so that E4 measures added
    data rather than a different draw. Achieve that by shuffling once with
    ``seed`` and taking prefixes.

    Args:
        convo_ids: Candidate conversation ids (typically novel-filtered train).
        fraction: One of ``data.learning_curve_fractions``.
        seed: ``data.novel_subflows_seed`` reused, or an explicit seed.

    Returns:
        ``round(fraction * len(convo_ids))`` ids, in the original relative order.
    """
    ids = list(convo_ids)
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ContractViolation(f"learning-curve fraction {fraction} is outside [0, 1]")
    take = int(round(fraction * len(ids)))
    order = list(range(len(ids)))
    random.Random(int(seed)).shuffle(order)
    chosen = set(order[:take])
    return [cid for position, cid in enumerate(ids) if position in chosen]


# --------------------------------------------------------------------------- #
# Context window (spec 6.1 step 4) -- and the leakage boundary
# --------------------------------------------------------------------------- #


def _context_k(cfg: dict[str, Any]) -> Optional[int]:
    """``data.context_turns_K`` as an int, or ``None`` for the full thread.

    Decision A of 2026-09-16: the default is ``full``. 49.5% of predicted turns
    have no customer message immediately before them, so a short window starves
    half the dataset. Measured here on all 13,392 dev agent-side contexts with
    the ModernBERT-base tokenizer: turns only, mean 168.6 subword tokens, p99
    512, 1.00% over ``data.max_len``; turns + the state line, mean 223.7, p99
    646, 4.05% over. The state line costs ~55 tokens and triples the truncation
    rate -- still a 96% fit, and the reason the state is rendered LAST. An int is
    still honoured (the K ablation sweeps 4/6/10/full), which is why this never
    hard-codes 6.
    """
    value = get_dotted(cfg, "data.context_turns_K")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContractViolation(f"data.context_turns_K must be an int or 'full', got {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ContractViolation(f"data.context_turns_K must be >= 0, got {value}")
        return value
    text = str(value).strip().lower()
    if text in {"full", "all", "none", ""}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ContractViolation(
            f"data.context_turns_K must be an int or 'full', got {value!r}"
        ) from exc
    if parsed < 0:
        raise ContractViolation(f"data.context_turns_K must be >= 0, got {parsed}")
    return parsed


def _as_surface_values(raw: Any) -> list[str]:
    """Surface strings a scenario leaf could be stated as; ``[]`` if not statable."""
    if raw is None or isinstance(raw, bool):
        return []
    if isinstance(raw, (str, int, float)):
        text = str(raw).strip()
        return [text] if text else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            out.extend(_as_surface_values(item))
        return out
    return []  # nested dicts are not a single statable fact


def _candidate_markers(leaf: str) -> tuple[str, ...]:
    """ABCD ``<slot>`` marker names that stand for a scenario leaf field.

    ABCD's marker vocabulary is ``ontology["values"]["non_enumerable"]`` and its
    names are the scenario leaf names, except that the two ``product`` fields are
    pluralized in the scenario (``<name>`` -> ``product.names``, ``<amount>`` ->
    ``product.amounts``). Exact-or-singular matching covers all 11 markers and
    never confuses ``<name>`` with ``personal.customer_name``.
    """
    if leaf.endswith("s") and not leaf.endswith("ss"):
        return (leaf, leaf[:-1])
    return (leaf,)


def _flatten_scenario(scenario: dict[str, Any], cfg: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """``[(dotted_field, surface_values)]`` for every DISCLOSABLE scenario leaf.

    ``flow`` and ``subflow`` are plain strings rather than blocks, so the
    dict-only walk already excludes them; ``data.disclosure_exclude_fields``
    names them anyway so the guard survives a future ontology that nests them.
    """
    groups = [str(g) for g in (get_dotted(cfg, "data.disclosure_scenario_groups") or [])]
    excluded = {str(f) for f in (get_dotted(cfg, "data.disclosure_exclude_fields") or [])}
    out: list[tuple[str, list[str]]] = []
    for group, block in (scenario or {}).items():
        if group in excluded or not isinstance(block, dict):
            continue
        if groups and group not in groups:
            continue
        for leaf, raw in block.items():
            field = f"{group}.{leaf}"
            if leaf in excluded or field in excluded:
                continue
            values = _as_surface_values(raw)
            if values:
                out.append((field, values))
    return out


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _contains_value(haystack: str, needle: str) -> bool:
    """Whole-token substring test -- no regex, so no per-value pattern cache."""
    if not needle:
        return False
    span = len(needle)
    start = 0
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return False
        before_ok = found == 0 or not haystack[found - 1].isalnum()
        after_ok = found + span == len(haystack) or not haystack[found + span].isalnum()
        if before_ok and after_ok:
            return True
        start = found + 1


#: ``(convo_id, customer texts, scenario leaves) -> {field: (turn_index, value)}``.
#: Keyed on everything the timeline depends on, so it is a memo, never a guess.
_TIMELINE_CACHE: dict[Any, dict[str, tuple[int, str]]] = {}


def _disclosure_timeline(
    turns: Sequence[NormalizedTurn],
    scenario: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, tuple[int, str]]:
    """``field -> (first turn index that disclosed it, disclosed surface value)``.

    THE LEAKAGE BOUNDARY. A field is disclosed only by a CUSTOMER turn, and only
    by that turn either stating the value (whole-token match, at least
    ``data.disclosure_min_value_chars`` characters, which keeps ``"ny"``,
    ``"yes"`` and ``"1"`` from matching every conversation) or carrying ABCD's
    ``<slot>`` marker for the field -- ``"order id: <order_id>"`` discloses the
    order id exactly as ``"order id: 3348917502"`` does, because ABCD masked the
    value the customer really typed.
    """
    fields = _flatten_scenario(scenario, cfg)
    min_chars = int(get_dotted(cfg, "data.disclosure_min_value_chars"))
    customer_texts = tuple(
        (turn.turn_index, turn.text) for turn in turns if turn.speaker == "customer"
    )
    # The key carries everything the timeline depends on -- the conversation, the
    # scenario leaves AND the disclosure policy -- so a config change can never
    # be served a stale answer.
    key = (
        int(turns[0].convo_id) if turns else -1,
        min_chars,
        customer_texts,
        tuple((field, tuple(values)) for field, values in fields),
    )
    cached = _TIMELINE_CACHE.get(key)
    if cached is not None:
        return cached

    pending = {
        field: (values, set(_candidate_markers(field.rsplit(".", 1)[-1])))
        for field, values in fields
    }
    timeline: dict[str, tuple[int, str]] = {}
    for turn_index, text in customer_texts:
        if not pending:
            break
        haystack = _normalize_text(text)
        markers = set(_MARKER_RE.findall(haystack))
        resolved: list[str] = []
        for field, (values, field_markers) in pending.items():
            if markers & field_markers:
                # The leakage boundary, in the small. A multi-valued leaf
                # (product.names, product.amounts) has SEVERAL surface values;
                # joining all of them would disclose values this turn never
                # carried, on the evidence of a single mask. Disclose only what
                # the turn evidences: the values it spells out, or -- since
                # ABCD masked them -- as many as it carries markers for. The
                # surface-text branch below cannot rescue these: amounts are
                # 2 characters and `len(v) >= min_chars` filters them out, so
                # this branch is the only way they are ever disclosed.
                stated = [
                    v
                    for v in values
                    if len(v) >= min_chars and _contains_value(haystack, _normalize_text(v))
                ]
                hits = sum(haystack.count(f"<{marker}>") for marker in field_markers)
                shown = stated or values[: max(1, hits)]
                timeline[field] = (turn_index, ", ".join(shown))
                resolved.append(field)
                continue
            matched = [v for v in values if len(v) >= min_chars and _contains_value(haystack, _normalize_text(v))]
            if matched:
                timeline[field] = (turn_index, ", ".join(matched))
                resolved.append(field)
        for field in resolved:
            pending.pop(field, None)

    if len(_TIMELINE_CACHE) >= _TIMELINE_CACHE_SIZE:
        _TIMELINE_CACHE.clear()
    _TIMELINE_CACHE[key] = timeline
    return timeline


def _render_state(disclosed: dict[str, str], actions_so_far: Sequence[dict[str, Any]]) -> str:
    facts = "; ".join(f"{field}={value}" for field, value in disclosed.items()) or _STATE_EMPTY
    taken = (
        "; ".join(
            "{}({})".format(entry["action"], ", ".join(str(v) for v in entry.get("values") or []))
            for entry in actions_so_far
        )
        or _STATE_EMPTY
    )
    return f"{_STATE_PREFIX}disclosed: {facts} || actions: {taken}"


def build_context(
    turns: Sequence[NormalizedTurn],
    turn_index: int,
    scenario: dict[str, Any],
    cfg: dict[str, Any],
) -> ContextWindow:
    """Build the spec 6.1 step 4 context for the agent turn at ``turn_index``.

    The context is the previous ``data.context_turns_K`` turns with speaker tags,
    plus a compact state string of DISCLOSED scenario fields and actions taken so
    far with their values.

    **The disclosure rule is a leakage guard.** A scenario field counts as
    disclosed only if its value (or ABCD's ``<slot>`` marker for it) has appeared
    in a customer turn at index ``< turn_index``. Passing the whole
    ``convo["scenario"]`` would hand the model the gold intent and the gold
    values, and every downstream number would be meaningless. Both the encoder
    and the LLM agent consume this same object (spec 6.9 step 1), so the guard
    applies equally to both arms.

    LAYOUT: turns oldest-first, then ONE ``state|`` line last::

        agent|hi! how can i help you?
        customer|i need to return an item, order id: <order_id>
        state|disclosed: order.order_id=3348917502 || actions: pull-up-account(crystal minh)

    The state goes last so that a tokenizer configured with
    ``truncation_side="left"`` keeps both the state and the most recent turns on
    the 1.5% of turns that exceed ``data.max_len``. K truncates the TURN LIST
    only: the state always summarizes the whole prefix, which is what makes a
    small K survivable at all.

    Args:
        turns: All turns of the conversation, from :func:`normalize_conversation`.
        turn_index: The agent turn being predicted. Only turns before it are read.
        scenario: The raw ``convo["scenario"]`` dict, consulted ONLY to decide
            which of its fields have been disclosed.
        cfg: Resolved config. Uses ``data.context_turns_K``.

    Returns:
        A :class:`ContextWindow` with ``context_hash`` populated.
    """
    if not turns:
        raise ContractViolation("cannot build a context for an empty conversation")
    if turn_index < 0 or turn_index >= len(turns):
        raise ContractViolation(f"turn_index {turn_index} is outside the conversation ({len(turns)} turns)")

    prior = list(turns[:turn_index])
    k = _context_k(cfg)
    windowed = prior if k is None else prior[max(0, len(prior) - k) :]
    turn_strings = [f"{turn.speaker}|{turn.text}" for turn in windowed]

    timeline = _disclosure_timeline(turns, scenario, cfg)
    disclosed = {
        field: value
        for field, (disclosed_at, value) in timeline.items()
        if disclosed_at < turn_index
    }
    actions_so_far = [
        {"action": turn.action, "values": list(turn.values or [])}
        for turn in prior
        if turn.speaker == "action" and turn.action
    ]

    text = "\n".join(turn_strings + [_render_state(disclosed, actions_so_far)])
    return ContextWindow(
        convo_id=int(turns[turn_index].convo_id),
        turn_index=int(turn_index),
        text=text,
        turns=turn_strings,
        disclosed=disclosed,
        actions_so_far=actions_so_far,
        context_hash=_sha16(text),
    )


# --------------------------------------------------------------------------- #
# Iteration, ids, hashing
# --------------------------------------------------------------------------- #


def iter_agent_turns(partition: dict[int, list[NormalizedTurn]]) -> Iterator[tuple[int, int, NormalizedTurn]]:
    """Iterate the AGENT-SIDE turns of a partition in deterministic order.

    "Agent-side" means every turn that has a non-``None`` ``nextstep``: agent
    (retrieve_utterance), action (take_action) and the synthetic
    end_conversation turn. Customer turns are inputs and are never predicted
    (spec 2), so they are skipped. This is the denominator of ``reflex_rate``
    (spec 8.1) and the turn set both arms are scored on (spec 8.2).

    Args:
        partition: ``convo_id -> turns``.

    Yields:
        ``(convo_id, turn_index, turn)`` in conversation order, then turn order.
    """
    for convo_id, turns in partition.items():
        for turn in turns:
            if turn.nextstep is not None:
                yield int(convo_id), int(turn.turn_index), turn


def turn_key(split: str, convo_id: int, turn_index: int) -> str:
    """Return the canonical turn id string ``"{split}:{convo_id}:{turn_index}"``.

    Used for :attr:`TurnLabel.turn_id` and :attr:`Template.example_context_ids`.
    One function so all three producers agree on the separator.

    Args:
        split: Partition name, e.g. ``"train"``.
        convo_id: Conversation id.
        turn_index: Dense turn index.

    Returns:
        e.g. ``"train:1234:12"``.
    """
    return f"{split}:{convo_id}:{turn_index}"


def dataset_hash(cfg: dict[str, Any]) -> str:
    """Return a stable 16-char sha256 prefix over the ABCD data files.

    Hash the raw BYTES of ``abcd_v<version>.json``, ``utterances.json``,
    ``ontology.json``, ``kb.json`` and ``guidelines.json``, in that fixed order,
    streaming in chunks. Do not hash parsed JSON: dict ordering and float
    repr would make the hash Python-version dependent.

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir`` and ``data.version``.

    Returns:
        16 lowercase hex chars, for the manifest and report Section 1.
    """
    digest = hashlib.sha256()
    for filename in (_abcd_filename(cfg),) + _HASHED_FILES:
        path = _data_path(cfg, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"ABCD file not found, cannot hash the dataset: {path}")
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()[:_HASH_HEX_LEN]
