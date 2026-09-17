"""Spec 4.5 / 6.5 -- Encoder+Selector at inference: SCORES ONLY, no decisions.

What this module does, and the three things it deliberately does not do:

* It produces one :class:`~reflex.schemas.SelectorScores` per agent turn
  (softmax probabilities per head, plus the novelty DISTANCE) and the argmax
  reading of those scores as a :class:`~reflex.schemas.Selection`.
* It does **not** route. ``route``/``reason`` belong to :mod:`reflex.gate`, which
  is the only decision point (contract boundary ruling 1: the distance is a
  signal computed here and merely thresholded there).
* It does **not** fill slots or check availability -- :mod:`reflex.fill` owns
  slot logic (boundary ruling 2).
* It does **not** read a gold field. ``turn`` is consulted for ``candidates``
  only. Reading ``turn.intent`` / ``action`` / ``values`` / ``utt_id`` /
  ``utt_rank`` -- or branching on ``turn.nextstep``, which is 1:1 with the
  speaker label the model is being asked to predict -- is leakage. Every branch
  below keys on the **predicted** nextstep.

PER-HEAD CONTEXT (DECISIONS D2b and D7), AND WHY IT IS OFF BY DEFAULT
---------------------------------------------------------------------
Three independent measured axes (window size, order sensitivity, and order
destruction under refit) agree that the heads want different inputs:

* ``nextstep`` (H1) and ``action`` (H3) are LOCAL and ORDERED -- k6 window with
  recency-tagged tokens measured 0.8345-0.8351 on nextstep, +2.8 over the best
  plain bar and +4.5 over a 33.7M-param frozen-encoder MLP.
* ``intent`` (H2) is GLOBAL and UNORDERED -- full thread 0.8074 vs 0.6153 at k6
  (+19.2); tagging HURTS it.
* ``skeleton`` (H5) and ``template`` (H7) are UNMEASURED on either axis. They
  inherit, and they are labelled unmeasured rather than assumed.

The mechanism is implemented here in full and is config-driven
(``select.head_context``): a distinct context string per head group, one encoder
pass per DISTINCT string, each head read from its own pass. Nothing about the
finding is discarded.

It is nevertheless **inactive against the checkpoints this build produces**, and
that is deliberate rather than a silent collapse:

1. :mod:`reflex.models` exposes ONE context representation. ``forward`` encodes
   one string into one ``query`` and every head reads that same vector; there is
   no per-head input in the model, so per-head context can only be realised by
   re-running the encoder from here.
2. More decisively, :mod:`reflex.train` builds exactly one ``context_text`` per
   example (``build_train_examples`` -> ``context.text``, the global
   ``data.context_turns_K`` window, untagged). D2b/D7 were measured with a probe
   REFIT on each window. Feeding a trained head a window and a tokenisation it
   never saw is a train/test mismatch that subtracts accuracy; it does not add
   the measured gain.
3. Recency tagging in particular was measured on TF-IDF, where it repairs a bag
   of words' permutation invariance. A transformer already encodes order. What
   tagging does to a fine-tuned encoder is UNMEASURED.

So ``select.head_context.require_checkpoint_match`` (default true) asks the
checkpoint whether it was trained under the same per-head plan -- the key
``metadata["extra"][select.head_context.checkpoint_key]`` -- and, finding no
such record, falls back to the training window for every head and warns loudly.
Train per-head and record the plan in the checkpoint, or set
``require_checkpoint_match: false`` to run the measured plan as an ablation.

NOVELTY IS NEVER RE-WINDOWED. :mod:`reflex.calibrate` indexes ``context.text``;
querying that index with any other rendering would compare vectors of two
different distributions and make ``novelty_distance`` meaningless. The novelty
pass always uses ``ContextWindow.text``.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import math
import re
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from reflex.config import get_dotted
from reflex.contracts import ContractViolation
from reflex.schemas import (
    NEXT_STEPS,
    Bank,
    Calibration,
    ContextWindow,
    NormalizedTurn,
    Selection,
    SelectorScores,
    SlotRegistry,
    SlotSources,
)

__all__ = [
    "build_selector",
    "score_turn",
    "prediction_set",
    "select_from_scores",
    "delexicalize_candidates",
    "map_to_candidate",
]


# --------------------------------------------------------------------------- #
# Module constants. VOCABULARY only -- every threshold, model id, path and
# window lives in configs/default.yaml under `select:` (spec 10).
# --------------------------------------------------------------------------- #

#: The head groups that may be given their own context window (D2b/D7).
#: H6 (act) is reporting-only and selection never reads it (spec 6.4).
_CONTEXT_HEADS = ("nextstep", "intent", "action", "values", "skeleton", "template")

#: ``K: inherit`` means "whatever ``data.context_turns_K`` produced" -- i.e. use
#: ``ContextWindow.text`` byte for byte, which is what training saw.
_INHERIT = "inherit"

#: ``K: full`` means the whole thread the ContextWindow carries.
_FULL = "full"

_WHITESPACE = re.compile(r"\s+")

#: Delexicalized-candidate cache, shared across calls because the same 100
#: utterance ids recur on thousands of turns (contract: "cached per candidate
#: id"). Keyed by (fingerprint of registry+cfg, the utterance TEXT) rather than
#: by the id: the id is only meaningful against one pool, and a cache keyed on it
#: returns another pool's string in any test or ablation that swaps
#: ``utterances``. Bounded by ``select.candidate_embed_cache``; eviction is
#: oldest-first (insertion order), which is deterministic.
_CANDIDATE_DELEX_CACHE: dict[tuple[str, str], str] = {}
_CANDIDATE_DELEX_LOCK = threading.Lock()

#: Candidate-mapping embedders, keyed by model id, for calls made without a
#: selector bundle to hang one on. Loading a sentence encoder per call would
#: dominate the fast path.
_EMBEDDERS: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _torch() -> Any:
    """Import torch lazily: ``import reflex.select`` must stay cheap."""
    import torch  # local import by design

    return torch


def _numpy() -> Any:
    import numpy as np

    return np


def _cfg_get(cfg: dict[str, Any], key: str, fallback_key: Optional[str] = None) -> Any:
    """``get_dotted`` with one named fallback key, for keys another module owns.

    ``train.truncation_side`` and ``train.multi_value_actions`` are read by
    :mod:`reflex.train` but are absent from ``configs/default.yaml``; rather than
    invent them here or crash, this falls back to the ``select.*`` mirror of the
    same knob and the divergence is reported.
    """
    try:
        return get_dotted(cfg, key)
    except KeyError:
        if fallback_key is None:
            raise
        return get_dotted(cfg, fallback_key)


def _softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a plain sequence. Empty in, empty out."""
    if len(values) == 0:
        return []
    top = max(values)
    if not math.isfinite(top):
        # Every column masked out (H4's copy positions are -inf when no token
        # backs them). Uniform is the honest answer; the gate's set will be
        # large and the turn escalates, which is correct.
        uniform = 1.0 / len(values)
        return [uniform] * len(values)
    exps = [math.exp(float(v) - top) if math.isfinite(v) else 0.0 for v in values]
    total = sum(exps)
    if total <= 0.0:  # pragma: no cover - only reachable with all -inf logits
        uniform = 1.0 / len(exps)
        return [uniform] * len(exps)
    return [e / total for e in exps]


def _argmax(values: Sequence[float]) -> int:
    """Argmax with ties broken toward the LOWEST index (spec 10 determinism)."""
    best_index = -1
    best_value = -math.inf
    for index, value in enumerate(values):
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def _normalize_text(text: str, normalizers: Sequence[str]) -> str:
    """Apply the configured normalizers, in the configured order."""
    out = text
    for step in normalizers:
        name = str(step).lower()
        if name == "casefold":
            out = out.casefold()
        elif name == "lower":
            out = out.lower()
        elif name == "collapse_whitespace":
            out = _WHITESPACE.sub(" ", out).strip()
        elif name == "strip":
            out = out.strip()
        elif name == "strip_punct_tail":
            out = out.rstrip(" .!?")
        else:
            raise ValueError(
                f"unknown normalizer {step!r} in select.candidate_normalizers; "
                "supported: casefold, lower, collapse_whitespace, strip, strip_punct_tail"
            )
    return out


def _evict(cache: dict, limit: int) -> None:
    """Bound a cache at ``select.candidate_embed_cache``, oldest insertion first.

    Deterministic: dicts preserve insertion order, so two runs evict the same
    entries in the same order and the cache never changes an answer, only how
    often one is recomputed. ``limit <= 0`` means unbounded.
    """
    if limit <= 0:
        return
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def _warn_once(bundle: Optional["_Selector"], message: str) -> None:
    """Warn, and record the note on the selector so a run can log it."""
    if bundle is not None:
        if message in bundle.notes:
            return
        bundle.notes.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- #
# Per-head context plan (D2b / D7)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _HeadContext:
    """One head group's context rendering instructions."""

    k: Any = _INHERIT  # int | "full" | "inherit"
    tag_recency: bool = False
    recency_buckets: int = 0
    recency_mode: str = "word"

    def is_inherited(self) -> bool:
        return self.k == _INHERIT and not self.tag_recency


def _parse_k(raw: Any, where: str) -> Any:
    if isinstance(raw, bool):  # bool is an int subclass; reject it explicitly
        raise ContractViolation(f"{where}: K must be an int, 'full' or 'inherit', got {raw!r}")
    if isinstance(raw, int):
        if raw < 0:
            raise ContractViolation(f"{where}: K must be >= 0, got {raw}")
        return int(raw)
    text = str(raw).strip().lower()
    if text in (_FULL, _INHERIT):
        return text
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ContractViolation(
            f"{where}: K must be an int, 'full' or 'inherit', got {raw!r}"
        ) from exc
    if parsed < 0:
        raise ContractViolation(f"{where}: K must be >= 0, got {parsed}")
    return parsed


def _resolve_head_plan(cfg: dict[str, Any]) -> dict[str, _HeadContext]:
    """Read ``select.head_context`` into one :class:`_HeadContext` per head."""
    block = get_dotted(cfg, "select.head_context")
    mode = str(get_dotted(cfg, "select.head_context.recency_mode")).lower()
    if mode not in ("word", "turn"):
        raise ValueError(
            f"select.head_context.recency_mode must be word|turn, got {mode!r}"
        )
    raw_plan = block.get("plan") or {}
    unknown = sorted(set(raw_plan) - set(_CONTEXT_HEADS))
    if unknown:
        raise ContractViolation(
            f"select.head_context.plan names heads that do not exist: {unknown}. "
            f"Known head groups: {list(_CONTEXT_HEADS)}"
        )
    plan: dict[str, _HeadContext] = {}
    for head in _CONTEXT_HEADS:
        entry = raw_plan.get(head) or {}
        if not isinstance(entry, dict):
            raise ContractViolation(
                f"select.head_context.plan.{head} must be a mapping, got {entry!r}"
            )
        plan[head] = _HeadContext(
            k=_parse_k(entry.get("K", _INHERIT), f"select.head_context.plan.{head}"),
            tag_recency=bool(entry.get("tag_recency", False)),
            recency_buckets=int(entry.get("recency_buckets", 0)),
            recency_mode=mode,
        )
    return plan


def _plan_fingerprint(plan: dict[str, _HeadContext]) -> dict[str, dict[str, Any]]:
    """The plan in the shape a checkpoint should record it (plain JSON types)."""
    return {
        head: {
            "K": spec.k,
            "tag_recency": bool(spec.tag_recency),
            "recency_buckets": int(spec.recency_buckets),
            "recency_mode": spec.recency_mode,
        }
        for head, spec in plan.items()
    }


def _inherit_plan() -> dict[str, _HeadContext]:
    return {head: _HeadContext() for head in _CONTEXT_HEADS}


def _split_context(context: ContextWindow) -> tuple[list[str], Optional[str]]:
    """Recover ``(turn_lines, state_line)`` from a :class:`ContextWindow`.

    ``reflex.data.build_context`` renders ``"\\n".join(turns + [state])``, so the
    state line is exactly the suffix after the joined turns. Recovering it by
    slicing -- rather than re-rendering it here -- keeps the state formatting
    owned by 4.1 and cannot drift from it. If the slice does not match (the
    layout changed), the caller falls back to ``context.text`` unmodified.
    """
    turns = list(context.turns)
    text = context.text or ""
    if not turns:
        return [], text or None
    prefix = "\n".join(turns) + "\n"
    if text.startswith(prefix):
        return turns, text[len(prefix) :]
    return turns, None


def _tag_turn(line: str, bucket: int, mode: str) -> str:
    """Prefix a rendered ``speaker|text`` turn with its recency bucket."""
    tag = f"r{bucket}"
    speaker, sep, body = line.partition("|")
    if not sep:
        speaker, body = "", line
    if mode == "turn":
        tagged = f"{tag}|{body}"
    else:  # "word" -- the representation D7 actually measured
        tagged = " ".join(f"{tag}|{token}" for token in body.split())
    return f"{speaker}|{tagged}" if sep else tagged


def _render_variant(
    context: ContextWindow,
    spec: _HeadContext,
    bundle: Optional["_Selector"],
) -> str:
    """Render the context string for one head group."""
    if spec.is_inherited():
        return context.text
    turns, state = _split_context(context)
    if state is None and turns:
        _warn_once(
            bundle,
            "select: ContextWindow.text is not 'turns + state line'; the per-head "
            "context plan cannot re-window it safely and every head is falling back "
            "to ContextWindow.text. reflex.data.build_context's layout changed.",
        )
        return context.text

    if spec.k == _FULL or spec.k == _INHERIT:
        chosen = list(turns)
    else:
        k = int(spec.k)
        if k > len(turns) and bundle is not None and bundle.context_k_is_truncated:
            _warn_once(
                bundle,
                f"select: head plan asks for the last {k} turns but data.context_turns_K "
                f"already truncated the window to {bundle.context_k}; the missing turns "
                "cannot be recovered from a ContextWindow. Widen data.context_turns_K.",
            )
        chosen = turns[max(0, len(turns) - k) :]

    if spec.tag_recency:
        buckets = max(0, int(spec.recency_buckets))
        n = len(chosen)
        chosen = [
            _tag_turn(line, min(n - 1 - i, buckets), spec.recency_mode)
            for i, line in enumerate(chosen)
        ]

    lines = list(chosen)
    if state:
        lines.append(state)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The selector bundle
# --------------------------------------------------------------------------- #


@dataclass
class _Selector:
    """The opaque handle :func:`build_selector` returns.

    Private by design: spec Section 4 gives :mod:`reflex.select` six public
    names and this is not one of them. Other modules receive it from
    :func:`build_selector` and hand it back to :func:`score_turn` /
    :func:`select_from_scores` / :func:`map_to_candidate`.
    """

    cfg: dict[str, Any]
    model: Any
    tokenizer: Any
    bank: Bank
    ontology: dict[str, Any]
    calibration: Calibration
    checkpoint_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    novelty_index: Any = None
    utterances: Sequence[str] = ()

    # class orders, taken from the model (which took them from the data)
    next_steps: list[str] = field(default_factory=list)
    subflows: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    skeleton_ids: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    value_list: list[str] = field(default_factory=list)

    # bank-derived lookups
    skeleton_acts: dict[str, list[str]] = field(default_factory=dict)
    template_index_by_act: dict[str, list[int]] = field(default_factory=dict)
    template_id_by_index: list[str] = field(default_factory=list)
    template_text_by_id: dict[str, str] = field(default_factory=dict)
    required_slots: dict[str, list[str]] = field(default_factory=dict)

    # ontology-derived lookups for H4
    enumerable: dict[str, list[str]] = field(default_factory=dict)
    value_by_action: dict[str, list[str]] = field(default_factory=dict)
    value_index: dict[str, int] = field(default_factory=dict)

    # inference knobs resolved once
    plan: dict[str, _HeadContext] = field(default_factory=_inherit_plan)
    requested_plan: dict[str, _HeadContext] = field(default_factory=_inherit_plan)
    head_context_active: bool = False
    context_k: Any = _FULL
    context_k_is_truncated: bool = False
    max_len: int = 0
    truncation_side: str = "left"
    infonce_temperature: float = 1.0
    multi_value_actions: frozenset = frozenset()
    slot_aliases: dict[str, str] = field(default_factory=dict)
    value_slots_source: str = "bank_then_union"
    value_restrict_to_action: bool = True
    score_inapplicable_heads: bool = False
    compose_join: str = " "
    normalizers: tuple = ()
    exact_match_similarity: float = 1.0
    cache_limit: int = 0

    # candidate mapping
    candidate_backend: str = "lexical"
    _candidate_embedder: Any = None
    _candidate_vectors: dict[str, Any] = field(default_factory=dict)

    notes: list[str] = field(default_factory=list)


def _as_selector(selector: Any) -> _Selector:
    if not isinstance(selector, _Selector):
        raise ContractViolation(
            "selector must be the handle returned by reflex.select.build_selector, "
            f"got {type(selector).__name__}"
        )
    return selector


def build_selector(
    cfg: dict[str, Any],
    checkpoint_path: str,
    bank: Bank,
    ontology: dict[str, Any],
    calibration: Calibration,
) -> Any:
    """Assemble the inference bundle: model, bank, template embeddings, novelty index.

    See :func:`reflex.contracts.build_selector` for the frozen contract.

    Everything expensive happens here, once: the checkpoint is loaded, the whole
    template bank is embedded with the model's FROZEN encoder copy (the contract
    forbids doing that per turn), the novelty index is read off disk, and the
    candidate utterance pool is loaded.

    The per-head context plan (D2b/D7) is resolved here too, and disabled -- with
    a warning -- when the checkpoint carries no record of having been trained
    under it. See the module docstring for why that is the safe default.
    """
    from reflex.calibrate import load_novelty_index
    from reflex.data import load_utterances
    from reflex.models import load_checkpoint

    torch = _torch()

    model, metadata = load_checkpoint(checkpoint_path, cfg, bank, ontology)
    model.eval()

    orders = model.class_orders
    bundle = _Selector(
        cfg=cfg,
        model=model,
        tokenizer=model.tokenizer,
        bank=bank,
        ontology=ontology,
        calibration=calibration,
        checkpoint_path=str(checkpoint_path),
        metadata=dict(metadata or {}),
        next_steps=list(orders["next_steps"]),
        subflows=list(orders["subflows"]),
        actions=list(orders["actions"]),
        skeleton_ids=list(orders["skeleton_ids"]),
        acts=list(orders["acts"]),
        value_list=list(orders["value_list"]),
    )

    # -- class-order sanity: the gate's conformal sets are defined on these --- #
    if tuple(bundle.next_steps) != tuple(NEXT_STEPS):
        raise ContractViolation(
            f"checkpoint nextstep order {bundle.next_steps} is not the normative "
            f"{list(NEXT_STEPS)}; cds_report branches on 0/1/2 with that meaning."
        )

    # -- bank lookups --------------------------------------------------------- #
    bundle.skeleton_acts = {s.skeleton_id: list(s.acts) for s in bank.skeletons}
    bundle.template_id_by_index = [t.template_id for t in bank.templates]
    bundle.template_text_by_id = {t.template_id: t.text_delex for t in bank.templates}
    index_of_template = {tid: i for i, tid in enumerate(bundle.template_id_by_index)}
    by_act: dict[str, list[int]] = {}
    for template in bank.templates:
        by_act.setdefault(template.act, []).append(index_of_template[template.template_id])
    bundle.template_index_by_act = by_act
    bundle.required_slots = {p.action: list(p.required_slots) for p in bank.actions}

    # -- ontology lookups for H4 ---------------------------------------------- #
    bundle.enumerable = dict(getattr(model, "enumerable", {}) or {})
    bundle.value_by_action = dict(getattr(model, "value_by_action", {}) or {})
    bundle.value_index = {value: i for i, value in enumerate(bundle.value_list)}

    # -- knobs ----------------------------------------------------------------- #
    bundle.max_len = int(get_dotted(cfg, "data.max_len"))
    bundle.truncation_side = str(
        _cfg_get(cfg, "train.truncation_side", "select.truncation_side")
    )
    bundle.infonce_temperature = float(get_dotted(cfg, "model.infonce_temperature"))
    bundle.multi_value_actions = frozenset(
        _cfg_get(cfg, "train.multi_value_actions", "select.multi_value_actions") or ()
    )
    bundle.slot_aliases = dict(get_dotted(cfg, "select.slot_category_aliases") or {})
    bundle.value_slots_source = str(get_dotted(cfg, "select.value_slots_source")).lower()
    if bundle.value_slots_source not in ("bank", "ontology_union", "bank_then_union"):
        raise ValueError(
            "select.value_slots_source must be bank|ontology_union|bank_then_union, "
            f"got {bundle.value_slots_source!r}"
        )
    bundle.value_restrict_to_action = bool(get_dotted(cfg, "select.value_restrict_to_action"))
    bundle.score_inapplicable_heads = bool(get_dotted(cfg, "select.score_inapplicable_heads"))
    bundle.compose_join = str(get_dotted(cfg, "select.compose_join"))
    bundle.normalizers = tuple(get_dotted(cfg, "select.candidate_normalizers") or ())
    bundle.exact_match_similarity = float(get_dotted(cfg, "select.exact_match_similarity"))
    bundle.cache_limit = int(get_dotted(cfg, "select.candidate_embed_cache"))
    bundle.candidate_backend = str(get_dotted(cfg, "select.candidate_match_backend")).lower()
    if bundle.candidate_backend not in ("lexical", "sentence_transformer"):
        raise ValueError(
            "select.candidate_match_backend must be lexical|sentence_transformer, "
            f"got {bundle.candidate_backend!r}"
        )

    raw_k = get_dotted(cfg, "data.context_turns_K")
    bundle.context_k = _parse_k(raw_k, "data.context_turns_K")
    bundle.context_k_is_truncated = isinstance(bundle.context_k, int)

    # -- per-head context plan, and the checkpoint-match guard ----------------- #
    bundle.requested_plan = _resolve_head_plan(cfg)
    bundle.plan = _effective_plan(bundle, cfg)

    # -- template embeddings: ONCE, here (contract) ---------------------------- #
    features = getattr(model, "template_features", None)
    n_templates = len(bundle.template_id_by_index)
    if features is None or int(getattr(features, "shape", [0])[0]) != n_templates:
        batch_size = int(get_dotted(cfg, "select.template_embed_batch_size"))
        with torch.no_grad():
            model.refresh_template_cache(batch_size=batch_size)

    # -- novelty index --------------------------------------------------------- #
    # A missing index does NOT default to distance 0.0 in silence: that reads as
    # "maximally familiar" and switches gate signal 2 off without saying so.
    required = bool(get_dotted(cfg, "select.require_novelty_index"))
    index_path = str(calibration.novelty_index_path or "")
    if not index_path:
        if required:
            raise ContractViolation(
                "Calibration.novelty_index_path is empty and select.require_novelty_index is "
                "true; reflex.calibrate must build and persist the novelty index before a "
                "selector can be built (spec 6.6 signal 2)."
            )
        _warn_once(
            bundle,
            "select: no novelty index (Calibration.novelty_index_path is empty) and "
            "select.require_novelty_index is false. novelty_distance is reported as 0.0 on "
            "every turn, so gate signal 2 never fires. Do not report novel_escalation_rate "
            "from such a run.",
        )
    else:
        try:
            bundle.novelty_index = load_novelty_index(index_path, cfg)
        except FileNotFoundError:
            if required:
                raise
            _warn_once(
                bundle,
                f"select: novelty index {index_path} is missing and "
                "select.require_novelty_index is false. novelty_distance is reported as 0.0 "
                "on every turn, so gate signal 2 never fires. Do not report "
                "novel_escalation_rate from such a run.",
            )
        query_dim = int(getattr(model, "query_dim", 0) or 0)
        if bundle.novelty_index is not None and query_dim and int(bundle.novelty_index.dim) != query_dim:
            raise ContractViolation(
                f"novelty index dim {bundle.novelty_index.dim} != model query dim {query_dim}. "
                f"The index at {index_path} was built with a different checkpoint; "
                "novelty_distance would be nonsense. Recalibrate."
            )

    bundle.utterances = load_utterances(cfg)
    return bundle


def _effective_plan(bundle: _Selector, cfg: dict[str, Any]) -> dict[str, _HeadContext]:
    """Apply ``select.head_context``'s enable flag and checkpoint-match guard."""
    requested = bundle.requested_plan
    if all(spec.is_inherited() for spec in requested.values()):
        bundle.head_context_active = False
        return _inherit_plan()

    if not bool(get_dotted(cfg, "select.head_context.enabled")):
        bundle.head_context_active = False
        bundle.notes.append(
            "select: per-head context (D2b/D7) is configured but "
            "select.head_context.enabled is false; every head reads ContextWindow.text."
        )
        return _inherit_plan()

    if not bool(get_dotted(cfg, "select.head_context.require_checkpoint_match")):
        bundle.head_context_active = True
        bundle.notes.append(
            "select: per-head context ACTIVE with require_checkpoint_match=false. The "
            "checkpoint's training context is unverified; if training used one global "
            "window (reflex.train does today) this is a train/test mismatch and the "
            "D2b/D7 gains do not transfer."
        )
        return requested

    key = str(get_dotted(cfg, "select.head_context.checkpoint_key"))
    recorded = (bundle.metadata.get("extra") or {}).get(key)
    if recorded is not None and recorded == _plan_fingerprint(requested):
        bundle.head_context_active = True
        return requested

    bundle.head_context_active = False
    _warn_once(
        bundle,
        "select: per-head context (DECISIONS D2b/D7 -- intent wants the full thread, "
        "nextstep/action want a k6 recency-tagged window) is configured but INACTIVE. "
        f"The checkpoint records no matching plan under metadata['extra'][{key!r}] "
        f"(found {recorded!r}), and reflex.train builds ONE context_text per example, "
        "so every head was trained on the data.context_turns_K window. Feeding a head a "
        "window it never saw subtracts accuracy instead of adding the measured gain. "
        "To activate: train per-head and record the plan in the checkpoint, or set "
        "select.head_context.require_checkpoint_match=false to run it as an ablation.",
    )
    return _inherit_plan()


# --------------------------------------------------------------------------- #
# Scoring one turn
# --------------------------------------------------------------------------- #


def _encode_batch(bundle: _Selector, texts: Sequence[str], extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One forward pass over one context string (batch of 1..n)."""
    torch = _torch()
    tokenizer = bundle.tokenizer
    tokenizer.truncation_side = bundle.truncation_side
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=bundle.max_len,
        return_tensors="pt",
    )
    batch: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }
    if extra:
        batch.update(extra)
    with torch.no_grad():
        return bundle.model(batch)


def _value_slots(bundle: _Selector, action: str) -> list[str]:
    """Which slots H4 is decoded for, under ``select.value_slots_source``."""
    bank_slots = list(bundle.required_slots.get(action, []))
    if bundle.value_slots_source == "bank":
        return bank_slots
    if bundle.value_slots_source == "bank_then_union" and bank_slots:
        return bank_slots
    # ontology_union: one slot standing for the whole action, which is what H4
    # was trained to predict (train.value_position='first' supervises one value).
    return ["*"] if bundle.value_by_action.get(action) else []


def _slot_columns(
    bundle: _Selector,
    action: str,
    slot: str,
    tokens: Sequence[str],
) -> tuple[list[int], list[str]]:
    """H4 columns and their value strings for one slot of ``action``.

    Two tiers, exactly the ones ``models.value_target_index`` uses:
    enumerable values occupy ``value_list`` columns; a non-enumerable category
    occupies the COPY column of its ``<marker>`` in the context tokens. A slot
    may draw on both (``name`` is an enumerable ``product`` category for
    make-purchase AND a ``<name>`` marker -- see DEFECTS_OPEN D-4).
    """
    n_values = len(bundle.value_list)
    columns: list[int] = []
    names: list[str] = []

    if slot == "*":
        categories = list(bundle.value_by_action.get(action, []))
    else:
        alias = bundle.slot_aliases.get(slot, slot)
        categories = [slot] if alias == slot else [slot, alias]

    for category in categories:
        for value in bundle.enumerable.get(category, []):
            column = bundle.value_index.get(value)
            if column is not None and column not in columns:
                columns.append(column)
                names.append(value)
        marker = f"<{category}>"
        if marker in tokens:
            column = n_values + list(tokens).index(marker)
            if column not in columns:
                columns.append(column)
                names.append(marker)
    return columns, names


def score_turn(selector: Any, context: ContextWindow, turn: NormalizedTurn, cfg: dict[str, Any]) -> SelectorScores:
    """Produce all head scores for one agent turn (spec 6.5 steps 1-3). NO decisions.

    See :func:`reflex.contracts.score_turn` for the frozen contract. Returns
    SOFTMAX PROBABILITIES over each head's canonical class order, because the
    gate's conformal sets are defined on probabilities.

    **No gold field is read.** Which heads are populated follows the PREDICTED
    nextstep, never ``turn.nextstep``: H3/H4 on a predicted ``take_action``,
    H5/H7 on a predicted ``retrieve_utterance``, per :class:`SelectorScores`'
    "empty when the turn is not ...". Computing the argmax needed to pick a
    branch, a skeleton's act positions or an action's value slots is arithmetic,
    not routing; :mod:`reflex.gate` remains the only decision point.

    One consequence worth knowing, because it biases calibration:
    ``calibrate._collect_dev_scores`` keeps an H3/H5/H7 row only when the head is
    populated, so a turn whose nextstep was mispredicted contributes nothing to
    those heads' conformal quantiles. Those are exactly the hard rows, so the
    quantiles come out optimistic (prediction sets too small, gate too
    permissive). Set ``select.score_inapplicable_heads: true`` to populate every
    head regardless of the predicted branch and remove the bias; the gate is
    unaffected either way because it reads applicability from the
    :class:`Selection`, not from emptiness.
    """
    bundle = _as_selector(selector)
    torch = _torch()

    # -- one rendered context per head group, one encoder pass per DISTINCT one #
    head_text = {head: _render_variant(context, bundle.plan[head], bundle) for head in _CONTEXT_HEADS}
    novelty_text = context.text  # never re-windowed; the index was built on this

    passes: dict[str, Any] = {}
    for text in list(head_text.values()) + [novelty_text]:
        if text not in passes:
            passes[text] = _encode_batch(bundle, [text])

    def _logits(head: str, key: str) -> list[float]:
        return [float(v) for v in passes[head_text[head]][key][0].tolist()]

    nextstep_probs = _softmax(_logits("nextstep", "nextstep_logits"))
    intent_probs = _softmax(_logits("intent", "intent_logits"))

    predicted_nextstep = NEXT_STEPS[_argmax(nextstep_probs)]
    want_action = predicted_nextstep == "take_action" or bundle.score_inapplicable_heads
    want_utterance = predicted_nextstep == "retrieve_utterance" or bundle.score_inapplicable_heads

    action_probs: list[float] = []
    value_probs: list[list[float]] = []
    value_candidates: list[list[str]] = []
    if want_action:
        action_probs = _softmax(_logits("action", "action_logits"))
        predicted_action = bundle.actions[_argmax(action_probs)]
        value_probs, value_candidates = _score_values(
            bundle, context, head_text["values"], predicted_action
        )

    skeleton_probs: list[float] = []
    template_probs: list[list[float]] = []
    template_candidates: list[list[str]] = []
    if want_utterance:
        skeleton_probs = _softmax(_logits("skeleton", "skeleton_logits"))
        skeleton_id = bundle.skeleton_ids[_argmax(skeleton_probs)]
        template_probs, template_candidates = _score_templates(
            bundle, passes[head_text["template"]]["query"], skeleton_id
        )

    # -- novelty: a SIGNAL, thresholded in the gate (boundary ruling 1) -------- #
    novelty_distance = 0.0
    if bundle.novelty_index is not None:
        query = passes[novelty_text]["query"]
        with torch.no_grad():
            vectors = query.detach().cpu().to(torch.float32).numpy()
        novelty_distance = 1.0 - float(bundle.novelty_index.max_cosine(vectors)[0])

    return SelectorScores(
        convo_id=int(context.convo_id),
        turn_index=int(context.turn_index),
        nextstep_probs=nextstep_probs,
        intent_probs=intent_probs,
        action_probs=action_probs,
        skeleton_probs=skeleton_probs,
        value_probs=value_probs,
        value_candidates=value_candidates,
        template_probs=template_probs,
        template_candidates=template_candidates,
        novelty_distance=novelty_distance,
        context_hash=str(context.context_hash),
    )


def _score_values(
    bundle: _Selector,
    context: ContextWindow,
    context_text: str,
    action: str,
) -> tuple[list[list[float]], list[list[str]]]:
    """H4: one renormalized distribution per value slot of the predicted action.

    The copy tier needs the official AST second input, so this runs one extra
    forward with ``context_input_ids`` built exactly as
    ``train.collate_batch`` builds it (``model.value_candidate_tokens`` ->
    ``convert_tokens_to_ids``).

    THE APPROXIMATION, stated rather than hidden: H4 has no position input and
    ``reflex.train`` supervises ``values[0]`` only (``train.value_position:
    first``), so the same distribution is restricted to each slot's candidate
    columns and renormalized. That gives the right ARITY for a multi-value
    button; it does not give three independently trained predictions.
    """
    torch = _torch()
    slots = _value_slots(bundle, action) if bundle.value_restrict_to_action else ["**raw**"]
    if not slots:
        return [], []

    context_texts = [
        piece.split("|", 1)[1] if "|" in piece else piece for piece in context.turns
    ]
    action_name = f"{action} a" if action in bundle.multi_value_actions else str(action)
    tokens = list(bundle.model.value_candidate_tokens(context_texts, action_name))

    extra: dict[str, Any] = {}
    if tokens:
        ids = bundle.tokenizer.convert_tokens_to_ids(tokens)
        extra["context_input_ids"] = torch.tensor([ids], dtype=torch.long)
        extra["context_attention_mask"] = torch.ones((1, len(ids)), dtype=torch.long)

    outputs = _encode_batch(bundle, [context_text], extra)
    logits = [float(v) for v in outputs["value_logits"][0].tolist()]
    expected = len(bundle.value_list) + int(get_dotted(bundle.cfg, "model.value_context_len"))
    if len(logits) != expected:
        raise ContractViolation(
            f"H4 emitted {len(logits)} columns but the official AST layout is "
            f"len(value_list)={len(bundle.value_list)} + model.value_context_len="
            f"{expected - len(bundle.value_list)}. A copy position would decode to the "
            "wrong value; the checkpoint and the config disagree."
        )

    if not bundle.value_restrict_to_action:
        # select.value_restrict_to_action=false: the raw head, unrestricted. Copy
        # columns with no token behind them are -inf and drop out of the softmax.
        n_values = len(bundle.value_list)
        names = list(bundle.value_list) + list(tokens)
        columns = list(range(n_values)) + [n_values + i for i in range(len(tokens))]
        return [_softmax([logits[c] for c in columns])], [names]

    probs: list[list[float]] = []
    candidates: list[list[str]] = []
    for slot in slots:
        columns, names = _slot_columns(bundle, action, slot, tokens)
        if not columns:
            continue
        probs.append(_softmax([logits[c] for c in columns]))
        candidates.append(names)
    return probs, candidates


def _score_templates(
    bundle: _Selector,
    query: Any,
    skeleton_id: str,
) -> tuple[list[list[float]], list[list[str]]]:
    """H7: one distribution over the act's templates, per act position of the skeleton.

    ``score_templates`` returns cosines; training's logits are
    ``cosine / model.infonce_temperature`` (``models.forward``), so the softmax
    is taken on that same scale or the probabilities would not be the ones the
    conformal quantiles were fitted to.
    """
    torch = _torch()
    acts = bundle.skeleton_acts.get(skeleton_id)
    if acts is None:
        raise ContractViolation(
            f"skeleton {skeleton_id!r} is in the checkpoint's class order but not in the "
            "bank handed to build_selector; the bank and the checkpoint disagree."
        )
    act_index = {act: i for i, act in enumerate(bundle.acts)}
    features = bundle.model.template_features

    probs: list[list[float]] = []
    candidates: list[list[str]] = []
    for act in acts:
        indices = bundle.template_index_by_act.get(act, [])
        if not indices:
            # An act with no template in the bank: no candidate can be offered.
            # The gate will see |set| = 0 and escalate (low_confidence), which is
            # the correct outcome -- do not invent a candidate here.
            probs.append([])
            candidates.append([])
            continue
        act_id = torch.tensor([act_index[act]], dtype=torch.long)
        with torch.no_grad():
            rows = torch.tensor(indices, dtype=torch.long)
            cosine = bundle.model.score_templates(query, act_id, features[rows])
        scores = [float(v) / bundle.infonce_temperature for v in cosine[0].tolist()]
        probs.append(_softmax(scores))
        candidates.append([bundle.template_id_by_index[i] for i in indices])
    return probs, candidates


# --------------------------------------------------------------------------- #
# Conformal prediction sets (spec 6.6 signal 1)
# --------------------------------------------------------------------------- #


def prediction_set(probs: Sequence[float], q: float) -> list[int]:
    """Return the conformal prediction set (spec 6.6 signal 1).

    See :func:`reflex.contracts.prediction_set` for the frozen contract.
    ``{i : probs[i] >= 1 - q}``, ascending. The set may be EMPTY -- that is a
    real outcome (``|set| != 1`` -> ``low_confidence``), not a bug to patch by
    inserting the argmax, which would silently break coverage.
    """
    threshold = 1.0 - float(q)
    return [index for index, prob in enumerate(probs) if float(prob) >= threshold]


# --------------------------------------------------------------------------- #
# Argmax reading of the scores (spec 6.5 steps 2-5)
# --------------------------------------------------------------------------- #


def select_from_scores(
    selector: Any,
    scores: SelectorScores,
    turn: NormalizedTurn,
    cfg: dict[str, Any],
) -> Selection:
    """Turn scores into the argmax selection (spec 6.5 steps 2-5). Still not routing.

    See :func:`reflex.contracts.select_from_scores` for the frozen contract.

    Branching follows the PREDICTED nextstep: ``take_action`` -> H3 then H4;
    ``retrieve_utterance`` -> H5 argmax skeleton, then top-1 template per act
    position, then the spec 6.5 step 5 candidate mapping;
    ``end_conversation`` -> nextstep and intent only. ``turn`` is read for
    ``candidates`` and nothing else.

    ``exact_template_match`` DEVIATES from the wording in
    :class:`~reflex.schemas.Selection`, which says "whether the GOLD utterance's
    delexicalized form equals ``composed_text_delex``". Selection cannot read
    gold without invalidating every number in the report, so the flag records
    what the mapping actually did: True when a candidate matched the composed
    text exactly (normalized), False when the nearest-cosine fallback was used,
    None when no utterance was composed. The gold-side version of the same
    question belongs to :mod:`reflex.evaluate`, which legitimately holds labels.
    """
    bundle = _as_selector(selector)

    if not scores.nextstep_probs:
        raise ContractViolation("SelectorScores.nextstep_probs is empty; score_turn must fill H1")
    nextstep = NEXT_STEPS[_argmax(scores.nextstep_probs)]
    if not scores.intent_probs:
        raise ContractViolation("SelectorScores.intent_probs is empty; score_turn must fill H2")
    intent = bundle.subflows[_argmax(scores.intent_probs)]

    action: Optional[str] = None
    values: Optional[list[str]] = None
    skeleton_id: Optional[str] = None
    template_ids: Optional[list[str]] = None
    composed: Optional[str] = None
    candidate_utt_id: Optional[int] = None
    candidate_rank = -1
    exact_match: Optional[bool] = None

    if nextstep == "take_action":
        if not scores.action_probs:
            raise ContractViolation(
                "predicted nextstep is take_action but SelectorScores.action_probs is empty; "
                "score_turn and select_from_scores disagree about the predicted branch."
            )
        action = bundle.actions[_argmax(scores.action_probs)]
        values = [
            candidates[_argmax(probs)]
            for probs, candidates in zip(scores.value_probs, scores.value_candidates)
            if probs and candidates
        ]

    elif nextstep == "retrieve_utterance":
        if not scores.skeleton_probs:
            raise ContractViolation(
                "predicted nextstep is retrieve_utterance but SelectorScores.skeleton_probs "
                "is empty; score_turn and select_from_scores disagree about the branch."
            )
        skeleton_id = bundle.skeleton_ids[_argmax(scores.skeleton_probs)]
        template_ids = [
            candidates[_argmax(probs)]
            for probs, candidates in zip(scores.template_probs, scores.template_candidates)
            if probs and candidates
        ]
        composed = bundle.compose_join.join(
            bundle.template_text_by_id[tid] for tid in template_ids
        )

        # spec 6.5 step 5: map the composed text onto the official candidate pool
        candidates_ids = list(turn.candidates or [])
        candidate_texts = delexicalize_candidates(
            candidates_ids, bundle.utterances, bundle.bank.slot_registry, cfg
        )
        # The similarity is not a Selection field; a caller that wants it calls
        # map_to_candidate itself.
        candidate_utt_id, candidate_rank = map_to_candidate(
            composed, candidates_ids, candidate_texts, bundle, cfg
        )[:2]
        if candidate_rank >= 0:
            normalized = _normalize_text(composed, bundle.normalizers)
            exact_match = _normalize_text(candidate_texts[candidate_rank], bundle.normalizers) == normalized
        else:
            exact_match = False

    return Selection(
        nextstep=nextstep,
        intent=intent,
        action=action,
        values=values,
        skeleton_id=skeleton_id,
        template_ids=template_ids,
        composed_text_delex=composed,
        candidate_utt_id=candidate_utt_id,
        candidate_rank=candidate_rank,
        exact_template_match=exact_match,
    )


# --------------------------------------------------------------------------- #
# Candidate delexicalization and mapping (spec 6.5 step 5)
# --------------------------------------------------------------------------- #


def _delex_fingerprint(registry: SlotRegistry, cfg: dict[str, Any]) -> str:
    """Cache key component: everything that changes a delexicalization result."""
    parts = [
        str(sorted(registry.slots)),
        str(sorted(registry.marker_to_slot.items())),
        str(get_dotted(cfg, "compile.guard_glued_markers")),
        str(get_dotted(cfg, "compile.min_literal_match_chars")),
        str(sorted(get_dotted(cfg, "compile.delex_regex_slots") or ())),
        str(sorted(get_dotted(cfg, "compile.delex_match_types") or ())),
        str(get_dotted(cfg, "compile.delex_ner_model")),
    ]
    return "|".join(parts)


def delexicalize_candidates(
    candidates: Sequence[int],
    utterances: Sequence[str],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> list[str]:
    """Delexicalize the 100 candidate utterances so they are comparable to templates.

    See :func:`reflex.contracts.delexicalize_candidates` for the frozen contract.

    ``utterances.json`` is LEXICALIZED, so a composed ``{order_id}`` template
    would score near zero against the raw candidate carrying ``"7654321"``.
    Delexicalization runs through :func:`reflex.compile.delexicalize_sentence`
    (4.2 owns delexicalization) with an EMPTY :class:`SlotSources`: a candidate
    is a pool string with no conversation attached, so only the two
    conversation-independent tiers can fire -- ABCD's own ``<marker>``
    conversion and the ``compile.delex_regex_slots`` shape regexes.

    Results are cached per (registry+config fingerprint, utterance id); the same
    ids recur across thousands of turns.
    """
    from reflex.compile import delexicalize_sentence

    fingerprint = _delex_fingerprint(registry, cfg)
    empty_sources = SlotSources()
    out: list[str] = []
    for raw_id in candidates:
        utt_id = int(raw_id)
        if utt_id < 0 or utt_id >= len(utterances):
            raise ContractViolation(
                f"candidate utterance id {utt_id} is outside the pool of {len(utterances)} "
                "utterances; candidates index utterances.json"
            )
        raw_text = str(utterances[utt_id])
        key = (fingerprint, raw_text)
        with _CANDIDATE_DELEX_LOCK:
            cached = _CANDIDATE_DELEX_CACHE.get(key)
        if cached is not None:
            out.append(cached)
            continue
        text_delex, _ = delexicalize_sentence(raw_text, empty_sources, registry, cfg)
        with _CANDIDATE_DELEX_LOCK:
            _CANDIDATE_DELEX_CACHE[key] = text_delex
            _evict(_CANDIDATE_DELEX_CACHE, int(get_dotted(cfg, "select.candidate_embed_cache")))
        out.append(text_delex)
    return out


def _lexical_vector(text: str) -> dict[str, float]:
    """Word 1-2 gram count vector. Deterministic, dependency-free, no training."""
    tokens = text.split()
    vector: dict[str, float] = {}
    for token in tokens:
        vector[token] = vector.get(token, 0.0) + 1.0
    for first, second in zip(tokens, tokens[1:]):
        key = f"{first}_{second}"
        vector[key] = vector.get(key, 0.0) + 1.0
    return vector


def _lexical_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    if dot == 0.0:
        return 0.0
    norm_left = math.sqrt(sum(w * w for w in left.values()))
    norm_right = math.sqrt(sum(w * w for w in right.values()))
    return float(dot / (norm_left * norm_right))


def _candidate_embedder(bundle: Optional[_Selector], cfg: dict[str, Any]) -> Any:
    """Load the sentence embedder used for candidate mapping, once, lazily.

    Reached only when ``select.candidate_match_backend`` is
    ``sentence_transformer``, which is NOT the default: measured on a synthetic
    near-miss probe (n=1000 -- right skeleton, one act's template swapped for a
    sibling, recover the true utterance from 100 candidates) the lexical word
    1-2gram cosine scores 51.6% recall@1 against all-MiniLM-L6-v2's 41.6%
    (paired McNemar chi2 = 35.3) at 1/14 the time. Candidate mapping is a
    near-duplicate match, not a paraphrase match.

    NOT the task encoder either, in any configuration: D12 measured raw
    ModernBERT-base putting 59% of ABCD sentence pairs at cosine >= 0.92 --
    "what is your membership level?" against "the shipping fee is $4.99 per
    item." scores 0.961.
    """
    if bundle is not None and bundle._candidate_embedder is not None:
        return bundle._candidate_embedder
    model_id = str(get_dotted(cfg, "select.candidate_embed_model"))
    if model_id in _EMBEDDERS:
        model = _EMBEDDERS[model_id]
        if bundle is not None:
            bundle._candidate_embedder = model
        return model
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        model.eval()
    except Exception as exc:  # pragma: no cover - depends on the local cache
        _warn_once(
            bundle,
            f"select: could not load select.candidate_embed_model {model_id!r} ({exc!r}); "
            "falling back to the lexical word 1-2 gram cosine for candidate mapping. "
            "The two backends do not rank ties identically.",
        )
        return None
    _EMBEDDERS[model_id] = model
    if bundle is not None:
        bundle._candidate_embedder = model
    return model


def _embed_texts(bundle: Optional[_Selector], model: Any, texts: Sequence[str], cfg: dict[str, Any]) -> Any:
    """Embed with caching per exact string; returns an ``(n, d)`` normalized array."""
    np = _numpy()
    cache = bundle._candidate_vectors if bundle is not None else {}
    missing = [text for text in dict.fromkeys(texts) if text not in cache]
    if missing:
        batch_size = int(get_dotted(cfg, "select.candidate_embed_batch_size"))
        vectors = model.encode(
            missing,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for text, vector in zip(missing, vectors):
            cache[text] = np.asarray(vector, dtype=np.float32)
        _evict(cache, int(get_dotted(cfg, "select.candidate_embed_cache")))
    return np.stack([cache[text] if text in cache else _encode_one(model, text, np) for text in texts])


def _encode_one(model: Any, text: str, np: Any) -> Any:
    """Re-embed a string the cache evicted mid-call. Same vector, just recomputed."""
    vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vector[0], dtype=np.float32)


def map_to_candidate(
    composed_text_delex: str,
    candidates: Sequence[int],
    candidate_texts_delex: Sequence[str],
    selector: Any,
    cfg: dict[str, Any],
) -> tuple[Optional[int], int, float]:
    """Map composed template text onto the official candidate pool (spec 6.5 step 5).

    See :func:`reflex.contracts.map_to_candidate` for the frozen contract.
    EXACT STRING MATCH WINS (after ``select.candidate_normalizers``); otherwise
    the highest cosine; ties break toward the LOWEST rank, so the mapping is
    deterministic.

    WHICH NUMBER THE EVALUATOR WANTS: the RANK. ``targets[4]`` in ABCD is a rank
    into this turn's ``candidates`` list, and ``utils/evaluate.py::ranking_report``
    compares ranks -- ``gold_utt_id = candidates[targets[4]]`` is the id, which is
    reported here too but is NOT what the official metric consumes. Pass
    ``candidate_rank`` to the evaluator and keep ``candidate_utt_id`` for
    human-readable logs.

    Returns ``(candidate_utt_id, candidate_rank, similarity)``; ``(None, -1,
    0.0)`` when there are no candidates.
    """
    bundle = selector if isinstance(selector, _Selector) else None
    if bundle is None and selector is not None:
        raise ContractViolation(
            "map_to_candidate's selector must come from build_selector (or be None to "
            f"force the dependency-free lexical backend), got {type(selector).__name__}"
        )
    if not candidates:
        return None, -1, 0.0
    if len(candidate_texts_delex) != len(candidates):
        raise ContractViolation(
            f"candidate_texts_delex has {len(candidate_texts_delex)} entries for "
            f"{len(candidates)} candidates; delexicalize_candidates returns one per candidate"
        )

    normalizers = (
        bundle.normalizers
        if bundle is not None
        else tuple(get_dotted(cfg, "select.candidate_normalizers") or ())
    )
    composed = _normalize_text(composed_text_delex or "", normalizers)
    normalized = [_normalize_text(text, normalizers) for text in candidate_texts_delex]

    # -- exact match wins, lowest rank first ---------------------------------- #
    exact_similarity = (
        bundle.exact_match_similarity
        if bundle is not None
        else float(get_dotted(cfg, "select.exact_match_similarity"))
    )
    for rank, text in enumerate(normalized):
        if text == composed:
            return int(candidates[rank]), rank, float(exact_similarity)

    if not composed:
        # Nothing was composed (no templates for the skeleton's acts). Refuse to
        # guess: rank 0 would be a silent fabrication the gate cannot see.
        return None, -1, 0.0

    backend = (
        bundle.candidate_backend
        if bundle is not None
        else str(get_dotted(cfg, "select.candidate_match_backend")).lower()
    )
    similarities: list[float]
    model = _candidate_embedder(bundle, cfg) if backend == "sentence_transformer" else None
    if model is not None:
        np = _numpy()
        matrix = _embed_texts(bundle, model, [composed] + normalized, cfg)
        similarities = [float(v) for v in np.asarray(matrix[1:] @ matrix[0])]
    else:
        query = _lexical_vector(composed)
        similarities = [_lexical_cosine(query, _lexical_vector(text)) for text in normalized]

    best = _argmax(similarities)
    return int(candidates[best]), int(best), float(similarities[best])
