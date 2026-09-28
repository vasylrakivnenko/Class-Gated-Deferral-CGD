"""THE FROZEN MODULE-BOUNDARY CONTRACT for REFLEXIVE v1.

Nine agents implement the twelve modules of spec Section 4 in parallel, without
talking to each other. This file is the only thing they share. Read it as law.

HOW TO IMPLEMENT A MODULE
-------------------------
Each ``src/reflex/<module>.py`` currently re-exports its functions from here so
the package imports and the CLI runs today. To implement module ``X``:

1. Open ``src/reflex/X.py``. It lists its own public names in ``__all__``.
2. Replace the ``from reflex.contracts import ...`` line with real
   implementations whose signatures are **byte-identical** to the ones below --
   same names, same parameter names, same order, same defaults, same return type.
3. Do not add a public name that is not in that module's ``__all__``, and do not
   implement a function this file assigns to a different module.
4. ``tests/test_contracts.py`` mechanically enforces 2 and 3. Run pytest.

If a signature here is genuinely wrong or impossible, SAY SO in your report and
propose the change. Do not silently diverge: a silent divergence is nine broken
integrations.

OWNERSHIP MAP (spec Section 4, MECE -- each function belongs to exactly ONE module)
----------------------------------------------------------------------------------
==== =============== =========================================================
4.1  ``data``        Reading files, normalizing keys, building splits, and
                     building the context window (spec 6.1 step 4). NO modeling.
4.2  ``compile``     Sentence splitting, delexicalization, act labeling,
                     template dedup, skeleton extraction, action-pattern
                     extraction, the slot registry (6.3). NO selector training.
 --  ``models``      Encoder + head *definitions* (6.4) and checkpoint I/O.
                     No training loop, no thresholds, no inference policy.
4.3  ``train``       Training the encoder and all heads. NO thresholding.
4.4  ``calibrate``   Computing gate thresholds on dev, and building the FAISS
                     novelty index. NO training.
4.5  ``select``      Inference producing scores (and the novelty DISTANCE, which
                     is a signal, not a decision). NO decision-making.
4.6  ``gate``        The ONLY place that decides fast-path vs escalate.
4.7  ``fill``        Turning a chosen template into text with values, and
                     reporting which slots are unavailable. NO selection.
4.8  ``llm_agent``   The ONLY place that calls an LLM at inference time.
4.9  ``evaluate``    The ONLY place metrics are computed. Wraps the official
                     ABCD scripts; spec 13 forbids reimplementing them.
4.10 ``report``      Rendering tables/figures from evaluator output.
                     NO computation of new numbers.
 --  ``run``         Orchestrates Arm A / Arm B over a split; owns run_id,
                     manifest and ``decisions.jsonl`` I/O.
 --  ``config``      Cross-cutting: loading ``configs/default.yaml`` and applying
                     ``--set`` overrides. (Not in spec 11.1; see README
                     "Deviations from the spec".)
==== =============== =========================================================

THREE BOUNDARY RULINGS, made here so nobody has to guess
--------------------------------------------------------
1. **Novelty distance** is computed in ``select`` and merely *thresholded* in
   ``gate``. The FAISS index is built and persisted by ``calibrate``. Rationale:
   4.6 says the gate DECIDES; computing a cosine is not deciding, and forcing
   the gate to hold an index would give it a second job.
2. **Missing-slot detection** lives in ``fill`` (``check_availability``), because
   4.7 owns slot logic. ``gate`` consumes the returned list and turns it into
   ``reason="unavailable_slot"``. ``fill`` never routes.
3. **Early-stopping score** lives in ``train`` as ``dev_selection_score`` and is
   explicitly NOT a reported metric. Everything in spec Section 8 is computed in
   ``evaluate`` and nowhere else.

HARD PROJECT RULES (not negotiable by any module)
-------------------------------------------------
* **ZERO PAID API CALLS.** ``llm_agent`` must be fully implemented but disabled
  whenever ``llm.enabled`` is false, and must raise :class:`LLMDisabledError`
  if invoked while disabled. No module may call a hosted inference endpoint by
  any other route.
* Act labeling defaults to the FREE LOCAL labeler
  (``compile.act_labeler: rules_plus_embed``). The ``llm`` labeler is behind the
  same :class:`ActLabeler` interface and is off by default.
* **No numeric threshold, model id, path or price in code** (spec 10). Read
  ``cfg``. A literal like ``0.92`` in a module body is a review failure.
* **Determinism:** same context -> same Decision, always (spec 10).
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional, Protocol, Sequence, runtime_checkable

from reflex.schemas import (
    ActionPattern,
    Bank,
    Calibration,
    ContextWindow,
    Decision,
    EvalRecord,
    GateOutput,
    LLMDecision,
    NormalizedTurn,
    Partitions,
    RunManifest,
    Selection,
    SelectorScores,
    Skeleton,
    SlotRegistry,
    SlotSources,
    SlotSpec,
    Template,
    TurnLabel,
)

__all__ = [
    # Errors
    "ReflexError",
    "LLMDisabledError",
    "BudgetExceededError",
    "PlaceholderConfigError",
    "ContractViolation",
    # Protocols
    "ActLabeler",
    "LLMBackend",
    "NoveltyIndex",
    "OfficialMetrics",
    # Module ownership table (used by tests/test_contracts.py)
    "MODULE_FUNCTIONS",
    # 0. config
    "load_config",
    "apply_overrides",
    "resolve_path",
    "require_filled",
    # 4.1 data
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
    # 4.2 compile
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
    # models
    "build_encoder",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
    # 4.3 train
    "train",
    "build_train_examples",
    "collate_batch",
    "dev_selection_score",
    # 4.4 calibrate
    "nonconformity_scores",
    "conformal_quantile",
    "build_novelty_index",
    "save_novelty_index",
    "load_novelty_index",
    "calibrate",
    "write_calibration",
    "load_calibration",
    # 4.5 select
    "build_selector",
    "score_turn",
    "prediction_set",
    "select_from_scores",
    "map_to_candidate",
    "delexicalize_candidates",
    # 4.6 gate
    "evaluate_gate",
    # 4.7 fill
    "collect_slot_sources",
    "check_availability",
    "fill_template",
    "compose_utterance",
    # 4.8 llm_agent
    "load_prompt",
    "render_agent_prompt",
    "parse_llm_json",
    "build_llm_agent",
    "llm_decide",
    "estimate_prompt_tokens",
    "llm_cost_usd",
    # 4.9 evaluate
    "load_official_metrics",
    "build_ast_arrays",
    "build_cds_arrays",
    "ast_metrics",
    "cds_metrics",
    "routing_metrics",
    "fastpath_metrics",
    "novelty_metrics",
    "calibration_metrics",
    "cost_latency_metrics",
    "bootstrap_ci",
    "mcnemar_pvalue",
    "parity_delta",
    "evaluate_run",
    "verdict",
    # 4.10 report
    "render_report",
    "render_table",
    "render_figure",
    # run
    "new_run_id",
    "write_manifest",
    "read_manifest",
    "append_decisions",
    "read_decisions",
    "run_arm",
    "run_sweep",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReflexError(RuntimeError):
    """Base class for every REFLEXIVE error. Catch this at the CLI boundary."""


class LLMDisabledError(ReflexError):
    """Raised when the LLM agent is invoked while ``llm.enabled`` is false.

    This is the paid-call kill switch. It is an ERROR, not a warning and not a
    silent no-op: a silent no-op would let Arm A "run" and produce a fake
    baseline. The message must name the config key so the operator knows the one
    thing they would have to change to start spending money.
    """


class BudgetExceededError(ReflexError):
    """Raised when a run's cumulative LLM spend would exceed ``llm.budget_usd_per_run``.

    Spec 10: "a hard budget cap per run in config aborts when exceeded." Abort
    before the call, not after.
    """


class PlaceholderConfigError(ReflexError):
    """Raised when code reads a config value still set to a ``<fill: ...>`` placeholder.

    ``llm.strong``, ``llm.cheap``, ``llm.provider``, ``llm.prices_usd_per_million``
    and ``llm.price_list_date`` ship unfilled on purpose. Never substitute a
    default; raise this instead.
    """


class ContractViolation(ReflexError):
    """Raised when a module's output breaks an invariant this file declares.

    Examples: a probability vector whose length is not the head's class count; a
    ``Template.slots`` entry absent from the slot registry; a
    ``candidate_rank`` outside ``[-1, 100)``.
    """


# ---------------------------------------------------------------------------
# Protocols (swappable implementations behind one interface)
# ---------------------------------------------------------------------------


@runtime_checkable
class ActLabeler(Protocol):
    """Spec 6.2 step 3a. Assigns one act from the 9-act inventory to each sentence.

    Two implementations live in :mod:`reflex.compile`, selected by
    ``compile.act_labeler``:

    * ``rules_plus_embed`` -- **THE DEFAULT AND THE ONLY ONE ENABLED.** Free,
      local, deterministic: cue-phrase rules first, then nearest-centroid over
      embeddings from ``compile.act_labeler_embed_model`` for sentences no rule
      fires on.
    * ``llm`` -- spec 6.2 step 3a's original design (fixed prompt
      ``prompts/act_labeling.txt``, temperature 0, batches of 50). Implemented
      behind this same interface; raises :class:`LLMDisabledError` unless
      ``llm.enabled`` is true.

    An implementation MUST be deterministic: the same sentence list must yield
    the same labels on every call and in every process.
    """

    name: str

    def label(self, sentences: Sequence[str]) -> list[str]:
        """Return one act per input sentence, same length and order as input.

        Args:
            sentences: Delexicalized sentences to label.

        Returns:
            Acts from ``ACT_INVENTORY``. ``"OTHER"`` is the fallback; never
            return a label outside the inventory.
        """
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """The single seam through which any hosted model could ever be called.

    Implementing this against a paid provider is allowed; CALLING it while
    ``llm.enabled`` is false is not. :func:`llm_decide` enforces the switch
    before a backend is ever reached.
    """

    model_id: str

    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int, int]:
        """Return ``(text, tokens_in, tokens_out)`` for one completion."""
        ...


@runtime_checkable
class NoveltyIndex(Protocol):
    """A nearest-neighbour index over train context vectors (spec 6.6 signal 2).

    Backed by ``faiss.IndexFlatIP`` over L2-NORMALIZED vectors, so inner product
    IS cosine. Built by :mod:`reflex.calibrate`, queried by :mod:`reflex.select`.
    """

    dim: int
    size: int

    def max_cosine(self, vectors: Any) -> list[float]:
        """Return the max cosine to any indexed train vector, per query row.

        Args:
            vectors: ``(n, dim)`` float32 array. Implementations must normalize
                defensively rather than trusting the caller.

        Returns:
            ``n`` cosines in ``[-1, 1]``. ``novelty_distance = 1 - cosine``.
        """
        ...


@runtime_checkable
class OfficialMetrics(Protocol):
    """Handle on the four official ABCD report functions. See :func:`load_official_metrics`.

    Attributes hold the REAL functions from the vendored
    ``<eval.official_utils_dir>/utils/evaluate.py``. Spec 13 forbids
    reimplementing them, so this is the only legitimate way in.
    """

    ast_report: Any
    cds_report: Any
    ranking_report: Any
    task_completion_report: Any


# ---------------------------------------------------------------------------
# 0. config  (cross-cutting; src/reflex/config.py)
# ---------------------------------------------------------------------------


def load_config(path: str = "configs/default.yaml", overrides: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Load the YAML config and apply ``--set`` overrides.

    Spec 10 forbids thresholds, model ids, paths and prices in code, so this is
    the single entry point every module uses to learn any of them.

    Args:
        path: Path to the YAML config, relative to the repo root or absolute.
        overrides: ``"dotted.key=value"`` strings from ``--set``, applied in
            order after the file is read.

    Returns:
        The fully resolved config as nested plain dicts. Values are NOT
        validated against ``<fill: ...>`` here -- call :func:`require_filled` at
        the point of use, so that a fully free run never trips over an unfilled
        LLM price.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if an override is not ``key=value`` or names a key whose
            parent path does not exist in the config.
    """
    raise NotImplementedError


def apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a new config with ``"dotted.key=value"`` overrides applied.

    Values are parsed as YAML scalars, so ``gate.alpha=0.05`` yields a float,
    ``llm.enabled=true`` a bool, and ``train.seeds=[1,2]`` a list. Overriding a
    key that does not already exist is an error: it is almost always a typo, and
    a typo that silently adds a dead key is worse than a crash.

    Args:
        cfg: Config to override. Not mutated.
        overrides: ``"dotted.key=value"`` strings.

    Returns:
        A deep-copied config with the overrides applied.

    Raises:
        ValueError: on a malformed override or an unknown key path.
    """
    raise NotImplementedError


def resolve_path(cfg: dict[str, Any], key: str) -> str:
    """Resolve a dotted config key holding a path into an absolute path.

    Relative paths resolve against the repo root (the parent of ``src/``), NOT
    against the process cwd, so a run started from anywhere writes to the same
    place.

    Args:
        cfg: Resolved config.
        key: Dotted key, e.g. ``"paths.bank_dir"`` or ``"data.abcd_dir"``.

    Returns:
        An absolute filesystem path. Does not create it.

    Raises:
        KeyError: if the key is absent.
    """
    raise NotImplementedError


def require_filled(cfg: dict[str, Any], key: str) -> Any:
    """Return ``cfg[key]``, refusing to return an unfilled placeholder.

    Args:
        cfg: Resolved config.
        key: Dotted key, e.g. ``"llm.strong"``.

    Returns:
        The value.

    Raises:
        PlaceholderConfigError: if the value is a string containing ``"<fill"``,
            or a dict/list containing one at any depth.
        KeyError: if the key is absent.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.1 data  (src/reflex/data.py) -- reading, normalizing, splitting, context
# ---------------------------------------------------------------------------


def load_raw_abcd(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read ``abcd_v<version>.json`` and return it unmodified.

    The file is ~116 MB unzipped; load it once per process and pass it around.

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def load_kb(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read ``kb.json``: 55 entries keyed by subflow name.

    Its key set is exactly :func:`subflow_list`'s output (verified). Used by the
    gate's ``unseen_action`` signal (spec 6.6 signal 3).

    Args:
        cfg: Resolved config. Uses ``data.abcd_dir``.

    Returns:
        ``subflow -> allowed/expected actions``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

    Args:
        turns: All turns of the conversation, from :func:`normalize_conversation`.
        turn_index: The agent turn being predicted. Only turns before it are read.
        scenario: The raw ``convo["scenario"]`` dict, consulted ONLY to decide
            which of its fields have been disclosed.
        cfg: Resolved config. Uses ``data.context_turns_K``.

    Returns:
        A :class:`ContextWindow` with ``context_hash`` populated.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.2 compile  (src/reflex/compile.py) -- the BANK, and the slot registry (6.3)
# ---------------------------------------------------------------------------


def build_slot_registry(ontology: dict[str, Any], cfg: dict[str, Any]) -> SlotRegistry:
    """Build the slot registry (spec 6.3).

    **ABCD's own ``<slot>`` vocabulary is the BACKBONE.** It is exactly
    ``ontology["values"]["non_enumerable"]`` flattened -- 11 names across 3
    categories: product (``amount``, ``name``), personal (``account_id``,
    ``email``, ``phone``, ``pin_number``, ``username``), order (``full_address``,
    ``order_id``, ``street_address``, ``zip_code``) -- and it is the same list
    the official ``utils/load.py`` adds to its tokenizer as
    ``[f"<{slot}>" for ...]``. 21,297 such markers already exist in the corpus.
    Register each with ``abcd_marker`` set, and DO NOT invent a parallel name for
    any entity ABCD already marks (no ``customer_email`` next to ``email``).

    Extend the registry only for entities ABCD leaves literal -- dates,
    product names, promo codes, counts -- and only with SPECIFIC names
    (``arrival_date``, not ``date``; spec 6.3 and acceptance item 5).

    Args:
        ontology: From :func:`load_ontology`.
        cfg: Resolved config. Uses ``compile.slot_registry_backbone``.

    Returns:
        A :class:`SlotRegistry` whose ``marker_to_slot`` covers all 11 ABCD
        markers.

    Raises:
        ContractViolation: if a generic slot name (``date``, ``value``, ``id``,
            ``name`` alone is permitted only because ABCD defines ``<name>``)
            would be registered for a REFLEX-added slot.
    """
    raise NotImplementedError


def split_sentences(text: str, cfg: dict[str, Any]) -> list[str]:
    """Split one agent utterance into sentences (spec 6.2 step 1).

    Rule-based only -- ``pysbd`` per ``compile.sentence_splitter``. The splitter
    must not break inside a ``<slot>`` marker or a ``{slot}`` placeholder.

    Args:
        text: One agent utterance (already partially delexicalized by ABCD).
        cfg: Resolved config. Uses ``compile.sentence_splitter``.

    Returns:
        Non-empty, whitespace-stripped sentences in order. Returns ``[]`` for
        blank input.
    """
    raise NotImplementedError


def delexicalize_sentence(
    sentence: str,
    sources: SlotSources,
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Delexicalize one sentence and report what was replaced (spec 6.2 step 2).

    Order of attempts (spec 6.2 step 2b): exact match against scenario values and
    ontology value lists; then regex for ids, emails, phone numbers, money and
    dates; then a small local NER model for names. **No hosted model, ever** --
    the NER step is a local pipeline or it is skipped.

    Two ABCD-specific rules:

    1. **Convert, do not re-derive.** ``<order_id>`` in the input becomes
       ``{order_id}`` in the output via ``registry.marker_to_slot``. ABCD already
       did that work for 21,297 spans; re-detecting them invites a second,
       disagreeing convention.
    2. **Guard glued markers.** 357 of those 21,297 markers are stuck to
       adjacent alphanumerics because ABCD's own delexicalizer did partial
       substring replacement -- real examples: ``"$1<amount>"`` for ``$164``,
       ``"<order_id>3"``, ``"4<street_address>"``, ``"$4.<amount>"`` for ``$4.99``.
       Such a span is NOT a clean slot and is NOT reversible by the filler. When
       ``compile.guard_glued_markers`` is true, leave the glued text as literal
       surface text and do not emit a slot for it. Silently emitting
       ``{amount}`` there would make the filler produce ``"$1123"``.

    Args:
        sentence: One sentence from :func:`split_sentences`.
        sources: Values available in this conversation, for exact matching.
        registry: The slot registry -- the ONLY source of slot names.
        cfg: Resolved config. Uses ``compile.guard_glued_markers``.

    Returns:
        ``(text_delex, slot_to_surface)`` where ``text_delex`` uses ``{slot}``
        braces and never contains ``<``, and ``slot_to_surface`` maps each
        emitted slot name to the surface value it replaced (spec 6.2 step 2b's
        per-turn mapping, stored in this direction because that is how
        :mod:`reflex.fill` reads it).

    Raises:
        ContractViolation: if a slot name not in ``registry`` would be emitted.
    """
    raise NotImplementedError


def get_act_labeler(cfg: dict[str, Any]) -> ActLabeler:
    """Return the configured act labeler (spec 6.2 step 3a).

    ``compile.act_labeler`` selects it: ``rules_plus_embed`` (free, local,
    DEFAULT) or ``llm``. The ``llm`` branch must additionally check
    ``llm.enabled`` and raise :class:`LLMDisabledError` if false -- checked here,
    at construction, so a disabled run fails before it spends an hour compiling.

    Args:
        cfg: Resolved config.

    Returns:
        An :class:`ActLabeler`.

    Raises:
        ValueError: on an unknown labeler name.
        LLMDisabledError: for ``llm`` while ``llm.enabled`` is false.
    """
    raise NotImplementedError


def label_acts(sentences: Sequence[str], cfg: dict[str, Any]) -> list[str]:
    """Label every train sentence with an act (spec 6.2 step 3a), via :func:`get_act_labeler`.

    Args:
        sentences: All delexicalized train sentences.
        cfg: Resolved config.

    Returns:
        One act per sentence, aligned by index.
    """
    raise NotImplementedError


def extract_templates(
    labeled_sentences: Sequence[tuple[str, str, str]],
    cfg: dict[str, Any],
) -> list[Template]:
    """Build raw :class:`Template` objects, one per distinct ``(act, text_delex)``.

    No dedup beyond exact match and no count filtering here -- that is
    :func:`dedup_templates`.

    Args:
        labeled_sentences: ``(turn_id, act, text_delex)`` triples for every
            sentence of every train retrieve_utterance turn.
        cfg: Resolved config. Uses ``compile.max_example_context_ids``.

    Returns:
        Templates with ``count``, ``slots`` (order of first appearance),
        ``surface_forms`` (just the canonical form at this stage) and
        ``example_context_ids`` populated. ``template_id`` is assigned as
        ``f"T{i:06d}"`` over templates sorted by descending count then
        ``text_delex``, so ids are reproducible across runs.
    """
    raise NotImplementedError


def dedup_templates(
    templates: Sequence[Template],
    cfg: dict[str, Any],
) -> list[Template]:
    """Merge near-duplicate templates and drop rare ones (spec 6.2 step 4).

    Normalize (lowercase, strip punctuation spacing); merge exact duplicates;
    then single-linkage merge at cosine ``>= compile.dedup_threshold`` **within
    the same act only**; canonical surface form is the most frequent, with
    ``text_delex`` alphabetically first among ties so the choice is
    deterministic. Then drop templates with ``count < compile.min_template_count``.

    Merging across acts is forbidden -- two identical sentences labeled ACK and
    CONFIRM are different bank entries, because the skeleton refers to acts.

    Args:
        templates: From :func:`extract_templates`.
        cfg: Resolved config. Uses ``compile.dedup_threshold``,
            ``compile.min_template_count`` and ``model.encoder`` for embeddings.

    Returns:
        Merged templates with summed ``count`` and unioned ``surface_forms``.
        ``template_id`` is REASSIGNED after merging, so ids are only meaningful
        within one bank.
    """
    raise NotImplementedError


def extract_skeletons(
    utterance_acts: Sequence[Sequence[str]],
    cfg: dict[str, Any],
) -> list[Skeleton]:
    """Extract skeletons: the ordered act tuple of each agent utterance (spec 6.2 step 5).

    Keep ALL observed skeletons -- spec 6.2 step 5 is explicit that there is no
    minimum count, because "rare skeletons are simply unlikely to be chosen".

    Args:
        utterance_acts: One act sequence per train retrieve_utterance turn.
        cfg: Resolved config.

    Returns:
        One :class:`Skeleton` per distinct act tuple, ``skeleton_id`` assigned as
        ``f"S{i:04d}"`` over skeletons sorted by descending count then act tuple.
    """
    raise NotImplementedError


def extract_action_patterns(
    action_turns: Sequence[NormalizedTurn],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> list[ActionPattern]:
    """Extract one :class:`ActionPattern` per button seen in train (spec 6.2 step 6).

    ``required_slots`` are the slot-registry names of the value positions
    actually filled in train. Observed ``len(values)`` on action turns is 0
    (11,327 turns), 1 (19,057) or 3 (6,098), so a button's arity is not fixed by
    the ontology -- take the MODAL arity per button and record slots for those
    positions.

    Args:
        action_turns: Train turns with ``nextstep == "take_action"``.
        registry: The slot registry, for typing the observed values.
        cfg: Resolved config.

    Returns:
        One pattern per observed button, ordered by :func:`action_list`.
    """
    raise NotImplementedError


def compile_bank(cfg: dict[str, Any], fraction: float = 1.0) -> Bank:
    """Run the whole compiler (spec 6.2) on the TRAIN split only and return the bank.

    Train-only is a hard boundary: compiling over dev or test would leak the
    answer into the bank, and the bank is what the selector chooses from.

    Args:
        cfg: Resolved config.
        fraction: Learning-curve fraction of train to compile from (spec 3.4 /
            E4). ``1.0`` for the headline bank.

    Returns:
        A :class:`Bank` with ``source_fraction`` and ``bank_hash`` set.
    """
    raise NotImplementedError


def write_bank(bank: Bank, cfg: dict[str, Any]) -> dict[str, str]:
    """Persist the bank to the four spec 6.2 step 7 files.

    ``bank/templates.jsonl``, ``bank/skeletons.jsonl``, ``bank/actions.jsonl``,
    ``bank/slot_registry.json`` under ``paths.bank_dir``. JSONL rows are written
    in ``template_id`` / ``skeleton_id`` / action order, with sorted JSON keys, so
    the files are byte-stable and ``bank_hash`` is meaningful.

    Args:
        bank: The bank to write.
        cfg: Resolved config. Uses ``paths.bank_dir``.

    Returns:
        ``logical_name -> absolute path`` for the four files written.
    """
    raise NotImplementedError


def load_bank(cfg: dict[str, Any]) -> Bank:
    """Load the bank written by :func:`write_bank`, rebuilding ``templates_by_act``.

    Args:
        cfg: Resolved config. Uses ``paths.bank_dir``.

    Returns:
        The :class:`Bank`.

    Raises:
        FileNotFoundError: if the bank has not been compiled.
        ContractViolation: if any ``Template.slots`` entry is absent from the
            registry (spec 14 acceptance item 5, checked on load so it cannot be
            skipped).
    """
    raise NotImplementedError


def write_turn_labels(labels: Sequence[TurnLabel], split: str, cfg: dict[str, Any]) -> str:
    """Persist per-turn labels to ``labels/<split>.jsonl`` (spec 6.2 step 7).

    Args:
        labels: One per agent-side turn of the split.
        split: e.g. ``"train"``.
        cfg: Resolved config. Uses ``paths.labels_dir``.

    Returns:
        The absolute path written.
    """
    raise NotImplementedError


def load_turn_labels(split: str, cfg: dict[str, Any]) -> list[TurnLabel]:
    """Load ``labels/<split>.jsonl``.

    Args:
        split: e.g. ``"train"``.
        cfg: Resolved config. Uses ``paths.labels_dir``.

    Returns:
        The labels in file order.
    """
    raise NotImplementedError


def write_delex_check(bank: Bank, cfg: dict[str, Any]) -> str:
    """Write the delexicalization quality sample to ``outputs/compile/delex_check.md``.

    Spec 6.2 step 2c: sample ``compile.delex_check_sample`` (200) sentences and
    lay them out for manual review; the acceptance bar is >= 98% correct masking.
    This function WRITES THE SAMPLE AND THE TALLY TEMPLATE; it does not invent a
    score. A machine-generated "98%" would defeat the purpose of a manual review.

    Args:
        bank: The compiled bank.
        cfg: Resolved config. Uses ``compile.delex_check_sample``.

    Returns:
        The absolute path written.
    """
    raise NotImplementedError


def write_act_check(bank: Bank, cfg: dict[str, Any]) -> str:
    """Write the act-labeling agreement sample to ``outputs/compile/act_check.md``.

    Spec 6.2 step 3b: 300 random labels for human checking. Same rule as
    :func:`write_delex_check` -- lay out the sample, do not fabricate the
    agreement number.

    Args:
        bank: The compiled bank.
        cfg: Resolved config. Uses ``compile.act_check_sample``.

    Returns:
        The absolute path written.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# models  (src/reflex/models.py) -- definitions only (spec 6.4)
# ---------------------------------------------------------------------------


def build_encoder(cfg: dict[str, Any], size: str = "base") -> Any:
    """Instantiate the context encoder (spec 6.4).

    Tries ``model.encoder``, falls back to ``model.fallback_encoder`` if the
    first will not load, and uses ``model.small_encoder`` when ``size ==
    "small"`` (ablation E3c). The fallback that actually fired must be recorded
    in the manifest -- a silent encoder swap would make two runs
    incomparable.

    Args:
        cfg: Resolved config.
        size: ``"base"`` or ``"small"``.

    Returns:
        ``(encoder, tokenizer, resolved_model_id)``. Typed ``Any`` so this
        contract does not import torch.

    Raises:
        ValueError: on an unknown ``size``.
    """
    raise NotImplementedError


def build_model(cfg: dict[str, Any], bank: Bank, ontology: dict[str, Any], size: str = "base") -> Any:
    """Build the multi-task model: shared encoder, query MLP, heads H1-H7 (spec 6.4).

    Head output widths come from the DATA, never from a literal: H1 =
    ``len(NEXT_STEPS)``, H2 = ``len(subflow_list(ontology))``, H3 =
    ``len(action_list(ontology))``, H5 = ``len(bank.skeletons)``, H6 =
    ``len(ACT_INVENTORY)``. H4 and H7 are scoring heads over per-turn candidate
    sets, not fixed-width classifiers. H7 scores
    ``cosine(W_act[act] @ q, e_t)``; template embeddings come from a FROZEN copy
    of the encoder, refreshed once per epoch.

    Args:
        cfg: Resolved config. Uses ``model.query_mlp``.
        bank: Needed for the skeleton head width and the template inventory.
        ontology: Needed for the intent and action head widths.
        size: ``"base"`` or ``"small"``.

    Returns:
        An ``nn.Module`` exposing ``encode_context``, ``encode_templates``,
        ``score_templates`` and ``forward``.
    """
    raise NotImplementedError


def save_checkpoint(model: Any, cfg: dict[str, Any], seed: int, extra: Optional[dict[str, Any]] = None) -> str:
    """Save a training checkpoint under ``train.checkpoint_dir``.

    The checkpoint must embed everything needed to reproduce inference class
    orders -- resolved encoder id, subflow list, action list, skeleton ids, act
    inventory, seed -- so that a calibration produced against it cannot be
    silently paired with a different head ordering.

    Args:
        model: The trained model.
        cfg: Resolved config.
        seed: The training seed.
        extra: Extra provenance to store alongside.

    Returns:
        The absolute checkpoint path.
    """
    raise NotImplementedError


def load_checkpoint(path: str, cfg: dict[str, Any], bank: Bank, ontology: dict[str, Any]) -> Any:
    """Load a checkpoint into a freshly built model, in eval mode.

    Args:
        path: Checkpoint path from :func:`save_checkpoint`.
        cfg: Resolved config.
        bank: Must be the same bank the checkpoint was trained against.
        ontology: The ontology.

    Returns:
        ``(model, checkpoint_metadata)``.

    Raises:
        ContractViolation: if the checkpoint's stored class orders disagree with
            the ones derived from ``bank`` and ``ontology``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.3 train  (src/reflex/train.py) -- no thresholding
# ---------------------------------------------------------------------------


def build_train_examples(
    partitions: Partitions,
    bank: Bank,
    labels: Sequence[TurnLabel],
    cfg: dict[str, Any],
    split: str = "train",
) -> list[dict[str, Any]]:
    """Assemble training examples: one per agent-side turn.

    Each example carries the rendered context string, the gold class INDEX for
    every applicable head (using the canonical class orders), the per-act gold
    template ids, and the H4 candidate value set. Head losses are masked where
    the head does not apply (spec 6.4: H3/H4 only on take_action, H5/H7 only on
    retrieve_utterance).

    Args:
        partitions: From :func:`build_partitions`.
        bank: The compiled bank.
        labels: From :func:`load_turn_labels`.
        cfg: Resolved config.
        split: Which partition to build from.

    Returns:
        Examples in deterministic order (conversation, then turn).
    """
    raise NotImplementedError


def collate_batch(examples: Sequence[dict[str, Any]], cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Collate examples into padded tensors, including H7 hard negatives.

    Hard negatives per spec 6.4 H7: ``model.hard_negatives_per_positive`` (10)
    other templates OF THE SAME ACT nearest the gold one, plus in-batch
    negatives for the InfoNCE term. Negative selection must be seeded from the
    example index, not from global RNG state, or batches stop being reproducible.

    Args:
        examples: From :func:`build_train_examples`.
        cfg: Resolved config. Uses ``data.max_len``.
        tokenizer: From :func:`build_encoder`.

    Returns:
        A batch dict of tensors plus per-head loss masks.
    """
    raise NotImplementedError


def dev_selection_score(model: Any, dev_examples: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, float]:
    """Compute the EARLY-STOPPING signal only (spec 6.4: dev H7 recall@1 + H3 accuracy).

    **This is not a reported metric.** Spec 4.9 makes :mod:`reflex.evaluate` the
    only place metrics are computed; nothing here may be quoted in the report.
    Kept in ``train`` because it is part of the training loop's control flow.

    Args:
        model: The model being trained.
        dev_examples: Dev examples.
        cfg: Resolved config.

    Returns:
        ``{"h7_recall_at_1", "h3_accuracy", "score"}`` where ``score`` is the sum
        that early stopping maximizes.
    """
    raise NotImplementedError


def train(cfg: dict[str, Any], seed: int, fraction: float = 1.0, size: str = "base") -> dict[str, Any]:
    """Train the encoder and all heads jointly (spec 6.4). No thresholding here.

    AdamW; ``train.lr``; ``train.epochs``; losses summed with
    ``train.loss_weights``; early stopping on :func:`dev_selection_score` with
    ``train.early_stopping_patience``. Seeds every RNG (python, numpy, torch)
    from ``seed`` and honours ``runtime.deterministic``.

    Args:
        cfg: Resolved config.
        seed: One of ``train.seeds``. Spec 6.4 requires all three to be run and
            the mean and spread reported.
        fraction: Learning-curve fraction the bank was compiled from (E4). Must
            match the bank's ``source_fraction``.
        size: ``"base"`` or ``"small"`` (E3c).

    Returns:
        ``{"checkpoint_path", "seed", "epochs_run", "best_dev": {...},
        "resolved_encoder", "bank_hash"}``.

    Raises:
        ContractViolation: if ``fraction`` disagrees with ``bank.source_fraction``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.4 calibrate  (src/reflex/calibrate.py) -- no training
# ---------------------------------------------------------------------------


def nonconformity_scores(
    probs: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
) -> list[float]:
    """Compute split-conformal nonconformity ``s = 1 - p(gold)`` (spec 6.8 step 1).

    Args:
        probs: One probability vector per turn where the head applies.
        gold_indices: Gold class index per turn. Rows with a negative index are
            SKIPPED, not scored -- the head did not apply there.

    Returns:
        One score per applicable row, in input order.
    """
    raise NotImplementedError


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Return the finite-sample conformal quantile (spec 6.8 step 1).

    ``q = the ceil((n+1)(1-alpha))/n quantile of s``. Implement the
    finite-sample correction literally: sort ``s`` ascending and take element
    ``ceil((n+1)*(1-alpha)) - 1`` (0-based), clamping the index to ``n-1``. Using
    ``numpy.quantile(s, 1-alpha)`` instead drops the ``(n+1)`` correction and
    quietly under-covers.

    Args:
        scores: Nonconformity scores.
        alpha: Miscoverage rate, e.g. ``gate.alpha``.

    Returns:
        ``q_h``. The prediction set is then
        ``{classes with softmax >= 1 - q_h}`` (spec 6.6 signal 1).

    Raises:
        ValueError: if ``scores`` is empty or ``alpha`` is outside ``(0, 1)``.
    """
    raise NotImplementedError


def build_novelty_index(context_vectors: Any, cfg: dict[str, Any]) -> NoveltyIndex:
    """Build the FAISS index over ALL train context vectors (spec 6.6 signal 2).

    ``faiss.IndexFlatIP`` over L2-normalized vectors, so inner product is
    cosine. Flat (exact) is required: an approximate index would make
    ``novelty_distance`` depend on index build order, and spec 10 demands
    determinism.

    Args:
        context_vectors: ``(n_train_turns, d)`` float32 array.
        cfg: Resolved config.

    Returns:
        A :class:`NoveltyIndex`.
    """
    raise NotImplementedError


def save_novelty_index(index: NoveltyIndex, cfg: dict[str, Any], seed: int) -> str:
    """Persist the novelty index; the path goes into :attr:`Calibration.novelty_index_path`.

    Args:
        index: From :func:`build_novelty_index`.
        cfg: Resolved config.
        seed: The training seed whose encoder produced the vectors.

    Returns:
        The absolute path written.
    """
    raise NotImplementedError


def load_novelty_index(path: str, cfg: dict[str, Any]) -> NoveltyIndex:
    """Load a persisted novelty index.

    Args:
        path: From :attr:`Calibration.novelty_index_path`.
        cfg: Resolved config.

    Returns:
        A :class:`NoveltyIndex`.
    """
    raise NotImplementedError


def calibrate(cfg: dict[str, Any], checkpoint_path: str, seed: int) -> Calibration:
    """Compute all gate thresholds on dev (spec 6.8). No training.

    **Dev, SEEN SUBFLOWS ONLY** -- spec 6.8's first line. ``Partitions.dev`` is
    already novel-filtered, so use it as given and do not re-add anything.

    Computes ``q_h`` for every head at ``gate.alpha``; the
    ``gate.novelty_percentile`` percentile of dev ``novelty_distance``; and the
    whole ``gate.alpha_sweep`` set of quantiles (spec 6.8 step 3) so E5 needs no
    recalibration.

    Args:
        cfg: Resolved config.
        checkpoint_path: The checkpoint to calibrate. Quantiles are NOT
            transferable across checkpoints.
        seed: That checkpoint's seed.

    Returns:
        A :class:`Calibration`.
    """
    raise NotImplementedError


def write_calibration(calibration: Calibration, cfg: dict[str, Any]) -> str:
    """Persist to ``paths.calibration_path`` (``outputs/calibration/gate.json``).

    Args:
        calibration: From :func:`calibrate`.
        cfg: Resolved config.

    Returns:
        The absolute path written.
    """
    raise NotImplementedError


def load_calibration(cfg: dict[str, Any], path: Optional[str] = None) -> Calibration:
    """Load a persisted calibration.

    Args:
        cfg: Resolved config. Uses ``paths.calibration_path`` when ``path`` is None.
        path: Explicit override.

    Returns:
        A :class:`Calibration`.

    Raises:
        FileNotFoundError: if it has not been produced yet.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.5 select  (src/reflex/select.py) -- scores only, NO decisions
# ---------------------------------------------------------------------------


def build_selector(
    cfg: dict[str, Any],
    checkpoint_path: str,
    bank: Bank,
    ontology: dict[str, Any],
    calibration: Calibration,
) -> Any:
    """Assemble the inference bundle: model, bank, template embeddings, novelty index.

    Template embeddings are computed ONCE here and cached, not per turn -- spec 9
    criterion 5 allows 20 ms p95 for the whole fast path.

    Args:
        cfg: Resolved config.
        checkpoint_path: Trained checkpoint.
        bank: The compiled bank.
        ontology: The ontology.
        calibration: Needed for ``novelty_index_path``.

    Returns:
        An opaque selector handle, passed to :func:`score_turn` and
        :func:`select_from_scores`.
    """
    raise NotImplementedError


def score_turn(selector: Any, context: ContextWindow, turn: NormalizedTurn, cfg: dict[str, Any]) -> SelectorScores:
    """Produce all head scores for one agent turn (spec 6.5 steps 1-3). NO decisions.

    Returns SOFTMAX PROBABILITIES, not logits, over each head's canonical class
    order -- the gate's conformal sets are defined on probabilities.

    Also fills :attr:`SelectorScores.novelty_distance` (the boundary ruling at
    the top of this file: distance is a signal computed here, thresholded in the
    gate).

    ``turn`` is passed for its ``candidates`` list and its nextstep-applicability
    only. **Reading ``turn.intent``, ``turn.action``, ``turn.values``,
    ``turn.utt_id`` or ``turn.utt_rank`` here is label leakage** and invalidates
    every number in the report.

    Args:
        selector: From :func:`build_selector`.
        context: From :func:`build_context`.
        turn: The turn being predicted (candidates only).
        cfg: Resolved config.

    Returns:
        A :class:`SelectorScores`.
    """
    raise NotImplementedError


def prediction_set(probs: Sequence[float], q: float) -> list[int]:
    """Return the conformal prediction set (spec 6.6 signal 1).

    ``{i : probs[i] >= 1 - q}``. Note the inequality direction and that the set
    can be EMPTY (when ``q`` is small and the model is diffuse) as well as large.
    Both count as ``|set| != 1``, i.e. ``low_confidence``. Do not "fix" an empty
    set by inserting the argmax: that would silently break coverage.

    Args:
        probs: A probability vector over a head's classes.
        q: The head's ``q_h`` from :class:`Calibration`.

    Returns:
        Class indices in ascending order.
    """
    raise NotImplementedError


def select_from_scores(
    selector: Any,
    scores: SelectorScores,
    turn: NormalizedTurn,
    cfg: dict[str, Any],
) -> Selection:
    """Turn scores into the argmax selection (spec 6.5 steps 2-5). Still not routing.

    Per spec 6.5: H1 argmax chooses the branch; take_action -> H3 then H4;
    retrieve_utterance -> H5 argmax skeleton, then top-1 template per act
    position; end_conversation -> nextstep only. Then spec 6.5 step 5 maps the
    composed text onto the 100 candidates via :func:`map_to_candidate`.

    Args:
        selector: From :func:`build_selector`.
        scores: From :func:`score_turn`.
        turn: For ``candidates``. Gold fields must not be read.
        cfg: Resolved config.

    Returns:
        A :class:`Selection`.
    """
    raise NotImplementedError


def delexicalize_candidates(
    candidates: Sequence[int],
    utterances: Sequence[str],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> list[str]:
    """Delexicalize the 100 candidate utterances so they are comparable to templates.

    Necessary because ``utterances.json`` is LEXICALIZED (see
    :func:`load_utterances`): comparing a ``{order_id}``-shaped composed template
    against a raw candidate containing ``"7654321"`` would score near zero for
    the right answer. Results should be cached per candidate id -- the same ids
    recur across turns.

    Args:
        candidates: The turn's 100 utterance ids.
        utterances: From :func:`load_utterances`.
        registry: The slot registry.
        cfg: Resolved config.

    Returns:
        100 delexicalized strings, aligned with ``candidates`` by index.
    """
    raise NotImplementedError


def map_to_candidate(
    composed_text_delex: str,
    candidates: Sequence[int],
    candidate_texts_delex: Sequence[str],
    selector: Any,
    cfg: dict[str, Any],
) -> tuple[Optional[int], int, float]:
    """Map composed template text onto the official candidate pool (spec 6.5 step 5).

    "Among the 100 candidates, pick the candidate whose delexicalized form has
    the highest cosine similarity to the composed text (exact string match wins
    if present)." Ties break toward the LOWEST rank, so the mapping is
    deterministic.

    Args:
        composed_text_delex: From :attr:`Selection.composed_text_delex`.
        candidates: The turn's 100 global utterance ids.
        candidate_texts_delex: From :func:`delexicalize_candidates`.
        selector: For the embedding model.
        cfg: Resolved config.

    Returns:
        ``(candidate_utt_id, candidate_rank, similarity)``. ``(None, -1, 0.0)``
        when there are no candidates. ``candidate_rank`` indexes ``candidates``
        and is what the official evaluator consumes; ``candidate_utt_id`` is the
        global id.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.6 gate  (src/reflex/gate.py) -- THE ONLY DECISION POINT
# ---------------------------------------------------------------------------


def evaluate_gate(
    scores: SelectorScores,
    selection: Selection,
    missing_slots: Sequence[str],
    unseen_action: bool,
    calibration: Calibration,
    cfg: dict[str, Any],
) -> GateOutput:
    """Decide fast-path vs escalate for one Arm B turn (spec 6.6). The only such place.

    Three signals, all of which must pass for ``route="reflex"``:

    1. **Confidence.** For every head REQUIRED BY THIS TURN'S PREDICTED NEXTSTEP
       -- H1 and H2 always; H3 only if take_action; H5 and H7-per-position only
       if retrieve_utterance -- the conformal :func:`prediction_set` must have
       size exactly 1. Otherwise ``low_confidence``. A head that does not apply
       is not consulted and reports ``0`` in ``set_sizes``.
    2. **Novelty.** ``scores.novelty_distance > calibration.novelty_threshold``
       -> ``novel``. Strictly greater, matching spec 6.6's wording.
    3. **Availability.** Non-empty ``missing_slots`` -> ``unavailable_slot``;
       ``unseen_action`` -> ``unseen_action``.

    When several fail, report per :data:`GATE_REASON_PRECEDENCE`:
    ``novel > unseen_action > unavailable_slot > low_confidence``.

    **The gate never modifies scores; it only routes** (spec 6.6, last line).
    No renormalizing, no argmax overriding, no threshold nudging.

    ``gate.affect_signal`` is ``off`` in v1 and spec 13 puts it out of scope; if
    it is ever ``on``, raise rather than silently ignoring it.

    Args:
        scores: From :func:`score_turn`.
        selection: From :func:`select_from_scores`, to know which heads apply.
        missing_slots: From :func:`check_availability`. The gate does not compute
            this itself -- :mod:`reflex.fill` owns slot logic.
        unseen_action: Whether the predicted action was never seen in train for
            the predicted intent (from the kb / action patterns).
        calibration: Quantiles and the novelty threshold.
        cfg: Resolved config. Uses ``gate.affect_signal``.

    Returns:
        A :class:`GateOutput` with ``set_sizes`` fully populated for every head
        key, applicable or not.

    Raises:
        ValueError: if ``gate.affect_signal`` is not ``off``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.7 fill  (src/reflex/fill.py) -- no selection
# ---------------------------------------------------------------------------


def collect_slot_sources(
    turns: Sequence[NormalizedTurn],
    turn_index: int,
    scenario: dict[str, Any],
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> SlotSources:
    """Gather available slot values for the turn at ``turn_index`` (spec 6.7).

    Three origins, in the spec's strict priority order: values the CUSTOMER
    stated (from the delex mapping of customer turns), values entered in PRIOR
    ACTIONS, and scenario fields ALREADY DISCLOSED. The disclosure rule is the
    same leakage guard as :func:`build_context`: an undisclosed scenario field is
    not available, however convenient it would be.

    Args:
        turns: All turns of the conversation.
        turn_index: The turn being answered. Only turns before it are read.
        scenario: Raw ``convo["scenario"]``, consulted only for disclosure.
        registry: The slot registry.
        cfg: Resolved config.

    Returns:
        A :class:`SlotSources`.
    """
    raise NotImplementedError


def check_availability(
    required_slots: Sequence[str],
    sources: SlotSources,
    cfg: dict[str, Any],
) -> list[str]:
    """Return the required slots that have NO value in any source (spec 6.6 signal 3).

    This is the availability signal the gate consumes. It is computed here
    because :mod:`reflex.fill` owns slot logic; this function does not route and
    must not construct a :class:`GateOutput`.

    Args:
        required_slots: Slots needed by the chosen templates or action.
        sources: From :func:`collect_slot_sources`.
        cfg: Resolved config.

    Returns:
        Missing slot names, in ``required_slots`` order. Empty means available.
    """
    raise NotImplementedError


def fill_template(
    template: Template,
    sources: SlotSources,
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> tuple[Optional[str], list[str]]:
    """Substitute values into one template's ``{slot}`` placeholders (spec 6.7).

    **A slot with no available value is never guessed.** Return
    ``(None, missing)`` and let the gate escalate. Formatting follows
    ``SlotSpec.format``.

    Args:
        template: The chosen template.
        sources: From :func:`collect_slot_sources`.
        registry: The slot registry.
        cfg: Resolved config.

    Returns:
        ``(filled_text, missing_slots)``. ``filled_text`` is ``None`` iff
        ``missing_slots`` is non-empty.
    """
    raise NotImplementedError


def compose_utterance(
    templates: Sequence[Template],
    sources: SlotSources,
    registry: SlotRegistry,
    cfg: dict[str, Any],
) -> tuple[Optional[str], list[str]]:
    """Fill and join one template per act position into the final utterance.

    Joining is a single space between filled sentences, and spec 13 forbids any
    rewriting, paraphrasing or generation on the fast path -- join only.

    Args:
        templates: One per act position, in skeleton order.
        sources: From :func:`collect_slot_sources`.
        registry: The slot registry.
        cfg: Resolved config.

    Returns:
        ``(utterance_text, missing_slots)``. ``utterance_text`` is ``None`` iff
        any template was unfillable; ``missing_slots`` is the union, deduplicated,
        in first-seen order.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.8 llm_agent  (src/reflex/llm_agent.py) -- the ONLY inference-time LLM caller
# ---------------------------------------------------------------------------


def load_prompt(cfg: dict[str, Any], which: str) -> tuple[str, str]:
    """Load a frozen prompt file and its hash.

    Spec 6.9 step 3: the frozen prompt is committed and its hash recorded in
    every run.

    Args:
        cfg: Resolved config. Uses ``paths.prompt_agent_a`` /
            ``paths.prompt_act_labeling``.
        which: ``"agent_A"`` or ``"act_labeling"``.

    Returns:
        ``(prompt_text, sha256_hex_16)``.

    Raises:
        FileNotFoundError: if the prompt file is missing.
        ValueError: on an unknown ``which``.
    """
    raise NotImplementedError


def render_agent_prompt(
    context: ContextWindow,
    turn: NormalizedTurn,
    candidate_texts: Sequence[str],
    guidelines: dict[str, Any],
    ontology: dict[str, Any],
    kb: dict[str, Any],
    prompt_template: str,
    cfg: dict[str, Any],
) -> tuple[str, str]:
    """Render the Arm A prompt (spec 6.9 step 1).

    System = guidelines rendered as text + the action list from ontology/kb +
    the output format. User = **the same context string the encoder sees**
    (``context.text``, spec 6.9 step 1) plus, for retrieve_utterance turns, the
    100 candidates NUMBERED 0-99 so the model's ``candidate_index`` is a rank
    that lines up with :attr:`NormalizedTurn.utt_rank`.

    Both arms must see identical context, or spec 8.7's paired comparison is not
    paired. Do not enrich this prompt with anything the encoder cannot see.

    Args:
        context: From :func:`build_context`.
        turn: The turn being predicted (candidates only; gold fields must not be
            rendered).
        candidate_texts: Candidate strings in ``turn.candidates`` order.
        guidelines: From :func:`load_guidelines`.
        ontology: The ontology.
        kb: From :func:`load_kb`.
        prompt_template: From :func:`load_prompt`.
        cfg: Resolved config.

    Returns:
        ``(system, user)``.
    """
    raise NotImplementedError


def parse_llm_json(raw: str, ontology: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Strictly parse the LLM's JSON output (spec 6.9 step 2).

    Expected keys: ``nextstep``, ``intent``, ``action``, ``values``,
    ``candidate_index``. "Parse strictly" means: reject unknown nextstep or
    subflow values, reject a ``candidate_index`` outside ``[-1, 100)``, and do
    not repair truncated JSON. Return ``None`` on failure so the caller can
    apply the single retry and then count the turn incorrect -- never fabricate a
    decision.

    Args:
        raw: The model's raw text.
        ontology: For validating nextstep / intent / action against the label sets.

    Returns:
        The parsed dict, or ``None`` if it does not parse or does not validate.
    """
    raise NotImplementedError


def build_llm_agent(cfg: dict[str, Any], model_key: str) -> Any:
    """Construct the LLM agent handle, honouring the kill switch.

    **Raises :class:`LLMDisabledError` immediately when ``llm.enabled`` is
    false.** Failing at construction rather than at the first call means a
    misconfigured run dies in a second instead of after an hour of setup.

    When enabled, resolves ``llm.strong`` / ``llm.cheap`` through
    :func:`require_filled` (so a ``<fill: model id>`` placeholder raises
    :class:`PlaceholderConfigError`), opens the ``llm.cache_dir`` cache (spec 6.9
    step 5, keyed by ``(model, prompt_hash, context_hash)``) and arms the
    ``llm.budget_usd_per_run`` cap.

    Args:
        cfg: Resolved config.
        model_key: ``"strong"`` or ``"cheap"``.

    Returns:
        An opaque agent handle for :func:`llm_decide`.

    Raises:
        LLMDisabledError: if ``llm.enabled`` is false.
        PlaceholderConfigError: if the model id or prices are still unfilled.
        ValueError: on an unknown ``model_key``.
    """
    raise NotImplementedError


def llm_decide(
    agent: Any,
    context: ContextWindow,
    turn: NormalizedTurn,
    candidate_texts: Sequence[str],
    cfg: dict[str, Any],
) -> LLMDecision:
    """Get one decision from the LLM (spec 6.9). The only inference-time LLM call.

    Checks the cache first (a cache hit is not re-billed and reports
    ``latency_ms=0.0``); checks the budget BEFORE calling; calls at
    ``llm.temperature`` (0); parses with :func:`parse_llm_json`, retrying
    ``llm.max_retries_on_parse_failure`` (1) time; on final failure returns
    ``parse_failed=True`` and the turn is counted INCORRECT and logged rather
    than dropped (spec 6.9 step 2). Records tokens and wall clock (spec 6.9
    step 4).

    Under ``llm.dry_run``, returns a decision with ``parse_failed=True`` and the
    ESTIMATED token counts from :func:`estimate_prompt_tokens` without calling
    anything (spec 10's dry-run flag).

    Args:
        agent: From :func:`build_llm_agent`.
        context: From :func:`build_context`.
        turn: The turn being predicted.
        candidate_texts: The 100 candidates, or ``[]``.
        cfg: Resolved config.

    Returns:
        An :class:`LLMDecision`.

    Raises:
        LLMDisabledError: if invoked while ``llm.enabled`` is false.
        BudgetExceededError: if the call would exceed ``llm.budget_usd_per_run``.
    """
    raise NotImplementedError


def estimate_prompt_tokens(system: str, user: str, cfg: dict[str, Any]) -> tuple[int, int]:
    """Estimate ``(tokens_in, tokens_out)`` for a prompt WITHOUT calling any API.

    Backs the dry-run flag. Use a local tokenizer; never a hosted token-counting
    endpoint (that is still a paid call).

    Args:
        system: System prompt.
        user: User prompt.
        cfg: Resolved config.

    Returns:
        ``(tokens_in, estimated_tokens_out)``.
    """
    raise NotImplementedError


def llm_cost_usd(tokens_in: int, tokens_out: int, model_key: str, cfg: dict[str, Any]) -> float:
    """Compute LLM cost per spec 8.6 from the config price table.

    ``tokens_in * price_in + tokens_out * price_out``, with prices read from
    ``llm.prices_usd_per_million[model_key]`` and divided by 1e6. Prices ship as
    ``<fill>`` placeholders, so this raises rather than returning 0.0 -- a
    silent zero would show up in the report as a free LLM.

    Args:
        tokens_in: Billed prompt tokens.
        tokens_out: Billed completion tokens.
        model_key: ``"strong"`` or ``"cheap"``.
        cfg: Resolved config.

    Returns:
        Cost in USD.

    Raises:
        PlaceholderConfigError: if the price table is still unfilled.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.9 evaluate  (src/reflex/evaluate.py) -- THE ONLY place metrics are computed
# ---------------------------------------------------------------------------


def load_official_metrics(cfg: dict[str, Any]) -> OfficialMetrics:
    """Import the REAL official ABCD metric functions. Spec 13 forbids reimplementing them.

    There is a real obstacle, and this function is where it is handled once.
    ``utils/evaluate.py`` begins with ``from components.systems import
    Application`` and ``from utils.help import prepare_inputs``, but the
    downloaded ABCD tree contains only ``data/`` and ``utils/`` -- there is no
    ``components/`` package. A bare import therefore fails with
    ``ModuleNotFoundError``.

    The sanctioned fix is a SHIM, not a rewrite: insert stub
    ``components``/``components.systems``/``components.tools`` modules into
    ``sys.modules`` before importing, where ``Application.prepare_masks`` raises
    :class:`NotImplementedError`. The four report functions only touch
    ``Application`` on the ``kb_labels is not None`` path, so every metric this
    project needs works untouched. Verified: ``ast_report`` and ``cds_report``
    both run to completion under the shim.

    Consequence: **``kb_labels`` MUST stay ``None``** (``eval.use_kb_labels:
    false``). The KB-masked variant of the official metrics is unavailable until
    ``components/`` is vendored. Report the unmasked numbers and say so.

    Args:
        cfg: Resolved config. Uses ``eval.official_utils_dir`` (the PARENT of
            ``utils/``) and ``eval.use_kb_labels``.

    Returns:
        An :class:`OfficialMetrics` holding the four real functions.

    Raises:
        FileNotFoundError: if ``utils/evaluate.py`` is not found.
        ContractViolation: if ``eval.use_kb_labels`` is true.
    """
    raise NotImplementedError


def build_ast_arrays(records: Sequence[EvalRecord], ontology: dict[str, Any]) -> tuple[Any, Any]:
    """Shape records into ``ast_report``'s expected arguments (spec 8.2).

    The official signature is ``ast_report(predictions, labels)`` where
    ``predictions = (bslot_preds, value_preds)`` are 2-D SCORE arrays
    (``np.argmax(..., axis=1)`` is applied inside) and ``labels =
    (bslot_labels, value_labels)`` are 1-D int arrays. AST scores TAKE_ACTION
    TURNS ONLY, and every AST denominator is ``len(bslot_preds)``, so filter to
    take_action turns before calling -- passing all turns silently deflates every
    AST number.

    Since REFLEXIVE emits discrete predictions rather than score vectors,
    synthesize one-hot rows: a one-hot's argmax is the prediction, which is all
    ``ast_report`` reads.

    Args:
        records: Records for ONE (arm, model, seed, split).
        ontology: For the action and value label spaces.

    Returns:
        ``(predictions, labels)`` ready to splat into ``ast_report``.
    """
    raise NotImplementedError


def build_cds_arrays(records: Sequence[EvalRecord], ontology: dict[str, Any], bank: Bank) -> tuple[Any, Any, Any]:
    """Shape records into ``cds_report``'s expected arguments (spec 8.2).

    The official signature is
    ``cds_report(predictions, labels, ci_and_tc, kb_labels=None)`` with
    ``predictions = (intent_pred, nextstep_pred, bslot_pred, value_pred,
    utterance_rank)`` as 2-D score arrays and ``labels`` the matching 1-D int
    arrays. Five things this function must get right:

    1. ``nextstep_label`` MUST be encoded by :data:`NEXT_STEPS` index --
       ``cds_report`` branches on ``== 0 / 1 / 2`` meaning retrieve_utterance /
       take_action / end_conversation.
    2. ``utterance_label`` is a RANK in ``[0, 100)`` (it is compared against an
       ``argpartition`` over a 100-wide score row), i.e.
       :attr:`NormalizedTurn.utt_rank`, NOT ``utt_id``. Likewise the prediction
       side uses :attr:`Decision.candidate_rank`.
    3. ``-1`` is the official "not applicable" marker: ``bslot_label``,
       ``value_label`` and ``utterance_label`` must be ``-1`` on turns where
       the head does not apply, because the official denominators are
       ``sum(label >= 0)``.
    4. ``ci_and_tc`` is ``(convo_ids, turn_counts)`` and the official code calls
       ``.detach().cpu().numpy()`` on both -- so they MUST BE TORCH TENSORS, not
       numpy arrays. ``turn_counts`` is ABCD's sparse ``turn_count``, not the
       dense ``turn_index``.
    5. Include the synthetic ``end_conversation`` turns. They are one third of
       the nextstep signal and ``utils/process.py`` emits them.

    Args:
        records: Records for ONE (arm, model, seed, split).
        ontology: For intent / action / value label spaces.
        bank: Unused by the official code; accepted so callers need not branch.

    Returns:
        ``(predictions, labels, ci_and_tc)`` ready to splat into ``cds_report``.
    """
    raise NotImplementedError


def ast_metrics(records: Sequence[EvalRecord], ontology: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    """Run the official ``ast_report`` and return its dict (spec 8.2).

    Args:
        records: Records for one (arm, model, seed, split).
        ontology: The ontology.
        cfg: Resolved config.

    Returns:
        ``{"Bslot_Accuracy", "Value_Accuracy", "Joint_Accuracy"}`` -- the
        official key names, unrenamed, so the report cannot drift from the
        source.
    """
    raise NotImplementedError


def cds_metrics(
    records: Sequence[EvalRecord],
    ontology: dict[str, Any],
    bank: Bank,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run the official ``cds_report`` and return its dict (spec 8.2).

    Args:
        records: Records for one (arm, model, seed, split).
        ontology: The ontology.
        bank: Passed through to :func:`build_cds_arrays`.
        cfg: Resolved config.

    Returns:
        The official keys: ``Intent_Accuracy``, ``Nextstep_Accuracy``,
        ``Action_Accuracy``, ``Value_Accuracy``, ``Joint_Accuracy``,
        ``Recall_at_1``, ``Recall_at_5``, ``Recall_at_10``, ``Turn_Accuracy``,
        ``Cascading_Score``.
    """
    raise NotImplementedError


def routing_metrics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute spec 8.1 routing metrics.

    ``reflex_rate`` = reflex turns / agent turns, overall AND per nextstep type;
    ``containment`` = conversations with no escalated turn / conversations;
    ``escalation_reason_share`` = the distribution of ``GateOutput.reason`` over
    escalated turns.

    Arm A is always ``route="escalated"`` (spec 5.5), so its reflex_rate is 0 by
    construction -- that is correct accounting, not a bug.

    Args:
        records: Arm B records for one (model, seed, split).
        cfg: Resolved config.

    Returns:
        ``{"reflex_rate", "reflex_rate_by_nextstep", "containment",
        "escalation_reason_share", "n_agent_turns", "n_conversations"}``.
    """
    raise NotImplementedError


def fastpath_metrics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute spec 8.3 fast-path-only quality.

    Error rate over REFLEX TURNS ONLY, per component (nextstep, intent,
    action+values, utterance recall@1), plus the ``exact_template_match`` rate.
    Escalated turns are excluded from the denominator -- the point is to measure
    what the fast path did when it chose to speak.

    Args:
        records: Arm B records for one (model, seed, split).
        cfg: Resolved config.

    Returns:
        ``{"fastpath_error_rate": {component -> rate},
        "exact_template_match_rate", "n_reflex_turns"}``.
    """
    raise NotImplementedError


def novelty_metrics(
    records_novel: Sequence[EvalRecord],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute spec 8.4 novelty metrics on ``test_novel``.

    ``novel_escalation_rate`` = escalated / agent turns; and
    ``novel_fastpath_error_rate`` = incorrect reflex / reflex turns. Spec 9
    criterion 3 wants the escalation rate >= 0.95.

    Args:
        records_novel: Arm B records on ``test_novel``.
        cfg: Resolved config.

    Returns:
        ``{"novel_escalation_rate", "novel_fastpath_error_rate",
        "n_agent_turns", "n_reflex_turns"}``.
    """
    raise NotImplementedError


def calibration_metrics(
    records: Sequence[EvalRecord],
    calibration: Calibration,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute spec 8.5 calibration metrics.

    ECE over ``eval.ece_bins`` (10) bins of the argmax probability for H1, H3,
    H5, H7 on ``test_seen``; and empirical coverage per head at the chosen alpha,
    which spec 9 criterion 4 requires to be ``>= 1 - alpha - 0.01``.

    Args:
        records: Arm B records on ``test_seen``.
        calibration: The calibration in force.
        cfg: Resolved config.

    Returns:
        ``{"ece": {head -> value}, "empirical_coverage": {head -> value},
        "alpha"}``.
    """
    raise NotImplementedError


def cost_latency_metrics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute spec 8.6 cost and latency.

    LLM cost via :func:`llm_cost_usd`; fast-path cost = ``cpu_seconds *
    cost.cpu_price_per_hour_usd / 3600``; per-arm cost per turn and per
    conversation; p50/p95 latency per arm, with Arm B's fast-path-only latency
    reported SEPARATELY from its escalated-turn latency (spec 8.6 is explicit).

    Cache hits cost nothing but still took a turn -- count them in the latency
    denominator and not in the spend.

    Args:
        records: Records for one (arm, model, seed, split).
        cfg: Resolved config.

    Returns:
        ``{"llm_cost_per_turn", "fastpath_cost_per_turn", "arm_cost_per_turn",
        "arm_cost_per_conversation", "latency_p50_ms", "latency_p95_ms",
        "latency_p50_ms_fastpath", "latency_p95_ms_fastpath",
        "latency_p50_ms_escalated", "latency_p95_ms_escalated"}``.

    Raises:
        PlaceholderConfigError: if LLM turns are present and prices are unfilled.
    """
    raise NotImplementedError


def bootstrap_ci(
    values_by_conversation: dict[int, Sequence[float]],
    statistic: str,
    cfg: dict[str, Any],
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a 95% CI **over conversations**, not turns (spec 8.7).

    Resampling turns would treat the ~9 agent turns of one conversation as
    independent and shrink every interval. Resample conversation ids with
    replacement, ``eval.n_bootstrap`` (1000) times, at ``eval.ci`` (0.95).
    Seeded, so the CI is reproducible.

    Args:
        values_by_conversation: ``convo_id -> per-turn values`` (typically 0/1
            correctness).
        statistic: ``"mean"`` or ``"rate"``.
        cfg: Resolved config.
        seed: Bootstrap seed.

    Returns:
        ``(point_estimate, ci_low, ci_high)``.
    """
    raise NotImplementedError


def mcnemar_pvalue(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
) -> float:
    """McNemar test on per-turn correctness (spec 8.7). Report the p-value only.

    The two sequences must be the SAME TURNS IN THE SAME ORDER -- the test is
    paired and is meaningless otherwise. Use the exact binomial test on the
    discordant pairs; the chi-square approximation misbehaves when discordant
    counts are small, which is exactly the regime a near-parity result sits in.

    Args:
        correct_a: Arm A correctness per turn.
        correct_b: Arm B correctness on the same turns, same order.

    Returns:
        Two-sided p-value.

    Raises:
        ValueError: if the lengths differ.
    """
    raise NotImplementedError


def parity_delta(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    metric_name: str,
) -> float:
    """Return ``metric(Arm B) - metric(Arm A)`` in PERCENTAGE POINTS (spec 8.7).

    Spec 9 states its tolerances in points (">= Arm A - 1.0 points"), while the
    official reports return fractions in ``[0, 1]``. Multiply by 100 here, once,
    so the comparison against spec 9 is apples to apples.

    Args:
        metrics_a: Arm A metrics.
        metrics_b: Arm B metrics on the same turns.
        metric_name: An official key, e.g. ``"Cascading_Score"``.

    Returns:
        The delta in percentage points.

    Raises:
        KeyError: if the metric is absent from either dict.
    """
    raise NotImplementedError


def evaluate_run(cfg: dict[str, Any], run_id: str, baseline_run_id: Optional[str] = None) -> dict[str, Any]:
    """Score one run end to end and write ``outputs/runs/<run_id>/metrics.json``.

    Reads ``decisions.jsonl``, runs every applicable spec 8 metric, and -- when
    ``baseline_run_id`` names an Arm A run over the same turns -- computes the
    spec 8.7 paired statistics.

    Args:
        cfg: Resolved config.
        run_id: The run to score.
        baseline_run_id: The Arm A run to compare against, for parity.

    Returns:
        The metrics dict that was written.

    Raises:
        FileNotFoundError: if the run directory or its decisions log is missing.
        ContractViolation: if the two runs do not cover an identical turn set
            (spec 8.2 requires both arms scored on identical turns).
    """
    raise NotImplementedError


def verdict(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the spec 9 PASS / PARTIAL / STOP criteria. Computation, not rendering.

    PASS requires ALL of: reflex_rate >= 0.50; CDS and AST joint each >= Arm A -
    1.0 points with the 95% CI lower bound of each delta >= -2.0 points;
    novel_escalation_rate >= 0.95; empirical coverage per head >= 1 - alpha -
    0.01; fast-path p95 latency <= 20 ms. STOP-AND-RETHINK if reflex_rate < 0.30
    at any alpha satisfying criterion 2. Everything between is PARTIAL.

    Spec 9's last line is binding: "report it, do not tune the baseline down to
    reach it." This function reports; it never adjusts an input.

    The thresholds above are quoted from the spec for documentation only -- read
    them from config at runtime, per spec 10.

    Args:
        metrics: From :func:`evaluate_run`, including parity statistics.
        cfg: Resolved config.

    Returns:
        ``{"verdict": "PASS"|"PARTIAL"|"STOP", "criteria": {name -> {"passed":
        bool, "value": float, "threshold": float}}}``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4.10 report  (src/reflex/report.py) -- NO new numbers
# ---------------------------------------------------------------------------


def render_report(metrics_paths: Sequence[str], out_path: str, cfg: dict[str, Any]) -> str:
    """Render ``report.md`` in the fixed spec 11.3 structure. No new numbers.

    The nine sections, in order: 1 Setup; 2 Table A Routing; 3 Table B Quality;
    4 Table C Novelty; 5 Table D Cost and latency; 6 Figure 1 coverage-error and
    Figure 2 learning curve; 7 Table E Ablations; 8 Table F Failure taxonomy with
    three anonymized examples; 9 Verdict, one paragraph, no adjectives.

    Spec 4.10 forbids computing anything here. If a number is not in a
    ``metrics.json``, the cell renders as ``n/a`` and the reason is stated --
    never computed on the fly. Spec 14 also requires every headline number to
    carry a 95% CI, so a headline without one renders as an explicit gap.

    Args:
        metrics_paths: One or more ``metrics.json`` paths.
        out_path: Where to write the report.
        cfg: Resolved config.

    Returns:
        The absolute path written.

    Raises:
        ContractViolation: if a required spec 11.3 section has no input at all.
    """
    raise NotImplementedError


def render_table(name: str, metrics: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """Render one spec 11.3 table as a markdown string.

    Args:
        name: ``"A"`` | ``"B"`` | ``"C"`` | ``"D"`` | ``"E"`` | ``"F"``.
        metrics: The metrics dicts feeding this table.
        cfg: Resolved config.

    Returns:
        A markdown table.

    Raises:
        ValueError: on an unknown table name.
    """
    raise NotImplementedError


def render_figure(name: str, metrics: Sequence[dict[str, Any]], out_dir: str, cfg: dict[str, Any]) -> str:
    """Render one spec 11.3 figure into ``outputs/runs/<run_id>/figures/``.

    Args:
        name: ``"coverage_error"`` (Figure 1, from E5) or ``"learning_curve"``
            (Figure 2, from E4).
        metrics: The metrics dicts feeding this figure.
        out_dir: Figures directory.
        cfg: Resolved config.

    Returns:
        The absolute path written.

    Raises:
        ValueError: on an unknown figure name.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# run  (src/reflex/run.py) -- orchestration, run_id, manifest, decisions log
# ---------------------------------------------------------------------------


def new_run_id(cfg: dict[str, Any], arm: str, model_key: str, split: str, seed: int) -> str:
    """Mint a run id (spec 10: "every run has a run_id").

    Format: ``"{UTC yyyymmddTHHMMSSZ}-arm{arm}-{model_key}-{split}-s{seed}"`` --
    sortable, and self-describing in a directory listing.

    Args:
        cfg: Resolved config.
        arm: ``"A"`` or ``"B"``.
        model_key: ``"strong"`` or ``"cheap"``.
        split: Partition name.
        seed: Training seed.

    Returns:
        The run id, also the directory name under ``paths.runs_dir``.
    """
    raise NotImplementedError


def write_manifest(manifest: RunManifest, cfg: dict[str, Any]) -> str:
    """Write ``outputs/runs/<run_id>/manifest.json`` (spec 10).

    Must be written BEFORE the first decision, so that an aborted run is still
    identifiable.

    Args:
        manifest: The manifest.
        cfg: Resolved config.

    Returns:
        The absolute path written.
    """
    raise NotImplementedError


def read_manifest(cfg: dict[str, Any], run_id: str) -> RunManifest:
    """Read a run's manifest.

    Args:
        cfg: Resolved config.
        run_id: The run.

    Returns:
        The :class:`RunManifest`.

    Raises:
        FileNotFoundError: if absent.
    """
    raise NotImplementedError


def append_decisions(records: Iterable[EvalRecord], cfg: dict[str, Any], run_id: str) -> int:
    """Append records to ``outputs/runs/<run_id>/decisions.jsonl`` (spec 10).

    One JSON object per line in the flattened spec 5.7 shape (see
    :meth:`EvalRecord.to_dict`). Spec 14 requires one record per turn per arm --
    including turns where the LLM failed to parse and turns the gate escalated.
    Nothing is ever dropped.

    Args:
        records: Records to append.
        cfg: Resolved config.
        run_id: The run.

    Returns:
        The number of records written.
    """
    raise NotImplementedError


def read_decisions(cfg: dict[str, Any], run_id: str) -> list[EvalRecord]:
    """Read a run's decisions log.

    Spec 10: "Nothing is computed from logs outside the Evaluator." This function
    is I/O only; do not aggregate here.

    Args:
        cfg: Resolved config.
        run_id: The run.

    Returns:
        The records in file order.

    Raises:
        FileNotFoundError: if the log is absent.
    """
    raise NotImplementedError


def run_arm(
    cfg: dict[str, Any],
    arm: str,
    model_key: str,
    split: str,
    seed: int,
    checkpoint_path: Optional[str] = None,
    alpha: Optional[float] = None,
) -> str:
    """Run one arm over one split and write manifest + decisions (spec 7 E1 / E2).

    Arm A calls the LLM on EVERY agent turn (spec 6.9) and marks every decision
    ``route="escalated"``. Arm B runs select -> gate -> (fill | LLM). Both arms
    must cover an identical turn set from :func:`iter_agent_turns`, or spec 8.2's
    paired comparison is invalid.

    With ``llm.enabled: false`` (the default), Arm A cannot run at all and this
    raises :class:`LLMDisabledError`. Arm B runs fully as long as no turn
    escalates; the first escalation raises unless ``llm.dry_run`` is set, in
    which case escalated turns are recorded as ``parse_failed`` with estimated
    tokens. That asymmetry is deliberate: the fast path is free and must be
    measurable without spending anything.

    Args:
        cfg: Resolved config.
        arm: ``"A"`` or ``"B"``.
        model_key: ``"strong"`` or ``"cheap"``.
        split: ``"test_seen"`` | ``"test_novel"`` | ``"dev"``.
        seed: Training seed.
        checkpoint_path: Required for Arm B.
        alpha: Overrides ``gate.alpha`` for E5.

    Returns:
        The ``run_id``.

    Raises:
        LLMDisabledError: for Arm A, or for Arm B escalation, while the LLM is
            disabled and not in dry-run.
        ValueError: if Arm B is requested without a checkpoint.
    """
    raise NotImplementedError


def run_sweep(cfg: dict[str, Any], experiment: str) -> list[str]:
    """Run an experiment matrix from spec Section 7.

    ``"E3"`` ablations (whole-utterance vs skeleton-first; gate variants;
    encoder size; context window K); ``"E4"`` learning curve over
    ``data.learning_curve_fractions``; ``"E5"`` coverage-error over
    ``gate.alpha_sweep``; ``"E6"`` the simulated growth loop, which spec 7 says
    runs only if E2 passes.

    Spec 7's execution order is E0 -> E1 -> E2 -> E5 -> E3 -> E4 -> E6; this
    function runs ONE of them and does not reorder anything.

    Args:
        cfg: Resolved config.
        experiment: ``"E3"`` | ``"E4"`` | ``"E5"`` | ``"E6"``.

    Returns:
        The ``run_id`` list produced, in execution order.

    Raises:
        ValueError: on an unknown experiment.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Ownership table -- mechanically enforced by tests/test_contracts.py
# ---------------------------------------------------------------------------

#: ``module name -> the public functions that module MUST define``.
#: Every name appears under exactly one module (MECE, spec Section 4). The test
#: asserts that each module defines exactly these names and that each signature
#: matches the one declared above.
MODULE_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "config": (
        "load_config",
        "apply_overrides",
        "resolve_path",
        "require_filled",
    ),
    "data": (
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
    ),
    "compile": (
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
    ),
    "models": (
        "build_encoder",
        "build_model",
        "save_checkpoint",
        "load_checkpoint",
    ),
    "train": (
        "build_train_examples",
        "collate_batch",
        "dev_selection_score",
        "train",
    ),
    "calibrate": (
        "nonconformity_scores",
        "conformal_quantile",
        "build_novelty_index",
        "save_novelty_index",
        "load_novelty_index",
        "calibrate",
        "write_calibration",
        "load_calibration",
    ),
    "select": (
        "build_selector",
        "score_turn",
        "prediction_set",
        "select_from_scores",
        "delexicalize_candidates",
        "map_to_candidate",
    ),
    "gate": ("evaluate_gate",),
    "fill": (
        "collect_slot_sources",
        "check_availability",
        "fill_template",
        "compose_utterance",
    ),
    "llm_agent": (
        "load_prompt",
        "render_agent_prompt",
        "parse_llm_json",
        "build_llm_agent",
        "llm_decide",
        "estimate_prompt_tokens",
        "llm_cost_usd",
    ),
    "evaluate": (
        "load_official_metrics",
        "build_ast_arrays",
        "build_cds_arrays",
        "ast_metrics",
        "cds_metrics",
        "routing_metrics",
        "fastpath_metrics",
        "novelty_metrics",
        "calibration_metrics",
        "cost_latency_metrics",
        "bootstrap_ci",
        "mcnemar_pvalue",
        "parity_delta",
        "evaluate_run",
        "verdict",
    ),
    "report": (
        "render_report",
        "render_table",
        "render_figure",
    ),
    "run": (
        "new_run_id",
        "write_manifest",
        "read_manifest",
        "append_decisions",
        "read_decisions",
        "run_arm",
        "run_sweep",
    ),
}

# Names imported for the benefit of implementers' type annotations; referenced
# here so linters do not strip them from the module namespace.
_SCHEMA_REEXPORTS = (
    ActionPattern, Bank, Calibration, ContextWindow, Decision, EvalRecord,
    GateOutput, LLMDecision, NormalizedTurn, Partitions, RunManifest, Selection,
    SelectorScores, Skeleton, SlotRegistry, SlotSources, SlotSpec, Template,
    TurnLabel, Iterator, Iterable,
)
