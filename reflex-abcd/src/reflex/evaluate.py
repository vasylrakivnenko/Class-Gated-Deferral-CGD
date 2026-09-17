"""Spec 4.9 -- Evaluator: THE ONLY PLACE metrics are computed.

This module WRAPS the official ABCD reports (spec 13 forbids reimplementing
them) and adds everything in spec Section 8 that the official code cannot
express.

WHAT THE OFFICIAL FUNCTIONS EXPECT (inferred by reading
``<eval.official_utils_dir>/utils/evaluate.py`` line by line; every claim below
was checked by running them)
-----------------------------------------------------------------------------
``ast_report(predictions, labels)``
    * ``predictions = (bslot_preds, value_preds)`` -- 2-D float arrays,
      ``np.argmax(..., axis=1)`` is applied inside.
    * ``labels = (bslot_labels, value_labels)`` -- 1-D int numpy arrays
      (``labels == argmax`` is an elementwise numpy comparison, so a plain list
      would silently produce a scalar ``False``).
    * EVERY denominator is ``len(bslot_preds)``, so the caller must pass
      take_action rows ONLY.

``cds_report(predictions, labels, ci_and_tc, kb_labels=None)``
    * ``predictions = (intent, nextstep, bslot, value, utterance_rank)`` -- five
      2-D float arrays; ``labels`` the five matching 1-D int arrays.
    * ``nextstep_label`` is an INDEX INTO :data:`NEXT_STEPS`; the cascade branches
      on ``== 0 / 1 / 2``.
    * ``utterance_label`` is a RANK in ``[0, n_candidates)`` compared against
      ``np.argpartition`` over the utterance score ROW, i.e.
      :attr:`NormalizedTurn.utt_rank` -- never the global ``utt_id``.
    * ``-1`` is the official "not applicable" marker: the action, value and
      utterance denominators are ``sum(label >= 0)``.
    * ``ci_and_tc = (convo_ids, turn_counts)`` and the official code calls
      ``.detach().cpu().numpy()`` on both, so they MUST be torch tensors.
    * ``kb_labels`` must stay ``None``: the ``components/`` package is missing
      from the download (see :func:`load_official_metrics`).

FIVE THINGS THE OFFICIAL SHAPES FORCE ON US, RECORDED HONESTLY
--------------------------------------------------------------
1. REFLEXIVE emits discrete predictions, not score vectors, so the prediction
   rows here are ONE-HOT. A one-hot's argmax is the prediction, which is all
   ``ast_report``/``cds_report`` read -- exact for every argmax-based metric.
2. ``Recall_at_5`` / ``Recall_at_10`` are DEGENERATE under a one-hot: four (nine)
   of the top-k slots are filled by tie-breaking, so a discrete predictor gets
   ``(1 - R@1) * 4/(width-1)`` of free credit. Only ``Recall_at_1`` is
   meaningful. ``metrics.json`` carries ``recall_at_k_degenerate: true`` and the
   estimated inflation; the report must not quote R@5/R@10 for this system.
3. The official ``joint_match = bslot_match & value_match`` with
   ``value_label = -1`` on no-value actions means a no-value action turn can
   NEVER be joint-correct: ``argmax`` cannot return ``-1``. That caps
   ``Joint_Accuracy`` at ``1 - (no-value action rows / action rows)``. This is
   the official code's behaviour, reproduced, not a bug introduced here. The cap
   is reported as ``joint_accuracy_ceiling``.
4. Action turns are EXPANDED one row per gold value position, exactly as
   ``utils/process.py`` does for ``verify-identity`` / ``validate-purchase``
   (verified: those are the only two 3-value buttons, 6,098 turns; and
   ``len(values) > 0`` iff the ontology requires values, 25,155/25,155).
   Unlike the official AST processor we never DROP a row whose value the copy
   mechanism could not locate, because we compare value STRINGS rather than
   re-running ABCD's tokenizer-dependent ``value_to_id``.
5. Value ids are assigned by a deterministic string table (ontology enumerable
   values first, in official order, then unseen strings in first-appearance
   order). The metric predicate is unchanged -- "predicted value equals gold
   value" -- but the integers differ from ABCD's copy-mechanism ids.

WHAT THE LOG CANNOT ANSWER (spec 8.5)
-------------------------------------
``Decision``/``GateOutput`` carry prediction-set SIZES, not membership, and no
probabilities at all. So ECE and exact per-head coverage are NOT derivable from
``decisions.jsonl``. :func:`calibration_metrics` therefore reports
* exact ECE and coverage when the optional side-car
  ``outputs/runs/<run_id>/<eval.probs_filename>`` is present, and
* COVERAGE BOUNDS otherwise (a singleton set contains the gold iff the
  prediction was correct; a set of size > 1 is unknown from the log),
never a guessed point value.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import math
import os
import sys
import types
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from reflex.config import get_dotted, resolve_path
from reflex.contracts import ContractViolation, OfficialMetrics, ReflexError
from reflex.schemas import NEXT_STEPS, Bank, Calibration, Decision, EvalRecord, RunManifest

__all__ = [
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
]


# ---------------------------------------------------------------------------
# Small structural constants. NOT thresholds: these are facts about the official
# code's shapes, not tunable knobs (spec 10 bans tunables in source, not shapes).
# ---------------------------------------------------------------------------

#: ``ranking_report`` / ``cds_report`` call ``np.argpartition(pred, kth=-10)``,
#: which requires at least ten columns in the utterance score row.
_MIN_UTTERANCE_WIDTH = 10

#: Heads the gate produces prediction sets for (spec 5.6 ``set_sizes`` keys).
_GATE_HEADS = ("nextstep", "intent", "action", "skeleton")

#: Components spec 8.3 asks for, in report order.
_FASTPATH_COMPONENTS = ("nextstep", "intent", "action_values", "utterance_recall_at_1", "turn")


# ---------------------------------------------------------------------------
# Official-metric loading (the components/ shim)
# ---------------------------------------------------------------------------

_OFFICIAL_CACHE: dict[str, Any] = {}


class _OfficialMetricsHandle:
    """Concrete :class:`OfficialMetrics`: holds the four REAL official functions."""

    def __init__(self, module: Any, source_path: str) -> None:
        self.ast_report = module.ast_report
        self.cds_report = module.cds_report
        self.ranking_report = module.ranking_report
        self.task_completion_report = module.task_completion_report
        self.source_path = source_path


def _install_components_shim() -> None:
    """Insert stub ``components.*`` modules so ``utils/evaluate.py`` imports.

    The download has only ``data/`` and ``utils/``. ``utils/evaluate.py`` imports
    ``components.systems.Application`` and (transitively, via ``utils/load.py``)
    ``components.tools``; ``utils/process.py`` wants ``components.datasets``.
    None of them are touched on the ``kb_labels is None`` path, so stubs whose
    members raise are enough -- and they FAIL LOUDLY if the KB path is ever
    entered, rather than silently returning a wrong mask.
    """
    for name in ("components", "components.systems", "components.tools", "components.datasets"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = module

    systems = sys.modules["components.systems"]
    if not hasattr(systems, "Application"):

        class Application:  # noqa: D401 - stub
            """Stub. Only ``cds_report(kb_labels=...)`` would use this."""

            @staticmethod
            def prepare_masks(*args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "components/ is not vendored: the KB-masked official metrics are "
                    "unavailable. eval.use_kb_labels must stay false."
                )

        systems.Application = Application  # type: ignore[attr-defined]

    tools = sys.modules["components.tools"]
    for attr in ("RAdam", "AdamW", "get_linear_schedule_with_warmup"):
        if not hasattr(tools, attr):
            setattr(tools, attr, _missing_component(f"components.tools.{attr}"))

    datasets = sys.modules["components.datasets"]
    for attr in ("ActionFeature", "CompletionFeature", "CascadeFeature"):
        if not hasattr(datasets, attr):
            setattr(datasets, attr, _missing_component(f"components.datasets.{attr}"))


def _missing_component(name: str) -> Any:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{name} is a shim stub; components/ is not vendored")

    return _raise


def load_official_metrics(cfg: dict[str, Any]) -> OfficialMetrics:
    """Import the REAL official ABCD metric functions. Spec 13 forbids reimplementing them.

    See the module docstring for the shapes these functions expect. The
    ``components/`` shim is installed here, once, and the imported module is
    cached per source path so repeated calls are free and deterministic.

    Args:
        cfg: Resolved config. Uses ``eval.official_utils_dir`` (the PARENT of
            ``utils/``) and ``eval.use_kb_labels``.

    Returns:
        An :class:`OfficialMetrics` holding the four real functions.

    Raises:
        FileNotFoundError: if ``utils/evaluate.py`` is not found.
        ContractViolation: if ``eval.use_kb_labels`` is true, or the imported
            module is missing one of the four reports.
    """
    if bool(get_dotted(cfg, "eval.use_kb_labels")):
        raise ContractViolation(
            "eval.use_kb_labels is true, but the KB-masked official metrics need the "
            "components/ package, which is not in the ABCD download. Vendor components/ "
            "from a full clone or keep eval.use_kb_labels: false."
        )

    base = resolve_path(cfg, "eval.official_utils_dir")
    source = os.path.join(base, "utils", "evaluate.py")
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"official ABCD evaluator not found at {source!r}. "
            "Set eval.official_utils_dir to the directory that CONTAINS utils/."
        )

    key = os.path.realpath(source)
    module = _OFFICIAL_CACHE.get(key)
    if module is None:
        _install_components_shim()
        added = False
        if base not in sys.path:
            sys.path.insert(0, base)
            added = True
        try:
            spec = importlib.util.spec_from_file_location("reflex._abcd_official_evaluate", source)
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                raise ContractViolation(f"could not build an import spec for {source!r}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if added and base in sys.path:
                sys.path.remove(base)
        _OFFICIAL_CACHE[key] = module

    missing = [
        name
        for name in ("ast_report", "cds_report", "ranking_report", "task_completion_report")
        if not hasattr(module, name)
    ]
    if missing:
        raise ContractViolation(f"{source!r} does not define {missing}; is this the official file?")
    return _OfficialMetricsHandle(module, key)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Label spaces. Prefer reflex.data (it owns ontology flattening); fall back to
# the official prepare_*_labels logic while data/ is still a stub so this module
# is testable on its own. The fallback is byte-for-byte the official flattening.
# ---------------------------------------------------------------------------


def _norm_value(value: Any) -> str:
    """Canonical form of a slot value for equality (official lowercases enumerables)."""
    return str(value).strip().lower()


class _LabelSpaces:
    """Deterministic string -> index maps for every head the official code scores."""

    def __init__(self, ontology: dict[str, Any]) -> None:
        self.nextsteps = list(NEXT_STEPS)
        ont_nextsteps = list(ontology.get("next_steps", self.nextsteps))
        if ont_nextsteps != self.nextsteps:
            raise ContractViolation(
                f"ontology['next_steps'] is {ont_nextsteps}, but cds_report branches on the "
                f"order {self.nextsteps}. Do not re-order NEXT_STEPS."
            )
        self.intents = _try_data("subflow_list", ontology, _fallback_subflows)
        self.actions = _try_data("action_list", ontology, _fallback_actions)
        self.intent_index = {name: i for i, name in enumerate(self.intents)}
        self.action_index = {name: i for i, name in enumerate(self.actions)}
        self.nextstep_index = {name: i for i, name in enumerate(self.nextsteps)}
        self.values: list[str] = []
        self.value_index: dict[str, int] = {}
        for category in ontology.get("values", {}).get("enumerable", {}).values():
            for val in category:
                self._value_id(val)

    def _value_id(self, value: Any) -> int:
        key = _norm_value(value)
        idx = self.value_index.get(key)
        if idx is None:
            idx = len(self.values)
            self.value_index[key] = idx
            self.values.append(key)
        return idx

    # -- prediction side: unknown strings collapse onto the sentinel column ----
    def pred_value_id(self, value: Any) -> int:
        if value is None:
            return -1
        return self._value_id(value)

    def gold_value_id(self, value: Any) -> int:
        return self._value_id(value)


def _fallback_subflows(ontology: dict[str, Any]) -> list[str]:
    """Official ``prepare_intent_labels``: flatten ``intents.subflows`` in file order."""
    out: list[str] = []
    for _flow, subflows in ontology["intents"]["subflows"].items():
        out.extend(subflows)
    return out


def _fallback_actions(ontology: dict[str, Any]) -> list[str]:
    """Official ``prepare_action_labels``: flatten the three action categories."""
    out: list[str] = []
    for _section, buttons in ontology["actions"].items():
        out.extend(buttons.keys())
    return out


def _try_data(func_name: str, ontology: dict[str, Any], fallback: Any) -> list[str]:
    """Call ``reflex.data.<func_name>``; use ``fallback`` while that module is a stub."""
    try:
        data_mod = importlib.import_module("reflex.data")
        func = getattr(data_mod, func_name)
        return list(func(ontology))
    except (ImportError, AttributeError, NotImplementedError):
        return list(fallback(ontology))


# ---------------------------------------------------------------------------
# Per-turn gold / prediction reading and correctness
# ---------------------------------------------------------------------------


def _gold_nextstep(record: EvalRecord) -> str:
    step = record.gold.get("nextstep")
    if step not in NEXT_STEPS:
        raise ContractViolation(
            f"gold nextstep {step!r} on convo {record.decision.convo_id} turn "
            f"{record.decision.turn_index} is not one of {NEXT_STEPS}"
        )
    return str(step)


def _gold_values(record: EvalRecord) -> list[str]:
    values = record.gold.get("values")
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values]


def _pred_values(record: EvalRecord) -> list[str]:
    values = record.decision.values
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values]


def _gold_utt_rank(record: EvalRecord) -> Optional[int]:
    """Gold utterance RANK. ``gold['utt_rank']`` wins; ``None`` when unavailable."""
    if "utt_rank" in record.gold and record.gold["utt_rank"] is not None:
        rank = int(record.gold["utt_rank"])
        return rank if rank >= 0 else None
    return None


def _turn_correct(record: EvalRecord) -> dict[str, Optional[bool]]:
    """Per-component correctness for one turn. ``None`` means "does not apply".

    The official reports return AGGREGATES only, but spec 8.1/8.3/8.4/8.7 need
    per-turn correctness on slices the official code cannot express (reflex turns
    only, the novel split, per-conversation bootstrap resamples, McNemar pairs).
    The turn-level predicate below is the same one the official
    ``task_completion_report`` uses, so the slices stay comparable to the
    headline numbers, which still come from the official functions.
    """
    dec = record.decision
    gold_step = _gold_nextstep(record)
    out: dict[str, Optional[bool]] = {
        "nextstep": dec.nextstep == gold_step,
        "intent": dec.intent == record.gold.get("intent"),
        "action": None,
        "values": None,
        "utterance": None,
    }
    if gold_step == "take_action":
        out["action"] = dec.action == record.gold.get("action")
        gold_vals = [_norm_value(v) for v in _gold_values(record)]
        pred_vals = [_norm_value(v) for v in _pred_values(record)]
        out["values"] = gold_vals == pred_vals
    elif gold_step == "retrieve_utterance":
        gold_rank = _gold_utt_rank(record)
        out["utterance"] = None if gold_rank is None else dec.candidate_rank == gold_rank

    if gold_step == "take_action":
        tail = bool(out["action"]) and bool(out["values"])
    elif gold_step == "retrieve_utterance":
        tail = bool(out["utterance"])
    else:
        tail = True
    out["turn"] = bool(out["nextstep"]) and bool(out["intent"]) and tail
    return out


def _turn_key(record: EvalRecord) -> tuple[int, int]:
    return (record.decision.convo_id, record.decision.turn_index)


# ---------------------------------------------------------------------------
# Array building (spec 8.2)
# ---------------------------------------------------------------------------


def _one_hot(indices: Sequence[int], width: int) -> np.ndarray:
    """One-hot rows; ``indices[i] < 0`` yields an all-zero row (never happens here)."""
    out = np.zeros((len(indices), width), dtype=np.float64)
    for row, idx in enumerate(indices):
        if 0 <= idx < width:
            out[row, idx] = 1.0
    return out


def _utterance_rows(pred_ranks: Sequence[int], width: int) -> np.ndarray:
    """Score rows for the utterance head.

    A discrete predictor gives one rank, so the row is ``1.0`` at that rank over a
    strictly DESCENDING ramp in ``(0, 0.5)``. The ramp only makes ``argpartition``
    tie-breaking explicit and reproducible; it carries no information, which is
    exactly why R@5/R@10 are flagged degenerate.
    """
    base = (np.arange(width - 1, -1, -1, dtype=np.float64) + 1.0) / (2.0 * (width + 1.0))
    out = np.tile(base, (len(pred_ranks), 1))
    for row, rank in enumerate(pred_ranks):
        if 0 <= rank < width:
            out[row, rank] = 1.0
        else:
            out[row, width - 1] = 1.0  # sentinel column: "no candidate predicted"
    return out


class _Rows:
    """Flat per-ROW arrays (one action turn can expand to three rows)."""

    def __init__(self) -> None:
        self.convo_ids: list[int] = []
        self.turn_counts: list[int] = []
        self.intent_label: list[int] = []
        self.intent_pred: list[int] = []
        self.nextstep_label: list[int] = []
        self.nextstep_pred: list[int] = []
        self.action_label: list[int] = []
        self.action_pred: list[int] = []
        self.value_label: list[int] = []
        self.value_pred: list[int] = []
        self.utt_label: list[int] = []
        self.utt_pred: list[int] = []
        self.missing_gold_rank = 0
        self.no_value_action_rows = 0
        self.action_rows = 0

    def __len__(self) -> int:
        return len(self.nextstep_label)


def _expand_rows(
    records: Sequence[EvalRecord],
    spaces: _LabelSpaces,
    action_only: bool,
) -> _Rows:
    """Turn records into official ROWS, replicating ``utils/process.py`` expansion."""
    rows = _Rows()
    n_actions = len(spaces.actions)
    for record in records:
        dec = record.decision
        gold_step = _gold_nextstep(record)
        if action_only and gold_step != "take_action":
            continue

        gold_intent = record.gold.get("intent")
        if gold_intent not in spaces.intent_index:
            raise ContractViolation(
                f"gold intent {gold_intent!r} is not in the 55-subflow label space "
                f"(convo {dec.convo_id} turn {dec.turn_index})"
            )
        intent_label = spaces.intent_index[gold_intent]
        intent_pred = spaces.intent_index.get(str(dec.intent), len(spaces.intents))
        nextstep_label = spaces.nextstep_index[gold_step]
        nextstep_pred = spaces.nextstep_index.get(str(dec.nextstep), len(spaces.nextsteps))

        turn_count = record.gold.get("turn_count")
        turn_count = int(turn_count) if turn_count is not None else int(dec.turn_index)

        if gold_step == "take_action":
            gold_action = record.gold.get("action")
            if gold_action not in spaces.action_index:
                raise ContractViolation(
                    f"gold action {gold_action!r} is not in the 30-button label space "
                    f"(convo {dec.convo_id} turn {dec.turn_index})"
                )
            action_label = spaces.action_index[gold_action]
            action_pred = spaces.action_index.get(str(dec.action), n_actions)
            gold_vals = _gold_values(record)
            pred_vals = _pred_values(record)
            n_rows = max(1, len(gold_vals))
            for position in range(n_rows):
                rows.action_rows += 1
                if position < len(gold_vals):
                    value_label = spaces.gold_value_id(gold_vals[position])
                else:  # the button takes no values: official sets value_id = -1
                    value_label = -1
                    rows.no_value_action_rows += 1
                if position < len(pred_vals):
                    value_pred = spaces.pred_value_id(pred_vals[position])
                else:
                    value_pred = -1  # -> sentinel column, never equals a gold id
                _append(
                    rows,
                    dec.convo_id,
                    turn_count,
                    intent_label,
                    intent_pred,
                    nextstep_label,
                    nextstep_pred,
                    action_label,
                    action_pred,
                    value_label,
                    value_pred,
                    -1,
                    -1,
                )
        else:
            utt_label, utt_pred = -1, -1
            if gold_step == "retrieve_utterance":
                gold_rank = _gold_utt_rank(record)
                if gold_rank is None:
                    rows.missing_gold_rank += 1
                else:
                    utt_label = gold_rank
                utt_pred = int(dec.candidate_rank)
            _append(
                rows,
                dec.convo_id,
                turn_count,
                intent_label,
                intent_pred,
                nextstep_label,
                nextstep_pred,
                -1,
                n_actions,
                -1,
                -1,
                utt_label,
                utt_pred,
            )
    return rows


def _append(rows: _Rows, *values: int) -> None:
    (
        convo_id,
        turn_count,
        intent_label,
        intent_pred,
        nextstep_label,
        nextstep_pred,
        action_label,
        action_pred,
        value_label,
        value_pred,
        utt_label,
        utt_pred,
    ) = values
    rows.convo_ids.append(convo_id)
    rows.turn_counts.append(turn_count)
    rows.intent_label.append(intent_label)
    rows.intent_pred.append(intent_pred)
    rows.nextstep_label.append(nextstep_label)
    rows.nextstep_pred.append(nextstep_pred)
    rows.action_label.append(action_label)
    rows.action_pred.append(action_pred)
    rows.value_label.append(value_label)
    rows.value_pred.append(value_pred)
    rows.utt_label.append(utt_label)
    rows.utt_pred.append(utt_pred)


def _utterance_width(rows: _Rows) -> int:
    """Width of the utterance score row, DERIVED from the data (ABCD always gives 100).

    ``+1`` reserves the last column for "no candidate predicted", so a turn where
    the system declined to rank cannot accidentally match gold rank 0.
    """
    observed = max([r for r in rows.utt_label] + [r for r in rows.utt_pred] + [-1])
    return max(_MIN_UTTERANCE_WIDTH, observed + 1) + 1


def build_ast_arrays(records: Sequence[EvalRecord], ontology: dict[str, Any]) -> tuple[Any, Any]:
    """Shape records into ``ast_report``'s expected arguments (spec 8.2).

    Filters to gold ``take_action`` turns (every AST denominator is
    ``len(bslot_preds)``), then expands one row per gold value position exactly
    as ``utils/process.py`` does.

    Args:
        records: Records for ONE (arm, model, seed, split).
        ontology: For the action and value label spaces.

    Returns:
        ``((bslot_preds, value_preds), (bslot_labels, value_labels))``, ready to
        splat into ``ast_report``. Prediction arrays are 2-D one-hot float
        arrays; label arrays are 1-D int numpy arrays with ``-1`` for N/A.
    """
    spaces = _LabelSpaces(ontology)
    rows = _expand_rows(records, spaces, action_only=True)
    action_width = len(spaces.actions) + 1
    value_width = len(spaces.values) + 1
    value_pred = [v if v >= 0 else value_width - 1 for v in rows.value_pred]
    predictions = (
        _one_hot(rows.action_pred, action_width),
        _one_hot(value_pred, value_width),
    )
    labels = (
        np.asarray(rows.action_label, dtype=np.int64),
        np.asarray(rows.value_label, dtype=np.int64),
    )
    return predictions, labels


def build_cds_arrays(records: Sequence[EvalRecord], ontology: dict[str, Any], bank: Bank) -> tuple[Any, Any, Any]:
    """Shape records into ``cds_report``'s expected arguments (spec 8.2).

    Includes the synthetic ``end_conversation`` turns (one third of the nextstep
    signal) and expands action turns per value position. ``ci_and_tc`` is a pair
    of TORCH TENSORS because the official code calls ``.detach().cpu().numpy()``
    on both. ``turn_counts`` prefers ``gold['turn_count']`` (ABCD's sparse count)
    and falls back to ``turn_index``; the official code uses it only to SORT
    within a conversation and both are monotone in turn order, so the cascade is
    identical either way.

    Args:
        records: Records for ONE (arm, model, seed, split).
        ontology: For intent / action / value label spaces.
        bank: Unused by the official code; accepted so callers need not branch.

    Returns:
        ``(predictions, labels, ci_and_tc)`` ready to splat into ``cds_report``.
    """
    del bank  # accepted for signature parity; the official code never reads it
    import torch  # local: keep module import cheap for callers that never score CDS

    spaces = _LabelSpaces(ontology)
    rows = _expand_rows(records, spaces, action_only=False)
    action_width = len(spaces.actions) + 1
    value_width = len(spaces.values) + 1
    utt_width = _utterance_width(rows)
    value_pred = [v if v >= 0 else value_width - 1 for v in rows.value_pred]

    predictions = (
        _one_hot(rows.intent_pred, len(spaces.intents) + 1),
        _one_hot(rows.nextstep_pred, len(spaces.nextsteps) + 1),
        _one_hot(rows.action_pred, action_width),
        _one_hot(value_pred, value_width),
        _utterance_rows(rows.utt_pred, utt_width),
    )
    labels = (
        np.asarray(rows.intent_label, dtype=np.int64),
        np.asarray(rows.nextstep_label, dtype=np.int64),
        np.asarray(rows.action_label, dtype=np.int64),
        np.asarray(rows.value_label, dtype=np.int64),
        np.asarray(rows.utt_label, dtype=np.int64),
    )
    ci_and_tc = (
        torch.tensor(rows.convo_ids, dtype=torch.long),
        torch.tensor(rows.turn_counts, dtype=torch.long),
    )
    return predictions, labels, ci_and_tc


def _array_diagnostics(records: Sequence[EvalRecord], ontology: dict[str, Any]) -> dict[str, Any]:
    """Facts about the arrays that the official reports do not return."""
    spaces = _LabelSpaces(ontology)
    rows = _expand_rows(records, spaces, action_only=False)
    utt_width = _utterance_width(rows)
    n_retrieve = sum(1 for lab in rows.utt_label if lab >= 0)
    ceiling = None
    if rows.action_rows:
        ceiling = 1.0 - rows.no_value_action_rows / rows.action_rows
    return {
        "n_rows": len(rows),
        "n_records": len(records),
        "n_action_rows": rows.action_rows,
        "n_no_value_action_rows": rows.no_value_action_rows,
        "joint_accuracy_ceiling": ceiling,
        "n_retrieve_rows_scored": n_retrieve,
        "n_retrieve_rows_missing_gold_rank": rows.missing_gold_rank,
        "utterance_row_width": utt_width,
        "recall_at_k_degenerate": True,
        "recall_at_5_max_tie_inflation": (5 - 1) / max(1, utt_width - 1),
        "recall_at_10_max_tie_inflation": (10 - 1) / max(1, utt_width - 1),
        "value_label_space_size": len(spaces.values),
    }


def ast_metrics(records: Sequence[EvalRecord], ontology: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    """Run the official ``ast_report`` and return its dict (spec 8.2).

    Returns an EMPTY dict when the records contain no gold ``take_action`` turn:
    every AST denominator would be zero and the official code would raise
    ``ZeroDivisionError``. The caller records that as "not applicable" rather
    than printing a zero.
    """
    official = load_official_metrics(cfg)
    predictions, labels = build_ast_arrays(records, ontology)
    if len(labels[0]) == 0:
        return {}
    report, _headline = official.ast_report(predictions, labels)
    return {str(k): float(v) for k, v in report.items()}


def cds_metrics(
    records: Sequence[EvalRecord],
    ontology: dict[str, Any],
    bank: Bank,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run the official ``cds_report`` and return its dict (spec 8.2).

    Note on cost: the official cascade groups by conversation with a nested
    Python loop (``O(conversations x rows)``), so a full test split takes tens of
    seconds. That is the official implementation; spec 13 forbids replacing it.
    """
    official = load_official_metrics(cfg)
    predictions, labels, ci_and_tc = build_cds_arrays(records, ontology, bank)
    if len(labels[0]) == 0:
        return {}
    report, _headline = official.cds_report(predictions, labels, ci_and_tc, None)
    return {str(k): float(v) for k, v in report.items()}


# ---------------------------------------------------------------------------
# 8.1 routing
# ---------------------------------------------------------------------------


def routing_metrics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute spec 8.1 routing metrics.

    ``reflex_rate`` = reflex turns / agent turns (overall and per GOLD nextstep);
    ``containment`` = conversations with no escalated turn / conversations;
    ``escalation_reason_share`` = distribution of ``GateOutput.reason`` over
    escalated turns. Arm A turns carry no gate, so they are counted under the
    reason ``"no_gate"`` -- Arm A's reflex_rate is 0 by construction (spec 5.5),
    which is correct accounting, not a bug.
    """
    del cfg
    n = len(records)
    reflex = [r for r in records if r.decision.route == "reflex"]
    escalated = [r for r in records if r.decision.route != "reflex"]

    by_step_total: Counter[str] = Counter()
    by_step_reflex: Counter[str] = Counter()
    for record in records:
        step = _gold_nextstep(record)
        by_step_total[step] += 1
        if record.decision.route == "reflex":
            by_step_reflex[step] += 1

    convos = {r.decision.convo_id for r in records}
    escalated_convos = {r.decision.convo_id for r in escalated}

    reasons: Counter[str] = Counter()
    for record in escalated:
        gate = record.decision.gate
        reasons[gate.reason if gate is not None else "no_gate"] += 1

    return {
        "reflex_rate": (len(reflex) / n) if n else 0.0,
        "reflex_rate_by_nextstep": {
            step: (by_step_reflex[step] / by_step_total[step]) for step in sorted(by_step_total)
        },
        "containment": ((len(convos) - len(escalated_convos)) / len(convos)) if convos else 0.0,
        "escalation_reason_share": {
            reason: count / len(escalated) for reason, count in sorted(reasons.items())
        },
        "n_agent_turns": n,
        "n_reflex_turns": len(reflex),
        "n_escalated_turns": len(escalated),
        "n_conversations": len(convos),
    }


# ---------------------------------------------------------------------------
# 8.3 fast-path-only quality
# ---------------------------------------------------------------------------


def _component_rates(records: Iterable[EvalRecord]) -> tuple[dict[str, float], dict[str, int]]:
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for record in records:
        flags = _turn_correct(record)
        for component in _FASTPATH_COMPONENTS:
            if component == "action_values":
                if flags["action"] is None:
                    continue
                ok = bool(flags["action"]) and bool(flags["values"])
            elif component == "utterance_recall_at_1":
                if flags["utterance"] is None:
                    continue
                ok = bool(flags["utterance"])
            else:
                value = flags[component]
                if value is None:
                    continue
                ok = bool(value)
            total[component] += 1
            correct[component] += int(ok)
    rates = {c: (1.0 - correct[c] / total[c]) if total[c] else float("nan") for c in _FASTPATH_COMPONENTS}
    return rates, dict(total)


def fastpath_metrics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute spec 8.3 fast-path-only quality.

    Error rate over REFLEX TURNS ONLY, per component, plus the
    ``exact_template_match`` rate. Escalated turns are excluded from the
    denominator: the point is what the fast path did when it chose to speak.
    """
    del cfg
    reflex = [r for r in records if r.decision.route == "reflex"]
    rates, denominators = _component_rates(reflex)
    matches = [r.decision.exact_template_match for r in reflex if r.decision.exact_template_match is not None]
    return {
        "fastpath_error_rate": rates,
        "fastpath_denominators": denominators,
        "exact_template_match_rate": (sum(1 for m in matches if m) / len(matches)) if matches else float("nan"),
        "n_exact_template_match_applicable": len(matches),
        "n_reflex_turns": len(reflex),
    }


# ---------------------------------------------------------------------------
# 8.4 novelty
# ---------------------------------------------------------------------------


def novelty_metrics(
    records_novel: Sequence[EvalRecord],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute spec 8.4 novelty metrics on ``test_novel``.

    ``novel_escalation_rate`` = escalated / agent turns;
    ``novel_fastpath_error_rate`` = turn-level-incorrect reflex / reflex turns.
    Spec 9 criterion 3 wants the escalation rate >= the configured floor.
    """
    del cfg
    n = len(records_novel)
    reflex = [r for r in records_novel if r.decision.route == "reflex"]
    escalated = n - len(reflex)
    wrong = sum(1 for r in reflex if not _turn_correct(r)["turn"])
    return {
        "novel_escalation_rate": (escalated / n) if n else float("nan"),
        "novel_fastpath_error_rate": (wrong / len(reflex)) if reflex else float("nan"),
        "n_agent_turns": n,
        "n_reflex_turns": len(reflex),
    }


# ---------------------------------------------------------------------------
# 8.5 calibration
# ---------------------------------------------------------------------------


def _ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int) -> float:
    """Expected calibration error over ``n_bins`` equal-width bins of confidence."""
    if not confidences:
        return float("nan")
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin i covers (edges[i], edges[i+1]]; the first bin also covers 0.0
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if not count:
            continue
        total += (count / len(conf)) * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def _probs_sidecar_path(cfg: dict[str, Any], run_id: str) -> str:
    runs_dir = resolve_path(cfg, "paths.runs_dir")
    return os.path.join(runs_dir, run_id, str(get_dotted(cfg, "eval.probs_filename")))


def _read_probs_sidecar(cfg: dict[str, Any], run_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = _probs_sidecar_path(cfg, run_id)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(int(row["convo_id"]), int(row["turn_index"]))] = row
    return out


def calibration_metrics(
    records: Sequence[EvalRecord],
    calibration: Calibration,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute spec 8.5 calibration metrics.

    EXACT ECE and coverage require per-head confidences and gold-in-set flags,
    which ``decisions.jsonl`` does not carry (``GateOutput`` logs set SIZES, not
    membership, and no probabilities). So:

    * if ``outputs/runs/<run_id>/<eval.probs_filename>`` exists, its rows --
      ``{"convo_id", "turn_index", "max_prob": {head: p}, "correct":
      {head: bool}, "gold_in_set": {head: bool}}`` -- give exact ECE over
      ``eval.ece_bins`` bins and exact empirical coverage per head;
    * otherwise ``ece`` is empty and ``empirical_coverage`` is ``None`` per head,
      with BOUNDS derived from the log instead: a singleton prediction set
      contains the gold iff the prediction was correct, and a set of size > 1 is
      unknown. A bound is not silently promoted to a point estimate.
    """
    n_bins = int(get_dotted(cfg, "eval.ece_bins"))
    run_ids = {r.run_id for r in records}
    run_id = next(iter(run_ids)) if len(run_ids) == 1 else ""
    sidecar = _read_probs_sidecar(cfg, run_id) if run_id else {}

    ece: dict[str, float] = {}
    coverage: dict[str, Optional[float]] = {}
    if sidecar:
        per_head_conf: dict[str, list[float]] = defaultdict(list)
        per_head_hit: dict[str, list[bool]] = defaultdict(list)
        per_head_cov: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            row = sidecar.get(_turn_key(record))
            if not row:
                continue
            for head, prob in (row.get("max_prob") or {}).items():
                hit = (row.get("correct") or {}).get(head)
                if prob is None or hit is None:
                    continue
                per_head_conf[head].append(float(prob))
                per_head_hit[head].append(bool(hit))
            for head, in_set in (row.get("gold_in_set") or {}).items():
                if in_set is not None:
                    per_head_cov[head].append(bool(in_set))
        ece = {head: _ece(per_head_conf[head], per_head_hit[head], n_bins) for head in sorted(per_head_conf)}
        coverage = {head: (sum(v) / len(v)) for head, v in sorted(per_head_cov.items()) if v}

    bounds: dict[str, dict[str, Any]] = {}
    for head in _GATE_HEADS:
        applicable = 0
        singleton_correct = 0
        unknown = 0
        for record in records:
            gate = record.decision.gate
            if gate is None:
                continue
            size = gate.set_sizes.get(head)
            if not isinstance(size, int) or size <= 0:
                continue
            applicable += 1
            if size == 1:
                flags = _turn_correct(record)
                key = "utterance" if head == "skeleton" else head
                flag = flags.get(key)
                if flag is None:
                    unknown += 1
                else:
                    singleton_correct += int(bool(flag))
            else:
                unknown += 1
        if applicable:
            low = singleton_correct / applicable
            bounds[head] = {
                "lower": low,
                "upper": low + unknown / applicable,
                "n_applicable": applicable,
                "n_unknown": unknown,
            }
        else:
            bounds[head] = {"lower": None, "upper": None, "n_applicable": 0, "n_unknown": 0}

    if not coverage:
        coverage = {head: None for head in _GATE_HEADS}

    return {
        "ece": ece,
        "empirical_coverage": coverage,
        "alpha": float(calibration.alpha),
        "coverage_bounds": bounds,
        "source": "probs_sidecar" if sidecar else "unavailable",
        "unavailable_reason": (
            ""
            if sidecar
            else (
                "decisions.jsonl carries prediction-set SIZES and no probabilities, so ECE "
                "and exact coverage are not derivable. Have the runner emit "
                f"{get_dotted(cfg, 'eval.probs_filename')!r} next to decisions.jsonl."
            )
        ),
        "ece_bins": n_bins,
        "n_turns": len(records),
    }


# ---------------------------------------------------------------------------
# 8.6 cost and latency
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float, method: str) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method=method))


def cost_latency_metrics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute spec 8.6 cost and latency.

    LLM spend goes through :func:`reflex.llm_agent.llm_cost_usd` (the price table
    lives in config and ships unfilled, so a run with billed tokens raises rather
    than reporting a free LLM). Cache hits are excluded from the spend and
    included in the latency denominator. Fast-path cost is wall-clock CPU seconds
    priced at ``cost.cpu_price_per_hour_usd``.

    Raises:
        PlaceholderConfigError: if LLM turns are present and prices are unfilled.
        ReflexError: if LLM turns are present but the model key cannot be
            resolved (no manifest), or ``llm_agent`` is still a stub.
    """
    method = str(get_dotted(cfg, "eval.percentile_method"))
    p_lo, p_hi = 50.0, 95.0
    n = len(records)
    convos = {r.decision.convo_id for r in records}

    tokens_in = sum(r.decision.llm_tokens_in for r in records if not r.decision.cache_hit)
    tokens_out = sum(r.decision.llm_tokens_out for r in records if not r.decision.cache_hit)
    n_llm_turns = sum(1 for r in records if r.decision.llm_tokens_in or r.decision.llm_tokens_out)
    n_cache_hits = sum(1 for r in records if r.decision.cache_hit)

    llm_cost = 0.0
    if tokens_in or tokens_out:
        model_key = _model_key_for(records, cfg)
        llm_cost = _llm_cost(tokens_in, tokens_out, model_key, cfg)

    cpu_seconds = sum(r.decision.latency_ms_fastpath for r in records) / 1000.0
    cpu_price = float(get_dotted(cfg, "cost.cpu_price_per_hour_usd"))
    fastpath_cost = cpu_seconds * cpu_price / 3600.0

    total_latency = [r.decision.latency_ms_fastpath + r.decision.latency_ms_llm for r in records]
    fastpath_latency = [r.decision.latency_ms_fastpath for r in records if r.decision.route == "reflex"]
    escalated_latency = [
        r.decision.latency_ms_fastpath + r.decision.latency_ms_llm
        for r in records
        if r.decision.route != "reflex"
    ]

    return {
        "llm_cost_per_turn": (llm_cost / n) if n else 0.0,
        "fastpath_cost_per_turn": (fastpath_cost / n) if n else 0.0,
        "arm_cost_per_turn": ((llm_cost + fastpath_cost) / n) if n else 0.0,
        "arm_cost_per_conversation": ((llm_cost + fastpath_cost) / len(convos)) if convos else 0.0,
        "latency_p50_ms": _percentile(total_latency, p_lo, method),
        "latency_p95_ms": _percentile(total_latency, p_hi, method),
        "latency_p50_ms_fastpath": _percentile(fastpath_latency, p_lo, method),
        "latency_p95_ms_fastpath": _percentile(fastpath_latency, p_hi, method),
        "latency_p50_ms_escalated": _percentile(escalated_latency, p_lo, method),
        "latency_p95_ms_escalated": _percentile(escalated_latency, p_hi, method),
        "llm_cost_usd_total": llm_cost,
        "fastpath_cost_usd_total": fastpath_cost,
        "llm_tokens_in_billed": int(tokens_in),
        "llm_tokens_out_billed": int(tokens_out),
        "n_llm_turns": n_llm_turns,
        "n_cache_hits": n_cache_hits,
        "cpu_seconds": cpu_seconds,
        "n_turns": n,
        "n_conversations": len(convos),
    }


def _llm_cost(tokens_in: int, tokens_out: int, model_key: str, cfg: dict[str, Any]) -> float:
    """Price billed tokens via the owning module (4.8), never with a local price."""
    try:
        llm_agent = importlib.import_module("reflex.llm_agent")
        return float(llm_agent.llm_cost_usd(int(tokens_in), int(tokens_out), model_key, cfg))
    except NotImplementedError as exc:
        raise ReflexError(
            "this run billed LLM tokens but reflex.llm_agent.llm_cost_usd is still a stub, "
            "so the spend cannot be priced. Implement module 4.8 before scoring a paid run."
        ) from exc


def _model_key_for(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> str:
    run_ids = {r.run_id for r in records}
    if len(run_ids) == 1:
        manifest = _try_read_manifest(cfg, next(iter(run_ids)))
        if manifest is not None and manifest.model_key:
            return str(manifest.model_key)
    raise ReflexError(
        "LLM tokens were billed but the model key is unknown: no single-run manifest.json "
        "was found next to decisions.jsonl, and Decision does not record it."
    )


# ---------------------------------------------------------------------------
# 8.7 statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values_by_conversation: dict[int, Sequence[float]],
    statistic: str,
    cfg: dict[str, Any],
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a CI **over conversations**, not turns (spec 8.7).

    Resampling turns would treat one conversation's ~9 agent turns as independent
    and shrink every interval. Conversation ids are resampled with replacement
    ``eval.n_bootstrap`` times at ``eval.ci``, seeded, percentile method.

    ``statistic`` is ``"mean"`` (turn-weighted micro average of the values) or
    ``"rate"`` (turn-weighted fraction of values that are non-zero). For 0/1
    correctness the two coincide; they differ for continuous values such as
    latency, where ``"rate"`` would be meaningless.

    Returns:
        ``(point_estimate, ci_low, ci_high)``; all NaN if there is no data.
    """
    if statistic not in ("mean", "rate"):
        raise ValueError(f"unknown statistic {statistic!r}; expected 'mean' or 'rate'")
    n_bootstrap = int(get_dotted(cfg, "eval.n_bootstrap"))
    ci = float(get_dotted(cfg, "eval.ci"))

    convo_ids = sorted(values_by_conversation)
    sums = np.zeros(len(convo_ids), dtype=np.float64)
    counts = np.zeros(len(convo_ids), dtype=np.float64)
    for i, convo_id in enumerate(convo_ids):
        vals = np.asarray(list(values_by_conversation[convo_id]), dtype=np.float64)
        if statistic == "rate":
            vals = (vals != 0.0).astype(np.float64)
        sums[i] = vals.sum()
        counts[i] = vals.size
    if counts.sum() == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = float(sums.sum() / counts.sum())
    if len(convo_ids) < 2 or n_bootstrap <= 0:
        return (point, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(convo_ids), size=(n_bootstrap, len(convo_ids)))
    boot_sums = sums[idx].sum(axis=1)
    boot_counts = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.where(boot_counts > 0, boot_sums / np.maximum(boot_counts, 1.0), np.nan)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:  # pragma: no cover - only if every resample is empty
        return (point, float("nan"), float("nan"))
    tail = (1.0 - ci) / 2.0
    lo = float(np.percentile(stats, 100.0 * tail))
    hi = float(np.percentile(stats, 100.0 * (1.0 - tail)))
    return (point, lo, hi)


def mcnemar_pvalue(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
) -> float:
    """McNemar test on per-turn correctness (spec 8.7). Exact binomial, two-sided.

    The two sequences must be the SAME TURNS IN THE SAME ORDER. The chi-square
    approximation misbehaves when discordant counts are small, which is exactly
    the regime a near-parity result sits in, so this uses the exact test.

    Returns:
        Two-sided p-value; ``1.0`` when there are no discordant pairs.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"McNemar needs paired sequences: got {len(correct_a)} and {len(correct_b)} turns"
        )
    b01 = sum(1 for a, b in zip(correct_a, correct_b) if (not a) and b)
    b10 = sum(1 for a, b in zip(correct_a, correct_b) if a and (not b))
    n = b01 + b10
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest  # type: ignore

        return float(binomtest(b01, n, 0.5, alternative="two-sided").pvalue)
    except ImportError:  # pragma: no cover - scipy is pinned in requirements.txt
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
        return float(min(1.0, 2.0 * tail))


def parity_delta(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    metric_name: str,
) -> float:
    """Return ``metric(Arm B) - metric(Arm A)`` in PERCENTAGE POINTS (spec 8.7).

    Spec 9 states tolerances in points while the official reports return
    fractions in ``[0, 1]``, so the x100 happens here, once.
    """
    if metric_name not in metrics_a:
        raise KeyError(f"{metric_name!r} is not in the Arm A metrics: {sorted(metrics_a)}")
    if metric_name not in metrics_b:
        raise KeyError(f"{metric_name!r} is not in the Arm B metrics: {sorted(metrics_b)}")
    return (float(metrics_b[metric_name]) - float(metrics_a[metric_name])) * 100.0


# ---------------------------------------------------------------------------
# Constant-predictor guard (standing project rule)
# ---------------------------------------------------------------------------


def _majority(values: Iterable[Any]) -> Optional[Any]:
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    # ties broken by the label's string form, so the guard is deterministic
    return sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _constant_records(records: Sequence[EvalRecord]) -> tuple[list[EvalRecord], dict[str, Any]]:
    """Rewrite every prediction to a LABEL-BLIND constant, keeping gold untouched.

    The constant: majority gold nextstep, majority gold intent, majority gold
    action, majority gold value (in every value position), and the FIRST
    candidate (rank 0) for utterance ranking. Routing is left as-is so the
    fast-path slices line up turn for turn.
    """
    nextstep = _majority(_gold_nextstep(r) for r in records) or NEXT_STEPS[0]
    intent = _majority(r.gold.get("intent") for r in records) or ""
    action = _majority(
        r.gold.get("action") for r in records if _gold_nextstep(r) == "take_action"
    )
    value = _majority(v for r in records for v in _gold_values(r))
    n_positions = max([len(_gold_values(r)) for r in records] + [1])

    out: list[EvalRecord] = []
    for record in records:
        decision = dataclasses.replace(
            record.decision,
            nextstep=str(nextstep),
            intent=str(intent),
            action=action,
            values=([str(value)] * n_positions) if value is not None else None,
            candidate_rank=0,
            candidate_utt_id=None,
            skeleton_id=None,
            template_ids=None,
            exact_template_match=None,
        )
        out.append(
            EvalRecord(
                decision=decision,
                gold=record.gold,
                correct={},
                seed=record.seed,
                run_id=record.run_id,
            )
        )
    description = {
        "nextstep": nextstep,
        "intent": intent,
        "action": action,
        "value": value,
        "utterance": "first candidate (rank 0)",
    }
    return out, description


def _constant_guard(
    records: Sequence[EvalRecord],
    ontology: dict[str, Any],
    bank: Bank,
    cfg: dict[str, Any],
    system: dict[str, Any],
) -> dict[str, Any]:
    """Score a label-blind constant on the same turns and flag every metric it wins.

    A metric a constant beats carries no signal: it is reported as DROPPED, not
    caveated. ``dropped`` lists ``"<family>.<metric>"`` names.
    """
    constant, description = _constant_records(records)
    const_cds = cds_metrics(constant, ontology, bank, cfg)
    const_ast = ast_metrics(constant, ontology, cfg)
    const_fast = fastpath_metrics(constant, cfg)

    dropped: list[str] = []
    comparisons: dict[str, Any] = {}
    for family, const_block, sys_block in (
        ("cds", const_cds, system.get("cds", {})),
        ("ast", const_ast, system.get("ast", {})),
    ):
        for key, const_value in const_block.items():
            sys_value = sys_block.get(key)
            if sys_value is None:
                continue
            wins = bool(const_value >= sys_value)
            comparisons[f"{family}.{key}"] = {
                "system": float(sys_value),
                "constant": float(const_value),
                "constant_wins": wins,
            }
            if wins:
                dropped.append(f"{family}.{key}")

    sys_fast = system.get("fastpath", {}).get("fastpath_error_rate", {})
    for key, const_err in const_fast["fastpath_error_rate"].items():
        sys_err = sys_fast.get(key)
        if sys_err is None or math.isnan(const_err) or math.isnan(float(sys_err)):
            continue
        wins = bool(const_err <= float(sys_err))  # lower error is better
        comparisons[f"fastpath.{key}_error_rate"] = {
            "system": float(sys_err),
            "constant": float(const_err),
            "constant_wins": wins,
        }
        if wins:
            dropped.append(f"fastpath.{key}_error_rate")

    return {
        "constant_predictor": description,
        "comparisons": comparisons,
        "dropped": sorted(set(dropped)),
        "flagged": bool(dropped),
        "note": (
            "A label-blind constant beating a metric means the metric carries no signal. "
            "Metrics listed in 'dropped' must be reported as dropped, not caveated."
        ),
    }


# ---------------------------------------------------------------------------
# Run-level I/O helpers (decisions.jsonl is owned by reflex.run; these prefer it
# and fall back to the frozen EvalRecord on-disk shape while run/ is a stub)
# ---------------------------------------------------------------------------


def _run_dir(cfg: dict[str, Any], run_id: str) -> str:
    return os.path.join(resolve_path(cfg, "paths.runs_dir"), run_id)


def _read_records(cfg: dict[str, Any], run_id: str) -> list[EvalRecord]:
    run_dir = _run_dir(cfg, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    try:
        run_mod = importlib.import_module("reflex.run")
        return list(run_mod.read_decisions(cfg, run_id))
    except NotImplementedError:
        pass
    path = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"decisions log not found: {path}")
    records: list[EvalRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EvalRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractViolation(f"{path}:{lineno} is not a spec 5.7 EvalRecord: {exc}") from exc
    if not records:
        raise ContractViolation(f"{path} is empty: nothing to score")
    return records


def _try_read_manifest(cfg: dict[str, Any], run_id: str) -> Optional[RunManifest]:
    try:
        run_mod = importlib.import_module("reflex.run")
        return run_mod.read_manifest(cfg, run_id)
    except (NotImplementedError, FileNotFoundError, KeyError, TypeError, ValueError):
        pass
    path = os.path.join(_run_dir(cfg, run_id), "manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return RunManifest.from_dict(json.load(fh))
    except (KeyError, TypeError, ValueError):
        return None


def _load_ontology(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        data_mod = importlib.import_module("reflex.data")
        return data_mod.load_ontology(cfg)
    except NotImplementedError:
        path = os.path.join(resolve_path(cfg, "data.abcd_dir"), "data", "ontology.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def _load_bank(cfg: dict[str, Any]) -> Bank:
    try:
        compile_mod = importlib.import_module("reflex.compile")
        return compile_mod.load_bank(cfg)
    except (NotImplementedError, FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        return Bank()


def _load_calibration(cfg: dict[str, Any]) -> Optional[Calibration]:
    try:
        calibrate_mod = importlib.import_module("reflex.calibrate")
        return calibrate_mod.load_calibration(cfg, None)
    except (NotImplementedError, FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        path = resolve_path(cfg, "paths.calibration_path")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return Calibration.from_dict(json.load(fh))
        except (KeyError, TypeError, ValueError):
            return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# evaluate_run / verdict
# ---------------------------------------------------------------------------


def evaluate_run(cfg: dict[str, Any], run_id: str, baseline_run_id: Optional[str] = None) -> dict[str, Any]:
    """Score one run end to end and write ``outputs/runs/<run_id>/metrics.json``.

    Runs every applicable spec 8 metric, the constant-predictor guard, and -- when
    ``baseline_run_id`` names a run over the SAME turns -- the spec 8.7 paired
    statistics (McNemar on per-turn correctness, parity deltas in points, and a
    conversation-level bootstrap CI of the delta).

    Bootstrap CIs pool every record in the log, so a log containing several
    training seeds is pooled across seeds, as spec 8.7 asks; ``seeds`` records
    which seeds were present.

    Raises:
        FileNotFoundError: if the run directory or its decisions log is missing.
        ContractViolation: if the two runs do not cover an identical turn set.
    """
    records = _read_records(cfg, run_id)
    manifest = _try_read_manifest(cfg, run_id)
    ontology = _load_ontology(cfg)
    bank = _load_bank(cfg)
    calibration = _load_calibration(cfg)

    arm = manifest.arm if manifest else _majority(r.decision.arm for r in records)
    split = manifest.split if manifest else ""
    seeds = sorted({r.seed for r in records})
    alpha = manifest.alpha if manifest else (calibration.alpha if calibration else float("nan"))

    official = {
        "cds": cds_metrics(records, ontology, bank, cfg),
        "ast": ast_metrics(records, ontology, cfg),
    }
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm,
        "model_key": manifest.model_key if manifest else "",
        "split": split,
        "seeds": seeds,
        "alpha": alpha,
        "n_records": len(records),
        "n_conversations": len({r.decision.convo_id for r in records}),
        "official": official,
        "official_notes": {
            "source": "wrapped utils/evaluate.py (spec 13: not reimplemented)",
            "kb_labels": "None -- components/ is not vendored, so the KB-masked variant is unavailable",
            "arrays": _array_diagnostics(records, ontology),
        },
        "routing": routing_metrics(records, cfg),
        "fastpath": fastpath_metrics(records, cfg),
        "cost_latency": cost_latency_metrics(records, cfg),
        "log_correct_mismatch": _log_correct_mismatch(records),
    }
    if split == "test_novel" or not split:
        metrics["novelty"] = novelty_metrics(records, cfg)
    if calibration is not None:
        metrics["calibration"] = calibration_metrics(records, calibration, cfg)
    else:
        metrics["calibration"] = {
            "ece": {},
            "empirical_coverage": {head: None for head in _GATE_HEADS},
            "alpha": None,
            "source": "unavailable",
            "unavailable_reason": "no calibration file; run `reflex calibrate` first",
        }

    metrics["constant_guard"] = _constant_guard(records, ontology, bank, cfg, metrics)
    metrics["statistics"] = _statistics(records, cfg)

    if baseline_run_id:
        metrics["parity"] = _parity_block(cfg, records, baseline_run_id, ontology, bank, official)

    metrics["verdict"] = verdict(metrics, cfg)
    metrics["dropped_metrics"] = list(metrics["constant_guard"]["dropped"])

    out_path = os.path.join(_run_dir(cfg, run_id), "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(metrics), fh, indent=2, sort_keys=False)
        fh.write("\n")
    metrics["metrics_path"] = out_path
    return metrics


def _log_correct_mismatch(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Cross-check the runner's logged ``correct`` flags against ours.

    The Evaluator owns correctness (spec 4.9), so a disagreement is a bug in the
    RUNNER, not something to absorb quietly.
    """
    mismatches: Counter[str] = Counter()
    checked = 0
    for record in records:
        ours = _turn_correct(record)
        for key, logged in (record.correct or {}).items():
            if key not in ours or ours[key] is None:
                continue
            checked += 1
            if bool(logged) != bool(ours[key]):
                mismatches[key] += 1
    return {"n_compared": checked, "mismatches": dict(mismatches), "clean": not mismatches}


def _statistics(records: Sequence[EvalRecord], cfg: dict[str, Any]) -> dict[str, Any]:
    """Conversation-level bootstrap CIs for the per-turn correctness components."""
    out: dict[str, Any] = {"bootstrap": {}, "n_bootstrap": int(get_dotted(cfg, "eval.n_bootstrap")),
                           "ci": float(get_dotted(cfg, "eval.ci"))}
    for component in _FASTPATH_COMPONENTS:
        by_convo: dict[int, list[float]] = defaultdict(list)
        for record in records:
            flags = _turn_correct(record)
            if component == "action_values":
                if flags["action"] is None:
                    continue
                ok = bool(flags["action"]) and bool(flags["values"])
            elif component == "utterance_recall_at_1":
                if flags["utterance"] is None:
                    continue
                ok = bool(flags["utterance"])
            else:
                if flags[component] is None:
                    continue
                ok = bool(flags[component])
            by_convo[record.decision.convo_id].append(float(ok))
        if by_convo:
            point, lo, hi = bootstrap_ci(by_convo, "mean", cfg, seed=0)
            out["bootstrap"][component] = {"point": point, "ci_low": lo, "ci_high": hi}
    return out


def _parity_block(
    cfg: dict[str, Any],
    records: Sequence[EvalRecord],
    baseline_run_id: str,
    ontology: dict[str, Any],
    bank: Bank,
    official: dict[str, Any],
) -> dict[str, Any]:
    """Spec 8.7 paired statistics against an Arm A run over the identical turns."""
    baseline = _read_records(cfg, baseline_run_id)
    keys_b = [_turn_key(r) for r in records]
    keys_a = [_turn_key(r) for r in baseline]
    if set(keys_a) != set(keys_b) or len(keys_a) != len(keys_b):
        only_b = sorted(set(keys_b) - set(keys_a))[:5]
        only_a = sorted(set(keys_a) - set(keys_b))[:5]
        raise ContractViolation(
            "spec 8.2 requires both arms scored on identical turns, but the turn sets differ: "
            f"{len(keys_b)} vs {len(keys_a)} rows; examples only in this run {only_b}, "
            f"only in the baseline {only_a}"
        )
    by_key = {_turn_key(r): r for r in baseline}
    paired_a = [by_key[k] for k in keys_b]

    correct_b = [bool(_turn_correct(r)["turn"]) for r in records]
    correct_a = [bool(_turn_correct(r)["turn"]) for r in paired_a]
    base_official = {
        "cds": cds_metrics(paired_a, ontology, bank, cfg),
        "ast": ast_metrics(paired_a, ontology, cfg),
    }

    deltas: dict[str, float] = {}
    for family in ("cds", "ast"):
        for key in official.get(family, {}):
            if key in base_official.get(family, {}):
                deltas[f"{family}.{key}"] = parity_delta(base_official[family], official[family], key)

    diff_by_convo: dict[int, list[float]] = defaultdict(list)
    for record, a_ok, b_ok in zip(records, correct_a, correct_b):
        diff_by_convo[record.decision.convo_id].append(float(b_ok) - float(a_ok))
    point, lo, hi = bootstrap_ci(diff_by_convo, "mean", cfg, seed=0)

    return {
        "baseline_run_id": baseline_run_id,
        "baseline_official": base_official,
        "delta_points": deltas,
        "turn_accuracy_delta_points": point * 100.0,
        "turn_accuracy_delta_ci_low_points": lo * 100.0,
        "turn_accuracy_delta_ci_high_points": hi * 100.0,
        "mcnemar_p": mcnemar_pvalue(correct_a, correct_b),
        "n_paired_turns": len(correct_b),
    }


def verdict(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the spec 9 PASS / PARTIAL / STOP criteria. Computation, not rendering.

    Every threshold is read from the ``verdict`` config block (spec 10). A
    criterion whose input is unavailable gets ``passed: null`` and is listed in
    ``unavailable``: an unmeasured criterion is never a pass. Spec 9's last line
    is binding -- this function reports, it never adjusts an input.
    """
    thresholds = get_dotted(cfg, "verdict")
    criteria: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Optional[float], threshold: float, ok: Optional[bool]) -> None:
        criteria[name] = {"passed": ok, "value": value, "threshold": threshold}

    reflex_rate = metrics.get("routing", {}).get("reflex_rate")
    min_reflex = float(thresholds["min_reflex_rate"])
    add("reflex_rate", reflex_rate, min_reflex, None if reflex_rate is None else reflex_rate >= min_reflex)

    parity = metrics.get("parity")
    tol = float(thresholds["parity_tolerance_points"])
    ci_floor = float(thresholds["parity_ci_lower_points"])
    parity_metrics = dict(thresholds["parity_metrics"])
    for family, metric_name in sorted(parity_metrics.items()):
        key = f"{family}.{metric_name}"
        delta = (parity or {}).get("delta_points", {}).get(key)
        add(f"parity_{key}", delta, -tol, None if delta is None else delta >= -tol)
    ci_low = (parity or {}).get("turn_accuracy_delta_ci_low_points")
    add("parity_delta_ci_low", ci_low, ci_floor, None if ci_low is None else ci_low >= ci_floor)

    novel = metrics.get("novelty", {}).get("novel_escalation_rate")
    min_novel = float(thresholds["min_novel_escalation_rate"])
    novel_ok = None if novel is None or (isinstance(novel, float) and math.isnan(novel)) else novel >= min_novel
    add("novel_escalation_rate", novel, min_novel, novel_ok)

    alpha = metrics.get("calibration", {}).get("alpha")
    slack = float(thresholds["coverage_slack"])
    coverage = metrics.get("calibration", {}).get("empirical_coverage", {}) or {}
    floor = (1.0 - float(alpha) - slack) if isinstance(alpha, (int, float)) else float("nan")
    observed = [v for v in coverage.values() if isinstance(v, (int, float))]
    if observed and not math.isnan(floor):
        worst = min(observed)
        add("empirical_coverage_min", worst, floor, worst >= floor)
    else:
        add("empirical_coverage_min", None, floor, None)

    p95 = metrics.get("cost_latency", {}).get("latency_p95_ms_fastpath")
    max_p95 = float(thresholds["max_fastpath_p95_latency_ms"])
    p95_ok = None if p95 is None or (isinstance(p95, float) and math.isnan(p95)) else p95 <= max_p95
    add("fastpath_p95_latency_ms", p95, max_p95, p95_ok)

    # DECISIONS D3: latency is a BUDGET, not a gate. When
    # `latency.gate_on_latency` is false the criterion is measured and REPORTED
    # but must not drag the verdict. Without this the p95 row entered all()
    # below and turned a PASS into PARTIAL on a criterion the maintainer had
    # explicitly demoted -- report.py labelled the row REPORT-ONLY while the
    # verdict string in metrics.json still said otherwise.
    gate_on_latency = bool(get_dotted(cfg, "latency.gate_on_latency"))
    criteria["fastpath_p95_latency_ms"]["gates_verdict"] = gate_on_latency
    verdict_criteria = {
        name: c for name, c in criteria.items()
        if c.get("gates_verdict", True)
    }

    unavailable = sorted(name for name, c in criteria.items() if c["passed"] is None)
    stop_floor = float(thresholds["stop_reflex_rate"])
    if reflex_rate is not None and reflex_rate < stop_floor:
        result = "STOP"
    elif all(c["passed"] for c in verdict_criteria.values()):
        result = "PASS"
    else:
        result = "PARTIAL"

    return {
        "verdict": result,
        "criteria": criteria,
        "unavailable": unavailable,
        "stop_threshold": {"reflex_rate": stop_floor},
        "dropped_metrics": list(metrics.get("constant_guard", {}).get("dropped", [])),
        "note": (
            "Criteria with passed=null were not measurable from this run's artifacts; an "
            "unmeasured criterion never counts as a pass."
        ),
    }
