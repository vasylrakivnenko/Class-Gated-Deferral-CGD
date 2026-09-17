"""Run orchestration for Arm A / Arm B over a split. Owns run_id, the manifest and
`decisions.jsonl` I/O.

WHAT THIS MODULE IS ALLOWED TO DO
---------------------------------
Sequence the other modules over a turn set, time them, and write two files. It
computes no metric (spec 10: "nothing is computed from logs outside the
Evaluator"), makes no routing decision (spec 4.6: that is the gate's only job)
and never touches a score.

THE THREE MODES, AND WHY THE THIRD ONE EXISTS
---------------------------------------------
``llm.enabled`` defaults to false -- ZERO PAID API CALLS is a hard project rule
by default -- so the arms are not symmetric while it stays off:

``arm="A"``
    The LLM answers every turn (spec 6.9) and every decision is
    ``route="escalated"`` for accounting (spec 5.5). It cannot run while the
    kill switch is on and dies at :func:`reflex.llm_agent.build_llm_agent`
    BEFORE any data is loaded, in about a second.

``arm="B"``
    select -> gate -> (fill | escalate). Fully runnable and free until the
    first escalation. If ``llm.enabled`` is true (and the price table is
    filled), the escalation is handed to :func:`reflex.llm_agent.llm_decide`
    and actually answered -- the agent handle is built once, up front, and
    reused across every escalated turn (spec 6.9 step 5's response cache and
    the running spend counter both depend on that). If ``llm.enabled`` is
    false, an escalation has nobody to answer it and raises, unless
    ``run.forced_reflex`` is set (below).

``arm="B"`` with ``run.forced_reflex``, ``llm.enabled`` still false
    **<-- the one that pays for itself**
    The gate still runs and its REAL verdict is recorded -- route, reason,
    prediction-set sizes, novelty distance -- but an escalated turn is not
    handed to anyone. It is logged unanswered: no utterance, no candidate, zero
    tokens. Every spec 8.1 routing metric, the escalation-reason distribution,
    spec 8.4 novelty escalation and per-nextstep coverage then come out of a $0
    run. **Arm-level QUALITY does not**, because nobody answered the escalated
    turns, and the manifest says so in ``forced_reflex`` so no reader can mistake
    the quality columns of such a run for Arm B's real quality.

LATENCY NUMBERS FROM THIS MODULE ARE CONTENDED
-----------------------------------------------
Fast-path and escalated wall clock are timed separately and written to separate
:class:`~reflex.schemas.Decision` fields, batch 1, CPU, ``runtime.torch_threads``
threads. They were produced on a machine that has been swapping hard enough to
take one fine-tune step from 0.306s to 241s. Every wall-clock number this module
emits is therefore an UPPER bound under contention, and the manifest records the
machine so the report can label it. Ratios survive contention; absolute
milliseconds do not.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Optional

from reflex.config import get_dotted, repo_root, resolve_path
from reflex.contracts import ContractViolation, LLMDisabledError
from reflex.schemas import (
    NEXT_STEPS,
    Calibration,
    Decision,
    EvalRecord,
    GateOutput,
    LLMDecision,
    NormalizedTurn,
    RunManifest,
    Selection,
    SelectorScores,
)

__all__ = [
    "new_run_id",
    "write_manifest",
    "read_manifest",
    "append_decisions",
    "read_decisions",
    "run_arm",
    "run_sweep",
]

#: Filenames the contract fixes by name (``outputs/runs/<run_id>/manifest.json``
#: and ``.../decisions.jsonl``). :mod:`reflex.evaluate` opens both by these exact
#: names as its fallback reader, so they are a shared constant, not a knob.
_MANIFEST_FILENAME = "manifest.json"
_DECISIONS_FILENAME = "decisions.jsonl"

#: The run-id timestamp format from the contract.
_RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"

#: Arms and model keys the CLI accepts. Spec 5.5 / 11.2.
_ARMS = ("A", "B")
_MODEL_KEYS = ("strong", "cheap")

#: Heads whose per-turn confidence the optional probs sidecar records
#: (``eval.probs_filename``). H7 is per act position and has no single scalar.
_SIDECAR_HEADS = ("nextstep", "intent", "action", "skeleton")


# --------------------------------------------------------------------------- #
# run_id, manifest, decisions log
# --------------------------------------------------------------------------- #


def new_run_id(cfg: dict[str, Any], arm: str, model_key: str, split: str, seed: int) -> str:
    """Mint a run id (spec 10: "every run has a run_id").

    See :func:`reflex.contracts.new_run_id` for the frozen contract. The format
    is ``"{UTC yyyymmddTHHMMSSZ}-arm{arm}-{model_key}-{split}-s{seed}"``.
    """
    del cfg
    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime(_RUN_ID_TIME_FORMAT)
    return f"{stamp}-arm{arm}-{model_key}-{split}-s{int(seed)}"


def _run_dir(cfg: dict[str, Any], run_id: str) -> str:
    return os.path.join(resolve_path(cfg, "paths.runs_dir"), run_id)


def write_manifest(manifest: RunManifest, cfg: dict[str, Any]) -> str:
    """Write ``outputs/runs/<run_id>/manifest.json`` (spec 10).

    See :func:`reflex.contracts.write_manifest` for the frozen contract. Written
    BEFORE the first decision, so an aborted run is still identifiable.
    """
    directory = _run_dir(cfg, manifest.run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, _MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    return os.path.abspath(path)


def read_manifest(cfg: dict[str, Any], run_id: str) -> RunManifest:
    """Read a run's manifest.

    See :func:`reflex.contracts.read_manifest` for the frozen contract.

    Raises:
        FileNotFoundError: if absent.
    """
    path = os.path.join(_run_dir(cfg, run_id), _MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return RunManifest.from_dict(json.load(handle))


def append_decisions(records: Iterable[EvalRecord], cfg: dict[str, Any], run_id: str) -> int:
    """Append records to ``outputs/runs/<run_id>/decisions.jsonl`` (spec 10).

    See :func:`reflex.contracts.append_decisions` for the frozen contract.
    One JSON object per line in the flattened spec 5.7 shape. Spec 14 requires
    one record per turn per arm, including turns the LLM failed to parse and
    turns the gate escalated: **nothing is ever dropped**.
    """
    directory = _run_dir(cfg, run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, _DECISIONS_FILENAME)
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=False, default=str))
            handle.write("\n")
            written += 1
    return written


def read_decisions(cfg: dict[str, Any], run_id: str) -> list[EvalRecord]:
    """Read a run's decisions log.

    See :func:`reflex.contracts.read_decisions` for the frozen contract. I/O
    only -- spec 10: "Nothing is computed from logs outside the Evaluator."

    Raises:
        FileNotFoundError: if the log is absent.
    """
    path = os.path.join(_run_dir(cfg, run_id), _DECISIONS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"decisions log not found: {path}")
    records: list[EvalRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EvalRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractViolation(
                    f"{path}:{lineno} is not a spec 5.7 EvalRecord: {exc}"
                ) from exc
    return records


# --------------------------------------------------------------------------- #
# Manifest ingredients
# --------------------------------------------------------------------------- #


def _git_commit() -> str:
    """HEAD sha, or ``"unknown"`` outside a git work tree (spec 10)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_root(), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = (out.stdout or "").strip()
    return sha if out.returncode == 0 and sha else "unknown"


def _machine() -> dict[str, Any]:
    """Spec 10: "document the actual machine"."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": "unavailable",
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return info


def _prompt_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    """sha256(16) of each frozen prompt file; ``"absent"`` when a file is missing."""
    out: dict[str, str] = {}
    for name, key in (("agent_A", "paths.prompt_agent_a"), ("act_labeling", "paths.prompt_act_labeling")):
        try:
            path = resolve_path(cfg, key)
        except Exception:
            out[name] = "unresolved"
            continue
        if not os.path.exists(path):
            out[name] = "absent"
            continue
        with open(path, "rb") as handle:
            out[name] = hashlib.sha256(handle.read()).hexdigest()[:16]
    return out


def _resolved_model_id(cfg: dict[str, Any], arm: str, model_key: str) -> str:
    """The model id this run actually uses.

    Arm B's fast path is the local encoder, not an LLM; recording ``llm.strong``
    for a run that never calls it would make the manifest claim a dependency the
    run does not have. The LLM id is recorded for Arm B only as the escalation
    target, and only when it is filled.
    """
    if arm == "B":
        return str(get_dotted(cfg, "model.encoder"))
    value = get_dotted(cfg, f"llm.{model_key}")
    return str(value)


# --------------------------------------------------------------------------- #
# Correctness flags logged beside each decision
# --------------------------------------------------------------------------- #


def _norm_value(value: Any) -> str:
    """Canonical form of a slot value for equality. Mirrors the Evaluator's rule."""
    return str(value).strip().lower()


def _correct_flags(decision: Decision, gold: dict[str, Any]) -> dict[str, bool]:
    """Per-component correctness for one turn, logged for spec 5.7.

    The EVALUATOR owns correctness (spec 4.9) and recomputes all of this; it also
    cross-checks these flags and reports any disagreement as a bug in the RUNNER.
    This is written as an independent implementation of the same rule rather than
    a call into ``reflex.evaluate``, so that the cross-check can actually fail.

    ``False`` is used where the Evaluator uses ``None`` ("does not apply"),
    because :attr:`EvalRecord.correct` is typed ``dict[str, bool]``; the
    Evaluator only compares the keys it considers applicable.
    """
    gold_step = str(gold.get("nextstep"))
    flags = {
        "nextstep": decision.nextstep == gold_step,
        "intent": decision.intent == gold.get("intent"),
        "action": False,
        "values": False,
        "utterance": False,
    }
    if gold_step == "take_action":
        flags["action"] = decision.action == gold.get("action")
        gold_values = [_norm_value(v) for v in (gold.get("values") or [])]
        pred_values = [_norm_value(v) for v in (decision.values or [])]
        flags["values"] = gold_values == pred_values
    elif gold_step == "retrieve_utterance":
        gold_rank = gold.get("utt_rank")
        flags["utterance"] = gold_rank is not None and int(gold_rank) >= 0 and decision.candidate_rank == int(gold_rank)
    return flags


def _gold_for(turn: NormalizedTurn) -> dict[str, Any]:
    """Spec 5.7's five gold keys, plus ``utt_rank`` (the Evaluator prefers it)."""
    return {
        "nextstep": turn.nextstep,
        "intent": turn.intent,
        "action": turn.action,
        "values": list(turn.values) if turn.values is not None else None,
        "utt_id": turn.utt_id,
        "utt_rank": turn.utt_rank,
    }


# --------------------------------------------------------------------------- #
# Arm B turn loop
# --------------------------------------------------------------------------- #


def _unseen_action_map(partition: dict[int, list[NormalizedTurn]]) -> dict[str, set[str]]:
    """``intent -> actions observed for it in TRAIN``, for gate signal 3.

    Built from the training partition only. Building it from the split being
    scored would make ``unseen_action`` un-triggerable on exactly the turns it
    exists to catch.
    """
    seen: dict[str, set[str]] = {}
    for turns in partition.values():
        for turn in turns:
            if turn.intent and turn.action:
                seen.setdefault(str(turn.intent), set()).add(str(turn.action))
    return seen


def _required_slots_for(selection: Selection, bank: Any) -> list[str]:
    """Slots the chosen answer needs: the action's pattern, or the templates'.

    Spec 6.6 signal 3 is about what the fast path is ABOUT to say, so the
    requirement follows the selection, not the gold.
    """
    required: list[str] = []
    if selection.nextstep == "take_action" and selection.action:
        for pattern in bank.actions:
            if pattern.action == selection.action:
                required.extend(str(s) for s in pattern.required_slots)
                break
    elif selection.nextstep == "retrieve_utterance" and selection.template_ids:
        by_id = {t.template_id: t for t in bank.templates}
        for template_id in selection.template_ids:
            template = by_id.get(template_id)
            if template is not None:
                required.extend(str(s) for s in template.slots)
    out: list[str] = []
    for slot in required:
        if slot not in out:
            out.append(slot)
    return out


def _templates_for(selection: Selection, bank: Any) -> list[Any]:
    by_id = {t.template_id: t for t in bank.templates}
    return [by_id[tid] for tid in (selection.template_ids or []) if tid in by_id]


def _sidecar_row(
    scores: SelectorScores,
    gate_output: GateOutput,
    turn: NormalizedTurn,
    class_orders: dict[str, list[str]],
    calibration: Calibration,
    gold_skeleton: Optional[str],
) -> dict[str, Any]:
    """One ``eval.probs_filename`` row: max prob, correctness and gold-in-set per head.

    ``decisions.jsonl`` logs prediction-set SIZES and no probabilities, so spec
    8.5's ECE and exact per-head coverage are not derivable from it. This sidecar
    is the only thing that makes them exact instead of bounded.
    """
    from reflex.select import prediction_set

    gold_by_head = {
        "nextstep": turn.nextstep,
        "intent": turn.intent,
        "action": turn.action,
        "skeleton": gold_skeleton,
    }
    probs_by_head = {
        "nextstep": scores.nextstep_probs,
        "intent": scores.intent_probs,
        "action": scores.action_probs,
        "skeleton": scores.skeleton_probs,
    }
    max_prob: dict[str, float] = {}
    correct: dict[str, bool] = {}
    gold_in_set: dict[str, bool] = {}
    for head in _SIDECAR_HEADS:
        probs = list(probs_by_head[head] or [])
        if not probs or not gate_output.set_sizes.get(head):
            continue
        order = class_orders.get(head, [])
        gold = gold_by_head.get(head)
        if gold is None or gold not in order:
            continue
        gold_index = order.index(gold)
        best = max(range(len(probs)), key=lambda i: probs[i])
        max_prob[head] = float(probs[best])
        correct[head] = bool(best == gold_index)
        q = float((calibration.quantiles or {}).get(head, 0.0))
        gold_in_set[head] = bool(gold_index in prediction_set(probs, q))
    return {
        "convo_id": int(turn.convo_id),
        "turn_index": int(turn.turn_index),
        "max_prob": max_prob,
        "correct": correct,
        "gold_in_set": gold_in_set,
    }


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

    See :func:`reflex.contracts.run_arm` for the frozen contract.

    Deviation from the contract's Arm B paragraph, made deliberately and recorded
    in the manifest: the contract says an escalation "raises unless
    ``llm.dry_run``". ``llm.dry_run`` cannot help while ``llm.enabled`` is false,
    because :func:`reflex.llm_agent.build_llm_agent` refuses at CONSTRUCTION --
    correctly, since a dry run still needs a model id and a price table it must
    not invent. So a third mode exists: ``run.forced_reflex`` records the
    escalation unanswered and keeps going, which is what makes every routing
    metric obtainable for $0.

    Raises:
        LLMDisabledError: for Arm A, or for an Arm B escalation, while the LLM is
            disabled and ``run.forced_reflex`` is off.
        ValueError: on an unknown arm/model_key, or Arm B without a checkpoint.
        FileNotFoundError: if the checkpoint or the calibration is absent.
    """
    if arm not in _ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {_ARMS}")
    if model_key not in _MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {_MODEL_KEYS}")

    forced_reflex = bool(get_dotted(cfg, "run.forced_reflex"))

    if arm == "A":
        # Fail before loading 116MB of JSON: a run that was never authorized to
        # spend money should die in a second, not after an hour of setup.
        from reflex.llm_agent import build_llm_agent

        build_llm_agent(cfg, model_key)  # raises LLMDisabledError while the switch is off
        raise ContractViolation(
            "Arm A reached its turn loop with the LLM enabled, but this build has "
            "never been authorized to make a paid call. Refusing to proceed."
        )

    if not checkpoint_path:
        raise ValueError("Arm B requires --checkpoint: there is no fast path without a model")

    return _run_arm_b(
        cfg,
        model_key=model_key,
        split=split,
        seed=seed,
        checkpoint_path=checkpoint_path,
        alpha=alpha,
        forced_reflex=forced_reflex,
    )


def _resolve_calibration(cfg: dict[str, Any], alpha: Optional[float]) -> Calibration:
    """Load the calibration and, for E5, swap in the sweep quantiles for ``alpha``."""
    from reflex.calibrate import load_calibration

    calibration = load_calibration(cfg, None)
    if alpha is None:
        return calibration
    key = repr(float(alpha))
    sweep = calibration.alpha_sweep_quantiles or {}
    quantiles = sweep.get(key)
    if quantiles is None:
        # Try the other float spellings JSON round-tripping can produce.
        for candidate, value in sweep.items():
            if abs(float(candidate) - float(alpha)) < 1e-12:
                quantiles = value
                break
    if quantiles is None:
        raise ContractViolation(
            f"alpha={alpha} was requested but the calibration carries no quantiles for it "
            f"(has {sorted(sweep)}). Add it to gate.alpha_sweep and re-run `reflex calibrate`; "
            "quantiles are not interpolable."
        )
    import dataclasses

    return dataclasses.replace(calibration, alpha=float(alpha), quantiles=dict(quantiles))


def _run_arm_b(
    cfg: dict[str, Any],
    *,
    model_key: str,
    split: str,
    seed: int,
    checkpoint_path: str,
    alpha: Optional[float],
    forced_reflex: bool,
) -> str:
    """Arm B: select -> gate -> (fill | escalate), one turn at a time, batch 1, CPU."""
    from reflex.compile import load_bank, load_turn_labels
    from reflex.data import (
        action_list,
        build_partitions,
        dataset_hash,
        iter_agent_turns,
        load_ontology,
        load_raw_abcd,
        load_utterances,
        subflow_list,
        turn_key,
    )
    from reflex.data import build_context
    from reflex.fill import check_availability, collect_slot_sources, compose_utterance
    from reflex.gate import evaluate_gate
    from reflex.llm_agent import build_llm_agent, llm_decide
    from reflex.select import (
        build_selector,
        delexicalize_candidates,
        score_turn,
        select_from_scores,
    )

    _apply_runtime(cfg)

    ontology = load_ontology(cfg)
    bank = load_bank(cfg)
    partitions = build_partitions(cfg)
    raw = load_raw_abcd(cfg)
    utterances = load_utterances(cfg)
    calibration = _resolve_calibration(cfg, alpha)

    # Arm B answers its own escalations when the kill switch is on: build the
    # agent handle ONCE (it carries the response cache and the running spend
    # counter, spec 6.9 step 5) rather than per turn. Building it never calls
    # the API -- only `llm_decide` does -- so this costs nothing when the run
    # goes on to escalate zero turns.
    llm_enabled = bool(get_dotted(cfg, "llm.enabled"))
    llm_agent_handle = build_llm_agent(cfg, model_key) if llm_enabled else None

    partition = getattr(partitions, split, None)
    if not isinstance(partition, dict):
        raise ValueError(f"unknown split {split!r}; expected one of dev/test_seen/test_novel")

    scenarios = {
        int(convo["convo_id"]): dict(convo.get("scenario") or {})
        for convos in raw.values()
        for convo in convos
    }
    seen_actions = _unseen_action_map(partitions.train)
    class_orders = {
        "nextstep": list(NEXT_STEPS),
        "intent": list(subflow_list(ontology)),
        "action": list(action_list(ontology)),
        "skeleton": [s.skeleton_id for s in bank.skeletons],
    }
    try:
        gold_skeletons = {
            label.turn_id: label.skeleton_id for label in load_turn_labels(split, cfg)
        }
    except FileNotFoundError:
        gold_skeletons = {}

    selector = build_selector(cfg, checkpoint_path, bank, ontology, calibration)

    run_id = new_run_id(cfg, "B", model_key, split, seed)
    max_turns = int(get_dotted(cfg, "run.max_turns"))
    manifest = RunManifest(
        run_id=run_id,
        created_utc=_datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        arm="B",
        model_key=model_key,
        model_id=_resolved_model_id(cfg, "B", model_key),
        split=split,
        seed=int(seed),
        alpha=float(calibration.alpha),
        config={k: v for k, v in cfg.items() if not str(k).startswith("_")},
        overrides=[str(o) for o in (cfg.get("_overrides") or [])],
        prompt_hashes=_prompt_hashes(cfg),
        dataset_hash=dataset_hash(cfg),
        bank_hash=str(bank.bank_hash),
        git_commit=_git_commit(),
        novel_subflows=list(partitions.novel_subflows),
        machine={
            **_machine(),
            "forced_reflex": forced_reflex,
            "max_turns": max_turns,
            "latency_note": (
                "wall clock is CONTENDED: measured on a machine observed swapping "
                "(a fine-tune step went 0.306s -> 241s). Treat every millisecond "
                "figure as an upper bound; ratios are the robust reading."
            ),
        },
        llm_enabled=bool(get_dotted(cfg, "llm.enabled")),
        price_list_date=str(get_dotted(cfg, "llm.price_list_date")),
    )
    write_manifest(manifest, cfg)  # BEFORE the first decision (spec 10)

    flush_every = max(1, int(get_dotted(cfg, "run.flush_every")))
    progress_every = int(get_dotted(cfg, "run.progress_every"))
    write_sidecar = bool(get_dotted(cfg, "run.write_probs_sidecar"))
    sidecar_path = os.path.join(_run_dir(cfg, run_id), str(get_dotted(cfg, "eval.probs_filename")))
    registry = bank.slot_registry

    buffer: list[EvalRecord] = []
    sidecar_buffer: list[dict[str, Any]] = []
    n_written = 0
    n_escalated = 0

    def _flush() -> None:
        nonlocal buffer, sidecar_buffer, n_written
        if buffer:
            n_written += append_decisions(buffer, cfg, run_id)
            buffer = []
        if sidecar_buffer:
            with open(sidecar_path, "a", encoding="utf-8") as handle:
                for row in sidecar_buffer:
                    handle.write(json.dumps(row))
                    handle.write("\n")
            sidecar_buffer = []

    for n_seen, (convo_id, turn_index, turn) in enumerate(iter_agent_turns(partition), start=1):
        if max_turns and n_seen > max_turns:
            break
        turns = partition[convo_id]
        scenario = scenarios.get(int(convo_id), {})

        # ---- the fast path, timed ------------------------------------------ #
        started = time.perf_counter()
        context = build_context(turns, turn_index, scenario, cfg)
        scores = score_turn(selector, context, turn, cfg)
        selection = select_from_scores(selector, scores, turn, cfg)
        sources = collect_slot_sources(turns, turn_index, scenario, registry, cfg)
        required = _required_slots_for(selection, bank)
        missing = check_availability(required, sources, cfg)
        unseen = bool(
            selection.nextstep == "take_action"
            and selection.action
            and selection.action not in seen_actions.get(str(selection.intent), set())
        )
        gate_output = evaluate_gate(scores, selection, missing, unseen, calibration, cfg)

        utterance_text: Optional[str] = None
        if gate_output.route == "reflex" and selection.nextstep == "retrieve_utterance":
            utterance_text, fill_gaps = compose_utterance(
                _templates_for(selection, bank), sources, registry, cfg
            )
            if fill_gaps:
                # check_availability and compose_utterance disagreed about the
                # same slots. That is a bug, not a route: fail loudly rather than
                # emitting text with a {placeholder} in it.
                raise ContractViolation(
                    f"convo {convo_id} turn {turn_index}: the gate passed availability but "
                    f"compose_utterance reports {fill_gaps} missing. fill.check_availability and "
                    "fill.fill_template must agree on the same sources."
                )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        escalated = gate_output.route != "reflex"
        llm_decision: Optional[LLMDecision] = None
        if escalated:
            n_escalated += 1
            if llm_agent_handle is not None:
                candidate_ids = turn.candidates or []
                candidate_texts = (
                    delexicalize_candidates(candidate_ids, utterances, registry, cfg)
                    if candidate_ids
                    else []
                )
                llm_decision = llm_decide(llm_agent_handle, context, turn, candidate_texts, cfg)
            elif not forced_reflex:
                _flush()
                raise LLMDisabledError(
                    f"convo {convo_id} turn {turn_index} escalated with reason "
                    f"{gate_output.reason!r}, but llm.enabled is false so there is nobody to "
                    f"answer it. {n_written} decisions were flushed to run {run_id}. "
                    "Re-run with `--forced-reflex` to record escalations unanswered and get "
                    "the routing metrics for $0, or set llm.enabled (and fill the price table) "
                    "to actually answer them."
                )

        decision = _build_decision(
            convo_id=convo_id,
            turn_index=turn_index,
            selection=selection,
            gate_output=gate_output,
            utterance_text=utterance_text,
            elapsed_ms=elapsed_ms,
            escalated=escalated,
            llm_decision=llm_decision,
        )
        if not escalated and selection.nextstep == "retrieve_utterance" and turn.candidates:
            decision = _with_exact_match(
                decision, selection, turn, utterances, registry, cfg, delexicalize_candidates
            )

        gold = _gold_for(turn)
        buffer.append(
            EvalRecord(
                decision=decision,
                gold=gold,
                correct=_correct_flags(decision, gold),
                seed=int(seed),
                run_id=run_id,
            )
        )
        if write_sidecar:
            sidecar_buffer.append(
                _sidecar_row(
                    scores,
                    gate_output,
                    turn,
                    class_orders,
                    calibration,
                    gold_skeletons.get(turn_key(split, convo_id, turn_index)),
                )
            )
        if len(buffer) >= flush_every:
            _flush()
        if progress_every and n_seen % progress_every == 0:
            print(
                f"[{run_id}] {n_seen} turns, {n_escalated} escalated "
                f"({n_escalated / n_seen:.1%})",
                file=sys.stderr,
                flush=True,
            )

    _flush()
    if n_written == 0:
        raise ContractViolation(
            f"run {run_id} produced no decisions: partition {split!r} yielded no agent turns"
        )
    print(
        f"[{run_id}] done: {n_written} decisions, {n_escalated} escalated"
        + (" (UNANSWERED -- forced-reflex mode, $0)" if forced_reflex else ""),
        file=sys.stderr,
        flush=True,
    )
    return run_id


def _build_decision(
    *,
    convo_id: int,
    turn_index: int,
    selection: Selection,
    gate_output: GateOutput,
    utterance_text: Optional[str],
    elapsed_ms: float,
    escalated: bool,
    llm_decision: Optional[LLMDecision] = None,
) -> Decision:
    """Assemble the spec 5.5 Decision for one Arm B turn.

    Three shapes, not two:

    * ``route == "reflex"``: the fast path answered -- below, unchanged.
    * ``escalated`` and ``llm_decision`` is not ``None``: the LLM actually
      answered it (``llm.enabled``). The Decision's H1/H2/action/values/
      utterance now come from the LLM's parsed response, not the fast path's
      (rejected) selection, and ``llm_tokens_in``/``llm_tokens_out``/
      ``latency_ms_llm`` are no longer zero. ``skeleton_id``/``template_ids``
      stay ``None`` -- the LLM answers freely (spec 6.9), it does not compose
      from the bank. ``candidate_rank`` carries ``LLMDecision.candidate_index``
      so :func:`_correct_flags` can score ``retrieve_utterance`` turns exactly
      as it already does for the fast path.
    * ``escalated`` and ``llm_decision`` is ``None`` (forced-reflex, UNANSWERED):
      everything the fast path would have said is withheld except H1 and H2,
      which :class:`Decision` types as non-optional strings. The withheld
      fields -- action, values, skeleton, templates, utterance, candidate --
      are what make the arm-level quality columns of such a run unusable, and
      that is the honest accounting: nobody answered this turn.
    """
    if escalated and llm_decision is not None:
        return Decision(
            convo_id=int(convo_id),
            turn_index=int(turn_index),
            arm="B",
            route="escalated",
            nextstep=str(llm_decision.nextstep),
            intent=str(llm_decision.intent),
            action=llm_decision.action,
            values=list(llm_decision.values) if llm_decision.values is not None else None,
            skeleton_id=None,
            template_ids=None,
            utterance_text=llm_decision.utterance_text,
            candidate_utt_id=None,
            gate=gate_output,
            llm_tokens_in=int(llm_decision.tokens_in),
            llm_tokens_out=int(llm_decision.tokens_out),
            latency_ms_fastpath=float(elapsed_ms),
            latency_ms_llm=float(llm_decision.latency_ms),
            candidate_rank=int(llm_decision.candidate_index),
            exact_template_match=None,
            cache_hit=bool(llm_decision.cache_hit),
        )
    if escalated:
        return Decision(
            convo_id=int(convo_id),
            turn_index=int(turn_index),
            arm="B",
            route="escalated",
            nextstep=str(selection.nextstep),
            intent=str(selection.intent),
            action=None,
            values=None,
            skeleton_id=None,
            template_ids=None,
            utterance_text=None,
            candidate_utt_id=None,
            gate=gate_output,
            llm_tokens_in=0,
            llm_tokens_out=0,
            latency_ms_fastpath=float(elapsed_ms),
            latency_ms_llm=0.0,
            candidate_rank=-1,
            exact_template_match=None,
            cache_hit=False,
        )
    return Decision(
        convo_id=int(convo_id),
        turn_index=int(turn_index),
        arm="B",
        route="reflex",
        nextstep=str(selection.nextstep),
        intent=str(selection.intent),
        action=selection.action,
        values=list(selection.values) if selection.values is not None else None,
        skeleton_id=selection.skeleton_id,
        template_ids=list(selection.template_ids) if selection.template_ids is not None else None,
        utterance_text=utterance_text,
        candidate_utt_id=selection.candidate_utt_id,
        gate=gate_output,
        llm_tokens_in=0,
        llm_tokens_out=0,
        latency_ms_fastpath=float(elapsed_ms),
        latency_ms_llm=0.0,
        candidate_rank=int(selection.candidate_rank),
        exact_template_match=None,
        cache_hit=False,
    )


def _with_exact_match(
    decision: Decision,
    selection: Selection,
    turn: NormalizedTurn,
    utterances: list[str],
    registry: Any,
    cfg: dict[str, Any],
    delexicalize_candidates: Any,
) -> Decision:
    """Fill :attr:`Decision.exact_template_match` from the GOLD utterance.

    Spec 6.5 step 5 asks whether "the gold utterance's delexicalized form equals
    the composed text". That needs the gold, so :mod:`reflex.select` cannot
    compute it without leaking a label into scoring -- it is computed HERE, where
    gold is legitimately in hand for logging, and after routing has already been
    decided.
    """
    import dataclasses

    if turn.utt_rank is None or turn.utt_rank < 0 or not selection.composed_text_delex:
        return decision
    texts = delexicalize_candidates(turn.candidates or [], utterances, registry, cfg)
    if turn.utt_rank >= len(texts):
        return decision
    gold_delex = str(texts[turn.utt_rank]).strip().lower()
    composed = str(selection.composed_text_delex).strip().lower()
    return dataclasses.replace(decision, exact_template_match=bool(gold_delex == composed))


def _apply_runtime(cfg: dict[str, Any]) -> None:
    """Pin threads and determinism before the first forward pass (spec 10).

    ``runtime.torch_threads`` must equal ``cost.inference_threads`` or the
    latency numbers describe a machine configuration the cost model does not.
    """
    threads = int(get_dotted(cfg, "runtime.torch_threads"))
    inference_threads = int(get_dotted(cfg, "cost.inference_threads"))
    if threads != inference_threads:
        raise ContractViolation(
            f"runtime.torch_threads={threads} != cost.inference_threads={inference_threads}. "
            "Latency is priced at cost.cpu_price_per_hour_usd, so the two must describe the "
            "same machine configuration."
        )
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        if bool(get_dotted(cfg, "runtime.deterministic")):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        pass


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #


def run_sweep(cfg: dict[str, Any], experiment: str) -> list[str]:
    """Run an experiment matrix from spec Section 7.

    See :func:`reflex.contracts.run_sweep` for the frozen contract. Spec 7's
    order is E0 -> E1 -> E2 -> E5 -> E3 -> E4 -> E6; this function runs ONE of
    them and reorders nothing.

    Raises:
        ValueError: on an unknown experiment.
        NotImplementedError: for E3, E4 and E6, which need artifacts this build
            does not have. See the note below -- this is a refusal to fabricate,
            not an oversight.
    """
    if experiment not in ("E3", "E4", "E5", "E6"):
        raise ValueError(f"unknown experiment {experiment!r}; expected E3|E4|E5|E6")

    checkpoint = os.environ.get("REFLEX_CHECKPOINT", "")
    seed = int(get_dotted(cfg, "train.seeds")[0])

    if experiment == "E5":
        # The one sweep that needs nothing but a checkpoint and a calibration:
        # every alpha reuses the SAME calibration's alpha_sweep_quantiles, which
        # is exactly why spec 6.8 step 3 precomputes them.
        if not checkpoint:
            raise ValueError(
                "E5 needs a checkpoint; set REFLEX_CHECKPOINT=<path> or run "
                "`reflex run --arm B --checkpoint ...` per alpha instead."
            )
        run_ids: list[str] = []
        for alpha in get_dotted(cfg, "gate.alpha_sweep"):
            run_ids.append(
                run_arm(
                    cfg,
                    arm="B",
                    model_key="strong",
                    split="test_seen",
                    seed=seed,
                    checkpoint_path=checkpoint,
                    alpha=float(alpha),
                )
            )
        return run_ids

    raise NotImplementedError(
        f"sweep {experiment} is not runnable in this build. E3 needs ablation checkpoints "
        "(whole-utterance vs skeleton-first, encoder size, context K), E4 needs one "
        "checkpoint per data.learning_curve_fractions, and E6 runs only if E2 passes -- "
        "and E2 needs Arm A, which needs a paid LLM. Producing run_ids without those "
        "artifacts would put numbers in the report that no experiment generated."
    )
