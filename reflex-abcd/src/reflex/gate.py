"""Spec 4.6 -- Gate: THE ONLY PLACE that decides fast-path vs escalate. Never modifies scores.

A CASCADE IS A COST DIAL, NOT AN ACCURACY DEVICE
------------------------------------------------
The standing finding of the wider project, and the frame for everything below:
**measured out of fold, no gate ever beat the better base arm.** A learned
router landed within -0.3/+1.4 points of the better arm and never above both. So
this gate is not trying to be right more often than the selector -- it cannot
be, it has strictly less information than the selector had. Its job is
COVERAGE: how much traffic the cheap arm can answer without the answer getting
worse. Every number it produces is a price, not an accuracy claim, and an
ablation that raises the reflex rate is a cost win, never an accuracy win.

The corollary is the failure mode to watch for: a gate tuned until the fast path
looks accurate is a gate that has quietly routed away its hard turns. Read
``reflex_rate`` and fast-path accuracy together, or neither means anything.

THE TWO USES OF CONFIDENCE ARE KEPT SEPARABLE AND MEASURABLE
-------------------------------------------------------------
**One confidence score cannot do two jobs.** Where a score is spent on a
scope/abstain decision it is no longer available for routing -- measured on
HINT3, the shipped pipeline's routing AUROC inverted to 0.240, worse than a
coin, because the signal had already been consumed. This module therefore never
collapses its signals into a single scalar "confidence":

* The **scope/abstain** decision is the conformal prediction-set SIZE per head,
  plus the novelty distance and slot availability. That, and only that, routes.
* Any future **routing** decision (which LLM, which arm, whether to retry) must
  use a score computed and validated separately. Nothing here may be reused
  for it.

SEPARABLE is enforced by the config: each of spec 6.6's three signals can be
switched off on its own (E3b below), so its contribution is attributable rather
than argued. MEASURABLE is enforced by :class:`GateOutput` carrying each signal
in its own field -- per-head ``set_sizes``, the raw ``novelty_distance``, the
``missing_slots`` list -- so :mod:`reflex.evaluate` can attribute every
escalation instead of trusting ``reason``, which by design reports only the
highest-precedence failure. Two gaps in that, recorded rather than glossed:
``unseen_action`` has no field of its own in the frozen schema, so it is
recoverable only when it IS the reason; and a template-consistency escalation
(below) shares the ``unavailable_slot`` reason, so it is separable only by
joining ``Decision.template_ids`` against the sidecar that flagged it.

WHAT THIS MODULE DOES (spec 6.6)
--------------------------------
Three signals, all of which must pass for ``route="reflex"``:

1. **Confidence.** For every head the PREDICTED nextstep requires -- H1 and H2
   always, H3 only on take_action, H5 and H7-per-position only on
   retrieve_utterance -- the conformal prediction set
   ``{c : softmax(c) >= 1 - q_h}`` must have size exactly 1. Size 0 (a diffuse
   head at a small q) and size > 1 both fail as ``low_confidence``. An empty set
   is never "fixed" by inserting the argmax: that silently breaks coverage. A
   head that does not apply is not consulted and reports ``0``, never a missing
   key and never a free pass.
2. **Novelty.** ``novelty_distance > calibration.novelty_threshold`` -> ``novel``.
   Strictly greater, matching spec 6.6's wording. The distance is computed in
   :mod:`reflex.select` and only compared here (boundary ruling 1).
3. **Availability.** ``unseen_action`` -> ``unseen_action``; non-empty
   ``missing_slots`` -> ``unavailable_slot``. Both are computed elsewhere --
   :mod:`reflex.fill` owns slot logic, the caller owns the kb lookup (boundary
   ruling 2) -- and merely consumed here.

Reported precedence when several fail (spec 6.6,
:data:`~reflex.schemas.GATE_REASON_PRECEDENCE`):
``novel > unseen_action > unavailable_slot > low_confidence``.

**The gate never modifies scores; it only routes.** No renormalizing, no argmax
overriding, no threshold nudging. Nothing in this file writes to ``scores`` or
``selection``, and a regression test pins that.

THE ONE THING CONFORMAL CONFIDENCE CANNOT CATCH (DECISIONS D16)
----------------------------------------------------------------
Single-linkage dedup once merged templates that request DIFFERENT named fields:
"what is your order id?" was canonicalised as "can i have your account id and
order id?", 35.1% of field-requesting merged forms diverged, and an expected
~4.9% of all retrieve turns would have asked the customer for the wrong
identifiers. **No prediction-set size can see that** -- the model is perfectly
confident in the template it picked; the template's own cluster is the thing
that is wrong. The fix belongs at BUILD time and is there
(``compile.merge_block_on_field_mismatch``, divergence 35.1% -> 0.0%). What
lives here is a backstop for a regression in that guard:
``gate.template_consistency_path``, off by default because the current bank has
nothing to flag and :class:`~reflex.schemas.Template` (spec 5.2) is frozen with
nowhere to carry a per-template flag. When a path IS configured it must exist --
a guard that cannot fire is worse than no guard.

E3b ABLATIONS (all behind config, all off by default)
------------------------------------------------------
``gate.confidence_mode: softmax_threshold`` replaces the calibrated set with a
raw ``p >= gate.softmax_threshold_ablation`` test, which is what prices the
conformal quantile. ``gate.use_novelty: false`` and
``gate.use_availability: false`` switch signals 2 and 3 off. A disabled signal is
still MEASURED and still reported in the :class:`GateOutput` fields; it simply
stops routing. That is the whole point of keeping the signals separable.

The D16 backstop has its OWN switch (``gate.use_template_consistency``) rather
than riding on ``gate.use_availability``: an arm that removes two mechanisms at
once measures neither, and this project has already been bitten twice today by
controls that could not fail and by defects measured against data another defect
had corrupted. The two share the ``unavailable_slot`` reason because the frozen
:class:`GateReason` vocabulary offers no other availability-class value, and
they are told apart in the output by ``missing_slots``, which is EMPTY on a
template-consistency escalation. **Report that as its own row**; folded into the
slot-availability count it would misstate the escalation-reason distribution.

Read ``cfg``, never a literal: spec 10 forbids any numeric threshold, model id,
path or price in code.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from reflex.config import get_dotted, repo_root
from reflex.contracts import ContractViolation
from reflex.schemas import (
    GATE_REASON_PRECEDENCE,
    NEXT_STEPS,
    Calibration,
    GateOutput,
    Selection,
    SelectorScores,
)

__all__ = [
    "evaluate_gate",
]


# ---------------------------------------------------------------------------
# Module constants: NAMES the frozen schemas already fix, not tunables. Spec 10
# bans thresholds, model ids, paths and prices from code -- not the vocabulary
# GateOutput.set_sizes and Calibration.quantiles define. A config knob for these
# would only let a config disagree with the schema.
# ---------------------------------------------------------------------------

#: :attr:`GateOutput.set_sizes` keys holding an int, in report order.
_SCALAR_HEADS: tuple[str, ...] = ("nextstep", "intent", "action", "skeleton")

#: The :attr:`GateOutput.set_sizes` key holding one int per act position.
_TEMPLATES_KEY: str = "templates"

#: :attr:`Calibration.quantiles` key shared by every H7 act position.
_TEMPLATE_QUANTILE_KEY: str = "template"

#: The :attr:`GateOutput.set_sizes` key holding one int per H4 value slot of
#: the predicted action (empty list when the turn is not ``take_action``).
_VALUES_KEY: str = "values"

#: :attr:`Calibration.quantiles` key shared by every H4 value slot.
_VALUE_QUANTILE_KEY: str = "value"

#: The only supported ``gate.affect_signal`` value (spec 13 puts affect out of
#: scope for v1). Compared case-insensitively against a STRING -- see
#: :func:`_check_affect_signal` for why the type matters.
_AFFECT_OFF: str = "off"

#: ``gate.confidence_mode`` values.
_MODE_CONFORMAL: str = "conformal"
_MODE_SOFTMAX: str = "softmax_threshold"


# ---------------------------------------------------------------------------
# Signal 1: the conformal prediction set
# ---------------------------------------------------------------------------


def _frozen_prediction_set(probs: Sequence[float], q: float) -> list[int]:
    """The frozen rule ``{i : probs[i] >= 1 - q}``, ascending.

    :func:`reflex.select.prediction_set` OWNS this rule (spec 4.5: a set is a
    score, not a decision). This copy exists only so the gate still routes while
    ``select`` is the contract stub; :func:`_confidence_set` prefers the owner's
    implementation the moment it lands, so the two cannot drift in a real run.
    """
    threshold = 1.0 - float(q)
    return [index for index, prob in enumerate(probs) if float(prob) >= threshold]


def _confidence_set(probs: Sequence[float], q: float, cfg: dict[str, Any]) -> list[int]:
    """The prediction set under the configured confidence mode.

    ``conformal`` (default) is spec 6.6 signal 1. ``softmax_threshold`` is the
    E3b ablation: the same "exactly one class survives" test against a RAW
    probability threshold no calibration produced, which is what prices the
    conformal quantile. Both return a SET, so ``set_sizes`` means the same thing
    in either arm and the two are directly comparable.
    """
    mode = str(get_dotted(cfg, "gate.confidence_mode")).strip().lower()
    if mode == _MODE_SOFTMAX:
        threshold = float(get_dotted(cfg, "gate.softmax_threshold_ablation"))
        return [index for index, prob in enumerate(probs) if float(prob) >= threshold]
    if mode != _MODE_CONFORMAL:
        raise ValueError(
            f"gate.confidence_mode must be {_MODE_CONFORMAL!r} or {_MODE_SOFTMAX!r}, "
            f"got {mode!r}"
        )
    # Imported inside the call, not at module scope, so a future select that
    # wants anything from gate cannot form an import cycle. `select` owns the
    # public rule; the frozen copy above is the stub-era fallback.
    from reflex import contracts as _contracts
    from reflex import select as _select

    owner = getattr(_select, "prediction_set", None)
    if owner is not None and owner is not _contracts.prediction_set:
        return list(owner(probs, q))
    return _frozen_prediction_set(probs, q)


# ---------------------------------------------------------------------------
# Input validation -- loud, because each of these is otherwise silent wrongness
# ---------------------------------------------------------------------------


def _check_affect_signal(cfg: dict[str, Any]) -> None:
    """Spec 13 puts ``gate.affect_signal`` out of scope for v1; refuse to ignore it.

    The type check is not pedantry, it is a shipped bug: YAML 1.1 coerces a bare
    ``off`` to the BOOLEAN ``False`` (likewise ``on``/``yes``/``no``/``y``/``n``),
    so a gate comparing ``str(value).lower()`` to ``"off"`` sees ``"false"`` and
    raises on every single turn. ``configs/default.yaml`` now quotes the value;
    this guard names the trap instead of papering over it, so the next bare
    boolean is diagnosed in one read rather than bisected.
    """
    value = get_dotted(cfg, "gate.affect_signal")
    if not isinstance(value, str):
        raise ValueError(
            f"gate.affect_signal must be the STRING {_AFFECT_OFF!r}, got "
            f"{value!r} ({type(value).__name__}). YAML 1.1 coerces a bare off/on/"
            f"yes/no/y/n to a boolean -- quote it in configs/default.yaml."
        )
    if value.strip().lower() != _AFFECT_OFF:
        raise ValueError(
            f"gate.affect_signal is {value!r}, but spec 13 puts the affect signal out of "
            f"scope for v1 and no implementation exists. Set it to {_AFFECT_OFF!r} rather "
            f"than having the gate silently ignore a signal the config says is on."
        )


def _validate_probs(head: str, probs: Sequence[float], cfg: dict[str, Any]) -> None:
    """Refuse anything that is not a probability vector.

    :class:`SelectorScores` says "SOFTMAX PROBABILITIES, not logits", and the
    conformal rule is meaningless on logits: a ``1 - q`` threshold against raw
    scores would pass or escalate nearly everything, and no metric would say
    why. Caught here, once, loudly.
    """
    tolerance = float(get_dotted(cfg, "gate.prob_sum_tolerance"))
    total = 0.0
    for index, raw in enumerate(probs):
        value = float(raw)
        if value < 0.0 or value > 1.0 + tolerance:
            raise ContractViolation(
                f"{head} score at index {index} is {value!r}, which is not a probability. "
                f"reflex.select.score_turn must return softmax probabilities, not logits "
                f"(SelectorScores docstring)."
            )
        total += value
    if abs(total - 1.0) > tolerance:
        raise ContractViolation(
            f"{head} probabilities sum to {total!r}, not 1 (tolerance {tolerance}). The "
            f"conformal set is defined on a distribution; a renormalization bug upstream "
            f"would silently change every route."
        )


def _required_scalar_probs(
    head: str, probs: Sequence[float], selection: Selection, cfg: dict[str, Any]
) -> Sequence[float]:
    """A fixed-width head that applies must actually carry a distribution."""
    if not probs:
        raise ContractViolation(
            f"nextstep {selection.nextstep!r} requires head {head!r}, but "
            f"SelectorScores.{head}_probs is empty. The gate cannot route on a head the "
            f"selector did not score; fix reflex.select rather than escalating silently."
        )
    _validate_probs(head, probs, cfg)
    return probs


def _quantile(calibration: Calibration, quantiles: dict[str, float], head: str) -> float:
    """Return ``q_h``, refusing to invent one.

    A missing quantile is NOT a zero. ``q_h = 0`` makes the prediction set
    ``{i : p_i >= 1}`` -- almost always empty, so everything escalates, and the
    run reads as a calibration finding rather than as a missing file.
    """
    if head not in quantiles:
        raise ContractViolation(
            f"calibration has no quantile for head {head!r} (has {sorted(quantiles)}). "
            f"Run `reflex calibrate` against "
            f"{calibration.checkpoint_path or 'this checkpoint'}; quantiles are not "
            f"transferable across checkpoints or seeds."
        )
    return float(quantiles[head])


def _quantiles_for_alpha(calibration: Calibration, cfg: dict[str, Any]) -> dict[str, float]:
    """The per-head quantiles for the alpha THIS RUN is configured at.

    ``Calibration.quantiles`` belongs to ``calibration.alpha``. Spec 7's E5
    sweeps ``gate.alpha`` without retraining, and :mod:`reflex.calibrate`
    precomputes ``alpha_sweep_quantiles`` precisely so the sweep needs no
    recalibration -- so when the run's alpha differs from the calibrated one,
    the sweep table is the right source, and silently reusing the calibrated
    quantiles would report a coverage curve the gate never applied. Keys are
    stringified floats, matched numerically rather than by spelling.
    """
    alpha = float(get_dotted(cfg, "gate.alpha"))
    if alpha == float(calibration.alpha):
        return {head: float(q) for head, q in (calibration.quantiles or {}).items()}
    for key, head_quantiles in (calibration.alpha_sweep_quantiles or {}).items():
        try:
            matches = float(key) == alpha
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if matches:
            return {head: float(q) for head, q in (head_quantiles or {}).items()}
    raise ContractViolation(
        f"gate.alpha is {alpha} but the calibration was computed at {calibration.alpha} "
        f"and its alpha sweep covers {sorted(calibration.alpha_sweep_quantiles or {})}. "
        f"Recalibrate, or add {alpha} to gate.alpha_sweep before the run -- do not gate "
        f"at an alpha whose quantiles nobody computed."
    )


def _template_positions(scores: SelectorScores, selection: Selection) -> list[Sequence[float]]:
    """The H7 distributions for a retrieve turn, one per act position.

    A retrieve turn whose skeleton has P act positions must present P template
    distributions AND P chosen template ids. A mismatch means the selector and
    the gate disagree about what is being answered, which is a cross-module
    contract break rather than a property of the turn: it is raised, not
    escalated, so it cannot hide inside the reflex rate. An EMPTY vector at a
    position is different and is allowed -- that is the bank having no template
    for that act, a coverage fact, and it escalates.
    """
    positions = list(scores.template_probs or [])
    chosen = list(selection.template_ids or [])
    if not positions:
        raise ContractViolation(
            "nextstep 'retrieve_utterance' requires H7, but "
            "SelectorScores.template_probs is empty. Every act position of the chosen "
            "skeleton must carry a distribution; a position with no CANDIDATE is an "
            "empty vector, not an absent one."
        )
    if len(chosen) != len(positions):
        raise ContractViolation(
            f"Selection.template_ids has {len(chosen)} entries but "
            f"SelectorScores.template_probs has {len(positions)} act positions. The "
            f"selector and the gate disagree about what is being answered."
        )
    return positions


# ---------------------------------------------------------------------------
# DECISIONS D16 backstop: templates whose merge cluster is internally inconsistent
# ---------------------------------------------------------------------------

#: ``resolved path -> frozenset(flagged template ids)``. A memo: the sidecar is
#: small and immutable during a run, and re-reading it per turn would put a file
#: read on the fast path.
_FLAGGED_CACHE: dict[str, frozenset[str]] = {}
_FLAGGED_CACHE_SIZE: int = 4


def _flagged_templates(cfg: dict[str, Any]) -> frozenset[str]:
    """Template ids the bank flags as internally inconsistent (D16), or empty.

    ``gate.template_consistency_path: null`` (the default) means no sidecar and
    no check -- today's honest state, because the build-time field-set merge
    guard took divergence to 0.0% and ``Template`` is frozen with nowhere to
    carry a flag. A CONFIGURED path that does not exist raises: a guard that
    cannot fire is worse than no guard, and silently passing is exactly how the
    original defect survived 35.1% divergence.
    """
    configured = get_dotted(cfg, "gate.template_consistency_path")
    if configured is None or not str(configured).strip():
        return frozenset()
    path = str(configured)
    if not os.path.isabs(path):
        path = os.path.join(repo_root(), path)
    cached = _FLAGGED_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        raise ContractViolation(
            f"gate.template_consistency_path points at {path}, which does not exist. "
            f"Set it to null to run without the D16 backstop, or write the sidecar -- do "
            f"not leave a guard configured that can never fire."
        )
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        flagged = frozenset(str(key) for key, value in raw.items() if value)
    elif isinstance(raw, list):
        flagged = frozenset(str(item) for item in raw)
    else:
        raise ContractViolation(
            f"{path} must hold a list of template ids or a {{template_id: flag}} object, "
            f"got {type(raw).__name__}"
        )
    if len(_FLAGGED_CACHE) >= _FLAGGED_CACHE_SIZE:
        _FLAGGED_CACHE.clear()
    _FLAGGED_CACHE[path] = flagged
    return flagged


# ---------------------------------------------------------------------------
# 4.6 evaluate_gate
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

    See :func:`reflex.contracts.evaluate_gate` for the frozen contract.

    Applicability follows the PREDICTED nextstep, never the gold one:

    ====================== ===================================================
    predicted nextstep     heads consulted
    ====================== ===================================================
    ``take_action``        H1, H2, H3
    ``retrieve_utterance`` H1, H2, H5, and H7 at every act position
    ``end_conversation``   H1, H2
    ====================== ===================================================

    Determinism (spec 10): the same ``scores`` / ``selection`` / ``calibration``
    / ``cfg`` always produce the same :class:`GateOutput`. Nothing here reads a
    clock, an RNG, a gold field or a model, and every comparison is a float
    comparison in a fixed order.
    """
    _check_affect_signal(cfg)

    nextstep = str(selection.nextstep)
    if nextstep not in NEXT_STEPS:
        raise ContractViolation(
            f"Selection.nextstep is {nextstep!r}, which is not one of {list(NEXT_STEPS)}. "
            f"The gate reads it to decide which heads apply, so an unknown value cannot "
            f"be routed."
        )
    quantiles = _quantiles_for_alpha(calibration, cfg)

    # --- signal 1: confidence, per APPLICABLE head -------------------------- #
    applicable: dict[str, bool] = {
        "nextstep": True,
        "intent": True,
        "action": nextstep == NEXT_STEPS[1],   # take_action
        "skeleton": nextstep == NEXT_STEPS[0],  # retrieve_utterance
    }
    head_probs: dict[str, Sequence[float]] = {
        "nextstep": scores.nextstep_probs,
        "intent": scores.intent_probs,
        "action": scores.action_probs,
        "skeleton": scores.skeleton_probs,
    }

    set_sizes: dict[str, Any] = {}
    low_confidence = False
    for head in _SCALAR_HEADS:
        if not applicable[head]:
            set_sizes[head] = 0
            continue
        probs = _required_scalar_probs(head, head_probs[head], selection, cfg)
        size = len(_confidence_set(probs, _quantile(calibration, quantiles, head), cfg))
        set_sizes[head] = size
        if size != 1:
            low_confidence = True

    template_sizes: list[int] = []
    if applicable["skeleton"]:
        q_template = _quantile(calibration, quantiles, _TEMPLATE_QUANTILE_KEY)
        for position, probs in enumerate(_template_positions(scores, selection)):
            if not probs:
                # The bank has no template for this act position: a coverage
                # fact (template coverage is 61.01% of agent utterances), not a
                # structural bug. Escalate; never manufacture a candidate.
                template_sizes.append(0)
                low_confidence = True
                continue
            _validate_probs(f"template[{position}]", probs, cfg)
            size = len(_confidence_set(probs, q_template, cfg))
            template_sizes.append(size)
            if size != 1:
                low_confidence = True
    set_sizes[_TEMPLATES_KEY] = template_sizes

    # --- signal 1b: H4 value confidence, per slot of a predicted action ----- #
    # Reviewer-confirmed gap: this block did not exist. A take_action turn's
    # argument values (SelectorScores.value_probs, one distribution per slot --
    # already produced by select._score_values, just never consulted here)
    # passed the gate regardless of confidence, including a 50/50-probability
    # value or an EMPTY argument list. Mirrors the template loop exactly: a
    # slot with no candidate distribution at all (an empty value_probs[i], the
    # "empty argument list" failure mode) is size 0, never a free pass; a
    # slot whose prediction set is not a singleton is low_confidence, exactly
    # like every other head. Requires calibration.quantiles["value"] -- run
    # `reflex calibrate` against a checkpoint calibrated under this change
    # before evaluate_gate is called on live take_action turns; there is no
    # silent fallback, per this module's existing "a missing quantile is not
    # a zero" rule for every other head.
    #
    # WHAT `and scores.value_probs` MEANS, and why the size-0 branch below is
    # reachable at all. Until select._score_values was fixed it DROPPED a slot
    # with no candidate column instead of emitting an empty distribution, so an
    # empty value_probs[i] was a state select could never produce and this
    # block's own promise was dead code: validate-purchase's three arguments
    # arrived as two confident slots and passed. select now emits one (possibly empty)
    # distribution PER SLOT of the predicted action, which makes the branch
    # live, and narrows an entirely empty value_probs to the one case it should
    # mean -- a button that takes no arguments at all. That is not a failure and
    # must not demand a "value" quantile, hence the guard stays. The two halves
    # only work together: if select ever goes back to dropping slots, this guard
    # becomes a free pass again.
    value_sizes: list[int] = []
    if applicable["action"] and scores.value_probs:
        q_value = _quantile(calibration, quantiles, _VALUE_QUANTILE_KEY)
        for slot_index, probs in enumerate(scores.value_probs):
            if not probs:
                value_sizes.append(0)
                low_confidence = True
                continue
            _validate_probs(f"value[{slot_index}]", probs, cfg)
            size = len(_confidence_set(probs, q_value, cfg))
            value_sizes.append(size)
            if size != 1:
                low_confidence = True
    set_sizes[_VALUES_KEY] = value_sizes

    # --- signal 2: novelty --------------------------------------------------- #
    novelty_distance = float(scores.novelty_distance)
    novel = bool(get_dotted(cfg, "gate.use_novelty")) and (
        novelty_distance > float(calibration.novelty_threshold)
    )

    # --- signal 3: availability ---------------------------------------------- #
    # Both inputs are computed elsewhere (boundary ruling 2) and only routed on
    # here. The D16 backstop reports the SAME reason because the frozen
    # GateReason vocabulary has no other availability-class value -- but it has
    # its OWN switch. Folding it under gate.use_availability would make the E3b
    # no-availability arm remove two distinct mechanisms at once, and its delta
    # would then measure neither: the arm has to isolate one thing or it is not
    # an ablation. The two are told apart in the output by `missing_slots`,
    # which is EMPTY on a template-consistency escalation.
    missing = list(dict.fromkeys(str(slot) for slot in (missing_slots or [])))
    flagged = _flagged_templates(cfg)
    inconsistent = bool(get_dotted(cfg, "gate.use_template_consistency")) and any(
        str(template_id) in flagged for template_id in (selection.template_ids or [])
    )
    use_availability = bool(get_dotted(cfg, "gate.use_availability"))
    unavailable = (use_availability and bool(missing)) or inconsistent
    unseen = use_availability and bool(unseen_action)

    # --- precedence: novel > unseen_action > unavailable_slot > low_confidence #
    failed: dict[str, bool] = {
        "novel": novel,
        "unseen_action": unseen,
        "unavailable_slot": unavailable,
        "low_confidence": low_confidence,
    }
    reason = next((name for name in GATE_REASON_PRECEDENCE if failed.get(name)), "ok")

    return GateOutput(
        route="reflex" if reason == "ok" else "escalated",
        reason=reason,
        set_sizes=set_sizes,
        novelty_distance=novelty_distance,
        # Reported even when gate.use_availability is false: a disabled signal is
        # still measured, it just stops routing (E3b).
        missing_slots=missing,
    )
