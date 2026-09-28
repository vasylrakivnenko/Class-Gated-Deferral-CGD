# Arm B escalation wiring fix

## The gap (confirmed by external review, verified independently)

`src/reflex/run.py::_run_arm_b`, on escalation, previously did one of two
things: raised `LLMDisabledError` unconditionally unless `run.forced_reflex`
was set — **with no check on `llm.enabled` at all** — or (under
`forced_reflex`) logged the escalation unanswered. `reflex.llm_agent.llm_decide`
existed and was already unit-tested in isolation
(`tests/test_no_paid_calls.py:121`), but nothing in the orchestrator ever
called it. Setting `llm.enabled=True` and filling the price table did not
complete the cascade — an enabled LLM was a dead end.

Separately confirmed and explicitly **not** part of this fix: Arm A's raise is
intentional, tested, correct behavior (the zero-paid-API-calls safety rail;
asserted by `test_run_report.py::test_arm_a_refuses_before_loading_any_data`).
Not touched. **Line reference refreshed 2026-09-20: it is now `run.py:514-522`**
(`build_llm_agent(cfg, model_key)` at line 519, followed by a
`ContractViolation`); this note originally said ~463-471.

## The fix

`src/reflex/run.py`:

1. `_run_arm_b` now builds the LLM agent handle **once**, up front, only if
   `llm.enabled` is true:
   ```python
   llm_enabled = bool(get_dotted(cfg, "llm.enabled"))
   llm_agent_handle = build_llm_agent(cfg, model_key) if llm_enabled else None
   ```
   (Building never calls the API — only `llm_decide` does — so this costs
   nothing on a run that escalates zero turns.)

2. On escalation, if the handle exists, it now actually calls `llm_decide`
   with the turn's context and delexicalized candidate texts, instead of
   immediately raising:
   ```python
   if llm_agent_handle is not None:
       candidate_ids = turn.candidates or []
       candidate_texts = (
           delexicalize_candidates(candidate_ids, utterances, registry, cfg)
           if candidate_ids else []
       )
       llm_decision = llm_decide(llm_agent_handle, context, turn, candidate_texts, cfg)
   elif not forced_reflex:
       raise LLMDisabledError(...)   # unchanged
   ```

3. `_build_decision` gained a third shape (previously only "fast path
   answered" / "unanswered, everything withheld"): an **answered escalation**,
   populated from the `LLMDecision` — nextstep, intent, action, values,
   utterance_text, candidate_rank (from `candidate_index`), llm_tokens_in/out,
   latency_ms_llm, cache_hit — never from the fast path's rejected
   `selection`. `skeleton_id`/`template_ids` stay `None`: the LLM answers
   freely (spec 6.9), it does not compose from the bank.

## Tests (`tests/test_arm_b_escalation.py`, new)

`_build_decision` is tested directly rather than `_run_arm_b` end-to-end,
because there is no trained checkpoint on this machine (same limitation
`test_run_report.py` already documents at its own top) — `build_selector`
requires one and none exists in this checkout, so a live `_run_arm_b` run is
not exercisable here regardless of this fix. That gap (the certified
TF-IDF+logreg selector was never wired to this runtime path at all) is
tracked separately. `_build_decision` is where the actual defect lived (the
decision-assembly logic), and it is a pure function, so it is fully testable
without the corpus:

1. `test_answered_escalation_uses_the_llm_decision_not_the_rejected_selection`
   — the core proof: constructs a `Selection` saying one thing
   (take_action/manage_dispute/offer-refund) and an `LLMDecision` saying
   another (retrieve_utterance/status_delivery_time), and asserts every field
   on the resulting `Decision` traces to the `LLMDecision`, never the
   `Selection`.
2. `test_answered_escalation_take_action_carries_action_and_values` —
   take_action shape specifically.
3. `test_unanswered_escalation_still_withholds_everything_but_h1_h2` —
   regression guard: `llm_decision=None` (forced-reflex / kill-switch-on)
   produces byte-identical output to the pre-fix behavior.
4. `test_fast_path_decision_is_unaffected_by_the_llm_decision_parameter` —
   regression guard: the reflex-route branch ignores the new parameter
   entirely.

All four fail with `TypeError: unexpected keyword argument 'llm_decision'`
against the pre-fix `_build_decision` signature — the tests do catch the
defect, not just describe the fix.

## Test-suite status

**Superseded 2026-09-20.** This note recorded "49 passed" for
`pytest tests/test_arm_b_escalation.py tests/test_run_report.py
tests/test_no_paid_calls.py` and deferred the full-suite run to a coordinator.
Both are now settled:

- those three modules collect **47** tests, not 49 — `test_arm_b_escalation.py`
  4 + `test_run_report.py` 24 + `test_no_paid_calls.py` 19 (counted with
  `pytest --collect-only -q` on 2026-09-20). All 47 pass.
- the deferred full-suite run **has been done**, after all four parallel fixes
  landed: `python -m pytest -p no:warnings` → **561 passed, 0 failed, 0 errors**.

The four `_build_decision` tests and the `llm_decision` keyword argument they
exercise are all present: `run.py:905-915` shows the current signature carrying
`llm_decision: Optional[LLMDecision] = None`, and the wiring described above is
at `run.py:681-682` (handle built once) and `run.py:824-831` (`llm_decide`
called on escalation).

## Left for later, deliberately

- Full `_run_arm_b` end-to-end exercise. **Still open, and only partly for the
  reason given.** Re-checked 2026-09-20: the compiled bank DOES exist
  (`outputs/compile/bank/`, bank hash `c270df0e249444bb`, 4,489 templates); the
  checkpoint and the calibration do not (`outputs/checkpoints/` is absent,
  `outputs/calibration/` is empty), and `faiss-cpu` is not installed, which the
  spec 6.6 novelty index needs. So the blocker is train + calibrate + faiss, not
  the bank.
- Whether `llm.enabled=True` together with `run.forced_reflex=True` should
  warn (forced_reflex's whole point is a $0 run; this fix makes `llm.enabled`
  take priority when both are set, silently ignoring `forced_reflex` in that
  combination). Not addressed here — flagging for a decision, not a defect.
