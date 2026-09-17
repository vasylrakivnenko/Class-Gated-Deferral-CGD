# Fix note: `required_slots` misuse + missing H4 value-confidence gate

Two reviewer-confirmed bugs, both fixed. Scope: `src/reflex/select.py`,
`src/reflex/gate.py`, `src/reflex/schemas.py` (docstring/field additions
only), `tests/test_select.py`, `tests/test_gate.py`. No other files touched.

## Sub-claim A: `required_slots` treated as the complete action schema

**Root cause.** `compile.py`'s `ActionPattern.required_slots` docstring
(line 1311) is explicit: it holds only the slots the FILLER could source from
disclosed state — a subset, not the schema. `select.py:_value_slots`'s
`bank_then_union` mode ignored that and returned `bank_slots` alone whenever
non-empty:

```python
# before
if bundle.value_slots_source == "bank_then_union" and bank_slots:
    return bank_slots
```

Ground truth: ontology `validate-purchase`/`verify-identity`-shaped actions
list 3 arguments; the compiled bank drops the ones the filler never had a
chance to source (e.g. `username`), even though train labels carry real
values for them. H4 was never even ASKED to decode the dropped slot.

**Fix.** `bank_then_union` now unions bank slots (kept first — the
filler-availability signal was tuned against that order) with the full
ontology slot list for the action, via a new `_ontology_action_slots`
helper (mirrors `reflex.llm_agent._flatten_actions` without importing that
module, to keep the hot inference path light):

```python
if bundle.value_slots_source == "bank_then_union":
    union = list(bank_slots)
    for slot in _ontology_action_slots(bundle.ontology, action):
        if slot not in union:
            union.append(slot)
    return union
```

`bundle.required_slots` itself is untouched — it still drives the
availability/`unavailable_slot` gate signal (D-4) and must stay narrow for
that purpose; only `_value_slots`'s USE of it changed.

**Tests** (`tests/test_select.py`):
- `test_value_slots_bank_then_union_no_longer_drops_ontology_only_slots` —
  bank `[customer_name, account_id]` + ontology `[..., order_id]` →
  `[customer_name, account_id, order_id]`; `bundle.required_slots` itself
  confirmed unchanged.
- `test_value_slots_bank_then_union_falls_back_to_bank_when_ontology_is_empty`
  — `ontology={}` (as every other existing fixture in this file uses)
  behaves exactly as before: no regression.
- `test_score_values_now_attempts_a_slot_bank_alone_used_to_drop` — end to
  end through `_score_values`: with the ontology wired and an `<order_id>`
  copy marker present in context, H4 now attempts 3 slots where bank-alone
  attempted 2.

## Sub-claim B: gate had no confidence check for H4 values

**Root cause.** `gate.py`'s `_SCALAR_HEADS` covers nextstep/intent/action/
skeleton, and there's a separate per-position loop for H7 templates — no
equivalent existed for H4 (`SelectorScores.value_probs` was computed by
`select._score_values` but never read in `evaluate_gate`). An action call
with 50/50 argument probabilities, or an empty value prediction, passed
unconditionally.

**Fix.** New block in `evaluate_gate`, structurally identical to the
template-position loop: for a `take_action` turn with populated
`value_probs`, each slot's prediction-set size is checked
(`_confidence_set` against a new `Calibration.quantiles["value"]` key,
reusing the exact same conformal/softmax-threshold mechanism every other
head already uses); size 0 (empty prediction, mirroring "no template
candidate") or size != 1 sets `low_confidence`, same reason precedence as
every other head. `GateOutput.set_sizes["values"]` (list of ints, one per
slot) is new; reported as `[]`, never a missing key, when the turn has no
value slots to check — same convention `"templates": []` already uses.

**Deliberate non-choice:** a missing `"value"` calibration quantile RAISES
(`ContractViolation`), exactly like a missing quantile for any other head
already does (`_quantile`'s documented "a missing quantile is not a zero"
rule) — no soft fallback was added. This means **`reflex calibrate` must be
rerun to populate `quantiles["value"]` before `evaluate_gate` can process a
real take_action turn** — a real, honest new prerequisite, not a hidden gap.
Extending `calibrate.py`'s dev-score collection to compute that quantile is
explicitly OUT OF SCOPE here (a parallel fix was touching `calibrate.py` for
the gate-collapse bug at the same time; avoiding a file-overlap race) and is
the natural next step for whoever owns that module.

**Tests** (`tests/test_gate.py`):
- `test_a_50_50_value_prediction_now_escalates` — the review's own
  reproduction (50/50 probs), now escalates.
- `test_an_empty_value_prediction_now_escalates` — the review's other
  reproduction (empty value list), now escalates, size 0.
- `test_a_confident_correct_value_prediction_still_routes_to_reflex` — proves
  the check discriminates, not just rejects everything.
- `test_one_bad_value_slot_escalates_the_whole_turn` — multi-slot action,
  one bad slot escalates the turn (mirrors the multi-position template test).
- `test_a_turn_with_no_value_slots_reports_empty_and_is_not_penalized` — no
  regression for turns with nothing to check.
- `test_value_confidence_without_a_calibrated_value_quantile_raises` — proves
  the "no silent fallback" choice above is real, not aspirational.

## Test-suite status

`tests/test_select.py` (36 tests), `tests/test_gate.py` (47 tests, 3
pre-existing `set_sizes` fixtures updated to include the new `"values": []`
key): **all pass.** Full suite (`pytest tests/ -q`, excluding
`test_run_report.py` — a different fork was actively editing `run.py` in
parallel, avoided to prevent reading it mid-edit): **all pass, no
regressions.**
