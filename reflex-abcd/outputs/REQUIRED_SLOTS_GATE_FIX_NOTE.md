# Fix note: `required_slots` misuse + missing H4 value-confidence gate

Two reviewer-confirmed bugs, both fixed. Scope: `src/reflex/select.py`,
`src/reflex/gate.py`, `src/reflex/schemas.py` (docstring/field additions
only), `tests/test_select.py`, `tests/test_gate.py`. No other files touched.

## Sub-claim A: `required_slots` treated as the complete action schema

**Root cause.** `compile.py`'s `ActionPattern.required_slots` docstring
(now `compile.py:1326`; it was line 1311 when this note was written) is
explicit: it holds only the slots the FILLER could source from disclosed state —
a subset, not the schema. `select.py:_value_slots`'s `bank_then_union` mode
ignored that and returned `bank_slots` alone whenever non-empty:

```python
# before
if bundle.value_slots_source == "bank_then_union" and bank_slots:
    return bank_slots
```

Ground truth, recomputed 2026-09-20 from
`/Users/vasyl/zadumai/data/abcd/data/ontology.json` and
`outputs/compile/bank/actions.jsonl` (the earlier "3 arguments" in this note was
right for one action and wrong for the other):

| action | ontology arguments | bank `required_slots` | dropped |
|---|---|---|---|
| `validate-purchase` | `username, email, order_id` (3) | `email, order_id` (2) | `username` |
| `verify-identity` | `customer_name, account_id, order_id, zip_code` (4) | `customer_name, account_id, order_id` (3) | `zip_code` |

The compiled bank drops the arguments the filler never had a chance to source,
even though train labels carry real values for them. H4 was never even ASKED to
decode the dropped slot.

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

**Superseded 2026-09-20 — re-counted and re-run, nothing excluded.** This note
originally claimed 36 tests in `tests/test_select.py` and a partial suite run
that skipped `test_run_report.py`. Current, verified by
`python -m pytest -p no:warnings --collect-only -q`:

| module | tests | note |
|---|---:|---|
| `tests/test_select.py` | **33** (not 36) | includes the three `_value_slots` / `_score_values` tests named above |
| `tests/test_gate.py` | **47** | includes all six value-confidence tests named above; 3 pre-existing `set_sizes` fixtures carry the new `"values": []` key |
| **whole suite** | **561** | **561 passed, 0 failed, 0 errors** — `test_run_report.py` (24 tests) included; no module is excluded any more |

All nine tests this note names by ID were confirmed present on 2026-09-20.

**The "Deliberate non-choice" prerequisite above has since been CLOSED, and the
paragraph that states it is out of date.** It said extending `calibrate.py` to
compute `quantiles["value"]` was out of scope and "the natural next step for
whoever owns that module". Somebody took it: `src/reflex/calibrate.py:135` now
reads `_HEADS = ("nextstep", "intent", "action", "skeleton", "template",
"value")`, `_collect_dev_scores` appends `probs["value"]` / `golds["value"]`
(lines 903-904), and the step-3 loop computes a conformal quantile for every
head in `_HEADS`, so a `reflex calibrate` run now writes `quantiles["value"]`
and `evaluate_gate` will not raise. `calibrate.py:131-134` records exactly why:
"every Arm B run died on its first slot-bearing take_action turn".

What is still true: **no calibration has ever been produced.**
`outputs/calibration/` is empty, so `evaluate_gate` has never seen a real
`take_action` turn and the `ContractViolation` path has never fired outside
`test_value_confidence_without_a_calibrated_value_quantile_raises`.
