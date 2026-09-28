# Tests

**Status 2026-09-20: 16 test modules, 561 tests, 561 passed / 0 failed**
(`cd reflex-abcd && python -m pytest -p no:warnings`). `pytest.ini` sets
`testpaths = tests`, `pythonpath = src`, `addopts = -q`, so a bare `pytest` from
the repo root is the whole suite.

`test_contracts.py` (228 of the 561) is the coherence mechanism for the
multi-agent build: it asserts each module defines exactly the public names it
owns and that every implemented signature still matches the frozen one in
`src/reflex/contracts.py`. It was written to pass while the modules were stubs;
all fourteen are now implemented and it still passes, which is the point.

## What exists today

| module | tests | covers |
|---|---:|---|
| `test_contracts.py` | 228 | the frozen module-boundary contract |
| `test_gate.py` | 47 | gate routing; all four `GATE_REASON_PRECEDENCE` reasons (`novel`, `unseen_action`, `unavailable_slot`, `low_confidence`) plus `ok`, and the new H4 value-confidence check |
| `test_featurize.py` | 38 | the shared feature builder |
| `test_models.py` | 34 | heads H1–H7, checkpoint save/load, eval-mode determinism |
| `test_select.py` | 33 | candidate mapping, `_value_slots`, per-turn scoring determinism |
| `test_fill.py` | 30 | slot sourcing and `check_availability` |
| `test_arm_b0.py` | 24 | the bank-only Arm B0 variant |
| `test_run_report.py` | 24 | run orchestration and report rendering inputs |
| `test_data.py` | 23 | loader keys, context construction, determinism |
| `test_probe_selection.py` | 20 | the probe track's conversation-grouped split |
| `test_no_paid_calls.py` | 19 | the zero-paid-API-calls rail |
| `test_compile.py` | 12 | value typing and required-slot derivation |
| `test_report_inputs.py` | 12 | report input contracts |
| `test_config_integrity.py` | 11 | config keys the code reads actually exist |
| `test_arm_b_escalation.py` | 4 | `_build_decision`'s answered-escalation shape |
| `test_calibrate.py` | 2 | the gate-collapse fix (absent rows excluded from the quantile) |
| **total** | **561** | |

## Still owed to spec 10

Spec 10 lists five tests by name. Three are satisfied by other modules than the
ones originally planned; two requirements are **not met**:

| spec 10 requirement | status |
|---|---|
| loader keys present (`data`) | **met** — `test_data.py` |
| delex round-trip on 20 fixtures (`compile`) | **NOT MET** — `test_compile.py` covers value typing and required slots only; there is no round-trip fixture test |
| skeleton extraction on 10 fixtures (`compile`) | **NOT MET** — no such test exists |
| gate routing on synthetic score fixtures, one per reason (`gate`) | **met** — `test_gate.py` |
| candidate mapping (`select`) | **met** — `test_select.py::test_map_to_candidate_is_deterministic` and neighbours |

`test_determinism.py` is called out separately by spec 10 ("Add a test that
asserts it") and by acceptance item 12. **No file by that name exists.** The
property is asserted piecemeal instead — `test_select.py::test_score_turn_is_deterministic_for_the_same_context`,
`test_models.py::test_forward_is_deterministic_in_eval_mode`,
`test_data.py::test_context_is_deterministic`,
`test_fill.py::test_collect_slot_sources_is_deterministic`,
`test_featurize.py::test_documents_are_deterministic` — but nothing asserts the
end-to-end "same context → same `Decision`" the spec asks for, because that
needs a checkpoint and a calibration, and neither exists in this checkout
(`outputs/checkpoints/` is absent, `outputs/calibration/` is empty).
