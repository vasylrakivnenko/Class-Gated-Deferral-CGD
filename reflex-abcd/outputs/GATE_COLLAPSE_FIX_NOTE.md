# Gate collapse fix (Priority 2)

## Root cause

`src/reflex/calibrate.py::_collect_dev_scores`, template-head bucketing
(`h7_absent_gold: max_nonconformity`, the shipped default): whenever a dev
template position's gold template was absent from the predicted skeleton's
candidate pool, the code appended an explicit `s = 1.0` nonconformity row
into the **same distribution** `conformal_quantile` reads to set the
accept/reject threshold for confident, covered turns. Coverage-absence (a
structural fact: is the true answer even a candidate here) and confidence
(given real candidates, how sure is the model) were being calibrated as one
signal. At the real absent rate (~50% of dev template positions — matches
D23's `gold_present_rate`), this forces `q → 1.0`, the prediction-set
threshold to `0.0` (every class admitted), and the singleton gate then
rejects every turn regardless of confidence. Reproduced synthetically at 3%
absent, α=0.02: a 99.9%-confident response was rejected.

## Fix

`_collect_dev_scores` now **excludes** absent rows from the score/gold pair
that feeds `conformal_quantile`, in both `h7_absent_gold` modes — `probs` and
`golds` stay index-aligned (every position still gets a `probs` row, for
array-length parity), but absent rows get gold index `-1`, which
`nonconformity_scores` already skips by contract ("Rows with a negative index
are SKIPPED, not scored"). The coverage gap is **not deleted** — it is
tracked in a new `template_coverage = {"n_positions", "n_gold_present"}`
return value, threaded through `calibrate()` into the diagnostics sidecar
(`_write_diagnostics`, `calibration_diagnostics_path`) as
`template_coverage.gold_present_rate`, with a note explaining why it is kept
separate from the routing threshold. `gate.py` was **not modified** — the fix
is entirely upstream, in what `quantiles["template"]` gets set to.

Before/after (calibrate.py):
```python
# BEFORE (collapses at ~50% absent rate)
elif absent_mode == "max_nonconformity":
    probs["template"].append(list(position_probs) + [0.0])
    golds["template"].append(len(position_probs))   # points at the 0.0 -> s=1.0

# AFTER
elif absent_mode == "max_nonconformity":
    probs["template"].append(list(position_probs))   # real probs, no extra column
    golds["template"].append(-1)                      # skipped by nonconformity_scores
    # (n_positions/n_gold_present still counted for the coverage diagnostic)
```

## Tests (`tests/test_calibrate.py`, new file — none existed for this module before)

1. `test_old_bucketing_collapses_the_gate_new_bucketing_does_not` — pure-math
   reproduction at the review's exact numbers (3% absent, α=0.02): OLD
   bucketing gives `q=1.0`, threshold `0.0`, admits every class. NEW bucketing
   gives `q<0.5` and a 99.9%-confident row forms a **singleton** prediction
   set (i.e., would be accepted, not rejected).
2. `test_collect_dev_scores_excludes_absent_rows_but_still_reports_coverage`
   — end-to-end through the real `_collect_dev_scores` (stubbed `score_turn`,
   10 synthetic turns, 6 present / 4 absent): confirms exactly the present
   rows contribute a score to the quantile (`len(scores) == 6`), the
   quantile is not collapsed, and `template_coverage` correctly reports
   `4/10` absent — proving the gap is visible, not silently dropped.

**Caught a real bug in my own first-draft fix**: my initial edit appended to
`golds["template"]` unconditionally but only appended to `probs["template"]`
for present rows, breaking the arrays' index alignment (would have raised
`ValueError` in `nonconformity_scores`'s length check). Test 2 failed
immediately and caught it before it shipped; fixed by always appending a
`probs` row too. Also confirmed both tests fail against the pre-fix code
(`git stash` the calibrate.py change, rerun — test 2 fails with
`ValueError: not enough values to unpack (expected 4, got 3)`, since the
function's return arity itself changed; test 1 doesn't depend on the source
change since it exercises the two pure functions directly with hand-built
synthetic arrays, so it was written to fail only if the OLD bucketing
behavior were reintroduced — it currently demonstrates the collapse exists in
principle and that the new bucketing avoids it, which is the intended
contract).

**Full test suite**: `pytest tests/` — all calibrate/gate-related tests pass
except one unrelated failure, `test_gate.py::test_a_50_50_value_prediction_now_escalates`,
which belongs to a **different, parallel fix** (the required_slots/value-confidence
gate work, touching `gate.py`/`select.py`, done by a separate agent working at
the same time) — not caused by this change; `calibrate.py` was the only file
this task touched.

## Left for later (explicitly out of scope here)

There is still no **runtime** mechanism to detect "this specific position's
true answer is absent from the bank" the way dev calibration can (dev has
gold labels; a live turn does not — per-candidate softmax alone cannot signal
"the right answer isn't even a candidate"). A proper fix needs a
Learn-Then-Test-style two-stage design (arxiv.org/abs/2110.01052):
calibrate "is this situation supported" separately from "is this specific
response correct," each against its own objective and its own alpha. That
redesign is a genuinely open architecture item, not attempted in this pass.
