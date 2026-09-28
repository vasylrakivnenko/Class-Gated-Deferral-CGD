# PLAN — correct every known discrepancy and republish the leaderboard

**This file is the dev-spec. Work it top to bottom. Update the status column as each task lands so a
rate-limit interruption never loses the place.**

| | |
|---|---|
| Created | 2026-09-21 |
| Last updated | 2026-09-21 — A, B, C, E1, E2, E3, F1, F2 all DONE. 561 tests pass. Only F3/F4 decisions remain. |
| **Resume at** | **F3 + F4 — user decisions only; no code work outstanding** |
| Test baseline | `cd /Users/vasyl/zadumai/reflex-abcd && /Users/vasyl/zadumai/.venv/bin/python -m pytest -p no:warnings` → **561 passed** |
| Python | `/Users/vasyl/zadumai/.venv/bin/python` (the only interpreter with the deps) |
| Git root | `/Users/vasyl/zadumai` (one level up) — use `git -C /Users/vasyl/zadumai/reflex-abcd …` |
| Ground truth | `scratchpad/VERIFIED.md`; never copy a number from prose, always recompute from row-level data |

## Status legend

`TODO` · `WIP` · `DONE` · `BLOCKED` (needs a decision) · `DROPPED` (with reason)

---

## Task table

| id | task | status | depends on |
|---|---|---|---|
| A1 | Teacher-forcing: add `--history-mode` to the two n-gram baselines, default free-running | **DONE** | — |
| A2 | Teacher-forcing: same fix at the three committee sites | **DONE** | — |
| A3 | Tie-break: `_argmax` at the three committee sites | **DONE** | — |
| A4 | Judge sampler: default to `simple_random`, always write `stratum_weight` | **DONE** | — |
| A5 | Verify every builder writes RAW context; fix any that does not | **DONE** | — |
| B1 | Re-run `ngram_skeleton_baseline.py` → artifact | **DONE** | A1, A3 |
| B2 | Re-run `ngram_skeleton_plus_h7.py` → artifact | **DONE** | A1, A3 |
| B3 | Re-run `committee_gate_h7.py` → artifact (new buckets) | **DONE** | A2, A3 |
| B4 | Re-run `committee_gate.py` → artifact | **DONE** | A2, A3 |
| C1 | Re-derive the ungated cache's `p_app` with a design-correct estimator | **DONE** | — |
| C2 | Re-derive the gated rows' `p_app` under the **new** buckets | **DONE** | B3, C1 |
| C3 | Bound the tagged-context confound as far as the data allows | **DONE** | — |
| D1 | **BLOCKED** — re-judge a clean-context self-weighting sample (paid, ~$1) | **DOWNGRADED** — optional | user approval |
| E1 | Rebuild `Chat_Leaderboard.md`, remove the banner | **DONE** | B1–B4, C1–C3 |
| E2 | Update `DECISIONS.md` (D32, D33, new entry) | **DONE** | E1 |
| E3 | Update `DEFECTS_OPEN.md` | **DONE** | E1 |
| F1 | Triage the 42 unreviewed findings from lenses L2/L4/L7/L8/L9 | **DONE** | — |
| F2 | Fold in the running workflow's results (L1/L4/L7/L9/L10 + verifiers) | **DONE** | workflow |
| F3 | Resolve the four carried-over open items | TODO | — |

---

## Phase A — code fixes (no re-runs, no cost)

### A1 · Teacher-forcing in the n-gram baselines · `DONE`

`sft/eval/ngram_skeleton_baseline.py:226` and `sft/eval/ngram_skeleton_plus_h7.py:222` both do
`history.append(row.gold_skeleton_id)` — the model predicts turn *n* from the **true** act sequence
of turns 1…*n−1*, which does not exist at inference.

Add `--history-mode {free,gold}` defaulting to **`free`**; in free mode append the model's own
prediction. Record the mode in the output JSON. Keep `gold` available so the old number stays
reproducible, and label it in the artifact as the teacher-forced diagnostic it is.

**Verify.** Free-running conditional skeleton@1 must come out at `0.3139272271016311`, bit-identical
to the label-blind constant (measured 2026-09-21). If it does not, stop — the fix is wrong.

### A2 · Teacher-forcing at the committee sites · `DONE`

Same defect at `committee_gate_h7.py:150`, `committee_gate.py:130`,
`build_committee_judge_sample.py`. Same `--history-mode` flag, same default. For the gate this is
not merely an unfair comparison but an **unimplementable design** — say so in the docstring.

**Expected.** Unanimous bucket coverage 47.1% → 36.2%, its compose@1 34.1% → 31.4% (measured).

### A3 · Tie-break at the committee sites · `DONE`

The deterministic `_argmax` landed on the two standalone baselines but not on `committee_gate.py`,
`committee_gate_h7.py` or `build_committee_judge_sample.py`, which still use
`Counter.most_common(1)` (insertion-order). The same model is currently two different models
depending on which script runs it — 698 of 8,889 predictions differ. Port `_argmax` across.

### A4 · Judge-sampler design · `DONE`

`build_llm_judge_sample.py:146` defaults to `--design stratified`: equal allocation over 47
act-sequence strata (1–4 rows each; `('ASK',)` is 25.1% of the pool and 3% of the sample), and the
rows carry no `stratum_weight`, so the published share is an unweighted mean over failure *modes*
rather than over *turns*. Every other arm is self-weighting.

Two changes: default to `simple_random`, and **always** write `stratum_weight` +
`stratum_population_share` regardless of design, so `run_llm_judge._weighted_shares` can fire.

### A5 · Raw context in every builder · `DONE`

`build_llm_judge_sample.py:256` writes `h5row.context.text` (raw, correct) and
`build_committee_judge_sample.py` now writes `raw_context_by_turn` (fixed). Audit the remaining five
builders and confirm none passes `render_context` output to the judge. The **artifacts** are stale
regardless — see C3/D1.

---

## Phase B — local re-runs (free, CPU, no API)

Each writes to `outputs/probes/response/`. Back up the current artifact first; record old → new in
this file. Run in the background with a log (each takes minutes); never block for >2 min.

- **B1** `ngram_skeleton_baseline.py` — expect compose@1 and skeleton@1 to fall materially.
- **B2** `ngram_skeleton_plus_h7.py` — leaderboard row 11.
- **B3** `committee_gate_h7.py` — new `n_dis` buckets; this is what rows 1 and 6 rest on.
- **B4** `committee_gate.py` — also re-point to the certified H5 winner (carried-over item F3).

---

## Phase C — judge re-analysis (no new API calls)

### C1 · Design-correct `p_app` for the ungated cache · `DONE`

The three candidate samples and why none is clean:

| sample | n | design | context | reproducible |
|---|---|---|---|---|
| `llm_judge_sample_corrected.jsonl` | 100 | equal allocation ✗ | **raw ✓** | yes |
| `cache_matched_judge_sample.jsonl` | 200 | self-weighting ✓ | **tagged ✗** | **no producing script in repo** |
| `committee_judge_results.jsonl` | 100 | self-weighting ✓ | tagged (47/100) ✗ | yes, but old gate |

Post-stratifying the clean n=100 gives an effective n of ~16 (Kish) — uselessly wide.
Post-stratifying the tagged n=200 gives **C2 78.4% [73.8–82.7], C3 14.4%** vs the published
71.5% / 18.1%.

Report **both**, take the n=200 figure as the working estimate, and flag that it rests on a sample
with mangled context and no producer. This is what D1 exists to fix properly.

### C2 · Gated rows under the new buckets · `DONE`

After B3, recompute `n_dis` for the judged turns and re-derive rows 1 and 6. Row 1 should stop using
a 32-row slice of the equal-allocation sample; prefer the n=100 self-weighting committee sample —
but note it was drawn under the **old** gate, so its population no longer matches. If the mismatch
is material, fall back to bucket-specific rates from the n=200 sample and say so.

### C3 · Bound the tagged-context confound · `DONE`

163/200 and 47/100 rows show the judge `agent|r3|good r3|afternoon, …`. Inside the existing sample
the tagged/untagged split is perfectly confounded with opener/non-opener (all 37 untagged rows were
judged appropriate), so it cannot be bounded from within. Establish what *can* be said: how many
leaderboard rows are affected, in which direction the confound plausibly runs, and what D1 would
settle.

---

## Phase D — paid (needs approval)

### D1 · Re-judge a clean, self-weighting sample · `OPTIONAL` (no longer blocking)

**Why it is needed.** Defects 2 and 4 are entangled: no sample on disk is both self-weighting *and*
clean-context, and the one that is self-weighting has no producing script. Fixing the estimator
properly requires a sample that is both.

**Proposal.** Rebuild with `build_llm_judge_sample.py --design simple_random` (raw context,
reproducible) and judge ~300 rows with `gpt-oss-120b` **serverless** — per-token, no GPU, no
dedicated deployment, so nothing can be left billing. Estimated well under $1.

**Before spending:** dry-run the builder, smoke-test 10 real rows, confirm the cost, then run.

**Decision needed from the user.** Until then C1/C2 ship with the caveat.

---

## Phase E — republish

- **E1** Rebuild `Chat_Leaderboard.md` from the corrected artifacts; delete the warning banner;
  keep the correction log and the provenance table; carry forward the "top four are a statistical
  tie" finding and re-check whether it still holds after the corrections.
- **E2** `DECISIONS.md`: D32's n-gram claims die with teacher-forcing (the arm has zero skill
  free-running); D33's judge table needs the estimator note; add a dated entry for this cycle.
- **E3** `DEFECTS_OPEN.md`: close what is fixed, add what is not.

---

## Phase F — backlog

- **F1** 42 findings from lenses L2 (populations), L4 (input parity), L7 (judge protocol),
  L8 (cross-reference drift), L9 (statistical claims) are **unread and unverified** — every verifier
  in that run died on usage limits. Extract from `scratchpad/mca_old_results.md`, triage, fold the
  real ones into this plan as new tasks.
- **F2** The replacement workflow (`wf_8db7dc5c-7ac`) re-runs L1/L4/L7/L9/L10 *with* verifiers.
- **F3** Carried-over open items:
  1. `run_response_probe.py` `validate`: the certified config reproduces 1 of D7's 7 cells; another
     reproduces 4 of 7. Switching re-certifies a different vectorizer and moves every headline —
     **user's call**.
  2. `committee_gate.json` is stale against its own script (folded into B4).
  3. n-gram `max_order=4` is the untuned argparse default; order 1–2 scores better. The cache got a
     certified sweep — either sweep the baselines too or disclose the asymmetry.
  4. 1-NN TF-IDF draws from 71,133 train turns, LangCache from 43,159; matching the pools raises
     1-NN compose@1 from 4.92% to 6.22%.

- **F4 (new, 2026-09-21) — the qwen3-0.6B arms are not a clean supervision-format experiment.**
  `sft/build_structured_sft_dataset.py` says "Only the SUPERVISION differs", but
  `abcd_train_lexical.meta.json` has `n_examples` 71,133 while `abcd_train_structured.meta.json` has
  43,159 with `skipped_no_template_text: 27974` — the structured arm was trained on exactly the
  bank-coverable turns, i.e. on the metric's own population, and its own metadata asserts "same
  population … directly comparable", which is false. Verified signature: structured beats
  unconstrained by **+2.91** skeleton@1 points on the 3,985 fully-covered turns but only **+0.71**
  on the 4,904 out-of-bank turns, so ~80% of the edge sits where it was exclusively trained.
  Leaderboard row 2's margin over row 7 is therefore partly a training-population effect.
  Documented in `Chat_Leaderboard.md` limitation 3. Quantifying it end to end needs the
  unconstrained arm retrained on the 43,159 subset — a paid Fireworks job. **Not run; needs a
  decision.** Also noted: `sft/eval/checkpoints/smollm2_360m_8h/train_history.json` records
  `n_train` = 17,034, not 71,133 — worth checking before quoting SmolLM2 as trained on the same set.

---

## Log

| when | task | result |
|---|---|---|
| 2026-09-21 | — | Plan written. |
| 2026-09-21 | A1 | `--history-mode {free,gold}` added to both n-gram baselines, default **free**; `history_mode` now recorded in the artifact. **Gate passed:** free-running conditional skeleton@1 = `0.3139272271016311`, bit-identical to the label-blind constant. Modal-template compose@1 fell 0.10514 → 0.06374. Also rewrote the docstring section that had *justified* the teacher forcing — it was a documented convention, not an oversight, and the justification conflates prior TEXT (observable) with prior LABELS (not). |
| 2026-09-21 | A2 | Same `history_mode` fix at `committee_gate_h7.py`, `committee_gate.py`, `build_committee_judge_sample.py`. |
| 2026-09-21 | A3 | `_argmax` ported to all three committee sites; they were still on `Counter.most_common(1)`. |
| 2026-09-21 | A4 | `build_llm_judge_sample.py --design` default flipped `stratified` → `simple_random`. Note: the stale `llm_judge_sample_corrected.jsonl` predates the weighting code, which is why it carries no `stratum_weight`. |
| 2026-09-21 | A5 | Audited all seven builders: every one now writes RAW context to the judge. The defect survives only in two stale ARTIFACTS (163/200 and 47/100 tagged rows), which is what D1 would fix. |
| 2026-09-21 | — | `561 passed` after Phase A. |
| 2026-09-21 | B1–B4 | All four re-ran clean. n-gram+modal compose@1 0.10514 → **0.06374**, n-gram+H7 0.20151 → **0.14806**, both skeleton@1 0.41957 → **0.31393**. Gate unanimous bucket n 2758 → 2365, share 31.1% → **26.7%**, compose@1_cond 0.3419 → **0.3158**. Overall cache compose@1 0.27654 → 0.27729 (tie-break drift, as predicted). |
| 2026-09-21 | C1 | Two design-correct routes disagree: n=200 self-weighting gives **78.4%** (Kish 183), n=100 post-stratified gives 92.5% (Kish **16** — unusable). Took the n=200. Published was 71.5%, so the direction is certain even though the magnitude is not. |
| 2026-09-21 | C2 | **Key finding.** Under ONE consistent estimator the gate's benefit is +0.7 accuracy points, not +18.2. The published swing came from using an unweighted mean for the ungated row and a 32-row slice for the gated row, then attributing the difference to gating. The gate declines 64% of traffic for no measurable quality gain. |
| 2026-09-21 | C3 | Paired evidence on the 10 turns judged in BOTH the tagged and clean samples: 8/10 identical, **all 4 "appropriate" held**, 2 flips both within the non-appropriate band and in opposite directions. So the tagged context does not appear to move `p_appropriate`, which is the only thing feeding ②. This downgrades D1 from blocking to optional. |
| 2026-09-21 | E2/E3 | `DECISIONS.md` (+194 lines: D32 rewritten, D33 estimator note, new D35) and `DEFECTS_OPEN.md` (+240: new A-0 fixed section, new section E with six open items). Both were drafted against my interim +0.7 figure, so I then corrected 10 sites in DECISIONS.md and 3 in DEFECTS_OPEN.md to the final within-bucket numbers (ungated **77.2%**, gate-all-3 **86.3%**, gate-≤1 **83.0%**, benefit **+9.1 / -3.8**) and left an explicit note that the interim estimator was superseded. |
| 2026-09-21 | — | Two claims from that pass verified and folded in: (1) the free-running n-gram emits `S0000` on **all 8,889 rows** — it does not merely tie the label-blind constant, it IS the constant, and row 11's compose@1 `0.06373902132998745` is bit-identical too; (2) the tagged/clean paired overlap holds **5** appropriate verdicts, not 4 (3 of them on tagged rows). Both corrected in `Chat_Leaderboard.md`. |
| 2026-09-21 | — | `recall_at_k.py` re-run — it was the last stale artifact. Its unanimous bucket goes n 1,869 @ 34.19% → **1,441 @ 31.58%**, now agreeing with `committee_gate_h7.json`. Cache conditional compose@1 settles at **0.27729**; both artifacts agree to 5 dp. |
| 2026-09-21 | F2 | Replacement lens workflow finished 10/10 with verifiers. Its L1 lens **caught a real error in my own E1 rebuild**: I had post-stratified every row to its bucket's act-sequence mix, which assumes the appropriate-rate within an act stratum is identical inside and outside the bucket — exactly what a gate violates. It washed the gate's effect out to +0.7 pts. Corrected to a within-bucket direct SRS estimate (60/145/200 judged rows), which is also the estimator every non-cache arm always used, so the whole table is now on one footing. **Gate benefit is +9.1 accuracy points, −3.8 errors.** Corroborated independently by `committee_judge_results.jsonl` (n=100 SRS of that bucket, p_app 0.88). Leaderboard rewritten a third time. |
| 2026-09-21 | F1 | Triaged the 42 orphaned findings. L4/L7/L9/L1 were re-covered with verifiers by the replacement run, so only L2 (populations) and L8 (cross-reference drift) were genuinely orphaned. Their top findings mostly DUPLICATE what is already fixed (the estimator, the 32-row slice, tagged context, the 1-NN pool, H5_SKELETON_BAR). **Two were new**, both verified and handled: (a) `redis_vector_5000.json` and `redis_vector_smoke.json` hardcoded LangCache compose@1 = 0.0825 where the artifact records 0.0275 — both patched, round-trip proven to change only that block; (b) **the qwen3-0.6B arms were trained on different populations** — see the new F4. |
| 2026-09-21 | E1 | `Chat_Leaderboard.md` rebuilt from the corrected artifacts with one estimator throughout; banner removed. **Qwen3-4B is now #1 on accuracy (87.4%)**; the cache's three rows collapse into a tight 78.4–80.3% cluster; rows 1 vs 6 are the one separable pair. |
