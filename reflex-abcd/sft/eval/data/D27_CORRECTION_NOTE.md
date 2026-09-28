# D27 correction: the judge sample's "predicted_text" was wrong for 86/100 rows

> **Re-verified 2026-09-20** against `sft/eval/data/judge_multi_corrected/`,
> `sft/eval/data/judge_multi/`, `llm_judge_sample.jsonl` and
> `llm_judge_sample_corrected.jsonl`. The bug description, the 86/100 count and
> the three before/after examples reproduce exactly. **Four quantitative claims
> were wrong and are corrected below** (marked CORRECTED); every per-judge rate
> now names the denominator it was computed over, which the original version of
> this note did not.

## The bug

`sft/eval/build_llm_judge_sample.py` built `predicted_text` from `_h7_arm`'s
`_pred_by_position`, which is keyed by `(turn_id, position)` and computed by
scoring each position WITHIN THAT ROW'S OWN GOLD ACT's per-act classifier
pool (`_h7_arm` partitions both fit and eval rows by `r.act`, and for an
`H7Row` that field is always the gold act at that position, regardless of
what any skeleton predictor said).

For the 86 of 100 sampled turns where `skeleton_ok == False` (the H5 skeleton
classifier picked the wrong skeleton), this means the text shown to the judge
as "the cache's predicted answer" was assembled using the *correct* act at
every position — something the real decoder (`select.py`'s
`_score_templates`, which scores `(query, act_id)` with no gold dependency)
would never produce once it has already committed to a wrong skeleton. It was
an easier, non-deployable hypothetical, not the system's actual output.

**Verified 2026-09-20:** exactly **86 of the 100 sampled rows have
`skeleton_ok == False`**, and exactly those **86 rows' `predicted_text`
changed** between `llm_judge_sample.jsonl` and `llm_judge_sample_corrected.jsonl`;
the two files carry the same 100 `turn_id`s.

## The fix

`build_llm_judge_sample.py` now fits one classifier **per act**
(`_fit_per_act_h7`, mirroring `_h7_arm`'s own per-act fit loop, including its
skip rule for acts with fewer than 2 fit classes) and keeps the fitted
objects. For a `skeleton_ok == False` row, it looks up the act sequence the
**predicted** skeleton actually implies (`spaces.skeleton_acts[pred_skeleton_id]`)
and scores each position against that act's classifier — not the row's own
gold act. `skeleton_ok == True` rows are unaffected (predicted and gold acts
coincide there by construction). Conditional compose@1 reproduction is
unchanged at **0.2765370138017566 on n=3,985** (this script doesn't touch
compose@1 itself, only the illustrative text shown to the judge).

> **Caveat added 2026-09-20:** that 0.27654 is the *published* value
> (`outputs/probes/response/select.json`, `committee_gate_h7.json`). Re-running
> the probe on the current working tree gives 1,105/3,985 = **0.27729** — the
> upstream `src/reflex/data.py` leakage fix moved 3 turns. Inside the CI
> (half-width 1.54 points), no conclusion changes, but this sample was built
> against the 0.27654 tree.

Corrected sample: `sft/eval/data/llm_judge_sample_corrected.jsonl`. Original
(buggy) sample left in place, untouched: `sft/eval/data/llm_judge_sample.jsonl`.

## Concrete before/after examples

All three re-read from the two jsonl files on 2026-09-20 and byte-identical to
what is printed here.

| turn_id | gold | OLD (buggy) predicted_text | NEW (corrected) predicted_text |
|---|---|---|---|
| `test_seen:5166:6` | "to help me verify your identity, would you please provide your zip code, phone number, and email address? two out of those three options would be fine." | "i need to verify your identity. is there anything else i can help you with today?" | "i need to verify your identity." |
| `test_seen:9624:10` | "now i need to validate your purchase can i get your username, email address and order id?" | "i need to verify your identity. can i get your username, email address and order id?" | "can i get your username, email address and order id?" |
| `test_seen:924:2` | "ok. let me help you with that. one moment please." | "i would happy to help. sure." | "can i have your full name?" |

The pattern across most of the 86 changed rows: the OLD text was frequently
*closer* to gold than the NEW text is (unsurprising — it was built from gold
acts), which is exactly the leniency the bug introduced.

## New per-judge results, corrected sample vs. original (buggy) sample

**Denominators differ by judge and between the two runs**, because unparseable
verdicts are dropped: the shares below are over `n_scored`, not over the 100
sampled rows. Both samples contain 100 rows.

| judge | n_scored (old → new) | parse errors (old → new) | appropriate (old → new) | borderline (old → new) | wrong (old → new) |
|---|---|---|---|---|---|
| GLM-5.3-Flash | 100 → **95** | 0 → **5** | 74.0% → 65.3% | 17.0% → 17.9% | 9.0% → **16.8%** |
| GLM-5.3 (full) | 97 → **93** | 3 → **7** | 68.0% → 62.4% | 24.7% → 26.9% | 7.2% → **10.8%** |
| gpt-oss-120b | 100 → 100 | 0 → 0 | 69.0% → 56.0% | 15.0% → 16.0% | 16.0% → **28.0%** |

Source: `judge_multi/summary.json` and `judge_multi_corrected/summary.json`,
`per_model.<model>.share_*` and `counts`.

**CORRECTED — "every judge's wrong rate roughly doubled".** It rose on all
three, by **1.5× to 1.9×**, not uniformly 2×: Flash 9.0% → 16.8% (×1.87, and
note the denominator shrank from 100 to 95), GLM-5.3 7.2% → 10.8% (×1.49),
gpt-oss-120b 16.0% → 28.0% (×1.75, same denominator).

## Agreement, corrected sample

Agreement is computed over the rows BOTH judges scored, so each cell has its own
denominator (`agreement.pairwise.<pair>.n_compared`).

| pair | n_compared | exact 3-way | coarse (adequate vs. wrong) |
|---|---:|---|---|
| Flash vs. GLM-5.3 | 89 | 88.8% | 94.4% |
| Flash vs. gpt-oss-120b | 95 | 86.3% | 91.6% |
| GLM-5.3 vs. gpt-oss-120b | 93 | 81.7% | 84.9% |
| all three unanimous | 89 | 80.9% | 86.5% |

**CORRECTED — "agreement stayed high (slightly higher, in fact, than the
original run)".** True of the *exact 3-way* figures only. Against the original
run (`judge_multi/summary.json`, n_compared 97 / 100 / 97 / 97):

- exact 3-way ROSE on every pair: 87.6 → 88.8, 81.0 → 86.3, 76.3 → 81.7,
  unanimous 73.2 → 80.9.
- coarse adequate-vs-wrong **FELL** on two of three pairs and on unanimity:
  Flash vs GLM-5.3 **100.0 → 94.4**, GLM-5.3 vs gpt-oss **89.7 → 84.9**,
  unanimous **89.7 → 86.5**; only Flash vs gpt-oss rose, 89.0 → 91.6.

So the correction did make the judges disagree more about *whether a row is
wrong* — which is the axis this note is about — while agreeing more about the
exact three-way label.

## Does "most misses are reasonable" survive?

Partially, weakened. On the corrected sample, "adequate" (appropriate +
borderline) is still the majority outcome for all three judges, and
the qualitative direction ("most compose@1 misses are not catastrophic
failures") is not reversed.

**CORRECTED — the two adequate ranges were both wrong.** Recomputed as
`share_appropriate + share_borderline` per judge, each over its own `n_scored`:

| sample | adequate range | wrong range |
|---|---|---|
| corrected | **72.0% (gpt-oss, n=100) – 89.2% (GLM-5.3, n=93)**; Flash 83.2% (n=95) | **10.8% – 28.0%**, median 16.8% |
| original (buggy) | **84.0% (gpt-oss, n=100) – 92.8% (GLM-5.3, n=97)**; Flash 91.0% (n=100) | 7.2% – 16.0%, median 9.0% |

(The note previously said "roughly 79-91% adequate vs. 9-28% wrong" against
"the original's 91-93% adequate". Both ranges dropped gpt-oss-120b, the
strictest judge, which is the one the conclusion turns on.)

The wrong-rate is materially higher and the range is wider: **10.8-28.0%
(median 16.8%) corrected vs. 7.2-16.0% (median 9.0%) buggy**. gpt-oss-120b —
the non-GLM judge, and the one that was already the stricter outlier before the
fix — now puts the wrong-rate at **28.0% of its 100 scored rows**, more than
1 in 4 misses being a genuine problem, not roughly 1 in 6 as the buggy run
suggested.

**Conclusion: the buggy version of D27 understated the real failure rate.**
The direction of the finding holds (most misses are not nonsense), but the
magnitude was optimistic by a meaningful margin — the corrected numbers,
not the original ones, are what should be quoted going forward.

## Two things to carry with these numbers wherever they are quoted

1. **D33's three-judge table prints the wrong row for the learned cache.** The
   appropriate-share range for this arm, read from
   `judge_multi_corrected/*.jsonl`, is **56.0% (gpt-oss, n=100) – 65.3% (Flash,
   n=95)**. D33 prints 62-78%; the 78% is the LangCache row above it. With the
   correct row the learned cache is the **lowest-appropriate arm of the five**,
   not fourth. D33's wrong-rate headline does survive: cache wrong 10.8-28.0%
   vs RNN skeleton+H7 wrong 17.4-30.0%, so the RNN is still worst on wrong-rate.
2. **This sample is stratified, not simple-random.** `build_llm_judge_sample.py`
   draws equal allocation per stratum and the shares above are UNWEIGHTED, so
   they are not the miss-pool shares and the effective sample size is
   materially below 100. The on-disk rows carry no `stratum_weight` field, so
   the weighted shares and the Kish effective n must be recomputed by rerunning
   the builder (`run_llm_judge._weighted_shares` now reports both). Do not place
   these percentages beside the simple-random arms without that caveat; for a
   directly comparable number, rebuild with `--design simple_random` and
   re-judge (paid API).
