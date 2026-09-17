# D27 correction: the judge sample's "predicted_text" was wrong for 86/100 rows

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

## The fix

`build_llm_judge_sample.py` now fits one classifier **per act**
(`_fit_per_act_h7`, mirroring `_h7_arm`'s own per-act fit loop, including its
skip rule for acts with fewer than 2 fit classes) and keeps the fitted
objects. For a `skeleton_ok == False` row, it looks up the act sequence the
**predicted** skeleton actually implies (`spaces.skeleton_acts[pred_skeleton_id]`)
and scores each position against that act's classifier — not the row's own
gold act. `skeleton_ok == True` rows are unaffected (predicted and gold acts
coincide there by construction). Verified: all 86 wrong-skeleton rows'
`predicted_text` actually changed (not a no-op); the same 100 turn_ids were
sampled (turn selection doesn't depend on `predicted_text`); conditional
compose@1 reproduction is unchanged at 0.2765 (this script doesn't touch
compose@1 itself, only the illustrative text shown to the judge).

Corrected sample: `sft/eval/data/llm_judge_sample_corrected.jsonl`. Original
(buggy) sample left in place, untouched: `sft/eval/data/llm_judge_sample.jsonl`.

## Concrete before/after examples

| turn_id | gold | OLD (buggy) predicted_text | NEW (corrected) predicted_text |
|---|---|---|---|
| `test_seen:5166:6` | "to help me verify your identity, would you please provide your zip code, phone number, and email address? two out of those three options would be fine." | "i need to verify your identity. is there anything else i can help you with today?" | "i need to verify your identity." |
| `test_seen:9624:10` | "now i need to validate your purchase can i get your username, email address and order id?" | "i need to verify your identity. can i get your username, email address and order id?" | "can i get your username, email address and order id?" |
| `test_seen:924:2` | "ok. let me help you with that. one moment please." | "i would happy to help. sure." | "can i have your full name?" |

The pattern across most of the 86 changed rows: the OLD text was frequently
*closer* to gold than the NEW text is (unsurprising — it was built from gold
acts), which is exactly the leniency the bug introduced.

## New per-judge results, corrected sample vs. original (buggy) sample

| judge | appropriate (old → new) | borderline (old → new) | wrong (old → new) |
|---|---|---|---|
| GLM-5.3-Flash | 74% → 65.3% | 17% → 17.9% | 9% → **16.8%** |
| GLM-5.3 (full) | 68% → 62.4% | 25% → 26.9% | 7% → **10.8%** |
| gpt-oss-120b | 69% → 56% | 15% → 16% | 16% → **28%** |

Every judge's "wrong" rate roughly doubled once the predicted text reflected
what the system would actually output on a wrong-skeleton turn.

## Agreement, corrected sample

| pair | exact 3-way | coarse (adequate vs. wrong) |
|---|---|---|
| Flash vs. GLM-5.3 | 88.8% | 94.4% |
| Flash vs. gpt-oss-120b | 86.3% | 91.6% |
| GLM-5.3 vs. gpt-oss-120b | 81.7% | 84.9% |
| all three unanimous | 80.9% | 86.5% |

Agreement stayed high (slightly higher, in fact, than the original run) — the
correction didn't make the judges disagree more, it shifted what they agree
*on*, toward more "wrong" verdicts.

## Does "most misses are reasonable" survive?

Partially, weakened. On the corrected sample, "adequate" (appropriate +
borderline) is still the majority outcome for all three judges — roughly
79-91% adequate vs. 9-28% wrong, compare to the original's 91-93% adequate.
So the qualitative direction ("most compose@1 misses are not catastrophic
failures") is not reversed. But the **wrong-rate is now materially higher and
the range is wider**: 10.8-28% (median ~17%) on the corrected sample vs.
7-16% (median ~9%) on the buggy one. gpt-oss-120b — the non-GLM judge, and
the one that was already the stricter outlier before the fix — now puts the
real wrong-rate at **28%**, more than 1-in-4 misses being a genuine problem,
not roughly 1-in-6 as the buggy run suggested.

**Conclusion: the buggy version of D27 understated the real failure rate.**
The direction of the finding holds (most misses are not nonsense), but the
magnitude was optimistic by a meaningful margin — the corrected numbers,
not the original ones, are what should be quoted going forward.
