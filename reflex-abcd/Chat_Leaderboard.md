# Deflection leaderboard — how much traffic each method keeps off the big LLM

**Rebuilt 2026-09-21** after a methodological-consistency audit found four defects in the
2026-09-20 version. Every figure below was recomputed from a named artifact, and every judged rate
uses **one** estimator. What changed and why is in the correction log at the bottom.

---

## What the columns mean

For a turn routed to a method instead of to the big LLM, the outcome is one of four things:

| outcome | share |
|---|---|
| exactly right by the mechanical metric | `c` = compose@1 |
| not exact, but an LLM judge called it appropriate | `(1−c) · p_appropriate` |
| borderline | `(1−c) · p_borderline` |
| wrong | `(1−c) · p_wrong` |

- **① compose@1** = `c`
- **② accuracy** = `c + (1−c)·p_appropriate` — the share you could actually stop sending
- **③ errors delivered** = `(1−c)·p_wrong` — the bad answers you ship to get it

② + ③ + borderline = 100%. **answers** is the share of the population the method takes on; a method
that abstains or is gated takes on less.

**Judge:** `gpt-oss-120b`, the same judge for every arm, scoring each arm's compose@1 **misses**.
It was the harshest of the three judges used elsewhere, so ② is a conservative floor.

**Estimator (this is what changed).** Every arm's `p_appropriate` and `p_wrong` is now the plain
share over that arm's own judged rows, with a Wilson interval — the design-correct estimate for a
simple random sample of misses, which is what every non-cache arm always used. For a gated row the
rows are those falling **inside that bucket** (an SRS restricted to a subset is still an SRS of the
subset). Previously the cache's rows used equal allocation over act-sequence strata with an
unweighted mean, and were ranked against arms using the plain share.

**Population:** `fully_covered` turns on `test_seen` — every gold position covered by the bank.
n = 3,985 unless the row says otherwise.

---

## The leaderboard

| # | method | answers | ① compose@1 | ② accuracy (95% CI) | ③ errors | errors per 100 saved | judged n | pop |
|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3-4B SFT** | 100% | **31.8%** | **87.0%** [83.0–90.3] | 7.5% | 8.6 | 200 | 415 |
| 2 | qwen3-0.6B **structured** SFT | 100% | 24.2% | 86.4% [79.8–91.1] | **6.1%** | **7.0** | 100 | 3,985 |
| 3 | learned cache, gate: all 3 agree | 36.2% | 31.6% | 86.3% [78.3–91.9] | 11.4% | 13.2 | 60 | 3,985 |
| 4 | learned cache, gate: ≤1 disagrees | 77.2% | 31.5% | 83.0% [77.8–87.3] | 11.8% | 14.2 | 145 | 3,985 |
| 5 | SmolLM2-8h SFT | 100% | 17.3% | 82.6% [75.2–88.3] | 10.8% | 13.0 | 100 | 191 |
| 6 | learned cache, **ungated** | 100% | 27.7% | 77.2% [72.4–81.6] | 15.2% | 19.7 | 200 | 3,985 |
| 7 | qwen3-0.6B unconstrained SFT | 100% | 21.8% | 74.2% [66.0–81.3] | 12.9% | 17.4 | 85 | 3,985 |
| 8 | RNN skeleton + certified H7 | 100% | 25.1% | 69.3% [61.9–76.1] | 22.5% | 32.4 | 100 | 3,985 |
| — | LangCache — *not comparable, see note* | 56.6% | 4.9% | ~70.5% | ~13.3% | ~18.9 | 100 | 800 |
| 9 | fully-RNN (skeleton + templates) | 100% | 19.3% | never judged | — | — | — | 3,985 |
| 10 | n-gram skeleton + certified H7 | 100% | **14.8%** | never judged | — | — | — | 3,985 |
| 11 | n-gram + modal template | 100% | **6.4%** | never judged | — | — | — | 3,985 |
| 12 | 1-NN TF-IDF retrieval | 100% | 5.2% | never judged | — | — | — | 3,985 |

**LangCache** is excluded from the ranking: its judged sample was drawn only from rows it *answered*,
and its compose@1 is not reproducible — it swings between 2.75% and 8.25% across identical re-runs
purely from tie-breaks among similarity-1.0 duplicate contexts. The stable part is the
non-duplicate bucket, 19/302 = 6.3%.

### What is and is not statistically separable

- **Rows 1–5 are one statistical cluster** — every pairwise CI overlaps. Do not read an ordering
  into 87.0 / 86.4 / 86.3 / 83.0 / 82.6. Row 3 rests on 60 judged rows and spans [78.3–91.9].
- **Row 1 vs row 6 IS separable**: [83.0–90.3] and [72.4–81.6] are disjoint. Qwen3-4B beats the
  *ungated* cache on accuracy.
- **Row 3 vs row 6** — gating the cache — is the one within-method contrast that is close to real:
  86.3% [78.3–91.9] against 77.2% [72.4–81.6], overlapping but only just.
- Row 8 (RNN) is separable from rows 1–5 and is the worst judged arm.

---

## What the table says

**Gating the cache is worth about nine accuracy points, not eighteen.**

| | ungated | gate: all 3 agree | delta |
|---|---|---|---|
| accuracy | 77.2% | 86.3% | **+9.1** |
| errors | 15.2% | 11.4% | **−3.8** |
| answers | 100% | 36.2% | −63.8 |
| traffic saved | 77.2% | **31.2%** | −46.0 |

The previous version of this file reported 71.5% → 89.7% and 18.1% → 6.2%, a +18.2/−11.9 swing.
Roughly half of that was real and roughly half was an estimator artifact: the ungated row used an
unweighted mean of an equal-allocation sample while the gated row used a 32-row slice of the same
sample. Corrected, the gate still buys a genuine quality improvement — it just costs 64% of the
traffic to get it, and it does not reach the LLM arms' error rates.

**The fine-tuned LLMs still lead on the combination.** Rows 1 and 2 match the gated cache's accuracy
while answering *every* turn, at roughly half its error rate. qwen3-0.6B structured is the standout
— 86.4% accuracy and the lowest error rate on the board at 6.1% — but see limitation 3 before
leaning on it.

**Where the cache competes outright is exact match.** Gated, its compose@1 is 31.6% against
Qwen3-4B's 31.8% — a tie, with McNemar p = 0.586 on the shared 415 turns. If you need an
exactly-correct canned response rather than an acceptable one, the cache matches a 4B model at a
fraction of the cost. That is about ①, not ②.

**The n-gram arm is the constant predictor.** Rows 10 and 11 fell from 20.2% and 10.5% because the
model was being fed the test conversation's gold act-sequence. Free-running it emits `S0000` on
**all 8,889 rows** — one distinct prediction — so it does not merely *tie* the label-blind constant,
it *is* it: skeleton@1 `0.3139272271016311`, and for row 11 compose@1 `0.06373902132998745`, both
bit-identical to the constant. Row 10 scores higher (14.8%) only because it pairs that constant
skeleton with the real text-based H7 template classifiers.

---

## All arms on one matched population

Rows 1 and 3 sit on smaller slices. Those slices are strict subsets of the 3,985 population
(and 191 ⊂ 415), so every arm can be scored on identical turns. compose@1:

| arm | same 415 turns | same 191 turns |
|---|---|---|
| Qwen3-4B | 0.3181 | 0.3246 |
| learned cache | 0.3036 | 0.2880 |
| RNN + certified H7 | 0.2723 | 0.2565 |
| qwen3-0.6B structured | 0.2578 | 0.2513 |
| qwen3-0.6B unconstrained | 0.2265 | 0.2199 |
| n-gram + H7 (teacher-forced) | 0.2169 | 0.2094 |

Cache vs Qwen3-4B on the 415: 39 cache-only wins, 45 Qwen-only wins, exact McNemar two-sided
**p = 0.586**.

---

## Correction log — 2026-09-20 → 2026-09-21

**1. The n-gram was teacher-forced on test gold labels.** `ngram_skeleton_plus_h7.py:222`,
`ngram_skeleton_baseline.py:226`, `committee_gate_h7.py:150`, `committee_gate.py:130` all appended
`row.gold_skeleton_id` to the prediction history. Those are labels, not observable input. Fixed with
`--history-mode {free,gold}`, default `free`; all four artifacts re-run.

| | was | now |
|---|---|---|
| n-gram + H7 compose@1 | 20.2% | **14.8%** |
| n-gram + modal compose@1 | 10.5% | **6.4%** |
| n-gram skeleton@1 (cond) | 42.8% | **31.4%** = the label-blind constant |
| gate unanimous coverage | 47.1% | **36.2%** |
| gate unanimous compose@1 | 34.2% | **31.6%** |

Note this was a *documented* convention, not an oversight — the script's docstring argued that other
heads also condition on true prior state. That argument conflates prior **text** (observable at
inference) with prior **labels** (not), and the arms being compared receive text alone.

**2. The cache's judged rate used the wrong estimator.** `build_llm_judge_sample.py` defaulted to
equal allocation over 47 act-sequence strata (1–4 rows each; `('ASK',)` is 25.1% of the pool and 3%
of the sample) with no `stratum_weight` written, so the published figure was an unweighted mean over
failure *modes* while every other arm averaged over *turns*. Default flipped to `simple_random`, and
all three cache rows re-estimated from the n=200 self-weighting sample. Ungated cache:
**71.5% → 77.2%**.

**3. All three cache rows were affected, not just the ungated one** — rows 3, 4 and 6 are nested
slices of that same equal-allocation sample, so the gated rows inherited the defect in full. Each is
now estimated from the judged rows falling inside its own bucket (60 / 145 / 200). Independent
corroboration for row 3: `committee_judge_results.jsonl` is a dedicated 100-row simple random sample
of exactly that bucket, judged by the same model, and gives p_appropriate 0.88 — 0.89 restricted to
the 74 of its rows still in the bucket after the gate was recomputed — against 0.80 from the n=200
route. Two independent samples, same direction.

*An earlier pass on 2026-09-21 post-stratified every row to its bucket's act-sequence mix. That was
wrong: it assumes the appropriate-rate within an act stratum is the same inside and outside the
bucket, which is precisely what a gate is supposed to violate, and it washed the gate's effect out
to +0.7 points. The within-bucket estimate above is the correct one.*

**4. The deterministic tie-break was missing at the committee sites.** `_argmax` had landed on the
two standalone baselines but not on `committee_gate.py`, `committee_gate_h7.py` or
`build_committee_judge_sample.py`, which still used `Counter.most_common(1)` — 698 of 8,889
predictions differ between the rules. Ported across.

**5. qwen3-0.6B unconstrained's judged sample was contaminated** (15 of 100 judged "misses" were
actually compose@1 hits, all judged appropriate). Those rows are now excluded: **78.1% → 74.9%**.

---

## Provenance

| figure | file |
|---|---|
| cache compose@1, gate buckets | `outputs/probes/response/committee_gate_h7.json` (re-run 2026-09-21) |
| cache skeleton@1 0.5821831870 (n=3,985) | `outputs/probes/response/recall_at_k.json` |
| n-gram arms | `ngram_skeleton_plus_h7.json`, `ngram_skeleton_baseline.json` (re-run, `history_mode: free`) |
| RNN + H7 | `outputs/probes/response/rnn_skeleton_plus_h7.json` |
| fully-RNN | `outputs/probes/response/rnn_h7_template.json` |
| 1-NN TF-IDF | `outputs/probes/response/retrieval_baseline.json` |
| qwen3-0.6B structured / unconstrained | `sft/eval/data/gen_{structured,unconstrained}_scored.json` |
| Qwen3-4B / SmolLM2-8h | `qwen3_4b_score_1000.json`, `smollm2_8h_score_500.json` |
| LangCache | `outputs/probes/response/langcache_baseline_mid.json` |
| all judge verdicts | `sft/eval/data/*_judge_multi/gpt-oss-120b.jsonl`, `sft/eval/data/*_judge_results.jsonl` |

**Two different fits are both called "the cache."** Every number here uses the **compose-winner**
fit (`{C: 4.0, sublinear_tf: true}` = `select.json` → `selection.winners.compose`), which is what
`recall_at_k.py` and `committee_gate_h7.py` use and the right choice for a compose@1 leaderboard.
`select.json`'s headline H5 (0.4317 over n=8,858) is the **certified standalone H5 head**, a
different fit. The compose-winner pipeline's own n=8,858 skeleton rate is 0.4274.

**Do not copy a number from prose.** Both `DECISIONS.md` and the `compare_against` strings inside
the artifacts have repeatedly carried wrong values. Go to the row-level data.

---

## Limitations

1. **The judges are not validated against humans** (the project's own D24 caveat). LLM-judge
   leniency bias is a known phenomenon; ② is a directional read, not a measured accuracy.
2. **Two cache judge samples show the judge mangled context.** 163/200 rows in
   `cache_matched_judge_sample.jsonl` and 47/100 in `committee_judge_sample.jsonl` carry the
   featurizer's recency-tagged rendering (`agent|r3|good r3|afternoon, …`) where every other arm's
   sample carries clean prose. All seven builders now write raw context, but these artifacts predate
   the fix. Direct evidence that this matters little for ②: on the 10 turns judged in *both* the
   tagged and the clean sample, 8 verdicts were identical, **all 5 "appropriate" verdicts held**
   (3 of them on tagged rows), and
   the 2 flips were both *within* the non-appropriate band and in opposite directions. Re-judging
   ~300 rows through `gpt-oss-120b` serverless (well under $1) would settle it.
3. **The qwen3-0.6B arms were trained on different populations**, though the build script says
   "Only the SUPERVISION differs". `sft/data/abcd_train_lexical.meta.json` records `n_examples`
   71,133; `sft/data/abcd_train_structured.meta.json` records 43,159 with
   `skipped_no_template_text: 27974` — the structured arm was trained on exactly the bank-coverable
   turns, i.e. on the metric's own population, while its metadata claims "same population …
   directly comparable". The signature is visible in the artifacts: structured beats unconstrained
   by **+2.91** skeleton@1 points on the 3,985 fully-covered turns but only **+0.71** on the 4,904
   out-of-bank turns, so ~80% of its edge sits where it was exclusively trained. Row 2's advantage
   over row 7 is therefore partly a training-population effect, not a supervision-format effect.
   Quantifying it end to end needs the unconstrained arm retrained on the 43,159 subset (a paid
   Fireworks job, not run). Qwen3-4B and SmolLM2 also train on the 71,133 set.
4. **Rows 1 and 3 sit on 415- and 191-turn slices**, which are contiguous blocks of conversations
   rather than random samples of `test_seen`. Use the matched-population table for comparisons.
5. **LangCache is not reproducible** — see the note under the table.
6. **Rows 9–12 were never judged**, so their ② and ③ are unknown, not zero.
7. `select.json` still records `certified: true` under a ranking rule that the corrected code would
   not grant; the certified config reproduces 1 of D7's 7 recorded cells where another reproduces 4
   of 7. Switching re-certifies a different vectorizer and would move every headline — an open
   decision, tracked in `PLAN.md` F3.
