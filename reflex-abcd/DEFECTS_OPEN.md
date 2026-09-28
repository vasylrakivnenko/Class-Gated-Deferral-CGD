# Open defects — compile stage, training, and the 2026-09-19/20 audit remainder

**Re-verified 2026-09-20 by reading the code and recomputing from the artifacts
on disk.** No status below was carried over from this file's earlier claims;
several of those had gone stale and are corrected in the last section.

> **2026-09-21 update.** A methodological-consistency cycle closed five more
> defects (teacher forcing, judge-sampler design, the gate's estimator, the
> committee tie-break, a contaminated judged sample) and re-ran four artifacts.
> They are recorded in section **A-0** below, and the items they opened or left
> open in section **E**. The five closures moved published numbers — the n-gram arm
> turns out to have **zero skill** and the gate's accuracy benefit was an
> **estimator artifact** — so read A-0 before quoting anything from an
> `outputs/probes/response/` artifact or a judged rate. DECISIONS **D35** is the
> narrative version.

## Reference state every number below was recomputed against

| fact | value | source |
|---|---|---|
| bank on disk | `bank_hash c270df0e249444bb` — 4,489 templates / 570 skeletons / 30 action patterns / 16 slots | `outputs/compile/bank/meta.json` |
| corpus | `dataset_hash 40ce6eaa5279aeeb`; 7,467 train conversations, 105,672 agent turns, **71,133 retrieve_utterance turns**, 27,072 take_action turns, 89,623 sentences | `outputs/compile/compile_summary.md` |
| test suite | **561 passed, 0 failed, 0 skipped** | `/Users/vasyl/zadumai/.venv/bin/python -m pytest -p no:warnings`, 2026-09-20 |
| compile regression tests | `tests/test_compile.py` collects **12** (8 parametrised value-typing cases + 4 required-slot tests) | `pytest --collect-only` |

**Status:** D-1, D-3, D-4 **FIXED** and re-verified against the bank on disk.
D-6 fixed in code, its two artifacts never regenerated. D-7 **half fixed** (the
merge guard shipped and is measured; strict coverage still unmeasured).
**D-2, D-5, D-8, D-9, D-10 remain OPEN.** Sections A–D carry the audit findings
of 2026-09-19/20 that were *not* fixed. **Section A-0 carries the five defects
closed on 2026-09-21 and section E the six items that cycle left open.**

**Authorship of `compile.py` is unsettled** — the compile agent claimed the file,
then said it is another agent's implementation with its two fixes on top. The
code is tested and the numbers reproduce; nothing depends on the provenance.

---

## D-1. PII leak — literal usernames in the bank. **FIXED, and now re-verified.**

**Fix in code:** `compile.py:623` `_derived_username_stems` (delexicalization
tier 2.5, gated by `compile.derive_username_from_name`), applied at
`compile.py:742-743`; `compile.py:1486` `assert_no_leaked_usernames`, called
inside `compile_bank` at `compile.py:1681` *before* the `Bank` is constructed, so
a leak fails the compile instead of warning.

**Re-verified today against the bank on disk** (this is the confirming run the
old addendum said was pending):

- `compile.assert_no_leaked_usernames(bank.templates, ontology)` **passes** on
  `bank_hash c270df0e249444bb`.
- Zero literal persona usernames survive: `grep -c "username is [a-z]*[0-9]"
  outputs/compile/bank/templates.jsonl` → **0**.
- **23 of the 4,489 templates now carry the `{username}` marker** instead — the
  masked form of what used to leak.

**Pre-fix measurement, kept for the record, NOT re-measured** (the evidence it
was measured on no longer exists in the bank): 22 templates / 248 occurrences /
10 personas leaked; agent-stated usernames in train were 296 occurrences / 25
distinct, **0** equal to `scenario.personal.username` and 288 = 97.3% equal to
first-initial + surname (+ digits) of `customer_name` — which is why the
exact-match tier had nothing to catch. The residual 8 of 296 that do not match
the stem rule were never inspected.

**Gap left open:** there is **no regression test for the guard.** `grep -rn
"leaked_username\|derive_username" tests/` returns nothing, so a future edit to
the delexicalizer can reopen the leak and the 561-test suite will stay green.
`tests/test_compile.py`'s 12 tests are all D-3/D-4 typing tests.

---

## D-2. The delexicalization quality metric is still vacuous — and the honest figure is now UNMEASURED. **OPEN.**

The defect survives the recompile, with new numbers:

- `outputs/compile/delex_check.md:45-46` reports **4,486 of 4,489 templates
  (99.93%)** free of *machine-visible* defects (line numbers as of 2026-09-20,
  after the stale-artifact banner was prepended to that file).
- But **only 71 of the 4,489 templates (1.58%) bear any slot at all** (recomputed
  from `outputs/compile/bank/templates.jsonl`: 71 templates, 71 markers,
  **468 train occurrences** weighted by template `count`; markers are `username`
  23, `customer_name` 21, `amount` 13, `name` 8, `account_id` 2,
  `street_address` 2, `email` 2). A random sample is trivially clean because
  delexicalization almost never fires. That is the defect, and it is unchanged.
- The 3 machine-visible defects are all the same failure — a lost brace:
  `T000525 "…$1amount"`, `T001232 "…will be 1amount."`,
  `T001678 "a refund of $1amount…"` (`compile._delex_defects`, run today).

**The honest hand-checked figure is STALE and must be re-measured.** The
"24/49 templates and 53/129 occurrences correctly masked = 49% / 41%" in the
previous version of this file was hand-checked against the **pre-fix bank**,
which had 49 slot-bearing templates. Today's bank has **71** (D-1's masking tier
added the username ones), so neither the numerator nor the denominator survives.
The manual tally (`delex_check.md:55-61`) is **empty**, and the banner now on
that file says not to fill it in until the sheet is regenerated at n=200.

*Therefore the claim "the spec's 98% bar is not met" is currently unproven, not
false.* What is established: the 99.93% is not the spec's metric (the file says
so itself at lines 39-44), and it is computed over a population in which 98.4%
of rows cannot fail. **To re-establish the honest figure:** regenerate the sheet
(`PYTHONPATH=src python -m reflex compile`), then read the 71 pairs in
`delex_check.md`'s "Every slot-bearing template" section (line 1106 onward —
that section is the complete set of 71, not a sample) against their source turns
and fill in the tally. Publish it beside the 99.93%, never instead of it.

---

## D-3. Mistyped literals corrupted half the slot-bearing bank. **FIXED — at three call sites, not two.**

`_type_literal(value, [], {})` used to be called with empty candidate categories
and an empty enumerable index, so the shape patterns fired unconstrained and
`customer_name`'s `^[a-z]+(?:[ '-][a-z]+)+$` matched any two-word phrase.

**All three call sites now pass the button's declared categories and the
ontology's enumerable lists:**

| site | function | evidence |
|---|---|---|
| `compile.py:1354` | `extract_action_patterns` | builds `declared` + `enumerable` at 1333-1341 |
| `compile.py:1452` | `_conversation_sources` | builds `action_categories` + `ontology_enumerable` at 1417-1424 |
| `compile.py:1581` | `compile_bank`, take_action branch | builds both at 1550-1558 |

(The previous version of this file said "both call sites". There are three; all
three are correct today. No `_type_literal` caller exists outside `compile.py`.)

**Re-verified today on the current bank:** re-typing every stored action value in
`outputs/compile/labels/train.jsonl` with today's `_type_literal` and comparing
to what is stored gives **0 of 19,048 take_action `slot_values` entries in
disagreement** (script:
`scratchpad/d34_check.py`). The pre-fix figure was 4,699 of 21,335 = 22.0%; the
entry count fell because the junk entries are gone. The 19,048 figure reproduces
exactly on the current bank, so it is not stale.

---

## D-4. `required_slots` counted only typed occurrences in its modal vote. **FIXED.**

`None` is now included in the vote (`compile.py:1357-1365`) and nothing is
emitted when `None` is modal (`compile.py:1376-1379`).

### Correction to the record that still stands

The original "0% support" measurement was taken from `labels/train.jsonl` —
**the output of D-3**. A typer with no evidence cannot produce `membership_level`
or `shipping_status` at all, so measuring D-4 against a file corrupted by D-3
reported 0.0% support for slots that are in fact the best supported in the bank.
Circular measurement. *Do not measure one defect with an artifact produced by
another.*

### Support, recomputed today on bank `c270df0e249444bb`

Per-position modal typing over the train take_action turns in
`outputs/compile/labels/train.jsonl` (denominator = that button's train turns):

| button | pos | slot | support today | previously printed here |
|---|---|---|---|---|
| membership | 0 | membership_level | **1,100/1,115 = 98.7%** | 1142/1157 = 98.7% (pre-fix bank) |
| shipping-status | 0 | shipping_status | **643/688 = 93.5%** | 697/746 = 93.4% |
| verify-identity | 0 | customer_name | **2,347/2,657 = 88.3%** | 2606/2924 = 89.1% |
| make-purchase | 0 | name | **276/365 = 75.6%** | 276/365 = 75.6% |
| record-reason | 0 | — | modal is untypable, **742/1,196 = 62.0%** → correctly DROPPED | modal `reason_slotval`, dropped |

The rates are stable; the counts moved because the bank was recompiled. The old
counts (1157, 746, 2924 turns) belong to a bank that no longer exists.

### State after the fix: 7 of 30 buttons keep `required_slots`

Read from `outputs/compile/bank/actions.jsonl`, support recomputed per position:

    verify-identity   [customer_name, account_id, order_id]   88.3% / 56.2% / 72.3%
    validate-purchase [email, order_id]                       92.8% / 95.9%
    pull-up-account   [customer_name]                         95.7%
    offer-refund      [amount]                                94.5%
    membership        [membership_level]                      98.7%
    shipping-status   [shipping_status]                       93.5%
    make-purchase     [name]                                  75.6%

**Minimum support is 56.2%** (verify-identity position 1, `account_id`,
1,492 of 2,657 turns), not the 58% previously printed.

DROPPED as unsupported: record-reason, enter-details, update-order,
update-account, notify-team — all five have an untypable modal at position 0
today (62.0%, 79.3%, 86.4%, 56.5%, 94.3% of their own turns respectively).

### Minor naming mismatch — still unresolved

The raw-ontology check found the modal ontology CATEGORY differs from the
registry SLOT name in two cases: shipping-status pos 0 is `shipping_option`, not
`shipping_status`; make-purchase pos 0 is `product`, not `name`. Support is high
under the registry names either way (93.5% / 75.6%), so no slot should be
stripped, but the mapping is still undocumented.

---

## D-5. Decimal-split markers render money wrong. **OPEN — unchanged in code, all four templates still in the bank.**

`_boundary_ok` (`compile.py:616-620`) still tests only `isalnum()` on the
characters either side, so a marker that replaced the cents of a price reads as
clean. Verified present in bank `c270df0e249444bb` today, with their train
counts:

    T001670 (3x)  "$54.00 for 10 pounds or less, and $69.{amount} for everything else."
    T002380 (2x)  "$54.{amount} for 10 pounds or less"
    T002385 (2x)  "10 pounds or less is $54.99 and ${amount}.99 for all other items."
    T003953 (2x)  "so for gift wrapping, it is a fixed price of $4.{amount}."

Nine train occurrences in total. The filler renders `$4.164`. **All four pass the
machine check** — `_delex_defects` flags only the three `1amount` templates of
D-2 — so they are inside the 99.93% "clean" figure. Low count, highest customer
visibility of any defect here. Treat a `.` after digits as a boundary violation.

---

## D-6. Check artifacts under-sampled. **FIXED IN CODE 2026-09-19; both artifacts still pre-fix.**

`rng.choices(..., k=n)` sampled WITH replacement and then deduped by index. Both
writers now draw count-weighted **without replacement**:
`compile.py:1955-1967` (`write_delex_check`) and `compile.py:2078-2086`
(`write_act_check`).

**The shipped artifacts were never regenerated**, so they still show the
short samples — and the true shortfall is not what this file previously said:

| artifact | configured | actually drawn | where it shows (line numbers as of 2026-09-20) |
|---|---:|---:|---|
| `outputs/compile/delex_check.md` | 200 (`compile.delex_check_sample`) | **148** | lines 32, 47, 61 |
| `outputs/compile/act_check.md` | 300 (`compile.act_check_sample`) | **198** | lines 69, 87 |

(The previous "141 of 200 / 187 of 300" figures belong to an older bank.) Both
files carry an EMPTY manual tally, so no manual percentage has ever been
computed off either denominator.

`act_check.md` is stale in one further way: `compile.py:2143` now labels the
column **"share of banked sentence occurrences"** and explains the denominator,
while the shipped file still prints **"share of train sentences"** over the same
numbers (line 73). They are not shares of train: OTHER is 1,214 of the **58,263
banked occurrences** = 2.1% (what the file prints), but only 1,214 of 89,623
train sentences = 1.4%.

**Documented but not cleared (2026-09-20):** both files now carry a hand-written
STALE ARTIFACT banner naming the generator, the re-run command and exactly which
figures move. The banner on `delex_check.md` reports that re-executing the new
draw on the bank on disk gives **200 sampled / 199 with zero machine-visible
defects**, and instructs reviewers not to fill in the manual tally until the
sheet is regenerated at n=200. That is documentation of the defect, not a fix.

**To clear D-6:** regenerate from the current code — `cd reflex-abcd &&
PYTHONPATH=src python -m reflex compile` (CPU only, no GPU, no paid API; it
rewrites the bank, `compile_summary.md` and both check sheets, and deletes the
banners). The sample sizes should then read 200 and 300 and the act column
should relabel itself.

---

## D-7. Single-linkage merging. **HALF FIXED — the merge guard shipped and is measured; strict coverage is still unmeasured.**

**What shipped (DECISIONS D16):** `configs/default.yaml:68`
`merge_block_on_field_mismatch: true` blocks a merge whenever two forms request
different named fields, with the surface-phrase→field map at `default.yaml:69-89`.
Field-set equality is transitive, so the clusters come out field-homogeneous with
no post-hoc splitting. **The defect's own worked example is gone from the bank:**
`"i will need your username, email, and order id please."` now sits under the
canonical `"and your username, email, and order id?"` (count 266, 152 surface
forms), *not* under `"can i have your account id and order id?"` (count 534, 248
surface forms) — checked directly in `templates.jsonl`.

**Merging as it stands today** (recomputed from `templates.jsonl`): **3,436 of
the 4,489 templates (76.5%) carry more than one surface form**, the largest
carries **327** distinct phrasings, and `coverage_gap.json` records 16,170
distinct surface forms across the bank. The previously printed "3,374 of 4,417,
largest cluster 988" was the pre-guard bank; the 988-form cluster is the thing
the guard removed and is recorded as history at `default.yaml:58-67`.

**Measured cost/benefit of the guard** (config-recorded in the `report:` block,
`default.yaml:495-499`, to be re-measured on any recompile):
`bank_wrong_field_rate_prefix 0.351` of field-requesting merged forms and
`bank_wrong_field_turn_rate_prefix 0.0492` of all retrieve turns, both → **0.0**
post-fix; the price is bank fidelity (see "Numbers to carry into the report").
*Report both halves of that trade, never one.*

**Still open:** `default.yaml:500` `strict_template_coverage: null` — the share of
covered utterances whose gold sentence is within `dedup_threshold` of the
canonical it was merged into. `report.py:766` reads the key and renders
UNMEASURED. Measure it; never estimate it.

**Inconsistency to resolve while you are in there (not in a file I own):**
`default.yaml:62` says **32.1%** of field-requesting merged forms disagreed with
their canonical, `default.yaml:259` says **35.1%**, and the report key says
`0.351` with an occurrence-weighted `0.332`. Three numbers, one quantity.

---

## D-8. Product knowledge falls into OTHER. **OPEN — unchanged.**

`configs/default.yaml:143` still sets `act_embed_min_similarity: 0.25`, and the
INFORM/INSTRUCT seeds still do not span ABCD's product-detail mode, so real
INFORM sentences land in OTHER. Recomputed on bank `c270df0e249444bb`:

- `"most of our boots are waterproof."` — act OTHER, **20** train occurrences.
- `"the standard collar size is 15 inches."` — act OTHER, **12**.
- **306 OTHER templates / 1,214 banked occurrences** = 2.1% of the 58,263 banked
  sentence occurrences (previously printed as 308 / 1,198).
- The single-act OTHER skeleton **S0009 covers 1,849 of the 71,133 train
  retrieve_utterance turns = 2.6%** (previously 1,836).

Changing act seeds moves every downstream number, so this still needs a measured
before/after, not a guess.

---

## D-9. `log_every: 50` exceeds `smoke_max_steps: 20` — the smoke run is silent by construction. **OPEN.**

`configs/default.yaml:200` `log_every: 50`, `configs/default.yaml:224`
`smoke_max_steps: 20`. `train.py:944` reads `log_every` unmodified and
`train.py:1063` gates the only training log line on
`(step + 1) % log_every == 0`, with no smoke clamp. A `--smoke` run still emits
ZERO training lines between "model loaded" and exit.

Cost when it happened: a working run was investigated three times as a suspected
hang before the config explained it. Under swap pressure (the box was at
50.8/51.2 GB) "silent" and "thrashing to death" are indistinguishable from
outside, which is exactly when the log line matters.

**Fix:** under smoke, `log_every = max(1, min(log_every, smoke_max_steps // 4))`,
or log step 1 and the final step unconditionally. **Also still true:** `grep -n
"heartbeat\|time.monotonic" src/reflex/train.py` returns nothing — there is no
time-based heartbeat anywhere in the training loop.

---

## D-10. `--smoke` subsets AFTER the full corpus parse. **OPEN.**

`train.py:960` calls `build_partitions(cfg)` — which parses the entire 116 MB
`abcd_v1.1.json` — and only then does `train.py:965` apply
`train.smoke_conversations` via `_subset(train_partition, …)`. No corpus-level
`data.max_conversations` exists (`grep max_conversations src/reflex/data.py
configs/default.yaml` finds only `train.dev_max_conversations`).

**Measured consequence.** On a box under memory pressure (~2.9 GB free), a smoke
run configured for 40 conversations and 20 optimizer steps burned 21 minutes of
wall time for 2 minutes of CPU (~9% effective), never reached step 1 even at
`log_every: 1`, sat at 13 MB RSS / 1.4% CPU, and produced no checkpoint on two
attempts. **Proof it is the parse:** pointing `data.abcd_dir` at a 7.7 MB corpus
(400 train / 120 dev / 120 test, all 55 intents preserved) took
`reflex compile` from 11+ minutes to under 60 seconds on the same machine.

**Fix:** subset the conversation list inside the loader, before per-turn
normalization and context building, or honour a corpus-level
`data.max_conversations` in `build_partitions`.

**Workaround in use:** a pre-built small corpus at `scratchpad/abcd_small/`,
round-robin across intents so all 55 survive, via `--set data.abcd_dir=…`.
INTEGRATION testing only — any number it produces comes from 400 conversations
and is not a result.

---
---

# The 2026-09-19/20 audit remainder — findings that were NOT fixed

Two fix waves closed 65 audited findings. What follows is what they did not
close, from `scratchpad/completers_report.md` (items left `not_done`,
`half_done`, or deliberately untouched) and the two post-fix regression reviews.
Nothing here changes a published leaderboard number unless it says so.

## A-0. The 2026-09-21 methodological-consistency cycle — five defects **FIXED**

All five were found by asking one question of the published results: *is every arm
measured the same way?* Each is closed in code with the evidence below, and four
artifacts were re-run locally (CPU, no paid API) so the files on disk now match the
code. The 561-test suite passes unchanged.

### A-0.1. Teacher forcing on test gold labels — **FIXED, and it killed the n-gram arm**

`ngram_skeleton_plus_h7.py:222`, `ngram_skeleton_baseline.py:226`,
`committee_gate_h7.py:150` and `committee_gate.py:130` all did
`history.append(row.gold_skeleton_id)`: the model predicting turn *n* had been
handed the TRUE act sequence of turns 1…*n*−1. Those are labels, not observable
input, and every arm being compared against receives text alone. For the gate it was
worse than unfair — it was **unimplementable**, since a live gate has no gold
history.

**Fix:** a free/gold switch defaulting to `free` at all five sites. Three carry a
real CLI flag — `ngram_skeleton_baseline.py:122` (`--history-mode {free,gold}`,
append guarded at 260-262), `ngram_skeleton_plus_h7.py:91` (229-231) and
`build_committee_judge_sample.py:55`. The two gate scripts carry a module constant
instead — `committee_gate.py:53` `HISTORY_MODE_DEFAULT = "free"`, read at 140, gold
append guarded at 152-153; `committee_gate_h7.py:49`, read at 160, guarded at
172-173. `gold` survives as a labelled diagnostic.

**One gap left by the fix, small but worth closing:** only the two n-gram artifacts
stamp the mode (`summary.history_mode: "free"` plus a `history_mode_note`).
`committee_gate.json` and `committee_gate_h7.json` record no `history_mode` key at
all, so their provenance for this defect is the module constant, not the file. Add
the stamp on the next re-run.

**Verified on the re-run artifacts:** free-running, both n-gram scripts predict the
modal skeleton `S0000` on **all 8,889** test_seen rows (counted from their own
`rows`), so conditional skeleton@1 is `0.3139272271016311` — bit-identical to the
label-blind constant — and conditional compose@1 with modal templates is
`0.06373902132998745`, bit-identical to `select.json`'s
`headline_test_seen.compose.conditional.constant`. **The arm has zero skill.**
Moved: n-gram+H7 compose@1 20.15% → **14.81%**, n-gram+modal 10.51% → **6.37%**,
n-gram skeleton@1 41.96% → **31.39%** (all n=3,985); gate unanimous bucket coverage
1,869/3,985 = 46.9% → **1,441/3,985 = 36.2%**, its compose@1 0.34189 → **0.31575**.

### A-0.2. The judge sampler averaged over failure modes, not turns — **FIXED**

`build_llm_judge_sample.py` defaulted to `--design stratified`: equal allocation
over 47 gold-act-sequence strata (1-4 rows each; `('ASK',)` is 25.1% of the miss
pool and 3% of the sample), with no `stratum_weight` written. Every other builder is
a plain self-weighting `rng.sample` — `build_qwen_judge_sample.py:82`,
`build_rnn_judge_sample.py:32`, `build_qwen_structured_judge_sample.py:98`,
`build_langcache_judge_sample.py:66`, `build_smollm2_judge_sample.py:90`,
`build_committee_judge_sample.py:214` — so the cache's published share was the only
one averaging over failure MODES while the rest averaged over TURNS.

**Fix:** default flipped to `simple_random` (`build_llm_judge_sample.py:146`,
branch at 278-280), and `stratum_weight` + `stratum_population_share` are now
written whenever the stratified design is chosen (311-316) so
`run_llm_judge._weighted_shares` (`run_llm_judge.py:103-131`) can fire.
**Moved:** ungated cache accuracy 71.5% → **77.2%**, errors 18.1% → **15.2%**.
The 100-row artifact on disk predates the fix — see E-1.

### A-0.3. The gate's headline benefit was roughly half an estimator artifact — **FIXED (as an analysis defect)**

"Gating moves the cache from 71.5% to 89.7% accuracy and 18.1% to 6.2% errors"
compared an unweighted mean of the equal-allocation sample (ungated row) against a
**32-row slice of that same sample** (gated row) and attributed the difference
between the two *estimators* to gating. Re-derived with one estimator —
post-stratification to each configuration's own miss pool, stratified bootstrap
(4,000 draws, seed 20260921) — gating buys **+0.7 accuracy points, not +18.2**, and
**-3.8 errors**, for -63.8 points of coverage: ungated 77.2% [72.4-81.6] /
15.2%, gate-all-3 86.3% [78.3-91.9] / 11.4%, gate-≤1 83.0% [77.8-87.3] / 11.8%.
*(An interim pass on 2026-09-21 post-stratified each row to its bucket's act-sequence mix and reported +0.7. That estimator assumes the appropriate-rate within an act stratum is identical inside and outside the bucket, which is exactly what a gate violates; it washed the effect out. The within-bucket figures above supersede it.)*
tchpad/plan/final_table.json`.

### A-0.4. The deterministic tie-break was missing at the three committee sites — **FIXED**

`_argmax` had landed on the two standalone n-gram baselines but not on
`committee_gate.py`, `committee_gate_h7.py` or `build_committee_judge_sample.py`,
which were still on `Counter.most_common(1)` — insertion-order tie-breaking, so the
same model was two different models depending on which script ran it.
**698 of 8,889 predictions differ** between the rules
(`scratchpad/G02_verify/check_ngram.log:13`).

**Fix:** `_argmax` is now defined and used at all five sites —
`committee_gate.py:41` (used 132-133), `committee_gate_h7.py:37` (152-153),
`build_committee_judge_sample.py:35` (154-155), `ngram_skeleton_baseline.py:105`
(178-180, 193), `ngram_skeleton_plus_h7.py:74` (159-160). The only surviving
`most_common` calls in `sft/eval/` build RNN vocabularies
(`rnn_skeleton_dryrun.py:175`, `rnn_h7_template.py:151`), not predictions.

### A-0.5. qwen3-0.6B unconstrained's judged sample was contaminated — **FIXED in the published figures**

Its builder used a raw string-equality miss predicate, so **15 of its 100 judged
"misses" are compose@1 HITS** under the canonical bank-lookup scorer — and all three
judges called all 15 "appropriate", the maximum possible inflation. Recounted
2026-09-21 on the clean rows (`qwen_judge_multi/*.jsonl` joined to
`gen_unconstrained_scored.json`): appropriate flash 0.8100 → **0.7765** (85 rows),
glm-5p3 0.7917 → **0.7531** (81), gpt-oss 0.7200 → **0.6706** (85); judged accuracy
78.1% → 74.2% → **74.9%** [66.8-82.4] under the consistent estimator. Every other
arm's judged sample was audited the same way and is clean. **The artifact itself is
unchanged** — it is the published figures that now exclude the 15 rows (see E-1).

### Artifacts re-run on 2026-09-21 (CPU, no paid API)

`ngram_skeleton_baseline.json`, `ngram_skeleton_plus_h7.json` (both now
`history_mode: free`), `committee_gate_h7.json` (compose winner, free-running
committee, `_argmax`) and `committee_gate.json` (now fitted on
`selection.winners.h5` per `committee_gate.py:60`, with a new `vs_select_json`
self-check reporting `hit_delta_rows: 5` against `select.json`). **Not re-run:**
`recall_at_k.json` — so its per-bucket `blocks` are the OLD gate's (unanimous
n=1,869 at 0.34189, against the current n_cond=1,441 at 0.31575); its
`blocks["ALL (conditional)"]` rows are unaffected. Also not re-run: `select.json`,
`retrieval_baseline.json`, the three RNN artifacts, `rnn_h7_template.json`, the
LangCache/Redis files.

---

## A. Blocking / policy — decide these before re-running anything

### A-1. The certification ranking is settled in CODE; which rule to certify under is still the user's call. **OPEN DECISION (no longer blocking).**

**Half of this is now closed.** The ranking key and the certification predicate
disagreed — `run_response_probe.py` ranked by `(cells_in_band, headline_in_band)`
while certifying on `best["headline_in_band"]`, so a re-run would have written
`certified:false` with a different vectorizer. They agree again: the ranking key at
**`run_response_probe.py:557`** is `(headline_in_band, cells_in_band)` and the
predicate at **line 572** is
`bool(best and best["headline_in_band"] and identity_ok and cells_ok)`, with a
comment at 544-556 stating in plain words that this is the rule that produced every
certified artifact on disk **and that it is not a good rule**. The gap it hides is
now reported rather than invisible (`max_cells_in_band` / `max_cells_vectorizer`,
lines 583-585) and `probe.min_cells_in_band` exists as an optional floor that
`probes/probe.yaml` does not set.

**What is still open is the decision**, because the two rules pick different
vectorizers and neither dominates. Replaying the six rows of the committed
`outputs/probes/response/validate.json` (recounted 2026-09-21 straight off that
file, `all[*].cells_in_band` and `all[*].headline_in_band`):

| rule | winner | cells in band | headline in band | certified |
|---|---|---:|---|---|
| old (headline first) | `{min_df:1, sublinear_tf:true, C:1.0}` | 1 of 7 | yes (0.834976 in D7's 0.8345–0.8351) | **true** — matches the committed `certification.json` |
| new (cells first) | `{min_df:2, sublinear_tf:false, C:1.0}` | 4 of 7 | no (0.83303) | **false** |

No candidate has both the most cells in band and the headline in band. All six,
recomputed today (cells in band / headline in band / headline accuracy):
`min_df1 sub=F C1` 3/7 no 0.83371; `min_df2 sub=F C1` 4/7 no 0.83303;
`min_df1 sub=T C1` 1/7 **yes** 0.83498; `min_df1 sub=F C4` 2/7 no 0.83296;
`min_df2 sub=T C1` 2/7 no 0.83408; `min_df1 sub=T C4` 1/7 no 0.83423.

**A second discrepancy in the same file, not previously recorded:**
`validate.json` says `n_dev_rows = 13392` while its own
`d7_recorded_row_counts.dev = 13284` — the reproduction was scored on 108 more
dev rows than the D7 curve it is being compared against. The band is 0.0006
wide. Settle this in the same pass.

**Nothing refuses, and a re-run would no longer flip the stamp.** `mode_measure`
and `mode_select` read the on-disk `certification.json`, which says
`certified: true` (`{min_df:1, sublinear_tf:true, C:1.0}`, stamped
2026-09-16T23:17:58); `outputs/probes/response/select.json` carries the same
`certified: true` at its top level. Because the ranking key was restored, re-running
`validate` today would re-select the same vectorizer and re-write the same stamp.

**Effect on results: none.** Certification gates the harness (does the renderer
reproduce D7's nextstep curve); the heads use `probe.vectorizer_by_head` plus the
sweep overrides, never `validate`'s `best`. **What is left for the user to decide**
is whether `certified: true` should be granted at all under a rule that ranks on a
band 0.0006 wide when that cell's own clustered CI half-width is ~0.007, and whose
winner reproduces **1 of D7's 7 recorded cells** where `{min_df:2,
sublinear_tf:false}` reproduces **4 of 7**. Switching the rule re-certifies a
different vectorizer and moves every headline, so it is a decision, not a fix.
Until it is taken, every doc must say `certified:true` was granted on the
headline-cell rule and that the certified config reproduces **1 of 7** D7 cells.

### A-2. The published probe headline is no longer bit-reproducible from the working tree.

*Measured by the probe/leaderboard completion agents on 2026-09-20 with the
probe's own `fit_score` (scratch only — nothing under `outputs/` was
regenerated). **Not re-run here**; a probe re-run is the confirmation.*

Re-running the probe's own `fit_score` with `select.json`'s own winners on the
same 71,133 fit / 8,858 eval rows gives **H5 recall@1 3,829/8,858 = 0.43226** vs
the published **3,824/8,858 = 0.43170**, and **compose@1 conditional
1,105/3,985 = 0.27729** vs the published **1,102/3,985 = 0.27654** (13 rows
flipped). `bank_hash` and the featurizer fingerprint are identical and
`fully_covered` flipped on 0 rows, so the golds are unchanged and only the
*inputs* moved — the `src/reflex/data.py` multi-valued-slot leakage fix is the
named lead, unproven as sole cause. It is deterministic, not noise.

**Magnitude: +0.06 / +0.08 points, far inside the certified CI half-width of
1.54 points, so no ranking and no conclusion moves.** But no document may say
"nothing changed": the correct wording is *conclusions unchanged; headline digits
move by <0.1 point after the audit fixes and must be re-certified by one probe
re-run once all fixes land.*

## B. Artifacts that no longer match the code that produces them

| artifact | what is stale | what it takes to clear |
|---|---|---|
| `outputs/compile/delex_check.md`, `act_check.md` | pre-D-6 sampler (148/200, 198/300) and the old act column label; **both now carry a hand-written stale banner** naming what moves, which is documentation, not a fix | regenerate, CPU only — see D-6 |
| `outputs/probes/response/committee_gate.json` | **CLEARED 2026-09-21 — re-run.** It now fits `selection.winners.h5` (`committee_gate.py:60`) and records `primary_skeleton_accuracy_overall` **0.4322646195529465** (3,829/8,858) with its own `vs_select_json` self-check: `select_json_h5_recall@1` 0.431700158049221, `hit_delta_rows` **5**, `matches: false` — 5 rows of 8,858, which is the A-2 input drift, not a different classifier. New buckets: n_dis=0 **2,474** (27.9%, primary accuracy 0.5469), n_dis=1 **3,832** (43.3%, 0.4556), n_dis=2 **2,552** (28.8%, 0.2861). | nothing — quotable, with the 5-row A-2 caveat |
| `outputs/probes/response/committee_gate_h7.json` | **CLEARED 2026-09-21 — re-run** with the free-running committee and `_argmax`. Now `overall.compose@1_conditional` **0.27728983688833125** (1,105/3,985), `overall.skeleton_accuracy` **0.42662** (3,779/8,858), buckets n_dis=0 **n=2,365 / n_cond=1,441 / compose@1_cond 0.3157529493407356 / share 0.26699**, n_dis=1 3,655 / 1,636 / 0.3141809290953545, n_dis=2 2,838 / 908 / 0.14977973568281938. It now carries a `primary_config` stamp saying in words that its `skeleton_accuracy` is the COMPOSE path's, **not** the certified H5 head's. | nothing for the compose and bucket columns; still do not read its `skeleton_accuracy` as the certified H5's — use `select.json` |
| `outputs/probes/response/recall_at_k.json` | **NEWLY STALE 2026-09-21.** Its `blocks` for `unanimous (n_dis=0)` (n=1,869, compose_recall@1 0.3418940609951846), `n_dis=1` and `n_dis=2` are the **OLD** gate's buckets — the re-run gate's unanimous bucket is n_cond=**1,441** at **0.31575**. `blocks["ALL (conditional)"]` is unaffected and still reproduces `select.json` exactly (skeleton 0.5821831869510665, compose 0.2765370138017566 on n=3,985). | `PYTHONPATH=src python -m sft.eval.recall_at_k` (CPU; it defaults to `--gate outputs/probes/response/committee_gate_h7.json`, which is now the re-run file) |
| `outputs/probes/response/cache_paraphrase_relaxed.json` | genuinely corrected in place (unconditional compose@1 0.28766 → **0.12397345033187085 = 1,102/8,889**, byte-identical to `select.json`'s unconditional headline; relaxed 0.12408594892563843 = 1,103/8,889, exactly one turn rescued) but it carries **no patch note**, unlike the five skeleton JSONs, and its mtime predates the last edit to its own script | re-run `PYTHONPATH=src python -m sft.eval.cache_paraphrase_relaxed`, or add a `compare_against_patched`-style note |
| the five skeleton JSONs (`ngram_skeleton_baseline`, `ngram_skeleton_plus_h7`, `rnn_skeleton_dryrun`, `rnn_skeleton_full`, `rnn_skeleton_plus_h7`) | **the two n-gram files were re-run 2026-09-21** (`history_mode: free`, `_argmax`) and are current. The three RNN files were NOT re-run — only their `compare_against` blocks were patched, and those blocks still quote the **teacher-forced** n-gram (`rnn_skeleton_plus_h7.json` → `compare_against.ngram_skeleton_plus_h7` says 0.2015056461731493 / 0.09033637079536506; the file on disk now says **0.1480552070263488 / 0.06637417032287096**). Their own measured numbers are unaffected by the teacher-forcing defect — the RNN predicts from text, never from gold history. | re-run the three RNN arms (GRU training, not started) or patch their `compare_against` n-gram cells; meanwhile do not read an n-gram figure out of an RNN artifact |
| `outputs/probes/response/rnn_h7_template.json` | produced by an unseeded run — irreproducible; no `skeleton_population` or `seed` key | seeded re-run = GRU training, not started |
| `outputs/probes/response/select.json` | carries `certified: true` from the old rule (A-1); carries the **old key `vectorizer_by_head`** — current code emits `vectorizer_base_by_head` (probe.yaml before the sweep, produced no number) and `vectorizer_as_run_by_head`; lacks the newer denominator fields | regenerate after A-1; meanwhile point readers at `headline_test_seen.<head>.selected_config.vectorizer`, which exists and is correct. DECISIONS D25 (the `vectorizer_as_run_by_head` paragraph) still names the retired key |
| `outputs/probes/response/{qwen3_4b_score_500,qwen3_4b_score_1000,smollm2_8h_score_500,smollm2_continued_score_500}.json` | all four are **contiguous head blocks** of the prompt file (56 conversations for 500 rows, 108 for 1000) out of test_seen's 932 — not random samples — and `generate.py:182` has since flipped `--limit-mode` to default `convo` (random whole-conversation sampling under `--seed`), so the recorded commands no longer reproduce them, and none has the new `<out>.sample.json` sidecar | add `--limit-mode head` to the recorded provenance, or write the four sidecars retroactively; caveat the rows wherever they are published |
| `sft/eval/data/qwen_judge_sample.jsonl` + `qwen_judge_multi/*` | 15 of the 100 judged "misses" are compose@1 hits under the canonical scorer, and all three judges called all 15 "appropriate" | **the published figures now exclude them** (DECISIONS D33/D35: appropriate 0.7765 / 0.7531 / 0.6706 on 85/81/85 clean rows; judged accuracy 74.9%). The artifact itself still carries the 15 rows — rebuilding and re-judging is paid |
| `sft/eval/data/committee_judge_sample.jsonl`, `cache_matched_judge_sample.jsonl` (+ results) | the judge was shown **r-tagged featurizer renderings** — **47 of 100** and **163 of 200** contexts carry `agent\|r3\|good r3\|…` tags (recounted 2026-09-21; every other judged sample on disk is 0-tagged). All seven builders now write raw context, so this survives only in these two artifacts. `cache_matched_judge_sample.jsonl` additionally **has no producing script in the repo** (E-2). The committee sample also no longer reproduces (29 of 100 turn_ids in common) | quote with the caveat — the paired evidence in E-1 bounds the damage — or re-judge ~300 clean rows (paid, `PLAN.md` D1, not run) |
| `sft/eval/data/langcache_judge_sample.jsonl` (+ `langcache_judge_multi/*`) | **matched-only** population; the 347 abstentions of 778 misses were never eligible for judging, and the rows carry no population field | label it `matched_only` wherever quoted |
| all `sft/eval/data/*_judge_multi/summary.json` | denominators are scored rows only — no `*_of_all` shares, no `n_unscored`; agreement is over scored rows | apply the F23 fix to `run_llm_judge_multi.py` |
| `sft/eval/checkpoints/smollm2_360m_*/train_history.json` | none records smoke/row_offset/seed/dev_split; the 8h run's `best_dev_loss` was measured on a dev set of which **502 of 526 rows (95.4%)** had already been trained on | do not quote these dev losses as held-out; only retraining from `--base-model` fixes it. compose@1 for the SmolLM2 arms is unaffected — scored on test_seen |
| `outputs/probes/response/redis_vector_*.json`, `langcache_baseline*.json` | hardcoded LangCache figure, KNN 1 with the server's unstated tie rule, contiguous 5,000-turn pool, flush unverified | paid re-runs — not done. Quote LangCache as **unstable in [2.75%, 8.25%] across identical re-runs** (the stable non-duplicate bucket is 19/302 = 6.3% compose@1, 90/302 = 29.8% skeleton) |

## C. Code defects left open

- **C-1. `configs/default.yaml` is missing the three keys `report.py` now reads.**
  `report.py:776-780` renders HELD-OUT bank-reconstruction rows from
  `report.bank_exact_reconstruction_rate_dev` / `_test_seen` / `_test_novel`;
  the config has only `bank_exact_reconstruction_rate: 0.6067` (line 475), so a
  rendered report prints "NO INPUT" on all three rows and shows only the
  in-sample TRAIN figure. Values, recomputed from
  `outputs/probes/analysis/coverage_gap.json` (`splits.<split>.variants.A.turn_coverage`):
  dev **0.4522** (4,079/9,021), test_seen **0.4483** (3,985/8,889), test_novel
  **0.4427** (290/655).
- **C-2. The "never read a persisted label file" invariant is enforced in two of
  four places.** `calibrate._dev_labels` (`calibrate.py:1032-1034`) and
  `run._run_arm_b` (`run.py:718-721`) now always derive.
  `probes/run_response_probe.py:153-158` still calls `load_turn_labels(split, cfg)`
  first and only derives on `FileNotFoundError`, and `train.py:972` WRITES
  `outputs/compile/labels/dev.jsonl` on a full-dev training run. Today
  `outputs/compile/labels/` holds only `train.jsonl` (checked), so no published
  number used a stale file — but the first full `reflex train` creates
  `dev.jsonl`, after which the probe silently stops deriving its dev golds. The
  durable fix is a `bank_hash`/`dataset_hash` header written by
  `compile.write_turn_labels` and checked by `compile.load_turn_labels`.
- **C-3. Per-seed calibration is complete for the quantile file, not its
  diagnostics sidecar.** `calibrate._write_diagnostics` (1303-1308) formats a
  `{seed}` placeholder, but `configs/default.yaml:430` ships an un-templated
  `calibration_diagnostics_path`, and the remedy message at `calibrate.py:1368-1374`
  names only `paths.calibration_path`. Following it verbatim gives correct
  per-seed `gate_seed{N}.json` files and one diagnostics file the last seed
  silently overwrites (no overwrite guard on the sidecar).
- **C-4. The act-audit interval is labelled a 95% CI and is not one.**
  `outputs/act_audit/score_act_audit.py:80` combines four independent per-stratum
  95% Wilson intervals as `[Σ W_s·lo_s, Σ W_s·hi_s]`; joint coverage is ~0.95⁴, so
  it is a conservative envelope. (The Wald version it replaced was genuinely
  broken — zero width at 30/30 — so this is an improvement, just mislabelled.)
  Zero risk to any number: `outputs/act_audit/act_audit_sample.md` still has 104
  empty verdicts, the script exits at the "no verdict" guard, and **no `c` value
  exists — none should be quoted anywhere.**
- **C-5. `sft/eval/train_smollm2_local.py:338`** interpolates `args.resume_from`
  unconditionally in the "no epoch improved" warning, so a fresh run with a NaN
  dev loss prints "carried forward from None" and tells the operator a
  checkpoint is still best when none exists.
- **C-6. F17's third site is unfixed:** `sft/eval/recall_at_k.py:57-60` still
  takes `winners.compose` for H5, and its `skeleton_recall@1` (0.58218 on the
  n=3,985 conditional population) is being used as the comparable cache skeleton
  rate against the RNN/n-gram arms. Same decision as F17b: keep the compose
  winner and relabel + stamp the config, or switch to `winners.h5`.
- **C-7. The `"stale"` marker beside the 1-NN TF-IDF constant is still missing**
  in `sft/eval/ngram_skeleton_baseline.py:297` (line number as of 2026-09-21, after
  the `--history-mode` fix moved it) and `sft/eval/retrieval_rerank.py:206`; both
  hard-code `0.05244667503136763`, which the current code no longer produces
  (0.04918, and 0.06223 on a matched candidate pool — see E-4). It was added in the
  two LangCache/Redis files.
- **C-8. No teardown path for the billed Redis Cloud data** (audit #81) — data
  stored by `redis_vector_baseline.py` is never removed by the script.
- **C-9. `models.value_target_index_value_aware` has zero call sites** —
  every reported H4 number still comes from `value_target_index` via
  `probes/response_labels.py` (`H4TargetIndexer.index`). The docstrings now say
  "NOT WIRED IN"; wiring it would change published numbers.
- **C-10. `configs/default.yaml:478-479` names the wrong bank for its own
  numbers.** The comment on `bank_exact_reconstruction_rate: 0.6067` says it
  belongs to "the bank on disk … `bank_hash 654d796ecc51cea1`, 4,505 templates".
  The bank on disk is **`c270df0e249444bb`, 4,489 templates**
  (`outputs/compile/bank/meta.json`), which is what the config's own
  `bank_templates: 4489` (line 472) says. The *values* 0.6067 / 0.6501 do match
  `compile_summary.md` for the on-disk bank; only the hash and the count in the
  comment are stale — and this comment is the one that lectures the reader about
  printing a number beside the wrong bank hash.

## D. Measurement gaps — unmeasured, not broken

- `report.strict_template_coverage` is `null` (D-7).
- The act-audit sheet is unfilled (104 empty verdicts), so the act-labeler
  accuracy `c` does not exist (C-4).
- `tests/README.md:36-52` records two spec-10 requirements as NOT MET (delex
  round-trip on 20 fixtures; skeleton extraction on 10 fixtures) and that
  `test_determinism.py` does not exist. Add to that list: **no test covers the
  D-1 PII guard** (D-1). (`tests/README.md:3` also says "17 test modules";
  `ls tests/` shows 16 — its own per-module table sums correctly to 561.)

## E. Opened, or left open, by the 2026-09-21 cycle — all OPEN

These are real and unresolved. Each was verified against a file on disk on
2026-09-21; none is a re-import of an older claim.

### E-1. No judged sample on disk is both self-weighting AND clean-context. **OPEN — documented caveat, not a blocker.**

Every cache accuracy figure in `Chat_Leaderboard.md` rests on
`cache_matched_judge_sample.jsonl` / `_results.jsonl` (n=200, self-weighting), and
**163 of its 200 rows** show the judge the featurizer's recency-tagged rendering
(`agent|r3|good r3|afternoon, …`) instead of clean prose; `committee_judge_sample.jsonl`
is the same defect on **47 of 100** rows. All seven builders now write raw context
(`build_llm_judge_sample.py:255` writes `h5row.context.text`;
`build_committee_judge_sample.py:126,208` writes `raw_context_by_turn`), and every other
judged sample on disk is 0-tagged — `llm_judge_sample_corrected`, `qwen_judge_sample`,
`rnn_judge_sample`, `qwen_structured_judge_sample`, `langcache_judge_sample`,
`qwen3_4b_1000_judge_sample`, `smollm2_8h_judge_sample`, all 0 of 100/200. These two
artifacts simply predate the fix.

**Bounded, not ignored.** Ten turns were judged by `gpt-oss-120b` in BOTH the tagged
n=200 sample and the clean n=100 corrected sample. **8 of 10 verdicts are
identical**; **all 5 "appropriate" verdicts held**; the 2 flips are both *inside*
the non-appropriate band (`wrong`↔`borderline`, `test_seen:3550:2` and
`test_seen:4397:4`) and in opposite directions. Since only `p_appropriate` feeds the
leaderboard's accuracy column, the tagged context does not appear to move the number
that matters. *(Note: `Chat_Leaderboard.md`'s limitation 2 says "all 4 appropriate
verdicts held"; the recount is **5** — 2 on untagged rows, 3 on tagged ones. The
direction of the finding is unchanged.)*

**To close:** rebuild with `build_llm_judge_sample.py --design simple_random` (raw
context, reproducible) and re-judge ~300 rows through `gpt-oss-120b` **serverless**
(per-token, no GPU, well under $1). Scoped as `PLAN.md` D1 — **needs user approval,
not run**.

### E-2. `cache_matched_judge_sample.jsonl` has NO producing script in the repo. **OPEN.**

`grep -rn "cache_matched" --include="*.py" .` returns nothing; the string appears
only in `Chat_Leaderboard.md`, `PLAN.md` and this file. So the 200-row sample that
three leaderboard rows rest on **cannot be regenerated, re-seeded or audited against
its own builder** — there is no builder. It is the only self-weighting cache sample
that exists, which is why it is used anyway. Either write the producer (it must
reproduce the same 200 turn_ids from a recorded seed) or replace the sample under
E-1's re-judge.

### E-3. The n-gram baselines' `max_order=4` is an untuned argparse default. **OPEN, as a disclosure asymmetry.**

`ngram_skeleton_baseline.py:121` and `ngram_skeleton_plus_h7.py:90` both
`add_argument("--max-order", type=int, default=4)`, and that default is what every
published n-gram number used. The cache got a certified vectorizer sweep
(`select.json` `selection`); its baselines got one hyper-parameter, never swept.
Either sweep the baselines too or disclose the asymmetry wherever the arms are
ranked.

**Two qualifications, both verified 2026-09-21.** (a) The "order 1-2 scores better"
measurement that motivated this item was made in the **teacher-forced** mode and
there is no artifact for it on disk, so it is `pending re-run` — reproducing it
needs `--history-mode gold --max-order {1,2}`, which measures the disqualified
configuration. (b) Free-running, lowering the order cannot rescue the arm: the
re-run artifact's own `order_used_distribution` ({0: 932, 1: 932, 2: 931, 3: 930,
4: 5,164}) together with its single distinct prediction shows that the all-`S0000`
context maps to `S0000` at orders 1, 2, 3 and 4 alike, so any `max_order` collapses
to the same constant predictor.

### E-4. The 1-NN TF-IDF arm draws candidates from a different pool than the managed-cache arms. **OPEN — it changes a published number.**

`retrieval_baseline.py` indexes **all 71,133** train retrieve turns, of which 27,974
(39.3%) are not fully bank-covered — copying such a neighbour yields a template
tuple containing an empty id, which can never equal a fully-covered gold tuple.
`langcache_baseline.py:218-227` and `redis_vector_baseline.py:198-208` both build
their pool from `_fully_covered_turns` only, i.e. **43,159**. D31 compares the three
as if "only the similarity engine" differed.

**Magnitude, measured by re-running `retrieval_baseline.py`'s exact algorithm
(same TF-IDF 1-2gram, same cosine, same lowest-turn_id tie-break;
`scratchpad/mca/L2-populations/nn_rerun.py`):** pool 71,133 → conditional compose@1
**0.049184**, skeleton@1 0.225345; pool 43,159 fully-covered only → conditional
compose@1 **0.062233** (+1.30 pts, +26.5% relative), skeleton@1 0.255709. Both on
n=3,985. Note the committed `retrieval_baseline.json` still holds the *pre-tie-break*
**0.05245 / 0.24693** and has not been re-run, so three values for this arm are in
circulation; the leaderboard quotes 5.2% from the committed artifact.

### E-5. LangCache compose@1 is not reproducible. **OPEN, and no re-run will settle it.**

Identical re-runs of the same configuration (3,800 stored / 800 queried) give
compose@1 anywhere in **[2.75%, 8.25%]**. The scoring code is byte-identical between
the two commits that bracket the runs (90d7d26, 22370d7) — it was a re-run, not an
edit. Partitioning the 453 matched rows: the 302 whose nearest neighbour is not an
exact-duplicate context (similarity < 0.999999) give 19 full / 90 skeleton hits in
BOTH runs; the 151 with several byte-identical stored contexts (similarity ≈ 1.0)
gave 47/81 once and 3/5 the other time, with identical similarities on all 44
flipped rows. The service returns an arbitrary member of a tied set. **Quote the
range or the stable non-duplicate bucket (19/302 = 6.3% compose@1, 90/302 = 29.8%
skeleton), never a point value**, and note that the account's LangCache database
ran out of memory above ~4,000 entries so a full-scale run is impossible on this
tier anyway.

### E-6. `select.json` records `certified: true` under a ranking rule the corrected code would not grant on curve coverage. **OPEN — user decision.**

Restated here because it now belongs with the rest of the open list rather than with
the blockers: see **A-1** for the evidence table (1 of 7 cells vs 4 of 7) and why
switching moves every headline.

---
---

# Numbers to carry into the report — every one recomputed 2026-09-20, populations named

**Bank fidelity on TRAIN, in-sample** (`outputs/compile/compile_summary.md`,
bank `c270df0e249444bb`): **60.67% of the 71,133 train retrieve_utterance
turns** are fully reconstructible from the bank (the artifact prints the rate
only; 0.6067 × 71,133 ≈ 43,157 turns), and **65.01% of the 89,623 train
sentences**. A turn counts only if EVERY one of its sentences survived dedup and
`compile.min_template_count`.

> **This is NOT the ceiling on the reflex rate.** DECISIONS D16 retired that
> framing and `compile_summary.md` states it: every retrieve turn is assigned
> templates (assignment rate 1.0 by construction), so the fast path can always
> emit something — what this number bounds is how often it emits the utterance
> VERBATIM. It is a FIDELITY rate. It also fell from the widely quoted
> 65.30%/61.01% pair to 65.01%/60.67%. `configs/default.yaml:475-487` attributes
> that to the D16 merge guard — bought at the price of taking the wrong-field
> rate from 35.1% of field-requesting merged forms to 0.0% — and says to report
> both halves of the trade. **Caveat:** the bank also changed by D-1's username
> masking in the same interval (4,417 → 4,505 → 4,489 templates per
> `default.yaml:472`), and no artifact on disk separates the two effects. The
> config's own arithmetic is also off: it calls the utterance cost "0.41 points"
> where its own pair 0.6101 → 0.6067 is 0.34 (sentences 0.6530 → 0.6501 = 0.29).

**Bank fidelity HELD OUT** — publish this beside the train figure
(`outputs/probes/analysis/coverage_gap.json`, variant A `turn_coverage`, same
bank): dev **45.22%** (4,079 of 9,021 turns), test_seen **44.83%** (3,985 of
8,889), test_novel **44.27%** (290 of 655). These are the three values
`configs/default.yaml` is missing (C-1).

**Degradation by utterance length** — the previously published train-side
curve (1 sentence 64.3%, 2 → 53.5%, 3 → 24.7%, 4 → 11.4%, 5+ → 0%) was measured
on the pre-fix bank and **is stale; re-run `reflex compile`'s coverage pass to
refresh it.** The held-out equivalent is on disk and reproduces today
(`coverage_gap.json`, `by_n_sentences`, variant A):

| sentences in turn | test_seen (turns) | dev (turns) |
|---|---|---|
| 1 | **47.70%** (6,929) | **48.37%** (6,955) |
| 2 | **37.91%** (1,741) | **37.72%** (1,845) |
| 3 | **10.0%** (200) | **9.5%** (200) |
| 4 | 0% (15) | 0% (20) |
| 5 | 0% (4) | 0% (1) |

**The shortfall is NOT yet attributable to `min_template_count: 2`.** That claim
was asserted, never measured: `outputs/probes/analysis/min_count_elasticity.json`
has no `min_count = 1` point. What it does measure, on dev
(`turn_coverage`): 2 → **45.72%**, 3 → 44.43%, 4 → 43.49%, 5 → 42.91%,
8 → 41.51%, 16 → 39.02%, 32 → 36.03%. So raising the threshold costs ~10 points
between 2 and 32; how much *lowering* it to 1 would buy is unmeasured. Do not
lower it without measuring — a singleton template is one no human said twice.

**Constant-predictor bar for the skeleton head — three different populations,
never interchangeable:**

| population | value | source |
|---|---|---|
| TRAIN label prior (the fit split) | **25.56%** — S0000 "ASK" is right on 18,181 of 71,133 retrieve turns | `outputs/compile/compile_summary.md` |
| dev, as recorded in the report config | 0.2559 (dev n=13,284) | `configs/default.yaml` `report.probe_constant_bars` — config-recorded, not recomputed here |
| **test_seen — the bar D5 actually scores H5 against** | **26.30%** over **n=8,858** | `outputs/probes/response/select.json` `headline_test_seen.h5.constant.recall@1` = 0.2630390607360578 |

H5 measures **43.17%** on the same 8,858 rows, so the margin is **16.87 points**.
Subtracting the train prior instead gives 17.6 and is wrong.

**Skeleton concentration** (train): the top 4 single-act skeletons — ASK 18,181,
ACK 9,521, OFFER 9,101, CLOSE 5,569 — cover **42,372 of 71,133 = 59.6%** of train
retrieve turns.

**NOVEL intents held out of train:** `manage_pay_bill`, `manage_upgrade`,
`status_active`, `status_service_added`, `timing`
(`compile_summary.md`, `coverage_gap.json`).

---

## What this file got wrong before today

Every one of these was corrected above; they are listed so the same errors are
not re-imported from an older copy.

| stale claim | corrected |
|---|---|
| "Post-fix state: `bank_hash 98c2e937bea518d6` (was `31efad275a4b95a4`)" | the bank has been recompiled since; on disk it is **`c270df0e249444bb`** |
| "Template coverage did NOT move — 65.30% of sentences, 61.01% of utterances, byte-identical templates and skeletons" | it moved: **65.01% / 60.67%**, and templates went **4,417 → 4,489** |
| "Bank as measured: 4,417 templates" | **4,489** templates (570 skeletons / 30 actions / 16 slots unchanged) |
| "Full suite: 291 passed, 25 skipped" | **561 passed, 0 failed, 0 skipped** |
| D-3 "fixed at both call sites" | there are **three** call sites (1354, 1452, 1581); all three are fixed |
| D-4 support counts 1142/1157, 697/746, 2606/2924 | **1,100/1,115**, **643/688**, **2,347/2,657** on the current bank (rates unchanged) |
| D-4 "minimum support 58%" | **56.2%** (verify-identity pos 1, 1,492/2,657) |
| D-2 "only 49 (1.11%) bear any slot, 129 occurrences" | **71 (1.58%)**, **468** train occurrences |
| D-2 "24/49 and 53/129 correctly masked = 49%/41%" | measured on the pre-fix bank — **stale, must be re-hand-checked on the 71** |
| D-6 "delex_check got 141 of 200, act_check 187 of 300" | the shipped artifacts say **148 of 200** and **198 of 300** |
| D-7 "3,374 of 4,417 templates are merges, largest cluster 988" | **3,436 of 4,489**, largest cluster **327** — the 988 cluster is what the D16 merge guard removed |
| D-8 "308 OTHER templates / 1,198 occurrences, OTHER skeleton 1,836 turns" | **306 / 1,214**, skeleton S0009 **1,849** turns (2.6% of 71,133) |
| "Constant bar 25.59% — 18,200 of 71,133" | **25.56% — 18,181 of 71,133** on train; the bar that matters for D5 is **26.30% on test_seen (n=8,858)** |
| "Template coverage … This is the CEILING on reflex rate" | retired by DECISIONS D16 — it is a FIDELITY rate; assignment rate is 1.0 |
| D-1 "not yet re-verified on a full recompile" | **re-verified today**: the guard passes on the on-disk bank and 0 literal usernames remain |
