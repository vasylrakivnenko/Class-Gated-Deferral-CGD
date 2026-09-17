# Compile-stage defects — status after the fix round

Found by the compile agent's self-audit, re-verified by the maintainer.
**D-3 and D-4 are FIXED and regression-tested. D-1, D-2, D-5..D-8 remain open.**

Post-fix state: `bank_hash 98c2e937bea518d6` (was `31efad275a4b95a4`; only
`actions.jsonl` changed). Template coverage did NOT move — 65.30% of sentences,
61.01% of utterances, byte-identical templates and skeletons, so no third defect
was introduced. Regression suite `tests/test_compile.py`, 12 tests, each PROVEN
to catch its defect on the pre-fix module rather than merely to pass. Full suite:
291 passed, 25 skipped, 0 failed.

**Authorship of `compile.py` is unsettled** — the agent claimed the file as its
own, then later said it is another agent's implementation with its two fixes on
top. The code is tested and the numbers reproduce; the provenance is not
established either way, and nothing depends on it.

Bank as measured: 4,417 templates / 570 skeletons / 30 action patterns / 16 slots.
`bank_hash 31efad275a4b95a4`, `dataset_hash 40ce6eaa5279aeeb`. Determinism holds
(two runs byte-identical). Train-only holds. No generic slot names.

---

## D-1. PII LEAK — literal usernames in the bank. **SAFETY. Fix first.**

**Verified worse than first reported: 22 templates, 248 occurrences, 10 personas.**
`your username is safzal1` (32x), `rdomingo1` (30x), `cminh1` (24x),
`jbanter1` (22x), and six more.

A fast path built on this bank will tell one customer another person's username.
Clean on emails, order ids, zips, phones and customer names — this is the one leak.

**Cause (genuine, not sloppiness):** in the make-password flow the agent states a
username the scenario does not contain, so the exact-match delexicalization tier
has nothing to match against, and `delex_regex_slots` covers only email/phone.

**Fix:** add a DERIVATION tier. These usernames are first-initial + surname +
digit of the scenario's `customer_name` (safzal1 = Sanya Afzal). Derive the
expected username from the scenario and mask on that. Do NOT mask on a bare
`[a-z]+\d+` regex — it will over-mask.

**Then add a build-breaking assertion:** no template may contain a
persona-derived username. A PII leak must fail the compile, not warn.

---

## D-2. The delex quality metric is vacuous as reported

`delex_check.md` reports **99.93%** of templates defect-free. That number is
computed over all 4,417 templates, but **only 49 (1.11%) bear any slot at all**
(129 occurrences total). A random sample is trivially clean because
delexicalization almost never fires.

The honest figure, hand-checked template by template against source turns:
**24/49 templates and 53/129 occurrences correctly masked = 49% / 41%.**

Report the honest number, with 99.93% shown beside it as the misleading figure.
The spec's 98% bar is not met; it was only appeared to be met.

This is the constant-predictor failure mode in a new costume: a metric scores
well because the thing it measures barely happens.

---

## D-3. FIXED — mistyped literals corrupted half the slot-bearing bank

`_type_literal(value, [], {})` was called with EMPTY candidate categories, and
`_conversation_sources` declared `ontology_enumerable: dict = {}` and never
populated it. Shape patterns fired unconstrained; `customer_name`'s
`^[a-z]+(?:[ '-][a-z]+)+$` matched any two-word phrase.

**FIXED.** Ontology categories now populated at both call sites.
Verified after fix: **0 of 19,048 stored `slot_values` entries disagree with
ontology-aware typing** (was 4,699 of 21,335 = 22.0%). Entry count fell because
the junk entries are gone.

---

## D-4. FIXED — and the maintainer's supporting measurement was WRONG

`ActionPattern.required_slots` counted only TYPED occurrences in its modal vote,
so a position whose modal value had no registry slot still emitted a minority
typed slot. **FIXED**: None is now included in the vote, and nothing is emitted
when None is modal.

### Correction to the record: five of the eleven "0% support" slots were artifacts

The maintainer measured per-position support from `labels/train.jsonl` — **but
that file is the OUTPUT of D-3.** A typer with no evidence cannot produce
`membership_level` or `shipping_status` at all, so measuring D-4 against a file
corrupted by D-3 reported 0.0% support for slots that are in fact the
best-supported in the bank. **Circular measurement; the 0.0% figures were an
artifact of the other defect, not evidence.**

Ground truth is ontology-aware typing of the RAW train values. Re-verified
independently by the maintainer against `ontology["values"]["enumerable"]`:

| button | pos | slot | true support |
|---|---|---|---|
| membership | 0 | membership_level | **1142/1157 = 98.7%** (reported as 0.0%) |
| shipping-status | 0 | shipping_status | **697/746 = 93.4%** (reported as 0.0%) |
| verify-identity | 0 | customer_name | **2606/2924 = 89.1%** (reported as 32.0%) |
| make-purchase | 0 | name | **276/365 = 75.6%** (reported as 0.0%) |
| record-reason | 0 | name | 496/1276 modal is `reason_slotval` — genuinely unsupported, DROPPED |

A second limitation of the maintainer's re-check, stated for the record: it typed
against the ENUMERABLE lists only, so it cannot adjudicate shape-matched
non-enumerable slots (`account_id`, `order_id`, `email`) — those show as
`<untypable>` under it regardless of truth. The compiler's typer handles both
tiers and is the authority for those.

### State after the fix: 7 of 30 buttons keep required_slots, minimum support 58%

    verify-identity   [customer_name, account_id, order_id]   89% / 58% / 74%
    validate-purchase [email, order_id]                       92% / 96%
    pull-up-account   [customer_name]                         96%
    offer-refund      [amount]                                95%
    membership        [membership_level]                      99%
    shipping-status   [shipping_status]                       93%
    make-purchase     [name]                                  76%

DROPPED as unsupported: record-reason, enter-details, update-order,
update-account, notify-team.

**Lesson worth keeping:** do not measure one defect using an artifact produced by
another. Establish ground truth from raw inputs.

### Minor naming mismatch, unresolved

The maintainer's raw-ontology check found the modal CATEGORY differs from the
registry SLOT name in two cases: shipping-status pos0 is `shipping_option` (not
`shipping_status`), make-purchase pos0 is `product` (not `name`). Support is high
either way so no slot should be stripped, but the registry name and the ontology
category should be reconciled or the mapping documented.

---

## D-5. Decimal-split markers render money wrong

`_boundary_ok` tests only alphanumerics, so a marker that replaced the cents of a
price reads as clean. Four templates, verified:

    "$54.00 for 10 pounds or less, and $69.{amount} for everything else."
    "$54.{amount} for 10 pounds or less"
    "10 pounds or less is $54.99 and ${amount}.99 for all other items."
    "so for gift wrapping, it is a fixed price of $4.{amount}."

The filler renders `$4.164`. Low count, highest customer visibility of any defect
here. Treat a `.` after digits as a boundary violation.

---

## D-6. Check artifacts under-sample

`rng.choices(..., k=n)` samples WITH replacement then dedups by index:
delex_check got 141 of 200 requested, act_check 187 of 300. Trivial fix.

---

## D-7. DECISION TAKEN, not a defect to fix blind: report strict coverage too

Single-linkage at `dedup_threshold 0.92` chains: **3,374 of 4,417 templates are
merges, the largest cluster merges 988 distinct phrasings**, and 89% of forms in
that cluster are below 0.92 against their own canonical (min 0.547).
Occurrence-weighted across 47 clusters, **80.6% of merged forms sit below
threshold**. `"i will need your username, email, and order id please."` is merged
into `"can i have your account id and order id?"`.

This is contract-compliant — the spec asks for single-linkage — but it means
"covered" does not mean "answered with something a human would have said".

**Decision: keep single-linkage, and report a second honest number beside the
61.01%** — `strict_template_coverage`, the share of covered utterances whose gold
sentence is within threshold of the canonical it was merged into. Measure it, do
not estimate it. If it lands far below 61%, THAT is the real ceiling on reflex
rate and 61.01% is an overstatement we must not ship.

---

## D-8. Report, do not fix blind: product knowledge falls into OTHER

`act_embed_min_similarity: 0.25` plus INFORM/INSTRUCT seeds that do not span
ABCD's product-detail mode send real INFORM sentences into OTHER:
`"most of our boots are waterproof."` (20), `"the standard collar size is 15
inches."` (12). 308 OTHER templates / 1,198 occurrences, plus an OTHER skeleton
covering 1,836 turns (2.6%).

Changing act seeds moves every downstream number, so this needs a measured
before/after, not a guess.

---

## Numbers to carry into the report regardless

- **Template coverage 61.01%** of train agent utterances (43,397/71,133); 65.30%
  of sentences. Degrades steeply by length: 1-sentence 64.3%, 2 → 53.5%,
  3 → 24.7%, 4 → 11.4%, 5+ → 0%. This is the CEILING on reflex rate.
  The 34.7% shortfall is `min_template_count: 2` dropping one-off phrasings —
  a config choice, not a compiler failure. Do not lower it without measuring:
  a singleton template is one no human said twice.
- **Constant-predictor bar for the skeleton head: 25.59%** — always answering
  S0000 "ASK" is right on 18,200 of 71,133 turns. The top 4 single-act skeletons
  (ASK, ACK, OFFER, CLOSE) cover 59.6% of all turns.
- NOVEL intents held out: manage_pay_bill, manage_upgrade, status_active,
  status_service_added, timing.

---

## D-1 ADDENDUM — the derivation rule is verified, so the fix is well-defined

Measured on the full train split before implementing:

| fact | value |
|---|---|
| agent-stated usernames in train | 296 occurrences, 25 distinct |
| equal to `scenario.personal.username` | **0** |
| equal to first-initial + surname (+ digits) of `customer_name` | **288 = 97.3%** |

    alessandro phoenix   scenario username: alessandrop146   STATED: aphoenix1
    crystal minh         scenario username: (absent)         STATED: cminh1
    albert sanders       scenario username: albertsanders128 STATED: asanders1

**Zero stated usernames match the scenario's own `username` field.** In the
make-password flow the agent GENERATES a new handle, which is precisely why the
exact-match delexicalization tier had nothing to catch and why the leak survived
to the bank.

**Fix, now fully specified:** derive `first_initial + surname` from
`scenario.personal.customer_name`, and mask any token equal to that stem followed
by digits. That covers 97.3% by construction. Then assert at bank level that no
template contains such a token, and FAIL the compile if one does — a PII leak
must break the build, not warn.

The residual 2.7% (8 of 296) do not match the stem rule and must be inspected
rather than assumed clean.

---

## D-9. `log_every: 50` exceeds `smoke_max_steps: 20` — the smoke run is silent by construction

A `--smoke` training run caps at 20 optimizer steps but only logs every 50, so it
emits ZERO training lines between "model loaded" and exit. The run looks hung.

Cost when it happened: a working run was investigated three times as a suspected
hang — process listing, CPU-time sampling, memory forensics — before the config
explained it. Under swap pressure (the box was at 50.8/51.2 GB) "silent" and
"thrashing to death" are indistinguishable from the outside, which is exactly when
you most need the log line.

**Fix:** under smoke, set `log_every = max(1, min(log_every, smoke_max_steps // 4))`,
or simply log step 1 and the final step unconditionally. A run that produces no
output is unobservable, and an unobservable run cannot be reported on.

Related: there is no heartbeat anywhere in the training loop. One line per N
seconds — not per N steps — would have made this a non-event.

---

## D-1 FIXED — persona-derived username masking + a build-breaking guard

Two changes in `compile.py`:

1. **New delexicalization tier 2.5**, `_derived_username_stems` +
   `compile.derive_username_from_name` (default true). Derives
   `first_initial+surname` AND `firstname+surname_initial` from the scenario's
   `customer_name` and masks `stem + digits` to the `username` slot. Verified on
   all ten personas: `sanya afzal -> safzal`, `crystal minh -> cminh`,
   `rodriguez domingo -> rdomingo`, etc. The second stem form also catches
   scenario-style handles such as `alessandrop146`.
2. **`assert_no_leaked_usernames()` called inside `compile_bank` before the Bank
   is constructed.** Builds stems from the ontology's own `customer_name` list —
   so the check does not depend on any one conversation's sources — and raises
   `ContractViolation` if any template or surface form contains one. Tested: it
   raises on the real leaked strings (`your username is safzal1`,
   `your username is nbouchard1`) and passes a clean bank without false positives.

**A PII leak now FAILS the compile rather than warning.** Not yet re-verified on a
full recompile — the box was swap-bound with training on the critical path, so the
confirming run is pending.

---

## D-10. `--smoke` subsets AFTER the full corpus parse, so it does not reduce the dominant cost

`train.py:965` applies `train.smoke_conversations` to `train_partition` — which
`build_partitions` has already produced by parsing the entire 116 MB
`abcd_v1.1.json`. So `--smoke` shrinks the training loop but not the data load,
and the data load is the expensive part: a 116 MB JSON expands to a multi-GB
Python object graph.

**Measured consequence.** On a box under memory pressure (~2.9 GB free), a smoke
run configured for 40 conversations and 20 optimizer steps:
- burned 21 minutes of wall time for 2 minutes of CPU (~9% effective),
- never reached step 1 — no training line even at `log_every: 1`,
- sat at 13 MB RSS / 1.4% CPU, i.e. almost entirely paged out,
- produced no checkpoint on two separate attempts.

**Proof it is the parse and not the training.** Pointing `data.abcd_dir` at a
7.7 MB corpus (400 train / 120 dev / 120 test, all 55 intents preserved) took
`reflex compile` from **11+ minutes to under 60 seconds** on the same machine.

**Fix:** subset the conversation list inside the loader, before the per-turn
normalization and context building — or support a corpus-level
`data.max_conversations` that `build_partitions` honours. A smoke path that loads
the production corpus is not a smoke path.

**Workaround in use:** a pre-built small corpus at `scratchpad/abcd_small/`,
selected round-robin across intents so all 55 survive, pointed at with
`--set data.abcd_dir=...`. Suitable for INTEGRATION testing only — any number it
produces is from 400 conversations and is not a result.
