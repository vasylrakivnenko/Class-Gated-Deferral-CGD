# Router: status and next steps (2026-09-30)

## Live now
- router.zadum.ai = `zadum-router` systemd service, running **straight from this working tree**
  (`legalbench_map/ask_ui.py`). Restarting it deploys whatever is on disk, committed or not.
- Deployed = commit the commit "Classify questions first, then answer facts and choices from the text" (`git log`) (branch `router-pretier0-tier0`, not pushed; 2026-09-30 19:35 UTC), which includes the B + D edits:
  `router/frames.py`, `router/pretier0.py`, `router/lexicon_extra.json` (new), `tests/test_frames.py`.
  They were reviewed, tested (135 pass) and deployed on 2026-09-30, and committed in the commit "Classify questions first, then answer facts and choices from the text" (`git log`) with M0–M4.
- Since 2026-09-30 ~17:00 UTC every `/api/ask` call is saved in `/var/lib/zadum-router/usage.db` (`router/usage.py`):
  `requests` has the question, answer, confidence, evidence (JSON) and `id` (= the reply's `request_id`, `req_…`);
  `documents` stores each document once under `document_id` (`doc_` + sha256 of the text). Older rows have NULLs.
  Backup from before the change: `usage.db.bak-2026-09-30`. The db is now mode 600 (it holds user text).
  Decided 2026-09-30: no auto-delete for now, and the saved calls aren't shown in the admin page for now.

Pipeline for each question:
1. **Pre-Tier 0** (`router/pretier0.py` + `router/frames.py` + `router/lexicon_extra.json`): regex + Aho-Corasick,
   ~0.2 ms (≤8 ms on 300 KB). Answers alone:
   - topic questions ("Is X discussed here?"); clause frames (party, may/must/not, action, object, conditions);
     presence questions ("Does the agreement specify…")
   - LegalBench held-out: first clean run 99.1% precise at 6.1% coverage (754 answers); latest run 99.7% at 5.8%.
     Unseen tasks: only 1.6–2.2% coverage, lower bound 0.92.
   - also reads "we" as the one party that does the action; rewords they/my questions for we/you documents
2. **Tier 0** (`router/tier0.py`): NLI `cross-encoder/nli-deberta-v3-xsmall`. Answers **"yes" only**; its "no" answers are off (46–60% precise).
   Since 2026-09-30 17:30 a "yes" also needs the deciding sentence to name the question's object, itself or by a
   lexicon synonym (`_unnamed_object`; it caught "Can we audit their tits?", said yes at 0.93). Eval: blocks 3 of 9
   wrong yeses and at most 27 of 364 correct ones (`extensive/results/*_v3obj.jsonl`; adv cases 81-84 added).
3. **Jev**: one call with the user's question. Free classifiers are on standby (`--classifiers` turns them back on) and routing is skipped.

Lexicon: 94 concepts / 767 phrases. The LLM-proposed ones are in `lexicon_extra.json`: `qwen3p8-max` proposes and
`kimi-k3` verifies, via Fireworks, key `FIREWORKS_API_KEY` in `/root/.env`; 202 requests used. Work files are in
`/root/zadumai_nli_proto/extensive/v3/`.

## ACTIVE WORK: the question tree (M0–M5), started 2026-09-30 18:00 UTC — M0–M4 DONE (see WHERE THINGS STAND)
The user asked for M0 → M1 → M2 → M3 → M4 to be built one by one, each with evals, fixing and re-running until its
gates pass, updating this file after every task, without stopping. **A new session: read "WHERE THINGS STAND" at
the end of the progress log; all milestones are ticked; the listed ideas are optional next work.** Commit only when the user asks (everything through M4 is in the commit "Classify questions first, then answer facts and choices from the text" (`git log`)).
Backups of the working tree before this work: `/root/backups/legalbench_map-router-*.tgz`.

**Why.** The spec "Zadumai Question Tree — Build Spec" (user's doc, 2026-09-30) classifies every question into a leaf
before answering. Reviewed 2026-09-30: adopt its tree (request/judgment exits first, defer-by-default leaves,
data-gated milestones), but not its handlers as written. Measured: its Boolean rule (lemma co-occurrence) is 70.5%
precise on LegalBench and 25% on the adversarial set, vs Pre-Tier 0's 99.7%; its "or → Choice" rule would misroute
all 2,683 LegalBench "or" questions (all are yes/no: "Is the license irrevocable or perpetual?"); spaCy NER breaks
on legal text ("thirty (30) days" → `30) days'`). Live bug it fixes: today Pre-Tier 0 answers "Is the non-compete
enforceable?" / "valid?" / "reasonable?" with yes because the contract says so, and Tier 0 answers "Is it legal for
the landlord to enter?" yes at 0.955.

**Design (refined spec).**
- `router/qtree.py`: `classify(question) -> QuestionFrame` — deterministic, one spaCy parse + rules, p95 < 5 ms;
  every cue word in `router/qtree_cues.py`. Fields: form (polar/alternative/wh/other), branch (exit/lookup/reasoning),
  leaf, lehnert (log only), wh, slots (QA-SRL style: aux, subject, verb, object, pp), options, answer_type, negated,
  trace. Ordered tests, first match wins: request → judgmental → branch (lookup/reasoning) → leaf.
- Leaves → handling in `router/harness.py`:
  - **request** (summarize/draft/list…): no answer, no LLM call. Embedded questions are unwrapped first
    ("Please tell me if X" / "Can you tell me whether X" → classify X).
  - **judgmental**: prudential/evaluative only (should, advisable, enforceable/valid/legal/fair/reasonable/standard
    as the predicate). Deontic lookups about the document stay lookups even in first person ("Do we have to give
    notice?", "Can we audit their books?"). Lookup tiers never answer; goes to Jev, labeled as a judgment.
  - **boolean** → today's cascade: Pre-Tier 0 → Tier 0 → Jev noul. Polar questions with "or" inside stay here.
  - **choice** (M3): only real alternatives (options are parties/entities/values: "Does the tenant or the landlord
    pay for water?"). Rule tier, then Jev `choice` over the options + NONE.
  - **span[type]** (M2): who/what/when/how much/how long/how often/where/which. QA-SRL-style: the frame minus the
    wh-slot finds the sentence; legal-aware candidate patterns (not spaCy NER) give typed candidates; 1 candidate →
    answer quoting it; 2+ → Jev `choice` over candidates + NONE (TypeSafe "select instead of generate"); 0/NONE → defer.
  - **backward/forward/process** (why / what happens if / how to): defer (no answer, no LLM). M5 only if data says so.
  - **unclassified** (no parse, two questions): defer.
- Answers never say "No" from absence. Pre-Tier 0's explicit-prohibition "no" stays (4 of 720 held-out answers).

**Eval workspace:** `/root/zadumai_nli_proto/qtree/` (data/, labeled/, results/, eval scripts). Labeled question
sets are split dev/test; rules are tuned on dev only, test is reported. Fireworks (LLM labeling) via
`/root/zadumai_nli_proto/extensive/v3/fw.py` (cap 1500 requests, usage log in `v3/llm/usage.jsonl`).

**Gates.**
- M0: judgmental recall ≥ 95% on the test split; 0 of the 52 LegalBench wordings + adversarial/prior questions caught;
  ≤ 1% of real lookup questions (PrivacyQA) caught; Pre-Tier 0 and Tier 0 evals unchanged.
- M1: every LegalBench/adversarial/prior question → boolean; request + judgmental recall ≥ 95%; lookup questions
  misrouted between leaves ≤ 2%; confusion matrix + leaf distribution printed; p95 < 5 ms; no eval regressions.
- M2: span precision ≥ 95% when firing (Wilson lower bound reported), coverage reported, on CUAD short answers
  (dates, parties, governing law, notice periods) + a hand-made contract set + a PolicyQA sample.
- M3: choice precision ≥ 95%; all 2,683 LegalBench "or" questions stay boolean.
- M4: real-question report (PrivacyQA + our saved questions): leaf distribution, false-fire rates; decides M5.

**Progress log** (newest last; tick when done, with the eval numbers):
- [x] 18:05 Plan written here; backup made.
- [x] M0.1 `router/qtree.py` (`classify`, `judgment_cue`, `request_cue`) + `router/qtree_cues.py`. Unit tests: M0.3.
- [x] M0.2 labeled sets in `qtree/labeled/`: `fixed.jsonl` (163: LegalBench 52 + adversarial + prior; all boolean but
  "When is rent due?" span, "Should I be worried about subletting?" judgmental); `generated_{dev,test}.jsonl`
  (469 by kimi-k3, relabeled blind by qwen3p8-max, 97% agree; 13 "A or B?" disagreements adjudicated choice;
  stratified 50/50); `generated_test2.jsonl` (474, 8 new doc types, blind); `privacyqa.jsonl` (1,533 real questions,
  two-model silver labels, 95% agree; train split = dev, test split = test). PrivacyQA leaf mix: boolean 63%,
  span 24%, judgmental 4%, process 2%. Scorer: `qtree/eval_classify.py dev|test|test2 [--m0]`.
  M0 gate on blind test2: judgmental recall 95.8% [0.90, 0.98], precision 98.9%; 0/162 fixed lookups caught;
  PrivacyQA lookups caught 0.3% (dev 0.94%); p95 2.4 ms. Tuned on dev + the first test split (seen once, then
  "standard"-as-noun fixed); after test2, fixed "require" (obligation) and "explain what..." (a request):
  those items of test2 are no longer blind, so M1's final numbers use a new `generated_test3.jsonl`.
- [x] M0.3 DEPLOYED 18:28 UTC. `Harness.answer` classifies first; judgmental → `_fallback` with path
  `llm_judgment` (Pre-Tier 0/Tier 0 skipped); every Answer has `qtree` (the frame). `ask_ui.App` warms the parser.
  UI: key "judgment", "Question type" row, "skipped: not a lookup". Tests: `tests/test_qtree.py` (41), 191 pass.
  End-to-end on :8777: "Is the non-compete enforceable?" → llm_judgment (Jev 0.68); audit question → pretier0.
  No LegalBench question is caught, so Pre-Tier 0/Tier 0 evals are unchanged by construction.
- [x] M1.1–M1.2 full `classify()`: unwrap ("Can you tell me whether X" → direct question, `_invert`), openers/typos,
  leading condition → main clause (`_main_clause`; lookup keeps the condition), form, branch, leaf, answer_type
  (DATE/DURATION/MONEY/PARTY/JURISDICTION/PERCENT/FREQUENCY/LOCATION/PROPERTY/DEFINITION/CARDINAL/QUANTITY/ENTITY),
  choice options (`_options`: clause alternatives, subject alternatives, values/numbers/frequencies, opposites,
  repeated prepositions — not after can/may/must; presence questions keep their "or"), QA-SRL slots, compound →
  unclassified, non-quantity "how" → process (the spec's rule). BLIND test3 (459 new questions, 8 new doc types):
  accuracy 97.8%; non-lookup→lookup-leaf 0.4% [0.001, 0.025] (gate ≤2%); judgmental and request recall 100%;
  choice R 87%/P 97%; span 98.9%/98.9%; fixed set 100% (all LegalBench "or" questions stay boolean); p95 2.6 ms.
  PrivacyQA (tuned on, not blind) 95.7% accuracy, gate 1/33. After the blind run: fixed "supposed to"
  (obligation), "valid if" (a stated condition), "Who should I contact" (a fact) → test3 no longer blind for those.
  Results: `qtree/results/m1_test3.txt`.
- [x] M1.3 DEPLOYED 18:45 UTC. Harness: boolean → cascade on `frame.lookup` (Answer.question stays as asked);
  request → path `declined`; span/choice/backward/forward/process/unclassified → path `deferred`
  (`NOT_ANSWERED` reasons in harness.py), answer "not answered", 0 LLM calls. Verbless fragments: head noun with an
  answer type → span ("Late fee?" MONEY), else boolean ("Audit rights?"). All 52 LegalBench wordings: boolean with
  unchanged text, so their cascade is identical. UI: "not answered" card, key "span · DATE". 223 tests pass.
  Incident: the 18:44 deploy went out with 1 failing test ("Audit rights?" was deferred); fixed at 18:45.
  Final classifier numbers: generated test 98.3% / test2 99.1% / test3 98.5%, fixed 100%.
- [ ] M2 span handler (`router/spans.py`), IN PROGRESS. Done so far: typed candidates (DATE/DURATION/MONEY/PERCENT/
  FREQUENCY/PARTY via find_parties/JURISDICTION via a gazetteer/DEFINITION of the asked term/CARDINAL), key terms
  (frames tokens + concepts, words inside multi-word concepts, light verbs dropped), safe vs loose synonyms, rule
  tier (all key terms in the clause + exactly one candidate attached in the parse subtree of a matched word + no
  condition + heading tie-break), document-level dates by anchors only (defined "Effective Date", "commence on",
  preamble "entered into as of"), LLM tier `_choose` (Jev `choice` over candidates + "none of these", MIN_P 0.80).
  Eval `qtree/eval_spans.py spans_dev|spans_test|cuad [--llm]` (Jev calls cached in `qtree/llm/span_choice_cache.jsonl`).
  Sets: `labeled/spans_{dev,test}.jsonl` (kimi-k3 documents + fact questions, gold copied from text, qwen-checked;
  ~25% unanswerable), CUAD `data/cuad/test.json` (102 real contracts × 6 short-answer categories; TUNED ON).
  Results: spans_dev rule+LLM 111/111 = 100% precise, coverage 82.8%, 0/45 false fires (104 Jev calls);
  spans_test BLIND 114/115 = 99.1% [0.952, 0.998], coverage 82.0%, 0/46 false fires; CUAD rule tier 138/141 = 97.9%
  [0.939, 0.993], coverage 46%, 2/314 false fires.
- [x] M2 DEPLOYED 19:07 UTC. More fixes: type (interest/APR/uptime → PERCENT, "what ... pay" → MONEY, typed
  nouns anywhere), "kick in" = effective, prefix-relation matching (not 5-letter prefixes: "electricity" ≠
  "electronic"), soft filler nouns, the LLM state holds every relevant clause (bug: only first occurrences),
  temporal qualifiers ("old rent") → no rule tier. LLM policy set on `cuad_blind`: Jev picks only among clauses
  that name every key term (strict pools were 14/14 right; loose 20/26; unanchored document dates 17/27; picks
  < 0.9 were 0/7) → MIN_P 0.90; loose pools only in documents < 8,000 chars (96/96 right there, 42/53 above).
  FINAL BLIND: `spans_test2` (8 new doc types) 123/123 = 100% [0.970, 1.000], coverage 86%, 0/45 false fires;
  `cuad_blind2` (60 new real contracts) 100/102 = 98.0% [0.931, 0.995], coverage 49.5%, 2/158 false fires.
  Harness: leaf span → `spans.answer` → path `span` (rule, 0 LLM calls) / `llm_span` / `deferred` with the
  reason; Answer.span holds the search. UI: key "span · DATE", "Where it says so" clause, placeholder invites
  any question. 233 tests pass. Not built: ENTITY/PROPERTY/QUANTITY/LOCATION answers (deferred), non-party
  "who" answers ("the Board of Directors"), document-level dates without anchors (deferred).
- [x] M3 DEPLOYED 19:15 UTC. `router/choice.py`: key terms minus option words find the clauses; rule = exactly
  one option stated (numbers "30 days" = "thirty (30) days", content words, not negated) or, for parties, the one
  that is the verb's subject; else Jev `choice` over options + "neither / the text doesn't say" with spans' policy
  (strict pool or document < 8k chars, MIN_P 0.9). qtree `_options` now: options are the conjuncts' whole phrases
  ("in court" / "in arbitration" — bug: were bare prepositions), `_parallel_values` (options differing in numbers,
  names, periods or negation are alternatives even after can/must; "first refusal or first offer" stays boolean),
  "or not <verb>", "upon" repeated; `_trim_options` ("the bonus guaranteed" → "guaranteed"); spans: headings merge
  into their clause, the old/new qualifier guard ignores names ("New York"). Eval `qtree/eval_choice.py
  dev|test|test2 [--llm]` on `labeled/choice_*.jsonl` (kimi-written over the span sets' documents, qwen-checked).
  dev 69/71 = 97.2% (coverage 90.8%); test 58/58 = 100% (95.1%); BLIND test2 60/62 = 96.8% [0.890, 0.991]
  (89.6%); misses are "X or only Y" and one broken gold label. Classifier unchanged (fixed 100%, test3 98.5%);
  spans unchanged or better. 240 tests pass. Choice→boolean misroutes remain (~13–15%: "a USB drive or an online
  gallery"), which then get the yes/no cascade.
- [x] M4 DEPLOYED 19:21 + 19:27 UTC. `qtree/m4_privacyqa.py [--llm]` → `qtree/results/m4_privacyqa.txt`: PrivacyQA
  test split, 400 real questions over 8 real app privacy policies (experts marked relevant sentences; no answers).
  Leaf mix: boolean 63.0%, span 23.8%, process 6.8%, judgmental 3.8%, backward 1.0%, unclassified 1.0%, choice
  0.5%, forward 0.2%. **M5 gate (forward + process ≥ 15%): 7.0% → M5 not built.** Local tiers answer 6/400 on these
  long policies (Pre-Tier 0 2, Tier 0 3, span rule 1), 0 answers to the 34 questions with no relevant sentence;
  the rest go to Jev (boolean) or defer (span/other). Fixes from it: an age isn't a duration ("users under
  eighteen years of age" was answered to "how long do you keep my data?"), and Pre-Tier 0 frames.py: a
  condition word in the verb's slot ("does it SAVE my health data?", "save" as in "save as provided") no longer
  collapses into "is health data mentioned?" (A/B on LegalBench: dev unchanged, held-out 720 answers 99.7% as
  before). Extra types: AGE ("how old", "minimum age"; candidates "16 years of age", "the age of 13"),
  "number of" → CARDINAL, "credit limit"/"coverage" → MONEY, "time limit" → DURATION; anchors climb from a verb
  that modifies a noun or hangs off "be" ("sixteen years of age to buy Premium"). Tried and reverted: minimum ≈
  "at least" (took "at least 25 images" for "a minimum of 400"). 248 tests pass.
  Our own saved questions (usage.db): only 3 so far — too few to measure.

**WHERE THINGS STAND (2026-09-30 19:30 UTC)** — M0–M4 done and live; M5 not justified by the data.
| | precision when answering | coverage | set |
|---|---|---|---|
| classifier (8 leaves) | accuracy 98.5% | — | blind generated_test3 (459) |
| judgment/request exits | recall 100% / 100% | — | blind generated_test3 |
| span (facts) | 124/124 = 100% | 86.7% | blind spans_test2 (8 new doc types) |
| span (facts) | 101/103 = 98.1% | 50.0% | blind cuad_blind2 (60 real contracts) |
| choice | 60/62 = 96.8% | 89.6% | blind choice_test2 |
| Pre-Tier 0 yes/no | 718/720 = 99.7% | 5.8% | LegalBench held-out (reused many times) |
Ideas not done: choice→boolean misroutes (~13%); span coverage on long documents (loose pools were 77% right
there, so they defer); ENTITY/PROPERTY/QUANTITY/LOCATION facts and non-party "who" answers; a fresh LegalBench
held-out set. (Committed: the commit "Classify questions first, then answer facts and choices from the text" (`git log`).)

## Next steps (from before the question tree)
1. ~~Commit the B + D edits~~ done in the commit "Classify questions first, then answer facts and choices from the text" (`git log`).
2. Improve **unseen-task coverage** (only 1.6–2.2%): generalize the question phrasings rather than matching per task.
   Known errors: loose date/duration pattern ("one year before"), and a successor term read as the initial term.
3. Get a **fresh held-out set**. The current one has been run 4 times and errors from it drove fixes.
4. Optional:
   - an "I am the: Licensee / Licensor" selector in the page
   - make Tier 0 faster on long documents (1.3 s p50 at 50+ sentences on 1 thread)
   - index documents at upload (option C)

## How to check things
- Tests: `cd legalbench_map && ../.venv/bin/python -m pytest -q tests/test_frames.py tests/test_pretier0.py tests/test_tier0.py tests/test_router.py tests/test_usage.py tests/test_qtree.py`
- Question-tree evals: `/root/zadumai_nli_proto/qtree/` — `eval_classify.py dev|test|test2|test3`, `eval_spans.py spans_dev|spans_test|spans_test2|cuad|cuad_blind|cuad_blind2 [--llm]`, `eval_choice.py dev|test|test2 [--llm]`, `m4_privacyqa.py [--llm]` (Jev calls are cached under `qtree/llm/`)
- Pre-Tier 0 eval: `PYTHONPATH=. ../.venv/bin/python /root/zadumai_nli_proto/extensive/v2/evalv2.py dev|heldout`
  (split: `extensive/v2_split.json`, 11 of the 52 tasks held out as unseen)
- Tier 0 eval and adversarial set: `/root/zadumai_nli_proto/extensive/` (`run_eval.py`, `adversarial.jsonl`); prototype cases: `/root/zadumai_nli_proto/cases.jsonl`
- Deploy: `systemctl restart zadum-router`. Before that, test on a copy with `ask_ui.py --port 8777`, which has no sign-in and no quota.

## Gotchas
- Check pytest's own exit code before a deploy (`pytest ... ; echo $?`): `pytest | tail` always exits 0.
- Live API checks count against your 50/day quota (it's per account, and the `/admin` page changes it).
- Port 8766 has an old test server from an earlier session; leave it alone.
- Git has no identity on this machine; commits use `-c user.name=... -c user.email=...` from earlier commits.
