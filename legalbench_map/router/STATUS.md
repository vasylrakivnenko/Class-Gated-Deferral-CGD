# Router: status and next steps (2026-09-30)

## Live now
- router.zadum.ai = `zadum-router` systemd service, running **straight from this working tree**
  (`legalbench_map/ask_ui.py`). Restarting it deploys whatever is on disk, committed or not.
- Deployed = the commit "Read deferred facts with a checked LLM (Tier 2); let the document decide 'or' questions"
  (`git log`; branch `router-pretier0-tier0`, pushed to origin; deployed 2026-09-30 23:40 UTC, 275 tests pass).
  Earlier commits on the branch: "Classify questions first, then answer facts and choices from the text" (M0–M4,
  19:35 UTC) and "Answer questions locally with Pre-Tier 0 and Tier 0 before calling Jev" (the B + D edits:
  `router/frames.py`, `router/pretier0.py`, `router/lexicon_extra.json`, `tests/test_frames.py`).
- Tier 2 (hosted LLM reader) is ON: `--tier2 priority` in /etc/systemd/system/zadum-router.service (see TIER 2 LIVE).
  Since 2026-10-01 02:10 UTC the reader is switched on /admin instead, between 4 readers incl. our own GPU (see TIER 2
  SWITCH). Until an admin picks one, the flag decides. 2026-10-01 ~04:45 UTC: that tab became "Routing Pipeline",
  which also switches Pre-Tier 0 / Tier 0 / Tier 1 (see ROUTING PIPELINE); live since 04:45 UTC (8926c9e on main;
  this box's checkout is now on main, not router-pretier0-tier0).
- Since 2026-09-30 ~17:00 UTC every `/api/ask` call is saved in `/var/lib/zadum-router/usage.db` (`router/usage.py`):
  `requests` has the question, answer, confidence, evidence (JSON) and `id` (= the reply's `request_id`, `req_…`);
  `documents` stores each document once under `document_id` (`doc_` + sha256 of the text). Older rows have NULLs.
  Since 23:40 UTC, requests where Tier 2 ran also log `tier2` (service tier), `tier2_ms`, `tier2_server_ms`,
  `tier2_result`, `tier2_usage`. Backup from before the change: `usage.db.bak-2026-09-30`. The db is now mode 600 (it holds user text).
  Decided 2026-09-30: no auto-delete for now, and the saved calls aren't shown in the admin page for now.

Pipeline for each question (`router/harness.py`):
0. **Question tree** (`router/qtree.py`) classifies it first: a request → declined; advice or a legal conclusion →
   Jev (`llm_judgment`); a fact → step 4; a choice → `router/choice.py` (rule, then Jev `choice`); why / what if /
   how → deferred; yes/no → steps 1–3 (a yes/no with "or" whose text states exactly one option → a choice).
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
3. **Jev** ("Tier 1" in /admin): one call with the user's question. Free classifiers are on standby (/admin's
   Routing Pipeline tab, or `--classifiers` at startup, turns them back on) and routing is skipped.
4. **Facts** (`router/spans.py`): a rule copies the one typed value attached to what the question names, else Jev
   picks among the text's candidates (≥ 0.9). What they defer goes to **Tier 2** (`router/reader.py`): gpt-oss-120b
   on Fireworks copies the answer from a clause, and it counts only if it's verbatim, fits the question
   (`reader.fits`) and Jev confirms the clause states it (≥ 0.7); otherwise the question defers. Kev mode skips Tier 2.

Lexicon: 94 concepts / 767 phrases. The LLM-proposed ones are in `lexicon_extra.json`: `qwen3p8-max` proposes and
`kimi-k3` verifies, via Fireworks, key `FIREWORKS_API_KEY` in `/root/.env`; 202 requests used. Work files are in
`/root/zadumai_nli_proto/extensive/v3/`.

## CONTRACT MAP (2026-10-02; deployed 23:30 UTC = 16:30 PDT; /admin switch "Contract map", off until an admin turns it on)
The user asked to use SALI/FOLIO, CUAD and ACORD for a tree of contract and clause types; research, benchmarks and
the build are in `/root/zadumai_nli_proto/contract_map/` (`bench/RESULTS.md` has every number; `taxonomy.json` = the
shared taxonomy: 146 clause types, 31 shared across datasets, 50 linked to FOLIO clause classes).
- `router/contract_map.py`: the document's kind (commercial / NDA / privacy policy / terms of service / lease / not a
  contract) and its sections' clause types, reported on each answer (`Answer.contract_map`; "Document" line in the
  playground). **Who answers is unchanged ("local first", the user's call).** Harness: `contract_map=` (ask_ui passes
  it when the switch is on); mask " m1" when on.
- Models: `models/contract_map/` (gitignored; built by `/root/zadumai_nli_proto/contract_map/build_models.py`): router
  95.1% on 1,156 held-out documents (99.1% on the 75% at p >= 0.9; texts < 3,000 chars get no kind); taggers by kind:
  LEDGAR 77 types (91.5% right on the 97% tagged), CUAD 37 (F1 0.60), ContractNLI 17 (0.68), OPP-115 9 (0.77),
  UNFAIR-ToS 8 (0.74); leases: fine-tuned MiniLM (0.28 on all 8,057 test paragraphs; TF-IDF 0.21), run in a background
  thread. Jev reads at most 8 unsure sections per new document (CUAD/ToS/privacy taggers; +0.015-0.02 F1 measured).
  Analysis is cached per document (64 documents).
- Built, measured and left off (`ROUTE_HIGH_RISK`, `ROUTE_LEASES`): sending questions about 35 hard-to-tag CUAD types
  (caps on liability, exclusivity, non-compete...) and every lease question straight to the LLM. On the open sets it
  would move 10% of user-style questions (57-70% of LegalBench's) off the local tiers, which answer them 99.6% right
  (open sets: Pre-Tier 0 604/604, network 469/472) where the LLM (Jev) got 88.4% (499 sampled; 1,000 Jev calls).
  Scripts: `contract_map/eval_routing.py`, `llm_check.py`. Lease rule unmeasured (no lease questions in our sets).
- Checked before the deploy: router suites 494 pass (exit 0); :8777 same answers as the p3 deploy; in-process with the
  live stages and the map on, 7 questions over 5 real documents (a 101k-char CUAD contract, NDA, privacy policy, ToS,
  lease): answers identical with the map on and off, kinds all right, first question per document +0.01-0.28 s, then
  cached; lease tags arrive in the background.
- Deploy note: an uncommitted Pre-Tier 0 fix ("f1", from the free public sets work) was in the tree; it was set aside
  for this deploy and put back afterwards, still not deployed or committed (`/root/backups/f1-working-tree-2026-10-02.patch`).

## ACTIVE WORK: Pre-Tier 0 "next level" (started 2026-10-01 ~20:15 UTC; user: "go build it to the next level")
DEPLOYED 2026-10-02 04:22 UTC (p3 = passive/noun shapes + routing to the network + rule fixes, the user's OK):
checked on :8777 and in-process with the live stages (routed questions answered on path tier0net in 44-113 ms; the
fixed rows answer or defer as intended), zadum-router restarted, live switches unchanged after it (Pre-Tier 0 ->
network -> Jev, NLI Tier 0 off, Tier 2 fireworks-priority). Pre-restart diff:
/root/backups/live-tree-before-restart-2026-10-02-p3.patch. Committed and pushed with c4 (818b4bf) to origin/main.
DEPLOYED 2026-10-01 23:13 UTC (c4, the user's OK): checked on :8777 (Pre-Tier 0 on the new shapes in 2-9 ms, defers
where it should) and in-process with the live stages (Pre-Tier 0 -> reader network -> Jev; NLI Tier 0 off), then
zadum-router restarted. Committed after the restart (see `git log`). Pre-restart diff:
/root/backups/live-tree-before-restart-2026-10-01-c4.patch (+ -new-files.tgz).
Note: ask_ui on :8777 without --require-user has no usage DB, so it runs the default stages (NLI Tier 0 on, reader
network off), not the saved live ones; check the live combination in-process (App + Stages).
Workspace: `/root/zadumai_nli_proto/extensive/v4/` (not in git). Tools there:
- `pt0eval.py [SETS] [--save NAME] [--diff NAME] [--groups]`: Pre-Tier 0 on every open set (dev, heldout, freshA,
  freshB, genval, conval, adv, prior, v4open) with a diff against a saved run (`runs/NAME.json`); `errs.py RUN SET`
  lists wrong answers; `bykind.py` = precision per frame kind. v4sealed only with `--final` (logged in sealed_runs.log).
- Scoring: "yes" right only on gold yes; on gold not_stated (genval, conval, v4 gen) any answer is wrong.
**Step 0, new sealed set (in progress):** `build_v4.py` -> `v4_rows.json`. Text no earlier eval or training used
(no 10-word run shared, `shingles.py`): LEGALBENCH unfair_tos sentences with 8 consumer questions written up front
(578 rows, dataset gold), and user-style questions in 6 shapes (direct, condition-first, passive, two actions, noun,
plain) written by gpt-oss-120b over 450 LEDGAR **test** provisions (no script had used that split) and 150 ToS
passages, kept only where the writer, gpt-oss-120b (blind) and deepseek-v4-pro (blind) agree. Split by passage:
1/3 "open" (tune on it), 2/3 "sealed" (run once at the end).
**Finding that changed the order:** on generated user-style questions (genval) Pre-Tier 0 was only 78.8% right
(25 wrong of 118) and gave 27 false yeses on near-misses (conval), vs 99.6% on LegalBench. So precision first:
- `router/qshapes.py` (new): rewrites user shapes to the frame's "Aux Subject Verb Object [condition]" ("Does the
  agreement require X to Y" -> "Must X Y", "If C, can X Y" -> "Can X Y if C", passives with/without "by X",
  "Is reverse engineering prohibited?"), and finds the question's named subject ("Must the Agent pay ...?").
- frames.py: a named subject the document uses as a party becomes an actor (the Agent must be the one paying); no
  content may come before the actor ("Is there a limit on the number of subsidiaries to which the Company can
  assign" isn't "Can the Company assign"); condition cues must be the same kind (after ~ upon ~ if; not before,
  during, except, without; "after July 31" vs "through July 31"); presence frames get relation guards (identity
  "Is X a Y?", before/after, "without", "owed by X", "limited to", a specific rate/amount/date needs one, "required
  to" needs an obligation word, words only named to be excluded); two modalities at once defer.
| set | baseline | checkpoint s3 |
|---|---|---|
| LegalBench dev | 896 = 12.0%, 99.6% | 896 = 12.0%, 99.6% |
| held-out (report only) | 1,087 = 8.7%, 99.8% | 1,084 = 8.7%, 99.8% |
| fresh A / B | 185, 99.5% / 191, 97.4% | same |
| genval (generated) | 118, **78.8%** | 67, **100%** |
| conval (near-misses, none yes) | **27 false yes** | **6 false yes** |
Tests: 638 pass (the 4 failing in test_determinism / test_per_item_dump fail on 4ef0122 too).
**v4 set built** (20:59 UTC): open 1,126 scored rows (661 LEDGAR-generated, 276 ToS-generated, 189 ToS category; 10
disputed ToS rows left out), sealed 2,204. On v4open (never used for the fixes above): baseline 52 answers 86.5% ->
s6 49 answers 98.0%: the precision fix holds out of sample. Coverage of user-style questions stays ~4%.
**Coverage, checkpoint s8 (21:45 UTC):**
- open-vocabulary actions (frames.OPEN "V:"): the question's own verb after its subject when the lexicon lacks it
  ("Must the Holder surrender the Note"), matched literally with the same actor/modal/negation/condition checks;
  "the shares" after a determiner is a noun, not "share"; passive sentences without "by X" answer any-party questions
  ("The Source Code shall be deposited"), "yes" only; a "must" question isn't answered by a bare "is identified"
  (a description); "am" is an auxiliary; a "not" inside the question's condition doesn't negate the question;
  topic questions skip negated sentences.
- `router/equivalences.py` (new, item 3): hand-checked formulas, only "yes", only when the frames defer, each with a
  veto: independent development carve-out, information received from a third party, existence/terms kept
  confidential, survival after termination (the question's own verb, no durations, same owner), no license granted.
  THING:ORAL concept (oral/verbal/spoken) in the lexicon.
| set | baseline | s8 |
|---|---|---|
| LegalBench dev | 896 = 12.0%, 99.6% | 968 = 13.0%, 99.6% |
| held-out (report only) | 1,087 = 8.7%, 99.8% | 1,176 = 9.4%, 99.7% |
| fresh A | 185 = 7.7%, 99.5% | **520 = 21.7%, 99.8%** |
| fresh B (open now) | 191 = 7.9%, 97.4% | **491 = 20.4%, 99.6%** |
| genval / v4open (user-style) | 118, 78.8% / 52, 86.5% | 74, 100% / 47, 100% |
| conval false yes | 27 | 5 |
**Final code = checkpoint s14 (2026-10-01 ~21:40 UTC)**, after s8: NOTIFY action + NOTICE thing ("provide ... with
prompt written notice"), "its" no longer required in the sentence, "the disclosing party" binds to the other party in
documents that never say "disclosing party", the document's own names for its confidential information
(`frames.info_aliases`: '"Information" means ... confidential', '(collectively, the "Evaluation Material")'; checked
first so "the Information" isn't just any information), wider no-license formulas, a payee isn't the actor
("compensation to such Grantor) to use"), a "no" whose thing has a restrictive clause defers ("employees who do not
have a need to know"), verbs after "from" ("prohibited from reverse engineering"), "reserve the right", fillers
("just", "actually") ignored, topic questions with a relational qualifier ("clause restricting the Executive") aren't
topic questions. Tried and dropped: requiring a passive's subject to be the question's object (lost 7 held-out audit
answers for 1 fresh error); LLM-mined lexicon from misses (`v4/mine_lexicon.py`: 782 single-slot misses, 36 phrases
in 2+ rows, the blind verifier kept 2; the misses are document binding and nominal forms, not synonyms).
Latency (300 KB worst case, `v4/latency_v4.py`): 9.2 ms vs 9.0 ms for 4ef0122 measured the same way (the formulas use
an Aho-Corasick cue scan before their patterns; without it they cost up to +10 ms). Tests: 685 pass (new:
tests/test_pretier0_v4.py, 47 cases); the 1 failure (test_determinism) fails on 4ef0122 too.
| set | 4ef0122 | s14 |
|---|---|---|
| LegalBench dev | 896 = 12.0%, 99.6% | 971 = 13.0%, 99.6% |
| held-out (report only) | 1,087 = 8.7%, 99.8% | 1,189 = 9.6%, 99.7% |
| fresh A | 185 = 7.7%, 99.5% | 602 = 25.2%, 99.7% |
| fresh B | 191 = 7.9%, 97.4% | 559 = 23.2%, 99.6% |
| genval (generated) | 118, 78.8% | 79, 100% |
| v4open (generated + ToS, never tuned on before s6) | 52, 86.5% | 48, 100% |
| conval false yes | 27 | 5 |
System (Pre-Tier 0, then the reader network as live; `v4/system_eval.py`, s10): fresh A 708 = 29.6% at 99.7%
(the network's own report had 403 = 16.8%), fresh B 634 = 26.3% at 99.7%, v4open 186 = 16.5% at 99.5%.
**Sealed run, pre-registered (written before running it):** v4sealed (2,204 rows; 10+20 disputed ToS rows left out),
once, totals only, by part: Pre-Tier 0 alone (`pt0eval.py v4sealed --final`) for 4ef0122 and s14, and the system
(`system_eval.py v4sealed --final`) for both. Scoring as above ("yes" right only on gold yes; any answer on
not_stated wrong). Every sealed run is logged in v4/sealed_runs.log.
**Sealed v4 result (run once, 21:35 UTC; 2,184 rows):**
| | Pre-Tier 0 alone | network on the rest | both |
|---|---|---|---|
| 4ef0122 | 112 = 5.1%, **81.2%** | 229 = 10.5%, 99.6% | 341 = 15.6%, 93.5% |
| s14 | 95 = 4.3%, **88.4%** | 233 = 10.7%, 99.6% | 328 = 15.0%, 96.3% |
By shape (s14, Pre-Tier 0): direct 60 answers 98.3%; noun 8 answers 25%; two actions 3, all wrong; ToS category 1.
So the precision fix generalized only partly: the open half's 48/48 was a small sample. On user-style questions the
network (99.6%) is now more precise than the rules. v4sealed is open since this run (`pt0eval.py v4all`).
**s15 (from v4's 11 sealed errors, now open; general causes only):** "its rights" after the verb is the object (not
"may"); "Can the Employee discuss X" isn't a topic question (only the text, or a passive, "discusses"); "including,
but not limited to" isn't a limit (stripped for word matching only); "limit on the amount" needs an amount; a value
the question puts on one side ("more than $175,000,000") needs that side next to it ("up to" doesn't); a presence
question asking permission isn't answered by a sentence that prohibits (and the reverse); "we"/"the company" and
"you" stay required in presence questions ("Is the company responsible ...?" vs "you are responsible"); an
unresolved "I" needs "I/me/my"; "Is there any right for me to own X" isn't a presence question. Tried and dropped:
blocking every presence question with a pronoun subject (lost 6 right for 1 error); the network as a veto on
Pre-Tier 0's answers (`v4/veto_study.py`: removes 3-8% of right answers for 1-2 errors per set).
s15 vs s14: v4all 131 answers 98.5% (s14: 143, 92.3%), dev/fresh/held-out unchanged, genval -2 right "no"s.
Tests 693 pass; latency 9.1 ms worst (300 KB).
**v5, a new sealed set for s15** (`build_v5.py`, generated questions only, over 600 LEDGAR test provisions and ~100
ToS passages v4 didn't use, same prompts and three-way agreement; all sealed; built 22:40 UTC: 3,208 rows): run once
with --final, Pre-Tier 0 alone and the system, for 4ef0122, s15 and **c2** (the second cycle below, also written
without seeing v5; frozen copies in the session scratchpad). Totals by part and shape only.
**v5 result (run once, ~22:20 UTC; 3,208 rows), Pre-Tier 0 alone:** 4ef0122 191 = 6.0% at **84.8%**; s15 156 = 4.9% at
**92.3%**; c2 154 = 4.8% at 92.2%. "direct" questions 93/93 right; errors in passive (5), condition-first, two
actions, noun, plain. System (4ef0122): network 438 = 13.7% at 99.3%, both 629 = 19.6% at 94.9%.
**Self-critique checks added (all open sets):**
- whole contracts (`v4/docs_pt0.py`, 102 CUAD test contracts x 38 questions, + `docs_judge_v4.py`: the teacher's
  literal verdict on changed answers): s15 answered 337 (4ef0122: 290); of its 51 new answers CUAD agreed with 23,
  the teacher with 48 (26 are "obligations that survive termination", which CUAD files only under post-termination
  services). 3 literal errors -> fixed in c2/c3.
- rewording (`v4/paraphrase_eval.py`: 4 user rewordings per benchmark question by gpt-oss-120b; 1-2 tuned on, 3-4
  held out): fresh A coverage original / held-out wordings: 4ef0122 7.7% / 2.4%, s15 25.2% / 8.8%, c3 26.2% / 11.5%
  (precision 99.6-100%); no row flips yes/no between wordings in c3.
- the network as a veto on Pre-Tier 0 (`v4/veto_study.py`): not worth it (3-8% of right answers for 1-2 errors).
**Cycle 2 (c2, from the critique checks):** questions about the text ("Is X specified in the agreement?", "Is the
clause about X?", "Does this provision cover X?", "Is X considered confidential information?"), "limit the use of X
to the purposes" = "use X only for the purposes", "once the agreement ends", "create copies", "that the agreement
exists", "if required by law" in its wordings (THING:LAW_REQUIRED), formulas with more verbs (create/produce/come up
with; given/conferred), "A, or is it B?" defers, mutual termination isn't without cause, a passive inside an
"if"-clause isn't a fact, "notify" composite needs "notice" right after the verb, a compatible cue covering the
question's condition excuses another cue on the same words, "not prohibited" = may.
**Cycle 3 (c3, from v5's 12 c2 errors, v5 open since):** an actor after "to/for/with/from/without" isn't the subject
("consent to Guarantor, assign"); "shall not be construed to require CBS to establish" asserts nothing; "from any
insurer" needs any/all in the text; "all X" isn't "some X"; a concept the question names twice needs two mentions
("Does the termination cancel ...?" vs "shall survive the termination"), not across a condition or for plain words;
"after fifteen days" vs "no later than fifteen days"; the consent reading of "require" only for consent questions;
"required/mandatory" presence questions need an obligation word; "can it be released?" (pronoun, no agent) defers.
Tried and dropped in c3: "waiver of" as a negation (broke questions about waivers), "must" presence questions need
an obligation word (lost "Must the free trial be free of charge?"), excluding "either party" from bare passives
(lost "may be executed in counterparts"), "once" as a contract condition word ("once per calendar year").
| set | 4ef0122 | s15 | c3 |
|---|---|---|---|
| dev | 896, 99.6% | 971, 99.6% | 969, 99.6% |
| held-out | 1,087, 99.8% | 1,189, 99.7% | 1,196, 99.7% |
| fresh A | 185, 99.5% | 602, 99.7% | 628 = 26.2%, 99.7% |
| fresh B | 191, 97.4% | 559, 99.6% | 570 = 23.6%, 99.6% |
| v4 all (open) | 164, 82.9% | 131, 98.5% | 130, 98.5% |
| v5 (open since its run; c3 tuned on its errors) | 191, 84.8% | 156, 92.3% | 145, 97.9% |
| conval false yes | 27 | 5 | 3 |
| whole contracts (CUAD gold) | 290, 86.9% | 337, 81.0% | 332, 81.6% |
Tests 693 pass; latency 9.2 ms worst (300 KB). **v6** (`build_v6.py`, new sealed set from LEDGAR test provisions and
ToS passages v4/v5 didn't use) is the blind check for c3: run once, 4ef0122 / s15 / c3. **OpenRouter credits ran out
during its build (HTTP 402, ~22:35 UTC; this session spent ~$4.03, 7,335 calls)**, so v6 kept only the passages both
blind checks finished: 698 rows (LEDGAR only; 284 yes, 130 no, 284 not_stated). Live router unaffected (Tier 2 =
fireworks-priority). Bulk LLM work needs the OpenRouter balance topped up.
**v6 result (run once, 22:53 UTC; 698 rows), Pre-Tier 0 alone:** 4ef0122 46 at 76.1%; s15 43 at 95.3%; c3 43 at 95.3%.
System (Pre-Tier 0 then the network): 4ef0122 171 = 24.5% at 93.6%; c3 162 = 23.2% at **98.8%** (network 119 at 100%).
Across v5+v6 (blind for s15), "direct" questions were 125/125 right; the errors are in rewritten shapes (passive,
noun/gerund, two actions) and on rows the network can't answer either.
**Tier order (`v4/policy_study.py`, open sets + v6):** where the network also answers Pre-Tier 0's rows, the two agree
(network 100% there), so "network first for risky paths" changes nothing, and "network only for risky paths" trades
~1 point of coverage for <=0.3 point of precision. Kept: Pre-Tier 0 first.
**Cycle 4 (c4 = FINAL, from v6's 2 errors; not blind-validated: no credits for a v7):** "no Corporation match ... can
be made" (a "no" right before the actor negates); "Is there an X provision ...?" isn't answered by a sentence that
denies X (topic "Is X discussed?" unchanged; test_pretier0's "Is there an audit clause?" over a denial now defers,
intentionally). Every open set unchanged otherwise; v6 now 42/42.
**FINAL (c4) vs 4ef0122:** dev 896 -> 969 (99.6%); held-out 1,087 -> 1,196 (99.7%); fresh A 7.7% -> 26.2% (99.7%);
fresh B 7.9% -> 23.6% (97.4% -> 99.6%); rewordings held out 2.4% -> 11.5%; user-style blind: 84.8% -> 92.3% (v5,
s15), 76.1% -> 95.3% (v6, c3); system on v6 93.6% -> 98.8%; whole contracts 290 -> 332 answers (teacher-literal right
on 48/51 of s15's new ones); latency 9.1 ms worst (300 KB) vs 9.0; tests 693 pass.
**Bulk LLM calls moved to Fireworks (2026-10-01 ~23:30 UTC, the user's choice):** reader_net/llm.py defaults to
Fireworks (OpenRouter ids mapped to Fireworks'), held to 20k output tokens/min and 8 requests at once of the account's
45k/min that the live Tier 2 shares (_FireworksGuard). deepseek-v4-pro isn't deployed on the Fireworks account
(404): the second blind checker on Fireworks is kimi-k3. **v6b (pre-registered):** the v6 passages OpenRouter left
unverified (never evaluated), verified by gpt-oss-120b + kimi-k3 on Fireworks (`v4/build_v6b.py`); run once for
4ef0122, s15 and c4 (= 818b4bf, live), Pre-Tier 0 alone and the system; totals by shape.
**v6b result (run once, 2026-10-02 00:10 UTC; 1,201 rows, 219 of them over ToS; 17 passages dropped after Fireworks
read timeouts):** Pre-Tier 0 alone: 4ef0122 73 at 84.9%; s15 67 at 89.6%; c4 66 at **89.4%** (7 wrong). System:
4ef0122 229 = 19.1% at 95.2%; c4 225 = 18.7% at **96.9%** (network 159/159). By path (c4, v6b): action 27/29,
action on a rewritten question 14/15, presence 1/3, catch-all 16/17, formula 1/2; dropping the risky paths (P3) would
give 97.6%. **Pooled over the blind sets (v5 s15, v6 c3, v6b c4): "direct" questions 168/168; every other shape ~79%
(79/100).** v6b's errors (not fixed; open since): survival formula on "Do we have to notify ... when obligations
survive" (the formula's question must have survive as its own verb, not after "have to ... when"); "except for X"
names X only to exclude it; "if the amendment is not adverse" vs the question's "if ... adverse" (negation inside the
condition); "it is your obligation to avoid making available" ("avoid" as a ban); "Is there a requirement that
Renren must provide ..." (an action question in a presence shape, roles swapped); "we may edit ... modifying ..., we
will not modify the meaning" read as a yes to "Must the Company modify ...?". Each blind set has turned up ~7 errors of
this kind in the non-direct shapes: patching alone won't take user-style questions to 99.5%.
**River AI (2026-10-02, the user's request):** client `river-client` in `/root/zadumai_nli_proto/river_venv` (uv,
Python 3.12; key RIVER_API_KEY in /root/.env); reader_net/llm.py has backend="river" (run it with that venv's python:
it has requests, so llm.py works there). 13 models on the key; Qwen3.5-397B/-122B, DeepSeek-V4.1-Flash and Nemotron
time out cold at first, then answer slowly. Checker benchmark (`v4/river_bench.py`, report in
`v4/river_bench_report.txt`; A = 156 ToS questions with the human dataset's gold, B = 50 v5 passages / 217 questions,
C = 120 near-misses where any "yes" is wrong), accuracy over answered items:
| model | A ToS | B generated | C false yes | missing | tokens/call |
|---|---|---|---|---|---|
| River Kimi-K2.6 | 91.0% | 97.7% | **12.2%** | 1.0% | 2,062 |
| River GLM-5.2 | 92.2% | 99.0% | 15.3% | 4.3% | 1,022 |
| River Qwen3.8-27B | 89.7% | 96.7% | 15.5% | 1.4% | 1,469 |
| River Qwen3.6-35B-A3B | 93.6% | 97.7% | 25.8% | 0% | 1,601 |
| River DeepSeek-V4-Flash | 91.0% | 98.6% | 55.0% | 0% | 39 |
| Fireworks gpt-oss-120b (in B's gold) | 90.4% | 98.6% | 22.5% | 0% | 308 |
| Fireworks kimi-k3 | 92.3% | 99.1% | 24.2% | 1.0% | 427 |
(partial runs: Qwen3.5-122B 93.4/99.4/22.0% with 12.6% missing; Qwen3.5-397B 92.3/100/0% with 64% missing;
Nemotron 89.3/93.7/13.4% with 14.6% missing; GLM-5.3-Flash 91.0/98.6/24.7%.) A doesn't separate the models (its
ceiling is the dataset's category labels); C does. llm.json_of now also handles reasoning that ends in a bare
"</think>" (Qwen3.8, Nemotron on River).
**Passive and noun shapes (2026-10-02, the user: "focus on the engine — fix the passive and noun shapes"); candidate
p2, working tree only (NOT deployed, NOT committed; live = c4 = 818b4bf):** on the open generated sets (v4all, v5all,
v6all, v6ball: 1,183 passive / 1,374 noun questions) c4 answered passive 28 (25 right), noun 16 (13 right); p2 answers
passive 33 (31 right), noun 21 (20 right). Changes: noun questions rewritten to verbs in qshapes ("Is there a
requirement for X to V" -> "Must X V", "a right for X to V" -> "Can X V", "a prohibition on assignment of X" -> "Is a
party prohibited from assigning X", "a cap/limit on X" -> "Is X capped", "Is assignment of X prohibited/permitted/
required" -> verb forms; "Must X refrain from V-ing" -> "Is X prohibited from V-ing"); a passive's modal carries over
"and" ("shall accrue, and be carried forward"); verb particles aren't things ("carried over"/"carried forward");
"except for X" (and the question's whole object, "permitted liens") names X only to exclude it, in positive sentences
(in a ban, "other than its employees" still permits them); a participle question ("Is the provision terminated ...?")
needs the verb, not just the noun; a single-word action after an article is a noun in sentences too ("construed as a
release"); list markers "(a)", "(iv)" are removed before judging ("shall not (a) use"); a question word matches a
sentence word the document made a name ("Vacation shall accrue"); formulas "changes in writing" and "exhibits are part
of it" (strict question forms only: the loose versions answered 14 near-misses "yes"); survival also for
indemnification / representations / sections. Tried and dropped: "no" from passives without "by X" (+4 answers, 2
wrong); passive gold-"no" questions are mostly contrasts ("by email" vs "by express mail"), out of reach of safe rules.
Open sets, p2 vs c4: dev 969 -> 967 (99.6%), fresh A 628 -> 627, fresh B 570 -> 572 (99.8%), held-out 1,196 -> 1,195
(99.7%), genval 75 -> 79 (100%), v4all 130 -> 144 (98.6%), v5all 145 -> 150, v6all 42 -> 44 (100%), v6ball 66 -> 64
(errors 7 -> 4), conval false yes 3. Tests 711 pass (+17 cases); latency 9.2 ms worst. **v7** (`v4/build_v7.py`,
sealed, passive + noun questions only, over the last 236 unused passages, Fireworks: gpt-oss writes and checks,
kimi-k3 checks) is the blind check: run once for 4ef0122 / c4 / p2.
**v7 result (blind, run once 2026-10-02 02:44, 1,120 passive + noun questions):** Pre-Tier 0 alone: 4ef0122 115
answered @ 84.3%, c4 100 @ 91.0%, p2 111 @ 90.1%. System (Pre-Tier 0 + network): c4 263 @ 96.6%, p2 269 @ 95.9%. The
network was right on every v7 row it answered. Rule patches raised coverage on these shapes, not precision.
**p3 = p2 + routing + fixes (2026-10-02; deployed 04:22 UTC, committed):**
- Routing: `pretier0.ROUTE_RISKY` (env `PRETIER0_ROUTE_RISKY`, default "1") leaves Pre-Tier 0's answer to the reader
  network on its two weakest paths on generated questions: catch-all presence frames (`strict`) and action frames on
  a question qshapes rewrote (`_left_to_the_network`). `v4/policy_study.py` (P1 = Pre-Tier 0 first, live order; P3 =
  network only on those paths; log `v4/policy_p2.log`): on every open set the network was right on all rows where both
  answered; the rows it declines are where the rules erred (12 wrong of 145 there).
- Fixes for the 7 rule errors left on v7 / v6b: "waive the right to V" defers (it was read as "can V"); "for my own
  benefit", "written notice" must be said about the same act (`_QUALIFIERS`, `_stretch_has`); "receives no
  consideration" and "avoid V-ing" negate; "it" right after a verb is an object ("translating it, modifying the
  size"), not the subject; the survival formula needs the obligations as the question's subject (not "notify ... when
  obligations survive"); "sell or transfer" takes "transfer" literally (not as disclosure). Diff on all 11 open sets:
  only the targeted rows changed, plus 2 newly right, no right answer lost.
- **Determinism bug fixed:** the extra-lexicon lemma table depended on the process's hash seed ("fees" was a form of
  "fee" in some processes, its own word in others), so an answer could change between restarts. Now file order,
  shortest base first; answers identical over 3 seeds on all open sets (24,600 rows); test covers 8 seeds.
- System (Pre-Tier 0 + network), answered / wrong, P1 (p2 in live order) -> p3 (`runs/sys_p3_routed.json`):
  v7all 269/11 -> 238/0 (c4 live: 263/9); v6ball 224/4 -> 207/0; v5all 595/6 -> 565/4; v4all 507/4 -> 472/2; genval
  537/0 -> 531/0; freshA 732/2 -> 727/2; freshB 652/1 -> 647/1; dev 1373/7 -> 1351/6; adv 21/0 -> 21/0. Held-out
  (report only) c4 live 1753/10 -> p3 1710/10 (Pre-Tier 0 1196/3 -> 1102/2). Generated user-style sets (v4-v7):
  errors 25 -> 6 for 7% fewer answers; LegalBench-style sets: about even.
- Tests 720 pass (+8; test_determinism's byte-identical check fails as before, on old commits too).
**Open:** a new sealed set to check p3 blind (v7 / v6b are open now; ~1 h of throttled Fireworks), top up
OpenRouter for more eval sets.
**Next levers (ranked):** (1) consumer/ToS documents: both tiers are weak there (Pre-Tier 0 ~1% on ToS category
questions; network 88% precise on ToS-generated questions vs 99.8% on LEDGAR) -> a network round with ToS data;
(2) rewording robustness of the NDA formulas (26% -> 11.5% on unseen wordings); (3) a literal-labeled whole-contract
set (CUAD's category gold understates literal yeses); (4) passive / noun / two-action question shapes.

## NEXT STEPS: Pre-Tier 0 coverage (review of 2026-10-01 evening; nothing built yet)
The user asked how much further Pre-Tier 0 can go, and whether it can be a highly accurate mechanical QA engine.
**Answer:** it's already highly accurate (99.6–99.8% on what it answers). It can be broad only for a fixed list of
questions about clauses that contracts word in standard ways (a checklist), not for any question. Speed isn't the
limit (~0.2 ms per question, ≤8 ms on 300 KB, so 10× more rules fit). The limit is how fast rules can be written
and each one proven to keep precision at ~99.5%.

**Ceiling** (the teacher LLM's quoted sentence = the deciding sentence; dev + fresh A only, held-out not touched;
script: `/root/zadumai_nli_proto/extensive/v3/ceiling_gap.py`):
| | LegalBench dev (41 questions, tuned on) | fresh A (246 questions) |
|---|---|---|
| text says "yes" outright | 33.7% | 69.7% |
| text says "no" outright | 1.7% | 18.2% |
| "no" only because nothing mentions it | 48.3% | 3.2% |
| Pre-Tier 0 answers today | 12.0% (34% of the "yes" rows) | 7.7% (10%) |
- Half of LegalBench is "no, this clause isn't about that". No sentence states it, so no reading engine answers it.
- Fixed wording does well: governing law reaches 82% of its ceiling, anti-assignment 81%, expiration date 67%.
  Ideas written many ways do badly: cap on liability 14%, revenue sharing 7%, exclusivity 2%, IP assignment 0%.

**Why it misses "yes" rows it could answer** (share of all rows, dev / fresh A):
- the deciding sentence lacks a word or concept from the question: 16.0% / 41.0%
- it can't parse the question's shape: 0.3% / 13.2%
- every word is there, but the structure check rejects it: 4.2% / 5.4%
Typical misses: "keep the existence of this agreement confidential" isn't matched to "prohibited from disclosing
the existence"; in "Borrower shall use the Information solely…" neither the receiving party ("Borrower") nor the
confidential information ("the Information") is found; "Confidential Information does not include information
lawfully obtained from a third party" isn't read as permission to obtain it.

**Plan, in order** (tune on v2 dev only; report held-out; keep each change only if precision holds at ~99.5%):
0. **Build a new sealed yes/no test set first.** None is left: fresh B was read after the final run. Without a new
   one, 99.5% becomes a number we tuned toward.
1. **Parse the question with qtree.** `_action_frame` (`frames.py:412`) needs the question to start with
   "does/can/must…" and name the actor before the verb. qtree already strips leading conditions ("If I cancel
   today, will I…") and finds subject, verb and object. Four question shapes are 13% of fresh A and get 0% today:
   notify-if-required-by-law, passive "Must CI be identified…", two verbs ("retain copies after returning or
   destroying") and "reverse engineering". The consumer contracts questions also get 0%, and they're the closest to
   how real users write. Low risk.
2. **Work out who and what the document's names refer to:** the receiving party ("Borrower", "VENDOR", we/you),
   what "the Information" or "the Material" means, and what "it" or "such party" points to. About 30% of the missing
   concepts on fresh A. Helps every question, most of all on whole contracts.
3. **A small, hand-checked set of legal equivalences:** "keep X confidential" = "not disclose X"; "nothing herein
   shall be construed as granting any licence" = no licence; "CI does not include information obtained from a third
   party or developed independently" = the receiving party may do that. The two "similar information" question
   types are at 0–1% today, though the text answers 95–98% of them outright.
4. **Grow the word list from the misses.** The teacher's quote shows the sentence and the missing phrase; keep a
   phrase only if dev precision holds. Upper bound: matching the 25 most often missing concepts gets 11% of dev rows
   and 37% of fresh A rows past word matching, before the structure checks. That curve is steep only because
   benchmarks repeat a few questions thousands of times; real users ask thousands of questions once each.
5. **Read sentence structure with the grammar parser** (passives, "undertakes: (a)… (b)…" lists, "Not disclose…").
   Covers the structure-rejected rows, ~4–5%. The biggest change: a rewrite of `_judge` (`frames.py:706`); spaCy is
   already loaded.
- Don't add more "no" answers beyond explicit bans: with the condition check lifted, its "no" answers were right
  only 40–55% of the time.

**Expected gains** (rough estimate, not measured): items 1–4 might take fresh A from 7.7% to ~15–25% and dev into
the high teens at the same precision, much of it from working through the 17 NDA questions one by one. Question
types nobody tuned for stay low (today's unseen LegalBench tasks: 3.8%). For scale: ~2 days of rule work took
held-out coverage from 5.8% to 8.7%; one overnight run of the reader network added 4.8% on held-out and 7.9% on fresh
B, at 99% or better. Item 5 and beyond turn it into a hand-built parser and reasoner. Approaches like that have
historically stalled, because every new idea needs someone to write its rule.

**Design that scales:** learned parts find the passage and match paraphrases. Pre-Tier 0's rules check party,
may/must, negation and conditions, and explain the answer (the Pre-Tier 0 + reader network setup that's live now).
Knowledge that proves stable gets turned back into rules (items 3–4). If Zadum offers fixed checklists per document
type (the 17 NDA questions, CUAD's 41 categories), the mechanical engine can cover a lot of them. Free-form questions
stay mostly with the network and Tier 2.
**Recommended start:** step 0 (new sealed set), then item 1. Waiting for the user's go.

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

- [x] "The document decides" DEPLOYED 20:09 UTC (the user picked option 2 for "or" questions the wording
  doesn't settle). qtree: a boolean with one "or" gets `maybe_options` unless it's a permission/obligation/
  presence question (`_EITHER_READING`: can/must/may/will..., entitled/prohibited/required/allowed/right of...,
  "is there", "any", "either"), or an option names nothing ("first" / "straight"). Harness: before the yes/no
  cascade, `choice.answer(..., llm=None)` on those options; if the clauses about it state exactly one → path
  `choice` (qtree reported as leaf choice). choice.py: prepositions don't identify an option ("via an online
  gallery" states "through an online gallery"); if the strict clause match settles nothing, one retry without
  the question's subject, the document's party names and period words (passive clauses drop the agent).
  Results: choice dev coverage 90.8→96.1%, test 95.1→96.7%, test2 89.6→91.0%, precision unchanged
  (97.3 / 100 / 96.8%); LegalBench "or" rows touched: 0 of 280 ("irrevocable or perpetual" is the only
  wording that qualifies); adversarial: none. 256 tests pass. Committed with Tier 2.

**WHERE THINGS STAND (2026-09-30 23:45 UTC)** — M0–M4 done and live; M5 not justified by the data; Tier 2 live.
| | precision when answering | coverage | set |
|---|---|---|---|
| classifier (8 leaves) | accuracy 98.5% | — | blind generated_test3 (459) |
| judgment/request exits | recall 100% / 100% | — | blind generated_test3 |
| span (facts) | 124/124 = 100% | 86.7% | blind spans_test2 (8 new doc types) |
| span (facts) | 101/103 = 98.1% | 50.0% | blind cuad_blind2 (60 real contracts) |
| span + Tier 2 | 137/137 = 100% | 95.8% | spans_test2 (blind for Tier 2) |
| span + Tier 2 | 133/141 = 94.3% (4 Tier 2 misses, all defensible) | 70.0% | **fresh cuad_blind3** (span alone: 95/99 = 96.0%, 50.0%) |
| choice | 60/62 = 96.8% | 89.6% | blind choice_test2 |
| Pre-Tier 0 yes/no | 718/720 = 99.7% | 5.8% | LegalBench held-out (reused many times) |
Ideas not done: choice→boolean misroutes (~13%); span coverage on long documents (loose pools were 77% right
there, so they defer); ENTITY/PROPERTY/QUANTITY/LOCATION facts and non-party "who" answers (Tier 2 could read them,
unmeasured); Tier 2 for deferred choice questions; a fresh LegalBench held-out set.

**TIER 2 BAKE-OFF (2026-09-30; led to TIER 2 LIVE below; `/root/zadumai_nli_proto/qtree/bakeoff.py`, `verify.py`)**
Reader: the LLM copies the answer verbatim from one of ≤8 selected clauses, or null; code checks the copy is
in the clause; then Jev noul checks the clause states that answer to the question. It runs only when today's
span tier defers. A key-term check on the cited clause (`bakeoff.about_the_question`, "+about") was too strict
(coverage 81→55%); Jev's check replaced it.
- Prices per 1M tokens (2026-09-30): Fireworks gpt-oss-120b Standard $0.15 in / $0.60 out, Priority $0.18 / $0.72;
  Gemma 4 26B-A4B via OpenRouter→NextBit $0.0765 / $0.255 (+5.5% OpenRouter credit fee). NextBit has no
  priority/fast tier. ~540 tokens in, ~60 out per Tier 2 call.
- Local CPU is too slow (4 cores, prefill ~45 tok/s → 10–16 s p50): Qwen3.5-4B, Gemma 4 26B-A4B, gpt-oss-20b.
  SaulLM-7B was worse and slower; Lawma-8B outputs choice letters only. GLM 5.3 was less precise.
- Finalists at 320 questions (`bakeoff.py 320`: 160 CUAD contracts, 221 answerable, 99 not). Today's system alone:
  164/165, coverage 74%. Combined with today's system:
  | reader | Jev check | precision | coverage | latency p50 / p99 | $ per 10k questions |
  |---|---|---|---|---|---|
  | gpt-oss-120b, Fireworks Priority | ≥ 0.7 | 202/204 = 99.0% | 91% | 0.33 / 0.74 s | 1.42 (Standard 1.18) |
  | Gemma 4 26B-A4B, OpenRouter pinned to NextBit | ≥ 0.8 | 199/202 = 98.5% | 90% | 0.52 / 1.15 s | 0.46 |
  Thresholds were picked on the first 160; on the unseen second 160 gpt-oss added 18 right, 0 wrong; Gemma 16
  right, 1 wrong (an exhibit's price-list date). Both share one arguable miss ("the date on which the Parties sign").
  Standard vs Priority on the same 145 calls: same replies; p50 0.35 vs 0.33 s, p99 0.82 vs 0.74 s.
- OpenRouter `:nitro` for Gemma: all 286 calls went to Makora; p50 0.32 s but p99 2.7 s (NextBit 1.15 s); same quality.

**TIER 2 LIVE (since 2026-09-30 23:16 UTC)** — gpt-oss-120b on Fireworks, the user's pick.
- Switch: **/admin → Routing Pipeline → Tier 2 reader** (no restart; see TIER 2 SWITCH and ROUTING PIPELINE). The
  flag `ask_ui.py --tier2 priority|standard` in /etc/systemd/system/zadum-router.service is now only the startup
  default, used until an admin picks a reader; `priority` since 23:40 UTC (`standard` 23:16–23:40).
- What runs: harness `_span` → `_read` when the span tier defers a fact of a type in `spans.ANSWERED_TYPES` (the
  types the bake-off measured), Jev reader only (Kev keeps documents local). Reader copies from ≤ 8 clauses
  (`select_clauses`) → verbatim check (`grounded`) → `fits` → Jev noul ≥ `CHECK_MIN` 0.7 (`CHECK` wording, shared with
  qtree/verify.py). Path `reader`; confidence = Jev's check; `Answer.reader` holds the attempt. A Fireworks outage or
  Jev error → the question defers (never a 502). Key: FIREWORKS_API_KEY from env / repo .env / ~/.env.
  UI: head "Tier 2 reader", cost line "+ 1 Tier 2 read (ms)". With Tier 1 (Jev) off in /admin, Tier 2 also
  decides the yes/no questions Jev would have answered (`reader.decide`, path `llm_decide`, head "Tier 2,
  unchecked") and reads facts without Jev's check — neither path was measured by the bake-off, which scored
  fact reading with the check on.
- **Service-tier A/B (the user's plan):** priority first, later standard for a week, then compare on live traffic.
  Requests where Tier 2 ran log in /var/lib/zadum-router/usage.db: `tier2` (standard|priority, as requested:
  Fireworks doesn't echo it), `tier2_ms` (round trip from this server), `tier2_server_ms` (Fireworks' own queue +
  compute, its `Fireworks-Server-Processing-Time` header), `tier2_result` (answered / not stated / dropped /
  unavailable), `tier2_usage` (tokens, ttft_s, attempts). Rows before 23:40 have NULLs. Compare with percentiles:
  `SELECT tier2, COUNT(*), AVG(tier2_ms), AVG(tier2_server_ms), SUM(tier2_result LIKE '%unavailable%') FROM requests
  WHERE tier2 IS NOT NULL GROUP BY tier2;`
- `reader.fits` (added after cuad_blind2 showed 93.0%): the answer must be the type asked for (a date, a duration...;
  redacted "[***] days" and capitals count); for the document's own date (`spans.document_date_kind`) the answer's
  sentence must name the document, with no other event before it (`spans.other_event`: terminate/assign/transfer/
  appoint..., skipping "this amendment" and "unless ... terminated,"), and the answer must say more than the question.
- Results, today's system → + Tier 2 (`qtree/eval_tier2.py SET`, strict CUAD scoring):
  | set | precision | coverage |
  |---|---|---|
  | bakeoff 320 (dev) | 164/164 → 201/203 = 99.0% | 74.2 → 91.0% |
  | spans_test2 (blind) | 124/124 → 137/137 = 100% | 86.7 → 95.8% (measured before `fits`) |
  | cuad_blind2 (seen: `fits` was designed after its errors) | 101/103 → 152/157 = 96.8% | 50.0 → 75.2% |
  | **cuad_blind3 (fresh, 60 new contracts, run once)** | 95/99 = 96.0% → 133/141 = 94.3% | **50.0 → 70.0%** |
  Tier 2's 4 misses on cuad_blind3 are all defensible readings CUAD scores wrong: "commencing on the date of
  execution by both Parties" / "as of the latest date referenced on the signature page" (gold = the signature date),
  "when two or more counterparts have been signed..." and "60 days" notice of intent not to renew (no CUAD label).
  Latency: reader call p50 ~0.29 s, Jev check ~65 ms. Today's own misses on cuad_blind3: 4 (3 false fires).
- Dec 31 fix (spans `_preamble_dates`): a date after "terminate/renew/assign... this Agreement, effective as of" is
  that event's date, not the start. cuad 167/171 → 167/170; other sets unchanged. Reason text "effectiv date" fixed.
- 2026-10-01 DEPLOYED (not committed): "who can review/audit the books?" over "Licensee shall have the right to
  audit the books" went to Jev (req_6071448f…). spans `_doer`: a verb without a subject of its own → "the right to
  X" (owner / the verb's subject; for grant/give/provide only a cleanly parsed recipient, else defer), "entitled to
  X", "permit Licensee to X" (the object); any "not" on the way → defer. SPAN_SYNONYMS: asked review/check →
  audit/inspect/examine (one way); audit = inspect = examine as in the lexicon. All eval sets unchanged (none had
  this pattern); 302 tests. Still open: Pre-Tier 0 yes/no "Can the licensee review the books?" doesn't fire
  (lexicon has no review≈audit; changing it needs the LegalBench A/B).
- Ideas: redo cuad scoring by hand for textual answers vs dates; ENTITY/LOCATION facts through Tier 2 (not measured);
  Tier 2 for deferred choice questions; the next fresh set is cuad_blind4 (228 unused contracts left).

**TIER 0 READER NETWORK (started 2026-10-01 ~06:45 UTC; LIVE since 15:51 UTC, committed)**
DEPLOYED 2026-10-01 15:51 UTC: zadum-router restarted with the user's OK, so Pre-Tier 0 step 1, the WITHOUT_CAUSE
fix and the reader network are live. The `tier0net` switch in /admin was off at restart, for the user to turn on.
Before the restart the switch snapped back to off: /admin re-reads admin.html from disk on every request, so a new
switch shows before the running server knows it. Saved stages at restart: Tier 0 (local NLI) off; the network
doesn't need it (Pre-Tier 0 -> network is the measured configuration). The weights (`legalbench_map/models/`,
279 MB) are gitignored and NOT pushed: copy them from /root/zadumai_nli_proto/reader_net/models/gpu/ (run xv4).
The user asked for the best possible version of "step 3": a small network finds and reads the passage, Pre-Tier 0's
tagger checks what it read (party, may/must/not, conditions), any failed check defers. Step 2's audit (what each
check blocks, right vs wrong) is folded in as its calibration. Workspace: `/root/zadumai_nli_proto/reader_net/`.

FINAL STATUS (2026-10-01 11:35 UTC) — read this first
DONE overnight, nothing deployed or committed (needs the user: restart zadum-router, turn on `tier0net` in /admin,
commit). GPU pod STOPPED at 11:20 (from inside, pod-scoped key; still listed in the Runpod account: terminate it
there if not needed; nothing on it is needed, everything was copied back and checksummed).
- INSTALLED (working tree): the reader network `legalbench_map/models/reader_net/` (run xv4: DeBERTa-v3-xsmall
  distilled from a DeBERTa-v3-large teacher on data v4; PROVENANCE.txt; gitignored) and router/netreader.py set to
  T_YES 0.935 (one-unit texts), T_YES_DOC 0.94 (documents read whole, <=12 units ~ 7 KB), T_YES_LONG off (longer
  documents return at once, ~2 ms, instead of embedding the contract), "no" off. The /admin switch `tier0net`
  stays off until an admin turns it on; ask_ui loads the network at startup. Latency per question (4 threads):
  44 ms clause, 71 ms lease, 343 ms 5 KB text. In-process router check (Tier 1/2 off): answers a fresh A clause
  via path tier0net in 114 ms, defers adv-38 (discretion), a scope near-miss, and a 145 KB contract (644 ms).
- RESULTS of the pre-registered configuration (Pre-Tier 0 first, then the network on what it leaves):
  | set | Pre-Tier 0 | network (xv4) | both |
  |---|---|---|---|
  | **fresh half B (sealed, run once)** | 191 = 7.9%, 97.4% | **191 = 7.9%, 100.0%** | 382 = 15.8%, 98.7% |
  | fresh half A (calibration) | 185 = 7.7%, 99.5% | 218 = 9.1%, 100.0% | 403 = 16.8%, 99.8% |
  | devx (dev rows not trained on) | 700 = 12.8%, 99.4% | 285 = 5.2%, 99.3% | 985 = 18.0%, 99.4% |
  | held-out (report-only) | 1,087 = 8.7%, 99.8% | 596 = 4.8%, 98.8% | 1,683 = 13.5%, 99.5% |
  | held-out seen / unseen tasks | 12.2%, 100% / 3.8%, 99.0% | 5.9%, 98.4% / 3.3%, 100% | 18.1%, 99.5% / 7.0%, 99.4% |
  | adversarial (84) | 11, 100% | 9, 100% | 20 = 23.8%, 100% |
  | genval (generated, held-out provisions) | 118, 78.8% | 449 = 18.5%, 100% | 567 = 23.3%, 95.6% |
  | conval (near-misses, none "yes") | 27 fire (all wrong) | 2 false yes (0.2%) | |
  **sealed cuad_blind3 (60 contracts, 2,280 questions), network at T_YES_DOC 0.94 if long documents were on:** 56
  "yes" (2.5%), 67.9% right by CUAD, 89.3% yes by the teacher reading the passage, 3.6% wrong by both (Pre-Tier 0:
  168, 87.5% by CUAD). Hence T_YES_LONG off. Base option bv3e (~3x slower: ~130 ms clause, ~1.2 s 5 KB) at
  0.94/0.945: fresh B 236 = 9.8% 100%, held-out network 8.3% 99.1% (both 17.0% 99.5%), cuad_blind3 96 yes 76.0% by
  CUAD, 3.1% wrong by both. Weights for both and the teacher (lv3c): /root/zadumai_nli_proto/reader_net/models/gpu/.
- Pre-Tier 0 changes waiting for the restart: step 1 ("yes, with a condition") + the WITHOUT_CAUSE fix (below). On
  fresh B step 1 added 85 answers at 97.6% (Pre-Tier 0 before step 1: 97.2% there); 2 of Pre-Tier 0's 5 fresh B
  errors are step-1 conditionals ("including ... the following:" quoted as a condition; "may only make such copies
  as are expressly authorised" read as a conditional yes, arguable). Pre-Tier 0 misfires on generated questions
  about other contract types (genval 78.8%, conval 27 wrong) before and after step 1: worth a look.
- Known cosmetic issue: condition_text counts "after" as a condition cue, so "Do the confidentiality obligations
  survive termination?" gets "yes, with a condition: after the termination of this Agreement ..." (answer right).
- Suggested next steps for the user: (1) restart zadum-router (test on :8777 first) to ship Pre-Tier 0 step 1 +
  fix and load the network; (2) turn on `tier0net` in /admin; (3) decide xsmall (installed) vs base (more coverage,
  ~3x slower); (4) commit. Not done: Kev path untested with the network on; ONNX speed-up; "no" answers.

HISTORY OF THE OVERNIGHT RUN (newest results above). The user went to bed at ~08:30 asking: keep going through every step (evals, fixes, new evals, more cycles),
checkpoint and update this file after each step, stop the GPU pod when done. Nothing deployed or committed; ask
before restarting zadum-router, turning on `tier0net`, or committing.
- GPU pod (Runpod RTX 4090, 24 GB; user's account; BILLS WHILE RUNNING, no volume, so Stop wipes it):
  `ssh -i ~/.ssh/runpod_ed25519 -o IdentitiesOnly=yes -p 40041 root@213.192.2.102` (direct TCP; the pod's
  authorized_keys was "null", fixed through the ssh.runpod.io proxy). Work dir /workspace/rn, venv /workspace/venv
  (system torch 2.8+cu128, transformers 5.17.0 as here). `job.sh NAME DATA [train.py args]` trains models/NAME and
  scores pairs/*.jsonl into probs/NAME; logs logs_NAME.txt; queue2.sh starts lv2 after bv2. To stop the pod from
  itself: the pod-scoped key is in /proc/1/environ (RUNPOD_API_KEY); `runpodctl stop pod 3bpkzc4u7w36u4`.
- train.py runs on GPU too (bf16 autocast, `--train-emb`, `--soft-alpha` distillation, `.tmp` checkpoints ignored
  on resume); xsmall 0.06 s/step there vs 1.31 s here. The CPU run (models/v1) was stopped at 08:21 (step-1000/1500
  checkpoints kept); the GPU's x1 is the same recipe.
- Tools: `score.py dump SETS` -> pairs/, pod `infer.py` -> probs/NAME, `pull.sh NAME` -> scores/NAME;
  `compare.py M...` (per model: lowest yes threshold reaching 99.5% on fresh A after Pre-Tier 0 + checks, "no"
  off; every set at it; cells = network's answers on rows Pre-Tier 0 leaves as share of all rows, precision);
  `docs.sh M` = CUAD whole-contract test in rounds (eval_docs.py); `docs_judge.py M` = teacher's verdict on M's
  contract-level "yes" answers (CUAD gold says "a clause of this category was annotated", the teacher reads the
  passage literally); conval = contrast questions of the val split (none is "yes": any "yes" is wrong).
- Data: v1 (86k pairs); v2 = v1 + 26.6k contrast pairs (gen_contrast.py: 8,000 "yes" questions rewritten to
  change party/scope/detail/force, blind-checked; 117k pairs).
- Results (clause level; label smoothing caps p at ~0.967):
  | model | t_yes | fresh A | held-out | adv | genval | conval false yes | contracts (CUAD gold) |
  |---|---|---|---|---|---|---|---|
  | x1 xsmall v1 1 ep | 0.962 | 21.5% 99.6% | 10.4% 98.5% | 2 wrong /12 | 22.9% 99.8% | 9.1% | 61.8% right @0.96 |
  | x3 xsmall v1 3 ep | none | max 99.3%: more epochs = confident errors | | | | | |
  | b2 base v1 2 ep | 0.97 | 26.9% 99.7% | 12.1% 99.3% | 10/10 | 23.6% 100% | 7.6% | 62.8% @0.96 |
  | l2 large v1 1 ep | (99.4% max) | 27.2% 99.4% @0.968; 37.3% 99.2% @0.96 | | | | | |
  | xv2 xsmall v2 1 ep | 0.955 | 18.6% 99.8% | 9.3% 98.2% | 9/10 | 20.2% 99.6% | 0.4% | 72.7% @0.96, 85% @0.97 |
  CPU latency per question (4 threads): xsmall 45 ms clause / 345 ms 5 KB / 963 ms 15 KB contract; base ~3x.
- PROBLEM FOUND: whole contracts. On CUAD test contracts the network's "yes" is far below the clause-level
  precision. Teacher check of xv2's 216 contract "yes" (t 0.96): 124 right by both, 26 literally right but not
  annotated by CUAD, 33 (15%) wrong by both (passages near the question that don't answer it: "terminate upon
  material breach" for "without cause", "shall not use the Marks" for "disparaging", exceptions to non-compete).
  Pre-Tier 0 itself is 85.6% right vs CUAD gold on these contracts.
- Fix in progress: `docs_mine.py`: passages retrieved from the 228 CUAD contracts no eval uses (CUADv1 minus
  test.json minus cuad_blind/2/3), for the 38 CUAD questions reworded by the teacher; xv2 scores them on the pod;
  the teacher labels those xv2 finds convincing (+ the top one and a random one per contract-question; "yes"
  confirmed blind); passages sharing text with any eval set dropped -> v3 data. Then large on v3 as the teacher,
  distill into xsmall.
- 09:45 mined data done: docs_mine.py retrieved 99,786 passages (8,664 contract-questions; bge-small vectors
  computed on the pod's GPU in 36 s, identical to CPU), xv2 scored them, the teacher labeled 21,909 (those xv2
  found convincing + top + random; 80 calls/s with 160 workers), "yes" and "no" re-checked blind. 8,734 passages
  share 10 words in a row with an eval text (LegalBench's CUAD rows and the fresh sets quote CUAD contracts) and
  were dropped: 12,957 kept (yes 708, no 694, doesn't settle 11,555) -> mine_train.json.
- LEAK FOUND AND CLOSED (09:48): 17,401 of v3's 130k training pairs (13%; also in v1/v2) share 10 words in a row
  with an eval text: LEDGAR and CUAD are both SEC filings, LegalBench's CNLI rows reuse each NDA across many
  hypotheses (most dev rows), its CUAD rows quote contracts; ~7,900 touch sealed fresh B. Part is boilerplate.
  `build_train.py --clean` drops them: v3c = 112,640 pairs (dev rows 6,245 -> 1,973). Models x1/b2/l2/xv2/bv2
  trained with the overlap, so their held-out/fresh A numbers may be a bit flattering; the final model is
  trained on clean data only.
- GPU now (09:50): lv2 (large v2) finishing; xv3c (xsmall v3c) training; queue3.sh then: lv3c (large v3c,
  teacher) -> soften.py (its probabilities on v3c train -> data/v3cd) -> xv3cd (xsmall distilled, --soft-alpha
  0.5). ETA ~10:45.
- bv2 (base v2 2 ep): fresh A 35.4% @99.5% (t 0.96), held-out 16.8% @97.8%, adv 1 wrong/12, conval 1.1%.
- 09:33 the pod's 30 GB disk filled (lv2 crashed saving a checkpoint; abandoned): finished runs' checkpoints
  deleted (final weights kept on the pod; copies of x1/x3/b2/xv2/bv2/l2 in reader_net/models/gpu/).
- STEP 1 REGRESSION FOUND AND FIXED (frames.py, not deployed): on whole CUAD contracts the teacher found Pre-Tier
  0's "yes" wrong by both CUAD and the teacher in 10.7% (pt0_docs_judge.py), 9 of 24 on "Can a party terminate
  the agreement without cause?": step 1's "yes, with a condition" made "Either party may terminate at any time
  ... if the other party fails materially to comply" a yes, but there the condition is the cause. Now a
  WITHOUT_CAUSE question defers on a condition unless the sentence itself says "without cause", "for
  convenience", "for any reason", "at will" (`_EXPLICIT_WITHOUT_CAUSE`), and "by mutual consent / written
  agreement" (the parties together) defers (`_MUTUAL`). Tests 136 pass (3 new). Pre-Tier 0 (with step 1):
  dev 905 -> 896 answers, 99.6% both; held-out 1,097 -> 1,087, 99.8%; fresh A unchanged; contracts 300 at
  85.3% -> 290 at 86.9% right vs CUAD. (pt0_<set>.json and docs_pt0.json re-dumped; the old ones are in
  pt0_before_without_cause/.) Also seen: Pre-Tier 0 is 78.8% right on genval (generated questions) and fires on
  27 conval near-misses (all wrong), mostly "yes, with a condition" from sentences about something else; noted
  for later, Pre-Tier 0's own eval sets are unaffected.
- 10:00-10:45 clean-data models (all trained on v3c or later; numbers comparable):
  | model | how | fresh A @99.5% | devx | held-out | adv | genval | conval | contracts (yes, wrong by both) |
  |---|---|---|---|---|---|---|---|---|
  | xv3c | xsmall v3c | t .95: 13.9% 99.7% | 99.3% | 6.5% 98.4% | 1/11 wrong | 99.2% | 0.3% | t .96: 82, 4.9% |
  | lv3c | large v3c (teacher) | t .95: 23.7% 99.6% | 98.0% | 12.2% 97.8% | 2/11 | 99.8% | 0.2% | t .96: 214, 5.1% |
  | xv3cd | xsmall distilled from lv3c (alpha .5) | t .91: 18.3% 99.5% | 97.4% | 9.3% 97.5% | 1/10 | 99.6% | 0.8% | t .94: 49, 2.0% |
  | xv3e | xv3cd + 51k contract passages labeled by lv3c | t .92: 15.5% 99.7% | 98.0% | 9.7% 97.8% | 1/11 | 99.6% | 0.5% | t .94: 78, 1.3% |
  devx = the 5,486 dev rows v3c doesn't train on (score.py devx): LegalBench-like, allowed for calibration and
  error reading; it tracks held-out closely. Distillation lowers the probabilities, so thresholds differ by model.
  Contracts need a stricter threshold than clauses: NetReader now has `t_yes_doc` (multi-unit documents; test).
  `docs_sweep.py M` = contract-level threshold table with the teacher's verdicts (cached in llm/docs_judge.jsonl).
- Lead candidate: xv3e, t_yes 0.92 (one-unit texts), t_yes_doc 0.94 (contracts). devx errors (13 of ~660, all
  CUAD tasks): half scope near-misses ("insurance limits" for a liability cap, license grants for "exceptions to
  non-compete", "exclusive right to use" for exclusive dealing), half CUAD-category gold vs a literal reading.
- NEXT (10:50): one more data round aimed at those near-misses: LEDGAR clauses nearest each CUAD question
  (embeddings), the ones the network finds convincing labeled by the teacher LLM -> v4; bv3e (base, v3e) is
  training on the pod for the speed/accuracy option.
- 10:40-11:00 last data round: ledgar_mine.py (LEDGAR clauses nearest each CUAD question; 2,549 LLM-labeled:
  yes 556, no 138, doesn't settle 1,855; xv3e's "yes" at 0.92 agreed with the LLM on 95.3% of these) -> v4 =
  v3e + those x3, teacher soft labels -> xv4. Also bv3e = base on v3e (distilled).
- Thresholds now must reach 99.5% on fresh A AND 99% on devx (compare.py --also devx:0.99); contracts use
  t_yes_doc picked on the CUAD test contracts for ~0-2% wrong by both (docs_sweep.py). "No" stays off: on devx
  the network's "no" is 0-11% right (LegalBench CUAD semantics), fresh A 80-91%.
  | model | t_yes | fresh A | devx | held-out (report) | adv | genval | conval | contracts t_yes_doc: yes, wrong by both |
  |---|---|---|---|---|---|---|---|---|
  | xv4 (xsmall) | 0.935 | 9.9% 100% | 5.3% 99.3% | 4.9% 98.7% | 9/9 | 18.5% 100% | 0.2% | 0.94: 75 (1.9%), 0.0% |
  | bv3e (base, ~3x slower) | 0.94 | 11.7% 99.6% | 9.1% 99.4% | 8.4% 99.1% | 9/9 | 20.2% 100% | 0.5% | 0.945: 143 (3.7%), 1.4% |
  (cells: network's answers on the rows Pre-Tier 0 leaves, share of all rows and precision)
- PRE-REGISTERED FINAL (11:05, before any sealed set is scored): primary = xv4 with t_yes 0.935 (one-unit
  texts), t_yes_doc 0.94 (several units), "no" off, conflict 0.5, k_lex 6 + k_emb 6, checks as now (incl.
  discretion), Pre-Tier 0 as now (step 1 + without-cause fix). Option = bv3e with t_yes 0.94, t_yes_doc 0.945.
  Sealed sets run once each for both: fresh half B (clause level) and cuad_blind3 (60 contracts). The choice
  between them is the user's (speed vs coverage), not made on the sealed numbers.
- Fix: netreader.check() "discretion" (adv-38), tests 22 pass. PROTOCOL SLIP: at 08:45 13 of b2's held-out errors
  were printed by mistake; nothing changed because of them. Held-out stays report-only.
- Weights copied back here: models/gpu/{x1,x3,b2} (the pod's models/ has all; copy before stopping it).
- **Step 1 done (not deployed: needs a zadum-router restart, ask first):** Pre-Tier 0's condition veto now turns a
  "yes" into "yes, with a condition" quoting it (`frames._conditional`, `condition_text`, `FrameResult.condition`);
  a "no" with a condition still defers. Not for property frames (a condition before the property can undo it),
  and not "discretion" in a presence frame (adv-38: "The Company may, in its sole discretion, pay a bonus" is not
  an entitlement). Answers stay "yes"; the reason and the evidence label carry the condition. Tests 111 pass.
  | set | before | after |
  |---|---|---|
  | LegalBench dev | 729 = 9.8%, 99.6% | 905 = 12.1%, 99.6% (1 new error of 176: a frame match) |
  | held-out | 811 = 6.5%, 99.8% (unseen tasks 1.6%) | 1097 = 8.8%, 99.8% (unseen 3.8%, 99.0%) |
  | fresh half A | 104 = 4.3%, 99.0% | 185 = 7.7%, 99.5% |
- Teacher: gpt-oss-120b. Bulk calls go through OpenRouter pinned to bf16 providers (DeepInfra first; `llm.py`),
  NOT Fireworks: Fireworks' 45k generated tokens/min limit is per account and the live Tier 2 shares it (a run
  there hit 429s at 06:56). On 100 dev rows its "yes" was right 41/41; generated "no"s are often silence read as
  "no", so every generated question is re-answered blind (`prompts.VERIFY`) and kept only if both agree.
- Data: `llm/dev_label.jsonl` (all 7,459 dev rows labeled; LegalBench CUAD gold means "the clause is about X",
  e.g. "Revenue shall not be shared" is gold yes for "Must a party share revenue?", so its "no"s are scored
  pessimistically); `gen_ledgar.py` → `gen_train.json`: 12,000 LEDGAR provisions (no eval set uses LEDGAR) × 6
  generated questions, + blind check. `build_train.py` assembles the pairs (labels 0 no, 1 yes, 2 doesn't settle).
- Code: `train.py` (CPU, ~1.5-2 s/step at batch 16), `score.py` (scores eval sets once), `analyze.py`
  (thresholds, checks audit, combined with Pre-Tier 0 from `pt0_<set>.json`), `retrieve.py` (recall on CUAD
  contracts: ~83% at ~12 units with lexical + bge-small; first question on a 50 KB contract embeds every unit,
  3.9 s; then 29 ms). Router module: `router/netreader.py` (NetReader, check(); model dir `legalbench_map/models/reader_net`).
- Fresh half B is SEALED for this project: score it only in the final run (`score.py ... freshB --final`).
- Eval protocol, fixed before any result (07:50): thresholds and error reading only on fresh half A, dev, adv,
  prior, the generated val split and CUAD's test-split contracts (whole-contract dev). LegalBench held-out is
  report-only (its errors are not read before the final run). Final, once: fresh half B and the 60 cuad_blind3
  contracts (whole-contract). Target: ≥99% right on the network's answers, threshold set for ≥99.5% on fresh A.
- 07:42 data: 71,988 generated questions over 12,000 provisions, writer and blind check agree on 59,126 (yes 23,194,
  no 12,773, not settled 23,159); OpenRouter cost $2.25 for ~21.8k calls. `build_train.py data/v1 --dev`: 86,473
  training pairs (gen 55k, the deciding sentence alone 8k, another provision type's text as "doesn't settle" 17k,
  dev rows where gold and teacher agree 6.2k; long provisions cut to ≤600-char windows around the quote, as the
  network reads documents), 101 tokens on average. Training `models/v1` started 07:43 (1 epoch, lr 4e-5, batch 16,
  8 threads at nice 10, 1.7 s/step, ~2.5 h). Calibration set: fresh half A (its questions are unseen in training
  except the 43 LegalBench ones); held-out's 11 unseen tasks are the generalization test.
- Router wiring (not live: the switch is off by default and the service hasn't restarted): /admin switch
  `tier0net` "Tier 0 (reader network)" (router/stages.py; mask gains " n1" only when on), Harness(netreader=...)
  runs it after Tier 0 (path "tier0net", Answer.netreader), ask_ui loads `NetReader()` at startup (logs and skips
  it if `legalbench_map/models/reader_net` is missing), ui.html shows it. Tests: tests/test_netreader.py (18).
  The full suite's 9 failures + 3 errors (test_phase2_runner, test_per_item_dump, test_determinism) and the 4
  phase2 collection errors (no module `downshift`) fail the same without these changes.
- Measured: int8 dynamic quantization is 30% faster but flips 8% of argmaxes: not used. bge-small embeds ~22-30
  ms per 600-char unit on this CPU (a 50 KB contract's first question ~2-3 s, then cached).

**PRE-TIER 0 + TIER 2 FIXES (2026-10-01 ~07:00 UTC; live; commits 5bf9078 + cb6c5b3)**
- Tier 2 reads every fact question the local tiers leave (any answer type; `reader.fits` checks only the measured
  types) and every why / what-if / how / unclassified question (`harness._answer_kind`); Jev's check still guards.
  Trigger: "what can the licensee do not more than once per calendar year?" was typed DATE ("calendar year") and
  dropped. qtree: "what can/must/may X do ..." is ACTION (`_ASKS_ACTION`). spans_test2 + Tier 2 133/133 (was 129/129).
- /admin's Pre-Tier 0 switch now also turns off the fact and choice rules (`spans.answer` / `choice.answer`
  `rules=`); with it off, Jev picks and Tier 2 reads.
- Topic matcher (`pretier0._topic`): word families via `_root` (terminate/termination, assign/assignment), roles kept
  apart (employer/employee, assignor/assign), terms of art whole (`_TERMS`: change of control, ROFR, MFN...). The
  old 6-letter prefix answered "Is the employer discussed?" from "The Employee shall...". Eval:
  `/root/zadumai_nli_proto/extensive/topic/eval_topic.py dev` (CUAD categories as topic questions): 20.8% answered at
  96.8% label agreement (was 18.4% at 96.4%; the disagreements are mostly mentions CUAD files elsewhere). Requiring
  the question's word pairs to stay together was tried and dropped (lost right answers).
- frames.py, from LegalBench dev misses where every concept was present: the agreement as the subject of its term's
  end (`_AGREEMENT_TERM`, +42 dev); recipients listed as the exception to a ban may receive it
  (`_permitted_by_exception`); "Recipient Party" is the receiver; an NDA that never says receiving party/recipient
  asks about any party (`_RECEIVER_WORDS`); property questions ignore cross-reference carve-outs and conditions
  after the property and its thing (+56 dev), not insurance limits as a liability cap; "shall not make more copies
  than necessary" is a limit, not a "no". Tried and dropped (no gain or net loss): singular forms of plural-only
  lexicon words; mapping nominalizations to verbs; not splitting sentences at "14.1".
  | set | before | after |
  |---|---|---|
  | LegalBench dev (tuned on) | 627 = 8.4%, 99.5% | 729 = 9.8%, 99.6% |
  | LegalBench held-out (reused) | 720 = 5.8%, 99.7% | 811 = 6.5%, 99.8% |
  | fresh half A (open) | 79 = 3.3%, 98.7% | 104 = 4.3%, 99.0% |
  | fresh half B (blind run) | 83 = 3.4%, 97.6% | 107 = 4.4%, 96.3% (then the comparative fix: 106, 97.2%) |
- **Fresh held-out set** (`/root/zadumai_nli_proto/extensive/v3/build_fresh.py`, `fresh_rows.json`,
  `eval_fresh.py --half A|B`): 4,805 rows: ContractNLI train/dev clauses (LegalBench used its test split; neutral left
  out: it means "not the evidence span", all 10 neutral "errors" read were right), its 3 hypotheses LegalBench never
  used, 164 unused CUAD clauses, LegalBench consumer_contracts_qa (396) and contract_qa (80). Split by clause into
  half A (open) and half B (blind). Half B was read after the final run above, so it is no longer blind: the next
  blind check needs another set. New question types (consumer_contracts_qa, the 3 new hypotheses) get ~0 answers.

**ROUTING PIPELINE (2026-10-01)** — /admin's second tab (was "Tier 2 reader"; `#tier2` links still open it). Merges
the laptop's commit bc11055 (tier switches, made without this box's uncommitted Tier 2 switch) with the Tier 2 switch.
- Tiers card: on/off switches for Pre-Tier 0, Tier 0 (NLI), Tier 1 (Jev) and the free classifiers, server-wide; the
  playground's Pre-Tier 0 / Tier 0 checkboxes are gone. `router/stages.py`; saved in usage.db `settings` key `stages`
  (with `updated_at`/`updated_by`, columns added to the old 2-column table on start); `POST /api/admin/stages`.
  Refused: Tier 1 and Tier 2 both off (nothing answers what the rules miss), classifiers without Tier 1.
- With Tier 1 off, Tier 2 reads facts unchecked (confidence null, "unchecked (Tier 1 off)") and decides yes/no
  itself (`reader.decide`, path `llm_decide`). Neither path was measured by the bake-off. Tested on :8777: all
  three local tiers off + gpt-oss-120b answered "$500 per animal" (reader) and "no" (llm_decide) in ~0.36 s.
- Each request logs the tiers that were on in `requests.stages`, e.g. `p1 t1 j1 c0 r:fireworks-priority`.
- `GET /api/admin/pipeline` returns both cards; `POST /api/admin/tier2` switches the reader (as below).

**TIER 2 SWITCH (2026-10-01)**
- /admin's Routing Pipeline tab has a "Tier 2 reader" card: Fireworks standard, Fireworks priority (both gpt-oss-120b, Jev check ≥ 0.7),
  Gemma 4 26B-A4B on our own GPU and Gemma 4 26B-A4B on OpenRouter pinned to NextBit (both ≥ 0.8, from the bake-off),
  or Off. A switch sends one test read first (`tier2.PROBE_*`) and keeps the old reader if it fails (409). The choice is
  saved in usage.db `settings` (key `tier2`) and beats `--tier2` after a restart. Each card shows its last 7 days of
  reads from usage.db (`Usage.tier2_stats`). Code: `router/tier2.py` (options, `Tier2`), `ask_ui.py`
  (`POST /api/admin/tier2`), `router/admin.html`. `--tier2` takes the option ids (old standard/priority still work).
- usage.db `tier2` now logs the option id (fireworks-standard, fireworks-priority, gpu-gemma, openrouter-gemma). Older
  rows say standard/priority; `usage.LEGACY_TIER2` maps them. For our GPU, `tier2_server_ms` is the gateway's time.
- **Our own GPU's API, zadum-gpu/1** (spec in the `router/gpu.py` docstring): `GET /v1/health`, `POST /v1/generate`
  (messages, max_tokens, temperature, json_schema, thinking → text, usage, timing), Bearer token. The router talks only to
  it (`gpu.OwnGPU`, wrapped by `reader.OwnGPULLM`). On the GPU box, `gpu/gpu_gateway.py` (stdlib only) serves it in front of
  the engine: adapters `llama.cpp` and `openai` (vLLM, SGLang, TGI). **A new deployment = a new Engine subclass in
  gpu/gpu_gateway.py; nothing in the router changes.** Setting a GPU up from scratch: `gpu/SETUP.md`.
- Today's deployment: Runpod RTX 4090 pod (`ssh -i ~/.ssh/runpod_ed25519 -p 12930 root@47.47.180.54`), llama.cpp
  serving Google's Gemma 4 26B-A4B Q4_0 QAT. Fix that made it batch: llama-server's host-RAM prompt cache stalled
  every request, so it runs with `--kv-unified --cache-ram 0 --ctx-checkpoints 0` (prefill went from 1.7k to 10.7k tok/s
  at 16 parallel). **Setting up a GPU from scratch, restarting one, and every flag: `gpu/SETUP.md`**: one command,
  `gpu/deploy.sh IP PORT [all|start]` (pod: `gpu/setup_pod.sh`; router-side check: `gpu/check_gpu.py`). Here:
  `zadum-gpu-tunnel.service` (SSH tunnel, 127.0.0.1:18000 → pod 127.0.0.1:8000, written by deploy.sh from
  `gpu/zadum-gpu-tunnel.service`; the pod's SSH port changes on a pod restart) and ZADUM_GPU_TOKEN in /root/.env.
- Measured from this server, 286 real reader prompts one at a time: 0.29 s p50, 0.36 s p90, 0.63 s p99 (GPU 0.25 s;
  the rest is the 33 ms network round trip); bad JSON 0. The same Q4_0 file gave NextBit's answer on 151/160 bake-off
  questions. An outage (tunnel down) defers the question in 0.2 s and the card shows the GPU as down.
- The pod bills by the hour whether used or not; stopping it makes the GPU option fail its test read (the switch refuses).

## Next steps (from before the question tree)
The current plan is NEXT STEPS: Pre-Tier 0 coverage, near the top. The items below are older.
1. ~~Commit the B + D edits~~ done in the commit "Classify questions first, then answer facts and choices from the text" (`git log`).
2. Improve **unseen-task coverage** (only 1.6–2.2%): generalize the question phrasings rather than matching per task.
   Known errors: loose date/duration pattern ("one year before"), and a successor term read as the initial term.
3. Get a **fresh held-out set**. The current one has been run 4 times and errors from it drove fixes.
4. Optional:
   - an "I am the: Licensee / Licensor" selector in the page
   - make Tier 0 faster on long documents (1.3 s p50 at 50+ sentences on 1 thread)
   - index documents at upload (option C)

## How to check things
- Tests: `cd legalbench_map && ../.venv/bin/python -m pytest -q tests/test_frames.py tests/test_pretier0.py tests/test_tier0.py tests/test_router.py tests/test_usage.py tests/test_qtree.py tests/test_tier2.py` (313 on 2026-10-01)
- Question-tree evals: `/root/zadumai_nli_proto/qtree/` — `eval_classify.py dev|test|test2|test3`, `eval_spans.py spans_dev|spans_test|spans_test2|cuad|cuad_blind|cuad_blind2|cuad_blind3 [--llm]`, `eval_choice.py dev|test|test2 [--llm]`, `m4_privacyqa.py [--llm]` (Jev calls are cached under `qtree/llm/`)
- Tier 2 evals (same folder): `eval_tier2.py bakeoff|spans_test2|cuad_blind2|cuad_blind3 [--tier priority]
  [--no-typecheck]` (today's system + Tier 2 exactly as live; reader replies cached in `results/tier2/`),
  `bakeoff.py N --models ...` (model bake-off, cached in `results/bakeoff/`), `verify.py MODEL N` (Jev-check
  thresholds). Used CUAD sets: test.json (tuned on), cuad_blind, cuad_blind2, cuad_blind3; next fresh: add cuad_blind4
  to both loaders (random.Random(17) over the 228 contracts left).
- The eval workspace `/root/zadumai_nli_proto/` is NOT in git (only `legalbench_map/` is); back it up before big changes.
- Pre-Tier 0 eval: `PYTHONPATH=. ../.venv/bin/python /root/zadumai_nli_proto/extensive/v2/evalv2.py dev|heldout`
  (split: `extensive/v2_split.json`, 11 of the 52 tasks held out as unseen)
- Tier 0 eval and adversarial set: `/root/zadumai_nli_proto/extensive/` (`run_eval.py`, `adversarial.jsonl`); prototype cases: `/root/zadumai_nli_proto/cases.jsonl`
- Deploy: `systemctl restart zadum-router`. Before that, test on a copy with `ask_ui.py --port 8777`, which has no sign-in and no quota.

## Gotchas
- Check pytest's own exit code before a deploy (`pytest ... ; echo $?`): `pytest | tail` always exits 0.
- Live API checks count against your 50/day quota (it's per account, and the `/admin` page changes it).
- Port 8766 has an old test server from an earlier session; leave it alone.
- `pkill -f <pattern>` over SSH or in a Bash call also matches the shell running it if the pattern is in the command
  line: kill by PID, or anchor the pattern (`pkill -f "^python3 /workspace/gpu_gateway.py"`).
- Stop the 8777 test server in its own Bash call, with a command line that doesn't contain its own pattern
  (`pkill -f "ask_ui.py --port 877[7]"` alone): if the same command line holds "--port 8777", pkill kills the shell.
- Git has no identity on this machine; commits use `-c user.name=... -c user.email=...` from earlier commits.
