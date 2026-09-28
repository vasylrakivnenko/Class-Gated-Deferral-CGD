# Research brief — prior art and competitive landscape for the downshift program

**Scope.** This covers everything *except* `reflex-abcd/` (skeleton-first response selection on
ABCD, which has its own brief at `reflex-abcd/RESEARCH_BRIEF.md`). This brief is the
classification-workload program: banking77, Financial PhraseBank, CLINC150, dair-ai emotion,
GoEmotions, TweetEval, ToxicChat, CUAD, LegalBench (37 + 22 tasks), FinBen, BLURB, MASSIVE (13
locales), HINT3, BeaverTails.

**Purpose.** You are a research agent. Find (a) prior work that anticipates any part of the system
below, (b) competing systems that beat its numbers on the same datasets, and (c) the strongest
published counter-evidence to its central claims. **Be adversarial: the goal is to discover that
this has already been done, not to confirm novelty.** Two of the findings below are the kind that
a single well-known paper could reduce to a replication — say so plainly if you find it.

---

## 1. What the system is, in one paragraph

A **measurement harness that decides, per workload, what share of production traffic can be served
by a free local classifier instead of a paid LLM** — and, where the answer is "some but not all",
what the routing rule should be. It runs a ladder of cheap candidates (majority baseline → TF-IDF
+ logistic regression → frozen sentence embeddings + logreg → a fine-tuned 68M-parameter encoder)
against a paid incumbent on *the same held-out items*, prices every row from measured token
counts, and then replays several deferral rules out-of-fold to find the cheapest setting that
holds a stated quality bar. The distinctive routing mechanism — the reason for this brief — is
**class-aware (label-conditional) deferral**: instead of escalating individual low-confidence
items, the system decides *per predicted class* whether the free model owns that class outright or
whether the entire class is handed to the LLM.

---

## 2. The algorithm

### 2.1 The free arm (identical across every workload)

| candidate | definition |
|---|---|
| majority / stratified-random | constant and label-blind baselines, drawn on every chart |
| **TF-IDF + logreg** | `FeatureUnion(word 1–2 grams, char_wb 3–5 grams)`, `sublinear_tf`, `min_df=2`, `LogisticRegression(class_weight="balanced")`, `C` selected by inner CV on training folds only |
| frozen embeddings + logreg | `bge-small/base/large-en-v1.5`, or `multilingual-e5-small`; encoder frozen, linear head trained |
| fine-tuned encoder | `jhu-clsp/ettin-encoder-68m` (English) / `ModernBERT-base`, full fine-tune on the task's own labels |
| zero-shot NLI | `deberta-v3-base-zeroshot-v2.0`, the no-labels control |

The **bag-of-words row is the one the headline usually rests on**, and the char-ngram half of the
union is load-bearing: on CJK locales where whitespace tokens average ~1 per utterance, the word
half collapses to near-chance (0.15) while the char half carries the locale intact (0.76–0.80).

### 2.2 The comparison protocol (this is where most published comparisons are wrong, so check it)

Five rules, each adopted after a measurement forced it:

1. **The classifier may see exactly the `{{placeholders}}` of the reference system's own prompt
   template — nothing else.** Enforced by a test pinning `input_columns` to the parsed placeholder
   set. This replaced a hand-written column blocklist that had let gold sub-answer columns through
   on six LegalBench tasks, producing ~100% accuracy that was pure leak.
2. **Paired, same-items comparison with McNemar mid-p**, not two independent confidence intervals.
   Pairing is the whole point: with n=250 and 220 agreements, the effective sample is 30.
3. **Non-inferiority testing against a margin δ**, not "no significant difference" — which a
   useless test set passes every time.
4. **Holm–Bonferroni** across the candidate family; a resolution warning printed on every run
   (at n=250, ±4.4%, so two rows closer than ~8.9% are not separable unpaired).
5. **Cost measured, never blended.** `cost_per_1k = 1000·(mean_in·price_in + mean_out·price_out)/1e6`
   with token counts from the provider's own usage counters during the run that produced the
   accuracy. Classification is ~87:1 input:output, nothing like the 50:50 blend rate cards assume,
   and reasoning models bill the `<think>` block as output (260 vs 37 output tokens for the same
   call, measured).

Evaluation is repeated stratified k-fold CV over the test split (5 folds × 3 repeats, every item
predicted out-of-fold in every repeat), with bootstrap CIs pooled over repeats.

### 2.3 The routing layer — five deferral rules, replayed offline out-of-fold

Every rule is fitted on training folds and applied to held-out items, so **no routing decision is
ever made by a rule that saw the item it routes.** All five were replayed from per-item
predictions already on disk, at $0, over 9 (task, LLM) pairs.

**(a) Per-item confidence dial** — escalate the least-confident x% of traffic; report accuracy on
what is kept. This is the "cascade dial".

**(b) Split conformal** — nonconformity `1 − p(true label)`; a 20% calibration slice carved out of
each fold's training pool; prediction set = every label under the finite-sample-corrected
`(1−α)` quantile; **keep iff the set is a singleton**, else escalate.

**(c) Committee disagreement** — the other cheap candidates vote; the signal is the number of
committee members disagreeing with the primary, `n_dis ∈ {0..K}`, optionally tie-broken by
`(1 − p_pred)`. Threshold chosen on training folds. (Query-by-committee, used as a deferral signal.)

**(d) ⭐ Class-aware / label-conditional ownership — the mechanism this brief is about:**

```
for each outer fold (train_idx, test_idx):
    for each class c that the cheap model PREDICTS on train_idx:
        m   = train_idx where cheap_pred == c
        adv = mean(incumbent_correct[m]) - mean(cheap_correct[m])
        if adv > margin:                       # margin ∈ {0, 0.02}
            escalate ALL test_idx where cheap_pred == c
```

Four properties that define it and that the search should match against:

- The partition is the **cheap model's own predicted class**, not the gold class — so it is
  available at inference time with no extra signal.
- The decision is **per-class and binary**: a class is owned outright or deferred entirely. There
  is no per-item confidence anywhere in the rule.
- It routes on **relative competence** (does the incumbent beat the cheap model *on this class*),
  not on absolute uncertainty. This is the deliberate contrast with (a)–(c).
- It therefore **requires per-item incumbent correctness on a calibration set** — one LLM pass
  over a few hundred items. That is a real deployment cost the conformal gate does not have.

A weaker variant of the same idea drives the traffic-coverage reporting across all other
workloads: **a class is "owned" iff its measured per-class accuracy clears an absolute reliability
bar** (e.g. 90%), and coverage = the share of traffic falling in owned classes. Note the trap
recorded against it: on CUAD only 3 of 29 categories clear 90% as whole categories, and those
three clear it because the clause is present in 73–86% of contracts (a constant predictor already
scores 0.85–0.92 on them), so their lift over a constant is the *smallest* in the set. **Report
lift or balanced accuracy for this variant, never positive-class F1, and never the count of
classes above the bar.**

**(e) Learned router** — two logistic regressions, `P(LLM right | x)` and `P(cheap right | x)`,
over `p_pred`, conformal set size, log length, predicted class, and a 384-d text embedding;
escalate where the predicted edge exceeds a margin.

**(f) Oracle** — escalate exactly the cheap model's errors. The ceiling.

### 2.4 The reported quantity

One number per workload: **the share of traffic routed to the free arm.** The ends count — a
workload sent 100% free is the dial at one extreme, not a different category. The only genuine
negative is *"no useful middle"*: the ends work but no intermediate setting beats them.

A two-part pre-flight gate decides whether a middle setting can exist at all, run on the row that
would actually be deployed:

| signal | gate | why |
|---|---|---|
| AUROC(confidence → correct) | **≥ ~0.75** | 0.5 means routing on confidence selects items at random |
| share of all errors in the least-confident 20% | **well above 20%** | ceiling on what a 20% budget can repair even with a perfect oracle |

**Calibration gap is explicitly *not* a gate** — an encoder 15 points overconfident routed
perfectly well. Ranking correctness and being calibrated are different properties.

---

## 3. Measured results

### 3.1 The dial, per workload

Free-arm share of traffic. `paired` = same items, against a *paid* incumbent we ran ourselves;
`published` = against the reference system's own reported number on its own protocol (weaker
evidence, labelled as such everywhere).

| workload | free arm | incumbent | free share | evidence |
|---|---|---|---|---|
| ToxicChat moderation | 0.744 F1 (toxic) | OpenAI moderation 0.237 F1 | **100%** | paired |
| banking77, 77 intents | 94.0% | Claude Sonnet 4.6 82.9%, $3.45/1k | **100%** | paired |
| Financial PhraseBank | 97.6% | 98.8%, $0.223/1k | **70%** | paired |
| LegalBench, 20 of 37 tasks | varies | varies | **100%** | published |
| LegalBench, 8 more | tie within split resolution | | **100%** | published |
| LegalBench, 9 reasoning tasks | loses by 7–47 pts | ahead | **0%** | published |
| CLINC150 | 88.5% | Claude Haiku zero-shot 88.5% | **100%** | published |
| dair-ai emotion | 92.4% | QLoRA LLaMA 3.1 90.9% | **100%** | published |
| GoEmotions | 0.450 macro-F1 | ChatGPT zero-shot 0.256 | **100%** | published |
| CUAD clause detection | 83.4% pooled | no comparable figure exists | **76% @ 90%** | coverage only |
| TweetEval sentiment | 67.5% | not run | **no useful middle** | — |

Supporting detail worth chasing prior art on separately:

- **The label-budget crossover.** Same 250 items, only training-set size varies. banking77 TF-IDF:
  19.9% @ 50 labels → 48.2% @ 200 → 85.3% @ 2,000 → 92.0% @ 12,701. Prompted LLMs span
  54.6–82.9% throughout. **A bag of n-grams passes the entire prompted field between 1,000 and
  2,000 labels.** At 100 labels the LLM leads by 32 points. Neither "just fine-tune" nor "just
  prompt" is a fact about the dataset; both are facts about the label budget.
- **Spread between LLMs exceeds the free-vs-paid gap.** On the three LegalBench tasks run against
  four 2026 models, the free arm is never more than 3 points behind the best LLM, while the spread
  *between* LLMs on the same task is 9–16 points, with no model best everywhere.
- **Tabular/serialized input is the largest margin in the program.** FinBen credit-and-fraud, six
  tasks: free 0.74–1.00 wF1 vs GPT-4 published 0.55–0.74, and the LLM's MCC is at or below chance
  on five of six. Serializing a table into a prompt discards the structure that made it easy.

### 3.2 The gate comparison — all nine (task, LLM) pairs, out-of-fold

Accuracy, with the share sent to the LLM. Cheap arm = the best free candidate for that task
(TF-IDF or a frozen-embedding row). p-values are **uncorrected**; see §7.

| task | LLM | cheap | LLM alone | conformal | confidence (ni) | committee+conf | **class-aware (any)** | share escalated | oracle |
|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | deepseek-v4-pro | 84.3 | 81.9 | 85.0 @37% | 84.3 @0% | 86.6 @21% | 84.1 | 36% | 93.0 |
| learned_hands_consumer | glm-5.3-flash | 84.3 | 87.5 | 87.8 @37% | 86.4 @16% | 87.1 @31% | 87.5 | **100%** | 93.1 |
| learned_hands_consumer | gpt-oss-120b | 84.3 | 71.5 | 81.6 @37% | 84.3 @0% | 84.6 @13% | **85.2** | 50% | 92.4 |
| opp115_data_retention | glm-5.3-flash | 79.7 | 78.6 | 83.0 @53% | 79.7 @1% | **85.9** @22% | 81.8 | 50% | 91.8 |
| opp115_data_retention | gpt-oss-120b | 79.7 | 73.0 | 80.4 @53% | 79.7 @0% | **85.3** @22% | 81.4 | 50% | 91.3 |
| opp115_data_retention | kimi-k2.6 | 79.7 | 75.7 | 80.9 @53% | 79.7 @0% | **84.8** @21% | 81.5 | 50% | 91.8 |
| supply_chain_audits | deepseek-v4-pro | 80.9 | 77.0 | 77.8 @58% | 80.9 @0% | 80.7 @14% | **85.0** | 76% | 91.6 |
| supply_chain_audits | glm-5.3-flash | 80.9 | 76.0 | 76.3 @58% | 80.9 @0% | 80.8 @14% | **84.4** | 76% | 90.5 |
| supply_chain_audits | gpt-oss-120b | 80.9 | 82.1 | 80.8 @58% | 81.8 @7% | 82.8 @34% | **85.4** | 76% | 91.7 |

Class-aware gate, significance (McNemar mid-p, uncorrected):

| pair | vs cheap | vs LLM | beats both on point estimate? |
|---|---|---|---|
| learned_hands / deepseek | p=0.79 | p=0.18 | no |
| learned_hands / glm | p=0.047 | p=1.0 | collapses to 100% LLM |
| learned_hands / gpt-oss | p=0.062 | p<0.001 | yes |
| opp115 / glm | **p=0.017** | p=0.27 | yes |
| opp115 / gpt-oss | p=0.078 | **p=0.008** | yes |
| opp115 / kimi | p=0.072 | p=0.056 | yes |
| supply_chain / deepseek | **p=0.003** | **p<0.001** | yes |
| supply_chain / glm | **p=0.003** | **p<0.001** | yes |
| supply_chain / gpt-oss | **p=0.001** | p=0.094 | yes |

**The finding, stated carefully.** The confidence-based gates behave exactly as the cascade
literature's pessimists would predict: the tuned confidence threshold chose *escalate nothing* in
7 of 9 pairs, and the learned router — trained on the LLMs' own per-item outcomes — landed within
−0.3 to +1.4 points of whichever base model was already better, never above both. The oracle sits
5–13 points above every honest gate. But the **class-aware gate beats both base models on the
point estimate in 7 of 9 pairs, and significantly against both arms in 2** (supply_chain ×
deepseek and × glm, p ≤ 0.003 both sides). The mechanism claim: *"the cheap model is uncertain" is
not "the LLM knows better"*, and relative competence is not recoverable from the cheap model's own
confidence — but it **is** partly recoverable from a coarse, label-conditional partition estimated
from a few hundred LLM-labelled calibration items.

The diagnostic that explains it: where the committee disagrees, the cheap model is right 55–74%
and the LLM 60–78%; where it agrees, the cheap model is right 87–93% and the LLM 74–91%. The
competence ordering **flips by region**, which is what a class-conditional rule can exploit and a
global confidence threshold cannot.

---

## 4. Research questions, in priority order

1. **Is label-conditional / per-predicted-class deferral established?** Any published routing,
   cascade, or learning-to-defer method whose escalation decision is made per *class* or per
   *partition* rather than per item, and does anyone report it beating both base models
   out-of-fold? This is the single most important question in this brief.
2. **Is the "confidence is not relative competence" result already published?** Strong suspects
   exist in the cascade literature (post-hoc deferral rules; token-level uncertainty for LM
   cascades). **If a paper already establishes that confidence-based deferral is provably
   suboptimal and that a learned post-hoc rule is required, our Part 4 is a replication on new
   data, not a discovery — say so.**
3. **What is the current published state of the art on each dataset?** banking77, CLINC150,
   dair-ai emotion, GoEmotions, ToxicChat, TweetEval sentiment, LegalBench (per task), CUAD
   (clause presence, not span AUPR), MASSIVE per locale, HINT3. We need to know where our free arm
   actually stands, including against *free* opponents.
4. **Is the label-budget crossover already characterized?** Published curves for "how many labels
   before a fine-tuned small model beats a prompted frontier model", by task type. Our 1,000–2,000
   figure on a 77-class task needs to be placed against existing measurements.
5. **Is committee disagreement (query-by-committee) established as an LLM-deferral signal**, as
   opposed to its standard use in active learning?
6. **Is the central claim refuted?** Published evidence that 2025–2026 frontier or small
   open-weight models now beat fine-tuned encoders on exactly these workloads, or that the cost
   saving does not survive realistic traffic and retraining.
7. **Commercial and industrial prior art.** LLM routers sold as products; "replace your LLM with a
   classifier" tooling; contact-center auto-resolution gated by per-intent precision bars (this is
   the class-aware coverage variant, almost certainly deployed somewhere without a paper). Patents
   matter.
8. **Contamination.** Has anyone else documented train/test prompt overlap in FinBen (we measure
   71.5% of test rows) or BeaverTails (99.8%)? If these are known, cite it; if not, it is a
   reportable dataset finding in its own right. Our audits, pinned and reproducible, are in
   `contamination/` -- see its README for the exact match criteria a prior-art claim would
   have to be compared against.

---

## 5. Literature areas and search terms

- **Already checked, do not re-search:** *Online Cascade Learning for Efficient Inference
  over Streams* (Nie et al., ICML 2024) -- the closest published relative: LR -> BERT ->
  GPT-3.5 cascade with a learned post-hoc confidence calibrator as the deferral rule.
  Reconstructed and compared in `ocl_compare/`. *Cache & Distil* (Ramirez et al., ACL
  Findings 2024) -- neural caching with Margin Sampling / Query-by-Committee policies;
  its benchmark has no per-class competence variation to exploit and was a dead end.
  *DESlib* (LCA / OLA / KNORA-E) -- dynamic classifier selection is classic; our
  per-class confidence rule is a variant of LCA and lost to the library's own
  implementations. Do not claim the ensemble as novel. **Now wired into the
  pipeline** (`classpipe.deslib_selection`, `pip install -e '.[deslib]'`) and run
  on all four datasets against a data-matched baseline: 12 of 12 comparisons lose
  to the best single model, every loss surviving Holm. The Oracle ceiling is
  +0.03 to +0.20, so the headroom is real and no selection rule reaches it.
- **LLM cascades / routing / deferral:** FrugalGPT; RouteLLM; "Hybrid LLM"; AutoMix; model
  cascading for LLMs; "when does confidence-based cascade deferral suffice"; post-hoc deferral
  rules; cascade-aware training; token-level uncertainty for LM cascades; "learning to defer";
  "learning to reject"; classification with a reject option; selective prediction / selective
  classification; Chow's rule.
- **The class-aware angle specifically:** class-conditional / label-conditional abstention;
  **per-class rejection thresholds**; group-wise or subgroup-conditional deferral; partition-based
  routing; "which categories to automate"; algorithmic triage; human-AI complementarity;
  "predict responsibly"; consistent estimators for learning to defer; deferral to multiple experts;
  class-wise coverage in selective prediction. *(Search both the ML-theory phrasing and the
  operations phrasing — "auto-approve rate by category", "straight-through processing rate".)*
- **Ensemble disagreement as an uncertainty signal:** query-by-committee; deep ensembles;
  disagreement-based active learning; ensemble disagreement for selective prediction.
- **Conformal prediction:** split conformal; conformal prediction sets for classification;
  conformal abstention / selective conformal; class-conditional (Mondrian) conformal prediction —
  **this is the closest formal relative of the class-aware gate and must be checked directly.**
- **Small model vs LLM on classification:** fine-tuned encoder vs zero-shot LLM text
  classification; "is GPT-4 a good annotator"; distillation for classification; prompting vs
  fine-tuning sample efficiency; "the unreasonable effectiveness of simple baselines"; TF-IDF vs
  transformer baselines.
- **Per-dataset:** ABCD is out of scope here; instead cite-chase banking77 (Casanueva et al.),
  CLINC150 (Larson et al. 2019), GoEmotions (Demszky et al. 2020), dair-ai emotion, TweetEval
  (Barbieri et al. 2020), **ToxicChat (Lin et al. 2023 — its own paper argues that moderation APIs
  miss its distribution, so check whether our headline finding is that paper's thesis restated)**,
  CUAD (Hendrycks et al. 2021), LegalBench (Guha et al. 2023), FinBen, BLURB (Gu et al.),
  MASSIVE (FitzGerald et al. 2022), HINT3 (Arora et al. 2020), BeaverTails.
- **Cost/efficiency methodology:** measured-token pricing; reasoning-token billing; prompt caching
  economics; inference cost accounting for classification workloads.
- **Benchmark contamination:** train/test overlap detection; near-duplicate leakage; n-gram
  contamination audits.

---

## 6. What a strong answer looks like

Per research question: the closest prior work with citation, what it did that overlaps, **what it
measured and the number**, and a one-line verdict — *anticipated / partially anticipated /
distinct*.

Prioritize (i) anything reporting a **traffic share routed to a cheap model at a stated quality
bar**, which is the exact quantity this system optimizes; (ii) anything reporting a routing rule
evaluated **out-of-fold against both pure strategies with a paired test** — most routing papers
compare against a single baseline or an in-sample curve, and that difference is the whole
argument; (iii) results on the specific datasets in §3.1.

Flag explicitly if you find: **(a)** a published per-class or partition-conditional deferral rule;
**(b)** a paper already establishing that confidence-based deferral cannot capture relative
competence; **(c)** a published number beating our free arm on any dataset in §3.1, *including by
another free model*; **(d)** an existing contamination audit of FinBen or BeaverTails; **(e)** any
work showing per-class routing degrades under distribution shift — this is the most likely failure
mode of the mechanism and we have not tested it.

---

## 7. Known weaknesses — do not let the search paper over these

- **Only three comparisons in the whole program are paired, same-items, against a *paid*
  incumbent:** banking77, Financial PhraseBank, ToxicChat. Everything else compares against
  published figures on their authors' protocols, or carries no baseline at all. Published
  comparison is the weakest evidence here and is labelled as such throughout.
- **Our cheap arm is not the best free arm, and two independent suites say so.** BLURB/ChemProt:
  our 44.70 micro-F1 vs GPT-4's 47.42 vs **PubMedBERT's 77.24**. MASSIVE: English ties published
  mT5/XLM-R, and **all 12 non-English locales lose by 2.9–5.2 points** — to open-weight models
  that also cost nothing. The true statement is "our particular cheap arm was the wrong free model
  here", not "free loses here". (One caveat that weakens the MASSIVE negative and is ours: the
  multilingual fine-tune did not converge, so a fine-tuned multilingual encoder's ceiling there is
  *untested*, not answered.)
- **Derive-the-answer tasks: 0 of 9.** LegalBench reasoning tasks lose by 7–47 points and no gate
  or representation recovered it. The boundary — *is the answer present in the input, or must it
  be derived using knowledge the input does not contain?* — predicted every win and loss in the
  program.
- **The class-aware gate's statistics are uncorrected.** Nine (task, LLM) pairs × ~8 gates, and
  the headline is a max over gates. Holm across that family would likely keep supply_chain
  (p ≤ 0.003 both sides) and kill the marginal cells. Treat 2 of 9 as the defensible count.
- **The class-aware gate is not a cost saver in the cells where it wins.** It escalates 36–100% of
  traffic (median 50%, 76% on supply_chain). It is an *accuracy* device bought with LLM spend, the
  inverse of the program's other findings. And it needs per-item LLM correctness on a calibration
  set to fit at all.
- **These gate results are not yet reflected in the project's own top-level documents.**
  `legalbench_map/POST_DRAFT.md` still states that no gate beats the better base
  model, which was true of the first replay (conformal, threshold, learned router) and is **not**
  true of the class-aware and committee gates in `legalbench_map/results_gate2/`. Do not treat the
  narrative documents as current on this point. `SCOPE.md` was updated 2026-09-24: a swept
  budget dial beats both arms on all four `ocl_compare/` datasets by resolvable margins,
  and the LegalBench null is now recorded as the boundary condition rather than the rule.
  Two things separate them -- those tasks carry 300-614 items (resolving ~3 points, while
  three of the four new gains are under 1.6) and the effect requires gold labels: fitted
  on LLM annotations instead, no dataset shows a resolvable gain over calling the LLM.
- **Three tasks, four LLMs, 300–614 items each** is the entire basis for §3.2. Test splits that
  size resolve about three points, so "tie" means "tie within three points". Each LLM ran once at
  temperature zero while the cheap arm got three cross-validated passes — an asymmetry chosen and
  reported, not hidden.
- **The LegalBench CV protocol uses more labels than the original few-shot setting** (each fold
  trains on the rest of the test split plus the small train split), so these are CV estimates, not
  leaderboard scores, and CV-vs-published comparability is an assumption.
- **The chart pages are built on the TF-IDF row only**, because that is the only row whose
  per-item confidences were persisted. The encoder beats TF-IDF on 6 of 7 datasets, so the routing
  curves are a floor, not the best available.
- **Two of four suites are contaminated:** FinBen headlines (71.5% of test rows share a prompt
  with train; 0.982 avg wF1 full split vs 0.967 leak-free) and BeaverTails (99.8%). CUAD and
  HINT3 are clean, verified rather than assumed. Both contamination audits are reproducible in
  `contamination/`; the 0.982/0.967 wF1 pair is not, having been lost with the FinBen arm.
- **Workloads with a "none of the above" class break the routing story.** On HINT3, the same
  confidence score cannot do out-of-scope rejection *and* routing: the gate posts AUROC 0.511 /
  0.451 / **0.240** (the last two inverted), and escalating the low-confidence tail *lowers* kept
  accuracy on all three bots. Ask of any cascade claim: does this workload have an out-of-scope
  class? If so the routing signal is already spent.
- **"No useful middle" is a real outcome.** TweetEval sentiment: AUROC 0.651–0.668, errors spread
  evenly across the confidence range, and a ceiling that is not a data shortage (67.3% on 12,000
  labels, 67.5% on 45,586).
- **A benchmark win is not a deployment forecast**, and the program's own track record is that
  measurement *changed the answer* repeatedly — a feature leak reading as 99–100%, a rate card
  claiming 3× that a stopwatch measured at 1.16×, a scorer bug that manufactured a p=0.049 out of
  nothing, and a CUAD headline that was backwards until lift over a constant was computed.
