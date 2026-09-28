# Research brief — prior art and competitive landscape for REFLEXIVE v1

**Purpose.** You are a research agent. Find (a) prior work that anticipates any part of the system
below, (b) competing systems that beat its numbers on the same task, and (c) the strongest
published counter-evidence to its central claim. Be adversarial: the goal is to discover that this
has already been done, not to confirm novelty.

---

## 1. What the system is, in one paragraph

A **response-selection cascade for task-oriented dialogue** that answers customer-service agent
turns by *choosing* a pre-compiled response rather than generating text, and escalates the turns it
is not confident about to a large LLM. The claim under test is a cost claim: a bag-of-words
classifier stack can absorb a large fraction of production dialogue traffic at a quality
indistinguishable from a fine-tuned small LLM, at a fraction of the inference cost. Dataset is
**ABCD** (Action-Based Conversations Dataset, Chen et al., NAACL 2021).

---

## 2. The algorithm

### 2.1 Compile time (train split only)

1. **Delexicalize** every agent utterance in train: replace surface values (names, order IDs,
   amounts, product names) with typed slot markers, using an ontology of button/action argument
   types plus shape heuristics.
2. **Cluster** the delexicalized sentences into **templates**; drop templates below a minimum
   occurrence count. Result on this corpus: **4,489 templates over 58,263 occurrences**, covering
   **65.01%** of the 89,623 train sentences.
3. **Act-label** every sentence into a small act inventory (ACK, ASK, OFFER, CLOSE, VERIFY,
   CONFIRM, INSTRUCT, INFORM, OTHER) and group templates per act.
4. Define a turn's **SKELETON** = the ordered tuple of acts in that turn, e.g. `(ACK, ASK)`.
   Skeletons are assigned ids; ~570 are learnable.

### 2.2 Inference, per agent turn

Context featurization: the last *k*≈6 turns, rendered as a token string in which every token is
prefixed with a **recency bucket tag** (`r0|`…`r3|`). This tagging is load-bearing — it is what lets
a bag-of-words model recover word order/recency information it would otherwise destroy.

Heads (each a TF-IDF + multinomial logistic regression over that rendering):

| head | decides |
|---|---|
| `nextstep` | branch: *say something* (`retrieve_utterance`) vs *click a button* (`take_action`) |
| `intent`, `action` | dialogue intent, and which button |
| `skeleton` (H5) | the ordered act tuple for this turn |
| `template` (H7) | per act position, which template id — one classifier **per act** |
| `value` (H4) | for `take_action`, which disclosed customer value fills each argument slot |

Then: fill the chosen templates' slots from values the customer has actually disclosed so far
(a disclosure timeline is maintained), and emit the composed text.

### 2.3 The confidence gate (the "deflection" mechanism)

Two mechanisms, evaluated separately:

- **Split-conformal abstention.** Per-head nonconformity quantiles fitted on a dev split give
  prediction sets at a target coverage 1−α; a turn whose set is too large or whose head is
  unconfident is escalated to the LLM. (This path is specified and implemented but has never been
  executed end to end in this repo — treat it as a design, not a result.)
- **Committee disagreement.** A primary predictor (the TF-IDF+logreg skeleton head) plus two
  cheap auditors — a GRU over the raw text, and an n-gram Markov model over skeleton-id sequences.
  Route the turn to the cache only when auditors agree with the primary; `n_dis ∈ {0,1,2}` buckets
  the traffic. This is the mechanism the measured results use.

### 2.4 Metric

**compose@1** = the predicted skeleton matches gold AND every position's predicted template id
matches gold. Strict, exact-match, all-or-nothing per turn. Reported on two populations that must
never be mixed: **n=3,985** turns whose gold is fully expressible in the bank ("conditional"), and
**n=8,889** all test_seen agent turns ("unconditional"). A third, **n=8,858**, is turns having a
gold skeleton.

---

## 3. Measured results (for comparison shopping)

Conditional population, n=3,985, unless stated. "Judged accuracy" = compose@1 plus the misses an
LLM judge (gpt-oss-120b) rated appropriate.

| arm | compose@1 | judged accuracy | notes |
|---|---|---|---|
| learned cache, ungated | 27.7% | 77.2% | TF-IDF+logreg selection |
| learned cache, committee-gated (answers 36.2%) | 31.6% | 86.3% | |
| Qwen3-4B, LoRA SFT, free generation | 31.8% | 87.0% | ties the gate; McNemar p=0.586 |
| qwen3-0.6B SFT, structured supervision | 24.2% | 86.4% | |
| GRU skeleton + same template layer | 25.1% | 69.3% | |
| 1-NN TF-IDF retrieval | 5.2% | — | |
| dense-embedding semantic cache (Redis LangCache) | 2.75–8.25% | — | irreproducible; tie-break unstable |
| n-gram over skeleton sequences | = the label-blind constant | — | zero skill once not teacher-forced |

Skeleton-only accuracy: cache **58.2%**, GRU 52.3%, label-blind constant 31.4%.
Bank ceiling: only **44.8%** of test_seen turns are fully bank-coverable, which caps unconditional
compose@1 at ~37.5% even with an oracle template choice.

---

## 4. Research questions, in priority order

1. **Has response selection from a compiled, delexicalized template bank been done for
   task-oriented dialogue, and how does it score?** Especially: anything that factorizes a turn into
   *act sequence* → *template per act* rather than predicting a flat response id.
2. **Is "cheap classifier absorbs traffic, LLM handles the rest" established, and with what
   gating signal?** We need the state of the art in LLM cascades/routing, and specifically whether
   **ensemble/committee disagreement** has been used as the deferral signal versus the more common
   confidence-threshold or trained-router approaches.
3. **What is the best published result on ABCD** for next-utterance selection, dialogue-act
   prediction, and the dataset's own AST (Action State Tracking) and CDS (Cascading Dialogue
   Success) metrics? Who holds it and with what method and compute?
4. **Has conformal prediction been used for per-head abstention in a multi-head dialogue
   pipeline?** Any work giving distribution-free coverage guarantees over a *pipeline* of
   dependent decisions rather than a single classifier.
5. **Is the central claim already refuted?** I.e. published evidence that retrieval/selection
   approaches are decisively worse than small fine-tuned generators on task-oriented dialogue, or
   that the cost saving does not survive realistic traffic.
6. **Commercial/industrial prior art** — semantic caching and canned-response suggestion in
   contact-center products (agent-assist, reply suggestion). Patents matter here.

---

## 5. Literature areas and search terms

- **LLM cascades / routing / deferral:** FrugalGPT; RouteLLM; "Hybrid LLM"; model cascading;
  "learning to defer"; selective prediction; classification with a reject option; "cascade
  deferral rule"; AutoMix.
- **Semantic caching for LLMs:** GPTCache; "semantic cache" LLM; embedding-based response reuse;
  cache hit-rate vs quality trade-off. (Note: *not* KV-cache / prefix caching — different topic,
  will pollute results.)
- **Retrieval-based dialogue / response selection:** Poly-encoder; ConveRT; bi-encoder response
  selection; Ubuntu Dialogue Corpus / DSTC response-ranking tracks; "retrieval-based chatbot".
- **Template-based NLG and delexicalization:** classic spoken-dialogue-system NLG; SC-LSTM
  (Wen et al. 2015); RNNLG; "delexicalization" dialogue; template mining from corpora;
  "response templates" customer service.
- **Task-oriented dialogue end-to-end:** SimpleTOD; SOLOIST; PPTOD; Schema-Guided Dialogue;
  MultiWOZ leaderboards (for methodology, not scores).
- **Dialogue act prediction:** next-act prediction, act-sequence modelling, hierarchical act
  taxonomies.
- **Conformal prediction in NLP:** split conformal; conformal prediction sets for classification;
  conformal abstention; coverage under distribution shift.
- **Ensemble disagreement as uncertainty:** query-by-committee; deep ensembles; disagreement-based
  active learning; "ensemble disagreement" selective prediction.
- **ABCD specifically:** cite-chase Chen et al. 2021 (NAACL), "Action-Based Conversations Dataset",
  and everything citing it.

---

## 6. What a strong answer looks like

For each research question return: the closest prior work with citation, what it did that overlaps,
**what it measured and the number**, and a one-line verdict — *anticipated / partially anticipated /
distinct*. Prioritize work that reports a comparable strict exact-match metric on task-oriented
dialogue, and any work that reports a **traffic-deflection rate at a stated quality bar**, which is
the specific quantity this system is optimizing.

Flag explicitly if you find: (i) a published system doing skeleton→template factorized selection;
(ii) committee disagreement used as an LLM-deflection gate; (iii) an ABCD result that beats 31.6%
on a comparable strict metric; (iv) evidence that LLM-judge "appropriateness" on selection outputs
is unreliable in a way that would undermine the 86.3% figure.

---

## 7. Known weaknesses — do not let the search paper over these

- **The bank ceiling is the real story.** 55.2% of turns cannot be answered exactly from the bank
  at all. Every headline is reported on the coverable subset. Look for how prior work handles or
  reports this; it is the most likely place the approach has been shown not to scale.
- **compose@1 is brutally strict** (exact template-id match at every position). Prior work using
  BLEU, ROUGE or recall@k is not directly comparable — note the metric mismatch rather than
  comparing numbers naively.
- **The LLM-judge numbers are not human-validated.** Treat the 77–87% accuracy band as a
  directional read.
- **The conformal gate has never been run.** Only the committee gate has measured results.
- **One competing arm (qwen3-0.6B structured) was trained on the metric's own population**, so its
  comparison is confounded.
