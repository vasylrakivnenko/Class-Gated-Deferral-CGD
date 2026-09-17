# What we can claim, and what we cannot

Written from the measured record only. Every number here is in `FINDINGS.md`,
`CASCADE_DIAL.md`, `legalbench_map/POST_DRAFT.md`, `finben/FINAL_SUMMARY.txt`
or `scratchpad/blurb_FINAL_TABLES.md`. Nothing is extrapolated.

The point of the file: we have a real boundary, and it is sharper and more
defensible than "small models are good enough". Claiming past it costs us the
part that is true.

---

## 1. The boundary that actually predicted the results

One question separated wins from losses across every suite:

> **Is the answer present in the input, or does it have to be derived from the
> input using knowledge the input does not contain?**

Present in the input -> we win or tie, usually outright.
Has to be derived -> we lose, and no amount of cheap-model engineering fixed it.

The cleanest demonstration is LegalBench, because the same harness ran both
sides on 37 tasks chosen before we saw any result:

| task family | tasks | free arm within 3 pts of best published LLM | free arm ahead outright |
|---|---|---|---|
| pattern recognition (clause presence, policy topic, disclosure type, forum intent) | 28 | 24 | 18 |
| genuine legal reasoning (hearsay, diversity jurisdiction, trademark strength) | 9 | 0 | **0** |

On the 9 reasoning tasks the free arm lost by **7 to 47 points**. That is not a
tuning gap. "Reasoning task" and "hard for a cheap model" turned out to name the
same set.

---

## 2. Three things we can claim, with the evidence

### A. Pattern classification against your own label set

| evidence | free arm | paid / published | margin |
|---|---|---|---|
| banking77, 77 intents, paired same-items | 94.0% | 82.9% measured | **+11 pts, -$3.45/1k** |
| FinBen headlines, 9 sub-tasks | 0.982 avg wF1 | 0.86 GPT-4 published | +0.12 |
| LegalBench pattern tasks | see above | | 18 outright wins |
| dair-ai emotion | 92.4% | 90.9% QLoRA-tuned LLaMA 3.1 | +1.5 |
| GoEmotions | 0.450 macro-F1 | 0.256 ChatGPT zero-shot | ~1.8x |

### B. Anywhere the taxonomy is yours and the vendor's is not

The strongest single result in the project, and paired same-items at zero cost
because the incumbent's per-item scores ship with the dataset:

| ToxicChat, 4,765 items | accuracy | F1 on toxic | AUPRC |
|---|---|---|---|
| OpenAI moderation (the deployed incumbent) | 0.939 | 0.237 | 0.615 |
| always answer "safe" | 0.932 | 0.000 | - |
| free fine-tuned encoder | **0.966** | **0.744** | **0.823** |

**3.1x the incumbent's F1.** State the reason fairly: a moderation API is
trained for its own content policy, not for your definition. That is the
structural advantage and it is the thesis in one row -- *an off-the-shelf LLM
classifier does not match your labels, and 5,000 of your own labels beat it by a
wide margin.* Claim the mechanism, not "OpenAI's classifier is bad."

### C. Tabular and semi-structured input

The least surprising and largest margins in the project. FinBen credit and fraud
tasks, held-out test splits, weighted F1:

| task | free best | GPT-4 published | margin | GPT-4 MCC |
|---|---|---|---|---|
| ccf | 1.000 | 0.55 | +0.450 | -0.02 |
| polish | 0.958 | 0.55 | +0.408 | -0.02 |
| ccfraud | 0.925 | 0.55 | +0.375 | -0.02 |
| lendingclub | 0.857 | 0.55 | +0.307 | -0.02 |
| german | 0.736 | 0.55 | +0.186 | -0.02 |
| australian | 0.856 | 0.74 | +0.116 | 0.47 |

Read the last column. On the imbalance-robust metric the LLM is at or below
chance on five of six, while gradient boosting over the columns is not. Serialising
a table into a prompt throws away the structure that made the task easy.

### D. The one measured middle: long documents, many rare categories

CUAD, 510 contracts, 29 usable clause categories, 14,790 decisions,
cross-validated with whole similarity clusters held out. Pooled 83.4%, and the
confidence gate passes on both parts (AUROC 0.797; 49.3% of all errors in the
least-confident fifth). **76% of decisions answerable for free at 90%
reliability.** The only workload we have measured that lands genuinely in the
middle rather than at an end.

---

## 3. What we should not claim

### Do not claim derive-the-answer work

0 of 9 on LegalBench reasoning, by 7 to 47 points. Neither a better
representation nor a cascade recovered it. Say the boundary out loud -- it is
more credible than pretending there isn't one, and the boundary is the product.

### Do not claim a cascade beats both arms

We built the router, replayed three gates plus an oracle out-of-fold, and it
does not:

- The oracle sits **5 to 13 points above every honest gate** on all nine
  (task, model) pairs.
- A learned router, trained on the LLMs' own per-item outcomes, landed within
  **-0.3 to +1.4 points of whichever base model was already better -- never
  above both.**
- A tuned threshold gate chose "escalate nothing" in **7 of 9** pairs.
- On the supply-chain task, escalating made accuracy **worse**: the items the
  free model doubted were items the LLM got wrong more often.

**The cascade is a cost dial, not an accuracy device.** "The cheap model is
uncertain" is not "the LLM knows better"; routing needs relative competence and
that signal is not recoverable from the cheap model's confidence. This includes
the CUAD number above: 76% free at 90% reliability is a *coverage* claim about
what needs no escalation, not a claim to beat an LLM.

### Do not claim a middle setting exists on every workload

TweetEval sentiment, re-checked on the row we would actually ship:

| base model | accuracy | majority | AUROC | errors in low 20% |
|---|---|---|---|---|
| TF-IDF + logreg | 0.604 | 0.487 | 0.651 | 28% |
| fine-tuned encoder | 0.674 | 0.482 | 0.668 | 30% |

Errors stay spread evenly across the confidence range, so no gate beats picking
at random. And the ceiling is real, not a data shortage: **67.3% on 12,000
labels, 67.5% on all 45,586.** Four times the data for two tenths of a point.
The honest output is "no useful middle" -- a real answer, not a missing one.

### Do not claim our cheap arm is the best free arm

**This is the most important line in this file, and two independent suites now
say it.** It is a pattern, not an anecdote.

BLURB is where it first bit:

| ChemProt, 16,038 items | micro-F1 |
|---|---|
| our best free row (TF-IDF + LR) | 44.70 |
| GPT-4, best published prompt | 47.42 |
| **PubMedBERT, fine-tuned** | **77.24** |
| **BioLinkBERT-large, fine-tuned** | **79.98** |

We lose to the paid arm *and* we lose by 33 points to the right free arm. The
true finding is "our particular cheap arm was the wrong free model for this
domain", not "free loses here". Do not quietly convert one into the other. Same
suite: DDI 44.2 against 44.7 published, a tie at a level where both are bad, and
PubMedQA 54.2 against a **55.2 constant predictor** -- below a label-blind
answer.

MASSIVE repeats it across 12 languages, which is cleaner because it is not one
domain-specific dataset. Best free arm per locale against published fine-tuned
mT5-Base / XLM-R-Base on the same official split and metric. **Convention: the
"best published" column is the strongest of the three published models for that
locale**, which is the harsher bar -- against XLM-R alone the gaps are 0.3 to
5.6 points, and the cheap arms (TF-IDF, frozen embeddings) lose in all 13
locales including English:

| locale | our best | best published | gap |
|---|---|---|---|
| en-US | 88.03 ±1.17 | 89.0 | **tie** |
| sw-KE | 82.78 | 85.8 | -3.0 |
| es-ES | 83.96 | 86.9 | -2.9 |
| de-DE | 82.89 | 86.8 | -3.9 |
| zh-CN | 81.91 | 85.8 | -3.9 |
| ja-JP | 81.78 | 85.8 | -4.0 |
| hi-IN | 82.11 | 86.2 | -4.1 |
| ar-SA | 77.94 | 82.2 | -4.3 |
| am-ET | 79.76 | 84.2 | -4.4 |
| th-TH | 80.90 | 85.5 | -4.6 |
| ru-RU | 82.08 | 87.2 | -5.1 |
| ko-KR | 81.34 | 86.5 | -5.2 |

English ties; **all 12 non-English locales lose, 2.9 to 5.2 points, without
exception.** mT5 and XLM-R are open-weight and free, so once again the opponent
we lose to costs nothing -- the gap is investment, not licensing.

**One caveat that weakens this negative, and it is ours to fix.** The arm that
could have competed was undertrained: fine-tuning `multilingual-e5-small` scored
*below simply freezing it* on all four locales it ran (en-US 0.8255 vs 0.8362,
de-DE 0.7283 vs 0.7989, zh-CN 0.7646 vs 0.8073, th-TH 0.7091 vs 0.7774). A model
that gets worse when fine-tuned is an optimisation failure, not a capability
ceiling -- final training loss 1.68 against the English encoder's 0.077, at 3
epochs and a learning rate chosen for a different encoder. **The consequence is
precise: this arm does not establish that a fine-tuned multilingual encoder
fails to close the off-Latin gap. That question is untested, not answered.**
Re-run before quoting the gap as final.

### Do not claim a workload that needs an out-of-scope decision

HINT3 -- three companies' live chatbot logs, 328/600/471 training rows, 21/28/59
intents -- is the arm that found this, and the reason matters more than the loss.

**An out-of-scope threshold cannot be selected from training data that contains
no out-of-scope examples.** HINT3's train splits contain none, which is not a
quirk of the benchmark: a production bot's training phrases are all in-scope by
construction. The stand-in rule (10th percentile of out-of-fold in-scope
confidence) landed within 0.001 of test-optimal on one bot and badly wrong on
another:

| bot | ours @ train-selected t | all-out-of-scope constant | out-of-scope share |
|---|---|---|---|
| sofmattress | 0.6801 | 0.4181 | 41.8% |
| curekart | 0.7023 | 0.5439 | 54.4% |
| powerplay11 | **0.5748** | **0.7202** | 72.0% |

Per the constant guard, **powerplay11 overall accuracy is dropped, not
caveated.** The free arm can beat that 0.7202 constant only by answering 12.1%
of traffic at 25% in-scope accuracy -- the constant wearing a hat. And on the
constant-proof metric (macro-F1 over in-scope intents), compared at the same
threshold granularity as the five NLU platforms, we rank **3rd, 3rd and 6th of
6** -- zero outright wins, last place on the hardest bot. A per-bot accuracy
table showing 7 paired McNemar wins and 8 ties exists and is real, but it is on
the metric the constant guard disqualifies here. **Do not lead with it.**

**There is no LLM on this arm at all.** The five baselines are 2020-era NLU
platforms and a fine-tuned BERT. Nothing here tests free-versus-paid, and at
7-14 labels per intent HINT3 sits deliberately below this project's measured
crossover (LLM +32 at 100 labels, -3 at 2,000), so the expected paid result here
is an LLM win.

**The transferable mechanism: you cannot use one confidence score twice.** If
confidence is doing out-of-scope rejection, it is no longer available for
routing. The cascade gate on the shipped pipeline posts AUROC 0.511 / 0.451 /
**0.240** -- the last two *inverted* -- with 7.9% / 5.1% / 3.4% of errors in the
least-confident 20% against a 20% random reference. Escalating the
low-confidence tail *lowers* kept-slice accuracy on all three bots. Measured
separately, the same score is a good **scope** detector (~0.85 AUROC, on par
with Dialogflow's 0.869) and a poor **correctness** detector (0.68-0.77). No
earlier workload had an out-of-scope class, so this conflict never surfaced.

Ask before quoting any cascade number: does this workload have a "none of the
above"? If yes, the routing signal is already spent.

### Do not claim non-English without measuring it

See the MASSIVE table above: English ties published fine-tuned multilingual
transformers, and every other language is 3-5 points behind.

**There is no script cliff, though, and that is worth saying because it is the
thing people assume.** The 12 non-English locales sit in a 6.0-point band
(ar-SA 0.7794 to es-ES 0.8396), 11 of 13 clear 0.80, and non-Latin scripts
interleave with Latin ones rather than sorting below them -- zh-CN at 0.8191
outranks hi-IN, ja-JP, ko-KR and th-TH, and sw-KE outranks ru-RU. The two
weakest locales are also near the bottom of the paper's own baselines, so the
ordering is a property of the locales, not of our arm.

What *does* transfer is routing. The cascade gate passes on **13 of 13
locales**, AUROC 0.828-0.908 (median 0.855) with 49-80% of errors in the
least-confident fifth, in Amharic and Thai as readily as in English; en-US at
0.908 / 79.8% essentially matches banking77's 0.905 / 80%. As coverage that is
**72%-97% of traffic answerable free at 90% reliability, median 83%**, against
CUAD's 76%. Coverage/cost framing only. So the escalation story is
language-independent; the accuracy story is not.

One thing we cannot claim: **cross-lingual zero-shot.** Training on English and
testing on another locale beat published XLM-R zero-shot on only **3 of 12**
locales (ja-JP +29.8, zh-CN +10.7, sw-KE +4.7) and lost on 9. Against the weaker
mT5-T2T baseline it wins 6 of 12; report the XLM-R comparison, which flatters us
least.

Two mechanism findings worth keeping:

- **Word-ngram features collapse without whitespace.** zh-CN averages 1.05
  whitespace tokens per utterance and ja-JP 1.17, so a word vectorizer sees
  roughly one token: it scores 0.1459 and 0.1715 against a 0.0703 constant. The
  character half of the same union carries those locales intact (0.7599,
  0.8013). Always keep `char_wb` in the union; on CJK it is the whole model.
- **hi-IN's 24-point word-arm deficit was our bug, not a language property.**
  It has 7.59 whitespace tokens per utterance -- *more* than English -- so
  segmentation never explained it. Cause: sklearn's default
  `token_pattern=r"\b\w\w+\b"` does not match Unicode combining marks, and 38.6%
  of Devanagari characters are marks, so **88.5% of hi-IN whitespace tokens were
  silently rejected.** With a mark-aware pattern the word arm recovers **+23.2
  points** (0.5797 -> 0.8117) while three control locales moved by <=0.2. Any
  mark-heavy script -- Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam --
  is affected by the default, which is most of South Asia.
  And the same fix makes th-TH **worse** by 7.3 points, because there the
  default pattern was accidentally acting as a Thai syllable segmenter. So the
  vectorizer's token pattern is a per-script decision, not a default to inherit.

Two predictions we made in advance and got wrong, recorded because the record is
the product: frozen `multilingual-e5-small` is **not** the best free arm outside
English (plain char TF-IDF beats it on 9 of 13 locales, by +8.5 on Swahili and
+8.4 on Amharic), and the **English-only** encoder was expected to fail on
non-Latin scripts but scored 0.8191 on zh-CN -- the best of any arm there, above
the multilingual encoder's 0.7646.

### Do not claim stance or interpretation over short text

FinBen FOMC, hawkish vs dovish: free arm 0.600, GPT-4 0.71, **behind by 11
points**, and the gate fails too (AUROC 0.696-0.699, 30-31% error recall). Same
shape as TweetEval's `neutral` class, which is a majority class defined by the
*absence* of a cue: only 8.8% of test tweets contain any learned cue word at
all, and the largest error cell is gold `negative` predicted `neutral`, 12% of
the test set.

### Do not present benchmark numbers as deployment numbers

Two of four suites are contaminated, which we found by checking rather than by
being told:

- **FinBen headlines: 71.5% of test rows have a prompt that appears in train.**
  Our avg wF1 is 0.982 on the full split and 0.967 on the 651 leak-free rows.
- **BeaverTails: 99.8% of test rows share a prompt with train** (distinct
  responses, so pairs do not repeat). And the label ceiling is set by the
  annotators, not the models: across texts appearing more than once, **27.7%
  disagree on `is_safe`**, and 23.9% on `non_violent_unethical_behavior`.
- CUAD by contrast is clean: 4 pairs above 0.90 similarity, none crossing the
  split. That is a finding about CUAD, not a property we can assume.
- **HINT3 is clean too:** 0.000% / 0.605% / 0.000% exact test-row overlap, and
  0% among out-of-scope items on every bot.
- **MASSIVE's parallel-translation hazard is handled upstream, and we verified
  rather than assumed it:** all 16,521 utterance ids appear in all 52 locales
  with **zero** partition disagreements, so the official split is id-aligned.
  Pooling locales and splitting at random would have leaked a Spanish utterance
  into train and its German twin into test. Within a locale, near-duplicates are
  not negligible and belong beside any per-locale number: **14.9% of en-US test
  rows** sit above 0.90 similarity to a training row, and zh-CN carries 223
  exact duplicate test rows, 13 of them label-conflicting. Exact-duplicate rates
  run 0.71%-14.43% across the 52 locales (median 5.41%, worst km-KH), and
  **en-US is the lowest of all 52** -- distinct English utterances collapse onto
  identical strings when translated. So cross-locale comparisons are mildly
  conservative about English, not generous to it.

A benchmark win is evidence the method works. It is not a forecast of the number
a customer will see.

### Do not overstate how much of the record is paired

Only **three** rows are paired, same-items, against a *paid* incumbent:
banking77 (82.9%), Financial PhraseBank (98.8%) and ToxicChat (0.2375 F1).

HINT3 adds a fourth paired, same-items comparison -- against five NLU platforms
(Dialogflow, LUIS, RASA, BERT, Haptik), item-aligned across all three bots, with
McNemar tests. It is genuinely paired, but **none of those five is an LLM**, so
it does not extend the free-versus-paid record. Label it accordingly.

Everything else compares against published figures on their authors' protocols
(MASSIVE's mT5/XLM-R included), or carries no baseline at all (CUAD,
deliberately -- published CUAD numbers are span-extraction AUPR and this is
clause presence). Published comparisons are the weakest evidence we hold; label
them.

### Two method caveats that limit how strongly the charts can be read

- Every chart panel is built on the **TF-IDF** row, because that is the only row
  whose per-item confidences were persisted. The encoder beats TF-IDF on 6 of 7
  datasets, so the routing curves are a **floor**, not the best available.
- The asymmetry in the LegalBench comparison was chosen and is reported: each
  LLM ran once at temperature zero; the free arm got three cross-validated
  passes. Test splits of 300-600 items resolve about three points, so "tie"
  means "tie within three points".

---

## 4. The claim that survives all of it

Not "you don't need an LLM". The record does not support that and it has now
refuted it four times.

**What we can claim is the measurement, and its track record is that it changed
the answer** -- it caught a feature leak that read as 99-100% accuracy, a
$3.45/1k saving that a rate card said was 3x and a stopwatch said was 1.16x, an
incumbent moderation API at 3.1x less F1 than a free encoder, 71.5% and 99.8%
benchmark contamination, a router that looked like it worked until it was scored
out-of-fold, our own headline CUAD framing, which was backwards until lift over
a constant was computed, a HINT3 headline that read as 7 paired wins on the
metric a label-blind constant beats, and two advance predictions about which
free model would win abroad that both went the other way.

Each of those was a decision someone would otherwise have made wrong, and none
of them was visible without running the measurement. That is the durable claim:
we tell you which side of the boundary your workload is on, in an afternoon, for
approximately nothing -- and we are specific about the side we lose on.
