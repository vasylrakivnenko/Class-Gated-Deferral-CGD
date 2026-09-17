# The cascade dial: what share of each workload goes to the free model

Terminology, so the file matches how we talk about it: **everything is a
cascade**, including the ends. A workload routed 100% to the free model is a
cascade with the dial all the way over, not a different thing. So the output per
workload is one dial setting.

Three things that setting can be, and the table below uses all three: a
percentage where both arms are measured; `unknown` where only one arm is;
and `no useful middle` where the ends work but no intermediate setting beats
them, because the free model cannot tell which of its own answers are wrong.
That last one is a real answer, not a missing one.

| workload | free arm | LLM | free share | note |
|---|---|---|---|---|
| ToxicChat moderation | 0.744 F1 | 0.237 F1 | **100%** | 3.1x better on the toxic class; API deleted |
| banking77 intents | 94.0% | 82.9% | **100%** | saves $3.45 per 1k items and gains 11 points |
| LegalBench, 20 of 37 tasks | varies | varies | **100%** | free arm ahead outright |
| LegalBench, 8 more tasks | varies | varies | **100%** | tie within the test split's resolution |
| Financial PhraseBank | 97.6% | 98.8% | **70%** | 98.0% at $0.067/1k vs 98.8% at $0.223/1k |
| LegalBench, 9 reasoning tasks | loses by 7-47 | ahead | **0%** | derive-the-answer tasks; keep paying |
| CLINC150 | 88.5% | 88.5% published | **100%** | statistical tie with zero-shot Claude Haiku |
| dair-ai emotion | 92.4% | 90.9% published | **100%** | beats a QLoRA-fine-tuned LLaMA 3.1 |
| GoEmotions | 0.450 macro-F1 | 0.256 published | **100%** | ~1.8x zero-shot ChatGPT; both arms weak |
| TweetEval sentiment | 67.5% | not run | **no useful middle** | see below |
| CUAD clause detection | 83.4% pooled | no comparable figure | **76%** | the only genuine mid-dial row; see below |

## CUAD: the first row that lands in the middle of the dial

Every other row here is 100% or 0%, which makes the dial look like a switch.
CUAD is the row that shows it is a dial.

Setup: 510 contracts, 41 clause categories reduced to 29 usable ones, scored as
"does this contract contain a clause of this type". 5-fold cross-validation
grouped by SIMILARITY CLUSTER rather than by contract, because 22 contract pairs
exceed 0.70 character-4gram similarity and are law-firm templates. Grouping on
contract id alone would put two copies of one template either side of a split.
Every contract is scored exactly once, out of fold. 14,790 decisions in total.

Pre-flight gate, both parts passed:

| signal | value | gate | |
|---|---|---|---|
| AUROC(confidence -> correct) | 0.797 | >= ~0.75 | PASS |
| errors in the least-confident 20% | 49.3% | well above 20% | PASS |

The dial, escalating the least-confident decisions first:

| escalated | accuracy on what is kept |
|---|---|
| 0% | 83.4% |
| 10% | 86.6% |
| 20% | 89.5% |
| 24% | ~90% |
| 30% | 92.0% |
| 40% | 93.9% |

So the shipping setting is a free share of about 76%. Roughly three of every
four clause checks are answered for nothing at 90% reliability, and the
remaining quarter goes to a lawyer or an LLM.

### Two cautions that belong with the number

**The per-class view understates this workload badly.** On the coverage page
CUAD reads 10% of traffic at >= 90%, because only 3 of 29 categories clear 90%
as whole categories. Those three are Governing Law, Expiration Date and
Anti-Assignment, which are present in 73-86% of contracts, so a constant
predictor already scores 0.92, 0.90 and 0.85 on them. They clear the bar because
the clause is common, not because the model is good: their lift over a constant
is +0.04, +0.03 and +0.07, the three smallest in the set. The categories the
model actually learned are Warranty Duration at +0.44, Uncapped Liability at
+0.36 and Irrevocable Or Perpetual License at +0.34. Report lift or balanced
accuracy on this workload, never F1 on the positive class alone, and never the
share of whole categories above 90%.

**There is no LLM line on this panel and that is correct.** Published CUAD
figures are area under precision-recall on extracted spans. This is binary
clause presence. The numbers are not comparable, so the panel carries no
baseline rather than a mislabelled one. Measuring the paid arm on these same
14,790 decisions is the open item.

Unlike FinBen at 71.5% and BeaverTails at 99.8%, CUAD is clean: only 4 contract
pairs exceed 0.90 similarity and none of them cross the official split.

## The three rows settled from the literature rather than by spending

No LLM was run on these. The numbers below are PUBLISHED, on their authors'
own protocols, so they are weaker evidence than the paired same-items
comparisons elsewhere in this project -- the same caveat FINDINGS section 7
raises about repeated case-study numbers, and the same one the LegalBench arm
attaches to its published column.

**CLINC150 — a tie, so take the free one.** A 2026 decision-framework paper runs
the full 5,500-item test split including its 1,000 out-of-scope examples, with
bootstrap CIs and paired significance tests:

| model | accuracy | 95% CI |
|---|---|---|
| fine-tuned RoBERTa | 89.1% | [88.3, 89.9] |
| Claude Haiku, zero-shot | 88.5% | [87.7, 89.3] |

Difference +0.6 points, p=0.24, not significant. Our fine-tuned encoder scores
**88.5%** on a 1,999-item sample of the same split at a near-identical 19.2%
out-of-scope share -- landing on both published figures. A separate systematic
evaluation of 41 open-weight models (135M-9B) puts the zero-shot cohort average
at **0.468** with only 3 of 41 clearing 80%, so the tie is specifically with a
strong hosted model, not with LLMs generally.

**dair-ai emotion — the free arm wins.** Published on the same 16k/2k six-class
split: zero-shot LLaMA 3.1 **11.9%**, the same model QLoRA-fine-tuned **90.9%**.
Our encoder is **92.4%** and our TF-IDF row **89.0%**. Treat the 11.9% with
suspicion rather than as a trophy: it sits far below the 34.7% majority
baseline, which is the signature of an output-format mismatch rather than
inability -- exactly the trap `legalbench_map/POST_DRAFT.md` documents, where a 2026 model answered
correctly in prose and a 2023 exact-match rule scored it wrong.

**GoEmotions — the free arm wins, and nobody does well.** Published macro-F1:
fine-tuned BERT **0.528**, ChatGPT zero-shot **0.256**. Our TF-IDF row is
**0.450** macro-F1, so roughly 1.8x the zero-shot model while still short of the
published fine-tuned figure. For scale on how hard fine-grained affect is for
frontier models, a 2026 zero-shot evaluation on a 13-class emotion taxonomy puts
Gemini 2.5-flash at 39.9% accuracy / 0.363 macro-F1, GPT-5.4 at 38.8% / 0.291
and Claude Sonnet 4.6 at 38.0% / 0.159.

**What would still be worth the money.** CLINC150, because a tie decided on
different protocols is the weakest kind of tie, and ours is the row where a
paired same-items run would most change the story. About 26 cents with one cheap
model.

### Sources

- CLINC150 head-to-head: <https://arxiv.org/html/2608.20371>
- CLINC150 41-model zero-shot sweep: <https://arxiv.org/html/2607.27421>
- GoEmotions BERT vs ChatGPT: <https://arxiv.org/html/2403.06108v2>
- Fine-grained emotion, frontier zero-shot: <https://arxiv.org/html/2607.00968>

"No useful middle" is the only interesting negative: the dial works at the ends
but nothing is gained by setting it partway, because the free model cannot tell
which of its own answers are wrong. That is what the check below measures.

## The check

Run on the held-out test split, **on the row you would actually deploy**:

| signal | what it is | gate |
|---|---|---|
| `AUROC(confidence -> correct)` | how well max predicted probability ranks right answers above wrong ones | **>= ~0.75.** 0.5 means confidence says nothing about correctness, so routing on it selects items at random |
| errors in the least-confident 20% | ceiling on what a 20% escalation budget can repair even with a perfect oracle | **well above 20%,** which is what random selection returns |

Reference values from rows that pass: banking77 0.905 / 80%, CLINC150 0.875 /
62%, dair-ai emotion 0.819 / 57%.

**Calibration gap is NOT a gate.** An earlier version of this file used it. The
CoLA encoder is 15 points overconfident and routes perfectly well, because
ranking correctness and being calibrated are different properties. Drop it.

## The mistake worth recording: measure the row you would ship

CoLA was first excluded on the TF-IDF row. That row scores **below its own
majority baseline**, so of course its confidence was uninformative -- a model
with no signal has no signal to be confident about. Checking the row anyone
would actually deploy reversed the verdict: the fine-tuned encoder scored
**81.4%** against a 68.9% baseline, with AUROC 0.790 and 47% of its errors in
the least-confident fifth, which passes.

**Calibration gap is NOT a gate.** An earlier version of this file used it. The
CoLA encoder was 15 points overconfident and routed perfectly well, because
ranking correctness and being calibrated are different properties.

### CoLA has since been REMOVED from the set

Not on this evidence. Removed at the maintainer's direction, for two reasons
that stand on their own: GLUE is a 2018 benchmark rather than something current
LLM leaderboards run, and a 2-class task cannot describe a class distribution --
its "0 of 2 classes above 90%" reading was an artifact of the bar, since the
majority class sat at 85.4% precision over 71% of the traffic and cleared an
85% bar comfortably.

What it measured while it was in the set, for the record:

| CoLA row | accuracy | majority | note |
|---|---|---|---|
| TF-IDF + logreg | 0.628 | 0.689 | below a constant answer at every training size |
| frozen embeddings | 0.542 | 0.689 | worse still; embeddings encode meaning, not well-formedness |
| fine-tuned encoder | **0.814** | 0.689 | least-confident 20% scored 56.4%, most-confident 20% scored 97.6% |

The transferable finding: grammatical acceptability is syntactic, so bag-of-words
cannot represent it, and the fix was a different representation rather than an
LLM.

## ToxicChat — the replacement, and the strongest result in the set

`lmsys/toxic-chat` (toxicchat0124), 4,978 train / 4,765 test after dedup, 6.8%
toxic. Current guardrail evaluation built from real user prompts, and the
incumbent on this job is not a chat model being quizzed but a moderation API
doing exactly this classification in production.

It ships `openai_moderation`: OpenAI's own per-category scores for every item.
So the paid incumbent's predictions were already on disk, and the comparison is
paired, same-items, and cost nothing.

| row | accuracy | balanced accuracy | F1 on toxic | AUPRC |
|---|---|---|---|---|
| OpenAI moderation (incumbent) | 0.9393 | 0.5684 | 0.2375 | 0.6148 |
| always answer "safe" | 0.9322 | 0.5000 | 0.0000 | — |
| TF-IDF + logreg | 0.9536 | 0.8100 | 0.6531 | 0.7227 |
| frozen embeddings | 0.9242 | 0.9019 | 0.6106 | 0.7584 |
| **fine-tuned encoder** | **0.9664** | 0.8528 | **0.7444** | **0.8227** |

Read the F1 and AUPRC columns, not accuracy: at a 6.8% base rate, answering
"safe" every time scores 93.2% while catching nothing, and the incumbent's
93.9% is barely above that. On the metric that matters the free encoder is
**3.1x** the incumbent's F1, and threshold-free AUPRC says the same, 0.823
against 0.615, so the conclusion does not rest on where the flag is set.

**State the reason fairly.** This is not "OpenAI's classifier is bad". A
moderation API is trained for its own content-policy taxonomy, not for this
dataset's definition of toxicity, and ToxicChat exists partly to show that gap.
The finding is the project's thesis in its purest form: **an off-the-shelf LLM
classifier does not match your labels, and 5,000 of your own labels beat it by a
wide margin.**

Cascade pre-flight, comfortably passed, and the best routing signal measured
anywhere in this project:

| row | AUROC | errors in least-confident 20% | kept @ 20% escalated |
|---|---|---|---|
| TF-IDF | 0.875 | 79% | 98.8% |
| frozen embeddings | 0.866 | 70% | 97.1% |
| fine-tuned encoder | **0.902** | **84%** | **99.3%** |

## TweetEval sentiment — no useful middle setting, on either base model

The same re-check was run rather than assumed, and the verdict held:

| TweetEval base model | accuracy | majority | AUROC | errors in low 20% |
|---|---|---|---|---|
| TF-IDF + logreg | 0.604 | 0.487 | 0.651 | 28% |
| fine-tuned encoder | 0.674 | 0.482 | 0.668 | 30% |

A better base model buys 7 points of accuracy and almost nothing of routing
signal. Errors stay spread evenly across the confidence range instead of
collecting in a tail, so no gate beats picking at random. Three measured causes:

1. **Cue sparsity.** The model learns the right words (positive: happy, great,
   good, best, love, nice, fun; negative: sad, worst, hate, worse). Only **8.8%
   of test tweets contain one.** Accuracy is 67.4% on those 175 items and 59.2%
   on the other 1,825. The cues work; they are almost never present.
2. **A majority class defined by absence.** `neutral` is 49% of the data and has
   no lexical signature, so the model's top "neutral" features are filler
   ("do you", "is there", "last day"). The largest single error cell is gold
   `negative` predicted `neutral`: 240 items, 12% of the test set.
3. **Temporal drift worth 6.4 points.** The same fitted model scores 66.4% on
   data held out of TRAIN and 60.0% on the official TEST split, which comes from
   a later SemEval round. A sixth of the apparent difficulty is not difficulty.

**The ceiling is real, not a data shortage.** The encoder scores 67.3% on 12,000
labels and 67.5% on all 45,586. Nearly four times the data for two tenths of a
point.

## What the chart pages carry

banking77, Financial PhraseBank, CLINC150, GoEmotions, dair-ai emotion,
ToxicChat (on its encoder base), and LegalBench with its 37 tasks treated as
classes. GoEmotions is the marginal one, AUROC 0.698 with 29% error recall, but
its class-based routing gains are real.

Three rows now carry a MEASURED incumbent rather than a break-even estimate:
banking77 (82.9%), Financial PhraseBank (98.8%) and ToxicChat (93.9% accuracy
but 0.2375 F1). LegalBench carries published per-task scores. The remaining
rows still need an LLM run to complete the comparison.

## Known limitation of those pages

Every panel except CoLA is built on the TF-IDF row, because that is the only row
whose per-item confidences were saved. `encoders.run_finetuned_encoder` takes
`logits.argmax(-1)` and discards the distribution, so the strongest free row
cannot be asked how sure it was. The encoder beats TF-IDF on 6 of 7 datasets, so
the routing curves shown are a floor rather than the best available. Persisting
per-item confidences from every free row is the fix.
