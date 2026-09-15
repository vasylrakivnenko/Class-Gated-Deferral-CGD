# Cascade pre-flight: what to check, and what got excluded

A cascade only pays if the base model knows when it is wrong. That is
measurable, so this is a measurement rather than a judgement. One dataset fails
it. A second one appeared to fail it and did not, for a reason worth recording.

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

## TweetEval sentiment — excluded, and it survives the better base model

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

## What the cascade pages carry

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
