# Decisions taken during the build, with the measurement behind each

Anything here overrides the original specification. Each entry names what was
measured, on what, and why the decision follows.

---

## D1. The NOVEL split keys on `targets[0]`, not `scenario["subflow"]`

**Found by:** the compile.py agent, as a hard `ContractViolation` from
`build_partitions` on real data ("convo 3695: scenario subflow 'timing_4' is not
in the ontology").

**Measured independently on the full corpus** (train+dev+test, 10,042
conversations, 220,983 turns):

| fact | value |
|---|---|
| ontology subflows | 55 |
| distinct `scenario["subflow"]` values | **96** |
| of those, NOT ontology subflows | **50**, covering 2,172 convos = **21.6%** |
| `targets[0]` outside the 55 | **0 / 220,983 turns** |
| conversations where `targets[0]` varies across turns | **0 / 10,042** |
| FAQ scenario ids mapping to >1 gold intent | **0 / 50** |

The 50 stray values are FAQ query ids (`timing_4`, `pricing_3`, `boots_how_2`,
`jacket_other_4`) drawn from
`ontology["values"]["enumerable"]["single_item_query" / "storewide_query"]` --
not subflow names.

**Decision.** Key the NOVEL partition on `targets[0]`.

**Why this and not the FAQ->parent mapping** (which the measurement shows is also
unambiguous, 0/50 conflicting): the NOVEL partition exists to hold out intents
the model has never seen, and `targets[0]` IS the label head H2 predicts. Keying
on `scenario["subflow"]` would hold out a field the model never predicts and
which only partly aligns with the label, so `novel_escalation_rate` (spec 8.4)
would not measure what the spec says it measures.

**Consequences for `data.py`:**
1. Select the 5 NOVEL intents from the 55 ontology subflows via `targets[0]`,
   seeded by `data.novel_subflows_seed` (13). Assert one distinct `targets[0]`
   per conversation and fail loudly if that ever stops holding -- it is true
   today and is load-bearing.
2. **Remove the hard `ContractViolation`** on `scenario["subflow"]` not being in
   the ontology. It is not an error; 21.6% of the corpus is legitimately an FAQ
   query id. Validate `targets[0]` instead, where 0 exceptions is the real
   invariant.
3. Keep `scenario["subflow"]` as a recorded field and also record the derived
   `faq_parent_intent` mapping -- for reporting, never for splitting.
4. Write the 5 chosen NOVEL intents into the manifest (spec 3.4).

---

## D2. Context is the FULL THREAD by default (was K=6)

**Measured** on the real train split, over 105,175 predicted agent turns:

| speaker of the immediately preceding turn | share |
|---|---|
| customer | 50.5% |
| action (tool result) | 22.5% |
| agent | 20.4% |
| nothing -- conversation opener | 6.6% |

**49.5% of the turns we must predict have no customer message before them.** A
last-user-message-only model has no input at all on half the dataset, and a short
window starves it.

Context length (multilingual-e5-small tokenizer, 6,609 turns / 500 convos):

| K | mean | median | p90 | p99 | over 512 |
|---|---|---|---|---|---|
| 4 | 52 | 51 | 76 | 105 | 0.0% |
| 6 (old default) | 74 | 77 | 108 | 140 | 0.0% |
| 10 | 110 | 120 | 166 | 210 | 0.0% |
| **full** | **174** | 158 | 334 | 551 | **1.5%** |

K=6 used 74 of a 512-token budget -- 85% of the input went unused.

**REVISED (D2b, below): the window is PER-HEAD, not global.** `context_turns_K: full`
remains the default for the intent head only. `K` may be an int or the string `full`; no code may hard-code 6
or assume short contexts. The thread is THREE-WAY (customer / agent / action):
29,190 action turns carry tool results, and 22.5% of predictions follow one.

---

## D3. Latency is a budget, not a gate

**Maintainer, verbatim:** "20ms is not so strict.. no one will notice 20ms vs
45ms.. what we need is a great engine!! it would be 100X faster than an LLM
anyways."

**Measured** (CPU, 4 threads, batch 1, median ms; machine contended by the build,
so absolute values are pessimistic and the ratios are the robust reading):

| tokens | context | ModernBERT-base (149M) | bge-small (33M) |
|---|---|---|---|
| 74 | K=6 mean | 34.3 | -- |
| 174 | full mean | 55.6 | 19.8 |
| 334 | full p90 | 95.0 | 30.6 |
| 551 | full p99 | 162.4 | -- |

Note that spec section 9 criterion 5 (p95 <= 20ms) was **already missed at the
spec's own default** -- 34ms at K=6, before any context change. The binding
constraint was encoder size, not context length.

**Decision.** Keep the strongest encoder that fits (`ModernBERT-base`).
`latency.target_p95_ms: 45`, `latency.gate_on_latency: false`. Criterion 5 is
recorded as measured-and-missed, never as failing the build. `bge-small-en-v1.5`
is demoted to the E3c ablation arm.

int8 dynamic quantization could NOT be tested here (`torch.quantization.
quantize_dynamic` fails on this platform with "Didn't find engine for operation
quantized::linear_prepack"). Untested, not ruled out.

---

## D4. A cheap baseline must be beaten before the encoder is justified

**Measured** on real ABCD dev (2,500 train convos, K=6 context, TF-IDF+logreg):

| target | constant | structured only | TF-IDF | TF-IDF + structured |
|---|---|---|---|---|
| nextstep | 0.7227 | 0.7369 | **0.8063** | 0.8136 |
| intent (55-way) | 0.0346 | 0.1840 | **0.6141** | 0.6192 |
| action | 0.7227 | 0.7313 | **0.8040** | 0.8226 |

Structured features alone (turn position, speaker history, prior actions, counts)
are nearly worthless -- +1.4 points over a constant on nextstep, +0.9 on action.
**The signal is textual.** An MLP over engineered features has almost nothing to
work with; the architecture question is about text representation.

**Decision.** TF-IDF + logreg runs as Arm B0, BEFORE the encoder. The 149M
encoder must beat the row above by a margin worth having. A separate architecture
probe (MLP over frozen turn-vector sequences vs fine-tuned encoder vs TF-IDF, on
one frozen shared dataset) is measuring this properly, including a turn-order
shuffle control to test whether sequence modelling is needed at all.

---

## D5. Standing project findings that constrain this build

Carried in from the wider project; not re-litigated here.

- **A cascade is a cost dial, not an accuracy device.** Measured out-of-fold, no
  gate ever beat the better base arm; a learned router landed within -0.3/+1.4 of
  it and never above both. The gate's job is coverage, not accuracy.
- **One confidence score cannot do two jobs.** Where a score is already spent on a
  scope/abstain decision, it is no longer available for routing -- measured on
  HINT3, where the shipped pipeline's routing AUROC inverted to 0.240. Keep the
  two uses separable and measurable.
- **Constant-predictor guard.** Every headline metric is reported beside what a
  label-blind constant scores; a metric a constant wins is dropped, not caveated.
  ABCD's `nextstep` is 1:1 with the speaker label, so a constant scores 0.7227
  and looks deceptively strong.
- **Fine-tuning can lose to freezing.** Measured on HINT3 and on MASSIVE, where a
  fine-tuned multilingual encoder scored below its own frozen version because it
  failed to converge. Always check the training loss before attributing a result
  to architecture.


---

## D2b. REVISION to D2 — the context window is PER-HEAD, and the two directions oppose

D2 set one global window. Measured on the frozen probe harness (38,795 train /
13,284 dev rows, TF-IDF+logreg, identical rows throughout), that is wrong: the
best window differs by target and points in OPPOSITE directions.

| window | nextstep | intent | action |
|---|---|---|---|
| full thread | 0.7861 | **0.8074** | 0.8013 |
| k6 (last 6 turns) | **0.8071** | 0.6153 | **0.8086** |
| first 2 turns only | 0.7066 | 0.5531 | 0.7213 |
| first2 + last6 | 0.8025 | 0.7293 | 0.8082 |
| first4 + last6 | 0.7911 | 0.7863 | 0.8047 |
| label-blind constant | 0.7227 | 0.0226 | 0.7227 |

**intent** is a conversation-level property -- what the customer came for. Full
thread is worth **+19.2 points** over k6.
**nextstep** and **action** are local decisions -- what to do at this instant.
Full thread COSTS 2.1 and 0.7 points; distant context is dilution, not signal.

**The head+tail shortcut does not substitute.** The hypothesis was that intent is
decided by the customer's opening statement, so "first few + last few" would buy
the intent gain cheaply. It does not:
- first 2 turns ALONE score 0.5531 on intent -- *below* even k6. The opening is
  not sufficient.
- first2+last6 recovers to 0.7293, still **7.8 points short** of full thread.
- first4+last6 reaches 0.7863, still **2.1 short**, and now costs nextstep 1.6.

So the intent evidence is DISTRIBUTED across the whole conversation rather than
concentrated at either end -- an ABCD dialogue reveals its subflow progressively.
There is no cheap asymmetric window that serves all three heads.

**Decision.** Per-head context:
- intent head (H2): FULL THREAD
- nextstep (H1), action (H3), values (H4): last 6 turns
- skeleton (H5) and template (H7): not yet measured -- measure before choosing,
  do not inherit a default.

Implementation: encode the full thread once and give the local heads a
suffix-masked pooling, or run the encoder twice. Measure both; the first is
cheaper, the second is simpler and latency is no longer a gate (D3).

**Caveat that travels with these numbers.** first2+last6 lands within the CI of
k6 on nextstep (0.8025 vs 0.8071, CI half-width ~0.68 points at 13,284 rows) and
ties on action. If a single global window is ever required for engineering
reasons, first2+last6 is the least-bad one -- but it gives up 7.8 points of
intent to buy that simplicity.

---

## D6. The intent baseline briefed to the architecture probe was wrong

Recorded because it changed a conclusion, and because the failure mode recurs.

Arms were briefed with `intent 0.6141` as the TF-IDF bar. That figure was measured
at **K=6 only**. On identical rows at full thread, plain TF-IDF+logreg reaches
**0.8074**. An arm reporting 0.72 and claiming to beat the reference was beating a
baseline crippled by its context window.

The error was invisible because the harness validated against the old numbers *at
the same K* (k6 here reproduces the original 2,500-convo run to within 0.5 points
on all three targets). A baseline is only a baseline WITH its configuration
attached.

Honest per-target bars, each at its best window, for anything scored on this
harness (digest `08ad053a37a67cc36aeec16e0a02306c`):

| target | constant | TF-IDF best | window |
|---|---|---|---|
| nextstep | 0.7227 | 0.8071 | k6 |
| intent | 0.0226 | 0.8074 | full thread |
| action | 0.7227 | 0.8086 | k6 |

**And this reframes D4.** TF-IDF is a bag -- unordered unigrams and bigrams, no
token order, no cross-turn entity binding, no pretraining -- and it now sits at
~0.81 on all three targets. A larger gain came from changing the WINDOW than
anything yet obtained from changing the MODEL. Whatever an encoder buys on ABCD
must be bought on top of 0.81.

---

## D7. Turn ORDER carries large signal — for local heads only. And the cheapest
## order-aware representation beats every neural arm tried.

This supersedes the framing of D4 and strengthens D2b. It is the answer to the
question that started the architecture probe ("why an encoder and not an MLP over
vectors?"), and the answer is neither.

### The control that could fail

Two earlier order controls were **incapable of failing** and both were nearly
reported as evidence:

1. The attention probe shuffled turn order and tested MEAN POOLING, which is
   permutation-invariant by construction: `dev_natural == dev_shuffled` to every
   decimal. Arithmetic, not a finding.
2. The maintainer then proposed shuffling turn lines and refitting plain TF-IDF.
   Measured before running: word unigrams are **exactly 0.00%** order-sensitive
   (permuting lines cannot change a multiset of tokens) and only **5.5%** of
   uni+bigram mass can respond at all (the bigrams straddling turn junctions).
   Run anyway as a check: nextstep -0.10, intent +0.03, action -0.42 — all inside
   CI. A null from that control would have meant nothing, and would have been
   read as "order does not matter".

The version that CAN fail: **recency-tagged TF-IDF.** Every word token is
prefixed with the recency bucket of its turn (r0 = most recent ... r6 = 6+ back),
then the same word 1-2gram TF-IDF + logreg, same C. Same words, same model. The
only addition is that it can distinguish "refund in the last turn" from "refund
six turns ago" — so it is not permutation-invariant and shuffling can hurt it.

### Result (dev n=13,284, CI half-width 0.68 pts, digest 08ad053a37a67cc36aeec16e0a02306c)

| target | plain TF-IDF | recency-tagged | tagged, scored SHUFFLED | order effect |
|---|---|---|---|---|
| nextstep | 0.7855 | **0.8287** | 0.7055 | **-12.3 pts** |
| intent | 0.8078 | 0.7981 | 0.7883 | -1.0 pts |

**Order is worth 12.3 points on nextstep and 1.0 on intent.** That is the SAME
local-vs-global split that made the best context window differ per target
(D2b: intent wants the full thread, +19.2; nextstep wants k6, and the full thread
costs it 2.1). Two independent axes now agree about the same two heads.

So D2b understates the case: **it is not only window size that is per-head, it is
whether sequence is modelled at all.**

### The cheapest order-aware representation wins outright

Recency-tagged TF-IDF scores **0.8287 on nextstep** — +2.2 points over the best
plain TF-IDF bar (0.8071 at k6), outside CI, and above every neural arm measured.
The 33.7M-parameter frozen-encoder MLP scores 0.7900 on the same target at ~50x
the latency. Tagging is still TF-IDF + logreg: no pretraining, no encoder, no
attention, ~0.3 ms/turn.

Conversely, tagging HURTS intent (0.7981 vs 0.8078) — consistent with intent being
order-invariant.

### The reframing

The neural arms were not beaten by "TF-IDF the bag". They were beaten by a cheap
representation carrying the RIGHT INDUCTIVE BIAS for its target: recency for a
local decision, a whole-thread bag for a global one. The useful question is not
"encoder or MLP" but **"which bias does this head need, and what is the cheapest
thing that encodes it"**.

### Correction to D4

D4 said the cheap arm wins on all three targets. That overstates one cell: on
intent, logreg on the frozen 384-d vector scores 0.8046 against TF-IDF's 0.8074 —
a 0.28-point gap, INSIDE the 0.68 CI. That is a TIE. The MLP losses
(-1.7 / -1.7 / -2.8) and the nextstep/action linear-probe losses are real.

### Open at time of writing

- `action` not yet measured on this axis. If it behaves like nextstep the
  local/global split is clean across all three heads.
- Recency-bucket granularity (r0..r6) is one arbitrary choice; unmeasured.
- Tagged + k6 on nextstep untested — nextstep already prefers k6, so the two may
  stack above 0.8287.

### D7 continued — completed, and independently verified

**ACTION behaves like nextstep. The local/global split is clean across all three
heads.** Recency-tagged TF-IDF, full window, three conditions:

| target | nat->nat | nat->shuf | **shuf->shuf (REFIT)** | signal destroyed |
|---|---|---|---|---|
| nextstep | 0.8287 | 0.7055 | 0.7712 | **-5.8 pts** |
| action | 0.8136 | 0.7263 | 0.7671 | **-4.7 pts** |
| intent | 0.7981 | 0.7883 | 0.7904 | -0.8 pts (inside CI) |

The **shuf->shuf column is the actual experiment.** nat->shuf alone shows only a
train/test mismatch. Refitting on scrambled data — giving the model every chance
to relearn — and still losing 5.8 and 4.7 points proves the information is
DESTROYED, not merely misaligned. intent loses 0.8, inside CI: it genuinely does
not care about order.

### The ship configuration, verified twice

**nextstep, k6 window, recency-tagged r0..r3 = 0.8351** (agent) /
**0.8345** (maintainer, independent re-measurement). 0.06 pts apart.
Supporting cells also reproduced: plain k6 0.8071 vs 0.8062, tagged full 0.8296
vs 0.8281-0.8297.

| config | nextstep |
|---|---|
| plain full | 0.7855 |
| plain k6 | 0.8071 |
| tagged full | 0.8296 |
| **tagged k6 (r0..r3)** | **0.8345-0.8351** |

That is **+2.8 over the best plain TF-IDF bar** and **+4.5 over the 33.7M-param
frozen-encoder MLP** (0.7900), both far outside CI, at ~0.3 ms/turn with no
encoder. The two cheap biases COMPOSE: local window and recency tagging stack.

### Granularity saturates at 3-4 buckets, and most of it is ONE BIT

| MAXB | none | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|
| full | .7855 | .8162 | .8266 | .8281 | .8290 | .8287 | .8297 | .8282 |
| k6 | .8062 | .8307 | .8345 | **.8351** | .8318 | .8301 | .8301 | .8301 |

**MAXB=1 is just "is this the most recent turn, yes or no" — one bit — and it
buys +3.1 on full and +2.5 on k6.** Most of a 12-point order effect is carried by
a single bit of recency. Past 4 buckets it is flat and the vocabulary cost is
pure loss (full goes 56k -> 148k features). Expose it as a tunable; default
r0..r3; do not search it hard.

Sanity check that the bucketing does what it claims: k6 at MAXB 6/8/12 is
byte-identical, because a 6-turn window only has 6 positions.

The order effect GROWS with granularity (+0.77 plain -> +12.9 at MAXB=8) — a
model that can see more order has more to lose. That is a non-tautological
control behaving correctly.

### It is NOT a repackaged prev-speaker feature

|  | plain | plain + struct | tagged r0..r3 | tagged + struct |
|---|---|---|---|---|
| full | 0.7855 | 0.7972 | 0.8281 | 0.8307 |
| k6 | 0.8062 | 0.8130 | 0.8351 | 0.8356 |

Explicit prev-speaker + n_prior_turns features buy +1.17 / +0.68. Recency tagging
buys +4.26 / +2.89 — about **4x more** — and the structured features add nothing
on top (+0.26 / +0.05, both inside CI). **Tagging subsumes prev-speaker**: it is
the recency profile of the whole vocabulary, not the identity of the last speaker.

### THE CAVEAT, which ships at the same prominence as the result

**This does NOT show that encoders are useless on ABCD.** The encoder arms lost
because a frozen mean-pooled whole-context vector DESTROYS recency — it is
permutation-invariant, the same property that made the meanpool shuffle control
vacuous — while also diluting the global signal. That is a REPRESENTATION
failure, not evidence that pretraining cannot help. **A sequence model given the
same recency bias and trained to convergence has not been fairly tested.**
Anyone reading "cheap wins" out of this is over-reading it.

### Consequence for the main build

The spec has one global context window and a transformer for every head. On three
independent axes — window size (D2b), order sensitivity, and order destruction
under refit — the same split holds:

- **nextstep (H1), action (H3): local and ordered.** k6 window, recency-tagged.
- **intent (H2): global and unordered.** Full thread, untagged.
- skeleton (H5), template (H7): not yet measured on any axis. Measure before
  choosing; do not inherit a default.

---

## D9-D14. Findings from the core build batch (recorded before they are lost)

The batch died on a session limit with 3 of 12 agents reporting cleanly, but the
surviving agents returned findings that must not be lost. `train.py` (1,169 lines)
and `evaluate.py` (1,720) also landed despite being marked failed.

### D9. `models.py` was CORRUPT, not a stub — and it hid from every tool

The killed earlier batch left a 1,182-line `models.py` containing a **raw NUL byte
at line 227**, where `"\x00".join(ids)` had been written with a literal NUL
instead of the escape. Consequences worth remembering:
- `file` reported it as **`data`**, not Python.
- **every `grep` silently returned nothing** — which is why an earlier status check
  read it as a 34-line stub.
- It was unimportable.

Repaired byte-wise; the rest of the file was sound and was kept. **Lesson: a
`grep` returning nothing is not proof of absence — check the file type.**

### D10. REAL BUG FOUND AND FIXED: H7 trained against a query no other head saw

`_template_loss` re-encoded the context from scratch instead of reusing
`outputs["query"]`. Two consequences, the second serious:
- 3 live-encoder passes per training step instead of 2.
- **With dropout live in training, H7's InfoNCE ran against a DIFFERENT dropout
  sample than H1-H6** — the retrieval head was learning against a query the rest
  of the model never produces.

Fixed to read `outputs["query"]`. Two regression tests added (encoder-pass count;
identity of the query H7 scores).

### D11. OPEN — cross-module interface mismatch that will raise at runtime

`calibrate._encode_contexts` calls `model.encode_context` with a `list[str]`
(falling back to `(texts, cfg)` / `texts=`), but `models.ModernBERT.encode_context`
takes tensors (`input_ids`, `attention_mask`). One side must move or calibration
raises `ContractViolation`. **Not fixed — no integration agent ran.**

### D12. Dedup must NOT use the task encoder — measured, not assumed

`dedup_templates` uses a sentence-trained MiniLM, not `model.encoder`, and the
contract's suggestion to use the latter is wrong. Measured on 15 real ABCD
sentences (105 pairs):
- raw ModernBERT-base puts **59% of ALL pairs at cosine >= 0.92** (CLS; 29%
  mean-pooled), and scores *"what is your membership level?"* against *"the
  shipping fee is $4.99 per item."* at **0.961**.
- Single-linkage at 0.92 on those embeddings would **fuse the entire bank into one
  cluster**.
- Same 105 pairs under MiniLM: **0 pairs >= 0.92**; true paraphrases land 0.80-0.82.

A raw MLM encoder's cosine is not a paraphrase metric. This generalises beyond
this project.

### D13. The truncation figure in D2 was measured WITHOUT the state line

D2 recorded "p99 551, 1.5% over max_len" for K=full. Re-measured on all 13,392
dev agent-side contexts with the ModernBERT-base tokenizer:

| context | mean | p99 | over max_len |
|---|---|---|---|
| turns only | 168.6 | 512 | 1.00% |
| **turns + state line** | **223.7** | **646** | **4.05%** |

Four times the truncation. Mitigation adopted: `ContextWindow.text` puts the state
line LAST and collation uses `truncation_side='left'`, so overflow drops the
OLDEST turns rather than the state or the most recent turns.

### D14. Two deliberate deviations from the official baseline, both documented

1. **The synthetic end turn carries `action=None`/`values=None`.** `process.py`
   copies all five target slots, but `CDSProcessor.collect_one_example` then fills
   `action_id`/`value_id` only under `if nextstep == 'take_action'`, emitting
   -1/-1. Since the official `cds_report`'s action denominator is
   `sum(bslot_label >= 0)`, inheriting the button would add 10,042 turns to that
   denominator and make action accuracy **incomparable to published numbers**.
   Intent and turn_count ARE inherited.
2. **H4 carries a deliberate off-by-one** relative to the official
   `convert_context_tokens`, which prepends [CLS] to the copy-context input
   without shifting the copy index. Documented rather than silently matched.

### D15. No measured act-labeling accuracy exists in this build

Both `delex_check.md` and `act_check.md` carry an explicitly labelled
automated/self-assessed number and a **blank human tally**. No human has checked
either sample. Any claim about act-labeling quality is currently unsupported.

---

## D16. "Template coverage 61.01%" is NOT a ceiling on reflex rate. It is a
## fidelity rate — and the operationally dangerous number is 4.92%.

Measured directly from `outputs/compile/labels/train.jsonl` and
`bank/templates.jsonl` while the runtime modules were being built.

### What 61.01% actually is

Every one of the **71,133** retrieve turns is assigned templates. Zero nulls,
zero empty lists, 100% assigned. So the fast path can always emit *something*;
61.01% is the share of utterances the bank reconstructs **exactly**, not the share
it can answer. Reflex RATE is not capped at 61%. Reflex **FIDELITY** is.

That distinction matters for how the headline is reported: DEFECTS_OPEN.md and
earlier notes call 61.01% "the CEILING on reflex rate". **That framing is wrong
and must not ship.**

### The number that actually bites

Single-linkage dedup merges 3,374 of 4,417 templates, covering **90.8% of all
template occurrences** (53,150 / 58,528). Within clusters whose canonical requests
named fields (account id, order id, username, email, ...), measured by comparing
the requested field-set of each merged surface form against its canonical:

| measure | value |
|---|---|
| field-requesting merged forms | 5,620 |
| forms whose requested FIELD-SET differs from canonical | **1,803 = 32.1%** |
| occurrence-weighted divergence | 3,501 / 11,169 = **31.3%** |

Propagated to turns:

| | turns | share of 71,133 |
|---|---|---|
| touch a cluster that CAN ask for the wrong fields | 7,628 | 10.72% |
| **EXPECTED to ask for the wrong fields** | **3,499** | **4.92%** |

### Why this is a functional error, not a style difference

From the 988-form cluster, canonical `"can i have your account id and order id?"`:

    merged form: "what is your order id?"                        -> asks for LESS
    merged form: "and your username, email, and order id?"       -> asks for DIFFERENT fields
    merged form: "can i get your username, email address and order id?"  -> DIFFERENT again

If the fast path emits the canonical where the human asked only for an order id,
it demands an account id the customer was never asked for. If it emits the
canonical where the human asked for username and email, the customer supplies the
wrong identifiers and the next action fails. **~1 in 20 of all retrieve turns.**

### Decision

1. Report **three** numbers, never one: assignment rate (100%), exact-reconstruction
   fidelity (61.01%), and **expected wrong-field rate (4.92%)**. The third is the
   one a customer would feel.
2. Stop calling 61.01% a ceiling on reflex rate anywhere in the repo.
3. The gate must treat a high-divergence template as a reason to escalate. A
   cluster whose forms disagree about *which fields are being requested* is
   exactly the case where the free model should not answer. This is a new gate
   signal, not covered by conformal confidence — the model can be perfectly
   confident in a template whose cluster is internally inconsistent.
4. `dedup_threshold: 0.92` with single-linkage is the cause. Do not change it
   without measuring: raising it fragments the bank and lowers fidelity; the
   principled fix is to block a merge when the two forms request different
   field-sets, which is a cheap exact check and needs no embedding.

---

## D17. The field-set merge guard: implemented, and it works

Fix for D16, landed in `compile.py::_union_find_clusters` + `_requested_fields`,
behind `compile.merge_block_on_field_mismatch` (default true) with the surface
phrase -> field map in `compile.merge_field_terms`.

Mechanism: a pairwise union is refused when the two forms request different named
fields. **Field-set equality is an equivalence relation**, so refusing mismatched
pairwise unions is sufficient to make every resulting cluster field-homogeneous —
no post-hoc cluster splitting is needed. It is an exact string check; no embedding.

### Measured, full recompile, before vs after

| | BEFORE | AFTER |
|---|---|---|
| templates in bank | 4,417 | 4,505 |
| merged templates | 3,374 | 3,445 |
| **largest cluster (surface forms)** | **988** | **327** |
| template occurrences | 58,528 | 58,211 |
| **field-set divergence (forms)** | **35.1%** | **0.0%** |
| **field-set divergence (occurrence-weighted)** | **33.2%** | **0.0%** |
| divergent forms | 1,694 / 4,830 | **0 / 4,463** |

The divergence is eliminated, not reduced. Cost: **+88 templates** (clusters
fragment slightly) and **-317 occurrences**. That is a cheap price for removing an
expected ~4.9% of retrieve turns that would have asked the customer for the wrong
identifiers.

Note the pre-fix divergence measured here (35.1%) is slightly higher than the
31.3% first reported, because the corrected `_requested_fields` strips punctuation
before matching — the original missed "order id?" with a question mark and so
undercounted. The first measurement was an underestimate of the problem.

---

## D18. Two YAML traps hit within one hour, both silent, both now guarded

Recording as a pair because they share a cause: **YAML fails quietly, and a
config that parses is not a config that is correct.**

### `#` in an unquoted key truncates the line

`account #: account_id` -> `#` starts a comment -> the line becomes a bare key
with no colon -> `ScannerError`, and EVERY `reflex` command and every test that
calls `load_config` fails at once. Caught within minutes by an agent. Fixed by
quoting: `"account #":`.

### A bare `off` is the BOOLEAN False, not the string "off"

`gate.affect_signal: off` parses to `False`, so code doing
`str(get_dotted(cfg, "gate.affect_signal")).lower() != "off"` compares against
`"false"` and **raises on every single `evaluate_gate` call**. The gate could not
route one turn. Same trap applies to bare `on`, `yes`, `no`, `y`, `n`.

Fixed at the ROOT — `affect_signal: "off"` is quoted with a comment naming the
trap — rather than by a string-coercion workaround in the gate, which would have
hidden the next instance. The whole file was audited: zero other bare
on/off/yes/no values.

### The guard that makes these permanent

Also discovered: **duplicate top-level keys are silently accepted.** Two agents
each added `select:`/`fill:`/`run:`/`report:` blocks; PyYAML keeps only the LAST
and raises nothing, so one agent's entire configuration was dead weight while both
believed theirs was live. Requested for the test suite: a duplicate-key-rejecting
loader that fails if any key at any depth appears twice.

**Transferable lesson:** three distinct silent-failure modes in one config file in
one hour. Config files deserve the same adversarial treatment as code — a parse
test, a duplicate-key test, and a test that every key the code READS actually
EXISTS (twenty such keys were missing here, each a latent runtime KeyError in a
module that had been called "finished" because it imported cleanly).

---

## D19. The E3b "no-availability" ablation is CONFOUNDED and must not be reported as-is

Found by the gate+fill agent while wiring the D16 template-consistency backstop.

`gate.use_availability: false` switches off two distinct mechanisms at once:
1. the availability signal proper (a required slot has no value), and
2. the D16 backstop that escalates a template whose merge cluster is internally
   inconsistent about which fields it requests.

So the E3b arm claims to isolate "availability on vs off" but its delta actually
measures the removal of both. **The number is uninterpretable as labelled.**

Either separate the flags, or report the arm as measuring both and say so. This is
the third instance today of the same class of error: a control that cannot fail
(mean-pool shuffle, permutation-invariant by construction), a defect measured
against data corrupted by a different defect (D-4's five false zeros), and now an
ablation that removes two things while naming one.

**An ablation is only as good as its claim to isolate one variable. Check the
claim, not just the code.**

## D20. `unavailable_slot` now covers two different failures — discriminate them

The D16 backstop routes as `unavailable_slot`, so that reason means either
"a required slot could not be sourced" or "this template's cluster is
inconsistent". They remain separable: **a real slot failure names slots, an
inconsistent template names none** — the discriminator is
`reason == "unavailable_slot" and missing_slots == []`.

The report must break these out as separate rows. Folded together, the escalation
reason distribution misleads.

## D21. Measured escalation from unavailable slots — and why it is CORRECT, not a defect

Measured on 200 train conversations / 2,825 agent-side turns:

| | |
|---|---|
| action turns carrying `required_slots` | 337 |
| of those, at least one slot unavailable | 45 = 13.4% |
| **share of ALL agent-side turns escalating as `unavailable_slot`** | **1.59%** |
| `collect_slot_sources` cost incl. `build_context` | 0.08 ms/turn |

Missing slots: order_id 23, account_id 21, customer_name 7, amount 5, email 5,
name 3, shipping_status 1. By button: verify-identity 24, validate-purchase 8,
offer-refund 5, pull-up-account 4, make-purchase 3, shipping-status 1.

**The cause is the SCENARIO, not the filler.** Across those 200 scenarios
`personal.account_id` is present in only 67 (33.5%) and `order.order_id` in 100
(50%). A slot the scenario does not contain and the customer never stated cannot
be sourced by anything, so escalating is the CORRECT behaviour. Anyone reading
"1.59% escalation" as something to optimise away has misread it.

Counterfactual, which quantifies how little loosening would buy: dropping
`account_id` from verify-identity moves 45 -> 43 (the same turns usually miss
order_id too); dropping account_id AND order_id moves it to 21/337 = 6.2%.

Slot filling is also not a latency item at 0.08 ms/turn.

## D22. DEFERRED DELIBERATELY: the D-4 support-metadata assertion

I asked for an assertion that loaded action patterns carry support metadata. It
**cannot be placed where I specified**: `ActionPattern` (spec 5.4) is frozen at
three fields, `actions.jsonl` therefore carries no support data, and
`check_availability` receives bare slot NAMES with no button attached. I specified
the assertion without checking the schema could express it.

**Deferred, not dropped, and the reasoning is that it is not load-bearing:** D-4
was fixed AT SOURCE (None included in the modal vote; only supported slots
emitted) and that fix carries a regression test proven to fail on the pre-fix
module. The assertion would be a second line of defence against a defect whose
first line is already tested.

A config-gated implementation exists and is OFF (`fill.require_slot_support_metadata`,
`fill.slot_support_path`, `fill.min_required_slot_support: 0.50`), with four tests
covering both states, so the intent is documented for whoever picks it up. Closing
it properly needs compile to write `{button: {slot: support}}` beside the bank.
