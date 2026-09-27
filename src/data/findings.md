# FINDINGS

Non-obvious things this benchmark turned up that you cannot learn from a rate
card, a model card, or a leaderboard. Collected for writeups.

Every entry states what was **measured**, on what, and how to reproduce it.
Where something is unverified or came from a run that is no longer in
`runs/*/results.json`, it says so. Numbers here are from a 250-item held-out
test split that no optimizer ever saw.

**Dates:** all measurements 2026-09-09. Prices are the published rate cards of
that date and will drift.

---

## 1. A rate card said 3x cheaper. The measurement said 1.16x.

**Claude Haiku 4.5 lists at $1/$5 per 1M tokens; Claude Sonnet 4.6 at $3/$15.
Sticker gap: exactly 3x. Measured gap on a real 77-class task: 1.16x.**

| | input tok | cached | measured $/1k calls |
|---|---|---|---|
| Claude Sonnet 4.6 | 2,641 | 1,884 | $3.211 |
| Claude Haiku 4.5 | 2,640 | **0** | $2.765 |

Same prompt, same task, same code path. Sonnet cached 71% of its input; Haiku
cached nothing.

**Why:** Anthropic publishes a *minimum cacheable prompt length* that differs
per model. From their prompt-caching docs:

> 1,024 tokens for Claude Opus 4.8, Claude Sonnet 5, Claude Sonnet 4.6 …
> **4,096 tokens for Claude Haiku 4.5**
>
> Shorter prompts cannot be cached, even if marked with `cache_control`. Any
> requests to cache fewer than this number of tokens will be processed without
> caching, **and no error is returned.**

Our prompt is 2,640 tokens: above Sonnet's threshold, below Haiku's. So the
cheaper model quietly loses its cheapness, and nothing in the response tells
you — you have to notice that `cache_creation_input_tokens` and
`cache_read_input_tokens` are both zero.

**The uncomfortable implication:** there is a band of prompt sizes
(1,024–4,095 tokens) where *upgrading* from Haiku to Sonnet costs you almost
nothing, and possibly less than nothing once you count the accuracy
difference. Nobody finds that by reading pricing pages.

Encoded as `ModelSpec.cache_min_tokens` / `cache_applies()` so a 0% cache rate
comes with an explanation rather than being a mystery.

---

## 2. Turning reasoning off is not free, and turning it on is not always better

Three separate measurements, all on the same held-out split.

### Nemotron 3.5 Lightning 30B-A3B (Fireworks), Financial PhraseBank

| config | accuracy | output tokens | $/1k calls |
|---|---|---|---|
| reasoning on (default) | 93.2% | 1,137 | $0.2407 |
| reasoning off | 86.0% | 24 | $0.0181 |
| reasoning off **+ GEPA** | 91.2% | 15 | $0.0315 |

Turning reasoning off made it **13x cheaper and 7.2 points worse**. Prompt
optimization then bought back 5.2 of those 7.2 points while still costing
**7.6x less** than the reasoning-on configuration.

That is the actual shape of the tradeoff, and none of the three rows is
obviously "correct" — it depends on what a point of accuracy is worth to you.
A benchmark that only reports the default configuration would have shown one
of these three numbers and called it "Nemotron".

### Qwen3 1.7B (local), Financial PhraseBank

Measured on the same 250-item held-out split:

| config | accuracy | output tokens | $/1k calls | wall |
|---|---|---|---|---|
| reasoning off (direct) | 80.4% | 50 | $0.0138 | 102s |
| reasoning on | 78.0% | 285 | $0.0418 | 1,360s |

**5.7x the output tokens, 3.0x the cost, 13.4x the wall time — and no
accuracy improvement at all.** The 2.4-point gap in favour of reasoning-*off*
is itself not significant (McNemar mid-p = 0.54 over 94 discordant items,
95% CI on the difference [-5.2pp, +10.0pp]), so the honest claim is not
"reasoning made it worse" — it is "reasoning changed nothing measurable and
tripled the bill."

But the two configurations are not interchangeable, and plain accuracy hides
why. See §6 — the same pair of runs reverses its ranking under balanced
accuracy.

### Not Kimi

Worth correcting a recollection: **Kimi K2.6 was never run on a dataset**, only
smoke-tested. In that smoke test it emitted 126 output tokens to answer a
one-word classification, and its Azure meter is literally named
`K2.6 Thinking` — so it very likely behaves the same way. But that is an
expectation, not a measurement, and should not be posted as one.

**The general point:** reasoning tokens are billed output tokens. A "cheap"
model that thinks for 300 tokens before saying one word can cost more per task
than a pricier model that answers directly. A $/1M rate card cannot show this;
only measuring the tokens can.

---

## 3. Prompt caching is opt-in on one provider and automatic on another

Measured on the 77-class task, where the prompt carries a ~1,900-token label
list that is byte-identical on every call:

- **OpenAI-family models cached ~60% automatically**, no code change.
- **Anthropic models cached 0%** until an explicit
  `cache_control: {"type": "ephemeral"}` marker was added.

Before the fix, Claude Opus 4.8 measured **$18.62/1k**. After: **$7.67/1k**.

That 2.4x was not a real price difference. It was a missing flag — and a
benchmark that shipped it would have been comparing one vendor's cached cost
against another's uncached cost while calling it a price comparison.

**If you publish cost comparisons across providers, you have to verify that
caching is enabled equivalently on both, because the defaults differ.**

---

## 4. On a realistic 77-class task, every LLM lost to TF-IDF

Banking77 (77 customer-intent classes, 12,701 labelled training rows available):

| | accuracy | $/1k items |
|---|---|---|
| Fine-tuned encoder (68M params) | **94.0%** | $0 |
| TF-IDF + logistic regression | **92.0%** | $0 |
| Frozen embeddings + logreg | **90.4%** | $0 |
| Claude Sonnet 4.6 | 82.9% | $3.45 |
| Claude Haiku 4.5 | 80.1% | $2.76 |
| GPT-5-nano | 78.1% | $0.20 |
| GPT-5.4-nano | 75.3% | $0.19 |
| Qwen3 1.7B | 54.6% | $0.06 |

**Pareto frontier: 1 of 13 candidates.** A free bag-of-words model beats every
frontier LLM by **9.1 points**.

The comparison is not rigged — it is the actual choice facing a team that has
labelled ticket history. But it is asymmetric and should always be stated as
such: the classifier trains on 12,701 examples, the LLMs get zero and must
choose from 77 confusable intents listed in the prompt.

**And the "corrected" column is our own bug, found while auditing this.** Of
banking77's 77 label strings, exactly one is not lowercase:
`Refund_not_showing_up`. Our scorer lowercases the model's answer before
matching it against the label set, so a model that answered that intent
*perfectly* produced `refund_not_showing_up`, matched nothing, and was recorded
as a **parse failure** and scored 0.

Four of the 251 test items carry that gold label, so a correct answer on any of
them was unmatchable **by construction** — the ceiling on what the fix can
recover is 4/251, or **1.6pp**. The classical and encoder rows predict by index
straight out of the label set, never touch the string matcher, and carry zero
parse failures. So the bug penalised **only the LLM arm**, in the one comparison
this section exists to make.

Parse failures on the original sweep ran 5–9 per LLM row, never 4, and the
surplus has a second cause: an item whose *gold* was lowercase but whose
*answer* was the capitalised label was blanked too. Those were already wrong and
stay wrong — but they were logged as "the model returned garbage" when it had
returned a real label, so the counter was overstating garbage across the board.

The conclusion survives — 82.9% still loses to 92.0% by nine points — but the
margin was overstated, and it was overstated in the direction of our own
headline. That is the kind of error a benchmark has to go looking for, because
it will never announce itself: the row just shows a slightly lower number and a
`parse_failures` count that looks like ordinary model sloppiness.

**Re-measured, every LLM row on the task, $3.91 of API:**

| row | before | after | delta | items recovered |
|---|---|---|---|---|
| GPT-5.4-nano | 72.1% | 75.3% | +3.2pp | 8 |
| GPT-5.4-nano + GEPA | 74.5% | 77.3% | +2.8pp | 7 |
| Nemotron 3.5 Lightning 30B | 64.5% | 67.3% | +2.8pp | 7 |
| Claude Haiku 4.5 | 77.7% | 80.1% | +2.4pp | 6 |
| **Claude Opus 4.8** | **79.7%** | **81.7%** | **+2.0pp** | **5** |
| DeepSeek-V4-Flash | 76.5% | 78.1% | +1.6pp | 4 |
| Qwen3 1.7B | 53.0% | 54.6% | +1.6pp | 4 |
| Claude Sonnet 4.6 | 81.7% | 82.9% | +1.2pp | 3 |
| GPT-4.1-mini | 75.7% | 76.5% | +0.8pp | 2 |
| DeepSeek-V4-Flash + GEPA | 78.1% | 78.9% | +0.8pp | 2 |
| GPT-5-nano | 77.7% | 78.1% | +0.4pp | 1 |

**`parse_failures` went to 0 on all eleven rows**, from 5–9 before.

### Do not read that delta column as the size of the bug

The fix can recover **at most four items**. Five rows gained more than four —
GPT-5.4-nano gained eight. So the delta column is not the bug's effect; it is
the bug's effect *plus re-sampling noise*, and the two cannot be separated after
the fact. The `gpt-5*` tiers reject a `temperature` parameter and therefore
sample at 1.0 while every other row is pinned at 0, which is why the largest
surplus sits on exactly those rows.

Read the other direction, this bounds the harness's noise — but be careful what
it bounds. These rows changed the SCORER between measurements, so the spread is
scorer-fix plus re-sampling, not a clean replicate: the fix alone can move a row
at most +1.6pp, and the observed deltas straddle that on both sides, so
re-measuring contributed **up to about ±1.6pp** of the movement.

The only genuinely unchanged-prompt replicate on disk is smaller: **±0.8pp**
(§5, DeepSeek-V4-Flash, 2 of 251 items). Both numbers are one observation each,
not a dispersion estimate — the harness still evaluates every configuration
exactly once, so no row anywhere in this document carries a standard deviation.
On present evidence, a claimed gain under roughly 2pp on a 251-item split is not
distinguishable from running the same thing twice.

Both numbers are kept in `runs/banking77/rerun_record.json`; the new one does not
replace the old.

**All eleven rows now share one scorer**, so the chart no longer mixes two.
Opus 4.8 was re-run last and separately, at $1.93 — more than the other ten
combined — because leaving it out was the cheaper option but left exactly one
row on the chart incomparable with its neighbours, and an annotated asterisk is
a worse artifact than a settled one.

It also moved Opus from *below* Haiku to *above* it, which is the sort of
reordering a scorer bug produces and a reader has no way to detect. Corrected,
Opus lands at 81.7% for $7.67/1k items: **1.2pp below Sonnet at 2.2x the
price.** It is strictly dominated, and that is the point of keeping it on the
chart.

And the cost of *not* re-running it would not have been cosmetic. While Opus
carried the old scorer's number and Sonnet carried the new one, the paired
McNemar test between them read **p = 0.049** — a difference that clears the
conventional 0.05 threshold. Re-scored consistently, the same pair reads
**p = 0.388**, and Holm-Bonferroni does not flag it at any level. The two models
are indistinguishable on this split.

Mixing two scorers across a *paired* test does not merely add noise, then: it
had produced a publishable-looking significant result out of nothing. This is
why the row was re-run rather than footnoted. An annotation warns a reader who
reads annotations; the p-value was wrong for everyone else.

### The same dataset is easy *or* hard depending on label budget

Same 250 held-out items, same code, only the number of labelled training rows
changes. banking77, 77 classes:

| labelled rows | per class | TF-IDF | frozen embeddings |
|---|---|---|---|
| 50 | 0.6 | 19.9% | 33.5% |
| 100 | 1.3 | 35.1% | 46.2% |
| 200 | 2.6 | 48.2% | 55.8% |
| 500 | 6.5 | 65.7% | 67.7% |
| 1,000 | 13 | 76.9% | 74.9% |
| 2,000 | 26 | **85.3%** | 82.9% |
| 5,000 | 65 | 88.8% | 86.5% |
| 12,701 | 165 | **92.0%** | 90.0% |

The 18 prompted LLMs on this task span **54.6% to 82.9%** — Claude Sonnet 4.6
tops them at 82.9%, for $3.45 per 1,000 classifications. A bag of n-grams passes
the entire field somewhere between **1,000 and 2,000 labelled rows**, and from
there it costs nothing per call. That is the whole commercial question — *how
many labels before a trained classifier beats a prompted model?* — and a single
headline accuracy for a dataset hides it completely.

It cuts the other way too. At 100 labels the best free option scores 46.2% and
prompted GPT-5-nano scores 78.1%: the LLM wins by 32 points. Neither "just
fine-tune a small model" nor "just call an LLM" is a fact about the dataset.
Both are facts about the label budget.

On Financial PhraseBank, TF-IDF plateaus at 88.8% and frozen embeddings at
90.8% — neither ever catches the 97.6% of the best prompted model. Only the
fine-tuned encoder does, reaching 97.6% at 1,889 rows.

---

## 5. GEPA's benefit tracks inversely with model strength

Stock vs GEPA-optimized on the same 250 items, paired McNemar (mid-p):

| model | stock → GEPA | p | survives Holm |
|---|---|---|---|
| Qwen3 0.6B | 28.8% → 52.4% | <0.0001 | **yes** |
| Qwen3 1.7B | 80.4% → 90.4% | <0.0001 | **yes** |
| Nemotron 30B | 86.0% → 91.2% | 0.0043 | **yes** |
| GPT-5-nano | 94.8% → 98.8% | 0.0010 | **yes** |
| GPT-4.1-mini | 96.0% → 97.6% | 0.0625 | no |
| DeepSeek-V4-Flash | 95.6% → 97.6% | 0.1094 | no |
| GPT-5.4-nano | 92.8% → 94.4% | 0.4421 | no |
| **Claude Sonnet 4.6** | **95.6% → 95.6%** | 1.0 | no |

Sonnet's zero is clean, not a failure: GEPA proposed one candidate, it tied the
baseline on validation (0.975 → 0.975), so the optimizer correctly kept the
original one-line prompt. Later iterations aborted with *"All subsample scores
perfect for parent."* **Prompt optimization helps where the prompt is the
bottleneck.** On a frontier model doing an easy task, it isn't.

### The ranking under one fixed prompt was noise — and it reversed

Holding one prompt constant across every candidate is how comparisons are made:
it is the obvious way to be fair, and it is what RouterBench, RouteLLM and the
public leaderboards are built on. On Financial PhraseBank it put the winner 4th.

Under the stock one-line instruction:

| rank | model | accuracy |
|---|---|---|
| 1 | GPT-4.1-mini | 96.0% |
| 2 | DeepSeek-V4-Flash | 95.6% |
| 3 | Claude Sonnet 4.6 | 95.6% |
| 4 | **GPT-5-nano** | **94.8%** |

Sonnet's lead over GPT-5-nano is **2 items out of 250** — McNemar mid-p
**0.549**. The problem is not that the fixed-prompt ranking was wrong. It is
that there was no ranking there to be right: 4th and 1st are indistinguishable.

Give each model its own optimized prompt and an ordering appears — pointing the
other way:

| comparison | gap | items | p |
|---|---|---|---|
| Sonnet vs GPT-5-nano, one shared prompt | +0.8pp | 2 | 0.549 |
| GPT-5-nano vs Sonnet, each with its own | **+3.2pp** | 8 | **0.022** |

GPT-5-nano finishes first at **$0.223/1k against Sonnet's $1.144/1k — 5.1x
cheaper**. A comparison that holds the prompt fixed cannot see this, because the
thing it holds fixed is the thing that was limiting the cheaper model. Optimizing
is also the safer half of the trade: **0 of 10** optimized rows scored below its
own stock row (p = 0.001 against a coin flip), while the cost penalty above is
certain.

**What this does not establish.** One flip, on one of the two tasks — banking77
has only two optimized rows and they did not reorder. And the inverse-strength
pattern this section is named for leans hard on a single model: across all ten
pairs, stock score and gain correlate at −0.89, but drop Qwen3 0.6B and that
falls to −0.31 (−0.19 on ranks). Read the per-model table above, which is
paired and Holm-corrected; do not read a law into the trend line.

### And the optimized prompt costs 2–3.6x more per classification

GEPA replaces a ~101-character instruction with a 1,800–2,500-character rubric.
Measured, per **classification** (not per billed call — see below):

| model | $/1k before → after | billed calls per 250 items |
|---|---|---|
| DeepSeek-V4-Flash | $0.041 → $0.123 (3.0x) | 250 → 250 |
| GPT-4.1-mini | $0.119 → $0.272 (2.3x) | 250 → 250 |
| Qwen3 1.7B | $0.014 → $0.030 (2.2x) | 250 → 250 |
| **GPT-5.4-nano** | **$0.069 → $0.250 (3.6x)** | **250 → 431** |
| Qwen3 0.6B | $0.003 → $0.011 (3.3x) | 250 → 347 |

GPT-4.1-mini bought a **statistically unproven** +1.6pp for a **certain** +128%
bill. "GEPA improved accuracy" is only half a sentence.

### The optimizer found a prompt that doubled the bill, and scored it as a win

On banking77, GPT-5.4-nano's optimized prompt was billed **500 calls to
classify 251 items** — essentially two calls per classification.

The mechanism: DSPy's `ChatAdapter` retries an item through `JSONAdapter` when
its own output format fails to parse. GEPA's reflection model had appended a
format directive to the instruction —

> `Return a JSON object with exactly: {"label": "<intent_label>"}`

— which is wrong for the adapter actually in use. The model complied (118 of 119
replies came back as `{"label": "..."}`), the parse failed, DSPy silently
retried, and both calls were billed. Accuracy was unaffected, because the retry
succeeds.

So the optimizer's metric saw a win and the bill saw a doubling, and **nothing
in the loop connected the two.** GEPA optimizes what you measure; if you measure
only accuracy, prompt cost is free by construction and it will spend it.

Two things had to be wrong at once for this to stay invisible:

1. Output format is the *adapter's* contract, not the instruction's — but
   nothing stopped a reflective optimizer from rewriting it.
2. Cost was being reported **per billed call** while the chart axis said
   **per classification**. At one call per item those are the same number, which
   is why it went unnoticed for every other row. This one was off by 1.99x.

If you optimize prompts and report cost, you have to divide by *items*, not by
API calls, or your own optimizer can hide its worst behaviour from you.

**The fix is one paragraph of prompt, and it is worth half the bill.** Pinning
the response envelope to the adapter's own format — and re-pinning it after
every optimization round, so the reflection model cannot have the last word —
was measured on 60 validation items:

| | billed calls | calls/item | accuracy | $/1k items |
|---|---|---|---|---|
| stored optimized prompt | 119 | 1.98 | 73.3% | $1.020 |
| same prompt, envelope pinned | 60 | **1.00** | 73.3% | **$0.484** |

Identical accuracy, identical unparseable count, **half the cost.** A follow-up
arm left the rogue JSON line in place and merely appended the pin *after* it:
still 1.00 calls per item. It is the last word that the model obeys.

**The second retry row has a different cause, and finding it took 1,290 items.**
GPT-5.4-nano on PhraseBank billed 431 calls for 250 items. Its instruction
contains no JSON directive at all — but it does say *"Do not output anything
else (no explanations, no extra keys, no punctuation outside the label)"*, and
the model obeyed **too literally**, emitting a bare `neutral` with no
`[[ ## label ## ]]` header. Unparseable for the same reason, from the opposite
instruction. Captured first-call shapes over 250 items:

```
chat-markers                205
NO MARKERS: 'neutral'        32
NO MARKERS: 'positive'        7
NO MARKERS: 'negative'        4
NO MARKERS: "label = 'neutral'"   1
```

Both fixes — pinning the envelope, or just deleting the `Output format:` block —
took it to **0 retries out of 250**, with accuracy differences inside the CIs.

**But the rate is not stable, and that is the finding.** Re-measured today it
retries ~15% of items, not the ~72% the stored row implies. Pooled over 1,290
items: 0.152 [0.133, 0.173], and wildly over-dispersed — χ² = 84.7 on 10 df
against a binomial expectation of ~10. Individual 40-item blocks ranged from
**2.5% to 42.5%**; the same 120 items gave 1 retry in one run and 18 in another.
Not concurrency (25% at 1 thread, 22.5% at 24), not input length (r = +0.01),
only mildly label-dependent.

The reason is upstream of the prompt: **the OpenAI reasoning tiers reject a
`temperature` parameter**, so we send none — and the default is 1.0. Every other
model in this benchmark runs at temperature 0. Format compliance for a
borderline prompt is therefore a *sampled* quantity for exactly those models,
and the stored 72% was a real measurement of a genuinely unstable thing. The
token totals confirm it was not an artifact: `mean_out_tokens=11` only
reconciles if ~181 of 250 first calls were 2-token bare labels.

A 15-item probe cannot see any of this. At p=0.15 it returns zero retries 8.7%
of the time; inside a low block, 68% of the time. The first pass at this question
used n=15 and concluded "does not reproduce" — wrong, and wrong in the
comfortable direction.

### Every single classification, billed twice

Cohere Command A Plus is the cleanest example this benchmark has produced of why
cost has to be measured per *classification* and not per *call*.

| task | calls / item | $/1k calls | $/1k items | accuracy |
|---|---|---|---|---|
| banking77 | 1.90 | $2.395 | **$4.562** | 74.9% |
| Financial PhraseBank | **2.00** | $1.040 | **$2.081** | 95.6% |

On PhraseBank the retry rate is 2.00 calls per item -- not "most items", *every*
item. The model never once produced output the ChatAdapter could parse, so every
classification went through the JSONAdapter fallback and was billed twice.
`parse_failures` is **0** on both rows, because the fallback worked: the answer
came back, correct, at double the price, silently.

Read from its rate card, Command A Plus is a mid-tier model at $0.80/$3.20 per
1M -- comfortably under Haiku. Measured, it is the **second most expensive row on
the banking77 chart at $4.56/1k**, above Claude Sonnet 4.6 and behind only Opus,
while scoring 8 points *below* Sonnet. It has the worst accuracy-per-dollar of
any of the 20 priced rows on that task, and nothing on the rate card predicts it.

Note also what it does to the other honesty problem: because the answer survives
the retry, a metric that reads only the final answer sees a competent model. GEPA
would have optimized against it happily. The doubling is visible only in the call
counter.

### The retry was doing real work, for the weakest model

Pinning the envelope is not free everywhere. On Qwen3 0.6B — the 1.39x retry row
— it removed all 104 retries and took accuracy from **53.6% to 30.0%**. The
reason is uncomfortable: the JSONAdapter retry is a *materially better
classifier* than the first ChatAdapter call for a 0.6B model. Accuracy on
retried items was **76.9%**, versus 37.0% on the items that parsed first time.
The retry's structured-output schema constrains the answer to the label set; the
first call does not.

What defuses it: the majority-class baseline on this task is 61.2%, and **every
arm of that experiment is below it**. The pin destroyed a number that had no
business being read as a win. But on a stronger model the same effect would be
invisible and benign, and on a weaker one it silently flatters the model through
a mechanism that has nothing to do with the model.

Which is also why the fallback stays on. Measured on banking77 at n=250, with
`use_json_adapter_fallback` on versus off:

| model | ChatAdapter fails on | extra calls | rescued | accuracy cost of turning it off |
|---|---|---|---|---|
| DeepSeek-V4-Flash | 3.6% of items | +2.0% | 10 of 13 correct | **−4.0pp** |
| GPT-5.4-nano | 0.8% | +0.8% | 2 of 2 correct | −0.8pp |
| Qwen3 1.7B | 0.4% | +0.4% | 0 of 1 correct | 0.0pp |

Four accuracy points for 2% of the bill is a trade worth taking. Two model-side
failure modes drive it, and **no prompt can fix either**: DeepSeek typos its own
scaffolding (`[[ ## label ##]]`, missing the space the parser requires) on 2–5%
of calls, and models emit near-miss labels (`get_virtual_card` for
`getting_virtual_card`) that the retry's enum schema repairs exactly.

So the 1.004–1.03x rows on disk are an irreducible model-side floor, not prompt
bugs. Only the 1.4–2.0x rows were fixable.

### Same prompt, different score

On banking77, GEPA left DeepSeek-V4-Flash's instruction **byte-identical** —
111 characters in, 111 characters out, no diff. The two evaluation runs of that
one unchanged prompt scored:

| | accuracy | $/1k items | mean cached tokens |
|---|---|---|---|
| run 1 | 78.1% | $0.17770 | 1,378 |
| run 2 | 78.9% | $0.16791 | 1,452 |

**+0.80pp and −5.5% cost, from the same prompt at `temperature=0`.** Predictions
differ on 2 of 251 items. The cost gap is nothing but cache warmth — the second
run hit a hotter prefix cache.

(These numbers were 76.5% → 78.1%, ±1.6pp and −12%, until two later corrections
landed: the case-insensitive scorer fix re-measured both rows, and the DeepSeek
cached-input rate was corrected from $0.007 to $0.030 per 1M. The old figures
survived in this table for a while after the data underneath them changed, which
is its own lesson about hand-copied numbers — the replicate is now read straight
off the rows on disk.)

This is the free replicate every optimization run hands you, and it calibrates
everything else in this document: on this task, **±0.8pp is what "no change"
looks like** on this pair. Any single-run improvement smaller than that is not a
result. Note this is ONE replicate, not a dispersion estimate — the harness
still runs each configuration once, so there is no standard deviation anywhere
in this document. The
chart was drawing this pair as a GEPA win until the label was gated on "did the
instruction actually change" rather than "did the numbers move".

---

## 6. The wrong test — and the wrong metric — flip conclusions

### The wrong test: independent CIs on paired runs

Four independent reviewers looked at the GEPA gains and concluded they were
inside the noise, because the stock and optimized 95% CIs overlapped heavily.

That is the underpowered comparison. These are **paired** observations on
identical items, so only the *discordant* items carry information:

- **GPT-5-nano, +4.0pp:** 10 discordant items, **all 10** favouring GEPA →
  p=0.0010, survives correction. Real.
- **GPT-5.4-nano, +1.6pp:** 26 discordant items, 15 one way and 11 the other →
  p=0.4421. Churn, not improvement.

Identical-looking "+1.6pp" and "+4.0pp" headlines; completely different
evidence underneath. With n=250 the unpaired CI half-width is ±4.4%, so
*anything* under ~9 points looks insignificant if you compare intervals — which
would have thrown away a real effect.

The same-prompt replicate in §5 puts a floor under this independently: a prompt
that did not change at all moved between runs (1.6pp when first measured; 0.8pp on the current rows). Two different tasks,
two different reasons, the same "+1.6pp" — and in neither case was it real.

### The wrong metric: accuracy hides which class a model gave up on

Qwen3 1.7B, reasoning off vs reasoning on, same 250 items (§2):

| | accuracy | balanced accuracy | negative | neutral | positive |
|---|---|---|---|---|---|
| reasoning off | **80.4%** | 65.1% | 35.3% | 98.0% | 61.9% |
| reasoning on | 78.0% | **86.9%** | 97.1% | 65.4% | 98.4% |

Plain accuracy says reasoning-off wins by 2.4 points. Balanced accuracy says
reasoning-on wins by **21.8 points**. Same two runs, opposite conclusions.

The prediction distributions explain it (gold is 153 neutral / 63 positive /
34 negative):

- **reasoning off** predicted `neutral` 192 times out of 250. It largely
  collapsed onto the majority class and scored well because the majority class
  is 61.2% of the data.
- **reasoning on** predicted `positive` 110 times. It read directional signal
  into neutral sentences, but it actually found the minority classes.

Two different failure modes, near-identical accuracy. On a class-balanced
deployment — or on any task where the minority class is the one you care about,
which is usually why you built the classifier — these two models are not
remotely equivalent.

For contrast, GEPA on the *reasoning-off* configuration fixed both at once:
90.4% accuracy **and** 86.7% balanced accuracy, with no reasoning tokens. The
class bias was a prompt problem, not a capability problem.

---

## 7. Two widely-repeated case-study numbers do not survive checking

Verified against primary sources (talk captions pulled and grepped, commits
checked via the GitHub API), independently double-checked by a second reviewer.

- **Shopify "~75x cheaper, ~2x more reliable" — REAL.** Verbatim on
  dspy.ai and spoken at 20:21 of the cited talk. But the speaker attributes the
  75x to the GPT-5 → Qwen swap **plus self-hosting**, not to GEPA; GEPA gets
  credit for the quality gain. He also explicitly disclaims precision:
  *"I cannot give you exact numbers, but I can give you relative improvements."*
  He hedges on the model and names a "Qwen-3-9B", which does not exist in the
  Qwen3 dense line. **Do not name a specific Qwen model.**

- **Shopify "~550x yearly cost reduction" — UNSUPPORTED.** Appears on dspy.ai's
  homepage and use-cases page. The strings "550", "yearly" and "annual" appear
  **zero times** in the cited talk. (A naive grep returns ~60 hits for "550" —
  every one is a WebVTT cue timestamp.) **Do not cite this.**

- **Dropbox "doubled accuracy" — WRONG.** The engineering blog never uses the
  words "accuracy" or "double". It reports a **45% NMSE reduction (8.83 → 4.86)**
  migrating from o3 to gpt-oss-120b.

- **GEPA paper "10% over GRPO" — superseded.** v1 said 10% across four tasks;
  v2 (ICLR 2026 Oral) says **6% across six tasks**. Citing 10% cites a
  withdrawn version.

The safest citation with disclosed methodology is the **Databricks GEPA post**,
not the Shopify anecdote — and even there the "20x/90x cheaper" is against
*baseline* (unoptimized) Claude, and those Claude versions have since retired.

---

## 8. Smaller engineering traps that cost real time

Each of these produced a wrong number or a silent failure, not an error message.

- **`datasets` 5.x removed loading-script support.** Bit us twice: the canonical
  `financial_phrasebank` and `PolyAI/banking77` both raise *"Dataset scripts are
  no longer supported"*. Parquet mirrors required.

- **DSPy's `context` and `track_usage` are thread-local.** A context opened on
  the main thread is invisible inside `ThreadPoolExecutor` workers, so every
  call raises "No LM is loaded" — which, caught per-item, looks *exactly* like a
  model scoring 0%. `track_usage` likewise silently reports zero tokens.

- **`from __future__ import annotations` breaks `Literal[labels]` signatures.**
  The annotation reaches pydantic as an unresolvable ForwardRef. It survives the
  ChatAdapter path and explodes only when DSPy falls back to the JSONAdapter —
  i.e. intermittently, mid-run.

- **Reasoning models return empty content when `max_tokens` is too low.** The
  think block consumes the whole budget. Looks like a model too dumb to answer;
  is a config bug.

- **Claude on Azure AI Foundry needs `/anthropic/v1/messages`.** The
  OpenAI-compatible route returns `404 api_not_supported` for Anthropic models
  while working fine for OpenAI-family models on the same resource.

- **litellm 1.100.0 cannot reach ollama's `think` flag.** Tested `ollama_chat`
  with `think=`, `extra_body`, `chat_template_kwargs`, and the OpenAI-compatible
  endpoint — all four still returned full reasoning traces. Fireworks *does*
  honour `extra_body.chat_template_kwargs` (verified: 328 tokens → 2), so this
  is host-specific, not a universal truth.

- **A "free" baseline that wasn't zero.** A constant-prediction row computed its
  cost from a large-but-finite throughput proxy and came out at `1.86e-11` —
  technically positive, so it became the smallest value on a log axis and made
  every tick label unreadable.

- **One capitalised label in a 77-label set cost every LLM 1.6 accuracy points.**
  Our scorer lowercases the answer before matching it against the label set, so
  a perfectly correct `Refund_not_showing_up` matched nothing and was booked as
  a *parse failure*. It hit only the LLM rows — the classical baselines predict
  by index and never touch the matcher — i.e. it biased exactly the comparison
  we were making. Case-normalise both sides, not one.

- **`parse_failures` was counting two different things.** A model that answers
  correctly in the wrong case and a model that invents a label out of thin air
  both landed in that counter, which defeats the point of having it: the field
  exists to separate "returned garbage" from "was inaccurate". Once the casing
  bug is fixed the count drops to the genuine invented-label cases (1–5 per row).

- **The OpenAI reasoning tiers reject `temperature`, so they run at 1.0.** Every
  other model here is pinned at 0. That makes borderline behaviour — like
  whether a prompt's output format parses — a sampled quantity for those models
  only, over-dispersed enough that the same 120 items gave 1 retry in one run
  and 18 in another. Anything you measure once on a `gpt-5*` tier, you have not
  measured.

- **DSPy retries a failed parse through a second adapter, and bills you twice.**
  `ChatAdapter(use_json_adapter_fallback=True)` is the default. The retry
  succeeds, so accuracy looks fine and nothing is logged as an error — the only
  trace is a call count higher than the item count. Measured up to 1.99 calls
  per item.

- **~2–4% of LLM outputs on a 77-class task are unparseable** (5–9 of 251 per
  model). Models invent label names that are not in the label set. On a 3-class
  task this never happened. Scored as wrong, but it is a *different* failure
  from being inaccurate and needs a different fix.

- **An endpoint can accept the system message, return 200, and discard it.**
  Azure's `Phi-4-mini-instruct` deployment does. Measured: the same question
  with and without a 3,200-token system message returned *byte-identical* text
  and *identical* `prompt_tokens` (10 both times), so the content was dropped in
  transport, not merely ignored by the model — the token counter is the proof.
  The `developer` role behaves the same way; the identical instruction inlined
  into the `user` turn is obeyed perfectly. This is not cosmetic here, because
  DSPy's ChatAdapter puts the instructions, every field definition, the
  `Literal[...]` label enumeration *and* the format contract in the system
  message. Unfixed, the model receives a bare unlabelled sentence, and the row
  reads as a hopeless model instead of a broken pipe. litellm cannot express it:
  `supports_system_messages: False` plus `modify_params = True` was verified not
  to fold on a generic `openai/` route, because that logic only runs for
  providers with a bespoke prompt template. Fixed with a `dspy.LM` subclass that
  folds system into the first user turn (`system_role_lm.py`).

- **A missing cached-input meter costs more than a higher headline rate.** No
  Phi model publishes one, so every input token bills at full price. On the
  3-class task that is invisible; on the 77-class task the prompt is ~1,900
  byte-identical tokens per call and the models that *do* cache get ~62% of it
  back at a tenth of the rate. Measured cost multiplier going from the 3-class
  to the 77-class task: **2.8–3.6x for models that cache, 7.0x for Phi-4-mini
  and 7.3x for Haiku 4.5** — and Haiku gets there by the other route, a
  4,096-token cache *minimum* that a 2,640-token prompt never reaches. Two
  different mechanisms, same 2x penalty. On paper Phi's $0.075/1M undercuts
  GPT-5-nano's $0.05; in practice it lands above DeepSeek-V4-Flash, which is
  both cheaper and 20 points more accurate.

- **Anthropic changed tokenizers.** Claude 4.7+ produces ~30% more tokens for
  identical text. On the same prompt: GPT-5.4-nano 1,923 tokens, Claude Sonnet
  4.6 (older tokenizer) 2,641, Claude Opus 4.8 (newer) 3,547. Per-token prices
  are not comparable across tokenizer generations.

---

## The thread running through all of it

Every finding above is the same shape: **the published number and the measured
number disagree, and the gap is only visible if you instrument the run.**

Rate cards do not price reasoning tokens. They do not mention cache minimums.
They assume caching is on when it may be opt-in. Leaderboards report one
configuration of a model that has three. Case studies get repeated with a digit
changed. The statistical test most people reach for is the one that hides real
effects, and the metric most people report hides which class the model gave up
on. An optimizer pointed at accuracy alone will happily double your bill and
book it as a win, because nothing it can see is denominated in money.

That gap is the product.
