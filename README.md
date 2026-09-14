# downshift

**Drop a labeled dataset. Get the cheapest model and prompt that clears your accuracy bar.**

A benchmark harness that ports a classification task from an expensive model down to
the cheapest thing that still works — and is honest about when that thing is not an
LLM at all.

It runs four families of candidate on the same held-out test set and prices them all
from *measured* token counts:

| Family | What it is |
|---|---|
| Prompted LLMs | frontier, proprietary small tier, and small open-weight models |
| GEPA-optimized LLMs | the same small models after DSPy's GEPA rewrites the prompt |
| Classical baselines | majority class, TF-IDF + logistic regression, frozen embeddings + logreg |
| Trained encoder | a small encoder (~68M params) fine-tuned on your labels |

---

## Quick start

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e ".[encoders]"

# local models (no API keys needed)
ollama pull qwen3:0.6b qwen3:1.7b qwen3:4b gpt-oss:20b

cp .env.example .env      # add API keys for hosted models; partial is fine
.venv/bin/python run_demo.py
```

Outputs `runs/cost_vs_accuracy.png` and `runs/results.json`. Any hosted model whose
key is missing is skipped with a note, so a partial `.env` yields a smaller chart
rather than a failed run.

---

## The methodology, and why each piece is there

This is the part that separates a benchmark from a marketing chart.

### Three-way split, and the test set is sacred

GEPA reads the **train** set to propose instructions and scores candidates on the
**val** set to pick survivors. Both are therefore contaminated: a number measured on
either is a training number in disguise. Only the **test** split — carved out first,
never shown to any optimizer — supports the claim the product sells.

The dataset is Financial PhraseBank `sentences_allagree`: 2,264 sentences where
*every* human annotator assigned the same label. That is a genuine human-verified
holdout, not synthetic labels, so we are not measuring agreement with a teacher
model's mistakes.

Two caveats that belong in any writeup of these numbers:

1. `allagree` is the **easy** subset by construction. Unanimous agreement selects for
   unambiguous sentences, so accuracies run higher here than the same model would
   score on messy production text. It is the right choice for a *trustworthy* holdout
   and the wrong one for estimating production accuracy.
2. The class balance is 13% negative / 61% neutral / 25% positive. **Always answering
   "neutral" scores 61.2%**, so that line is drawn on every chart. A three-class task
   with a 61% floor flatters weak models badly.

### Cost is measured, never blended

The usual "$/1M tokens, 50/50 input/output" figure is close to meaningless for
classification. A sentiment call here is ~260 input tokens and 5–40 output tokens —
about 87:1, nothing like 50:50 — so a blend flatters expensive models with cheap
input and can invert the ranking against the actual bill.

Worse, reasoning models bill their `<think>` block as output tokens. Measured on this
machine, the identical classification cost **260 output tokens with reasoning on and
37 with it off**. No headline rate can see that. So every row is priced as:

```
cost_per_1k = 1000 * (mean_input_tokens * price_in + mean_output_tokens * price_out) / 1e6
```

with token counts taken from the provider's own usage counters during the same run
that produced the accuracy number. Rate cards set the price; the run sets the quantity.

### The statistics (`src/downshift/stats.py`)

- **Wilson intervals**, not normal approximation (which returns intervals above 100%
  near the boundary) and not Clopper–Pearson (exact but conservative, which would
  make the chart look less decisive than the data warrants).
- **McNemar mid-p** for comparing two models on the same items. Pairing matters: with
  n=250 and two models agreeing on 220 items, the effective sample size is 30, not 250.
  Mid-p is the default because Fagerland, Lydersen & Laake (2013) conclude verbatim:
  *"We do not recommend use of the McNemar exact conditional test nor the asymptotic
  test with CC in any situation."* Both under-reject; for us that means telling a user
  their cheap model is fine when it is not.
- **Non-inferiority testing**, not significance testing. Our claim is "not worse by
  more than δ", which is a one-sided test against −δ. "No significant difference" is
  the trap: failing to reject is usually just evidence of a small test set, and a
  useless test set would pass that bar every time.
- **Holm–Bonferroni** across ~6–10 candidates. Uncorrected, roughly one run in three
  would throw a spurious winner.
- **A resolution warning printed on every run.** At n=250 the 95% half-width is ±4.4%,
  so two models closer than ~8.9% cannot be separated by unpaired comparison. Saying
  so is the product.

---

## Engineering notes (things that cost real time)

- **litellm cannot reach ollama's `think` flag.** Verified against litellm 1.100.0:
  `ollama_chat` raises on `think=`, and neither `extra_body` nor
  `chat_template_kwargs` survives the trip through the OpenAI-compatible endpoint —
  all three still returned full reasoning traces. Hence `ollama_lm.py`, a ~100-line
  DSPy `BaseLM` that speaks ollama's native `/api/chat`. Buys the `think` switch,
  exact token counters, and reasoning tokens reported separately.
- **`dspy.context` is thread-local.** A context opened on the main thread is invisible
  inside `ThreadPoolExecutor` workers, so every call raises "No LM is loaded" — which
  looks exactly like a model scoring 0%. Re-enter the context inside the worker.
- **`dspy.track_usage()` is thread-local too**, and silently reports zero for calls
  made on worker threads. Token accounting reads `lm.history` instead.
- **`from __future__ import annotations` breaks `Literal[labels]` signatures.** The
  annotation reaches pydantic as the unresolvable ForwardRef `'Literal[labels]'`. It
  survives the ChatAdapter path and explodes when DSPy falls back to the JSONAdapter —
  i.e. intermittently, mid-run. Build signatures programmatically.
- **GEPA validates its metric arity at construction**: exactly
  `(gold, pred, trace, pred_name, pred_trace)`.
- **transformers v5 removed `reference_compile`** and renamed `evaluation_strategy` →
  `eval_strategy`, `use_mps_device` → auto-detected. Most ModernBERT tutorials break.
- **Reasoning models return empty content when `max_tokens` is too low** — the think
  block eats the whole budget. Looks like a stupid model; is actually a config bug.
  `ollama_lm.py` raises a named error instead.
- **ollama throughput peaks around 12 concurrent requests** on an M4 Pro (2.8 items/s)
  and *degrades* past 16.

---

## Citing the case studies — read this before you post

The public numbers behind "port your task to a small model" are shakier than they
look. Verified against primary sources (talk captions pulled and grepped, commits
checked via the GitHub API), independently double-checked:

| Claim | Verdict |
|---|---|
| Shopify "~75x cheaper, ~2x more reliable" (dspy.ai) | **Real.** Verbatim on dspy.ai and in the talk at 20:21. |
| …but attributable to GEPA | **No.** The speaker attributes 75x to the GPT-5→Qwen swap *plus self-hosting*. GEPA gets credit for the quality gain. |
| …and it says "reliable" | Speaker says "quality"; the event blurb says "reliability". Either is defensible; "2x more reliable" is DSPy's paraphrase. |
| …on a specific Qwen model | **Unverifiable.** Speaker hedges and names a "Qwen-3-9B", which does not exist in the Qwen3 dense line. Do not name a model. |
| Shopify "~550x yearly cost reduction" (dspy.ai homepage) | **Unsupported.** "550", "yearly" and "annual" appear **zero times** in the cited talk. Do not cite. |
| Dropbox "doubled accuracy" | **Wrong.** The blog never uses "accuracy" or "double". It reports a 45% NMSE reduction (8.83 → 4.86), migrating from o3 to gpt-oss-120b. |
| GEPA paper "10% over GRPO" | **Superseded.** v1 said 10% across four tasks; v2 (ICLR 2026 Oral) says **6% across six tasks**. Citing 10% cites the withdrawn version. |
| Databricks "20x/90x cheaper" | **Real but qualified.** The same post also says 22x, and the comparison is against *baseline* (unoptimized) Claude. Also: Sonnet 4 / Opus 4.1 have since been retired. |

The safest primary citation with disclosed methodology is the **Databricks GEPA post**,
not the Shopify anecdote.

---

## Layout

```
src/downshift/
  data.py         dataset load, dedupe, stratified 3-way split, leakage asserts
  models.py       candidate registry + measured-token cost model
  ollama_lm.py    DSPy LM over ollama's native API (the `think` switch)
  program.py      DSPy signature + direct/reasoning cost profiles
  metric.py       scoring + the textual feedback GEPA reflects on
  optimize.py     GEPA runner, tracks reflection cost separately
  evaluate.py     threaded eval with token accounting
  encoders.py     majority / TF-IDF / frozen-embedding / fine-tuned encoder
  stats.py        Wilson, McNemar mid-p, paired bootstrap, non-inferiority, Holm
  chart.py        the cost-vs-accuracy chart
  experiment.py   end-to-end orchestration
run_demo.py       the 10-minute demo entry point
```
