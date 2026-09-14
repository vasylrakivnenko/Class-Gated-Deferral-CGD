# The free classifier beat GPT-4 on 24 of 28 legal tasks. The router I built to hedge didn't beat either.

*A $14 experiment in when you don't need an LLM — and why the hybrid everyone builds next doesn't help.*

---

Every team that ships an LLM classifier eventually asks the same question: *do we actually need this?* The prompt works, the bill grows, and somewhere in the data warehouse sits a few hundred labeled examples that nobody has tried training a real model on. Then a second question follows: *if the cheap model is almost as good, can we route the hard cases to the LLM and keep the easy ones cheap?*

I spent a week and about $14 answering both, on 37 legal classification tasks, with four 2026-era LLMs, and with the statistics done properly. The first answer is "more often than you think." The second is "no — and here is the experiment that shows why."

## Part 1: The map

[LegalBench](https://arxiv.org/abs/2308.11462) is 162 small legal tests written by lawyers — *does this contract clause grant audit rights, is this privacy-policy sentence about data retention, is this piece of evidence hearsay.* Each has human-verified answers, and the paper's authors published scores for 20 models including GPT-4 and Claude. That makes it perfect for a cheap experiment: train a free classifier on each task's own labels, compare against the published LLM numbers, spend nothing.

I picked 28 tasks that had at least 300 labeled examples, plus 9 "reasoning" tasks that were expected to be hard, and ran four free models on each: TF-IDF with logistic regression, two sizes of frozen sentence embeddings with logistic regression, and a majority-class baseline. Evaluation was 5-fold cross-validation repeated three times, with a conformal confidence gate, bootstrap confidence intervals, and a verdict rule: *match* if the free model's lower confidence bound cleared the best published LLM score within one point.

The result, after fixing one mistake I'll get to:

| | tasks | match | close | short |
|---|---|---|---|---|
| pattern-recognition tasks (clauses, policies, forum posts, disclosures) | 28 | **18** | 6 | 4 |
| genuine legal-reasoning tasks (hearsay, jurisdiction, trademark strength) | 9 | 0 | 0 | **9** |

On 24 of 28 pattern tasks, a classifier trained on a few hundred labels sits within three points of GPT-4 or Claude. On 18 it *beats* them outright. TF-IDF — the dumbest model in the set — won 17 of 37 tasks. And on all 9 reasoning tasks, the free models lose by 7 to 47 points.

That boundary is the finding. "Reasoning task" and "hard for a cheap model" turned out to mean the same thing, and the tool measures which side of the line your task is on in an afternoon, for free.

### The mistake

Six of those "reasoning" tasks initially came back at 99–100% for the free model, which should have been impossible. It was: the dataset shipped the intermediate reasoning steps as extra columns (`parties_are_diverse`, `aic_is_met`), I'd kept them thinking they were legitimate inputs, and the classifier learned `answer = A and B`. The original prompt given to GPT-4 was just the raw fact pattern. I'd built a leak.

The fix is worth more than the correction: a classifier may only see the columns that appear as placeholders in the original prompt template — mechanically checkable, now enforced by a test that fails the build if any task's inputs drift. My collaborator caught it by asking the only question that matters: *would those columns exist in real life?* They wouldn't. Ask that about every feature.

## Part 2: Modern models, same items

Published 2023 scores are a weak opponent. So I took three tasks from the middle of the map — a privacy-policy sentence task (304 items), a supply-chain-disclosure task (379), and a legal-aid forum task (614) — and ran four current models on every item: GLM-5.3-flash, gpt-oss-120b, Kimi K2.6, and DeepSeek V4 Pro, all on Fireworks, all with a few-shot prompt, scored item-by-item against the free model on identical test items with paired McNemar tests. Total cost: $4.20.

| task | free model | best modern LLM | worst modern LLM |
|---|---|---|---|
| privacy policy | 79.7% | GLM 78.6% (tie) | gpt-oss 73.0% |
| supply chain | 80.9% | gpt-oss 82.1% (tie) | GLM 76.0% |
| legal-aid forum | 84.3% | **GLM 87.5%** (+3, significant) | gpt-oss 71.5% |

Three things fell out of this that I didn't expect.

**Which LLM matters more than whether to use one.** The free model is never more than three points behind the best modern LLM. But the spread *between* LLMs on the same task is 9 to 16 points. GLM is the best model on two tasks and the worst on one; gpt-oss is best on one and worst on two; DeepSeek V4 Pro — the most expensive — is never best. There is no "better model." There is "better on this task," and only a measurement tells you which.

**2023 benchmarks and 2026 models don't speak the same language.** The LegalBench prompt is completion-style few-shot ending in `Label:`. GPT-4 in 2023 replied `Yes`. GLM in 2026 replies `**Label: Yes**` followed by a paragraph of reasoning — right answer, scored wrong by the benchmark's exact-match rule. Parse rate: 1 in 30. One added line — *"Answer with exactly one word: Yes or No"* — took it to 30 of 30. If you compare a modern model to a published number without checking what it actually emitted, you are measuring instruction-following, not the task.

**Sticker price is not cost.** A one-word answer from Kimi K2.6 arrived with 637 hidden reasoning tokens billed at output rates: $2.73 per thousand classifications versus $0.16 for gpt-oss, for accuracy 2.6 points *lower*. GLM's cost tripled on the long forum posts because it reasoned harder. Cost per classification is a property of the model *and* the task, and the only way to know it is to run it.

## Part 3: The hybrid

So the free model is close, sometimes ahead, occasionally behind. The obvious next move — the one every routing startup sells — is a *cascade*: let the cheap model answer when it's confident, escalate to the LLM when it isn't. You keep most of the savings and, in theory, get the LLM's accuracy on the hard cases.

I built it. Conformal prediction on the cheap model's confidence decides what to escalate; the LLM answers the rest. Then I measured it properly, per task, against both pure strategies, with paired tests.

| | privacy policy | supply chain | legal-aid forum |
|---|---|---|---|
| gate escalates | 53% of items | 58% | 37% |
| LLM's accuracy on the escalated items vs the free model's | +2 to +7 pts (not significant) | **−0 to −8 pts** | +9 (GLM only) |
| cascade vs just using the free model | +0.7 to +3.3 (not significant) | **−0.1 to −4.7 (significantly worse vs GLM)** | +3.5 with GLM, −2.7 with gpt-oss |

On the supply-chain task, escalating to GLM made things *worse* — the items the free model was unsure about were items GLM got wrong even more often. On the privacy task, the LLM was no better than the free model on what it received. Only one of nine (task, model) pairs showed the cascade beating the free model, and there the LLM was simply the better model overall.

The reason is in the middle row: **"the cheap model is uncertain" is not the same as "the LLM knows better."** The gate routes on the cheap model's confidence. What it needs to route on is *relative* competence — where does this LLM beat this classifier — and that's a different signal. Maybe a learnable one. So I tried to learn it.

## Part 4: The $0 experiment

Everything needed was already on disk: the free model's per-item predictions and confidences, and every LLM's per-item answers. So I could replay any gate offline and score it honestly — five-fold out-of-fold over items, meaning every routing decision is made by a rule that never saw that item.

Three gates: the deployed conformal one; a plain confidence threshold with the cutoff chosen on training folds; and a learned router — two small models predicting *P(LLM is right)* and *P(cheap model is right)* from the cheap model's confidence plus a text embedding, trained on the LLMs' own per-item results, escalating where the LLM's predicted edge exceeds a margin. Plus the oracle: escalate exactly the cheap model's errors, the ceiling any gate could reach.

Across all nine pairs, the oracle sits 5 to 13 points above every real gate. **No honest gate recovers any of that gap.** The threshold gate's chosen cutoff was "escalate nothing" in seven of nine pairs. The learned router, with the LLMs' own labels to learn from, landed within −0.3 to +1.4 points of whichever base model was better — never above both. The frontier chart says it in one glance: three repeats disagree by several points at the same escalation rate, that spread *is* the noise floor, and no out-of-fold gate ever climbs above it.

Why? The oracle knows which items the cheap model gets wrong. If that were predictable from the text or the model's confidence, the cheap model would have used that signal to be right. Same for the LLM's errors. From the gate's point of view, both systems' mistakes are noise, not structure — so the only reliable routing rule is "mostly cheap" or "mostly LLM," chosen by which is better overall.

What survives is narrower and still useful: on the two pairs where the LLM genuinely wins, a threshold gate escalates 7–16% of items and lands within a point of LLM-alone at 84–93% less cost. **The cascade is a cost dial, not an accuracy device.** It never beats the better base model. Anyone selling a router that "beats both models" should be asked for the out-of-fold curve.

## What I'd tell you to do on Monday

1. **Count your labels.** Below about 500, the LLM wins — on a 77-class banking task, the LLM led a free model by 32 points at 100 labels and lost by 3 at 2,000. Nothing else matters until you know this number.
2. **Train the dumbest thing first.** TF-IDF and logistic regression, cross-validated, with a confidence interval. It won 17 of 37 tasks here and reads long documents the embedders truncate.
3. **Run your actual incumbent on the same items** — the same items, so you can do a paired test. A few hundred items costs cents. Compare per-item cost, not list price.
4. **Pick the better one.** If it's the free model, you're done and your LLM bill is zero. If it's the LLM, add a confidence threshold and escalate the least-confident tenth; you'll keep most of the accuracy at a fraction of the cost.
5. **Don't build the router.** Or if you do, score it out-of-fold against both pure strategies before believing it.

## What this doesn't show

Three tasks and four models is a map of a corner, not the world. The published 2023 scores are old, which is exactly why Part 2 exists. Each LLM ran once at temperature zero while the cheap model got three cross-validated passes — an asymmetry I chose and reported, not hid. Test sets of 300–600 items resolve about three points, so "tie" means "tie within three points." And the one cheap-model upgrade that had a real mechanism behind it — fine-tuning a long-context encoder so it could read entire 2,000-token disclosures instead of the first 512 — I couldn't run: Apple's MPS backend spent 40 minutes compiling Metal kernels per fold, a tooling limit, not a result.

Everything else is code: the harness, the tests (339 of them, including the one that would have caught my leak), the per-item data, the replay. Total spend across the whole thing was about $14, most of it on an earlier benchmark. The legal experiment cost $4.20 and the experiment that mattered most cost nothing.

---

*The instrumented harness, the per-item data for every (task, model) pair, and the gate replay are in the repo. If you run the replay on your own task and a learned gate beats both base models out-of-fold, I want to see it — it would be the first.*
