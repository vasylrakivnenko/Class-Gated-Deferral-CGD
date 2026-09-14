# Phase 2 summary: paired cascade analysis

- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.
- Labels: incumbent = measured (this run); cheap = CV-estimated (Phase 1 OOF); published-2023 = contrast only.
- Cheap/cascade/sent columns are MEANS across cheap repeats; per-repeat numbers are in each pair's .md/.json.
- $/1k: incumbent-alone = measured sum(cost_usd)/n*1000; cascade = LLM share x incumbent $/1k (cheap side costs $0).

- Two scorings are reported. EXACT (first table) is benchmark-faithful; LENIENT (second table) is what the cascade decision uses. Definitions:
  - exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.
  - lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

## Exact-match scoring (benchmark-faithful)

| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc | cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc | sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | deepseek-v4-pro-fireworks | 614 | no | 81.92% [78.68%, 84.76%] | 81.92% | 84.31% | 84.31% | 85.02% | 85.02% | 37.30% | 2.9964 | 1.1176 | 229.0 | 69.85% | 67.94% | 93.00% | 15.69% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| learned_hands_consumer | glm-5.3-flash | 614 | no | 85.34% [82.32%, 87.92%] | 85.34% | 84.31% | 84.31% | 86.10% | 86.10% | 37.30% | 0.5909 | 0.2204 | 229.0 | 72.78% | 67.94% | 92.40% | 15.69% | 76.20% (GPT-4) | **unclear** (unclear, unclear, unclear) |
| learned_hands_consumer | gpt-oss-120b-fireworks | 614 | no | 71.50% [67.80%, 74.93%] | 71.50% | 84.31% | 84.31% | 81.60% | 81.60% | 37.30% | 0.1935 | 0.0722 | 229.0 | 60.69% | 67.94% | 92.40% | 15.69% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |

## Lenient scoring (used for the cascade decision)

| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc | cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc | sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | deepseek-v4-pro-fireworks | 614 | no | 81.92% [78.68%, 84.76%] | 81.92% | 84.31% | 84.31% | 85.02% | 85.02% | 37.30% | 2.9964 | 1.1176 | 229.0 | 69.85% | 67.94% | 93.00% | 15.69% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| learned_hands_consumer | glm-5.3-flash | 614 | no | 87.46% [84.60%, 89.85%] | 87.46% | 84.31% | 84.31% | 87.79% | 87.79% | 37.30% | 0.5909 | 0.2204 | 229.0 | 77.29% | 67.94% | 93.11% | 15.69% | 76.20% (GPT-4) | **unclear** (unclear, earns, incumbent) |
| learned_hands_consumer | gpt-oss-120b-fireworks | 614 | no | 71.50% [67.80%, 74.93%] | 71.50% | 84.31% | 84.31% | 81.60% | 81.60% | 37.30% | 0.1935 | 0.0722 | 229.0 | 60.69% | 67.94% | 92.40% | 15.69% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |

Parse-fail rates per scoring are in each pair's .md/.json (exact vs lenient).

```
VERDICT RULES (applied verbatim by phase2/analyze.py)

Definitions. NI(x vs y) = downshift.stats.non_inferiority_test(correct_x, correct_y,
margin=delta) with delta = 0.01 (95% one-sided, paired item bootstrap); "passes" means
the lower bound of the one-sided CI on acc_x - acc_y lies above -delta. McNemar = mid-p
McNemar on paired per-item correctness. Holm = Holm-Bonferroni at alpha = 0.05 over the
family of 4 McNemar p-values computed for one (task, model, repeat): (3) cheap vs incumbent
on all items, (4) cheap vs incumbent on the SENT subset, (5a) cascade vs cheap-alone,
(5b) cascade vs incumbent-alone. "cheaper" = cascade $/1k < incumbent-alone $/1k, i.e.
LLM share < 1. All numbers are exact-match accuracy on the joined items.

Per-repeat verdict, first rule that fires wins:
  cheap-alone : NI(cheap-alone vs incumbent-alone) passes.  No LLM needed.
  earns       : NI(cascade vs incumbent-alone) passes AND cheaper AND cascade beats
                cheap-alone (cascade acc - cheap acc > 0 AND McNemar (5a) p < 0.05
                after Holm).
  incumbent   : neither NI(cascade vs incumbent-alone) nor NI(cheap-alone vs
                incumbent-alone) passes.
  unclear     : anything else.

(task, model) verdict: the per-repeat verdict if EVERY cheap repeat agrees; otherwise
'unclear'. Repeat-to-repeat disagreement is itself evidence that the call is not settled.
```
