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
| supply_chain_disclosure_best_practice_audits | deepseek-v4-pro-fireworks | 379 | no | 77.04% [72.55%, 80.99%] | 65.04% | 80.91% | 74.47% | 77.84% | 66.07% | 57.87% | 1.4951 | 0.8652 | 219.3 | 67.77% | 73.11% | 91.56% | 19.09% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 379 | no | 75.99% [71.44%, 80.02%] | 62.90% | 80.91% | 74.47% | 76.25% | 63.32% | 57.87% | 0.2066 | 0.1196 | 219.3 | 65.05% | 73.11% | 90.50% | 19.09% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 379 | no | 82.06% [77.88%, 85.59%] | 72.34% | 80.91% | 74.47% | 80.83% | 70.46% | 57.87% | 0.2222 | 0.1286 | 219.3 | 72.94% | 73.11% | 91.73% | 19.09% | 76.60% (GPT-3.5) | **incumbent** (incumbent, incumbent, incumbent) |

## Lenient scoring (used for the cascade decision)

| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc | cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc | sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supply_chain_disclosure_best_practice_audits | deepseek-v4-pro-fireworks | 379 | no | 77.04% [72.55%, 80.99%] | 65.04% | 80.91% | 74.47% | 77.84% | 66.07% | 57.87% | 1.4951 | 0.8652 | 219.3 | 67.77% | 73.11% | 91.56% | 19.09% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 379 | no | 75.99% [71.44%, 80.02%] | 62.90% | 80.91% | 74.47% | 76.25% | 63.32% | 57.87% | 0.2066 | 0.1196 | 219.3 | 65.05% | 73.11% | 90.50% | 19.09% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 379 | no | 82.06% [77.88%, 85.59%] | 72.34% | 80.91% | 74.47% | 80.83% | 70.46% | 57.87% | 0.2222 | 0.1286 | 219.3 | 72.94% | 73.11% | 91.73% | 19.09% | 76.60% (GPT-3.5) | **incumbent** (incumbent, incumbent, incumbent) |

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
