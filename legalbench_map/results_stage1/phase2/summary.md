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
| opp115_data_retention | glm-5.3-flash | 304 | no | 78.62% [73.67%, 82.86%] | 78.62% | 79.71% | 79.71% | 83.00% | 83.00% | 53.18% | 0.2042 | 0.1086 | 161.7 | 73.97% | 67.42% | 91.78% | 20.29% | 70.50% (GPT-3.5) | **unclear** (unclear, unclear, unclear) |
| opp115_data_retention | gpt-oss-120b-fireworks | 304 | no | 73.03% [67.77%, 77.71%] | 73.03% | 79.71% | 79.71% | 80.37% | 80.37% | 53.18% | 0.1578 | 0.0839 | 161.7 | 69.06% | 67.42% | 91.34% | 20.29% | 70.50% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | kimi-k2.6-fireworks | 304 | no | 75.66% [70.53%, 80.14%] | 75.66% | 79.71% | 79.71% | 80.92% | 80.92% | 53.18% | 2.7260 | 1.4497 | 161.7 | 70.02% | 67.42% | 91.78% | 20.29% | 70.50% (GPT-3.5) | **unclear** (unclear, cheap-alone, cheap-alone) |

## Lenient scoring (used for the cascade decision)

| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc | cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc | sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opp115_data_retention | glm-5.3-flash | 304 | no | 78.62% [73.67%, 82.86%] | 78.62% | 79.71% | 79.71% | 83.00% | 83.00% | 53.18% | 0.2042 | 0.1086 | 161.7 | 73.97% | 67.42% | 91.78% | 20.29% | 70.50% (GPT-3.5) | **unclear** (unclear, unclear, unclear) |
| opp115_data_retention | gpt-oss-120b-fireworks | 304 | no | 73.03% [67.77%, 77.71%] | 73.03% | 79.71% | 79.71% | 80.37% | 80.37% | 53.18% | 0.1578 | 0.0839 | 161.7 | 69.06% | 67.42% | 91.34% | 20.29% | 70.50% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | kimi-k2.6-fireworks | 304 | no | 75.66% [70.53%, 80.14%] | 75.66% | 79.71% | 79.71% | 80.92% | 80.92% | 53.18% | 2.7260 | 1.4497 | 161.7 | 70.02% | 67.42% | 91.78% | 20.29% | 70.50% (GPT-3.5) | **unclear** (unclear, cheap-alone, cheap-alone) |

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
