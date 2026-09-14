# PILOT (n=10) -- not evidence

# Phase 2 summary: paired cascade analysis

**PILOT -- not evidence**: 18 of 18 (task, model) pairs were run with fewer items than n_test (see the n column).

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
| learned_hands_consumer | glm-5.3-flash | 10 | PILOT | 90.00% [59.58%, 98.21%] | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% | 30.00% | 0.4175 | 0.1253 | 3.0 | 63.89% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| learned_hands_consumer | gpt-oss-120b-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 90.00% | 90.00% | 100.00% | 100.00% | 30.00% | 0.3291 | 0.0987 | 3.0 | 100.00% | 63.89% | 100.00% | 10.00% | 76.20% (GPT-4) | **unclear** (unclear, unclear, unclear) |
| learned_hands_consumer | kimi-k2.6-fireworks | 10 | PILOT | 90.00% [59.58%, 98.21%] | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% | 30.00% | 5.1717 | 1.5515 | 3.0 | 63.89% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | glm-5.3-flash | 10 | PILOT | 50.00% [23.66%, 76.34%] | 54.17% | 76.67% | 76.39% | 53.33% | 56.94% | 73.33% | 0.1941 | 0.1423 | 7.3 | 50.00% | 81.55% | 83.33% | 23.33% | 70.50% (GPT-3.5) | **unclear** (unclear, cheap-alone, cheap-alone) |
| opp115_data_retention | gpt-oss-120b-fireworks | 10 | PILOT | 70.00% [39.68%, 89.22%] | 75.00% | 76.67% | 76.39% | 73.33% | 77.78% | 73.33% | 0.2175 | 0.1595 | 7.3 | 77.38% | 81.55% | 86.67% | 23.33% | 70.50% (GPT-3.5) | **unclear** (unclear, unclear, unclear) |
| opp115_data_retention | kimi-k2.6-fireworks | 10 | PILOT | 60.00% [31.27%, 83.18%] | 66.67% | 76.67% | 76.39% | 63.33% | 69.44% | 73.33% | 5.1397 | 3.7691 | 7.3 | 63.69% | 81.55% | 86.67% | 23.33% | 70.50% (GPT-3.5) | **unclear** (unclear, unclear, unclear) |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 0.1641 | 0.0821 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 0.2134 | 0.1067 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | kimi-k2.6-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 1.5815 | 0.7908 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |

## Lenient scoring (used for the cascade decision)

| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc | cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc | sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | glm-5.3-flash | 10 | PILOT | 90.00% [59.58%, 98.21%] | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% | 30.00% | 0.4175 | 0.1253 | 3.0 | 63.89% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| learned_hands_consumer | gpt-oss-120b-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 90.00% | 90.00% | 100.00% | 100.00% | 30.00% | 0.3291 | 0.0987 | 3.0 | 100.00% | 63.89% | 100.00% | 10.00% | 76.20% (GPT-4) | **unclear** (unclear, unclear, unclear) |
| learned_hands_consumer | kimi-k2.6-fireworks | 10 | PILOT | 90.00% [59.58%, 98.21%] | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% | 30.00% | 5.1717 | 1.5515 | 3.0 | 63.89% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | glm-5.3-flash | 10 | PILOT | 50.00% [23.66%, 76.34%] | 54.17% | 76.67% | 76.39% | 53.33% | 56.94% | 73.33% | 0.1941 | 0.1423 | 7.3 | 50.00% | 81.55% | 83.33% | 23.33% | 70.50% (GPT-3.5) | **unclear** (unclear, cheap-alone, cheap-alone) |
| opp115_data_retention | gpt-oss-120b-fireworks | 10 | PILOT | 70.00% [39.68%, 89.22%] | 75.00% | 76.67% | 76.39% | 73.33% | 77.78% | 73.33% | 0.2175 | 0.1595 | 7.3 | 77.38% | 81.55% | 86.67% | 23.33% | 70.50% (GPT-3.5) | **unclear** (unclear, unclear, unclear) |
| opp115_data_retention | kimi-k2.6-fireworks | 10 | PILOT | 60.00% [31.27%, 83.18%] | 66.67% | 76.67% | 76.39% | 63.33% | 69.44% | 73.33% | 5.1397 | 3.7691 | 7.3 | 63.69% | 81.55% | 86.67% | 23.33% | 70.50% (GPT-3.5) | **unclear** (unclear, unclear, unclear) |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 0.1641 | 0.0821 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 0.2134 | 0.1067 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | kimi-k2.6-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 1.5815 | 0.7908 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |

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
