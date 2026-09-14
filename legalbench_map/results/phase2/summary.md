# INVALID: learned_hands_consumer__qwen3.7-plus (10 API-error rows), opp115_data_retention__qwen3.7-plus (10 API-error rows), supply_chain_disclosure_best_practice_audits__qwen3.7-plus (10 API-error rows)

# PILOT (n=10) -- not evidence

# Phase 2 summary: paired cascade analysis

**INVALID -- API errors, not model answers**: 3 of 9 (task, model) pairs contain runner '<error: ...>' rows; their incumbent/cascade numbers are meaningless and their verdict is 'invalid-api-errors'. Re-run phase2/run_incumbents.py (it retries error rows on resume).

**PILOT -- not evidence**: 9 of 9 (task, model) pairs were run with fewer items than n_test (see the n column).

- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.
- Labels: incumbent = measured (this run); cheap = CV-estimated (Phase 1 OOF); published-2023 = contrast only.
- Cheap/cascade/sent columns are MEANS across cheap repeats; per-repeat numbers are in each pair's .md/.json.
- $/1k: incumbent-alone = measured sum(cost_usd)/n*1000; cascade = LLM share x incumbent $/1k (cheap side costs $0).

| task | model | n | pilot | incumbent acc (measured) | incumbent bal_acc | cheap acc (CV-est) | cheap bal_acc | cascade acc | cascade bal_acc | LLM share | $/1k incumbent | $/1k cascade | sent n | sent: incumbent acc | sent: cheap acc | oracle acc | oracle share | published-2023 best | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | glm-5.3-flash | 10 | PILOT | 0.00% [0.00%, 27.75%] | 0.00% | 90.00% | 90.00% | 70.00% | 70.00% | 30.00% | 0.5086 | 0.1526 | 3.0 | 0.00% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| learned_hands_consumer | gpt-oss-120b-fireworks | 10 | PILOT | 80.00% [49.02%, 94.33%] | 80.00% | 90.00% | 90.00% | 83.33% | 83.33% | 30.00% | 0.4058 | 0.1217 | 3.0 | 38.89% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| learned_hands_consumer | qwen3.7-plus | 10 | PILOT | 0.00% [0.00%, 27.75%] | 0.00% | 90.00% | 90.00% | 70.00% | 70.00% | 30.00% | 0.0000 | 0.0000 | 3.0 | 0.00% | 63.89% | 90.00% | 10.00% | 76.20% (GPT-4) | **invalid-api-errors** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | glm-5.3-flash | 10 | PILOT | 0.00% [0.00%, 27.75%] | 0.00% | 76.67% | 76.39% | 16.67% | 13.89% | 73.33% | 0.3497 | 0.2565 | 7.3 | 0.00% | 81.55% | 76.67% | 23.33% | 70.50% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | gpt-oss-120b-fireworks | 10 | PILOT | 30.00% [10.78%, 60.32%] | 25.00% | 76.67% | 76.39% | 33.33% | 27.78% | 73.33% | 0.3841 | 0.2817 | 7.3 | 22.62% | 81.55% | 76.67% | 23.33% | 70.50% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| opp115_data_retention | qwen3.7-plus | 10 | PILOT | 0.00% [0.00%, 27.75%] | 0.00% | 76.67% | 76.39% | 16.67% | 13.89% | 73.33% | 0.0000 | 0.0000 | 7.3 | 0.00% | 81.55% | 76.67% | 23.33% | 70.50% (GPT-3.5) | **invalid-api-errors** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 10 | PILOT | 10.00% [1.79%, 40.42%] | 10.00% | 100.00% | 100.00% | 50.00% | 50.00% | 50.00% | 0.2379 | 0.1189 | 5.0 | 0.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 10 | PILOT | 100.00% [72.25%, 100.00%] | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 50.00% | 0.1831 | 0.0916 | 5.0 | 100.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **cheap-alone** (cheap-alone, cheap-alone, cheap-alone) |
| supply_chain_disclosure_best_practice_audits | qwen3.7-plus | 10 | PILOT | 0.00% [0.00%, 27.75%] | 0.00% | 100.00% | 100.00% | 50.00% | 50.00% | 50.00% | 0.0000 | 0.0000 | 5.0 | 0.00% | 100.00% | 100.00% | 0.00% | 76.60% (GPT-3.5) | **invalid-api-errors** (cheap-alone, cheap-alone, cheap-alone) |

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
