# LegalBench Cheap-Model-Map -- Summary

One row per task, showing the BEST cheap candidate for that task (all numbers below are CV-estimated unless marked estimated cascade), sorted by estimated LLM share ascending (tasks needing the LLM least come first).

| task | best candidate | metric | CV-estimated mean | 95% CI | keep rate | est. LLM share | estimated cascade | published best | source | gap (pts) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | embed_small_logreg | balanced_accuracy | 0.850 | [0.833, 0.867] | 0.627 | 0.373 | 0.874 | 0.762 | https://arxiv.org/pdf/2308.11462 (Table 64, page 123) | -8.76 | match |
| opp115_data_retention | embed_base_logreg | balanced_accuracy | 0.812 | [0.786, 0.839] | 0.468 | 0.532 | 0.812 | 0.705 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | -10.75 | match |
| supply_chain_disclosure_best_practice_audits | tfidf_logreg | balanced_accuracy | 0.759 | [0.733, 0.787] | 0.421 | 0.579 | 0.795 | 0.766 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 0.70 | close |

**Verdict counts** (best candidate per task, n=3 tasks): match=2, close=1, short=0, no_reference=0.

**CV-vs-published comparability assumption.** Published LLM scores were
computed by their original authors on the fixed LegalBench test split
with their own methodology (typically a single pass, no CV). The
CV-estimated numbers in this report evaluate the same test-split items
via repeated stratified 5-fold cross-validation so every item gets an
out-of-fold prediction. We treat the two as comparable estimates of
"score on the LegalBench test split for this task" -- this is an
assumption, not a guarantee, and every CV-derived number is labeled
"CV-estimated" to keep that assumption visible.

**Estimated-cascade assumption.** `est_cascade_metric` = keep_rate *
metric_on_kept + (1 - keep_rate) * published_best_llm_score. Sent items
are assumed to score at the LLM's published AVERAGE, which may be
optimistic: sent items are exactly the ones the conformal gate flagged
as uncertain, i.e. selected to be harder than a random draw from the
test set. For balanced_accuracy, the estimate is computed per class
(assuming the published task-average score applies uniformly across
classes, since no per-class published number exists) and then averaged.
Every such number is labeled "estimated cascade".
