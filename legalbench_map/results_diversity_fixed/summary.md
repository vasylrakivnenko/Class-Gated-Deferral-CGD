# LegalBench Cheap-Model-Map -- Summary

One row per task, showing the BEST cheap candidate for that task (all numbers below are CV-estimated unless marked estimated cascade), sorted by estimated LLM share ascending (tasks needing the LLM least come first).

| task | best candidate | metric | CV-estimated mean | 95% CI | keep rate | est. LLM share | estimated cascade | published best | source | gap (pts) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diversity_1 | embed_base_logreg | balanced_accuracy | 0.707 | [0.670, 0.743] | 0.414 | 0.586 | 0.939 | 1.000 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 29.34 | short |
| diversity_2 | embed_base_logreg | balanced_accuracy | 0.712 | [0.678, 0.747] | 0.304 | 0.696 | 0.958 | 0.998 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 28.58 | short |
| diversity_4 | embed_base_logreg | balanced_accuracy | 0.674 | [0.644, 0.703] | 0.160 | 0.840 | 0.973 | 1.000 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 32.63 | short |
| diversity_5 | embed_small_logreg | balanced_accuracy | 0.644 | [0.612, 0.674] | 0.120 | 0.880 | 0.913 | 0.932 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 28.85 | short |
| diversity_3 | embed_small_logreg | balanced_accuracy | 0.611 | [0.577, 0.644] | 0.118 | 0.882 | 0.935 | 0.970 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 35.88 | short |
| diversity_6 | tfidf_logreg | balanced_accuracy | 0.555 | [0.522, 0.586] | 0.098 | 0.902 | 0.875 | 0.905 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 34.96 | short |

**Verdict counts** (best candidate per task, n=6 tasks): match=0, close=0, short=6, no_reference=0.

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
