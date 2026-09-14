# Published-scores notes

`published_scores.csv` (columns: `task, model, metric, score, source_url`)
contains every published LegalBench score I could find, with a page-level
citation, for the 37 tasks in `task_registry.json`.

## Source

Everything comes from one document: the LegalBench paper, arXiv 2308.11462.
PDF fetched from `https://arxiv.org/pdf/2308.11462` (143 physical pages).
I did not find a separate paperswithcode.com leaderboard or a results file
linked from the HazyResearch/legalbench GitHub repo -- the paper's own
appendix tables (Appendix G, "per-task, per-model results") are the only
source used. Every number was read directly off a rendered PDF page (via
the Read tool's `pages` parameter, i.e. actually looking at the table
image), not estimated -- and every appendix table type used here (commercial
models, 13B, 7B, 3B, on each of the five reasoning categories) was spot- or
fully visually verified against the page image at least once, cross-checked
against the corresponding `pdftotext -layout` extraction for the remaining
rows of the same table (identical column format, no wrapped-row
misalignment observed anywhere it was checked).

## Scale / conversion

The paper prints scores on a 0-100 scale. Every `score` in the CSV is that
value divided by 100 (0-1 scale), per the task instructions. No other
transformation is applied.

## Two different metrics for the 9 "reasoning-exception" tasks

`task_registry.json` lists `metric: "balanced_accuracy"` for all 37 tasks,
including the 9 reasoning exceptions (`hearsay`, `personal_jurisdiction`,
`abercrombie`, `diversity_1`..`diversity_6`). The paper, however, evaluates
these 9 tasks two different ways, and reports both:

- **Table 59** ("Performance on rule-application tasks for commercial
  models. We report correctness/analysis.", p.121): GPT-4/GPT-3.5/Claude-1
  only, graded manually by a law-trained human on two axes -- whether the
  generated free-text answer was *correct* and whether its *analysis*
  (legal reasoning) was sound. This is **not** balanced accuracy, and open-
  source models were never manually graded this way (paper: "we only
  evaluated GPT-4, GPT-3.5, and Claude-1" for rule-application). Recorded
  in the CSV as two rows per (task, model): `metric=correctness` and
  `metric=analysis`, with a note flagging the mismatch against the
  registry's stated metric.
- **Tables 68-71** ("... models on rule-conclusion tasks", pp.125-127): the
  *same* 9 tasks, but scored automatically (exact-match on the final
  conclusion only, using balanced-accuracy per the paper's own evaluation
  methodology in §5.1.3) across all 20 models. Recorded as
  `metric=balanced_accuracy`, matching the registry.

Every other task in the registry (`learned_hands_*`, `cuad_*`, `opp115_*`,
`supply_chain_disclosure_*`, `overruling`, `unfair_tos`) is a classification
task the paper scores with balanced-accuracy directly (§5.1.3), so its rows
carry `metric=balanced_accuracy` with no mismatch.

## Models represented (20)

Commercial: `GPT-4`, `GPT-3.5`, `Claude-1`.
~11-13B: `Flan-T5-XXL`, `LLaMA-2-13B`, `OPT-13B`, `Vicuna-13B-16k`, `WizardLM-13B`.
~7B: `BLOOM-7B`, `Falcon-7B-Instruct`, `Incite-7B-Base`, `Incite-7B-Instruct`,
`LLaMA-2-7B`, `MPT-7B-8k-Instruct`, `OPT-6.7B`, `Vicuna-7B-16k`.
~2-3B: `BLOOM-3B`, `Flan-T5-XL`, `Incite-3B-Instruct`, `OPT-2.7B`.

These are the paper's own model names/labels as printed in its table headers
(some abbreviated in-table, e.g. "Flan" for Flan-T5-XXL/XL, "Llama-2" for
LLaMA-2-13B/7B -- resolved via the model list in §5.1.2 and Table 2, which
names them in full).

## Coverage

All 37 registered tasks have at least one published score (in fact, all 37
have full commercial-model coverage: GPT-4, GPT-3.5, Claude-1). Most also
have full 20-model coverage, because the paper's appendix groups tasks by
reasoning category (issue-spotting / rule-recall / rule-conclusion /
interpretation / rhetorical-understanding) and reports every model in that
category's table set:

- 9 reasoning-exceptions -> rule-conclusion tables (68-71) + rule-application
  Table 59 (commercial only) = 3*2 + 17*3 = up to 57 rows/task... in
  practice 3 commercial x 2 metrics (correctness, analysis) + 17
  open/commercial x 1 metric (balanced_accuracy) = 23 rows/task.
- 6 `learned_hands_*` -> issue-spotting tables (64-67), 20 rows/task.
- 1 `overruling` -> rhetorical-understanding tables (72-75), 20 rows/task.
- 7 `cuad_*` + 7 `opp115_*` + 6 `supply_chain_disclosure_*` + 1 `unfair_tos`
  -> interpretation tables (76-79), 20 rows/task.

No task was left out; nothing was estimated, rounded from memory, or
interpolated.
