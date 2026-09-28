# LegalBench expansion: 22 additional tasks

This is a separate experiment from `../legalbench_map/`. It adds the 22
tasks selected on 2026-09-20; none overlaps the original 37-task registry.
No new full-cohort scores or LLM comparisons have been produced yet.

The candidates are **TF-IDF + logistic regression** and a majority-class
baseline. This runner does not call LLMs, download models, or access the
network. Its data and evaluation-code snapshot are already included.

## Run

From `/Users/vasyl/zadumai`:

```bash
.venv/bin/python legalbench_expansion_22/run.py --list
.venv/bin/python legalbench_expansion_22/run.py --validate
.venv/bin/python legalbench_expansion_22/run.py --smoke
.venv/bin/python legalbench_expansion_22/run.py --run-name expansion-22-first
```

The last command runs all 22 tasks, both candidates, five folds and three
repeats. It can take substantial time because TF-IDF hyperparameters are
selected inside each training fold. To run a subset:

```bash
.venv/bin/python legalbench_expansion_22/run.py \
  --tasks definition_classification,function_of_decision_section \
  --candidates tfidf_logreg --run-name rhetorical-first
```

Use a new run name for every invocation. Existing results cannot be
overwritten. Without `--run-name`, the runner generates a unique time-based
name. Unknown tasks, original-cohort tasks, output path traversal and
redirecting the results directory with a symlink are rejected.

## Where things live

| Path | Contents |
|---|---|
| `data/task_registry.json` | Only these 22 tasks, with exact input fields and label counts. |
| `data/raw/<task>/` | Pinned train/test TSVs; no shared Hugging Face cache. |
| `data/task_instructions/` | Original prompt templates defining permissible model inputs. |
| `data/source_manifest.json` | Source URLs and file hashes for all data and prompts. |
| `data/selection.json` | Frozen task selection and upstream revisions. |
| `data/original_37_task_names.json` | Exclusion list copied from the prior experiment. |
| `engine/` | Unmodified snapshot of the existing runner, core and two candidates. |
| `results/<run-name>/` | New experiment results only. |
| `smoke_results/<run-name>/` | Tiny wiring checks, never full benchmark results. |
| `.cache/` | Local plotting/candidate cache, separate from the prior experiment. |

The wrapper validates the data and engine hashes, input fields, split sizes,
label counts and cohort separation before fitting. A run saves its own
registry, validation report and manifest. A failed or partial run is marked
`failed` and returns an error; it is not silently treated as complete.

Smoke mode uses six test examples per class, two folds and one repeat on
`function_of_decision_section`. It exercises both candidates and the real
data-loading/scoring/output path. Its manifest says `smoke_only: true` and
`eligible_for_benchmark_reporting: false`. Its output folder also contains
`SMOKE_ONLY.txt`.

## What was added

- 14 CUAD clause-detection tasks.
- Four supply-chain disclosure tasks.
- `learned_hands_traffic` and `opp115_user_choice_control`.
- `definition_classification` and `function_of_decision_section`.

Every selected task has at least 300 original test examples. Selection was
based on task format and data size, before measuring new scores. The four
possible extraction extensions discussed in the feasibility audit are not
part of this batch; they need a different adapter.

## Evaluation meaning and limits

The engine snapshot preserves the existing TF-IDF method: word 1–2 grams
plus character 3–5 grams, balanced logistic regression, and training-only
inner cross-validation over C = 0.1, 0.3, 1, 3, 10.

It also preserves the existing **supervised adaptation** protocol: each
held-out fold comes from the benchmark test split, while other folds plus
the original small training split provide training labels. This uses more
labels than the original few-shot LegalBench setting. Outputs are labelled
CV estimates, not official LegalBench leaderboard scores.

The wrapper removes repeated identical inputs and removes any training
example whose input also occurs in the evaluation set. Pinned raw data stay
unchanged; each run records effective counts and removed rows. This removes
12 duplicate test rows from `opp115_user_choice_control` (1,254 to 1,242)
and one overlapping training row from `function_of_decision_section`
(seven to six training examples). Conflicting labels on identical inputs
cause an error instead of silent removal.

The inherited folds still operate on rows, not source-document groups.
Before using new scores in the poster, assess
document/duplicate grouping and align the LLM comparison to the same
held-out items and input information.

Two inherited reporting limitations are recorded in every run manifest:

- The confidence interval code pools repeated predictions for the same
  items. Those rows are dependent, so these are not validated intervals
  that properly account for that dependence.
- The engine's `per_item/` files contain predictions from its separately
  fitted conformal model. They do not reproduce the full-training model's
  `cv_mean_CV_estimated` column.

`data/published_scores.csv` intentionally contains only its header. No
matched LLM references were supplied for this new cohort, so the runner
reports no reference rather than borrowing scores from the old tasks or
claiming new wins.

## Check the separation safeguards

```bash
.venv/bin/python -B -m unittest discover -s legalbench_expansion_22/tests -v
```

The checks cover cohort overlap, rejection of original tasks, keeping smoke
outputs apart, refusing to overwrite an existing run, and blocking paths
or symlinks that could send results elsewhere.

## Provenance

Data revision:
`nguha/legalbench@daec8237410aa23e3faf4bc41ad8b3a7e1696826`.
Prompt revision:
`HazyResearch/legalbench@b46bf4ffae90524b2b72aaa30e7745fe9db64481`.
The inherited engine file hashes are in `data/engine_provenance.json`.
Original benchmark data retain their upstream licenses; the source URLs
are recorded per file.

The original `legalbench_map/` registry, code, cache and results are not
modified by this experiment.
