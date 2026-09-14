# LegalBench Cheap-Model-Map

Can cheap, non-LLM models (TF-IDF+LogReg, small/base sentence embeddings
+ LogReg, zero-shot NLI, a fine-tuned encoder, plus two trivial
baselines) match published LLM scores on LegalBench tasks -- and for the
tasks where they can't, how much of the test set could still be
answered cheaply if the rest is routed ("cascaded") to an LLM, gated by
split conformal prediction?

This repo is a CV/conformal/output harness. It does not call any LLM
API, does not add new labels, and does not hand-write label
descriptions or prompts for any candidate.

## Methodology

### Data

`core/data.py` reads `data/task_registry.json` (one entry per task: task
id, `hf_config`, family, expected n_train/n_test, `input_columns`, the
canonical (normalized) `class_distribution`, and the task's `metric`).
For each task it loads BOTH the train and test splits of
`nguha/legalbench` (config = `hf_config`) from the Hugging Face Hub.
Per row, model input = that row's `input_columns` values joined, IN
ORDER, with `" [SEP] "`.

**`input_columns` = exactly the `{{placeholders}}` of the task's original
LegalBench prompt template (`tasks/<task>/base_prompt.txt`)** -- i.e.
exactly the columns the LLM saw when the published score we compare
against was produced. Every other column in the HF dataset is withheld,
however useful it looks: it is information the LLM never had. The
per-task placeholder sets live in `data/prompt_placeholders.json`, and
`tests/test_input_columns_match_prompt.py` pins the registry to them.
This rule replaced an earlier hand-written column blocklist after that
blocklist let `diversity_1..6`'s gold sub-answer columns
(`parties_are_diverse`, `aic_is_met`) through -- a leak that scored
~100% -- see `data/REGISTRY_NOTES.md` section 4.

Labels are normalized (`strip().lower()`); any
row whose normalized label is not one of the task's registered
`class_distribution` keys is dropped and the drop count is logged.

### CV protocol (`core/cv.py`)

For every task with a supported metric, EVERY item in the test split is
evaluated via 5-fold (`--folds`) stratified cross-validation **over the
test split**, where each fold's training data = (the other 4/5 of the
test split) UNION (the entire train split). This is repeated
`--repeats` (default 3) times with independent fold-partition seeds,
each repeat producing full out-of-fold (OOF) coverage of the test split.

**Aggregation choice (repeats x bootstrap CI).** The reported mean is
the plain arithmetic mean of the 3 per-repeat metric values (each
already computed on that repeat's full-coverage OOF predictions). The
95% CI is a 1000-resample bootstrap over the 3 repeats' per-item
predictions POOLED together (3 x n_test rows total, each test item
contributing once per repeat) -- this lets the CI reflect both
item-level sampling variability and fold-partition (repeat-to-repeat)
variability in one number, while the point estimate stays a simple,
repeat-symmetric average. This exact recipe is used everywhere a
mean+CI is reported (main metric, accuracy, macro-F1, conformal keep
rate, coverage, metric-on-kept, metric-on-sent) -- see the
`core/cv.py` module docstring for the full reasoning.

**No leakage into hyperparameter selection.** Candidates do their own
inner-CV hyperparameter search using only the `X_train`/`y_train` given
to them for a fold; the harness's fold-calling code never constructs a
call that includes the held-out fold's labels (there is no parameter
through which they could reach the candidate). This is checked
structurally, not just by convention.

**Leak check.** For every fold of every repeat, an assertion verifies
the held-out test-split positions are disjoint from that fold's
training test-split positions (train-split items are, by design, always
part of every fold's training pool and are never held out -- that is
correct, not a leak). See `tests/test_leak.py`.

### Conformal gate (`core/conformal.py`)

Split conformal prediction, implemented by hand (no external conformal
library). Nonconformity score = `1 - p(true label)`. Within EACH fold
of EACH repeat, a 20% calibration slice is carved OUT OF that fold's
training pool; the candidate is fit on the REMAINING 80% and, via one
`fit_predict_proba` call, used to score BOTH the calibration slice and
the held-out fold -- so the calibration threshold and the held-out
prediction sets come from the exact same fitted model. The threshold is
the standard finite-sample-corrected `(1-alpha)` quantile of calibration
nonconformity scores. A held-out item's prediction SET = every label
with nonconformity <= that threshold; `kept` = set size 1, `sent` =
otherwise. `metric_on_kept`/`metric_on_sent` use this same 80%-trained
model's point predictions (the singleton for kept items, argmax
otherwise) -- see `core/conformal.py`'s docstring for why.

### Estimated cascade & verdict (`core/verdict.py`)

    est_cascade_metric = keep_rate * metric_on_kept
                          + (1 - keep_rate) * published_best_llm_score

using the BEST available published LLM score (max over models) from
`data/published_scores.csv`. This is an ESTIMATE: sent items are
assumed to score at the LLM's published AVERAGE, which may be
optimistic since sent items are exactly the ones flagged as uncertain
by the gate (selected to be harder than a random draw). For
`balanced_accuracy` specifically, this is computed per class (assuming
the published task-average score applies uniformly across classes,
since no per-class published number exists) and then averaged. Every
such number is labeled "estimated cascade" in the outputs.

**CV-vs-published comparability assumption.** Published scores were
computed by their original authors on the same fixed test split, with
their own (typically single-pass, no-CV) methodology. We treat our
repeated-CV numbers as comparable estimates of the same quantity; this
is an assumption, not a guarantee, and every CV-derived number is
labeled "CV-estimated" throughout the outputs.

**Verdict**, per task, comparing the BEST cheap candidate's CV mean+CI
to the best published LLM score, with `gap = published_best - cv_mean`
(points) and `--delta` (default 1.0 point):

  * `no_reference` -- no published score for the task
  * `match` -- CV lower-CI-bound `>= published_best - delta`
  * `short` -- (not match) and `gap > 3`
  * `close` -- (not match) and `gap <= 3`

## Reproduce

```
/Users/vasyl/zadumai/.venv/bin/python /Users/vasyl/zadumai/legalbench_map/cli.py \
    --folds 5 --repeats 3 --alpha 0.05 --delta 1.0 --seed 0
```

Useful flags for a smaller/faster run:

```
/Users/vasyl/zadumai/.venv/bin/python /Users/vasyl/zadumai/legalbench_map/cli.py \
    --tasks abercrombie --candidates majority,stratified_random,tfidf_logreg --seed 0
```

Outputs land in `results/`: `tasks.csv`, `summary.md`,
`chart_llm_share.png`, `per_class/<task>.csv`, `configs/<task>.json`,
and `run_errors.log` (task/candidate failures are logged here and never
stop the rest of the run).

Same `--seed` + same inputs -> byte-identical `results/tasks.csv`
across two runs (see `tests/test_determinism.py`).

### Re-running a subset and merging

A full run is hours; a corrected task shouldn't cost that. Re-run only
the affected tasks into a fresh directory, then merge them into the full
results with the same writers `cli.py` uses:

```
python cli.py --tasks diversity_1,diversity_2 --results-dir results_fixed ...
python merge_results.py --base results --patch results_fixed \
    --out results_merged --tasks diversity_1,diversity_2
```

The merge replaces those tasks' rows in place, carries every other task
through untouched, regenerates `summary.md` / `chart_llm_share.png`, and
refuses a patch directory containing tasks you didn't name. Prove it is
lossless before trusting it: `python merge_results.py --base results
--out /tmp/x --self-test` must report `tasks.csv` and `summary.md`
IDENTICAL.

## Adding a new task

Append an object to `data/task_registry.json` with: `task` (unique id),
`hf_config` (the `nguha/legalbench` config name), `family`,
`is_reasoning_exception` (bool, informational), `n_train`/`n_test`
(expected sizes, sanity-checked not enforced), `input_columns` (ordered
list of dataset columns to join -- MUST equal the `{{placeholders}}` in
the task's `base_prompt.txt`; also record that list under the task id in
`data/prompt_placeholders.json`, which
`tests/test_input_columns_match_prompt.py` enforces -- never add a column
because it "looks like input"), `n_classes`, `class_distribution`
(dict of normalized-label -> count, whose KEYS define the canonical
class set), `metric` (one of `accuracy`, `balanced_accuracy`,
`macro_f1`), and `notes`. Optionally add rows to
`data/published_scores.csv` (`task,model,metric,score,source_url` --
`source_url` is required and validated non-empty) so the task gets a
verdict instead of `no_reference`.

## Do not

  * No LLM API calls anywhere in this harness or its candidates.
  * No new labels -- every label comes from the LegalBench `answer`
    column, normalized (strip + lowercase), never hand-written.
  * No hand-written label descriptions/prompts fed to any candidate.
  * No tuning on held-out folds -- candidates only ever see
    `X_train`/`y_train` for their own inner-CV; held-out `y` never
    reaches a `fit_predict_proba` call.
  * No input columns the original LLM prompt did not expose -- a
    candidate may see exactly the `{{placeholders}}` of the task's
    `base_prompt.txt`, nothing else (enforced by
    `tests/test_input_columns_match_prompt.py`).

## Tests

```
/Users/vasyl/zadumai/.venv/bin/python -m pytest /Users/vasyl/zadumai/legalbench_map/tests -q
```
