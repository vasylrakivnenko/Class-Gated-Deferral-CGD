# Registry Notes

This file explains four decisions baked into `task_registry.json` that are not
self-evident from the JSON alone.

## 1. The family round-robin trim

LegalBench's `nguha/legalbench` dataset on the Hugging Face Hub has 162
configs. The candidate pool for this experiment (every config whose name
starts with `cuad_`, `maud_`, `contract_nli_`, `opp115_`, `learned_hands_`,
`supply_chain_disclosure_`, plus the four singles `unfair_tos`,
`overruling`, `proa`, `telemarketing_sales_rule`) is 125 configs.

After loading every candidate's metadata and applying the two selection
filters (test split >= 300 items; answer field categorical with <= 20
normalized classes covering >= 99% of test items), 48 candidates qualify.
The filter is almost entirely a *test-size* filter here: every candidate's
answer column happened to be small-cardinality categorical (binary yes/no
in nearly every case), so `pass_cardinality_filter` is `true` for all 125
candidates -- the cardinality filter never actually eliminates anyone in
this pool. Test-set size is what does the work: `maud_` (34 configs) and
`contract_nli_` (14 configs) are eliminated *in their entirety* because
every one of their test splits is under 300 rows (typically 90-190). The
surviving 48 are distributed very unevenly: cuad_ (21), supply_chain_
disclosure_ (10), opp115_ (8), learned_hands_ (7), and 2 of the 4 singles
(overruling, unfair_tos -- proa and telemarketing_sales_rule also fail the
test>=300 filter, at 95 and 47 test rows respectively).

If we simply took the first N qualifying tasks in family-priority order
(cuad_ first, since it's first in the priority list and has the most
members), the result would be dominated by cuad_ and would badly
under-represent the other families -- exactly the failure mode the task
brief warned about. Instead we **round-robin across families**: walk the
family priority order (`cuad_, maud_, contract_nli_, opp115_,
learned_hands_, supply_chain_disclosure_`, then the singles) repeatedly,
taking one qualifying task from the next family in the cycle each step
(using each family's own qualifying tasks in HF's own config order), until
28 tasks have been selected. Since `maud_` and `contract_nli_` contribute
zero qualifying tasks, the effective cycle is
`cuad_ -> opp115_ -> learned_hands_ -> supply_chain_disclosure_ -> single`.

We stopped at 28 (the top of the requested 24-28 range) rather than a lower
number in that range, to keep as much of the qualifying pool as the budget
allows. The two singles (`overruling`, `unfair_tos`) are exhausted after
round 2 of the cycle and drop out of subsequent rounds; `learned_hands_`
(7 qualifying tasks) is exhausted after round 7. The final 28 qualifying tasks break down as: 7 cuad_, 7 opp115_,
6 learned_hands_, 6 supply_chain_disclosure_, 1 overruling, 1 unfair_tos
(see `task_registry.json` for the exact list and selection order;
`is_reasoning_exception` is `false` for all 28).

## 2. diversity_jurisdiction -> diversity_1..diversity_6

The task brief that generated this registry named a fixed reasoning-exception
config called `diversity_jurisdiction`. That config does not exist on
`nguha/legalbench` -- LegalBench splits this task into six numbered variants,
`diversity_1` through `diversity_6`, each testing a different structural
variant of the diversity-jurisdiction fact pattern (number of plaintiffs/
defendants/claims). We use **all six** (`diversity_1` ... `diversity_6`) as
the exception family standing in for the single named
`diversity_jurisdiction` config. All six are marked
`is_reasoning_exception: true` in the registry and are included regardless
of the 24-28 qualifying-task filters and budget, per the fixed
reasoning-exception list:
`hearsay, personal_jurisdiction, abercrombie, diversity_1..diversity_6`
(9 tasks total).

## 3. Metric mapping source

Metrics are **not** assumed to be balanced accuracy by default. They are
looked up, per task, from the authoritative source:
`https://raw.githubusercontent.com/HazyResearch/legalbench/main/evaluation.py`
(fetched once during registry construction; nothing downstream may hit
github.com again -- see section 5 below).

That file defines exactly two task-list constants that assign a metric:

- `EXACT_MATCH_BALANCED_ACC_TASKS` -- a list of ~150 task names evaluated
  with `sklearn.metrics.balanced_accuracy_score` over normalized
  (lowercased, punctuation-stripped) exact-match strings. This is the vast
  majority of LegalBench tasks and is where `metric: "balanced_accuracy"`
  in the registry comes from.
- `MANUAL_EVAL_TASKS` -- currently just `["rule_qa"]`, tasks requiring a
  human to score. `rule_qa` is not in our candidate pool or exception list,
  so this constant contributes nothing to our registry.

Every other task name that reaches the file's `evaluate()` dispatcher
without matching either list, or one of a handful of hardcoded
`elif task == "..."` / `elif task.startswith("ssla")` branches
(`sara_numeric`, `successor_liability`, `citation_prediction_open`,
`definition_extraction`, `ssla*`), raises `Exception(f"Unknown task: {task}")`
-- i.e., the repository itself has no defined metric for it. We checked
every one of our 37 selected tasks (28 qualifying + 9 exceptions) against
`EXACT_MATCH_BALANCED_ACC_TASKS` and `MANUAL_EVAL_TASKS`: **all 37 are in
`EXACT_MATCH_BALANCED_ACC_TASKS`**, so every selected task's `metric` field
in the registry is `"balanced_accuracy"`, and none had to be marked
`"unknown"`. This is a real lookup against the file's contents for each
task, not a default -- it happens that this particular candidate pool
(binary/small-cardinality classification tasks from cuad_, opp115_,
learned_hands_, supply_chain_disclosure_, the singles, and the fixed
exception list) is fully covered by that one list.

## 4. Input-column rule: the classifier may only see what the LLM saw

**Rule.** For every task, `input_columns` = exactly the `{{placeholder}}`
names in that task's original prompt template
(`tasks/<task>/base_prompt.txt` in the LegalBench repo). Those placeholders
are, by definition, the only dataset columns the LLM was shown when the
published score we compare against was produced. Anything else in the HF
dataset -- however plausible it looks -- is information the LLM never had,
and feeding it to a cheap candidate makes the comparison invalid.

The extracted placeholder set for every selected task lives in
`data/prompt_placeholders.json` (mechanically parsed from each
`base_prompt.txt`), and `tests/test_input_columns_match_prompt.py` pins the
registry to it, so a mismatch fails the suite rather than silently
producing a flattering number.

Kept columns are concatenated with `" [SEP] "` in the dataset's own column
order (not resorted).

**Why this replaced the earlier blocklist rule.** The first version of
this registry used `all columns except {index, answer, slice, document_name}`
-- i.e. it *guessed* which columns were metadata. The guess was right for
`slice` (hearsay/personal_jurisdiction's post-hoc analysis category, which
encodes the answer) and `document_name` (CUAD/contract_nli filenames,
memorizable), and those exclusions are subsumed by the placeholder rule.
But it was **wrong** for `diversity_1..6`: the dataset ships
`parties_are_diverse` and `aic_is_met`, and the blocklist reasoning kept
them as "compositional inputs by design". They are not inputs -- the
original prompt is `{{text}}` only -- they are gold annotations of the two
intermediate reasoning steps. With them present, a logistic regression
learned `answer = diverse AND aic` and scored ~100% on all six tasks. That
was a leak, caught by a placeholder audit after the first full run, and
the six tasks were rerun with `input_columns = ['text']`.

The lesson generalizes: for any benchmark-vs-published comparison, derive
"what the classifier sees" from the artifact that defines what the
reference system saw (here, the prompt template) -- never from a
human-written allow/deny list of columns.

## 5. Network-access boundary (read this before wiring up later modules)

Step 7 of registry construction (pre-fetching `README.md` and
`base_prompt.txt` for every selected task, written to
`data/task_instructions/<task>.json`) is the **only** point in this whole
pipeline that is allowed to contact `github.com` /
`raw.githubusercontent.com` / `api.github.com`, and it happens exactly once,
offline, during registry construction. All 37 selected tasks' instruction
files were resolved successfully (0 fallbacks) using a single fetch of the
repository's full git tree (to handle case-sensitive filename mismatches,
e.g. `abercrombie/README.MD` vs. the usual `README.md`), so no task needs a
generic fallback template.

**Nothing downstream of this registry -- the candidate scoring modules, the
CV harness, the zero-shot NLI baseline, the CLI -- may ever make a
non-Hugging-Face network call at runtime.** The only host any later module
may contact is `huggingface.co` (for `datasets.load_dataset`, and only for
configs already cached under `cache/hf/` from this construction run). If a
later module needs task framing text for a zero-shot NLI hypothesis
template, it must read it from `data/task_instructions/<task>.json`, never
re-fetch it from GitHub.
