# Benchmark contamination audits

Two of this project's four benchmark suites have train/test overlap large enough
that a score on the official split is part memorisation. This directory is the
evidence for the numbers SCOPE.md quotes, in a form a reviewer can re-run.

| Suite | Finding | Script | Artifact |
|---|---|---|---|
| BeaverTails | **99.8%** of `330k_test` rows reuse a prompt seen in `330k_train` (99.2% on `30k_test`), while 0.0% of (prompt, response) *pairs* repeat | `beavertails_contam.py` | [`results/beavertails_contam.json`](results/beavertails_contam.json) |
| FinBen Headlines | **71.5%** of test rows carry a headline that appears verbatim in train, at 99.7-100% label agreement | `finben_headlines_contam.py` | [`results/finben_headlines_contam.json`](results/finben_headlines_contam.json) |

Two suites were checked and came back clean; those checks live with their arms,
not here — CUAD (4 contract pairs above 0.90 similarity, none crossing the
split) and HINT3 (0.000% / 0.605% / 0.000% exact test-row overlap,
`intent_arms/hint3_REPORT.md`). MASSIVE's cross-lingual twin hazard is audited
in `intent_arms/massive_contam.py`.

## Reproducing

```bash
.venv/bin/python contamination/beavertails_contam.py       # ~75s, 300k rows
.venv/bin/python contamination/finben_headlines_contam.py  # ~15s
```

Each script downloads its dataset at a **pinned revision**, measures, writes a
JSON artifact and a log to `results/`, and re-asserts every number the docs
quote. It exits non-zero if any of them has moved. Nothing here is a
run-once-and-file result: the audits are a regression test on a published
claim, and the exit code is the point.

For a reviewer who wants to check the claim without the download:

```bash
.venv/bin/python -m pytest contamination/ -q                # 10 tests, offline
```

Those read only the committed artifacts, confirm they match the published
numbers, confirm they record the dataset revision and file hashes they came
from, and confirm the drift guard actually fires when a value is perturbed. An
audit that cannot fail is not evidence, so that last one is what makes the rest
mean anything.

## Pins, and how to tell you have the same data

| Dataset | Revision |
|---|---|
| `PKU-Alignment/BeaverTails` | `8401fe609d288129cc684a9b3be6a93e41cfe678` |
| `TheFinAI/flare-headlines` | `39e4f9ba3515a7cf464ed07ece80a4f7f4189134` |

Every artifact carries three layers of data identity, because a revision string
alone is a promise rather than a check:

1. **`dataset.snapshot_files`** — sha256 and byte count of every file in the
   snapshot that was read. Re-download the same revision and the hashes match,
   or the upstream data moved and no number below is comparable.
2. **`content_fingerprints`** — sha256 over the columns actually used, in row
   order. Survives a change of file format or `datasets` version, which the
   file hashes do not.
3. **`environment`** — interpreter, library versions, platform, repo commit and
   whether the tree was dirty.

To audit a different revision without editing the pin:
`--revision <sha>`. To measure without asserting: `--no-verify` (the artifact
then records `verified_against_published_claim: false`, and the offline tests
reject it as evidence).

## What the numbers mean, precisely

**BeaverTails.** The dataset pairs each red-team prompt with responses from
several models and labels each *(prompt, response)* pair. The official splits
divide pairs, not prompts. The label is a judgement about the response, but the
prompt must be in the model input for that judgement to be well posed — so a
classifier can learn `prompt -> category` and score well without ever reading
the response. That is why the pair-overlap figure is reported next to the
prompt-overlap figure: 99.8% and 0.0% together are the finding. Matching is
exact string equality, the least arguable form of the claim. Both split sizes
are checked because the published baselines (LoRA-Guard, WildGuard,
GuardReasoner, MD-Judge) evaluate on 30k.

The same script measures the label ceiling. `330k_train` collapses to 99,734
unique prompt+response texts, **every one of them appearing at least three
times**. Across those repeats the annotators disagree 27.7% of the time on
`is_safe` and 23.9% on `non_violent_unethical_behavior`; only 47.1% of repeated
texts are identical on the full 14-category vector. A majority vote over the
copies tops out at 90.8% on `is_safe`. No model beats that, so it belongs beside
any F1 on this suite.

**FinBen Headlines.** Nine stacked binary sub-tasks over gold commodity
headlines. The sub-task is not a column in train — `label_type` is populated
only in test — so it has to be recovered from the instruction text. A
hand-written sub-task list is exactly the shortcut that produced this project's
`diversity_1-6` feature leak, so the vocabulary is read from the dataset's own
`label_type` column and the recovery is *proved* before any leak rate is
quoted: it reproduces `label_type` on **20,547/20,547** test rows, with no row
matching zero or multiple sub-tasks. The script refuses to report per-sub-task
leakage if that check fails.

Given a sound split, 1,632 of each sub-task's 2,283 test rows carry a headline
that is verbatim in train, leaving 651 leak-free rows per sub-task. Label
agreement on the leaked rows runs 99.7-100%, so the leak is directly
answerable by lookup rather than merely topical. Train also contradicts itself:
on Direction Up, 159 headlines appear more than once and 8 carry conflicting
labels.

One supplementary figure, not part of the published claim and not asserted:
counting the validation split as visible too raises test-row coverage from
71.5% to **81.2%**.

## Limits — what this does and does not establish

- These are **exact-match** audits. They are a lower bound on contamination;
  near-duplicate and paraphrase overlap are not measured here, so the true
  figures can only be higher.
- The FinBen leak is measured on `TheFinAI/flare-headlines`, the FinBen/PIXIU
  packaging of the Headlines task. It is a statement about that release.
- The audits establish that the splits overlap. They do **not**, on their own,
  establish what any particular model's score would have been without the
  overlap.
- SCOPE.md pairs the FinBen leak with "0.982 avg wF1 on the full split and
  0.967 on the 651 leak-free rows". That pair is a *modelling* result from the
  FinBen arm, whose code was lost in the same crash described below and is
  **not** restored by this directory. Treat it as an unbacked number until that
  arm is rebuilt.

## Provenance

The original audits ran on 2026-09-14 in a `/tmp` scratchpad that a machine OOM
wiped on 2026-09-16 (see commit `1f69f07`). The headline numbers survived in
SCOPE.md; the code and logs did not. The scripts here were recovered verbatim
from the session transcript — `bt_leakcheck.py`, `bt_dupcheck.py`,
`headlines_verify.py`, `headlines_leak.py` — and the measurement logic is
unchanged, which is why the re-run reproduces the original output to the digit,
including the full per-category disagreement table. The pinning, fingerprints,
JSON artifacts, assertions and tests are new.

One inconsistency from the original session, resolved: an early docstring in
`bt_prep_disjoint.py` cited **94.8%** for BeaverTails prompt overlap. That was
measured on the subsampled working set used for the modelling run
(10,508/11,088 rows), before `bt_leakcheck.py` ran on the full official splits.
The published figure, 99.8%, is the full-split number and is the correct one to
quote.
