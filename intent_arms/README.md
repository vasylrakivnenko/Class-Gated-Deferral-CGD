# Intent-classification arms: HINT3 and MASSIVE

Two measurement arms run 2026-09-16. **Both landed on the "do not claim" side**,
which is why they live here rather than on the chart pages. The verdicts they
produced are recorded in `../SCOPE.md`; this directory is the evidence behind
those numbers and the code to re-run them.

Zero paid API calls were made in either arm.

## What each one settled

**HINT3** (3 companies' live chatbot logs; 328/600/471 train rows; 21/28/59
intents; 41.8%/54.4%/72.0% out-of-scope test traffic).
The finding is a precondition, not a score: **an out-of-scope threshold cannot be
selected from training data that contains no out-of-scope examples**, and a
production bot's training phrases are all in-scope by construction. Shipped
accuracy 0.6801 / 0.7023 / **0.5748**, against all-out-of-scope constants of
0.4181 / 0.5439 / **0.7202** -- the third bot loses to a label-blind answer and
its accuracy is dropped, not caveated. On the constant-proof metric (macro-F1
over in-scope intents) we rank 3rd, 3rd and 6th of 6 against five NLU platforms.
Secondary finding, now general: **you cannot use one confidence score twice** --
if it is doing scope rejection it is unavailable for routing, which is why the
cascade gate posts an inverted AUROC of 0.240 on powerplay11.
No LLM ran here, so nothing in this arm tests free-versus-paid.

**MASSIVE** (60 intents, 13 of 52 locales modelled, official id-aligned splits).
English ties published fine-tuned mT5-Base / XLM-R-Base (88.03 +/-1.17 vs
89.0 / 88.3 / 87.9); **all 12 non-English locales lose by 2.9-5.2 points.** The
models that beat us are open-weight and free, so this is the second independent
confirmation -- after BLURB/ChemProt -- that *our cheap arm is not the best free
arm*. Routing, by contrast, transfers everywhere: the gate passes 13 of 13
locales at AUROC 0.828-0.908, i.e. 72-97% of traffic answerable free at 90%
reliability.

## The live bug this arm found

`token_pattern` is never set anywhere in this project, so the word half of the
shared TF-IDF arm (`src/downshift/encoders.py:146`) inherits sklearn's default
`r"\b\w\w+\b"`, which does not match Unicode combining marks. On Devanagari that
**silently discards 88.5% of tokens**; a mark-aware pattern recovers +23.2
points on hi-IN (0.5797 -> 0.8117) with control locales moving <=0.2. Affects
Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam. No existing project
number is wrong -- every dataset measured so far is Latin or Cyrillic script.
**Not yet fixed.** Note the tension: the same pattern change makes th-TH *worse*
by 7.3 points, because there the broken default was accidentally acting as a
Thai syllable segmenter. So it is a per-script decision, and applying it
globally would perturb existing published numbers by up to ~0.2. Evidence:
`massive_tokenfix.py` / `.json`.

## Open items

1. The fine-tuned `multilingual-e5-small` control **did not converge** (final
   loss 1.68 vs the English encoder's 0.077; frozen beat fine-tuned on 4 of 4
   locales). Consequence stated precisely: this arm does **not** establish that a
   fine-tuned multilingual encoder fails to close the off-Latin gap. That is
   untested, not answered, and it is the one re-run that could move the
   non-English verdict.
2. Decide the `token_pattern` question above -- per-script parameter, or leave
   the default and document the limit.

## Files

| file | what |
|---|---|
| `hint3_arm.py`, `massive_arm.py` | the runnable arms; docstrings record each design decision and the measurement that forced it |
| `hint3_verify.py`, `massive_verify.py` | independent re-derivations (HINT3: 855 checks in pure numpy, no `sklearn.metrics`; MASSIVE: 73 accuracies + reversed-FeatureUnion refit) |
| `hint3.json`, `massive.json` | all metrics, threshold sweeps, gates, contamination. **Per-item `pred`/`conf`/`gold` arrays were dropped on archive** -- regenerate from the arm scripts if needed |
| `hint3_REPORT.md`, `massive_REPORT.md` | the full human reports |
| `massive_published.json` | mT5/XLM-R per-locale baselines parsed from arXiv:2204.08582 Table 8; parse self-checked against the paper's own Table 3a means (85.11 vs 85.1, 85.33 vs 85.3) |
| `massive_contam.json` | the 52-locale id-alignment audit: 16,521 ids in all locales, 0 partition disagreements |
| `massive_tokenfix.*` | the Devanagari token-pattern experiment |
| `massive_zeroshot.*` | cross-lingual zero-shot: beats published XLM-R zero-shot on only 3 of 12 locales |
| `hint3_data/` | the 6 source CSVs (HINT3 v1 full), byte-identical to the upstream repo |
| `hint3_run.log` | includes the metric-stack validation: all 675 of the authors' published numbers reproduced to 2.22e-16 |
