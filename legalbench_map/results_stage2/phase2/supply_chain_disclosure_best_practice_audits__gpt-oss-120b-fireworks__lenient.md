# Phase 2 paired cascade analysis: supply_chain_disclosure_best_practice_audits x gpt-oss-120b-fireworks [scoring: lenient]

- scoring: lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

- task: `supply_chain_disclosure_best_practice_audits`  incumbent: `gpt-oss-120b-fireworks`  cheap candidate: `tfidf_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 379 (n_test expected 379); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=82.06% Wilson95=[77.88%, 85.59%] (n=379)
- balanced accuracy=72.34%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=756.2 out=196.5 cached=67.4; mean latency=1.91s
- total cost=0.0842 USD; measured $/1k items=0.2222
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.60% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=80.47%, bal_acc=73.92%, keep_rate=39.05%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0158 CI95=[-0.0633, 0.0290], a-only=37 b-only=43 discordant=80, mid-p=0.5052
  - non-inferiority cheap vs incumbent: diff=-0.0158 one-sided-lo=-0.0554 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=231 (60.95%); incumbent acc on sent=73.16%, cheap acc on sent=72.73%; (on kept: incumbent=95.95%, cheap=92.57%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0043 CI95=[-0.0779, 0.0693], a-only=36 b-only=37 discordant=73, mid-p=0.9076
- (5) Cascade: acc=80.74%, bal_acc=70.47%, LLM share=60.95%, $/1k items=0.1354 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0026 CI95=[-0.0422, 0.0475], a-only=37 b-only=36 discordant=73, mid-p=0.9076; NI diff=0.0026 one-sided-lo=-0.0343 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=-0.0132 CI95=[-0.0290, 0.0000], a-only=1 b-only=6 discordant=7, mid-p=0.0703; NI diff=-0.0132 one-sided-lo=-0.0264 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=91.82%, LLM share=19.53% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.5052 survives=no, sent: p=0.9076 survives=no, cascade_vs_cheap: p=0.9076 survives=no, cascade_vs_incumbent: p=0.0703 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: incumbent**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=81.00%, bal_acc=73.62%, keep_rate=43.27%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0106 CI95=[-0.0554, 0.0343], a-only=35 b-only=39 discordant=74, mid-p=0.6445
  - non-inferiority cheap vs incumbent: diff=-0.0106 one-sided-lo=-0.0475 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=215 (56.73%); incumbent acc on sent=73.02%, cheap acc on sent=73.02%; (on kept: incumbent=93.90%, cheap=91.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[-0.0698, 0.0698], a-only=31 b-only=31 discordant=62, mid-p=0.8991
- (5) Cascade: acc=81.00%, bal_acc=70.66%, LLM share=56.73%, $/1k items=0.1261 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[-0.0396, 0.0396], a-only=31 b-only=31 discordant=62, mid-p=0.8991; NI diff=0.0000 one-sided-lo=-0.0343 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=-0.0106 CI95=[-0.0290, 0.0079], a-only=4 b-only=8 discordant=12, mid-p=0.2668; NI diff=-0.0106 one-sided-lo=-0.0264 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=91.29%, LLM share=19.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.6445 survives=no, sent: p=0.8991 survives=no, cascade_vs_cheap: p=0.8991 survives=no, cascade_vs_incumbent: p=0.2668 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: incumbent**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=81.27%, bal_acc=75.86%, keep_rate=44.06%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0079 CI95=[-0.0528, 0.0369], a-only=38 b-only=41 discordant=79, mid-p=0.7376
  - non-inferiority cheap vs incumbent: diff=-0.0079 one-sided-lo=-0.0475 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=212 (55.94%); incumbent acc on sent=72.64%, cheap acc on sent=73.58%; (on kept: incumbent=94.01%, cheap=91.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0094 CI95=[-0.0660, 0.0849], a-only=35 b-only=33 discordant=68, mid-p=0.8099
- (5) Cascade: acc=80.74%, bal_acc=70.24%, LLM share=55.94%, $/1k items=0.1243 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0053 CI95=[-0.0475, 0.0369], a-only=33 b-only=35 discordant=68, mid-p=0.8099; NI diff=-0.0053 one-sided-lo=-0.0396 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=-0.0132 CI95=[-0.0317, 0.0026], a-only=3 b-only=8 discordant=11, mid-p=0.1460; NI diff=-0.0132 one-sided-lo=-0.0290 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=92.08%, LLM share=18.73% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.7376 survives=no, sent: p=0.8099 survives=no, cascade_vs_cheap: p=0.8099 survives=no, cascade_vs_incumbent: p=0.1460 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: incumbent**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=80.91%, bal_acc=74.47%, keep_rate=42.13%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0114 CI95=[-0.0572, 0.0334], a-only=36.6667 b-only=41.0000 discordant=77.6667, mid-p=0.6291
  - non-inferiority cheap vs incumbent: diff=-0.0114 one-sided-lo=-0.0501 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=219.3 (57.87%); incumbent acc on sent=72.94%, cheap acc on sent=73.11%; (on kept: incumbent=94.62%, cheap=91.68%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0017 CI95=[-0.0712, 0.0746], a-only=34.0000 b-only=33.6667 discordant=67.6667, mid-p=0.8722
- (5) Cascade: acc=80.83%, bal_acc=70.46%, LLM share=57.87%, $/1k items=0.1286 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0009 CI95=[-0.0431, 0.0413], a-only=33.6667 b-only=34.0000 discordant=67.6667, mid-p=0.8722; NI diff=-0.0009 one-sided-lo=-0.0361 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=-0.0123 CI95=[-0.0299, 0.0035], a-only=2.6667 b-only=7.3333 discordant=10.0000, mid-p=0.1611; NI diff=-0.0123 one-sided-lo=-0.0273 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=91.73%, LLM share=19.09% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=0.00
- verdict inputs (fraction of repeats): cheap_ni_passes=0.00, cascade_ni_passes=0.00, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

## Verdict

- per-repeat verdicts: ['incumbent', 'incumbent', 'incumbent']
- **(task, model) verdict: incumbent**

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
