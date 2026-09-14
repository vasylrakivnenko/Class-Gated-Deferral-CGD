# Phase 2 paired cascade analysis: supply_chain_disclosure_best_practice_audits x deepseek-v4-pro-fireworks [scoring: exact]

- scoring: exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.

- task: `supply_chain_disclosure_best_practice_audits`  incumbent: `deepseek-v4-pro-fireworks`  cheap candidate: `tfidf_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 379 (n_test expected 379); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=77.04% Wilson95=[72.55%, 80.99%] (n=379)
- balanced accuracy=65.04%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=695.3 out=145.8 cached=0.0; mean latency=3.46s
- total cost=0.5666 USD; measured $/1k items=1.4951
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.60% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=80.47%, bal_acc=73.92%, keep_rate=39.05%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0343 CI95=[-0.0158, 0.0844], a-only=54 b-only=41 discordant=95, mid-p=0.1843
  - non-inferiority cheap vs incumbent: diff=0.0343 one-sided-lo=-0.0079 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=231 (60.95%); incumbent acc on sent=67.97%, cheap acc on sent=72.73%; (on kept: incumbent=91.22%, cheap=92.57%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0476 CI95=[-0.0303, 0.1255], a-only=48 b-only=37 discordant=85, mid-p=0.2354
- (5) Cascade: acc=77.57%, bal_acc=65.88%, LLM share=60.95%, $/1k items=0.9113 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0290 CI95=[-0.0765, 0.0185], a-only=37 b-only=48 discordant=85, mid-p=0.2354; NI diff=-0.0290 one-sided-lo=-0.0686 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0053 CI95=[-0.0106, 0.0211], a-only=6 b-only=4 discordant=10, mid-p=0.5488; NI diff=0.0053 one-sided-lo=-0.0079 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.29%, LLM share=19.53% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.1843 survives=no, sent: p=0.2354 survives=no, cascade_vs_cheap: p=0.2354 survives=no, cascade_vs_incumbent: p=0.5488 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=81.00%, bal_acc=73.62%, keep_rate=43.27%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0396 CI95=[-0.0079, 0.0897], a-only=53 b-only=38 discordant=91, mid-p=0.1174
  - non-inferiority cheap vs incumbent: diff=0.0396 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=215 (56.73%); incumbent acc on sent=68.37%, cheap acc on sent=73.02%; (on kept: incumbent=88.41%, cheap=91.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0465 CI95=[-0.0326, 0.1209], a-only=41 b-only=31 discordant=72, mid-p=0.2416
- (5) Cascade: acc=78.36%, bal_acc=66.69%, LLM share=56.73%, $/1k items=0.8481 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0264 CI95=[-0.0712, 0.0158], a-only=31 b-only=41 discordant=72, mid-p=0.2416; NI diff=-0.0264 one-sided-lo=-0.0633 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0132 CI95=[-0.0079, 0.0369], a-only=12 b-only=7 discordant=19, mid-p=0.2632; NI diff=0.0132 one-sided-lo=-0.0053 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.03%, LLM share=19.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.1174 survives=no, sent: p=0.2416 survives=no, cascade_vs_cheap: p=0.2416 survives=no, cascade_vs_incumbent: p=0.2632 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=81.27%, bal_acc=75.86%, keep_rate=44.06%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0422 CI95=[-0.0079, 0.0950], a-only=58 b-only=42 discordant=100, mid-p=0.1109
  - non-inferiority cheap vs incumbent: diff=0.0422 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=212 (55.94%); incumbent acc on sent=66.98%, cheap acc on sent=73.58%; (on kept: incumbent=89.82%, cheap=91.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0660 CI95=[-0.0189, 0.1509], a-only=50 b-only=36 discordant=86, mid-p=0.1329
- (5) Cascade: acc=77.57%, bal_acc=65.65%, LLM share=55.94%, $/1k items=0.8363 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0369 CI95=[-0.0844, 0.0106], a-only=36 b-only=50 discordant=86, mid-p=0.1329; NI diff=-0.0369 one-sided-lo=-0.0765 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0053 CI95=[-0.0132, 0.0237], a-only=8 b-only=6 discordant=14, mid-p=0.6072; NI diff=0.0053 one-sided-lo=-0.0106 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=92.35%, LLM share=18.73% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.1109 survives=no, sent: p=0.1329 survives=no, cascade_vs_cheap: p=0.1329 survives=no, cascade_vs_incumbent: p=0.6072 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=80.91%, bal_acc=74.47%, keep_rate=42.13%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0387 CI95=[-0.0106, 0.0897], a-only=55.0000 b-only=40.3333 discordant=95.3333, mid-p=0.1375
  - non-inferiority cheap vs incumbent: diff=0.0387 one-sided-lo=-0.0026 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=219.3 (57.87%); incumbent acc on sent=67.77%, cheap acc on sent=73.11%; (on kept: incumbent=89.82%, cheap=91.68%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0534 CI95=[-0.0272, 0.1325], a-only=46.3333 b-only=34.6667 discordant=81.0000, mid-p=0.2033
- (5) Cascade: acc=77.84%, bal_acc=66.07%, LLM share=57.87%, $/1k items=0.8652 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0308 CI95=[-0.0774, 0.0150], a-only=34.6667 b-only=46.3333 discordant=81.0000, mid-p=0.2033; NI diff=-0.0308 one-sided-lo=-0.0695 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0079 CI95=[-0.0106, 0.0273], a-only=8.6667 b-only=5.6667 discordant=14.3333, mid-p=0.4731; NI diff=0.0079 one-sided-lo=-0.0079 margin=0.0100 passes=0.6667 (None)
- (6) Oracle gate (route exactly the cheap errors): acc=91.56%, LLM share=19.09% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=0.00
- verdict inputs (fraction of repeats): cheap_ni_passes=1.00, cascade_ni_passes=0.67, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

## Verdict

- per-repeat verdicts: ['cheap-alone', 'cheap-alone', 'cheap-alone']
- **(task, model) verdict: cheap-alone**

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
