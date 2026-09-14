# Phase 2 paired cascade analysis: supply_chain_disclosure_best_practice_audits x glm-5.3-flash [scoring: lenient]

- scoring: lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

- task: `supply_chain_disclosure_best_practice_audits`  incumbent: `glm-5.3-flash`  cheap candidate: `tfidf_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 379 (n_test expected 379); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=75.99% Wilson95=[71.44%, 80.02%] (n=379)
- balanced accuracy=62.90%
- parse-fail rate=0.26% (1 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=701.1 out=202.9 cached=0.0; mean latency=2.43s
- total cost=0.0783 USD; measured $/1k items=0.2066
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.60% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=80.47%, bal_acc=73.92%, keep_rate=39.05%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0449 CI95=[-0.0053, 0.0923], a-only=54 b-only=37 discordant=91, mid-p=0.0758
  - non-inferiority cheap vs incumbent: diff=0.0449 one-sided-lo=0.0026 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=231 (60.95%); incumbent acc on sent=64.94%, cheap acc on sent=72.73%; (on kept: incumbent=93.24%, cheap=92.57%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0779 CI95=[0.0000, 0.1558], a-only=50 b-only=32 discordant=82, mid-p=0.0475
- (5) Cascade: acc=75.73%, bal_acc=62.71%, LLM share=60.95%, $/1k items=0.1259 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0475 CI95=[-0.0950, -0.0026], a-only=32 b-only=50 discordant=82, mid-p=0.0475; NI diff=-0.0475 one-sided-lo=-0.0871 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=-0.0026 CI95=[-0.0185, 0.0132], a-only=4 b-only=5 discordant=9, mid-p=0.7539; NI diff=-0.0026 one-sided-lo=-0.0158 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=90.24%, LLM share=19.53% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0758 survives=no, sent: p=0.0475 survives=no, cascade_vs_cheap: p=0.0475 survives=no, cascade_vs_incumbent: p=0.7539 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=81.00%, bal_acc=73.62%, keep_rate=43.27%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0501 CI95=[0.0026, 0.0976], a-only=53 b-only=34 discordant=87, mid-p=0.0422
  - non-inferiority cheap vs incumbent: diff=0.0501 one-sided-lo=0.0106 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=215 (56.73%); incumbent acc on sent=65.58%, cheap acc on sent=73.02%; (on kept: incumbent=89.63%, cheap=91.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0744 CI95=[0.0000, 0.1488], a-only=43 b-only=27 discordant=70, mid-p=0.0568
- (5) Cascade: acc=76.78%, bal_acc=63.94%, LLM share=56.73%, $/1k items=0.1172 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0422 CI95=[-0.0844, 0.0000], a-only=27 b-only=43 discordant=70, mid-p=0.0568; NI diff=-0.0422 one-sided-lo=-0.0792 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0079 CI95=[-0.0132, 0.0290], a-only=10 b-only=7 discordant=17, mid-p=0.4807; NI diff=0.0079 one-sided-lo=-0.0106 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=89.97%, LLM share=19.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0422 survives=no, sent: p=0.0568 survives=no, cascade_vs_cheap: p=0.0568 survives=no, cascade_vs_incumbent: p=0.4807 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=81.27%, bal_acc=75.86%, keep_rate=44.06%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0528 CI95=[0.0026, 0.1055], a-only=58 b-only=38 discordant=96, mid-p=0.0417
  - non-inferiority cheap vs incumbent: diff=0.0528 one-sided-lo=0.0106 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=212 (55.94%); incumbent acc on sent=64.62%, cheap acc on sent=73.58%; (on kept: incumbent=90.42%, cheap=91.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0896 CI95=[0.0094, 0.1745], a-only=50 b-only=31 discordant=81, mid-p=0.0352
- (5) Cascade: acc=76.25%, bal_acc=63.32%, LLM share=55.94%, $/1k items=0.1156 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0501 CI95=[-0.0976, -0.0053], a-only=31 b-only=50 discordant=81, mid-p=0.0352; NI diff=-0.0501 one-sided-lo=-0.0897 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.0026 CI95=[-0.0185, 0.0211], a-only=8 b-only=7 discordant=15, mid-p=0.8036; NI diff=0.0026 one-sided-lo=-0.0132 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=91.29%, LLM share=18.73% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0417 survives=no, sent: p=0.0352 survives=no, cascade_vs_cheap: p=0.0352 survives=no, cascade_vs_incumbent: p=0.8036 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=80.91%, bal_acc=74.47%, keep_rate=42.13%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0493 CI95=[0.0000, 0.0985], a-only=55.0000 b-only=36.3333 discordant=91.3333, mid-p=0.0532
  - non-inferiority cheap vs incumbent: diff=0.0493 one-sided-lo=0.0079 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=219.3 (57.87%); incumbent acc on sent=65.05%, cheap acc on sent=73.11%; (on kept: incumbent=91.10%, cheap=91.68%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0807 CI95=[0.0031, 0.1597], a-only=47.6667 b-only=30.0000 discordant=77.6667, mid-p=0.0465
- (5) Cascade: acc=76.25%, bal_acc=63.32%, LLM share=57.87%, $/1k items=0.1196 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0466 CI95=[-0.0923, -0.0026], a-only=30.0000 b-only=47.6667 discordant=77.6667, mid-p=0.0465; NI diff=-0.0466 one-sided-lo=-0.0853 margin=0.0100 passes=0.0000 (None)
  - (5b) vs incumbent-alone: McNemar diff=0.0026 CI95=[-0.0167, 0.0211], a-only=7.3333 b-only=6.3333 discordant=13.6667, mid-p=0.6794; NI diff=0.0026 one-sided-lo=-0.0132 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=90.50%, LLM share=19.09% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=0.00
- verdict inputs (fraction of repeats): cheap_ni_passes=1.00, cascade_ni_passes=0.00, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

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
