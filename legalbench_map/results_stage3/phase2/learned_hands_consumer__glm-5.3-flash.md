# Phase 2 paired cascade analysis: learned_hands_consumer x glm-5.3-flash [scoring: exact]

- scoring: exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.

- task: `learned_hands_consumer`  incumbent: `glm-5.3-flash`  cheap candidate: `embed_small_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 614 (n_test expected 614); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=85.34% Wilson95=[82.32%, 87.92%] (n=614)
- balanced accuracy=85.34%
- parse-fail rate=3.58% (22 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=1694.9 out=673.3 cached=0.0; mean latency=7.50s
- total cost=0.3628 USD; measured $/1k items=0.5909
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.20% by GPT-4 (https://arxiv.org/pdf/2308.11462 (Table 64, page 123))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=84.85%, bal_acc=84.85%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0049 CI95=[-0.0358, 0.0261], a-only=44 b-only=47 discordant=91, mid-p=0.7547
  - non-inferiority cheap vs incumbent: diff=-0.0049 one-sided-lo=-0.0309 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=71.98%, cheap acc on sent=69.40%; (on kept: incumbent=93.46%, cheap=94.24%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0259 CI95=[-0.0948, 0.0431], a-only=31 b-only=37 discordant=68, mid-p=0.4704
- (5) Cascade: acc=85.83%, bal_acc=85.83%, LLM share=37.79%, $/1k items=0.2233 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0098 CI95=[-0.0179, 0.0358], a-only=37 b-only=31 discordant=68, mid-p=0.4704; NI diff=0.0098 one-sided-lo=-0.0130 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0049 CI95=[-0.0098, 0.0195], a-only=13 b-only=10 discordant=23, mid-p=0.5413; NI diff=0.0049 one-sided-lo=-0.0081 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.51%, LLM share=15.15% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.7547 survives=no, sent: p=0.4704 survives=no, cascade_vs_cheap: p=0.4704 survives=no, cascade_vs_incumbent: p=0.5413 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=83.88%, bal_acc=83.88%, keep_rate=63.68%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0147 CI95=[-0.0456, 0.0163], a-only=43 b-only=52 discordant=95, mid-p=0.3584
  - non-inferiority cheap vs incumbent: diff=-0.0147 one-sided-lo=-0.0407 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=223 (36.32%); incumbent acc on sent=72.65%, cheap acc on sent=65.47%; (on kept: incumbent=92.58%, cheap=94.37%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0717 CI95=[-0.1435, 0.0000], a-only=25 b-only=41 discordant=66, mid-p=0.0498
- (5) Cascade: acc=86.48%, bal_acc=86.48%, LLM share=36.32%, $/1k items=0.2146 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0261 CI95=[0.0000, 0.0521], a-only=41 b-only=25 discordant=66, mid-p=0.0498; NI diff=0.0261 one-sided-lo=0.0049 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0114 CI95=[-0.0049, 0.0277], a-only=18 b-only=11 discordant=29, mid-p=0.2005; NI diff=0.0114 one-sided-lo=-0.0033 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.35%, LLM share=16.12% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.3584 survives=no, sent: p=0.0498 survives=no, cascade_vs_cheap: p=0.0498 survives=no, cascade_vs_incumbent: p=0.2005 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=84.20%, bal_acc=84.20%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0114 CI95=[-0.0423, 0.0195], a-only=43 b-only=50 discordant=93, mid-p=0.4705
  - non-inferiority cheap vs incumbent: diff=-0.0114 one-sided-lo=-0.0375 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=73.71%, cheap acc on sent=68.97%; (on kept: incumbent=92.41%, cheap=93.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0474 CI95=[-0.1121, 0.0172], a-only=24 b-only=35 discordant=59, mid-p=0.1550
- (5) Cascade: acc=85.99%, bal_acc=85.99%, LLM share=37.79%, $/1k items=0.2233 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0179 CI95=[-0.0065, 0.0423], a-only=35 b-only=24 discordant=59, mid-p=0.1550; NI diff=0.0179 one-sided-lo=-0.0016 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0065 CI95=[-0.0114, 0.0244], a-only=19 b-only=15 discordant=34, mid-p=0.4996; NI diff=0.0065 one-sided-lo=-0.0081 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.35%, LLM share=15.80% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.4705 survives=no, sent: p=0.1550 survives=no, cascade_vs_cheap: p=0.1550 survives=no, cascade_vs_incumbent: p=0.4996 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=84.31%, bal_acc=84.31%, keep_rate=62.70%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0103 CI95=[-0.0413, 0.0206], a-only=43.3333 b-only=49.6667 discordant=93.0000, mid-p=0.5278
  - non-inferiority cheap vs incumbent: diff=-0.0103 one-sided-lo=-0.0364 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=229.0 (37.30%); incumbent acc on sent=72.78%, cheap acc on sent=67.94%; (on kept: incumbent=92.82%, cheap=94.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0483 CI95=[-0.1168, 0.0201], a-only=26.6667 b-only=37.6667 discordant=64.3333, mid-p=0.2251
- (5) Cascade: acc=86.10%, bal_acc=86.10%, LLM share=37.30%, $/1k items=0.2204 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0179 CI95=[-0.0081, 0.0434], a-only=37.6667 b-only=26.6667 discordant=64.3333, mid-p=0.2251; NI diff=0.0179 one-sided-lo=-0.0033 margin=0.0100 passes=0.6667 (None)
  - (5b) vs incumbent-alone: McNemar diff=0.0076 CI95=[-0.0087, 0.0239], a-only=16.6667 b-only=12.0000 discordant=28.6667, mid-p=0.4138; NI diff=0.0076 one-sided-lo=-0.0065 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.40%, LLM share=15.69% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=0.00
- verdict inputs (fraction of repeats): cheap_ni_passes=0.00, cascade_ni_passes=1.00, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

## Verdict

- per-repeat verdicts: ['unclear', 'unclear', 'unclear']
- **(task, model) verdict: unclear**

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
