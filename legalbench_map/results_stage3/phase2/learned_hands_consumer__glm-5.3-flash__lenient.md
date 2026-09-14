# Phase 2 paired cascade analysis: learned_hands_consumer x glm-5.3-flash [scoring: lenient]

- scoring: lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

- task: `learned_hands_consumer`  incumbent: `glm-5.3-flash`  cheap candidate: `embed_small_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 614 (n_test expected 614); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=87.46% Wilson95=[84.60%, 89.85%] (n=614)
- balanced accuracy=87.46%
- parse-fail rate=1.14% (7 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=1694.9 out=673.3 cached=0.0; mean latency=7.50s
- total cost=0.3628 USD; measured $/1k items=0.5909
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.20% by GPT-4 (https://arxiv.org/pdf/2308.11462 (Table 64, page 123))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=84.85%, bal_acc=84.85%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0261 CI95=[-0.0554, 0.0049], a-only=35 b-only=51 discordant=86, mid-p=0.0857
  - non-inferiority cheap vs incumbent: diff=-0.0261 one-sided-lo=-0.0505 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=76.72%, cheap acc on sent=69.40%; (on kept: incumbent=93.98%, cheap=94.24%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0733 CI95=[-0.1379, -0.0043], a-only=24 b-only=41 discordant=65, mid-p=0.0356
- (5) Cascade: acc=87.62%, bal_acc=87.62%, LLM share=37.79%, $/1k items=0.2233 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0277 CI95=[0.0016, 0.0537], a-only=41 b-only=24 discordant=65, mid-p=0.0356; NI diff=0.0277 one-sided-lo=0.0049 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0016 CI95=[-0.0130, 0.0163], a-only=11 b-only=10 discordant=21, mid-p=0.8318; NI diff=0.0016 one-sided-lo=-0.0098 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=93.16%, LLM share=15.15% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0857 survives=no, sent: p=0.0356 survives=no, cascade_vs_cheap: p=0.0356 survives=no, cascade_vs_incumbent: p=0.8318 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=83.88%, bal_acc=83.88%, keep_rate=63.68%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0358 CI95=[-0.0668, -0.0065], a-only=34 b-only=56 discordant=90, mid-p=0.0206
  - non-inferiority cheap vs incumbent: diff=-0.0358 one-sided-lo=-0.0603 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
- (4) SENT subset (kept==0): n_sent=223 (36.32%); incumbent acc on sent=77.13%, cheap acc on sent=65.47%; (on kept: incumbent=93.35%, cheap=94.37%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.1166 CI95=[-0.1839, -0.0448], a-only=19 b-only=45 discordant=64, mid-p=0.0011
- (5) Cascade: acc=88.11%, bal_acc=88.11%, LLM share=36.32%, $/1k items=0.2146 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0423 CI95=[0.0179, 0.0684], a-only=45 b-only=19 discordant=64, mid-p=0.0011; NI diff=0.0423 one-sided-lo=0.0212 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0065 CI95=[-0.0098, 0.0228], a-only=15 b-only=11 discordant=26, mid-p=0.4421; NI diff=0.0065 one-sided-lo=-0.0065 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=93.00%, LLM share=16.12% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0206 survives=yes, sent: p=0.0011 survives=yes, cascade_vs_cheap: p=0.0011 survives=yes, cascade_vs_incumbent: p=0.4421 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=yes
- **per-repeat verdict: earns**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=84.20%, bal_acc=84.20%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0326 CI95=[-0.0635, -0.0033], a-only=35 b-only=55 discordant=90, mid-p=0.0354
  - non-inferiority cheap vs incumbent: diff=-0.0326 one-sided-lo=-0.0570 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=78.02%, cheap acc on sent=68.97%; (on kept: incumbent=93.19%, cheap=93.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0905 CI95=[-0.1552, -0.0259], a-only=19 b-only=40 discordant=59, mid-p=0.0062
- (5) Cascade: acc=87.62%, bal_acc=87.62%, LLM share=37.79%, $/1k items=0.2233 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0342 CI95=[0.0098, 0.0586], a-only=40 b-only=19 discordant=59, mid-p=0.0062; NI diff=0.0342 one-sided-lo=0.0147 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0016 CI95=[-0.0163, 0.0195], a-only=16 b-only=15 discordant=31, mid-p=0.8601; NI diff=0.0016 one-sided-lo=-0.0130 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (6) Oracle gate (route exactly the cheap errors): acc=93.16%, LLM share=15.80% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0354 survives=no, sent: p=0.0062 survives=yes, cascade_vs_cheap: p=0.0062 survives=yes, cascade_vs_incumbent: p=0.8601 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=no, cheaper=yes, cascade_beats_cheap_after_holm=yes
- **per-repeat verdict: incumbent**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=84.31%, bal_acc=84.31%, keep_rate=62.70%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0315 CI95=[-0.0619, -0.0016], a-only=34.6667 b-only=54.0000 discordant=88.6667, mid-p=0.0472
  - non-inferiority cheap vs incumbent: diff=-0.0315 one-sided-lo=-0.0559 margin=0.0100 passes=0.0000 (None)
- (4) SENT subset (kept==0): n_sent=229.0 (37.30%); incumbent acc on sent=77.29%, cheap acc on sent=67.94%; (on kept: incumbent=93.51%, cheap=94.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0935 CI95=[-0.1590, -0.0250], a-only=20.6667 b-only=42.0000 discordant=62.6667, mid-p=0.0143
- (5) Cascade: acc=87.79%, bal_acc=87.79%, LLM share=37.30%, $/1k items=0.2204 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0347 CI95=[0.0098, 0.0603], a-only=42.0000 b-only=20.6667 discordant=62.6667, mid-p=0.0143; NI diff=0.0347 one-sided-lo=0.0136 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0033 CI95=[-0.0130, 0.0195], a-only=14.0000 b-only=12.0000 discordant=26.0000, mid-p=0.7113; NI diff=0.0033 one-sided-lo=-0.0098 margin=0.0100 passes=0.6667 (None)
- (6) Oracle gate (route exactly the cheap errors): acc=93.11%, LLM share=15.69% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.33, sent=0.67, cascade_vs_cheap=0.67, cascade_vs_incumbent=0.00
- verdict inputs (fraction of repeats): cheap_ni_passes=0.00, cascade_ni_passes=0.67, cheaper=1.00, cascade_beats_cheap_after_holm=0.67

## Verdict

- per-repeat verdicts: ['unclear', 'earns', 'incumbent']
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
