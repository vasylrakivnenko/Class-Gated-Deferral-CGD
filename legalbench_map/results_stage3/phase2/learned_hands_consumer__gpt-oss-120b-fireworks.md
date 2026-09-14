# Phase 2 paired cascade analysis: learned_hands_consumer x gpt-oss-120b-fireworks [scoring: exact]

- scoring: exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.

- task: `learned_hands_consumer`  incumbent: `gpt-oss-120b-fireworks`  cheap candidate: `embed_small_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 614 (n_test expected 614); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=71.50% Wilson95=[67.80%, 74.93%] (n=614)
- balanced accuracy=71.50%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=1737.1 out=184.0 cached=1314.8; mean latency=1.44s
- total cost=0.1188 USD; measured $/1k items=0.1935
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.20% by GPT-4 (https://arxiv.org/pdf/2308.11462 (Table 64, page 123))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=84.85%, bal_acc=84.85%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.1336 CI95=[0.0928, 0.1743], a-only=130 b-only=48 discordant=178, mid-p=0.0000
  - non-inferiority cheap vs incumbent: diff=0.1336 one-sided-lo=0.0993 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=61.64%, cheap acc on sent=69.40%; (on kept: incumbent=77.49%, cheap=94.24%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0776 CI95=[0.0000, 0.1552], a-only=54 b-only=36 discordant=90, mid-p=0.0586
- (5) Cascade: acc=81.92%, bal_acc=81.92%, LLM share=37.79%, $/1k items=0.0731 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0293 CI95=[-0.0603, 0.0000], a-only=36 b-only=54 discordant=90, mid-p=0.0586; NI diff=-0.0293 one-sided-lo=-0.0554 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.1042 CI95=[0.0765, 0.1336], a-only=76 b-only=12 discordant=88, mid-p=0.0000; NI diff=0.1042 one-sided-lo=0.0814 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.67%, LLM share=15.15% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0000 survives=yes, sent: p=0.0586 survives=no, cascade_vs_cheap: p=0.0586 survives=no, cascade_vs_incumbent: p=0.0000 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=83.88%, bal_acc=83.88%, keep_rate=63.68%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.1238 CI95=[0.0814, 0.1661], a-only=127 b-only=51 discordant=178, mid-p=0.0000
  - non-inferiority cheap vs incumbent: diff=0.1238 one-sided-lo=0.0879 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=223 (36.32%); incumbent acc on sent=59.64%, cheap acc on sent=65.47%; (on kept: incumbent=78.26%, cheap=94.37%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0583 CI95=[-0.0269, 0.1435], a-only=52 b-only=39 discordant=91, mid-p=0.1750
- (5) Cascade: acc=81.76%, bal_acc=81.76%, LLM share=36.32%, $/1k items=0.0703 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0212 CI95=[-0.0521, 0.0098], a-only=39 b-only=52 discordant=91, mid-p=0.1750; NI diff=-0.0212 one-sided-lo=-0.0472 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.1026 CI95=[0.0749, 0.1319], a-only=75 b-only=12 discordant=87, mid-p=0.0000; NI diff=0.1026 one-sided-lo=0.0782 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.18%, LLM share=16.12% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0000 survives=yes, sent: p=0.1750 survives=no, cascade_vs_cheap: p=0.1750 survives=no, cascade_vs_incumbent: p=0.0000 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=84.20%, bal_acc=84.20%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.1270 CI95=[0.0863, 0.1694], a-only=128 b-only=50 discordant=178, mid-p=0.0000
  - non-inferiority cheap vs incumbent: diff=0.1270 one-sided-lo=0.0928 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=60.78%, cheap acc on sent=68.97%; (on kept: incumbent=78.01%, cheap=93.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0819 CI95=[0.0000, 0.1638], a-only=57 b-only=38 discordant=95, mid-p=0.0519
- (5) Cascade: acc=81.11%, bal_acc=81.11%, LLM share=37.79%, $/1k items=0.0731 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0309 CI95=[-0.0635, 0.0000], a-only=38 b-only=57 discordant=95, mid-p=0.0519; NI diff=-0.0309 one-sided-lo=-0.0570 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0961 CI95=[0.0684, 0.1254], a-only=71 b-only=12 discordant=83, mid-p=0.0000; NI diff=0.0961 one-sided-lo=0.0733 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.35%, LLM share=15.80% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0000 survives=yes, sent: p=0.0519 survives=no, cascade_vs_cheap: p=0.0519 survives=no, cascade_vs_incumbent: p=0.0000 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=84.31%, bal_acc=84.31%, keep_rate=62.70%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.1281 CI95=[0.0868, 0.1699], a-only=128.3333 b-only=49.6667 discordant=178.0000, mid-p=0.0000
  - non-inferiority cheap vs incumbent: diff=0.1281 one-sided-lo=0.0934 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=229.0 (37.30%); incumbent acc on sent=60.69%, cheap acc on sent=67.94%; (on kept: incumbent=77.92%, cheap=94.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0726 CI95=[-0.0090, 0.1542], a-only=54.3333 b-only=37.6667 discordant=92.0000, mid-p=0.0952
- (5) Cascade: acc=81.60%, bal_acc=81.60%, LLM share=37.30%, $/1k items=0.0722 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0271 CI95=[-0.0586, 0.0033], a-only=37.6667 b-only=54.3333 discordant=92.0000, mid-p=0.0952; NI diff=-0.0271 one-sided-lo=-0.0532 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.1010 CI95=[0.0733, 0.1303], a-only=74.0000 b-only=12.0000 discordant=86.0000, mid-p=0.0000; NI diff=0.1010 one-sided-lo=0.0776 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.40%, LLM share=15.69% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=1.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=1.00
- verdict inputs (fraction of repeats): cheap_ni_passes=1.00, cascade_ni_passes=1.00, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

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
