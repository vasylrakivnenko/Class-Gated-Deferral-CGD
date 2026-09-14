# PILOT (n=10) -- not evidence

# Phase 2 paired cascade analysis: opp115_data_retention x kimi-k2.6-fireworks [scoring: lenient]

- scoring: lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

**PILOT (n=10) -- not evidence** -- joined items 10 < n_test 304.

- task: `opp115_data_retention`  incumbent: `kimi-k2.6-fireworks`  cheap candidate: `embed_base_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 10 (n_test expected 304); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=2000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=60.00% Wilson95=[31.27%, 83.18%] (n=10)
- balanced accuracy=66.67%
- parse-fail rate=10.00% (1 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=474.8 out=1179.4 cached=36.6; mean latency=13.62s
- total cost=0.0514 USD; measured $/1k items=5.1397
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=70.50% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=70.00%, bal_acc=70.83%, keep_rate=30.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.1000 CI95=[-0.2000, 0.4000], a-only=2 b-only=1 discordant=3, mid-p=0.6250
  - non-inferiority cheap vs incumbent: diff=0.1000 one-sided-lo=-0.2000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=7 (70.00%); incumbent acc on sent=71.43%, cheap acc on sent=71.43%; (on kept: incumbent=33.33%, cheap=66.67%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[-0.4286, 0.4286], a-only=1 b-only=1 discordant=2, mid-p=0.5000
- (5) Cascade: acc=70.00%, bal_acc=75.00%, LLM share=70.00%, $/1k items=3.5978 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[-0.3000, 0.3000], a-only=1 b-only=1 discordant=2, mid-p=0.5000; NI diff=0.0000 one-sided-lo=-0.2000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.1000 CI95=[0.0000, 0.3000], a-only=1 b-only=0 discordant=1, mid-p=0.5000; NI diff=0.1000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=80.00%, LLM share=30.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.6250 survives=no, sent: p=0.5000 survives=no, cascade_vs_cheap: p=0.5000 survives=no, cascade_vs_incumbent: p=0.5000 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=80.00%, bal_acc=79.17%, keep_rate=20.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.2000 CI95=[-0.2000, 0.6000], a-only=3 b-only=1 discordant=4, mid-p=0.3750
  - non-inferiority cheap vs incumbent: diff=0.2000 one-sided-lo=-0.1000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=8 (80.00%); incumbent acc on sent=62.50%, cheap acc on sent=87.50%; (on kept: incumbent=50.00%, cheap=50.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.2500 CI95=[-0.2500, 0.6250], a-only=3 b-only=1 discordant=4, mid-p=0.3750
- (5) Cascade: acc=60.00%, bal_acc=66.67%, LLM share=80.00%, $/1k items=4.1118 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.2000 CI95=[-0.6000, 0.2000], a-only=1 b-only=3 discordant=4, mid-p=0.3750; NI diff=-0.2000 one-sided-lo=-0.5000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.00%, LLM share=20.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.3750 survives=no, sent: p=0.3750 survives=no, cascade_vs_cheap: p=0.3750 survives=no, cascade_vs_incumbent: p=1.0000 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=80.00%, bal_acc=79.17%, keep_rate=30.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.2000 CI95=[-0.2000, 0.6000], a-only=3 b-only=1 discordant=4, mid-p=0.3750
  - non-inferiority cheap vs incumbent: diff=0.2000 one-sided-lo=-0.1000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=7 (70.00%); incumbent acc on sent=57.14%, cheap acc on sent=85.71%; (on kept: incumbent=66.67%, cheap=66.67%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.2857 CI95=[-0.2857, 0.7143], a-only=3 b-only=1 discordant=4, mid-p=0.3750
- (5) Cascade: acc=60.00%, bal_acc=66.67%, LLM share=70.00%, $/1k items=3.5978 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.2000 CI95=[-0.6000, 0.2000], a-only=1 b-only=3 discordant=4, mid-p=0.3750; NI diff=-0.2000 one-sided-lo=-0.5000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.00%, LLM share=20.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.3750 survives=no, sent: p=0.3750 survives=no, cascade_vs_cheap: p=0.3750 survives=no, cascade_vs_incumbent: p=1.0000 survives=no
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=76.67%, bal_acc=76.39%, keep_rate=26.67%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.1667 CI95=[-0.2000, 0.5333], a-only=2.6667 b-only=1.0000 discordant=3.6667, mid-p=0.4583
  - non-inferiority cheap vs incumbent: diff=0.1667 one-sided-lo=-0.1333 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=7.3 (73.33%); incumbent acc on sent=63.69%, cheap acc on sent=81.55%; (on kept: incumbent=50.00%, cheap=61.11%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.1786 CI95=[-0.3214, 0.5893], a-only=2.3333 b-only=1.0000 discordant=3.3333, mid-p=0.4167
- (5) Cascade: acc=63.33%, bal_acc=69.44%, LLM share=73.33%, $/1k items=3.7691 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.1333 CI95=[-0.5000, 0.2333], a-only=1.0000 b-only=2.3333 discordant=3.3333, mid-p=0.4167; NI diff=-0.1333 one-sided-lo=-0.4000 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0333 CI95=[0.0000, 0.1000], a-only=0.3333 b-only=0.0000 discordant=0.3333, mid-p=0.8333; NI diff=0.0333 one-sided-lo=0.0000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=86.67%, LLM share=23.33% -- the ceiling for ANY gate on this cheap model
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
