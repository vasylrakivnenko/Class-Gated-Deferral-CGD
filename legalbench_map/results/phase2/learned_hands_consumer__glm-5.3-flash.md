# PILOT (n=10) -- not evidence

# Phase 2 paired cascade analysis: learned_hands_consumer x glm-5.3-flash

**PILOT (n=10) -- not evidence** -- joined items 10 < n_test 614.

- task: `learned_hands_consumer`  incumbent: `glm-5.3-flash`  cheap candidate: `embed_small_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 10 (n_test expected 614); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=0.00% Wilson95=[0.00%, 27.75%] (n=10)
- balanced accuracy=0.00%
- parse-fail rate=100.00% (10 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=1598.9 out=537.6 cached=0.0; mean latency=6.71s
- total cost=0.0051 USD; measured $/1k items=0.5086
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.20% by GPT-4 (https://arxiv.org/pdf/2308.11462 (Table 64, page 123))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=90.00%, bal_acc=90.00%, keep_rate=60.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.9000 CI95=[0.7000, 1.0000], a-only=9 b-only=0 discordant=9, mid-p=0.0020
  - non-inferiority cheap vs incumbent: diff=0.9000 one-sided-lo=0.7000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=4 (40.00%); incumbent acc on sent=0.00%, cheap acc on sent=75.00%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.7500 CI95=[0.2500, 1.0000], a-only=3 b-only=0 discordant=3, mid-p=0.1250
- (5) Cascade: acc=60.00%, bal_acc=60.00%, LLM share=40.00%, $/1k items=0.2035 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.3000 CI95=[-0.6000, 0.0000], a-only=0 b-only=3 discordant=3, mid-p=0.1250; NI diff=-0.3000 one-sided-lo=-0.5000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.6000 CI95=[0.3000, 0.9000], a-only=6 b-only=0 discordant=6, mid-p=0.0156; NI diff=0.6000 one-sided-lo=0.3000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.00%, LLM share=10.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0020 survives=yes, sent: p=0.1250 survives=no, cascade_vs_cheap: p=0.1250 survives=no, cascade_vs_incumbent: p=0.0156 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=90.00%, bal_acc=90.00%, keep_rate=80.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.9000 CI95=[0.7000, 1.0000], a-only=9 b-only=0 discordant=9, mid-p=0.0020
  - non-inferiority cheap vs incumbent: diff=0.9000 one-sided-lo=0.7000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=2 (20.00%); incumbent acc on sent=0.00%, cheap acc on sent=50.00%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.5000 CI95=[0.0000, 1.0000], a-only=1 b-only=0 discordant=1, mid-p=0.5000
- (5) Cascade: acc=80.00%, bal_acc=80.00%, LLM share=20.00%, $/1k items=0.1017 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.1000 CI95=[-0.3000, 0.0000], a-only=0 b-only=1 discordant=1, mid-p=0.5000; NI diff=-0.1000 one-sided-lo=-0.3000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.8000 CI95=[0.5000, 1.0000], a-only=8 b-only=0 discordant=8, mid-p=0.0039; NI diff=0.8000 one-sided-lo=0.6000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.00%, LLM share=10.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0020 survives=yes, sent: p=0.5000 survives=no, cascade_vs_cheap: p=0.5000 survives=no, cascade_vs_incumbent: p=0.0039 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=90.00%, bal_acc=90.00%, keep_rate=70.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.9000 CI95=[0.7000, 1.0000], a-only=9 b-only=0 discordant=9, mid-p=0.0020
  - non-inferiority cheap vs incumbent: diff=0.9000 one-sided-lo=0.7000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=3 (30.00%); incumbent acc on sent=0.00%, cheap acc on sent=66.67%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.6667 CI95=[0.0000, 1.0000], a-only=2 b-only=0 discordant=2, mid-p=0.2500
- (5) Cascade: acc=70.00%, bal_acc=70.00%, LLM share=30.00%, $/1k items=0.1526 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.2000 CI95=[-0.5000, 0.0000], a-only=0 b-only=2 discordant=2, mid-p=0.2500; NI diff=-0.2000 one-sided-lo=-0.4000 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.7000 CI95=[0.4000, 1.0000], a-only=7 b-only=0 discordant=7, mid-p=0.0078; NI diff=0.7000 one-sided-lo=0.5000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.00%, LLM share=10.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0020 survives=yes, sent: p=0.2500 survives=no, cascade_vs_cheap: p=0.2500 survives=no, cascade_vs_incumbent: p=0.0078 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=90.00%, bal_acc=90.00%, keep_rate=70.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.9000 CI95=[0.7000, 1.0000], a-only=9.0000 b-only=0.0000 discordant=9.0000, mid-p=0.0020
  - non-inferiority cheap vs incumbent: diff=0.9000 one-sided-lo=0.7000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=3.0 (30.00%); incumbent acc on sent=0.00%, cheap acc on sent=63.89%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.6389 CI95=[0.0833, 1.0000], a-only=2.0000 b-only=0.0000 discordant=2.0000, mid-p=0.2917
- (5) Cascade: acc=70.00%, bal_acc=70.00%, LLM share=30.00%, $/1k items=0.1526 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.2000 CI95=[-0.4667, 0.0000], a-only=0.0000 b-only=2.0000 discordant=2.0000, mid-p=0.2917; NI diff=-0.2000 one-sided-lo=-0.4000 margin=0.0100 passes=0.0000 (None)
  - (5b) vs incumbent-alone: McNemar diff=0.7000 CI95=[0.4000, 0.9667], a-only=7.0000 b-only=0.0000 discordant=7.0000, mid-p=0.0091; NI diff=0.7000 one-sided-lo=0.4667 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.00%, LLM share=10.00% -- the ceiling for ANY gate on this cheap model
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
