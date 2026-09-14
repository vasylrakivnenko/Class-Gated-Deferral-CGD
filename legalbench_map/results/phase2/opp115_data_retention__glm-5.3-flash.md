# PILOT (n=10) -- not evidence

# Phase 2 paired cascade analysis: opp115_data_retention x glm-5.3-flash

**PILOT (n=10) -- not evidence** -- joined items 10 < n_test 304.

- task: `opp115_data_retention`  incumbent: `glm-5.3-flash`  cheap candidate: `embed_base_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 10 (n_test expected 304); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=0.00% Wilson95=[0.00%, 27.75%] (n=10)
- balanced accuracy=0.00%
- parse-fail rate=100.00% (10 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=469.8 out=558.5 cached=0.0; mean latency=5.95s
- total cost=0.0035 USD; measured $/1k items=0.3497
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=70.50% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=70.00%, bal_acc=70.83%, keep_rate=30.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.7000 CI95=[0.4000, 1.0000], a-only=7 b-only=0 discordant=7, mid-p=0.0078
  - non-inferiority cheap vs incumbent: diff=0.7000 one-sided-lo=0.5000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=7 (70.00%); incumbent acc on sent=0.00%, cheap acc on sent=71.43%; (on kept: incumbent=0.00%, cheap=66.67%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.7143 CI95=[0.4286, 1.0000], a-only=5 b-only=0 discordant=5, mid-p=0.0312
- (5) Cascade: acc=20.00%, bal_acc=16.67%, LLM share=70.00%, $/1k items=0.2448 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.5000 CI95=[-0.8000, -0.2000], a-only=0 b-only=5 discordant=5, mid-p=0.0313; NI diff=-0.5000 one-sided-lo=-0.8000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.2000 CI95=[0.0000, 0.5000], a-only=2 b-only=0 discordant=2, mid-p=0.2500; NI diff=0.2000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=70.00%, LLM share=30.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0078 survives=yes, sent: p=0.0312 survives=no, cascade_vs_cheap: p=0.0313 survives=no, cascade_vs_incumbent: p=0.2500 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=80.00%, bal_acc=79.17%, keep_rate=20.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.8000 CI95=[0.5000, 1.0000], a-only=8 b-only=0 discordant=8, mid-p=0.0039
  - non-inferiority cheap vs incumbent: diff=0.8000 one-sided-lo=0.6000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=8 (80.00%); incumbent acc on sent=0.00%, cheap acc on sent=87.50%; (on kept: incumbent=0.00%, cheap=50.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.8750 CI95=[0.6250, 1.0000], a-only=7 b-only=0 discordant=7, mid-p=0.0078
- (5) Cascade: acc=10.00%, bal_acc=8.33%, LLM share=80.00%, $/1k items=0.2798 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.7000 CI95=[-1.0000, -0.4000], a-only=0 b-only=7 discordant=7, mid-p=0.0078; NI diff=-0.7000 one-sided-lo=-0.9000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.1000 CI95=[0.0000, 0.3000], a-only=1 b-only=0 discordant=1, mid-p=0.5000; NI diff=0.1000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=80.00%, LLM share=20.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0039 survives=yes, sent: p=0.0078 survives=yes, cascade_vs_cheap: p=0.0078 survives=yes, cascade_vs_incumbent: p=0.5000 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=80.00%, bal_acc=79.17%, keep_rate=30.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.8000 CI95=[0.5000, 1.0000], a-only=8 b-only=0 discordant=8, mid-p=0.0039
  - non-inferiority cheap vs incumbent: diff=0.8000 one-sided-lo=0.6000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=7 (70.00%); incumbent acc on sent=0.00%, cheap acc on sent=85.71%; (on kept: incumbent=0.00%, cheap=66.67%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.8571 CI95=[0.5714, 1.0000], a-only=6 b-only=0 discordant=6, mid-p=0.0156
- (5) Cascade: acc=20.00%, bal_acc=16.67%, LLM share=70.00%, $/1k items=0.2448 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.6000 CI95=[-0.9000, -0.3000], a-only=0 b-only=6 discordant=6, mid-p=0.0156; NI diff=-0.6000 one-sided-lo=-0.8000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.2000 CI95=[0.0000, 0.5000], a-only=2 b-only=0 discordant=2, mid-p=0.2500; NI diff=0.2000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=80.00%, LLM share=20.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0039 survives=yes, sent: p=0.0156 survives=yes, cascade_vs_cheap: p=0.0156 survives=yes, cascade_vs_incumbent: p=0.2500 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=76.67%, bal_acc=76.39%, keep_rate=26.67%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.7667 CI95=[0.4667, 1.0000], a-only=7.6667 b-only=0.0000 discordant=7.6667, mid-p=0.0052
  - non-inferiority cheap vs incumbent: diff=0.7667 one-sided-lo=0.5667 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=7.3 (73.33%); incumbent acc on sent=0.00%, cheap acc on sent=81.55%; (on kept: incumbent=0.00%, cheap=61.11%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.8155 CI95=[0.5417, 1.0000], a-only=6.0000 b-only=0.0000 discordant=6.0000, mid-p=0.0182
- (5) Cascade: acc=16.67%, bal_acc=13.89%, LLM share=73.33%, $/1k items=0.2565 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.6000 CI95=[-0.9000, -0.3000], a-only=0.0000 b-only=6.0000 discordant=6.0000, mid-p=0.0182; NI diff=-0.6000 one-sided-lo=-0.8333 margin=0.0100 passes=0.0000 (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.1667 CI95=[0.0000, 0.4333], a-only=1.6667 b-only=0.0000 discordant=1.6667, mid-p=0.3333; NI diff=0.1667 one-sided-lo=0.0000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=76.67%, LLM share=23.33% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=1.00, sent=0.67, cascade_vs_cheap=0.67, cascade_vs_incumbent=0.00
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
