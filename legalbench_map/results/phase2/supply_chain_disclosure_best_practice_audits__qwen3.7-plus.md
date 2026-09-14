# INVALID: 10 of 10 incumbent rows are API errors ('<error: ...>' from the runner), not model answers -- re-run phase2/run_incumbents.py (it retries error rows on resume)

# PILOT (n=10) -- not evidence

# Phase 2 paired cascade analysis: supply_chain_disclosure_best_practice_audits x qwen3.7-plus

**INVALID: 10 of 10 incumbent rows are API errors ('<error: ...>' from the runner), not model answers -- re-run phase2/run_incumbents.py (it retries error rows on resume)** -- every incumbent number below is meaningless.

**PILOT (n=10) -- not evidence** -- joined items 10 < n_test 379.

- task: `supply_chain_disclosure_best_practice_audits`  incumbent: `qwen3.7-plus`  cheap candidate: `tfidf_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 10 (n_test expected 379); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=0.00% Wilson95=[0.00%, 27.75%] (n=10)
- balanced accuracy=0.00%
- parse-fail rate=100.00% (10 items; unparseable output counts as wrong, never coerced)
- API-error rows=10 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=0.0 out=0.0 cached=0.0; mean latency=0.14s
- total cost=0.0000 USD; measured $/1k items=0.0000
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.60% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=40.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=10 b-only=0 discordant=10, mid-p=0.0010
  - non-inferiority cheap vs incumbent: diff=1.0000 one-sided-lo=1.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=6 (60.00%); incumbent acc on sent=0.00%, cheap acc on sent=100.00%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=6 b-only=0 discordant=6, mid-p=0.0156
- (5) Cascade: acc=40.00%, bal_acc=40.00%, LLM share=60.00%, $/1k items=0.0000 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.6000 CI95=[-0.9000, -0.3000], a-only=0 b-only=6 discordant=6, mid-p=0.0156; NI diff=-0.6000 one-sided-lo=-0.8000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.4000 CI95=[0.1000, 0.7000], a-only=4 b-only=0 discordant=4, mid-p=0.0625; NI diff=0.4000 one-sided-lo=0.2000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0010 survives=yes, sent: p=0.0156 survives=yes, cascade_vs_cheap: p=0.0156 survives=yes, cascade_vs_incumbent: p=0.0625 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=50.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=10 b-only=0 discordant=10, mid-p=0.0010
  - non-inferiority cheap vs incumbent: diff=1.0000 one-sided-lo=1.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=5 (50.00%); incumbent acc on sent=0.00%, cheap acc on sent=100.00%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=5 b-only=0 discordant=5, mid-p=0.0312
- (5) Cascade: acc=50.00%, bal_acc=50.00%, LLM share=50.00%, $/1k items=0.0000 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.5000 CI95=[-0.8000, -0.2000], a-only=0 b-only=5 discordant=5, mid-p=0.0313; NI diff=-0.5000 one-sided-lo=-0.8000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.5000 CI95=[0.2000, 0.8000], a-only=5 b-only=0 discordant=5, mid-p=0.0312; NI diff=0.5000 one-sided-lo=0.2000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0010 survives=yes, sent: p=0.0312 survives=no, cascade_vs_cheap: p=0.0313 survives=no, cascade_vs_incumbent: p=0.0312 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=60.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=10 b-only=0 discordant=10, mid-p=0.0010
  - non-inferiority cheap vs incumbent: diff=1.0000 one-sided-lo=1.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=4 (40.00%); incumbent acc on sent=0.00%, cheap acc on sent=100.00%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=4 b-only=0 discordant=4, mid-p=0.0625
- (5) Cascade: acc=60.00%, bal_acc=60.00%, LLM share=40.00%, $/1k items=0.0000 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.4000 CI95=[-0.7000, -0.1000], a-only=0 b-only=4 discordant=4, mid-p=0.0625; NI diff=-0.4000 one-sided-lo=-0.7000 margin=0.0100 passes=no (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.6000 CI95=[0.3000, 0.9000], a-only=6 b-only=0 discordant=6, mid-p=0.0156; NI diff=0.6000 one-sided-lo=0.3000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0010 survives=yes, sent: p=0.0625 survives=no, cascade_vs_cheap: p=0.0625 survives=no, cascade_vs_incumbent: p=0.0156 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=50.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=10.0000 b-only=0.0000 discordant=10.0000, mid-p=0.0010
  - non-inferiority cheap vs incumbent: diff=1.0000 one-sided-lo=1.0000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=5.0 (50.00%); incumbent acc on sent=0.00%, cheap acc on sent=100.00%; (on kept: incumbent=0.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=1.0000 CI95=[1.0000, 1.0000], a-only=5.0000 b-only=0.0000 discordant=5.0000, mid-p=0.0365
- (5) Cascade: acc=50.00%, bal_acc=50.00%, LLM share=50.00%, $/1k items=0.0000 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.5000 CI95=[-0.8000, -0.2000], a-only=0.0000 b-only=5.0000 discordant=5.0000, mid-p=0.0365; NI diff=-0.5000 one-sided-lo=-0.7667 margin=0.0100 passes=0.0000 (clearly worse by more than the 1% margin)
  - (5b) vs incumbent-alone: McNemar diff=0.5000 CI95=[0.2000, 0.8000], a-only=5.0000 b-only=0.0000 discordant=5.0000, mid-p=0.0365; NI diff=0.5000 one-sided-lo=0.2333 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=1.00, sent=0.33, cascade_vs_cheap=0.33, cascade_vs_incumbent=0.33
- verdict inputs (fraction of repeats): cheap_ni_passes=1.00, cascade_ni_passes=1.00, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

## Verdict

- per-repeat verdicts: ['cheap-alone', 'cheap-alone', 'cheap-alone']
- **(task, model) verdict: invalid-api-errors**

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
