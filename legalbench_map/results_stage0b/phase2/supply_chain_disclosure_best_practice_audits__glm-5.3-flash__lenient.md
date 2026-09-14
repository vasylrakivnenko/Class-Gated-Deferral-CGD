# PILOT (n=10) -- not evidence

# Phase 2 paired cascade analysis: supply_chain_disclosure_best_practice_audits x glm-5.3-flash [scoring: lenient]

- scoring: lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

**PILOT (n=10) -- not evidence** -- joined items 10 < n_test 379.

- task: `supply_chain_disclosure_best_practice_audits`  incumbent: `glm-5.3-flash`  cheap candidate: `tfidf_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 10 (n_test expected 379); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=2000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=100.00% Wilson95=[72.25%, 100.00%] (n=10)
- balanced accuracy=100.00%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=724.6 out=110.9 cached=0.0; mean latency=1.46s
- total cost=0.0016 USD; measured $/1k items=0.1641
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.60% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=40.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000
  - non-inferiority cheap vs incumbent: diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=6 (60.00%); incumbent acc on sent=100.00%, cheap acc on sent=100.00%; (on kept: incumbent=100.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000
- (5) Cascade: acc=100.00%, bal_acc=100.00%, LLM share=60.00%, $/1k items=0.0985 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=1.0000 survives=no, sent: p=1.0000 survives=no, cascade_vs_cheap: p=1.0000 survives=no, cascade_vs_incumbent: p=1.0000 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=50.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000
  - non-inferiority cheap vs incumbent: diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=5 (50.00%); incumbent acc on sent=100.00%, cheap acc on sent=100.00%; (on kept: incumbent=100.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000
- (5) Cascade: acc=100.00%, bal_acc=100.00%, LLM share=50.00%, $/1k items=0.0821 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=1.0000 survives=no, sent: p=1.0000 survives=no, cascade_vs_cheap: p=1.0000 survives=no, cascade_vs_incumbent: p=1.0000 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=60.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000
  - non-inferiority cheap vs incumbent: diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=4 (40.00%); incumbent acc on sent=100.00%, cheap acc on sent=100.00%; (on kept: incumbent=100.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000
- (5) Cascade: acc=100.00%, bal_acc=100.00%, LLM share=40.00%, $/1k items=0.0657 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0 b-only=0 discordant=0, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=1.0000 survives=no, sent: p=1.0000 survives=no, cascade_vs_cheap: p=1.0000 survives=no, cascade_vs_incumbent: p=1.0000 survives=no
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=100.00%, bal_acc=100.00%, keep_rate=50.00%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0.0000 b-only=0.0000 discordant=0.0000, mid-p=1.0000
  - non-inferiority cheap vs incumbent: diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=5.0 (50.00%); incumbent acc on sent=100.00%, cheap acc on sent=100.00%; (on kept: incumbent=100.00%, cheap=100.00%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[0.0000, 0.0000], a-only=0.0000 b-only=0.0000 discordant=0.0000, mid-p=1.0000
- (5) Cascade: acc=100.00%, bal_acc=100.00%, LLM share=50.00%, $/1k items=0.0821 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0.0000 b-only=0.0000 discordant=0.0000, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0000 CI95=[0.0000, 0.0000], a-only=0.0000 b-only=0.0000 discordant=0.0000, mid-p=1.0000; NI diff=0.0000 one-sided-lo=0.0000 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=100.00%, LLM share=0.00% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=0.00
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
