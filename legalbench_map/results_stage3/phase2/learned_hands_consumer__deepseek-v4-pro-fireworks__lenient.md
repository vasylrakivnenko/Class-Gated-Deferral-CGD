# Phase 2 paired cascade analysis: learned_hands_consumer x deepseek-v4-pro-fireworks [scoring: lenient]

- scoring: lenient: the earliest whole-word label occurrence in the normalized output (ties -> longest label); no label anywhere -> unparseable, counts as wrong. Reported alongside exact, never instead of it. Measures whether the model KNOWS the answer rather than whether it obeys a 2023 completion-style format; the cascade decision uses this scoring.

- task: `learned_hands_consumer`  incumbent: `deepseek-v4-pro-fireworks`  cheap candidate: `embed_small_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 614 (n_test expected 614); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=81.92% Wilson95=[78.68%, 84.76%] (n=614)
- balanced accuracy=81.92%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=1687.3 out=194.2 cached=0.0; mean latency=4.11s
- total cost=1.8398 USD; measured $/1k items=2.9964
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=76.20% by GPT-4 (https://arxiv.org/pdf/2308.11462 (Table 64, page 123))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=84.85%, bal_acc=84.85%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0293 CI95=[-0.0049, 0.0651], a-only=69 b-only=51 discordant=120, mid-p=0.1014
  - non-inferiority cheap vs incumbent: diff=0.0293 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=70.69%, cheap acc on sent=69.40%; (on kept: incumbent=88.74%, cheap=94.24%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0129 CI95=[-0.0862, 0.0603], a-only=36 b-only=39 discordant=75, mid-p=0.7310
- (5) Cascade: acc=85.34%, bal_acc=85.34%, LLM share=37.79%, $/1k items=1.1322 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0049 CI95=[-0.0228, 0.0326], a-only=39 b-only=36 discordant=75, mid-p=0.7310; NI diff=0.0049 one-sided-lo=-0.0179 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0342 CI95=[0.0130, 0.0554], a-only=33 b-only=12 discordant=45, mid-p=0.0016; NI diff=0.0342 one-sided-lo=0.0163 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=93.16%, LLM share=15.15% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.1014 survives=no, sent: p=0.7310 survives=no, cascade_vs_cheap: p=0.7310 survives=no, cascade_vs_incumbent: p=0.0016 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=83.88%, bal_acc=83.88%, keep_rate=63.68%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0195 CI95=[-0.0147, 0.0554], a-only=68 b-only=56 discordant=124, mid-p=0.2831
  - non-inferiority cheap vs incumbent: diff=0.0195 one-sided-lo=-0.0098 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=223 (36.32%); incumbent acc on sent=68.16%, cheap acc on sent=65.47%; (on kept: incumbent=89.77%, cheap=94.37%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0269 CI95=[-0.1076, 0.0538], a-only=37 b-only=43 discordant=80, mid-p=0.5052
- (5) Cascade: acc=84.85%, bal_acc=84.85%, LLM share=36.32%, $/1k items=1.0883 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0098 CI95=[-0.0195, 0.0391], a-only=43 b-only=37 discordant=80, mid-p=0.5052; NI diff=0.0098 one-sided-lo=-0.0147 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0293 CI95=[0.0098, 0.0505], a-only=31 b-only=13 discordant=44, mid-p=0.0066; NI diff=0.0293 one-sided-lo=0.0130 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=93.00%, LLM share=16.12% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.2831 survives=no, sent: p=0.5052 survives=no, cascade_vs_cheap: p=0.5052 survives=no, cascade_vs_incumbent: p=0.0066 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=84.20%, bal_acc=84.20%, keep_rate=62.21%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0228 CI95=[-0.0130, 0.0586], a-only=67 b-only=53 discordant=120, mid-p=0.2029
  - non-inferiority cheap vs incumbent: diff=0.0228 one-sided-lo=-0.0065 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=232 (37.79%); incumbent acc on sent=70.69%, cheap acc on sent=68.97%; (on kept: incumbent=88.74%, cheap=93.46%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0172 CI95=[-0.0863, 0.0517], a-only=33 b-only=37 discordant=70, mid-p=0.6353
- (5) Cascade: acc=84.85%, bal_acc=84.85%, LLM share=37.79%, $/1k items=1.1322 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0065 CI95=[-0.0212, 0.0326], a-only=37 b-only=33 discordant=70, mid-p=0.6353; NI diff=0.0065 one-sided-lo=-0.0163 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0293 CI95=[0.0065, 0.0521], a-only=34 b-only=16 discordant=50, mid-p=0.0110; NI diff=0.0293 one-sided-lo=0.0098 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.83%, LLM share=15.80% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.2029 survives=no, sent: p=0.6353 survives=no, cascade_vs_cheap: p=0.6353 survives=no, cascade_vs_incumbent: p=0.0110 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=84.31%, bal_acc=84.31%, keep_rate=62.70%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0239 CI95=[-0.0109, 0.0597], a-only=68.0000 b-only=53.3333 discordant=121.3333, mid-p=0.1958
  - non-inferiority cheap vs incumbent: diff=0.0239 one-sided-lo=-0.0054 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=229.0 (37.30%); incumbent acc on sent=69.85%, cheap acc on sent=67.94%; (on kept: incumbent=89.09%, cheap=94.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0190 CI95=[-0.0934, 0.0553], a-only=35.3333 b-only=39.6667 discordant=75.0000, mid-p=0.6239
- (5) Cascade: acc=85.02%, bal_acc=85.02%, LLM share=37.30%, $/1k items=1.1176 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0071 CI95=[-0.0212, 0.0347], a-only=39.6667 b-only=35.3333 discordant=75.0000, mid-p=0.6239; NI diff=0.0071 one-sided-lo=-0.0163 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0309 CI95=[0.0098, 0.0527], a-only=32.6667 b-only=13.6667 discordant=46.3333, mid-p=0.0064; NI diff=0.0309 one-sided-lo=0.0130 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=93.00%, LLM share=15.69% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=1.00
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
