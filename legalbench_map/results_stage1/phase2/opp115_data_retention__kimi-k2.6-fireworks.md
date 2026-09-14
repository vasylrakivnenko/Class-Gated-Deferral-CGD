# Phase 2 paired cascade analysis: opp115_data_retention x kimi-k2.6-fireworks [scoring: exact]

- scoring: exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.

- task: `opp115_data_retention`  incumbent: `kimi-k2.6-fireworks`  cheap candidate: `embed_base_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 304 (n_test expected 304); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=75.66% Wilson95=[70.53%, 80.14%] (n=304)
- balanced accuracy=75.66%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=466.5 out=637.4 cached=337.5; mean latency=8.00s
- total cost=0.8287 USD; measured $/1k items=2.7260
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=70.50% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=78.29%, bal_acc=78.29%, keep_rate=54.93%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0263 CI95=[-0.0330, 0.0855], a-only=48 b-only=40 discordant=88, mid-p=0.3966
  - non-inferiority cheap vs incumbent: diff=0.0263 one-sided-lo=-0.0263 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=137 (45.07%); incumbent acc on sent=69.34%, cheap acc on sent=62.77%; (on kept: incumbent=80.84%, cheap=91.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0657 CI95=[-0.1679, 0.0438], a-only=23 b-only=32 discordant=55, mid-p=0.2288
- (5) Cascade: acc=81.25%, bal_acc=81.25%, LLM share=45.07%, $/1k items=1.2285 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0296 CI95=[-0.0164, 0.0789], a-only=32 b-only=23 discordant=55, mid-p=0.2288; NI diff=0.0296 one-sided-lo=-0.0099 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0559 CI95=[0.0197, 0.0921], a-only=25 b-only=8 discordant=33, mid-p=0.0029; NI diff=0.0559 one-sided-lo=0.0263 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.45%, LLM share=21.71% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.3966 survives=no, sent: p=0.2288 survives=no, cascade_vs_cheap: p=0.2288 survives=no, cascade_vs_incumbent: p=0.0029 survives=yes
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=80.92%, bal_acc=80.92%, keep_rate=40.46%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0526 CI95=[-0.0066, 0.1118], a-only=50 b-only=34 discordant=84, mid-p=0.0821
  - non-inferiority cheap vs incumbent: diff=0.0526 one-sided-lo=0.0033 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=181 (59.54%); incumbent acc on sent=71.27%, cheap acc on sent=71.82%; (on kept: incumbent=82.11%, cheap=94.31%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0055 CI95=[-0.0829, 0.0884], a-only=32 b-only=31 discordant=63, mid-p=0.9007
- (5) Cascade: acc=80.59%, bal_acc=80.59%, LLM share=59.54%, $/1k items=1.6231 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0033 CI95=[-0.0526, 0.0461], a-only=31 b-only=32 discordant=63, mid-p=0.9007; NI diff=-0.0033 one-sided-lo=-0.0461 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0493 CI95=[0.0230, 0.0789], a-only=18 b-only=3 discordant=21, mid-p=0.0009; NI diff=0.0493 one-sided-lo=0.0263 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.11%, LLM share=19.08% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0821 survives=no, sent: p=0.9007 survives=no, cascade_vs_cheap: p=0.9007 survives=no, cascade_vs_incumbent: p=0.0009 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=79.93%, bal_acc=79.93%, keep_rate=45.07%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0428 CI95=[-0.0164, 0.1020], a-only=49 b-only=36 discordant=85, mid-p=0.1606
  - non-inferiority cheap vs incumbent: diff=0.0428 one-sided-lo=-0.0066 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=167 (54.93%); incumbent acc on sent=69.46%, cheap acc on sent=67.66%; (on kept: incumbent=83.21%, cheap=94.89%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0180 CI95=[-0.1078, 0.0778], a-only=30 b-only=33 discordant=63, mid-p=0.7080
- (5) Cascade: acc=80.92%, bal_acc=80.92%, LLM share=54.93%, $/1k items=1.4975 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0099 CI95=[-0.0428, 0.0625], a-only=33 b-only=30 discordant=63, mid-p=0.7080; NI diff=0.0099 one-sided-lo=-0.0329 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0526 CI95=[0.0230, 0.0822], a-only=19 b-only=3 discordant=22, mid-p=0.0005; NI diff=0.0526 one-sided-lo=0.0296 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.78%, LLM share=20.07% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.1606 survives=no, sent: p=0.7080 survives=no, cascade_vs_cheap: p=0.7080 survives=no, cascade_vs_incumbent: p=0.0005 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=79.71%, bal_acc=79.71%, keep_rate=46.82%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0406 CI95=[-0.0187, 0.0998], a-only=49.0000 b-only=36.6667 discordant=85.6667, mid-p=0.2131
  - non-inferiority cheap vs incumbent: diff=0.0406 one-sided-lo=-0.0099 margin=0.0100 passes=0.6667 (None)
- (4) SENT subset (kept==0): n_sent=161.7 (53.18%); incumbent acc on sent=70.02%, cheap acc on sent=67.42%; (on kept: incumbent=82.05%, cheap=93.41%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0260 CI95=[-0.1195, 0.0700], a-only=28.3333 b-only=32.0000 discordant=60.3333, mid-p=0.6125
- (5) Cascade: acc=80.92%, bal_acc=80.92%, LLM share=53.18%, $/1k items=1.4497 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0121 CI95=[-0.0373, 0.0625], a-only=32.0000 b-only=28.3333 discordant=60.3333, mid-p=0.6125; NI diff=0.0121 one-sided-lo=-0.0296 margin=0.0100 passes=0.3333 (None)
  - (5b) vs incumbent-alone: McNemar diff=0.0526 CI95=[0.0219, 0.0844], a-only=20.6667 b-only=4.6667 discordant=25.3333, mid-p=0.0014; NI diff=0.0526 one-sided-lo=0.0274 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.78%, LLM share=20.29% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=1.00
- verdict inputs (fraction of repeats): cheap_ni_passes=0.67, cascade_ni_passes=1.00, cheaper=1.00, cascade_beats_cheap_after_holm=0.00

## Verdict

- per-repeat verdicts: ['unclear', 'cheap-alone', 'cheap-alone']
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
