# Phase 2 paired cascade analysis: opp115_data_retention x glm-5.3-flash [scoring: exact]

- scoring: exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.

- task: `opp115_data_retention`  incumbent: `glm-5.3-flash`  cheap candidate: `embed_base_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 304 (n_test expected 304); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=78.62% Wilson95=[73.67%, 82.86%] (n=304)
- balanced accuracy=78.62%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=471.8 out=266.9 cached=0.0; mean latency=2.92s
- total cost=0.0621 USD; measured $/1k items=0.2042
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=70.50% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=78.29%, bal_acc=78.29%, keep_rate=54.93%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=-0.0033 CI95=[-0.0625, 0.0526], a-only=39 b-only=40 discordant=79, mid-p=0.9111
  - non-inferiority cheap vs incumbent: diff=-0.0033 one-sided-lo=-0.0526 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=137 (45.07%); incumbent acc on sent=73.72%, cheap acc on sent=62.77%; (on kept: incumbent=82.63%, cheap=91.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.1095 CI95=[-0.2044, -0.0073], a-only=17 b-only=32 discordant=49, mid-p=0.0328
- (5) Cascade: acc=83.22%, bal_acc=83.22%, LLM share=45.07%, $/1k items=0.0920 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0493 CI95=[0.0065, 0.0954], a-only=32 b-only=17 discordant=49, mid-p=0.0328; NI diff=0.0493 one-sided-lo=0.0132 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0461 CI95=[0.0099, 0.0822], a-only=22 b-only=8 discordant=30, mid-p=0.0107; NI diff=0.0461 one-sided-lo=0.0164 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.45%, LLM share=21.71% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.9111 survives=no, sent: p=0.0328 survives=no, cascade_vs_cheap: p=0.0328 survives=no, cascade_vs_incumbent: p=0.0107 survives=yes
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=80.92%, bal_acc=80.92%, keep_rate=40.46%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0230 CI95=[-0.0329, 0.0789], a-only=41 b-only=34 discordant=75, mid-p=0.4222
  - non-inferiority cheap vs incumbent: diff=0.0230 one-sided-lo=-0.0230 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=181 (59.54%); incumbent acc on sent=75.14%, cheap acc on sent=71.82%; (on kept: incumbent=83.74%, cheap=94.31%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0331 CI95=[-0.1160, 0.0442], a-only=25 b-only=31 discordant=56, mid-p=0.4270
- (5) Cascade: acc=82.89%, bal_acc=82.89%, LLM share=59.54%, $/1k items=0.1216 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0197 CI95=[-0.0296, 0.0691], a-only=31 b-only=25 discordant=56, mid-p=0.4270; NI diff=0.0197 one-sided-lo=-0.0197 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0428 CI95=[0.0164, 0.0724], a-only=16 b-only=3 discordant=19, mid-p=0.0026; NI diff=0.0428 one-sided-lo=0.0197 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=92.11%, LLM share=19.08% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.4222 survives=no, sent: p=0.4270 survives=no, cascade_vs_cheap: p=0.4270 survives=no, cascade_vs_incumbent: p=0.0026 survives=yes
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=79.93%, bal_acc=79.93%, keep_rate=45.07%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0132 CI95=[-0.0428, 0.0691], a-only=40 b-only=36 discordant=76, mid-p=0.6488
  - non-inferiority cheap vs incumbent: diff=0.0132 one-sided-lo=-0.0329 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=167 (54.93%); incumbent acc on sent=73.05%, cheap acc on sent=67.66%; (on kept: incumbent=85.40%, cheap=94.89%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0539 CI95=[-0.1437, 0.0359], a-only=24 b-only=33 discordant=57, mid-p=0.2370
- (5) Cascade: acc=82.89%, bal_acc=82.89%, LLM share=54.93%, $/1k items=0.1122 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0296 CI95=[-0.0197, 0.0789], a-only=33 b-only=24 discordant=57, mid-p=0.2370; NI diff=0.0296 one-sided-lo=-0.0132 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0428 CI95=[0.0164, 0.0724], a-only=16 b-only=3 discordant=19, mid-p=0.0026; NI diff=0.0428 one-sided-lo=0.0197 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.78%, LLM share=20.07% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.6488 survives=no, sent: p=0.2370 survives=no, cascade_vs_cheap: p=0.2370 survives=no, cascade_vs_incumbent: p=0.0026 survives=yes
- verdict inputs: cheap_ni_passes=no, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: unclear**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=79.71%, bal_acc=79.71%, keep_rate=46.82%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0110 CI95=[-0.0461, 0.0669], a-only=40.0000 b-only=36.6667 discordant=76.6667, mid-p=0.6607
  - non-inferiority cheap vs incumbent: diff=0.0110 one-sided-lo=-0.0362 margin=0.0100 passes=0.0000 (inconclusive -- test set too small to rule out a 1% loss)
- (4) SENT subset (kept==0): n_sent=161.7 (53.18%); incumbent acc on sent=73.97%, cheap acc on sent=67.42%; (on kept: incumbent=83.93%, cheap=93.41%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0655 CI95=[-0.1547, 0.0243], a-only=22.0000 b-only=32.0000 discordant=54.0000, mid-p=0.2323
- (5) Cascade: acc=83.00%, bal_acc=83.00%, LLM share=53.18%, $/1k items=0.1086 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0329 CI95=[-0.0143, 0.0811], a-only=32.0000 b-only=22.0000 discordant=54.0000, mid-p=0.2323; NI diff=0.0329 one-sided-lo=-0.0066 margin=0.0100 passes=0.3333 (None)
  - (5b) vs incumbent-alone: McNemar diff=0.0439 CI95=[0.0143, 0.0757], a-only=18.0000 b-only=4.6667 discordant=22.6667, mid-p=0.0053; NI diff=0.0439 one-sided-lo=0.0186 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.78%, LLM share=20.29% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.00, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=1.00
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
