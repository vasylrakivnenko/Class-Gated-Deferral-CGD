# Phase 2 paired cascade analysis: opp115_data_retention x gpt-oss-120b-fireworks [scoring: exact]

- scoring: exact: LegalBench's verbatim exact-match rule -- the whole normalized output must equal a label. Benchmark-faithful; a correct answer wrapped in any other words counts as wrong.

- task: `opp115_data_retention`  incumbent: `gpt-oss-120b-fireworks`  cheap candidate: `embed_base_logreg` (best_candidate from configs)
- items joined on (task, item_idx): 304 (n_test expected 304); text_sha256 and gold asserted equal on every joined row
- cheap repeats: [0, 1, 2]; delta=0.01; Holm alpha=0.05; bootstrap n_boot=10000, seed=0
- ASYMMETRY (chosen, not hidden): the incumbent is a SINGLE temperature-0 pass; the cheap side has one out-of-fold prediction per item per CV repeat (normally 3). Every paired statistic pairs the cheap model's repeat-r OOF prediction with the same single incumbent prediction, and is reported per repeat and as the mean across repeats.

## (1) Incumbent alone [measured, this run]

- accuracy=73.03% Wilson95=[67.77%, 77.71%] (n=304)
- balanced accuracy=73.03%
- parse-fail rate=0.00% (0 items; unparseable output counts as wrong, never coerced)
- API-error rows=0 (runner '<error: ...>' rows; must be 0 for the numbers above to mean anything)
- mean tokens: in=530.3 out=211.5 cached=360.7; mean latency=1.66s
- total cost=0.0480 USD; measured $/1k items=0.1578
- published-2023 best for this task (contrast only, not this run): balanced_accuracy=70.50% by GPT-3.5 (https://arxiv.org/pdf/2308.11462 (Table 76, page 132))

## Per-repeat paired analysis (cheap repeat r vs the single incumbent pass)

### repeat 0

- (2) Cheap alone [CV-estimated]: acc=78.29%, bal_acc=78.29%, keep_rate=54.93%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0526 CI95=[-0.0099, 0.1151], a-only=54 b-only=38 discordant=92, mid-p=0.0966
  - non-inferiority cheap vs incumbent: diff=0.0526 one-sided-lo=0.0000 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=137 (45.07%); incumbent acc on sent=69.34%, cheap acc on sent=62.77%; (on kept: incumbent=76.05%, cheap=91.02%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0657 CI95=[-0.1679, 0.0438], a-only=23 b-only=32 discordant=55, mid-p=0.2288
- (5) Cascade: acc=81.25%, bal_acc=81.25%, LLM share=45.07%, $/1k items=0.0711 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0296 CI95=[-0.0164, 0.0757], a-only=32 b-only=23 discordant=55, mid-p=0.2288; NI diff=0.0296 one-sided-lo=-0.0099 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
  - (5b) vs incumbent-alone: McNemar diff=0.0822 CI95=[0.0461, 0.1217], a-only=31 b-only=6 discordant=37, mid-p=0.0000; NI diff=0.0822 one-sided-lo=0.0493 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=90.79%, LLM share=21.71% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0966 survives=no, sent: p=0.2288 survives=no, cascade_vs_cheap: p=0.2288 survives=no, cascade_vs_incumbent: p=0.0000 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 1

- (2) Cheap alone [CV-estimated]: acc=80.92%, bal_acc=80.92%, keep_rate=40.46%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0789 CI95=[0.0197, 0.1382], a-only=57 b-only=33 discordant=90, mid-p=0.0115
  - non-inferiority cheap vs incumbent: diff=0.0789 one-sided-lo=0.0296 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=181 (59.54%); incumbent acc on sent=70.17%, cheap acc on sent=71.82%; (on kept: incumbent=77.24%, cheap=94.31%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0166 CI95=[-0.0664, 0.1050], a-only=33 b-only=30 discordant=63, mid-p=0.7080
- (5) Cascade: acc=79.93%, bal_acc=79.93%, LLM share=59.54%, $/1k items=0.0939 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=-0.0099 CI95=[-0.0592, 0.0428], a-only=30 b-only=33 discordant=63, mid-p=0.7080; NI diff=-0.0099 one-sided-lo=-0.0526 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0691 CI95=[0.0362, 0.1020], a-only=24 b-only=3 discordant=27, mid-p=0.0000; NI diff=0.0691 one-sided-lo=0.0428 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.78%, LLM share=19.08% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0115 survives=yes, sent: p=0.7080 survives=no, cascade_vs_cheap: p=0.7080 survives=no, cascade_vs_incumbent: p=0.0000 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### repeat 2

- (2) Cheap alone [CV-estimated]: acc=79.93%, bal_acc=79.93%, keep_rate=45.07%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0691 CI95=[0.0066, 0.1316], a-only=56 b-only=35 discordant=91, mid-p=0.0280
  - non-inferiority cheap vs incumbent: diff=0.0691 one-sided-lo=0.0164 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=167 (54.93%); incumbent acc on sent=67.66%, cheap acc on sent=67.66%; (on kept: incumbent=79.56%, cheap=94.89%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=0.0000 CI95=[-0.0958, 0.0958], a-only=32 b-only=32 discordant=64, mid-p=0.9007
- (5) Cascade: acc=79.93%, bal_acc=79.93%, LLM share=54.93%, $/1k items=0.0867 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0000 CI95=[-0.0526, 0.0526], a-only=32 b-only=32 discordant=64, mid-p=0.9007; NI diff=0.0000 one-sided-lo=-0.0428 margin=0.0100 passes=no (inconclusive -- test set too small to rule out a 1% loss)
  - (5b) vs incumbent-alone: McNemar diff=0.0691 CI95=[0.0362, 0.1020], a-only=24 b-only=3 discordant=27, mid-p=0.0000; NI diff=0.0691 one-sided-lo=0.0428 margin=0.0100 passes=yes (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.45%, LLM share=20.07% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni (alpha=0.05, family=4): paired_all: p=0.0280 survives=no, sent: p=0.9007 survives=no, cascade_vs_cheap: p=0.9007 survives=no, cascade_vs_incumbent: p=0.0000 survives=yes
- verdict inputs: cheap_ni_passes=yes, cascade_ni_passes=yes, cheaper=yes, cascade_beats_cheap_after_holm=no
- **per-repeat verdict: cheap-alone**

### MEAN across repeats

- (2) Cheap alone [CV-estimated]: acc=79.71%, bal_acc=79.71%, keep_rate=46.82%
- (3) Paired, all items -- McNemar cheap (a) vs incumbent (b): diff=0.0669 CI95=[0.0055, 0.1283], a-only=55.6667 b-only=35.3333 discordant=91.0000, mid-p=0.0453
  - non-inferiority cheap vs incumbent: diff=0.0669 one-sided-lo=0.0154 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (4) SENT subset (kept==0): n_sent=161.7 (53.18%); incumbent acc on sent=69.06%, cheap acc on sent=67.42%; (on kept: incumbent=77.62%, cheap=93.41%)
  - McNemar on sent, cheap (a) vs incumbent (b): diff=-0.0164 CI95=[-0.1100, 0.0815], a-only=29.3333 b-only=31.3333 discordant=60.6667, mid-p=0.6125
- (5) Cascade: acc=80.37%, bal_acc=80.37%, LLM share=53.18%, $/1k items=0.0839 (measured incumbent $/1k x share)
  - (5a) vs cheap-alone: McNemar diff=0.0066 CI95=[-0.0428, 0.0570], a-only=31.3333 b-only=29.3333 discordant=60.6667, mid-p=0.6125; NI diff=0.0066 one-sided-lo=-0.0351 margin=0.0100 passes=0.3333 (None)
  - (5b) vs incumbent-alone: McNemar diff=0.0735 CI95=[0.0395, 0.1086], a-only=26.3333 b-only=4.0000 discordant=30.3333, mid-p=0.0000; NI diff=0.0735 one-sided-lo=0.0450 margin=0.0100 passes=1.0000 (non-inferior (and not worse on the point estimate))
- (6) Oracle gate (route exactly the cheap errors): acc=91.34%, LLM share=20.29% -- the ceiling for ANY gate on this cheap model
- (7) Holm-Bonferroni survival (fraction of repeats): paired_all=0.33, sent=0.00, cascade_vs_cheap=0.00, cascade_vs_incumbent=1.00
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
