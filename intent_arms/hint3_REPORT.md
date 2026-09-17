# HINT3 free arm — report

**Dataset:** HINT3 v1 *full* train splits — Arora et al., *"HINT3: Raising the bar for Intent Detection in the Wild"*, EMNLP 2020 Insights Workshop, arXiv:2009.13833. Three real production chatbot intent datasets from three Indian companies (sofmattress, curekart, powerplay11).
**Repo:** `https://github.com/hellohaptik/HINT3` (master).
**Paid API calls made: 0.** Every number below comes from local sklearn/torch or from the authors' own committed prediction files. Zero spend.
**SEED = 0** everywhere. Device: `mps`.

---

## 1. Headline — we won two bots and lost one, and the loss is the important part

**Mixed verdict, and one of the three bots is a loss that also disqualifies its own headline metric.**

| bot | shipped free arm, overall acc @ train-selected t* | all-`NO_NODES_DETECTED` constant | verdict |
|---|---|---|---|
| sofmattress | **0.6801** | 0.4181 | **WIN**, +26.20 pts over the constant |
| curekart | **0.7023** | 0.5439 | **WIN**, +15.84 pts over the constant |
| powerplay11 | **0.5748** | 0.7202 | **LOSS, −14.55 pts — the label-blind constant beats us** |

### The single headline number: **−14.55 points**

On powerplay11 the shipped free arm, at the operating threshold chosen on training folds, scores **57.48% overall accuracy against a label-blind constant's 72.02%**. It loses to a predictor that identifies zero intents.

This is not fixable by threshold choice. Sweeping the confidence cutoff over the whole range, the *best accuracy the free arm can reach anywhere on powerplay11* is 0.7548 at t=0.09 — a mere **+3.46 points** over the constant, and it buys that by answering only **12.1% of traffic** and getting in-scope accuracy of 0.2545. At the threshold that maximises macro-F1 (t=0.06) accuracy is 0.6907, still **2.95 points below the constant**.

**Per the project's constant-predictor guard, overall accuracy on powerplay11 is hereby flagged as unusable, not merely caveated.** So is out-of-scope recall on all three bots (the constant scores a perfect 1.0000 by definition).

### The competitive result on the metrics a constant cannot win

Macro-F1 over in-scope intents is 0.0000 for both constants on all three bots, as are in-scope accuracy and MCC. On that footing, against the five published systems (paired, same items — see §5):

| bot | our macro-F1 | best published | gap | our rank |
|---|---|---|---|---|
| sofmattress | 0.6480 | 0.6713 (Dialogflow) | **−0.0233** | 3rd of 6 |
| curekart | 0.6462 | 0.6499 (BERT) | **−0.0037** | 2nd of 6 |
| powerplay11 | 0.4480 | 0.4874 (Haptik) | **−0.0394** | 2nd of 6 |

**We win 0 of 3 bots outright** — an honest, narrow loss. A frozen 118M-parameter embedding model plus logistic regression, trained on 328–600 labels, lands within 0.4–3.9 macro-F1 points of the best of four commercial NLU platforms and BERT. Against published BERT alone it is 1 win / 2 narrow losses (both inside 0.005) on macro-F1. **It does not beat the field on any bot.**

### Cascade gate: no useful middle

The gate **FAILS on all three bots** for the row we would actually ship, under the definition that matches what we would ship. Details in §6.

### The constraint that limits what this arm can conclude

**There is no LLM arm in this measurement.** The hard rule was zero paid API calls, so HINT3 carries no paid-model number. The five published baselines are 2020-era NLU platforms and BERT — *not* LLMs. **This arm therefore cannot confirm or refute the project's free-vs-paid thesis on its own.** See §9.

---

## 2. Protocol, and why it is open-set rather than plain classification

`NO_NODES_DETECTED` appears **only in test, never in train** (verified: 0/328, 0/600, 0/471 train rows). It marks a real user query the production bot had no intent for. So: train on in-scope intents only; at test time answer with `argmax` only if `max P >= t`, else emit `NO_NODES_DETECTED`.

| bot | train rows | intents | median labels/intent | test rows | test in-scope | test OOS | OOS share |
|---|---|---|---|---|---|---|---|
| sofmattress | 328 | 21 | 12 (min 9) | 397 | 231 | 166 | 41.81% |
| curekart | 600 | 28 | 14 (min 3) | 991 | 452 | 539 | 54.39% |
| powerplay11 | 471 | 59 | 7 (min 1) | 983 | 275 | 708 | 72.02% |

All shapes match the brief exactly. Every in-scope intent appearing in test also appears in train (0 unseen intents on all three bots). Going the other way, 1 / 7 / 1 train intents never appear in test.

### Metric stack validated against the authors' own code

Before measuring anything, the metric implementation was checked by **reproducing the authors' 15 committed result files** (`results/{platform}_{bot}.csv` — 3 bots × 5 platforms × 9 thresholds × 5 metrics = **675 published numbers**) from their own committed per-item predictions.

**Max absolute deviation over all 675 numbers: 2.220446e-16** — floating-point noise.

This also pinned down an otherwise-invisible convention: the comparison is `score >= t`. With `score > t` the max deviation is 4.5e-2. Every metric in this report is computed by the code that reproduces the authors' numbers exactly, which is what makes the comparison in §5 trustworthy.

---

## 3. Constant-predictor guard (project rule)

| bot | all-OOS const acc | majority-intent const acc | const in-scope acc | const OOS recall | const macro-F1 | const MCC | intents identified |
|---|---|---|---|---|---|---|---|
| sofmattress | **0.4181** | 0.0176 | 0.0000 | **1.0000** | 0.0000 | 0.0000 | 0 |
| curekart | **0.5439** | 0.2109 | 0.0000 | **1.0000** | 0.0000 | 0.0000 | 0 |
| powerplay11 | **0.7202** | 0.0509 | 0.0000 | **1.0000** | 0.0000 | 0.0000 | 0 |

**Metrics dropped/flagged as constant-dominated:**
- **Overall accuracy on powerplay11 — DROPPED.** The constant wins it outright against our shipped row, and against 4 of the 5 published platforms at most thresholds.
- **Out-of-scope recall on all three bots — FLAGGED.** Trivially 1.0000 for the constant; only meaningful read jointly with in-scope accuracy.
- **Overall accuracy on sofmattress / curekart — FLAGGED but usable.** The constant scores 41.81% / 54.39%; our margins (+26.2 / +15.8) are far outside that.

**Trustworthy metrics: in-scope accuracy, macro-F1 over in-scope intents, MCC** (all exactly 0 for any constant predictor).

### The guard also indicts the published baselines

This is a finding about HINT3 itself, not about our arm. On powerplay11, against the 0.7202 constant, using each platform's own best threshold from the authors' own files:

| platform | best published acc | vs constant | thresholds below the constant |
|---|---|---|---|
| Haptik | 0.7416 | +0.0214 | 6 of 9 |
| LUIS | 0.7314 | +0.0112 | 7 of 9 |
| RASA | 0.7314 | +0.0112 | 8 of 9 |
| Dialogflow | 0.7284 | +0.0081 | 7 of 9 |
| **BERT** | **0.6653** | **−0.0549** | **9 of 9** |

Published BERT is **5.49 points below a label-blind constant at every threshold the authors report**, and the four commercial platforms clear it by 0.8–2.1 points. On powerplay11, accuracy is close to worthless as a discriminator — for us and for the published field alike. **That is the headline caveat of this arm, not a footnote.**

---

## 4. The four free arms

| arm | configuration |
|---|---|
| 1. majority / constant | most frequent train intent; plus the all-OOS constant of §3 |
| 2. TF-IDF + logreg | word 1-2 grams `sublinear_tf=True` ∪ `char_wb` 3-5 grams; `LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=0)` |
| 3. frozen embeddings + logreg | `intfloat/multilingual-e5-small`, `"query: "` prefix, `normalize_embeddings=True`; `LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced", random_state=0)` |
| 4. fine-tuned encoder | `jhu-clsp/ettin-encoder-68m`, truncate 256, batch 16, lr 5e-5, AdamW wd 0.01, OneCycleLR `pct_start=0.1` linear anneal, class-weighted CE, grad clip 1.0, `mps` |

### At t=0 (argmax always answers) — no rejection

| bot | arm | in-scope acc | macro-F1 | MCC | overall acc | CV macro-F1 (train OOF) |
|---|---|---|---|---|---|---|
| sofmattress | majority | 0.0303 | 0.0017 | 0.0000 | 0.0176 | 0.0089 |
| sofmattress | tfidf_lr | 0.7359 | 0.5734 | 0.4625 | 0.4282 | 0.8644 |
| sofmattress | **frozen_e5_lr** ← shipped | **0.7792** | **0.5852** | **0.4847** | 0.4534 | 0.8731 |
| sofmattress | ettin_encoder_3ep | 0.4848 | 0.3233 | 0.2898 | 0.2821 | 0.5676 |
| sofmattress | ettin_encoder_20ep | 0.5714 | 0.3958 | 0.3485 | 0.3325 | 0.7158 |
| curekart | majority | 0.4624 | 0.0166 | 0.0000 | 0.2109 | 0.0098 |
| curekart | tfidf_lr | 0.8186 | **0.5170** | 0.3771 | 0.3734 | 0.8549 |
| curekart | **frozen_e5_lr** ← shipped | **0.8540** | 0.5033 | **0.4023** | 0.3895 | 0.8549 |
| curekart | ettin_encoder_3ep | 0.5199 | 0.2930 | 0.2394 | 0.2371 | 0.5751 |
| curekart | ettin_encoder_10ep | 0.7611 | 0.4089 | 0.3405 | 0.3471 | 0.7332 |
| powerplay11 | majority | 0.1818 | 0.0017 | 0.0000 | 0.0509 | 0.0030 |
| powerplay11 | tfidf_lr | **0.6145** | **0.3361** | **0.2355** | 0.1719 | 0.7226 |
| powerplay11 | **frozen_e5_lr** ← shipped | 0.5855 | 0.2874 | 0.2312 | 0.1638 | 0.7385 |
| powerplay11 | ettin_encoder_3ep | 0.2109 | 0.1187 | 0.0798 | 0.0590 | 0.4276 |
| powerplay11 | ettin_encoder_20ep | 0.3491 | 0.1488 | 0.1298 | 0.0977 | 0.5765 |

Three things to note honestly:

1. **The fine-tuned encoder is the worst non-constant arm on every bot, by a wide margin.** The spec's 3 epochs is 63/114/90 optimizer steps total; CV-selecting 10–20 epochs (on train folds only) recovers 8–14 in-scope accuracy points but still leaves it 20–24 points behind the frozen embeddings. 328–600 labels is not enough to fine-tune a 68M encoder with a freshly initialised head. This is an expected label-budget result, reported as such.
2. **Train-only arm selection cost us on powerplay11.** The shipped row is chosen by highest train out-of-fold CV macro-F1. On powerplay11 that picked `frozen_e5_lr` (CV 0.7385) over `tfidf_lr` (CV 0.7226), but on test `tfidf_lr` is actually better (in-scope 0.6145 vs 0.5855; macro-F1 0.3361 vs 0.2874). **The train-only rule picked the worse row by 2.9 in-scope points.** We are keeping the train-selected row, because picking on test is the error this rule exists to prevent — but the cost is stated.
3. **The train→test generalisation gap is enormous:** CV macro-F1 0.86–0.87 versus test macro-F1 0.50–0.59 on sofmattress/curekart. This is precisely the effect the HINT3 authors built the benchmark to expose ("all systems latch on to unintended patterns in training data"). Train-fold CV numbers on HINT3 are not predictive of test performance in absolute terms, only useful for ranking.

### Encoder epoch selection (train CV only, no test)

| bot | CV macro-F1 @3ep | @10ep | @20ep | selected |
|---|---|---|---|---|
| sofmattress | 0.5676 | 0.7050 | 0.7158 | 20 |
| curekart | 0.5751 | 0.7332 | 0.7326 | 10 |
| powerplay11 | 0.4276 | 0.5660 | 0.5765 | 20 |

I changed epochs from the specified 3 and am flagging it, as instructed. Both the 3-epoch and CV-selected rows are reported everywhere.

### Threshold selection — chosen on training folds only

**A methodological point that must be stated: the rejection threshold cannot be tuned on train in any way that reflects the accuracy/OOS-recall trade-off, because train contains zero out-of-scope rows.** With no OOS examples in any CV fold, every rejection is a pure loss on train, so a "maximise fold accuracy" objective is monotone in lowering t and selects the degenerate t=0. This is a property of the HINT3 protocol.

Resolution used: **t\* = the 10th-percentile of out-of-fold in-scope confidence** — "reject the least-confident 10% of in-scope traffic," a target in-scope rejection rate. This is standard selective prediction and **requires no OOS labels, so it never touches test.** (IRR=20% and 30% are also in `hint3.json`.)

Sanity check on the rule: on sofmattress the train-selected t\*=0.1261 yields test macro-F1 0.6485, while the true test-optimal over a 0.0005 grid is 0.6495 at t=0.1165 — **the train-only rule lands within 0.001 of optimal.** On curekart it costs 0.0315 macro-F1 versus test-optimal, on powerplay11 0.0407.

### At the train-selected operating point (IRR=10%)

| bot | arm | t\* | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | overall acc | beats all-OOS const? |
|---|---|---|---|---|---|---|---|---|
| sofmattress | tfidf_lr | 0.2103 | 0.6407 | 0.6928 | 0.6203 | 0.5887 | 0.6625 | yes |
| sofmattress | **frozen_e5_lr** ← shipped | 0.1261 | 0.7013 | 0.6506 | **0.6485** | 0.6184 | 0.6801 | yes (+0.2620) |
| sofmattress | ettin_encoder_20ep | 0.5465 | 0.5368 | 0.4217 | 0.4455 | 0.4131 | 0.4887 | yes |
| sofmattress | ettin_encoder_3ep | 0.3128 | 0.4675 | 0.1988 | 0.3383 | 0.3039 | 0.3552 | **NO (−0.0630)** |
| curekart | tfidf_lr | 0.1834 | 0.7721 | 0.6234 | 0.5670 | 0.5937 | 0.6912 | yes |
| curekart | **frozen_e5_lr** ← shipped | 0.1170 | 0.7854 | 0.6327 | **0.6147** | 0.6004 | 0.7023 | yes (+0.1584) |
| curekart | ettin_encoder_10ep | 0.5472 | 0.7367 | 0.3135 | 0.4800 | 0.4198 | 0.5066 | **NO (−0.0373)** |
| curekart | ettin_encoder_3ep | 0.3338 | 0.4867 | 0.3952 | 0.3582 | 0.3065 | 0.4369 | **NO (−0.1070)** |
| powerplay11 | tfidf_lr | 0.1589 | 0.5273 | 0.6215 | 0.4291 | 0.3677 | 0.5951 | **NO (−0.1251)** |
| powerplay11 | **frozen_e5_lr** ← shipped | 0.0505 | 0.5236 | 0.5946 | 0.4073 | 0.3626 | 0.5748 | **NO (−0.1455)** |
| powerplay11 | ettin_encoder_20ep | 0.4213 | 0.3018 | 0.4040 | 0.1764 | 0.1647 | 0.3754 | **NO (−0.3449)** |
| powerplay11 | ettin_encoder_3ep | 0.2046 | 0.1927 | 0.2090 | 0.1312 | 0.0798 | 0.2045 | **NO (−0.5158)** |

⚑ = constant-dominated metric (constant scores 1.0000).

---

## 5. Threshold sweep — shipped row

`t*` in §4 was chosen **on training folds only**. The sweeps below are the full curves, reported instead of a hand-picked point, as the paper does.

**sofmattress** — `frozen_e5_lr`; constant acc 0.4181

| t | overall acc | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | answered % |
|---|---|---|---|---|---|---|
| 0.1 | 0.5995 | 0.7619 | 0.3735 | 0.6295 | 0.5757 | 81.9 |
| 0.2 | 0.6398 | 0.4372 | 0.9217 | 0.5027 | 0.5269 | 32.7 |
| 0.3 | 0.5189 | 0.1732 | 1.0000 | 0.2683 | 0.3522 | 10.8 |
| 0.4 | 0.4484 | 0.0519 | 1.0000 | 0.1102 | 0.1892 | 3.3 |
| 0.5 | 0.4257 | 0.0130 | 1.0000 | 0.0560 | 0.0967 | 0.8 |
| 0.6–0.9 | 0.4181 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |

**curekart** — `frozen_e5_lr`; constant acc 0.5439

| t | overall acc | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | answered % |
|---|---|---|---|---|---|---|
| 0.1 | 0.5913 | 0.8296 | 0.3915 | 0.5686 | 0.5276 | 77.0 |
| 0.2 | 0.7366 | 0.5022 | 0.9332 | 0.5792 | 0.5751 | 27.5 |
| 0.3 | 0.6519 | 0.2456 | 0.9926 | 0.3830 | 0.4344 | 11.8 |
| 0.4 | 0.5964 | 0.1195 | 0.9963 | 0.2298 | 0.3047 | 5.7 |
| 0.5 | 0.5762 | 0.0708 | 1.0000 | 0.1743 | 0.2415 | 3.2 |
| 0.6 | 0.5540 | 0.0221 | 1.0000 | 0.0631 | 0.1344 | 1.0 |
| 0.7 | 0.5449 | 0.0022 | 1.0000 | 0.0159 | 0.0428 | 0.1 |
| 0.8–0.9 | 0.5439 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |

**powerplay11** — `frozen_e5_lr`; constant acc 0.7202. **Every row's overall-accuracy column is constant-dominated or barely above; read in-scope accuracy and macro-F1 instead.**

| t | overall acc | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | answered % |
|---|---|---|---|---|---|---|
| 0.1 | 0.7497 | 0.2000 | 0.9633 | 0.3025 | 0.3277 | 9.0 |
| 0.2 | 0.7233 | 0.0109 | 1.0000 | 0.0184 | 0.0972 | 0.3 |
| 0.3 | 0.7223 | 0.0073 | 1.0000 | 0.0115 | 0.0793 | 0.2 |
| 0.4–0.9 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |

**Caveat on the paper's 0.1–0.9 grid.** Our arms' confidences are compressed low (frozen-e5 mean max-prob 0.220 on in-scope sofmattress items, versus published BERT's 0.930), so the entire informative region for our arms lies **below t=0.2** and the paper grid collapses them to the constant by t=0.4. A 0.01-resolution sweep (101 points) is stored in `hint3.json` under `sweep_fine` for every arm and every platform. **Comparing different methods "at the same t" is meaningless on HINT3** — confidence scales differ by design. §6's comparison therefore gives each method its own best threshold, symmetrically.

---

## 6. Published baselines and the paired comparison

The paper evaluated four NLU platforms plus a BERT classifier — Dialogflow, LUIS, RASA, Haptik, BERT — on exactly these splits. I obtained **more than the brief anticipated**:

- `results/{platform}_{bot}.csv` — the authors' own threshold sweeps (accuracy, weighted-F1, in-scope accuracy, OOS recall, MCC at t=0.1…0.9) for **all five** systems.
- `preds/{platform}_{bot}.csv` — **per-item** `sentence,label,predicted_node,predicted_node_score` for **all five** systems, not just BERT.

### The alignment succeeded, so this is a paired same-items comparison

Stated explicitly, since the brief asked me to skip rather than guess if alignment were uncertain:

- Our `hint3/{bot}_{train,test}.csv` are **byte-identical** to the repo's `dataset/v1/{train,test}/` (`pandas.DataFrame.equals` → `True`, all six files).
- For **all 5 platforms × 3 bots**, `preds/` matches the test CSV **row-for-row in order** on both `sentence` and `label` (397/991/983 rows; 0 NaN predictions).

So the items, their order, and the gold labels are identical, and per-item McNemar tests are legitimate. This is **stronger than published-on-authors'-protocol evidence** — it is a paired same-items run — and it cost nothing. (The authors' aggregate numbers, reproduced to 2.22e-16 in §2, remain published-on-authors'-protocol for the platforms' *training* side: their models were trained by the authors on their own infrastructure, which I did not re-run.)

### Each method at its own test-optimal macro-F1 threshold

**Optimistic for every row, applied symmetrically.** Our threshold here is tuned on test and is *not* the shipped t\* of §4.

**sofmattress**

| method | t | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | overall acc |
|---|---|---|---|---|---|---|
| Dialogflow (published) | 0.46 | 0.7013 | 0.6867 | **0.6713** | **0.6353** | 0.6952 |
| BERT (published) | 0.85 | 0.6840 | 0.7108 | 0.6521 | 0.6331 | 0.6952 |
| **ours: frozen_e5_lr** | 0.12 | **0.7100** | 0.6084 | 0.6480 | 0.6097 | 0.6675 |
| Haptik (published) | 0.29 | 0.6667 | 0.6687 | 0.6268 | 0.5987 | 0.6675 |
| RASA (published) | 0.68 | 0.5801 | 0.6265 | 0.5937 | 0.5118 | 0.5995 |
| LUIS (published) | 0.10 | 0.5931 | 0.7530 | 0.5933 | 0.5807 | 0.6599 |
| all-OOS constant | — | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4181 |

**curekart**

| method | t | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | overall acc |
|---|---|---|---|---|---|---|
| BERT (published) | 0.99 | 0.7412 | 0.7514 | **0.6499** | **0.6213** | 0.7467 |
| **ours: frozen_e5_lr** | 0.15 | 0.6881 | 0.7996 | 0.6462 | 0.6160 | **0.7508**† |
| RASA (published) | 0.68 | **0.7965** | 0.4100 | 0.6056 | 0.5008 | 0.5863 |
| Dialogflow (published) | 0.58 | 0.6062 | 0.8479 | 0.5871 | 0.5930 | 0.7376 |
| Haptik (published) | 0.72 | 0.5265 | 0.9295 | 0.5640 | 0.5918 | 0.7457 |
| LUIS (published) | 0.32 | 0.6372 | 0.6475 | 0.5377 | 0.5014 | 0.6428 |
| all-OOS constant | — | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5439 |

† our best accuracy anywhere on the sweep (t=0.17); at the macro-F1-optimal t=0.15 it is 0.7487, still the best of the six.

**powerplay11**

| method | t | in-scope acc | OOS recall ⚑ | macro-F1 | MCC | overall acc |
|---|---|---|---|---|---|---|
| Haptik (published) | 0.31 | **0.6036** | 0.6285 | **0.4874** | **0.4230** | 0.6216 |
| **ours: frozen_e5_lr** | 0.06 | 0.4727 | 0.7754 | 0.4480 | 0.4202 | 0.6907 |
| BERT (published) | 0.62 | 0.5236 | 0.5268 | 0.4443 | 0.3182 | 0.5259 |
| LUIS (published) | 0.27 | 0.3855 | 0.7867 | 0.3978 | 0.3660 | 0.6745 |
| Dialogflow (published) | 0.56 | 0.5127 | 0.5946 | 0.3670 | 0.3463 | 0.5717 |
| RASA (published) | 0.61 | 0.3927 | 0.7429 | 0.3442 | 0.3259 | 0.6450 |
| **all-OOS constant** | — | 0.0000 | 1.0000 | 0.0000 | 0.0000 | **0.7202** ← beats all six |

### McNemar, overall correctness, paired same items

| bot | vs | ours right / theirs wrong | theirs right / ours wrong | exact p | reading |
|---|---|---|---|---|---|
| sofmattress | Dialogflow | 47 | 58 | 0.329 | tie |
| sofmattress | LUIS | 65 | 62 | 0.859 | tie |
| sofmattress | RASA | 76 | 49 | **0.0197** | we win |
| sofmattress | BERT | 42 | 53 | 0.305 | tie |
| sofmattress | Haptik | 50 | 50 | 1.000 | tie |
| curekart | Dialogflow | 125 | 114 | 0.518 | tie |
| curekart | LUIS | 222 | 117 | **1.25e-08** | we win |
| curekart | RASA | 253 | 92 | **1.64e-18** | we win |
| curekart | BERT | 120 | 118 | 0.948 | tie |
| curekart | Haptik | 132 | 129 | 0.902 | tie |
| powerplay11 | Dialogflow | 210 | 93 | **1.5e-11** | we win |
| powerplay11 | LUIS | 131 | 115 | 0.339 | tie |
| powerplay11 | RASA | 154 | 109 | **0.00655** | we win |
| powerplay11 | BERT | 239 | 77 | **1.84e-20** | we win |
| powerplay11 | Haptik | 167 | 99 | **3.64e-05** | we win |

**We never lose a paired test.** 7 significant wins, 8 ties, 0 losses. But note this table is on overall accuracy, which is the constant-dominated metric on powerplay11 — five of the seven wins are on bots/metrics where the constant also performs well, so the McNemar table should be read alongside the macro-F1 ranking above, where we place 3rd/2nd/2nd and win nothing outright.

---

## 7. Cascade pre-flight gate — **FAILS on all three bots**

Computed on the row we would actually ship (`frozen_e5_lr`, train-CV-selected), at the train-selected t\*. Two correctness definitions are reported because on this task shape they disagree, and reporting only one would hide a choice:

- **A — shipped pipeline:** `correct = (thresholded prediction == gold)` over all test items. This is the output we would actually ship, OOS rejections included. **This is the definition that matters for routing.**
- **B — answering subset:** `correct = (argmax == gold)` over gold-in-scope items only. "When the bot does answer, does confidence track correctness." This is the definition comparable to the project's reference rows.

Gates: AUROC ≥ ~0.75, and error share in the least-confident 20% well above the 20% that random selection returns. **Calibration gap deliberately not used as a gate.**

| bot | def | AUROC(conf→correct) | ≥0.75? | err share in least-conf 20% | ≫20%? | **verdict** |
|---|---|---|---|---|---|---|
| sofmattress | **A shipped** | **0.5106** | FAIL | **0.0787** | FAIL | **FAIL** |
| sofmattress | B answering | 0.6781 | FAIL | 0.3725 | pass | **FAIL** |
| curekart | **A shipped** | **0.4507** | FAIL | **0.0508** | FAIL | **FAIL** |
| curekart | B answering | 0.7678 | pass | 0.4091 | pass | pass |
| powerplay11 | **A shipped** | **0.2405** | FAIL | **0.0335** | FAIL | **FAIL** |
| powerplay11 | B answering | 0.7384 | FAIL | 0.3509 | pass | **FAIL** |

Reference rows that pass elsewhere in this project: banking77 0.905 / 80%, ToxicChat 0.902 / 84%, CUAD 0.797 / 49%.

### **Verdict: no useful middle.**

Under definition A — the shipped pipeline — **all three bots fail both gates, badly.** AUROC 0.5106 is coin-flip; 0.4507 and **0.2405 are actively inverted** (low confidence *predicts being correct*). Error shares of 3–8% in the least-confident 20% are *far below* the 20% random selection returns: the least-confident quintile is where the pipeline is most nearly *right*, not most wrong. Routing the low-confidence tail to an expensive model would spend money on the items the free pipeline already handles correctly.

**Why, mechanically:** on an OOS-heavy set, the threshold has already spent the low-confidence region on the OOS decision. A very-low-confidence item gets predicted `NO_NODES_DETECTED` and is therefore *correct* exactly when the gold is OOS — which is 72.0% of powerplay11, 54.4% of curekart, 41.8% of sofmattress. So confidence→correctness is **non-monotone**, and AUROC, which measures monotone ranking, collapses. A low AUROC under definition A is a statement about the task shape, not about the confidence signal being uninformative.

Two supporting measurements that separate the two properties:

- **Confidence is a good *scope* detector.** AUROC(conf → item is in-scope) on sofmattress: ours 0.8520 (frozen-e5) / 0.8609 (TF-IDF), versus published Dialogflow 0.8690, BERT 0.8680, Haptik 0.8387, RASA 0.7925, LUIS 0.7973. We are on par with the best published systems at deciding *whether to answer*.
- **Confidence is a poor *correctness* detector.** Definition-B AUROC is only 0.6781 / 0.7678 / 0.7384 — one of three clears 0.75.

**These are different properties and only the second is what a cascade needs.** The gate is correctly reported as a failure.

### Escalation dial

Gate A fails on all three bots, so **the dial is not offered as a shippable result.** It is recorded for completeness, and it confirms the failure — escalating the least-confident items **lowers** kept-slice accuracy under the shipping definition:

| bot | def | kept-slice accuracy at 0% / 10% / 20% / 30% / 40% escalated |
|---|---|---|
| sofmattress | A shipped | 0.6801 → 0.6555 → 0.6321 → 0.6223 → 0.6513 (**worse**) |
| curekart | A shipped | 0.7023 → 0.6805 → 0.6469 → 0.6138 → 0.5950 (**worse**) |
| powerplay11 | A shipped | 0.5748 → 0.5322 → 0.4860 → 0.4331 → 0.3678 (**much worse**) |
| sofmattress | B answering | 0.7792 → 0.8029 → 0.8270 → 0.8519 → 0.8489 |
| curekart | B answering | 0.8540 → 0.8796 → 0.8923 → 0.9177 → 0.9446 |
| powerplay11 | B answering | 0.5855 → 0.6275 → 0.6636 → 0.6995 → 0.7091 |

Curekart's definition-B row is the only one that both passes the gate and produces a usable coverage/cost curve: escalating the least-confident 30% of *answered* items raises accuracy on the kept 70% from 0.8540 to 0.9177. **Framed strictly as coverage/cost** — this project measured out-of-fold that a cascade never beats the better base model, and nothing here contradicts that. It is also not a shippable pipeline on its own, because it presupposes the in-scope/OOS decision has already been made correctly.

**Diagnostic (test-tuned, optimistic, not a result):** errors *are* concentrated somewhere — a middle-confidence *band* of the same 20% budget captures 38.6% / 45.4% / 41.6% of all errors (bands at conf 0.127–0.171, 0.117–0.161, 0.051–0.063). A band router, not a tail router, is the only shape that could work on HINT3. The band position was chosen on test, so this is a direction for future work, not a measured result.

---

## 8. Contamination, label budget, and tiny intents

### Train/test contamination — **this suite is clean**

| bot | test rows | exact overlap | normalized overlap | of which in-scope | of which OOS | train internal dups |
|---|---|---|---|---|---|---|
| sofmattress | 397 | **0 (0.000%)** | 0 (0.000%) | 0/231 | 0/166 | 4 exact / 10 normalized |
| curekart | 991 | **6 (0.605%)** | 8 (0.807%) | 6/452 | 0/539 | 0 / 0 |
| powerplay11 | 983 | **0 (0.000%)** | 1 (0.102%) | 0/275 | 0/708 | 7 exact / 8 normalized |

Normalization = NFKC, lowercase, strip non-alphanumerics, collapse whitespace. Reported whatever they are, as instructed — and they are good news. Compare the project's two contaminated suites at 71.5% and 99.8% of test rows sharing a prompt with train. **HINT3 has at most 0.8% overlap, and zero on two of three bots.** No OOS test item is a train duplicate on any bot, which is what matters most for the open-set protocol. No result in this report needs a contamination discount.

### Label budget

| bot | train rows | intents | median/intent | min | intents <3 ex | intents =1 ex | test items in <3-ex intents | test items in 1-ex intents |
|---|---|---|---|---|---|---|---|---|
| sofmattress | 328 | 21 | 12 | 9 | 0 | 0 | 0 | 0 |
| curekart | 600 | 28 | 14 | 3 | 0 | 0 | 0 | 0 |
| powerplay11 | 471 | 59 | 7 | **1** | **14** | **7** | **27** of 275 | **14** of 275 |

**This sits deliberately below the project's crossover line.** The project's own measurement on a 77-class task was that an LLM led a free model by 32 points at 100 labels and lost by 3 at 2,000. At 328–600 labels total — 7–14 per intent — HINT3 is in the region where the project has already measured LLMs to lead. **A loss to a paid model here would be an expected, publishable finding, not a surprise.** It is also the region where the fine-tuned-encoder arm collapses (§4), which is the same phenomenon seen from the free side.

### What I did about intents too small to learn

**I kept them. I did not drop any intent.** Dropping the 14 powerplay11 intents with fewer than 3 training examples would have made their **27 test items (9.8% of the 275 in-scope items) unanswerable by construction** — a silent ceiling cut disguised as a modelling choice. Instead:

- All 59 powerplay11 intents are trained, including the 7 with a **single** training example (14 test items).
- `class_weight="balanced"` / class-weighted CE up-weight them, which is the only concession made.
- CV folds use stratified round-robin assignment rather than `StratifiedKFold`, which cannot handle singleton classes; a singleton's one example lands in exactly one fold's validation set and is necessarily missed there. This is honest and costs nothing on test.

**How they actually did on the shipped row:** the 14 sub-3-example intents own 27 test items; argmax gets **21 of 27** right, and **18 of 27** survive the operating threshold. Surprisingly strong — these intents are apparently lexically distinctive rather than hard. sofmattress and curekart have no intents below 3 examples, so this affects powerplay11 only.

### Other decisions recorded by measurement

- **TF-IDF `sublinear_tf` on the char block** (spec ambiguity): applying it to both blocks vs word-only changes in-scope accuracy by 0.0000 / 0.0000 / −0.0036. Resolved by number, not guess; does not affect any verdict.
- **Macro-F1 averaged over in-scope intents with non-zero test support.** curekart trains 28 intents but only 21 appear in test; averaging over the 7 empty classes would depress macro-F1 by a factor of 21/28 for reasons unrelated to the model. Both variants are in `hint3.json` (`macro_f1_inscope` vs `macro_f1_all_train_intents`).
- **Full probability matrices persisted** to `hint3_probs_{bot}.npz` for all four arms, and per-item `{pred, gold, conf}` to `hint3.json`. This closes the project's standing known limitation — the earlier encoder runner that kept only argmax and left every chart stuck on the TF-IDF row. **Every arm here, including the encoder, can be asked how sure it was.**

### Self-verification

Beyond the 675-number reproduction of the authors' files (§2), all headline figures were **re-derived a second way** by `hint3_verify.py`: it reads the persisted `.npz` probability matrices rather than in-memory objects, and recomputes in-scope accuracy, OOS recall, macro-F1, MCC (Gorodkin formula from the confusion matrix), gate AUROC (Mann-Whitney U with tie-averaged ranks instead of `sklearn.roc_auc_score`) and the 20%-error-share in **pure numpy with no `sklearn.metrics` at all**. **855 checks, all agreeing to 1e-8/1e-9.** I did run this, and it passed.

---

## 9. What we can and cannot claim

### Can claim

1. **A free, local, zero-marginal-cost classifier is competitive with 2020-era production NLU platforms on real production chatbot traffic.** Frozen `multilingual-e5-small` + logistic regression, on 328–600 labels, places **3rd / 2nd / 2nd of six** on macro-F1 over in-scope intents and **never loses a paired McNemar test** (7 wins, 8 ties, 0 losses) across 15 comparisons.
2. **Against published BERT specifically, stated precisely:**
   - *macro-F1:* we win **1 of 3** — powerplay11 0.4480 vs 0.4443. We lose curekart by 0.0037 (0.6462 vs 0.6499) and sofmattress by 0.0041 (0.6480 vs 0.6521), i.e. two losses inside 0.005.
   - *overall accuracy:* we win **2 of 3** — curekart 0.7508 vs 0.7467 and powerplay11 0.6907 vs 0.5259; we lose sofmattress 0.6675 vs 0.6952.
   - *MCC:* we win **1 of 3** — powerplay11 0.4202 vs 0.3182.
   - The powerplay11 accuracy win is decisive on paired items (239 vs 77 discordant, p=1.8e-20) — but it is on the metric the constant dominates on that bot, so it should not be leaned on.
3. **This comparison is unusually trustworthy** — paired same-items, byte-identical splits, and a metric stack that reproduces the authors' own 675 published numbers to 2.22e-16.
4. **HINT3 is not contaminated.** ≤0.8% train/test overlap, 0% on two of three bots, 0% among OOS items everywhere.
5. **Confidence on HINT3 is a good scope detector (AUROC ~0.85, on par with the best published systems) and a poor correctness detector (0.68–0.77).**
6. **Fine-tuning a 68M encoder on 328–600 labels loses badly to frozen embeddings + logreg** (20–24 in-scope accuracy points), even after CV-selecting 10–20 epochs instead of 3.

### Cannot claim

1. **Nothing about free-vs-paid. There is no LLM arm.** The zero-spend rule means HINT3 carries no paid-model number, and the five published baselines are commercial NLU platforms and BERT, not LLMs. **This arm does not test the project's central thesis** — it establishes where the free ceiling sits on this benchmark, which is a prerequisite for that test, not the test itself. Anyone reading a free-vs-paid conclusion into these numbers is reading something that is not there.
2. **We cannot claim a win over the published field.** We win 0 of 3 bots outright on macro-F1. The gaps are small (0.4–3.9 points) but they are losses.
3. **We cannot claim a usable cascade on HINT3.** The gate fails on all three bots on the row we would ship. "No useful middle" is the answer.
4. **We cannot use overall accuracy on powerplay11 at all** — a label-blind constant beats our shipped row by 14.55 points and beats all five published platforms too. Any powerplay11 accuracy figure in this project must carry the 0.7202 constant beside it or be struck.
5. **We cannot claim the published platforms were trained identically.** Their training was done by the authors on their own infrastructure in 2020; only the *evaluation* items are paired. Their training side remains published-on-authors'-protocol evidence.
6. **We cannot generalise the label-budget result upward.** At 328–600 labels HINT3 sits below the project's measured crossover; it says nothing about the 2,000-label regime where the project measured the free model to lead.
7. **The band-router finding is test-tuned and is not a result.** It is a direction.

### If asked "did the free arm hold up on HINT3?"

**Two wins and one loss, and we did not beat the published field anywhere.** The free arm holds up as a *peer* of commercial NLU on two bots, is worse than a constant on the third bot's headline metric, and offers **no routing headroom on any of them.** The thesis remains untested here for want of a paid arm.

---

## Files

| file | contents |
|---|---|
| `hint3_arm.py` | the runnable script; docstring records design decisions D1–D13 with the measurement that forced each |
| `hint3.json` | machine-readable results: per bot × arm — all metrics, 9-point and 101-point threshold sweeps, train-selected operating points, gate numbers, contamination, label budget, per-item `{pred, gold, conf}`, plus all five published platforms' sweeps and per-item predictions |
| `hint3_probs_{bot}.npz` | full probability matrices + class orders for all four arms (closes the argmax-only limitation) |
| `hint3_verify.py` | independent re-derivation, pure-numpy metrics, 855 checks |
| `hint3_report.py` | table generator |
| `hint3_tables.md` | generated tables (superset of those inlined above) |
| `hint3_run.log` | run log |
| `hint3_encoder_cache/` | cached encoder CV/final probabilities (re-runs are cheap) |
