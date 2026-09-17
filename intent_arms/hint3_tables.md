## T1. Constant-predictor guard

| bot | test | in-scope | OOS | all-OOS const acc | maj-intent const acc | const in-scope acc | const macro-F1 | const MCC |
|---|---|---|---|---|---|---|---|---|
| sofmattress | 397 | 231 | 166 | **0.4181** | 0.0176 | 0.0000 | 0.0000 | 0.0000 |
| curekart | 991 | 452 | 539 | **0.5439** | 0.2109 | 0.0000 | 0.0000 | 0.0000 |
| powerplay11 | 983 | 275 | 708 | **0.7202** | 0.0509 | 0.0000 | 0.0000 | 0.0000 |

## T2. Free arms, no rejection (t=0, argmax always answers)

| bot | arm | in-scope acc | macro-F1 | MCC | overall acc | CV macro-F1 (train OOF) |
|---|---|---|---|---|---|---|
| sofmattress | majority | 0.0303 | 0.0017 | 0.0000 | 0.0176 | 0.0089 |
| sofmattress | tfidf_lr | 0.7359 | 0.5734 | 0.4625 | 0.4282 | 0.8644 |
| sofmattress | frozen_e5_lr **<-- shipped** | 0.7792 | 0.5852 | 0.4847 | 0.4534 | 0.8731 |
| sofmattress | ettin_encoder_3ep | 0.4848 | 0.3233 | 0.2898 | 0.2821 | 0.5676 |
| sofmattress | ettin_encoder_cv20ep | 0.5714 | 0.3958 | 0.3485 | 0.3325 | 0.7158 |
| curekart | majority | 0.4624 | 0.0166 | 0.0000 | 0.2109 | 0.0098 |
| curekart | tfidf_lr | 0.8186 | 0.5170 | 0.3771 | 0.3734 | 0.8549 |
| curekart | frozen_e5_lr **<-- shipped** | 0.8540 | 0.5033 | 0.4023 | 0.3895 | 0.8549 |
| curekart | ettin_encoder_3ep | 0.5199 | 0.2930 | 0.2394 | 0.2371 | 0.5751 |
| curekart | ettin_encoder_cv10ep | 0.7611 | 0.4089 | 0.3405 | 0.3471 | 0.7332 |
| powerplay11 | majority | 0.1818 | 0.0017 | 0.0000 | 0.0509 | 0.0030 |
| powerplay11 | tfidf_lr | 0.6145 | 0.3361 | 0.2355 | 0.1719 | 0.7226 |
| powerplay11 | frozen_e5_lr **<-- shipped** | 0.5855 | 0.2874 | 0.2312 | 0.1638 | 0.7385 |
| powerplay11 | ettin_encoder_3ep | 0.2109 | 0.1187 | 0.0798 | 0.0590 | 0.4276 |
| powerplay11 | ettin_encoder_cv20ep | 0.3491 | 0.1488 | 0.1298 | 0.0977 | 0.5765 |

## T3. Free arms at the TRAIN-SELECTED operating point (IRR=10%, no test tuning)

| bot | arm | t* | in-scope acc | OOS recall | macro-F1 | MCC | overall acc | beats all-OOS const? |
|---|---|---|---|---|---|---|---|---|
| sofmattress | majority | 0.1037 | 0.0303 | 0.0000 | 0.0017 | 0.0000 | 0.0176 | **NO (-0.4005)** |
| sofmattress | tfidf_lr | 0.2103 | 0.6407 | 0.6928 | 0.6203 | 0.5887 | 0.6625 | YES |
| sofmattress | frozen_e5_lr **<-- shipped** | 0.1261 | 0.7013 | 0.6506 | 0.6485 | 0.6184 | 0.6801 | YES |
| sofmattress | ettin_encoder_3ep | 0.3128 | 0.4675 | 0.1988 | 0.3383 | 0.3039 | 0.3552 | **NO (-0.0630)** |
| sofmattress | ettin_encoder_cv20ep | 0.5465 | 0.5368 | 0.4217 | 0.4455 | 0.4131 | 0.4887 | YES |
| curekart | majority | 0.1583 | 0.4624 | 0.0000 | 0.0166 | 0.0000 | 0.2109 | **NO (-0.3330)** |
| curekart | tfidf_lr | 0.1834 | 0.7721 | 0.6234 | 0.5670 | 0.5937 | 0.6912 | YES |
| curekart | frozen_e5_lr **<-- shipped** | 0.1170 | 0.7854 | 0.6327 | 0.6147 | 0.6004 | 0.7023 | YES |
| curekart | ettin_encoder_3ep | 0.3338 | 0.4867 | 0.3952 | 0.3582 | 0.3065 | 0.4369 | **NO (-0.1070)** |
| curekart | ettin_encoder_cv10ep | 0.5472 | 0.7367 | 0.3135 | 0.4800 | 0.4198 | 0.5066 | **NO (-0.0373)** |
| powerplay11 | majority | 0.0977 | 0.1818 | 0.0000 | 0.0017 | 0.0000 | 0.0509 | **NO (-0.6694)** |
| powerplay11 | tfidf_lr | 0.1589 | 0.5273 | 0.6215 | 0.4291 | 0.3677 | 0.5951 | **NO (-0.1251)** |
| powerplay11 | frozen_e5_lr **<-- shipped** | 0.0505 | 0.5236 | 0.5946 | 0.4073 | 0.3626 | 0.5748 | **NO (-0.1455)** |
| powerplay11 | ettin_encoder_3ep | 0.2046 | 0.1927 | 0.2090 | 0.1312 | 0.0798 | 0.2045 | **NO (-0.5158)** |
| powerplay11 | ettin_encoder_cv20ep | 0.4213 | 0.3018 | 0.4040 | 0.1764 | 0.1647 | 0.3754 | **NO (-0.3449)** |

## T4. Threshold sweep, shipped row (paper grid)


**sofmattress** -- shipped arm `frozen_e5_lr`; all-OOS constant acc = 0.4181

| t | overall acc | in-scope acc | OOS recall | macro-F1 | MCC | answered % |
|---|---|---|---|---|---|---|
| 0.1 | 0.5995 | 0.7619 | 0.3735 | 0.6295 | 0.5757 | 81.9 |
| 0.2 | 0.6398 | 0.4372 | 0.9217 | 0.5027 | 0.5269 | 32.7 |
| 0.3 | 0.5189 | 0.1732 | 1.0000 | 0.2683 | 0.3522 | 10.8 |
| 0.4 | 0.4484 | 0.0519 | 1.0000 | 0.1102 | 0.1892 | 3.3 |
| 0.5 | 0.4257 | 0.0130 | 1.0000 | 0.0560 | 0.0967 | 0.8 |
| 0.6 | 0.4181 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.7 | 0.4181 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.8 | 0.4181 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.9 | 0.4181 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |

**curekart** -- shipped arm `frozen_e5_lr`; all-OOS constant acc = 0.5439

| t | overall acc | in-scope acc | OOS recall | macro-F1 | MCC | answered % |
|---|---|---|---|---|---|---|
| 0.1 | 0.5913 | 0.8296 | 0.3915 | 0.5686 | 0.5276 | 77.0 |
| 0.2 | 0.7366 | 0.5022 | 0.9332 | 0.5792 | 0.5751 | 27.5 |
| 0.3 | 0.6519 | 0.2456 | 0.9926 | 0.3830 | 0.4344 | 11.8 |
| 0.4 | 0.5964 | 0.1195 | 0.9963 | 0.2298 | 0.3047 | 5.7 |
| 0.5 | 0.5762 | 0.0708 | 1.0000 | 0.1743 | 0.2415 | 3.2 |
| 0.6 | 0.5540 | 0.0221 | 1.0000 | 0.0631 | 0.1344 | 1.0 |
| 0.7 | 0.5449 | 0.0022 | 1.0000 | 0.0159 | 0.0428 | 0.1 |
| 0.8 | 0.5439 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.9 | 0.5439 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |

**powerplay11** -- shipped arm `frozen_e5_lr`; all-OOS constant acc = 0.7202

| t | overall acc | in-scope acc | OOS recall | macro-F1 | MCC | answered % |
|---|---|---|---|---|---|---|
| 0.1 | 0.7497 | 0.2000 | 0.9633 | 0.3025 | 0.3277 | 9.0 |
| 0.2 | 0.7233 | 0.0109 | 1.0000 | 0.0184 | 0.0972 | 0.3 |
| 0.3 | 0.7223 | 0.0073 | 1.0000 | 0.0115 | 0.0793 | 0.2 |
| 0.4 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.5 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.6 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.7 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.8 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |
| 0.9 | 0.7202 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0 |

## T5. Paired comparison vs published platforms (same items, same order)

Each method at its OWN test-optimal macro-F1 threshold -> optimistic for every row, symmetrically.


**sofmattress**

| method | t | in-scope acc | OOS recall | macro-F1 | MCC | overall acc |
|---|---|---|---|---|---|---|
| **ours: frozen_e5_lr** | 0.12 | 0.7100 | 0.6084 | **0.6480** | 0.6097 | 0.6675 |
| dialogflow (published) | 0.46 | 0.7013 | 0.6867 | 0.6713 | 0.6353 | 0.6952 |
| luis (published) | 0.10 | 0.5931 | 0.7530 | 0.5933 | 0.5807 | 0.6599 |
| rasa (published) | 0.68 | 0.5801 | 0.6265 | 0.5937 | 0.5118 | 0.5995 |
| bert (published) | 0.85 | 0.6840 | 0.7108 | 0.6521 | 0.6331 | 0.6952 |
| haptik (published) | 0.29 | 0.6667 | 0.6687 | 0.6268 | 0.5987 | 0.6675 |
| all-OOS constant | - | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4181 |

McNemar (overall correctness, paired):

| vs | ours right / theirs wrong | theirs right / ours wrong | exact p |
|---|---|---|---|
| dialogflow | 47 | 58 | 0.329 |
| luis | 65 | 62 | 0.859 |
| rasa | 76 | 49 | 0.0197 |
| bert | 42 | 53 | 0.305 |
| haptik | 50 | 50 | 1 |

**curekart**

| method | t | in-scope acc | OOS recall | macro-F1 | MCC | overall acc |
|---|---|---|---|---|---|---|
| **ours: frozen_e5_lr** | 0.15 | 0.6881 | 0.7996 | **0.6462** | 0.6160 | 0.7487 |
| dialogflow (published) | 0.58 | 0.6062 | 0.8479 | 0.5871 | 0.5930 | 0.7376 |
| luis (published) | 0.32 | 0.6372 | 0.6475 | 0.5377 | 0.5014 | 0.6428 |
| rasa (published) | 0.68 | 0.7965 | 0.4100 | 0.6056 | 0.5008 | 0.5863 |
| bert (published) | 0.99 | 0.7412 | 0.7514 | 0.6499 | 0.6213 | 0.7467 |
| haptik (published) | 0.72 | 0.5265 | 0.9295 | 0.5640 | 0.5918 | 0.7457 |
| all-OOS constant | - | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5439 |

McNemar (overall correctness, paired):

| vs | ours right / theirs wrong | theirs right / ours wrong | exact p |
|---|---|---|---|
| dialogflow | 125 | 114 | 0.518 |
| luis | 222 | 117 | 1.25e-08 |
| rasa | 253 | 92 | 1.64e-18 |
| bert | 120 | 118 | 0.948 |
| haptik | 132 | 129 | 0.902 |

**powerplay11**

| method | t | in-scope acc | OOS recall | macro-F1 | MCC | overall acc |
|---|---|---|---|---|---|---|
| **ours: frozen_e5_lr** | 0.06 | 0.4727 | 0.7754 | **0.4480** | 0.4202 | 0.6907 |
| dialogflow (published) | 0.56 | 0.5127 | 0.5946 | 0.3670 | 0.3463 | 0.5717 |
| luis (published) | 0.27 | 0.3855 | 0.7867 | 0.3978 | 0.3660 | 0.6745 |
| rasa (published) | 0.61 | 0.3927 | 0.7429 | 0.3442 | 0.3259 | 0.6450 |
| bert (published) | 0.62 | 0.5236 | 0.5268 | 0.4443 | 0.3182 | 0.5259 |
| haptik (published) | 0.31 | 0.6036 | 0.6285 | 0.4874 | 0.4230 | 0.6216 |
| all-OOS constant | - | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.7202 |

McNemar (overall correctness, paired):

| vs | ours right / theirs wrong | theirs right / ours wrong | exact p |
|---|---|---|---|
| dialogflow | 210 | 93 | 1.5e-11 |
| luis | 131 | 115 | 0.339 |
| rasa | 154 | 109 | 0.00655 |
| bert | 239 | 77 | 1.84e-20 |
| haptik | 167 | 99 | 3.64e-05 |

## T6. Cascade pre-flight gate (shipped row)

| bot | arm | def | AUROC(conf->correct) | gate >=0.75 | err share in least-conf 20% | gate >>20% | overall |
|---|---|---|---|---|---|---|---|
| sofmattress | frozen_e5_lr | A shipped pipeline | 0.5106 | FAIL | 0.0787 | FAIL | FAIL |
| sofmattress | frozen_e5_lr | B answering subset | 0.6781 | FAIL | 0.3725 | PASS | FAIL |
| curekart | frozen_e5_lr | A shipped pipeline | 0.4507 | FAIL | 0.0508 | FAIL | FAIL |
| curekart | frozen_e5_lr | B answering subset | 0.7678 | PASS | 0.4091 | PASS | **PASS** |
| powerplay11 | frozen_e5_lr | A shipped pipeline | 0.2405 | FAIL | 0.0335 | FAIL | FAIL |
| powerplay11 | frozen_e5_lr | B answering subset | 0.7384 | FAIL | 0.3509 | PASS | FAIL |

## T7. Escalation dial (coverage/cost), shipped row

- **sofmattress** / A shipped pipeline: kept-slice accuracy at 0/10/20/30/40% escalated = 0.6801 | 0.6555 | 0.6321 | 0.6223 | 0.6513
- **sofmattress** / B answering subset: kept-slice accuracy at 0/10/20/30/40% escalated = 0.7792 | 0.8029 | 0.8270 | 0.8519 | 0.8489
- **curekart** / A shipped pipeline: kept-slice accuracy at 0/10/20/30/40% escalated = 0.7023 | 0.6805 | 0.6469 | 0.6138 | 0.5950
- **curekart** / B answering subset: kept-slice accuracy at 0/10/20/30/40% escalated = 0.8540 | 0.8796 | 0.8923 | 0.9177 | 0.9446
- **powerplay11** / A shipped pipeline: kept-slice accuracy at 0/10/20/30/40% escalated = 0.5748 | 0.5322 | 0.4860 | 0.4331 | 0.3678
- **powerplay11** / B answering subset: kept-slice accuracy at 0/10/20/30/40% escalated = 0.5855 | 0.6275 | 0.6636 | 0.6995 | 0.7091

## T8. Contamination

| bot | test rows | exact train/test overlap | normalized overlap | of which in-scope | of which OOS | train internal dups |
|---|---|---|---|---|---|---|
| sofmattress | 397 | 0 (0.0%) | 0 (0.0%) | 0/231 | 0/166 | 4 exact / 10 normalized |
| curekart | 991 | 6 (0.605%) | 8 (0.807%) | 6/452 | 0/539 | 0 exact / 0 normalized |
| powerplay11 | 983 | 0 (0.0%) | 1 (0.102%) | 0/275 | 0/708 | 7 exact / 8 normalized |

## T9. Label budget and tiny intents

| bot | train rows | intents | median/intent | min | intents <3 ex | intents ==1 ex | test items in <3 | test items in ==1 | train intents absent from test |
|---|---|---|---|---|---|---|---|---|---|
| sofmattress | 328 | 21 | 12 | 9 | 0 | 0 | 0 | 0 | 1 |
| curekart | 600 | 28 | 14 | 3 | 0 | 0 | 0 | 0 | 7 |
| powerplay11 | 471 | 59 | 7 | 1 | 14 | 7 | 27 | 14 | 1 |

### Tiny-intent outcomes on the shipped row

- **sofmattress**: no intents with <3 training examples.
- **curekart**: no intents with <3 training examples.
- **powerplay11**: 14 intents with <3 train examples own 27 test items; argmax gets 21/27 right, 18/27 survive the operating threshold.

## T10. Encoder epoch selection (train CV only)

| bot | CV macro-F1 @3ep | @10ep | @20ep | selected |
|---|---|---|---|---|
| sofmattress | 0.5676 | 0.7050 | 0.7158 | 20 |
| curekart | 0.5751 | 0.7332 | 0.7326 | 10 |
| powerplay11 | 0.4276 | 0.5660 | 0.5765 | 20 |

## T11. TF-IDF sublinear_tf-on-char ablation (D11)

| bot | sublinear on both (used) | sublinear on word only | delta |
|---|---|---|---|
| sofmattress | 0.7359 | 0.7359 | +0.0000 |
| curekart | 0.8186 | 0.8186 | +0.0000 |
| powerplay11 | 0.6145 | 0.6182 | -0.0036 |
