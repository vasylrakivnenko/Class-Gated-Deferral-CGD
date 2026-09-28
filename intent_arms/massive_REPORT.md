# MASSIVE: does the free arm survive outside English?

60-intent virtual-assistant utterance classification, **13 of 51 locales measured** (list pre-committed before any result; see `massive_arm.py` docstring). Amazon Science, arXiv:2204.08582.

- Anchor **en-US**, all four arms. Breadth: the two cheap arms across all 13 locales. Fine-tuned encoders on 4 locales (en-US, de-DE, zh-CN, th-TH).
- Every arm local. **Zero paid API calls, zero spend.** SEED = 0.
- Test set is the official per-locale split: 2,974 utterances per locale, 60 intents.

## Headline

**Non-English is not what breaks the free arm. What broke were a vectorizer default, the choice of encoder, and our own training protocol -- all three ours, none of them the language's.**

On the anchor en-US the fine-tuned encoder reaches **0.8803**, against published full-data fine-tuned baselines of 0.8790 (mT5-Base T2T), 0.8830 (XLM-R-Base) and 0.8900 (mT5-Base encoder) -- a **tie**, at +-1.3 points of resolution on a 2,974-item test split. Across the 12 non-English locales the best free arm ranges **0.7794** (ar-SA) to **0.8396** (es-ES), median **0.8188**, against a majority baseline of 0.0703.

**Where the free arm breaks -- stated first, before the wins:**

1. **The cheap arms alone lose to the published fine-tuned baselines in every one of the 13 locales measured**, by 0.3 to 5.6 points (TF-IDF or frozen embeddings, whichever is better per locale). There is no locale where a cheap arm wins. Buying breadth cheaply costs real accuracy, and the 9 locales where we ran only cheap arms should be read as a floor, not a result.
2. **We pre-registered that the English encoder would collapse on non-Latin scripts. It did not -- the prediction is REFUTED.** `jhu-clsp/ettin-encoder-68m` scores 0.8191 on zh-CN, the *best of any arm we ran there*, on a tokenizer that needs 1.43 pieces per character against 0.26 on English. Where it does underperform its own cheap arms is **de-DE** (0.7942 against 0.8289 TF-IDF) -- Latin, high-resource, the case we expected to be safest. Tokenizer fertility turned out to be a good mechanism story and a bad predictor. Section 4.
3. **Our own multilingual-encoder control did not converge, so the obvious fix is UNTESTED, not confirmed.** Fine-tuned `multilingual-e5-small` came in at 0.8255 on en-US -- *below* the frozen version of the same checkpoint (0.8362), with training loss still at 1.68 when the fixed 3-epoch schedule ended against ettin's 0.077. We are reporting that as a failed control rather than as evidence about multilingual encoders. Section 4a.
4. **A sklearn default silently cost hi-IN 23 points.** The default `token_pattern` does not match Unicode combining marks, so Devanagari words were shredded; word-only TF-IDF ran at 0.5797 instead of 0.8117. Nothing in the union score revealed it. See section 3.
5. **The cascade gate fails on 0 of 13 locales** (none); it passes on 13 (en-US, de-DE, es-ES, fr-FR, ru-RU, zh-CN, ja-JP, ko-KR, ar-SA, hi-IN, th-TH, sw-KE, am-ET). Where it fails there is no useful middle, which is a real answer.

**What does survive, and it is the point of this arm:** there is no script cliff. The 12 non-English locales sit in a 6.0-point band (0.7794-0.8396), and non-Latin locales interleave with Latin ones rather than sitting below them: the ranking from 2nd to 11th place runs es-ES 0.8396 > fr-FR 0.8376 > de-DE 0.8289 > sw-KE 0.8278 > ru-RU 0.8208 > zh-CN 0.8191 > hi-IN 0.8184 > ja-JP 0.8178 > ko-KR 0.8093 > th-TH 0.8043, i.e. zh-CN (Han) outranks hi-IN, ja-JP, ko-KR and th-TH, and sw-KE -- Latin but low-resource -- outranks ru-RU. The full anchor-to-worst spread is 10.1 points, but 11 of 13 locales clear 0.80. The two weakest, ar-SA and am-ET, are also the two the *paper's own* fine-tuned baselines rank at or near the bottom (XLM-R 0.807 and 0.817 against 0.883 on en-US) -- the ordering is a property of the locales, not of our arm. Non-English is not, by itself, the thing that breaks a free classifier.

## 1. Contamination and id-alignment (run before any modelling)

MASSIVE's locales are parallel translations sharing an `id`, so a pooled random split would put one utterance's Spanish copy in train and its German twin in test. Measured, not assumed:

- **16,521 distinct ids**, = 11,514 train + 2,033 dev + 2,974 test.
- Every id is present in **all 52 locale directories** (100.0000%).
- **ids whose partition disagrees between locales: 0 (0.0000%).** The official split is exactly id-aligned, so a per-locale run using it is structurally free of the cross-lingual twin leak, and any cross-lingual transfer run that reuses these partitions holds out by `id` for free.
- Train/test `id` overlap within a locale: **0** in all 52 locales.

Within-locale surface duplication between train and test, on normalised `utt` (all 52 locale dirs checked, 13-locale subset shown):

| locale | exact-dup test rows | share | label conflicts | near-dup >=0.90 | share |
|---|---|---|---|---|---|
| en-US | 21 | 0.71% | 2 | 442 | 14.86% |
| de-DE | 115 | 3.87% | 7 | 346 | 11.63% |
| es-ES | 136 | 4.57% | 2 | 334 | 11.23% |
| fr-FR | 150 | 5.04% | 8 | 338 | 11.37% |
| ru-RU | 208 | 6.99% | 6 | 366 | 12.31% |
| zh-CN | 223 | 7.50% | 13 | 244 | 8.20% |
| ja-JP | 207 | 6.96% | 13 | 225 | 7.57% |
| ko-KR | 187 | 6.29% | 9 | 344 | 11.57% |
| ar-SA | 237 | 7.97% | 20 | 344 | 11.57% |
| hi-IN | 133 | 4.47% | 7 | 349 | 11.74% |
| th-TH | 247 | 8.31% | 14 | 374 | 12.58% |
| sw-KE | 170 | 5.72% | 2 | 373 | 12.54% |
| am-ET | 105 | 3.53% | 2 | 202 | 6.79% |

**A translation artefact worth naming, because it runs against a naive reading.** en-US has the *lowest* exact-duplicate overlap of any locale (0.71%), while the translated locales run 2.49%-14.43% (median 5.41% over all 52, worst km-KH). Distinct English utterances collapse onto identical strings when translated. So the non-English locales have *more* train/test surface overlap than English, which if anything flatters them relative to the anchor -- the cross-locale comparisons in this report are therefore mildly conservative about English, not generous to it. It does not change any verdict here: the cheap arms trail published fine-tuned baselines in every locale regardless.

424 exact duplicates across all locales carry a test intent that never appears with that text in train, i.e. an irreducible label-noise floor.

## 2. Per-locale results, all arms

Intent accuracy (MASSIVE's own headline metric). `pub XLM-R` is **published-on-authors'-protocol** (arXiv:2204.08582 Table 8, full-data fine-tuned XLM-R-Base) -- *not* a paired same-items run, which is the weakest evidence class this project holds.

| locale | majority | TF-IDF union | frozen e5 | ettin-68m FT | e5-small FT | **best free** | pub XLM-R | gap |
|---|---|---|---|---|---|---|---|---|
| en-US | 0.0703 | 0.8440 | 0.8362 | 0.8803 | 0.8255 | **0.8803** | 0.8830 | -0.3 |
| de-DE | 0.0703 | 0.8289 | 0.7989 | 0.7942 | 0.7283 | **0.8289** | 0.8570 | -2.8 |
| es-ES | 0.0703 | 0.8396 | 0.8124 | - | - | **0.8396** | 0.8690 | -2.9 |
| fr-FR | 0.0703 | 0.8376 | 0.8171 | - | - | **0.8376** | 0.8630 | -2.5 |
| ru-RU | 0.0703 | 0.8208 | 0.8204 | - | - | **0.8208** | 0.8720 | -5.1 |
| zh-CN * | 0.0703 | 0.7465 | 0.8073 | 0.8191 | 0.7646 | **0.8191** | 0.8490 | -3.0 |
| ja-JP * | 0.0703 | 0.7898 | 0.8178 | - | - | **0.8178** | 0.8390 | -2.1 |
| ko-KR | 0.0703 | 0.8093 | 0.7952 | - | - | **0.8093** | 0.8650 | -5.6 |
| ar-SA | 0.0703 | 0.7794 | 0.7313 | - | - | **0.7794** | 0.8070 | -2.8 |
| hi-IN | 0.0703 | 0.8184 | 0.8040 | - | - | **0.8184** | 0.8580 | -4.0 |
| th-TH * | 0.0703 | 0.8043 | 0.7774 | 0.7744 | 0.7091 | **0.8043** | 0.8470 | -4.3 |
| sw-KE | 0.0703 | 0.8278 | 0.7424 | - | - | **0.8278** | 0.8310 | -0.3 |
| am-ET | 0.0703 | 0.7976 | 0.7135 | - | - | **0.7976** | 0.8170 | -1.9 |

`*` = no whitespace word boundaries. Macro-F1 over the 60 intents:

| locale | majority | TF-IDF union | frozen e5 | ettin-68m FT | e5-small FT | smallest intent in test |
|---|---|---|---|---|---|---|
| en-US | 0.0022 | 0.8033 | 0.8105 | 0.8670 | 0.7094 | 1 |
| de-DE | 0.0022 | 0.7859 | 0.7561 | 0.7574 | 0.5754 | 1 |
| es-ES | 0.0022 | 0.8084 | 0.7788 | - | - | 1 |
| fr-FR | 0.0022 | 0.8103 | 0.7809 | - | - | 1 |
| ru-RU | 0.0022 | 0.7913 | 0.7812 | - | - | 1 |
| zh-CN | 0.0022 | 0.7082 | 0.7635 | 0.7702 | 0.6336 | 1 |
| ja-JP | 0.0022 | 0.7709 | 0.7887 | - | - | 1 |
| ko-KR | 0.0022 | 0.7757 | 0.7665 | - | - | 1 |
| ar-SA | 0.0022 | 0.7368 | 0.6779 | - | - | 1 |
| hi-IN | 0.0022 | 0.7788 | 0.7637 | - | - | 1 |
| th-TH | 0.0022 | 0.7859 | 0.7559 | 0.7742 | 0.5801 | 1 |
| sw-KE | 0.0022 | 0.8057 | 0.7017 | - | - | 1 |
| am-ET | 0.0022 | 0.7632 | 0.6627 | - | - | 1 |

**Per-class resolution, and it is thin enough to matter.** The intents are strongly unbalanced and the test split is parallel, so these counts hold for every locale: **59 of 60 intents appear in test at all** (`cooking_query` has zero test items), the smallest present intent has **1 test utterance**, the median is 35 and the largest is 209. **6 intents have 10 or fewer test items and 14 have 20 or fewer.**

Consequences, stated so the per-class numbers are not over-read:

- A single item flips the rarest intent's recall by 100 points. For the 6 intents at n<=10, one item is worth >=10 points of recall, so **no per-intent claim is resolvable at finer than ~10-20 points** for that group.
- Macro-F1 averages all 59 present intents equally, so it is dominated by exactly these thin classes -- the 10 rarest intents carry 2.7% of test items but 16.9% of the macro-F1 weight. Treat macro-F1 here as a noisy tail-sensitivity indicator and **intent accuracy as the reliable figure**; accuracy does not inherit this problem, and it is also MASSIVE's own headline metric.
- This is the main reason the fine-tuned e5 row's macro-F1 (0.7094) collapses further than its accuracy (0.8255): an undertrained model gives up the rare classes first.

**Constant-predictor guard.** Majority intent accuracy is 0.0703 in every locale; every arm in every locale clears it by a wide margin, so no row is dropped or flagged.

## 2a. The frozen multilingual embedder is NOT the best cheap arm -- another prediction wrong

The brief's expectation, and ours, was that a frozen multilingual embedder would be the arm that carries non-English. It is not. Plain character-ngram TF-IDF beats it on most locales:

| locale | TF-IDF char-only | TF-IDF union | frozen e5 | char - frozen | union - frozen |
|---|---|---|---|---|---|
| en-US | 0.8376 | 0.8440 | 0.8362 | +0.1 | +0.8 |
| de-DE | 0.8191 | 0.8289 | 0.7989 | +2.0 | +3.0 |
| es-ES | 0.8238 | 0.8396 | 0.8124 | +1.1 | +2.7 |
| fr-FR | 0.8241 | 0.8376 | 0.8171 | +0.7 | +2.1 |
| ru-RU | 0.8208 | 0.8208 | 0.8204 | +0.0 | +0.0 |
| zh-CN * | 0.7599 | 0.7465 | 0.8073 | -4.7 | -6.1 |
| ja-JP * | 0.8013 | 0.7898 | 0.8178 | -1.6 | -2.8 |
| ko-KR | 0.8134 | 0.8093 | 0.7952 | +1.8 | +1.4 |
| ar-SA | 0.7751 | 0.7794 | 0.7313 | +4.4 | +4.8 |
| hi-IN | 0.8211 | 0.8184 | 0.8040 | +1.7 | +1.4 |
| th-TH * | 0.8090 | 0.8043 | 0.7774 | +3.2 | +2.7 |
| sw-KE | 0.8204 | 0.8278 | 0.7424 | +7.8 | +8.5 |
| am-ET | 0.7972 | 0.7976 | 0.7135 | +8.4 | +8.4 |

**Char-ngram TF-IDF beats frozen `multilingual-e5-small` on 11 of 13 locales.** The two it loses are exactly the no-whitespace pair, zh-CN and ja-JP (frozen wins by 4.7 and 1.6 points), which is the one place character n-grams have no word-like unit to latch onto.

**The largest gaps against the embedder are on the two low-resource locales**: am-ET +8.4 and sw-KE +7.8 points, with ar-SA +4.4 next. That is the interpretable part: a frozen embedder can only supply what its pretraining mixture contains, and for Amharic and Swahili that is thin, whereas character n-grams are fitted on the locale's own 11,514 training utterances and do not care how much Amharic was on the web. 11.5k in-language labels beat a multilingual prior.

Two consequences worth stating, because they invert the intuition this arm started with:

- If you are picking ONE cheap arm for an unknown language, pick **char-ngram TF-IDF**, not a frozen multilingual embedder -- unless the script has no whitespace, which you can detect in one line without any model.
- The frozen embedder's value here is *cross-lingual transfer* (section 5a), not in-language accuracy. Those are different jobs and this arm separates them.

## 3. Segmentation and the word/char split

The prediction made before running: a word 1-2gram TF-IDF must collapse on the three no-whitespace scripts, because sklearn's token pattern splits on whitespace, so a "word" becomes a whole clause. First, the precondition, model-free:

| locale | mean whitespace tokens / utterance | chars | chars per token |
|---|---|---|---|
| en-US | 6.92 | 35.0 | 5.06 |
| de-DE | 6.57 | 39.6 | 6.02 |
| es-ES | 7.22 | 39.6 | 5.49 |
| fr-FR | 7.46 | 42.8 | 5.73 |
| ru-RU | 5.88 | 36.9 | 6.27 |
| zh-CN * | 1.05 | 10.6 | 10.08 |
| ja-JP * | 1.17 | 15.6 | 13.39 |
| ko-KR | 4.88 | 15.6 | 3.19 |
| ar-SA | 5.26 | 27.0 | 5.13 |
| hi-IN | 7.59 | 35.8 | 4.72 |
| th-TH * | 3.35 | 31.4 | 9.35 |
| sw-KE | 6.47 | 38.2 | 5.90 |
| am-ET | 5.58 | 24.1 | 4.31 |

zh-CN averages **1.05** whitespace tokens per utterance and ja-JP **1.17** -- the entire utterance is one token. Now the two halves scored separately:

| locale | word-only | char-only | union | char - word | word-only features |
|---|---|---|---|---|---|
| en-US | 0.8278 | 0.8376 | 0.8440 | +1.0 | 29,040 |
| de-DE | 0.7841 | 0.8191 | 0.8289 | +3.5 | 35,272 |
| es-ES | 0.8073 | 0.8238 | 0.8396 | +1.6 | 30,867 |
| fr-FR | 0.8083 | 0.8241 | 0.8376 | +1.6 | 30,693 |
| ru-RU | 0.7915 | 0.8208 | 0.8208 | +2.9 | 36,060 |
| zh-CN * | 0.1459 | 0.7599 | 0.7465 | +61.4 | 11,579 |
| ja-JP * | 0.1715 | 0.8013 | 0.7898 | +63.0 | 13,304 |
| ko-KR | 0.7603 | 0.8134 | 0.8093 | +5.3 | 33,536 |
| ar-SA | 0.7471 | 0.7751 | 0.7794 | +2.8 | 38,020 |
| hi-IN | 0.5797 | 0.8211 | 0.8184 | +24.1 | 14,113 |
| th-TH * | 0.7095 | 0.8090 | 0.8043 | +10.0 | 38,006 |
| sw-KE | 0.8050 | 0.8204 | 0.8278 | +1.5 | 32,458 |
| am-ET | 0.7603 | 0.7972 | 0.7976 | +3.7 | 42,616 |

**The prediction was half right, and the half that was wrong is the more interesting finding.** Reported plainly because it was pre-registered:

| locale | word-only | char-only | char - word | predicted? |
|---|---|---|---|---|
| zh-CN | 0.1459 | 0.7599 | +61.4 | yes -- confirmed |
| ja-JP | 0.1715 | 0.8013 | +63.0 | yes -- confirmed |
| th-TH | 0.7095 | 0.8090 | +10.0 | yes, but **much weaker than predicted** |
| hi-IN | 0.5797 | 0.8211 | +24.1 | **no -- not predicted at all** |

On the other 9 whitespace-delimited locales the same char-minus-word difference is only +1.0 to +5.3 points, so the effect is specific to these four locales, not a general preference for character features. **Note that hi-IN is whitespace-delimited and still loses 24 points, so segmentation cannot be the explanation for it.**

**There are two separate mechanisms, and the pre-registered prediction conflated them.**

*Mechanism 1 -- no whitespace.* zh-CN and ja-JP average 1.05 and 1.17 whitespace tokens per utterance: the whole utterance is one token, so almost every feature is unique to one training row and nothing generalises. The word half falls to 0.1459 / 0.1715 -- only 2.1x and 2.4x the 0.0703 majority baseline, against 12x on en-US. zh-CN's word vectorizer extracts only 11,579 features from 11,514 utterances against en-US's 29,040.

*Mechanism 2 -- sklearn's default `token_pattern` drops Unicode combining marks, and this is a config bug rather than a language property.* The default is `r"(?u)\b\w\w+\b"`, and Devanagari vowel signs and virama are Unicode categories Mn/Mc, which `\w` does not match. So a Hindi word is shredded into the consonant runs between its marks:

```
'शुक्रवार को सुबह नौ बजे मुझे जगा दो'  ->  ['रव', 'बह', 'बज', 'जग']
```
| locale | whitespace tokens rejected by the default pattern | chars that are combining marks | utterances left with ZERO word features |
|---|---|---|---|
| en-US | 6.7% | 0.0% | 0.0% |
| de-DE | 1.3% | 0.0% | 0.0% |
| es-ES | 3.8% | 0.0% | 0.0% |
| fr-FR | 13.4% | 0.0% | 0.0% |
| ru-RU | 10.3% | 0.0% | 0.0% |
| zh-CN | 1.4% | 0.0% | 0.1% |
| ja-JP | 4.7% | 0.0% | 0.1% |
| ko-KR | 17.7% | 0.0% | 0.3% |
| ar-SA | 0.8% | 0.1% | 0.0% |
| hi-IN | 88.5% | 38.6% | 2.6% |
| th-TH | 80.2% | 22.5% | 0.2% |
| sw-KE | 0.7% | 0.0% | 0.0% |
| am-ET | 4.8% | 0.0% | 0.0% |

hi-IN loses **88.5%** of its whitespace tokens to the default pattern and 2.6% of its utterances end up with no word features at all, despite having *more* whitespace tokens per utterance (7.59) than English. th-TH loses 80.2%. Every other locale measured loses at most 17.7%.

This also explains why th-TH degraded only 10.0 points instead of collapsing: MASSIVE's Thai carries artificial spacing (3.35 whitespace tokens per utterance -- the paper itself notes Thai spacing is optional and that models learn from the artificial spacing around slot boundaries), so Thai is hurt mainly by mechanism 2, not mechanism 1. **Our prediction that th-TH would collapse for want of whitespace was wrong on both the size and the reason.**

Practical consequence: the `char_wb` half is load-bearing and the union is not decoration -- dropping the char half costs 60.1 points on zh-CN and 23.9 on hi-IN.

### 3a. Testing the fix, instead of asserting it

Mechanism 2 predicts a word-only TF-IDF should recover once the `token_pattern` admits combining marks. Rather than leave that as a hypothesis, it was run: `token_pattern=r"[^\s]{2,}"` (split on whitespace only), deliberately crude, with unaffected locales as controls.

| locale | word-only, default pattern | word-only, mark-aware | delta | char-only | role |
|---|---|---|---|---|---|
| hi-IN | 0.5797 | 0.8117 | +23.2 | 0.8211 | affected |
| th-TH | 0.7095 | 0.6362 | -7.3 | 0.8090 | affected |
| en-US | 0.8278 | 0.8302 | +0.2 | 0.8376 | control |
| ru-RU | 0.7915 | 0.7902 | -0.1 | 0.8208 | control |
| zh-CN | 0.1459 | 0.1446 | -0.1 | 0.7599 | control |

**hi-IN: confirmed.** The word half recovers +23.2 points, from 0.5797 to 0.8117, essentially reaching the char-only score (0.8211), and the feature count goes from 14,113 to 35,640. The Hindi deficit was a vectorizer default, not a property of Hindi. The three controls move -0.1 to +0.2 points, so the effect is specific.

**th-TH: refuted, in the opposite direction.** The mark-aware pattern makes Thai *worse*, -7.3 points. The reason is that Thai whitespace chunks span several words, so the default pattern's mark-splitting was accidentally acting as a crude syllable segmenter, and removing it hands the model longer, rarer units. We would have reported "combining marks break abugida scripts" as a single clean story; it is true for Devanagari and backwards for Thai.

So the honest recommendation is narrower than the tidy version: for Devanagari, fix the `token_pattern`; for Thai and for the no-whitespace scripts, keep the `char_wb` half, which is what carries them either way.

## 4. The encoder arm: a pre-registered prediction that was REFUTED

Prediction recorded in `massive_arm.py` before the run: ettin-68m would hold on en-US, degrade on Latin non-English, and **collapse on non-Latin scripts** because its tokenizer shatters unseen scripts into byte pieces. A multilingual encoder was run alongside to avoid the BLURB error of blaming the task for a bad model choice.

| locale | ettin-68m (EN) FT | e5-small (multi) FT | best cheap arm | ettin - cheap | pub XLM-R | ettin pieces/char |
|---|---|---|---|---|---|---|
| en-US | 0.8803 | 0.8255 | 0.8440 | +3.6 | 0.8830 | 0.265 |
| de-DE | 0.7942 | 0.7283 | 0.8289 | -3.5 | 0.8570 | 0.372 |
| zh-CN * | 0.8191 | 0.7646 | 0.8073 | +1.2 | 0.8490 | 1.434 |
| th-TH * | 0.7744 | 0.7091 | 0.8043 | -3.0 | 0.8470 | 1.198 |

**The prediction was wrong, and it was wrong in the direction that matters -- we predicted a collapse and there wasn't one.** On the non-Latin locales the English encoder scores zh-CN 0.8191, th-TH 0.7744, and it beats the best cheap arm on 1 of 2 of them (zh-CN). On zh-CN specifically, ettin is the strongest arm we ran -- 0.8191 against 0.8073 frozen multilingual and 0.7465 TF-IDF -- despite needing 1.434 pieces per character against 0.265 on English.

**Tokenizer fertility does not predict accuracy, which is the substantive correction.** The fertility probe was a good mechanism story and a bad predictor: ettin's vocabulary has no Chinese subwords and falls back to byte pieces at 0% UNK, and 11,514 training examples are apparently enough to learn intents over those byte pieces. Meanwhile the locale where ettin actually underperforms its own cheap arms is **de-DE** (0.7942 against 0.8289) -- Latin script, high-resource, the case the prediction said would be mildest. We have no mechanism for that and are not going to invent one; with +-1.5 points of resolution it is a real gap, not noise, but one run per locale cannot separate it from seed variance in the fine-tune.

What survives from the prediction: nothing about scripts. What survives as a usable rule: **the encoder arm is locale-idiosyncratic and has to be measured per locale**, which is the project's general claim rather than a new one.

### 4a. Unfavourable result, reported plainly: the multilingual fine-tune is undertrained, so it cannot carry the claim we wanted from it

**The fine-tuned `multilingual-e5-small` row is not a valid measure of that model's capability, and we are not going to present it as one.** On 4 of 4 encoder locales (en-US, de-DE, zh-CN, th-TH) the FROZEN version of the same checkpoint with a logistic-regression head scores *higher* than the fine-tuned version. On en-US that is 0.8362 frozen against 0.8255 fine-tuned. Fine-tuning a model cannot genuinely be worse than freezing it and fitting a linear head on top, so this is an optimisation failure, not a capability finding.

The training loss shows it directly, on en-US, same protocol (3 epochs, lr 5e-5, batch 16, OneCycleLR):

| epoch | ettin-68m loss | e5-small loss |
|---|---|---|
| 1 | 1.6545 | 3.5336 |
| 2 | 0.3226 | 2.2552 |
| 3 | 0.0774 | 1.6796 |

ettin converges; e5-small is still at 1.68 when the schedule ends. The protocol this project fixed (3 epochs at lr 5e-5) was calibrated on the 68M English encoder and does not converge a 118M model whose 250k-row embedding table is most of its parameters. **Consequence: this arm does NOT establish that a multilingual encoder recovers the off-Latin gap.** That remains untested, and it would need a longer schedule or a higher learning rate to test. We are flagging it rather than quietly reporting 0.8255 as "what multilingual-e5 can do".

### 4b. The fertility probe, kept because it is what misled us

Subword pieces per character, all 13 locales, measured without training anything. ettin's vocabulary has no subwords for most of these scripts, so characters fall back to byte pieces. **UNK rate is 0.00% everywhere**, so nothing is dropped -- the text is merely represented less compactly. The table is kept in the report because it is a clean measurement that produced a *wrong* prediction, and that is more useful to record than to delete.

| locale | ettin-68m pieces/char | e5-small pieces/char | ratio |
|---|---|---|---|
| en-US | 0.265 | 0.279 | 0.95x |
| de-DE | 0.372 | 0.281 | 1.32x |
| es-ES | 0.371 | 0.271 | 1.37x |
| fr-FR | 0.375 | 0.302 | 1.24x |
| ru-RU | 0.604 | 0.283 | 2.14x |
| zh-CN | 1.434 | 0.728 | 1.97x |
| ja-JP | 1.029 | 0.597 | 1.72x |
| ko-KR | 2.208 | 0.753 | 2.93x |
| ar-SA | 0.831 | 0.369 | 2.25x |
| hi-IN | 1.201 | 0.337 | 3.56x |
| th-TH | 1.198 | 0.326 | 3.68x |
| sw-KE | 0.469 | 0.303 | 1.55x |
| am-ET | 3.179 | 0.610 | 5.22x |

ettin needs 3.179 pieces per character on am-ET against 0.265 on en-US -- a 12x inflation -- while multilingual e5 stays within 0.27-0.75 across all 13.

## 5. Cascade pre-flight gate

Run on the row we would actually **ship** per locale (the most accurate arm available), because scoring the gate on the weakest row reversed a verdict in this project once. Gates: AUROC(confidence -> correct) >= ~0.75, and share of all errors in the least-confident 20% well above the 20% that random selection returns. **Calibration gap is deliberately not used** -- ranking correctness and being calibrated are different properties.

Reference passing rows elsewhere in this project: banking77 0.905 / 80%, ToxicChat 0.902 / 84%, CUAD 0.797 / 49%.

| locale | ship arm | accuracy | AUROC | errors in least-conf 20% | gate |
|---|---|---|---|---|---|
| en-US | ettin-68m (EN) FT | 0.8803 | 0.908 | 79.8% | PASS |
| de-DE | TF-IDF union | 0.8289 | 0.856 | 59.1% | PASS |
| es-ES | TF-IDF union | 0.8396 | 0.840 | 58.5% | PASS |
| fr-FR | TF-IDF union | 0.8376 | 0.859 | 59.8% | PASS |
| ru-RU | TF-IDF union | 0.8208 | 0.863 | 60.4% | PASS |
| zh-CN | ettin-68m (EN) FT | 0.8191 | 0.880 | 61.9% | PASS |
| ja-JP | frozen e5 | 0.8178 | 0.839 | 56.8% | PASS |
| ko-KR | TF-IDF union | 0.8093 | 0.853 | 58.9% | PASS |
| ar-SA | TF-IDF union | 0.7794 | 0.850 | 53.8% | PASS |
| hi-IN | TF-IDF union | 0.8184 | 0.855 | 59.4% | PASS |
| th-TH | TF-IDF union | 0.8043 | 0.828 | 49.1% | PASS |
| sw-KE | TF-IDF union | 0.8278 | 0.869 | 61.9% | PASS |
| am-ET | TF-IDF union | 0.7976 | 0.842 | 54.5% | PASS |

AUROC spans 0.828-0.908 (median 0.855); error recall in the low fifth spans 49.1%-79.8% (median 59.1%).

**Escalation dial** for the 13 passing locales, framed as **coverage/cost** and not as accuracy: this project measured out-of-fold that a cascade never beats the better base model, so these are claims about what needs no escalation, nothing more.

| locale | 0% escalated | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| en-US | 0.8803 | 0.9365 | 0.9697 | 0.9817 | 0.9871 |
| de-DE | 0.8289 | 0.8782 | 0.9126 | 0.9448 | 0.9608 |
| es-ES | 0.8396 | 0.8831 | 0.9168 | 0.9457 | 0.9580 |
| fr-FR | 0.8376 | 0.8879 | 0.9185 | 0.9476 | 0.9641 |
| ru-RU | 0.8208 | 0.8741 | 0.9113 | 0.9424 | 0.9613 |
| zh-CN | 0.8191 | 0.8734 | 0.9138 | 0.9481 | 0.9664 |
| ja-JP | 0.8178 | 0.8655 | 0.9016 | 0.9280 | 0.9496 |
| ko-KR | 0.8093 | 0.8599 | 0.9021 | 0.9323 | 0.9563 |
| ar-SA | 0.7794 | 0.8282 | 0.8726 | 0.9116 | 0.9378 |
| hi-IN | 0.8184 | 0.8700 | 0.9079 | 0.9356 | 0.9568 |
| th-TH | 0.8043 | 0.8487 | 0.8756 | 0.9073 | 0.9417 |
| sw-KE | 0.8278 | 0.8767 | 0.9180 | 0.9496 | 0.9636 |
| am-ET | 0.7976 | 0.8461 | 0.8848 | 0.9131 | 0.9406 |

Read a row as: escalate the least-confident X% to something better, and the remaining (100-X)% is answered for free at the stated accuracy.

Expressed the way this project states the CUAD result -- **how much of the traffic is answerable for free at 90% reliability** (keep the most-confident slice whose accuracy is still >=0.90):

| locale | ship arm | free coverage at >=90% accuracy | accuracy on that slice |
|---|---|---|---|
| en-US | ettin-68m (EN) FT | 96.6% | 0.9001 |
| de-DE | TF-IDF union | 83.3% | 0.9002 |
| es-ES | TF-IDF union | 85.0% | 0.9003 |
| fr-FR | TF-IDF union | 86.4% | 0.9000 |
| ru-RU | TF-IDF union | 84.0% | 0.9003 |
| zh-CN | ettin-68m (EN) FT | 84.2% | 0.9001 |
| ja-JP | frozen e5 | 80.7% | 0.9000 |
| ko-KR | TF-IDF union | 80.6% | 0.9003 |
| ar-SA | TF-IDF union | 73.0% | 0.9001 |
| hi-IN | TF-IDF union | 82.0% | 0.9000 |
| th-TH | TF-IDF union | 71.7% | 0.9001 |
| sw-KE | TF-IDF union | 84.7% | 0.9000 |
| am-ET | TF-IDF union | 75.7% | 0.9001 |

Free coverage at 90% reliability spans **72% to 97%** (median 83%), against CUAD's 76% which is the project's existing reference point for a genuine middle. The weakest are th-TH (72%) and the strongest en-US (97%). **This is a coverage/cost claim only.** It says what fraction needs no escalation; it does not say the cascade beats the escalation target, and this project measured out-of-fold that it does not.

## 5a. Cross-lingual zero-shot: a split decision, not a win

One classifier, frozen `multilingual-e5-small` + logistic regression, trained **once on en-US train** and applied unchanged to 12 other languages. Nothing is refit per locale. This is the only setting on MASSIVE where a published figure and a free arm share a protocol, because the paper reports zero-shot columns (train en-US, test elsewhere) in the same Table 8.

It is leak-safe for a measured reason, not an assumed one: the official partition is exactly id-aligned across locales, so en-US train ids and locale-xx test ids are disjoint by construction. The script asserts it per locale.

| locale | free zero-shot | pub XLM-R zero-shot | gap | pub mT5-T2T zero-shot | free supervised | zero-shot cost |
|---|---|---|---|---|---|---|
| en-US | 0.8362 | - | - | - | 0.8362 | +0.0 |
| de-DE | 0.7004 | 0.7760 | -7.6 | 0.7730 | 0.7989 | -9.9 |
| es-ES | 0.7394 | 0.7880 | -4.9 | 0.7660 | 0.8124 | -7.3 |
| fr-FR | 0.7609 | 0.8080 | -4.7 | 0.7690 | 0.8171 | -5.6 |
| ru-RU | 0.7276 | 0.8130 | -8.5 | 0.7620 | 0.8204 | -9.3 |
| zh-CN | 0.7256 | 0.6190 | +10.7 | 0.5570 | 0.8073 | -8.2 |
| ja-JP | 0.7458 | 0.4480 | +29.8 | 0.2570 | 0.8178 | -7.2 |
| ko-KR | 0.6137 | 0.7700 | -15.6 | 0.6000 | 0.7952 | -18.2 |
| ar-SA | 0.5541 | 0.6280 | -7.4 | 0.5900 | 0.7313 | -17.7 |
| hi-IN | 0.7451 | 0.7480 | -0.3 | 0.6240 | 0.8040 | -5.9 |
| th-TH | 0.6843 | 0.7740 | -9.0 | 0.7280 | 0.7774 | -9.3 |
| sw-KE | 0.5131 | 0.4660 | +4.7 | 0.4560 | 0.7424 | -22.9 |
| am-ET | 0.4734 | 0.5190 | -4.6 | 0.3680 | 0.7135 | -24.0 |

**Predicted a win, got a split decision -- reported as such.** The pre-registered guess (in `massive_zeroshot.py`) was that this would be the one setting where the free arm beats a published baseline. It beats published XLM-R-Base zero-shot on **3 of 12** non-English locales and **loses on 9**:

- Wins: **ja-JP +29.8**, **zh-CN +10.7**, **sw-KE +4.7** points.
- Losses: ko-KR -15.6, th-TH -9.0, ru-RU -8.5, de-DE -7.6, ar-SA -7.4, es-ES -4.9, fr-FR -4.7, am-ET -4.6, hi-IN -0.3 points.

That headline gap is against **XLM-R-Base**, which is the stronger of the paper's two zero-shot models and therefore the conservative comparison. Against **mT5-Base T2T** zero-shot the free arm wins on 6 of 12 (zh-CN, ja-JP, ko-KR, hi-IN, sw-KE, am-ET) -- so the verdict depends on which published model you pick, and we are quoting the one that flatters us least.

The wins are concentrated where published zero-shot falls off a cliff -- ja-JP, where XLM-R zero-shot manages only 0.4480, and zh-CN at 0.6190. Both are no-whitespace scripts, where a model fine-tuned on English word structure transfers worst and a sentence embedder trained for cross-lingual alignment transfers best. The losses are concentrated on locales where XLM-R zero-shot is already strong (ko-KR 0.7700, ru-RU 0.8130); there is no headroom to take and the small frozen embedder does not have it.

Two things limit this either way: it is a published comparison, not a paired same-items run, and zero-shot is well below supervised on every locale -- -24.0 to -5.6 points, worst on the low-resource pair (sw-KE -22.9, am-ET -24.0). **Zero-shot is the cheap option, not the good one**, and the honest summary is that the free zero-shot arm is competitive with published zero-shot rather than better than it.

## 6. What we can and cannot claim

**Can claim:**

- The free arm is **not** broken by non-English input per se. Across 13 locales spanning 9 writing systems, best-arm accuracy lands in a 10.1-point band, 11 of 13 clear 0.80, and every locale beats a majority-class predictor by 71+ points. Script and resource level did not produce a cliff.
- The `char_wb` half of the TF-IDF union is load-bearing on four locales (zh-CN, ja-JP, th-TH, hi-IN), worth 10-63 points over the word half. Two of those four were predicted in advance and two were not.
- **A cheap arm can be chosen badly in a way only measurement reveals.** Char-ngram TF-IDF beats a frozen multilingual embedder on 11 of 13 locales, by up to 8.4 points (am-ET), and sklearn's default `token_pattern` costs hi-IN 23 points. Neither is visible without running the comparison.
- The confidence gate passes on all 13 locales, giving 72-97% free coverage at 90% reliability. That is a coverage result, not an accuracy one.
- The official MASSIVE split is id-aligned across locales and safe to use per-locale. That is a finding about MASSIVE, not a property to assume of the next parallel corpus.

**Cannot claim:**

- **Cannot claim the CHEAP arms match fine-tuned multilingual baselines.** TF-IDF and frozen embeddings are behind published XLM-R-Base on 13/13 locales by 0.3-5.6 points. The cheap arms are a breadth instrument here, not a competitive row.
- **Cannot claim the fine-tuned tie generalises beyond the locales it was measured on.** It is a tie on 1 of the 4 locales where an encoder was run (en-US), and encoders were run on only 4 of 51 locales. Extrapolating it to the other 47 is exactly the move this project's SCOPE file forbids.
- **Cannot claim this is a paired comparison.** The mT5/XLM-R figures are the authors' own protocol on the same test split but a different run; no per-item incumbent scores exist here. This arm has **no paired baseline**, like CUAD.
- **Cannot claim the English encoder is a bad choice off English** -- we predicted that and it is refuted. ettin-68m was the best arm we ran on zh-CN. What we can say is that it is *unpredictable* off English: best of any arm on zh-CN, below its own TF-IDF baseline on de-DE and th-TH.
- **Cannot claim anything about whether a properly-trained multilingual encoder closes the gap.** Our control did not converge (section 4a).
- Cannot claim anything about the 39 locale directories not modelled (listed below).
- Cannot make fine per-intent claims. 6 intents have <=10 test items and one has 1, so per-intent resolution is ~10-20 points for the tail and macro-F1 is dominated by it. Intent accuracy is the reliable metric.
- Cannot claim a cascade beats an LLM anywhere. The dial is cost/coverage, and no LLM was run on this suite at all.

## 7. What was not run, explicitly

- **39 of the 52 locale directories were not modelled at all**: af-ZA, az-AZ, bn-BD, ca-ES, cy-GB, da-DK, el-GR, fa-IR, fi-FI, he-IL, hu-HU, hy-AM, id-ID, is-IS, it-IT, jv-ID, ka-GE, km-KH, kn-IN, lv-LV, ml-IN, mn-MN, ms-MY, my-MM, nb-NO, nl-NL, pl-PL, pt-PT, ro-RO, sl-SL, sq-AL, sv-SE, ta-IN, te-IN, tl-PH, tr-TR, ur-PK, vi-VN, zh-TW. The contamination and id-alignment audit does cover all 52; only the four arms are restricted to 13.
- Fine-tuned encoders ran on 4 locales (en-US, de-DE, zh-CN, th-TH), not 13. The other 9 locales have cheap arms only, so their "best free" column is a floor, not a ceiling.
- The cross-lingual zero-shot arm (section 5a) was run on the frozen embedder only. No zero-shot encoder fine-tune was run, so we cannot say whether a fine-tuned encoder transfers better or worse than the frozen one.
- Only one seed (SEED=0) and one run per (locale, arm). The de-DE encoder gap in particular cannot be separated from fine-tuning seed variance.
- No hyperparameter search of any kind. The protocol was fixed in advance, which is what made the e5 non-convergence in 4a visible rather than tuned away.
- `ca-ES` ships on the parquet branch (52 locale dirs) but is absent from the paper's 51-language Table 8, so it has no published baseline; it was audited but not modelled.
- No paid API call was made, so there is no GPT-class comparison on this suite at all.

## 8. Verification

- The published-baseline parse was checked against an independent figure in the same paper: mean XLM-R-full over the 51 parsed locales is 85.11 against the 85.1 the paper reports in Table 3a, and mean mT5-T2T-full is 85.33 against 85.3. The row alignment was additionally asserted by checking en-US carries exactly 3 populated cells (no zero-shot entry, since it is the zero-shot training language).
- Headline accuracies were re-derived a second way in `massive_verify.py`, and all four checks passed exactly:
  1. Every one of the **73 (locale, arm) accuracies** recomputed from the persisted per-item `{pred, gold}` arrays with a hand-rolled mean instead of sklearn. Max absolute difference **0.00e+00**.
  2. en-US TF-IDF-union **refitted from scratch** with the FeatureUnion built in the opposite order: 0.843981 refit against 0.843981 stored, and **100.0000% per-item prediction agreement**. That is both a second derivation and a determinism check under SEED=0.
  3. The gate's "errors in the least-confident 20%" re-derived by a different route (quantile threshold with explicit tie handling, instead of the lexsort used in the runner) for all 13 ship rows -- identical to 6 decimal places on every locale.
  4. The persisted `gold` array checked against the parquet `intent` column for all 13 locales: identical, so nothing was reordered in persistence.
- The default-pattern refits in `massive_tokenfix.py` independently reproduced the stored `tfidf_word` accuracies for all five locales it touched, which is a third determinism check.
- Per-item `pred`, `gold` and `conf` are persisted in `massive.json` for **every arm and every locale run**, including both fine-tuned encoders, so this arm does not inherit the project's standing limitation of charts stuck on the TF-IDF row.

