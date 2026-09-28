# Class-Gated Deferral (CGD): Spending LLM Calls Only Where the LLM Is Better

Vasyl Rakivnenko, … · September 2026 · **[One-pager (PDF)](CGD_one_pager.pdf)**

## Abstract

We introduce class-gated deferral (CGD), to our knowledge the first cascade algorithm that decides class by class, which model should answer: a computationally 'free' (CPU-run) NLP classifier trained on the task's labels, or a large language model (LLM). For every class the classifier can predict, CGD checks which of the two models is more accurate on held-out data and hands that whole class to the winner, so LLM calls are spent only where the LLM is actually better. Uncertainty-based cascades, such as Online Cascade Learning [5], work differently: they send the LLM whatever the free model is least sure about, even when the LLM is worse on those items. On LegalBench [1], across nine task–LLM combinations, CGD beats both the free classifier and the LLM in seven, and it beats a confidence cascade with the same LLM budget in six, by up to 17 points of balanced accuracy. On CUAD [3] contract clauses with a production API LLM, CGD again beats both models and outperforms a same-budget cascade by 4.8–8.1 points. On Financial PhraseBank [4], it keeps the majority neutral class on the free classifier, cuts LLM calls by 60%, and beats both components in 16 of 28 LLM setups. The gain grows with how class-dependent the LLM's competence is (*r* = 0.57 across 59 pairs), which suggests a simple test for when to gate by class.

**Table 1.** LegalBench [1], balanced accuracy (LegalBench's metric). *Confidence cascade* gives the LLM the same share of items as CGD, escalating the free classifier's least-confident ones. opp115_data_retention comes from the OPP-115 privacy-policy corpus [2].

| Task | LLM | 'Free' (CPU run) NLP alone | LLM alone | Confidence cascade (same budget) | CGD | Δ vs better | LLM-call share | Outcome |
|---|---|--:|--:|--:|--:|--:|--:|:-:|
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b | 74.5% | 72.3% | 72.2% | **82.4%**<sup>\*†</sup> | +7.9pp | 76% | Win (CGD) |
| supply_chain_disclosure_best_practice_audits | deepseek-v4-pro | 74.5% | 65.0% | 65.7% | **82.1%**<sup>\*†</sup> | +7.7pp | 76% | Win (CGD) |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 74.5% | 62.9% | 63.6% | **80.8%**<sup>\*†</sup> | +6.4pp | 76% | Win (CGD) |
| opp115_data_retention | glm-5.3-flash | 79.7% | 78.6% | 83.1% | **81.6%**<sup>\*</sup> | +1.9pp | 51% | Win (CGD) |
| opp115_data_retention | kimi-k2.6 | 79.7% | 75.7% | 80.8% | **81.5%**<sup>\*</sup> | +1.8pp | 50% | Win (CGD) |
| opp115_data_retention | gpt-oss-120b | 79.7% | 73.0% | 80.2% | **81.4%**<sup>\*</sup> | +1.6pp | 50% | Win (CGD) |
| learned_hands_consumer | gpt-oss-120b | 84.3% | 71.5% | 80.2% | **85.2%**<sup>\*†</sup> | +0.9pp | 50% | Win (CGD) |
| learned_hands_consumer | deepseek-v4-pro | 84.3% | 81.9% | 84.8% | 84.4% | +0.1pp | 42% | Tie |
| learned_hands_consumer | glm-5.3-flash | 84.3% | 87.5% | 87.5% | 87.4% | 0.0pp | 100% | Tie |

**Table 2.** CUAD [3] contract clauses with a production API LLM. Accuracy (the test sets are class-balanced, so this equals balanced accuracy).

| Task | LLM | 'Free' (CPU run) NLP alone | LLM alone | Confidence cascade (same budget) | CGD | Δ vs better | LLM-call share | Outcome |
|---|---|--:|--:|--:|--:|--:|--:|:-:|
| cuad_covenant_not_to_sue | Jev (TypeSafe API) | 94.5% | 81.8% | 86.9% | **95.0%**<sup>†</sup> | +0.5pp | 50% | Win (CGD) |
| cuad_change_of_control | Jev (TypeSafe API) | 90.6% | 82.5% | 87.9% | **92.7%**<sup>\*†</sup> | +2.1pp | 49% | Win (CGD) |

**Table 3.** Financial PhraseBank [4], accuracy. CGD keeps the neutral class on the free classifier and hands positive and negative items to the LLM (40% of calls). Six of 28 LLM setups shown; CGD beats both components in 16 of the 28.

| Task | LLM | 'Free' (CPU run) NLP alone | LLM alone | Confidence cascade (same budget) | CGD | Δ vs better | LLM-call share | Outcome |
|---|---|--:|--:|--:|--:|--:|--:|:-:|
| financial_phrasebank | grok-4-1-fast-non-reasoning | 91.2% | 90.0% | 94.0% | **96.8%**<sup>\*</sup> | +5.6pp | 40% | Win (CGD) |
| financial_phrasebank | gpt-5-mini | 91.2% | 92.8% | 97.0% | **97.2%**<sup>\*</sup> | +4.4pp | 40% | Win (CGD) |
| financial_phrasebank | gpt-5.4-nano | 91.2% | 92.8% | 96.2% | **96.8%**<sup>\*</sup> | +4.0pp | 40% | Win (CGD) |
| financial_phrasebank | qwen3-1.7b-reasoning | 91.2% | 78.0% | 88.2% | **95.2%**<sup>\*†</sup> | +4.0pp | 40% | Win (CGD) |
| financial_phrasebank | mistral-large-3 | 91.2% | 92.8% | 95.0% | **95.6%** | +2.8pp | 40% | Win (CGD) |
| financial_phrasebank | gpt-5-nano | 91.2% | 94.8% | 96.8% | **97.2%** | +2.4pp | 40% | Win (CGD) |

<sub>**'Free' (CPU run) NLP alone**: a classifier trained on each task's labeled data (5-fold cross-validation for LegalBench and CUAD; a separate training split for Financial PhraseBank) that runs on CPU. **CGD** decides which predicted classes go to the LLM from both models' accuracy on the training folds; all routing decisions are out-of-fold, averaged over 10 fold splits (×3 cross-validation repeats for LegalBench and CUAD). <sup>\*</sup> p < 0.05 vs. the better single model; <sup>†</sup> p < 0.05 vs. the same-budget cascade (paired McNemar test for accuracy, paired bootstrap for balanced accuracy). *Tie*: within 0.5 points.</sub>

### References

[1] N. Guha, J. Nyarko, D. E. Ho, C. Ré, et al. "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models." arXiv:2308.11462, 2023.

[2] S. Wilson, F. Schaub, A. A. Dara, F. Liu, S. Cherivirala, P. G. Leon, et al. "The Creation and Analysis of a Website Privacy Policy Corpus." ACL 2016, pp. 1330–1340.

[3] D. Hendrycks, C. Burns, A. Chen, S. Ball. "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." NeurIPS 2021 Datasets and Benchmarks Track.

[4] P. Malo, A. Sinha, P. Korhonen, J. Wallenius, P. Takala. "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts." Journal of the Association for Information Science and Technology 65(4):782–796, 2014.

[5] L. Nie, Z. Ding, E. Hu, C. Jermaine, S. Chaudhuri. "Online Cascade Learning for Efficient Inference over Streams." ICML 2024, PMLR 235:38071–38090.
