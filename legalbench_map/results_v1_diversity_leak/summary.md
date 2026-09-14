# LegalBench Cheap-Model-Map -- Summary

One row per task, showing the BEST cheap candidate for that task (all numbers below are CV-estimated unless marked estimated cascade), sorted by estimated LLM share ascending (tasks needing the LLM least come first).

| task | best candidate | metric | CV-estimated mean | 95% CI | keep rate | est. LLM share | estimated cascade | published best | source | gap (pts) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuad_audit_rights | tfidf_logreg | balanced_accuracy | 0.986 | [0.982, 0.990] | 1.000 | 0.000 | 0.985 | 0.979 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | -0.67 | match |
| cuad_cap_on_liability | tfidf_logreg | balanced_accuracy | 0.971 | [0.966, 0.976] | 1.000 | 0.000 | 0.967 | 0.956 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | -1.54 | match |
| diversity_1 | embed_small_logreg | balanced_accuracy | 0.999 | [0.996, 1.000] | 1.000 | 0.000 | 0.999 | 1.000 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 0.15 | match |
| diversity_2 | embed_small_logreg | balanced_accuracy | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | 1.000 | 0.998 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | -0.20 | match |
| diversity_3 | embed_small_logreg | balanced_accuracy | 0.999 | [0.997, 1.000] | 1.000 | 0.000 | 0.996 | 0.970 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | -2.91 | match |
| diversity_4 | embed_small_logreg | balanced_accuracy | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | 1.000 | 1.000 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 0.00 | match |
| diversity_5 | embed_small_logreg | balanced_accuracy | 1.000 | [1.000, 1.000] | 1.000 | 0.000 | 0.997 | 0.932 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | -6.80 | match |
| cuad_anti-assignment | tfidf_logreg | balanced_accuracy | 0.964 | [0.958, 0.970] | 0.992 | 0.008 | 0.968 | 0.924 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | -3.99 | match |
| cuad_expiration_date | tfidf_logreg | balanced_accuracy | 0.962 | [0.954, 0.969] | 0.992 | 0.008 | 0.962 | 0.870 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | -9.16 | match |
| overruling | tfidf_logreg | balanced_accuracy | 0.961 | [0.957, 0.966] | 0.991 | 0.009 | 0.959 | 0.954 | https://arxiv.org/pdf/2308.11462 (Table 72, page 127) | -0.71 | match |
| learned_hands_family | embed_base_logreg | balanced_accuracy | 0.947 | [0.942, 0.952] | 0.983 | 0.017 | 0.949 | 0.862 | https://arxiv.org/pdf/2308.11462 (Table 64, page 123) | -8.47 | match |
| learned_hands_employment | embed_small_logreg | balanced_accuracy | 0.960 | [0.950, 0.968] | 0.982 | 0.018 | 0.961 | 0.742 | https://arxiv.org/pdf/2308.11462 (Table 65, page 123) | -21.76 | match |
| learned_hands_housing | embed_base_logreg | balanced_accuracy | 0.944 | [0.940, 0.948] | 0.979 | 0.021 | 0.948 | 0.850 | https://arxiv.org/pdf/2308.11462 (Table 64, page 123) | -9.41 | match |
| cuad_covenant_not_to_sue | tfidf_logreg | balanced_accuracy | 0.965 | [0.955, 0.976] | 0.944 | 0.056 | 0.976 | 0.958 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | -0.74 | match |
| opp115_international_and_specific_audiences | embed_base_logreg | balanced_accuracy | 0.943 | [0.933, 0.953] | 0.914 | 0.086 | 0.958 | 0.923 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | -2.03 | match |
| supply_chain_disclosure_disclosed_accountability | embed_base_logreg | balanced_accuracy | 0.697 | [0.636, 0.761] | 0.884 | 0.116 | 0.762 | 0.807 | https://arxiv.org/pdf/2308.11462 (Table 77, page 135-136) | 10.95 | short |
| diversity_6 | embed_small_logreg | balanced_accuracy | 0.960 | [0.946, 0.973] | 0.856 | 0.144 | 0.953 | 0.905 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | -5.47 | match |
| cuad_change_of_control | tfidf_logreg | balanced_accuracy | 0.921 | [0.906, 0.936] | 0.800 | 0.200 | 0.941 | 0.897 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | -2.45 | match |
| opp115_policy_change | tfidf_logreg | balanced_accuracy | 0.917 | [0.901, 0.932] | 0.792 | 0.208 | 0.952 | 0.919 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 0.19 | close |
| cuad_exclusivity | embed_base_logreg | balanced_accuracy | 0.881 | [0.868, 0.895] | 0.768 | 0.232 | 0.933 | 0.929 | https://arxiv.org/pdf/2308.11462 (Table 76, page 130) | 4.75 | short |
| opp115_third_party_sharing_collection | tfidf_logreg | balanced_accuracy | 0.865 | [0.857, 0.873] | 0.737 | 0.263 | 0.898 | 0.801 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | -6.36 | match |
| opp115_data_security | tfidf_logreg | balanced_accuracy | 0.875 | [0.861, 0.889] | 0.736 | 0.264 | 0.914 | 0.875 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 0.01 | close |
| opp115_user_access,_edit_and_deletion | tfidf_logreg | balanced_accuracy | 0.903 | [0.887, 0.919] | 0.699 | 0.301 | 0.932 | 0.902 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | -0.14 | close |
| opp115_first_party_collection_use | embed_base_logreg | balanced_accuracy | 0.848 | [0.840, 0.855] | 0.688 | 0.312 | 0.891 | 0.806 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | -4.18 | match |
| learned_hands_crime | embed_base_logreg | balanced_accuracy | 0.886 | [0.873, 0.900] | 0.688 | 0.312 | 0.911 | 0.830 | https://arxiv.org/pdf/2308.11462 (Table 65, page 123) | -5.61 | match |
| learned_hands_consumer | embed_small_logreg | balanced_accuracy | 0.850 | [0.833, 0.867] | 0.627 | 0.373 | 0.874 | 0.762 | https://arxiv.org/pdf/2308.11462 (Table 64, page 123) | -8.76 | match |
| learned_hands_torts | embed_base_logreg | balanced_accuracy | 0.852 | [0.833, 0.870] | 0.621 | 0.379 | 0.845 | 0.706 | https://arxiv.org/pdf/2308.11462 (Table 64, page 123) | -14.59 | match |
| supply_chain_disclosure_best_practice_accountability | embed_base_logreg | balanced_accuracy | 0.739 | [0.700, 0.779] | 0.603 | 0.397 | 0.785 | 0.746 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 0.73 | close |
| opp115_data_retention | embed_base_logreg | balanced_accuracy | 0.812 | [0.786, 0.839] | 0.468 | 0.532 | 0.812 | 0.705 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | -10.75 | match |
| supply_chain_disclosure_best_practice_training | tfidf_logreg | balanced_accuracy | 0.666 | [0.635, 0.697] | 0.442 | 0.558 | 0.827 | 0.871 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 20.45 | short |
| supply_chain_disclosure_best_practice_audits | tfidf_logreg | balanced_accuracy | 0.759 | [0.733, 0.787] | 0.421 | 0.579 | 0.795 | 0.766 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 0.70 | close |
| unfair_tos | embed_base_logreg | balanced_accuracy | 0.890 | [0.873, 0.906] | 0.367 | 0.633 | 0.623 | 0.153 | https://arxiv.org/pdf/2308.11462 (Table 79, page 143) | -73.68 | match |
| supply_chain_disclosure_best_practice_verification | tfidf_logreg | balanced_accuracy | 0.706 | [0.681, 0.731] | 0.348 | 0.652 | 0.758 | 0.705 | https://arxiv.org/pdf/2308.11462 (Table 77, page 135-136) | -0.10 | close |
| supply_chain_disclosure_best_practice_certification | tfidf_logreg | balanced_accuracy | 0.601 | [0.574, 0.628] | 0.332 | 0.668 | 0.769 | 0.777 | https://arxiv.org/pdf/2308.11462 (Table 76, page 132) | 17.63 | short |
| abercrombie | embed_base_logreg | balanced_accuracy | 0.379 | [0.329, 0.428] | 0.000 | 1.000 | n/a | 0.853 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 47.41 | short |
| hearsay | tfidf_logreg | balanced_accuracy | 0.765 | [0.714, 0.819] | 0.000 | 1.000 | n/a | 0.838 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 7.33 | short |
| personal_jurisdiction | tfidf_logreg | balanced_accuracy | 0.533 | [0.458, 0.611] | 0.000 | 1.000 | n/a | 0.914 | https://arxiv.org/pdf/2308.11462 (Table 68, page 125) | 38.14 | short |

**Verdict counts** (best candidate per task, n=37 tasks): match=24, close=6, short=7, no_reference=0.

**CV-vs-published comparability assumption.** Published LLM scores were
computed by their original authors on the fixed LegalBench test split
with their own methodology (typically a single pass, no CV). The
CV-estimated numbers in this report evaluate the same test-split items
via repeated stratified 5-fold cross-validation so every item gets an
out-of-fold prediction. We treat the two as comparable estimates of
"score on the LegalBench test split for this task" -- this is an
assumption, not a guarantee, and every CV-derived number is labeled
"CV-estimated" to keep that assumption visible.

**Estimated-cascade assumption.** `est_cascade_metric` = keep_rate *
metric_on_kept + (1 - keep_rate) * published_best_llm_score. Sent items
are assumed to score at the LLM's published AVERAGE, which may be
optimistic: sent items are exactly the ones the conformal gate flagged
as uncertain, i.e. selected to be harder than a random draw from the
test set. For balanced_accuracy, the estimate is computed per class
(assuming the published task-average score applies uniformly across
classes, since no per-class published number exists) and then averaged.
Every such number is labeled "estimated cascade".
