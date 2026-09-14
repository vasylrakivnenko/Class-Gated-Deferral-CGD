# Gate replay 2 -- committee disagreement and per-class ownership (offline, $0, out-of-fold)

All gates OOF (5-fold over items, rule fitted without the item), mean of 3 cheap repeats. 'ni' = cheapest training-fold point with accuracy >= incumbent-alone; 'maxacc' = training-fold max accuracy. committee = other cheap models (tfidf / bge-small / bge-base / bge-large minus the primary).

## Who is right where the committee disagrees? (the diagnostic that decides it)

| task | incumbent | primary | committee | disagree rate | on DISAGREE: cheap / LLM | on AGREE: cheap / LLM | committee majority vote (no LLM) |
|---|---|---|---|---|---|---|---|
| learned_hands_consumer | deepseek-v4-pro-fireworks | embed_small_logreg (84.3%) | tfidf_logreg, embed_base_logreg, embed_large_logreg | 18.7% | 55.6% / 69.7% | 90.9% / 84.8% | 84.1% |
| learned_hands_consumer | glm-5.3-flash | embed_small_logreg (84.3%) | tfidf_logreg, embed_base_logreg, embed_large_logreg | 18.7% | 55.6% / 72.4% | 90.9% / 90.9% | 84.1% |
| learned_hands_consumer | gpt-oss-120b-fireworks | embed_small_logreg (84.3%) | tfidf_logreg, embed_base_logreg, embed_large_logreg | 18.7% | 55.6% / 59.7% | 90.9% / 74.2% | 84.1% |
| opp115_data_retention | glm-5.3-flash | embed_base_logreg (79.7%) | tfidf_logreg, embed_small_logreg, embed_large_logreg | 35.9% | 55.4% / 68.2% | 93.3% / 84.5% | 83.1% |
| opp115_data_retention | gpt-oss-120b-fireworks | embed_base_logreg (79.7%) | tfidf_logreg, embed_small_logreg, embed_large_logreg | 35.9% | 55.4% / 64.5% | 93.3% / 77.8% | 83.1% |
| opp115_data_retention | kimi-k2.6-fireworks | embed_base_logreg (79.7%) | tfidf_logreg, embed_small_logreg, embed_large_logreg | 35.9% | 55.4% / 66.7% | 93.3% / 80.7% | 83.1% |
| supply_chain_disclosure_best_practice_audits | deepseek-v4-pro-fireworks | tfidf_logreg (80.9%) | embed_small_logreg, embed_base_logreg, embed_large_logreg | 47.9% | 74.3% / 71.9% | 87.0% / 81.8% | 74.1% |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | tfidf_logreg (80.9%) | embed_small_logreg, embed_base_logreg, embed_large_logreg | 47.9% | 74.3% / 71.0% | 87.0% / 80.6% | 74.1% |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | tfidf_logreg (80.9%) | embed_small_logreg, embed_base_logreg, embed_large_logreg | 47.9% | 74.3% / 78.0% | 87.0% / 85.8% | 74.1% |

## Gates: accuracy @ LLM share (OOF)

| task | incumbent | cheap | LLM | conformal | confidence-ni | committee-ni | committee-maxacc | committee+conf-maxacc | per-class any | per-class 2pt | oracle | best gate vs cheap (diff, p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | deepseek-v4-pro-fireworks | 84.3% | 81.9% | 85.0% @ 37.3% | 84.3% @ 0.0% | 84.3% @ 0.0% | 86.9% @ 18.7% | 86.6% @ 21.1% | 84.1% @ 36.3% | 84.0% @ 9.9% | 93.0% @ 15.7% | committee_maxacc: +2.6pp, p=0.040 |
| learned_hands_consumer | glm-5.3-flash | 84.3% | 87.5% | 87.8% @ 37.3% | 86.4% @ 16.0% | 86.6% @ 58.0% | 86.6% @ 58.0% | 87.1% @ 31.0% | 87.5% @ 100.0% | 86.1% @ 73.7% | 93.1% @ 15.7% | perclass_any: +3.1pp, p=0.047 |
| learned_hands_consumer | gpt-oss-120b-fireworks | 84.3% | 71.5% | 81.6% @ 37.3% | 84.3% @ 0.0% | 84.3% @ 0.0% | 84.4% @ 14.5% | 84.6% @ 12.9% | 85.2% @ 50.4% | 84.5% @ 20.2% | 92.4% @ 15.7% | perclass_any: +0.9pp, p=0.062 |
| opp115_data_retention | glm-5.3-flash | 79.7% | 78.6% | 83.0% @ 53.2% | 79.7% @ 0.8% | 80.2% @ 1.0% | 84.4% @ 19.2% | 85.9% @ 22.0% | 81.8% @ 50.3% | 81.8% @ 50.3% | 91.8% @ 20.3% | committee_conf_maxacc: +6.1pp, p=0.001 |
| opp115_data_retention | gpt-oss-120b-fireworks | 79.7% | 73.0% | 80.4% @ 53.2% | 79.7% @ 0.0% | 79.7% @ 0.0% | 84.1% @ 18.0% | 85.3% @ 21.7% | 81.4% @ 50.3% | 80.9% @ 43.5% | 91.3% @ 20.3% | committee_conf_maxacc: +5.6pp, p=0.003 |
| opp115_data_retention | kimi-k2.6-fireworks | 79.7% | 75.7% | 80.9% @ 53.2% | 79.7% @ 0.0% | 79.7% @ 0.0% | 84.5% @ 21.6% | 84.8% @ 20.9% | 81.5% @ 50.3% | 81.1% @ 46.5% | 91.8% @ 20.3% | committee_conf_maxacc: +5.0pp, p=0.008 |
| supply_chain_disclosure_best_practice_audits | deepseek-v4-pro-fireworks | 80.9% | 77.0% | 77.8% @ 57.9% | 80.9% @ 0.0% | 80.9% @ 0.0% | 80.1% @ 9.6% | 80.7% @ 14.2% | 85.0% @ 76.4% | 85.0% @ 76.4% | 91.6% @ 19.1% | perclass_any: +4.0pp, p=0.003 |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 80.9% | 76.0% | 76.3% @ 57.9% | 80.9% @ 0.0% | 80.9% @ 0.0% | 80.2% @ 9.6% | 80.8% @ 14.0% | 84.4% @ 76.4% | 84.4% @ 76.4% | 90.5% @ 19.1% | perclass_any: +3.5pp, p=0.003 |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 80.9% | 82.1% | 80.8% @ 57.9% | 81.8% @ 6.8% | 80.7% @ 23.9% | 82.6% @ 38.9% | 82.8% @ 34.3% | 85.4% @ 76.4% | 85.4% @ 76.4% | 91.7% @ 19.1% | perclass_any: +4.5pp, p=0.001 |