# Gate replay -- can a better gate recover the oracle gap? (offline, $0)

All gate numbers are out-of-fold over items (5-fold, rule fitted without the item), mean of 3 cheap repeats. Incumbent = single temperature-0 pass (lenient scoring). 'gap recovered' = (gate - conformal) / (oracle - conformal). 'ni' = cheapest point whose training-fold accuracy >= incumbent-alone; 'maxacc' = training-fold max accuracy.

| task | incumbent | cheap | LLM | conformal acc / share | threshold-ni acc / share | router-ni acc / share | router-maxacc acc / share | oracle acc / share | gap recovered: thr-ni / router-ni / router-maxacc | router-maxacc vs cheap (diff, p) | router-maxacc vs LLM (diff, p) | $/1k LLM -> router-ni |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_hands_consumer | deepseek-v4-pro-fireworks | 84.3% | 81.9% | 85.0% / 37.3% | 84.3% / 0.0% | 84.2% / 7.0% | 83.9% / 21.4% | 93.0% / 15.7% | -0.09 / -0.10 / -0.14 | -0.4pp, p=0.460 | +2.0pp, p=0.227 | 2.996 -> 0.210 |
| learned_hands_consumer | glm-5.3-flash | 84.3% | 87.5% | 87.8% / 37.3% | 86.4% / 16.0% | 86.0% / 47.1% | 87.2% / 87.1% | 93.1% / 15.7% | -0.26 / -0.34 / -0.10 | +2.9pp, p=0.045 | -0.2pp, p=0.714 | 0.591 -> 0.278 |
| learned_hands_consumer | gpt-oss-120b-fireworks | 84.3% | 71.5% | 81.6% / 37.3% | 84.3% / 0.0% | 85.1% / 7.2% | 84.9% / 10.6% | 92.4% / 15.7% | +0.25 / +0.32 / +0.31 | +0.6pp, p=0.433 | +13.4pp, p=0.000 | 0.193 -> 0.014 |
| opp115_data_retention | glm-5.3-flash | 79.7% | 78.6% | 83.0% / 53.2% | 79.7% / 0.8% | 80.7% / 11.0% | 81.1% / 28.5% | 91.8% / 20.3% | -0.38 / -0.26 / -0.21 | +1.4pp, p=0.418 | +2.5pp, p=0.306 | 0.204 -> 0.022 |
| opp115_data_retention | gpt-oss-120b-fireworks | 79.7% | 73.0% | 80.4% / 53.2% | 79.7% / 0.0% | 80.3% / 10.7% | 80.4% / 29.7% | 91.3% / 20.3% | -0.06 / -0.01 / +0.00 | +0.7pp, p=0.620 | +7.3pp, p=0.033 | 0.158 -> 0.017 |
| opp115_data_retention | kimi-k2.6-fireworks | 79.7% | 75.7% | 80.9% / 53.2% | 79.7% / 0.0% | 80.0% / 9.8% | 79.8% / 29.4% | 91.8% / 20.3% | -0.11 / -0.08 / -0.10 | +0.1pp, p=0.605 | +4.2pp, p=0.159 | 2.726 -> 0.266 |
| supply_chain_disclosure_best_practice_audits | deepseek-v4-pro-fireworks | 80.9% | 77.0% | 77.8% / 57.9% | 80.9% / 0.0% | 80.3% / 7.7% | 79.6% / 32.9% | 91.6% / 19.1% | +0.22 / +0.18 / +0.13 | -1.3pp, p=0.345 | +2.6pp, p=0.261 | 1.495 -> 0.114 |
| supply_chain_disclosure_best_practice_audits | glm-5.3-flash | 80.9% | 76.0% | 76.3% / 57.9% | 80.9% / 0.0% | 80.4% / 6.4% | 80.3% / 18.4% | 90.5% / 19.1% | +0.33 / +0.29 / +0.28 | -0.6pp, p=0.533 | +4.3pp, p=0.067 | 0.207 -> 0.013 |
| supply_chain_disclosure_best_practice_audits | gpt-oss-120b-fireworks | 80.9% | 82.1% | 80.8% / 57.9% | 81.8% / 6.8% | 81.5% / 30.8% | 81.5% / 52.2% | 91.7% / 19.1% | +0.09 / +0.06 / +0.06 | +0.6pp, p=0.664 | -0.5pp, p=0.736 | 0.222 -> 0.068 |

Router features: p_pred,set_size,log_len,pred_class,bge-small-384. Training the router needs per-item LLM labels on a calibration set (one LLM pass over a few hundred items) -- a small real deployment cost the conformal gate does not have.
