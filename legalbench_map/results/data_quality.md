# Data quality report (RECONSTRUCTED)

> **Note:** the original `results/data_quality.md` produced by the task-registry agent was accidentally deleted during this integration pass's smoke-test cleanup (`rm -rf results`) before it had been read/archived. This file is a **reconstruction** computed independently, directly from `core.data.load_task_data` against the real `data/task_registry.json`, using the same per-task duplicate/length methodology the original agent described. Numbers here are freshly computed and should match the original in substance, but exact wording/formatting differs.

| task | n_train | n_test | dropped_train | dropped_test | train dup texts | test dup texts | train len (min/median/max chars) | test len (min/median/max chars) |
|---|---|---|---|---|---|---|---|---|
| cuad_anti-assignment | 6 | 1172 | 0 | 0 | 0 | 0 | 153/275/334 | 54/256/4220 |
| opp115_data_retention | 8 | 304 | 0 | 0 | 0 | 1 | 195/369/1171 | 49/423/2227 |
| learned_hands_consumer | 6 | 614 | 0 | 0 | 0 | 0 | 519/969/1872 | 152/939/12224 |
| supply_chain_disclosure_best_practice_accountability | 8 | 379 | 0 | 0 | 0 | 0 | 980/2175/4262 | 468/2686/24886 |
| overruling | 6 | 2394 | 0 | 0 | 0 | 0 | 17/99/131 | 1/147/958 |
| cuad_audit_rights | 6 | 1216 | 0 | 0 | 0 | 0 | 183/359/419 | 47/265/2363 |
| opp115_data_security | 8 | 738 | 0 | 0 | 0 | 3 | 318/671/1219 | 36/439/2279 |
| learned_hands_crime | 6 | 688 | 0 | 0 | 0 | 0 | 440/1146/1969 | 113/891/8361 |
| supply_chain_disclosure_best_practice_audits | 8 | 379 | 0 | 0 | 0 | 0 | 1432/2597/5327 | 468/2682/24886 |
| unfair_tos | 9 | 3813 | 0 | 0 | 0 | 0 | 34/197/1113 | 24/150/1677 |
| cuad_cap_on_liability | 6 | 1246 | 0 | 0 | 0 | 0 | 187/331/513 | 47/303/1921 |
| opp115_first_party_collection_use | 8 | 3034 | 0 | 0 | 0 | 55 | 111/521/779 | 30/403/2279 |
| learned_hands_employment | 6 | 710 | 0 | 0 | 0 | 0 | 985/1360/4502 | 68/919/9180 |
| supply_chain_disclosure_best_practice_certification | 8 | 378 | 0 | 0 | 0 | 0 | 980/2597/5327 | 468/2681/24886 |
| cuad_change_of_control | 6 | 416 | 0 | 0 | 0 | 0 | 90/271/419 | 76/304/2908 |
| opp115_international_and_specific_audiences | 8 | 694 | 0 | 0 | 0 | 6 | 176/449/640 | 30/419/2279 |
| learned_hands_family | 6 | 2265 | 0 | 0 | 0 | 0 | 687/1305/2549 | 68/956/18318 |
| supply_chain_disclosure_best_practice_training | 8 | 379 | 0 | 0 | 0 | 0 | 1449/2597/5327 | 468/2682/24886 |
| cuad_covenant_not_to_sue | 6 | 308 | 0 | 0 | 0 | 0 | 201/340/448 | 57/319/2263 |
| opp115_policy_change | 8 | 362 | 0 | 0 | 0 | 4 | 69/299/1136 | 30/393/1780 |
| learned_hands_housing | 6 | 4494 | 0 | 0 | 0 | 0 | 557/1764/2736 | 60/965/18318 |
| supply_chain_disclosure_best_practice_verification | 8 | 379 | 0 | 0 | 0 | 0 | 2504/3059/5327 | 468/2678/24886 |
| cuad_exclusivity | 6 | 762 | 0 | 0 | 0 | 0 | 112/215/372 | 64/300/2908 |
| opp115_third_party_sharing_collection | 8 | 2364 | 0 | 0 | 0 | 32 | 189/564/1145 | 30/414/2279 |
| learned_hands_torts | 6 | 432 | 0 | 0 | 0 | 0 | 329/644/1129 | 80/963/13071 |
| supply_chain_disclosure_disclosed_accountability | 8 | 378 | 0 | 0 | 0 | 0 | 980/2320/4262 | 468/2693/24886 |
| cuad_expiration_date | 6 | 876 | 0 | 0 | 0 | 0 | 79/163/345 | 47/229/2471 |
| opp115_user_access,_edit_and_deletion | 8 | 452 | 0 | 0 | 0 | 3 | 69/509/1564 | 30/421/2227 |
| hearsay | 5 | 94 | 0 | 0 | 0 | 0 | 90/122/141 | 85/133/285 |
| personal_jurisdiction | 4 | 50 | 0 | 0 | 0 | 0 | 307/311/572 | 262/350/553 |
| abercrombie | 5 | 95 | 0 | 0 | 0 | 0 | 27/38/54 | 27/41/81 |
| diversity_1 | 6 | 300 | 0 | 0 | 0 | 0 | 112/124/135 | 106/126/147 |
| diversity_2 | 6 | 300 | 0 | 0 | 0 | 0 | 159/163/175 | 145/168/194 |
| diversity_3 | 6 | 300 | 0 | 0 | 0 | 0 | 146/155/167 | 134/158/184 |
| diversity_4 | 6 | 300 | 0 | 0 | 0 | 0 | 161/169/176 | 143/167/188 |
| diversity_5 | 6 | 300 | 0 | 0 | 0 | 0 | 183/198/204 | 170/198/223 |
| diversity_6 | 6 | 300 | 0 | 0 | 0 | 0 | 317/326/339 | 292/323/361 |
