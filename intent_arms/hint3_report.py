#!/usr/bin/env python
"""Builds the numeric tables for hint3_REPORT.md from hint3.json.
Prints markdown to stdout; the prose in the report is written by hand around it."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "hint3.json"))
BOTS = ["sofmattress", "curekart", "powerplay11"]
PLATS = ["dialogflow", "luis", "rasa", "bert", "haptik"]


def f(x, n=4):
    return "n/a" if x is None else f"{float(x):.{n}f}"


print("## T1. Constant-predictor guard\n")
print("| bot | test | in-scope | OOS | all-OOS const acc | maj-intent const acc | const in-scope acc | const macro-F1 | const MCC |")
print("|---|---|---|---|---|---|---|---|---|")
for b in BOTS:
    B = R["bots"][b]
    c, m = B["constants"]["const_all_oos"], B["constants"]["const_majority_intent"]
    print(f"| {b} | {B['n_test']} | {B['n_test_inscope']} | {B['n_test_oos']} | "
          f"**{f(c['accuracy'])}** | {f(m['accuracy'])} | {f(c['inscope_accuracy'])} | "
          f"{f(c['macro_f1_inscope'])} | {f(c['mcc'])} |")

print("\n## T2. Free arms, no rejection (t=0, argmax always answers)\n")
print("| bot | arm | in-scope acc | macro-F1 | MCC | overall acc | CV macro-F1 (train OOF) |")
print("|---|---|---|---|---|---|---|")
for b in BOTS:
    B = R["bots"][b]
    for a, A in B["arms"].items():
        z = A["no_rejection_t0"]
        star = " **<-- shipped**" if a == B["shipped_row"]["arm"] else ""
        print(f"| {b} | {a}{star} | {f(z['inscope_accuracy'])} | {f(z['macro_f1_inscope'])} | "
              f"{f(z['mcc'])} | {f(z['accuracy'])} | {f(A['cv_macro_f1_train_oof'])} |")

print("\n## T3. Free arms at the TRAIN-SELECTED operating point (IRR=10%, no test tuning)\n")
print("| bot | arm | t* | in-scope acc | OOS recall | macro-F1 | MCC | overall acc | beats all-OOS const? |")
print("|---|---|---|---|---|---|---|---|---|")
for b in BOTS:
    B = R["bots"][b]
    cacc = B["constants"]["const_all_oos"]["accuracy"]
    for a, A in B["arms"].items():
        op = A["train_selected_operating_points"]["irr_0.10"]
        z = op["metrics"]
        w = "YES" if z["accuracy"] > cacc else f"**NO ({z['accuracy']-cacc:+.4f})**"
        star = " **<-- shipped**" if a == B["shipped_row"]["arm"] else ""
        print(f"| {b} | {a}{star} | {f(op['threshold'])} | {f(z['inscope_accuracy'])} | "
              f"{f(z['oos_recall'])} | {f(z['macro_f1_inscope'])} | {f(z['mcc'])} | "
              f"{f(z['accuracy'])} | {w} |")

print("\n## T4. Threshold sweep, shipped row (paper grid)\n")
for b in BOTS:
    B = R["bots"][b]
    a = B["shipped_row"]["arm"]
    print(f"\n**{b}** -- shipped arm `{a}`; all-OOS constant acc = {f(B['constants']['const_all_oos']['accuracy'])}\n")
    print("| t | overall acc | in-scope acc | OOS recall | macro-F1 | MCC | answered % |")
    print("|---|---|---|---|---|---|---|")
    for z in B["arms"][a]["sweep_paper_grid"]:
        print(f"| {z['threshold']:.1f} | {f(z['accuracy'])} | {f(z['inscope_accuracy'])} | "
              f"{f(z['oos_recall'])} | {f(z['macro_f1_inscope'])} | {f(z['mcc'])} | "
              f"{100*z['answered_frac']:.1f} |")

print("\n## T5. Paired comparison vs published platforms (same items, same order)\n")
print("Each method at its OWN test-optimal macro-F1 threshold -> optimistic for every row, symmetrically.\n")
for b in BOTS:
    B = R["bots"][b]
    a = B["shipped_row"]["arm"]
    ours = B["arms"][a]["best_on_test_macro_f1_OPTIMISTIC"]
    print(f"\n**{b}**\n")
    print("| method | t | in-scope acc | OOS recall | macro-F1 | MCC | overall acc |")
    print("|---|---|---|---|---|---|---|")
    print(f"| **ours: {a}** | {ours['threshold']:.2f} | {f(ours['inscope_accuracy'])} | "
          f"{f(ours['oos_recall'])} | **{f(ours['macro_f1_inscope'])}** | {f(ours['mcc'])} | {f(ours['accuracy'])} |")
    for p in PLATS:
        z = B["published_platforms"][p]["best_on_test_macro_f1_inscope"]
        print(f"| {p} (published) | {z['threshold']:.2f} | {f(z['inscope_accuracy'])} | "
              f"{f(z['oos_recall'])} | {f(z['macro_f1_inscope'])} | {f(z['mcc'])} | {f(z['accuracy'])} |")
    print(f"| all-OOS constant | - | 0.0000 | 1.0000 | 0.0000 | 0.0000 | "
          f"{f(B['constants']['const_all_oos']['accuracy'])} |")
    print("\nMcNemar (overall correctness, paired):\n")
    print("| vs | ours right / theirs wrong | theirs right / ours wrong | exact p |")
    print("|---|---|---|---|")
    for p in PLATS:
        m = B["paired_mcnemar_vs_published"][p]
        print(f"| {p} | {m['ours_right_theirs_wrong']} | {m['theirs_right_ours_wrong']} | {m['mcnemar_exact_p']:.3g} |")

print("\n## T6. Cascade pre-flight gate (shipped row)\n")
print("| bot | arm | def | AUROC(conf->correct) | gate >=0.75 | err share in least-conf 20% | gate >>20% | overall |")
print("|---|---|---|---|---|---|---|---|")
for b in BOTS:
    S = R["bots"][b]["shipped_row"]
    for tag, key in [("A shipped pipeline", "gate_A_shipped_pipeline"),
                     ("B answering subset", "gate_B_answering_subset_inscope_only")]:
        g = S[key]
        print(f"| {b} | {S['arm']} | {tag} | {f(g['auroc_conf_vs_correct'])} | "
              f"{'PASS' if g['gate_auroc_pass'] else 'FAIL'} | "
              f"{f(g['error_share_least_confident_20pct'])} | "
              f"{'PASS' if g['gate_error_share_pass'] else 'FAIL'} | "
              f"{'**PASS**' if g['gate_overall_pass'] else 'FAIL'} |")

print("\n## T7. Escalation dial (coverage/cost), shipped row\n")
for b in BOTS:
    S = R["bots"][b]["shipped_row"]
    for tag, key in [("A shipped pipeline", "escalation_dial_A_shipped_pipeline"),
                     ("B answering subset", "escalation_dial_B_answering_subset")]:
        row = " | ".join(f(d["kept_accuracy"]) for d in S[key])
        print(f"- **{b}** / {tag}: kept-slice accuracy at 0/10/20/30/40% escalated = {row}")

print("\n## T8. Contamination\n")
print("| bot | test rows | exact train/test overlap | normalized overlap | of which in-scope | of which OOS | train internal dups |")
print("|---|---|---|---|---|---|---|")
for b in BOTS:
    c = R["bots"][b]["contamination"]
    print(f"| {b} | {c['test_rows']} | {c['exact_overlap_n']} ({c['exact_overlap_pct']}%) | "
          f"{c['normalized_overlap_n']} ({c['normalized_overlap_pct']}%) | "
          f"{c['exact_overlap_inscope_n']}/{c['exact_overlap_inscope_of']} | "
          f"{c['exact_overlap_oos_n']}/{c['exact_overlap_oos_of']} | "
          f"{c['train_internal_dup_exact']} exact / {c['train_internal_dup_normalized']} normalized |")

print("\n## T9. Label budget and tiny intents\n")
print("| bot | train rows | intents | median/intent | min | intents <3 ex | intents ==1 ex | test items in <3 | test items in ==1 | train intents absent from test |")
print("|---|---|---|---|---|---|---|---|---|---|")
for b in BOTS:
    B = R["bots"][b]
    L = B["labels_per_intent"]
    print(f"| {b} | {B['n_train']} | {B['n_train_intents']} | {L['median']:.0f} | {L['min']} | "
          f"{B['intents_with_lt3_train_ex']} | {B['intents_with_1_train_ex']} | "
          f"{B['test_items_in_lt3_intents']} | {B['test_items_in_singleton_intents']} | "
          f"{B['train_intents_absent_from_test']} |")

print("\n### Tiny-intent outcomes on the shipped row\n")
for b in BOTS:
    pi = R["bots"][b]["tiny_intent_breakdown_shipped_row"]
    if not pi:
        print(f"- **{b}**: no intents with <3 training examples.")
        continue
    tot_i = sum(x["test_items"] for x in pi)
    tot_c = sum(x["argmax_correct"] for x in pi)
    tot_k = sum(x["kept_at_operating_t"] for x in pi)
    print(f"- **{b}**: {len(pi)} intents with <3 train examples own {tot_i} test items; "
          f"argmax gets {tot_c}/{tot_i} right, {tot_k}/{tot_i} survive the operating threshold.")

print("\n## T10. Encoder epoch selection (train CV only)\n")
print("| bot | CV macro-F1 @3ep | @10ep | @20ep | selected |")
print("|---|---|---|---|---|")
for b in BOTS:
    e = R["bots"][b]["encoder_epoch_selection"]
    c = e["cv_macro_f1_by_epochs"]
    print(f"| {b} | {f(c.get('3'))} | {f(c.get('10'))} | {f(c.get('20'))} | {e['selected_epochs']} |")

print("\n## T11. TF-IDF sublinear_tf-on-char ablation (D11)\n")
print("| bot | sublinear on both (used) | sublinear on word only | delta |")
print("|---|---|---|---|")
for b in BOTS:
    a = R["bots"][b]["tfidf_sublinear_char_ablation"]
    x, y = a["sublinear_on_both_inscope_acc_at_t0"], a["sublinear_word_only_inscope_acc_at_t0"]
    print(f"| {b} | {f(x)} | {f(y)} | {x-y:+.4f} |")
