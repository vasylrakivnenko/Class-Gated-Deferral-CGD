"""Merge the isolated runs/<task>/gepa/gepa_<key>.json files back into results.json.

The parallel jobs each wrote their own file precisely so they could not race
on results.json. This is the single-threaded step that folds them in, then
recomputes every comparison against the reference and re-renders the chart.

    DS_TASK=banking77 python merge_gepa.py
"""
import glob, json, os, sys
sys.path.insert(0, "src")

import numpy as np
from downshift import stats
from downshift.chart import ChartRow, cost_footnote, plot_cost_vs_accuracy
from downshift.evaluate import row_cost_per_1k_items, row_n_calls
from downshift.program import record_prompt_changed
from downshift.experiment import _family_for, ALWAYS_LABEL_KEYS, cost_basis_for


def _drop_unpriced(rows):
    """Keep the plotter's refusal out of the end of a finished script.

    `plot_cost_vs_accuracy` rejects NaN and negative costs, and NaN became
    reachable for any row whose every call failed, not just registry rows with
    no rate card. Every script that charts from results.json has to make the
    same call run_experiment makes, or re-charting a finished run dies at the
    very last step.
    """
    import math
    keep, drop = [], []
    for r in rows:
        c = r.cost_per_1k
        (keep if c is not None and not math.isnan(c) and c >= 0 else drop).append(r)
    for r in keep:
        if r.linked_from is not None:
            t = r.linked_from[0]
            if t is None or math.isnan(t) or t < 0:
                r.linked_from = None
    if drop:
        print(f"  NOTE: {len(drop)} row(s) left off the chart for want of a verified "
              f"price: {[r.label for r in drop]}")
    return keep


TASK = os.environ.get("DS_TASK", "financial_phrasebank")
RESULTS, CHART = f"runs/{TASK}/results.json", f"runs/{TASK}/cost_vs_accuracy.png"
payload = json.load(open(RESULTS))
results, cfg, task = payload["results"], payload["config"], payload["task"]




merged, failed = [], []
for path in sorted(glob.glob(f"runs/{TASK}/gepa/gepa_*.json")):
    j = json.load(open(path))
    if not j.get("ok"):
        failed.append((j.get("key", path), j.get("error", "unknown")))
        continue
    # Scratch files written before evaluate.py recorded n_calls carry neither
    # n_calls nor cost_per_1k_items, and merging one verbatim would put a row
    # back into results.json missing both -- reintroducing the per-call cost
    # under a per-classification axis. Normalise on the way in.
    r = j["result"]
    r.setdefault("n_calls", row_n_calls(r))
    r.setdefault("cost_per_1k_items", row_cost_per_1k_items(r))
    # And take the table's costs from the rows, not from the file's own
    # stock_cost/gepa_cost: those were written per billed call.
    j["gepa_cost"] = row_cost_per_1k_items(r)
    stock = results.get(j["key"])
    j["stock_cost"] = row_cost_per_1k_items(stock) if stock else j.get("stock_cost")
    j["spend_usd"] = j["gepa_cost"] * r["n"] / 1000
    results[j["result_key"]] = r
    payload["optimizations"][j["key"]] = j["optimization"]
    merged.append(j)

for j in sorted(merged, key=lambda x: -(x["gepa_acc"] or 0)):
    d = (j["gepa_acc"] - j["stock_acc"]) * 100
    print(f"  {j['key']:<22} {j['stock_acc']:.3f} -> {j['gepa_acc']:.3f} ({d:+.1f}pp)  "
          f"${j['stock_cost']:.5f} -> ${j['gepa_cost']:.5f}/1k items  "
          f"{'rewritten' if j['instruction_changed'] else 'UNCHANGED'}  ${j['spend_usd']:.3f}")
for k, e in failed:
    print(f"  {k:<22} FAILED: {e[:120]}")
print(f"\nspend on these GEPA evals: ${sum(j['spend_usd'] for j in merged):.2f}")

# ── recompute comparisons ──
ref_key = payload["reference_model"]; ref = results[ref_key]
margin, bar = cfg["non_inferiority_margin"], cfg["accuracy_bar"]
comparisons, raw_p = {}, {}
for key, res in results.items():
    if key == ref_key or res["n"] != ref["n"]:
        continue
    a, b = np.array(res["correct"], bool), np.array(ref["correct"], bool)
    mc, ni = stats.mcnemar_test(a, b), stats.non_inferiority_test(a, b, margin=margin)
    comparisons[key] = {"mcnemar": mc.to_dict(), "non_inferiority": ni.to_dict()}
    raw_p[key] = mc.p_value
for (key, p), rej in zip(raw_p.items(), stats.holm_bonferroni(list(raw_p.values()))):
    comparisons[key]["holm_significant"] = bool(rej)
payload["comparisons"] = comparisons

print(f"\nFull ranking vs {ref['label']} ({ref['accuracy']['point']:.1%}):")
for key, res in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]["point"]):
    if key == ref_key:
        print(f"  [ REF] {res['label']:<42} {res['accuracy']['point']:6.1%}  ${row_cost_per_1k_items(res):.5f}/1k items")
        continue
    ni = comparisons[key]["non_inferiority"]
    print(f"  [{'PASS' if ni['passes'] else '----'}] {res['label']:<42} "
          f"{res['accuracy']['point']:6.1%}  ${row_cost_per_1k_items(res):.5f}/1k items  {ni['verdict']}")

# ── re-chart ──
rows = []
for key, res in results.items():
    if key == "majority":
        continue
    base = key.replace("+gepa", "")
    linked = None
    if key.endswith("+gepa") and base in results:
        b = results[base]
        linked = (row_cost_per_1k_items(b), b["accuracy"]["point"])
    rows.append(ChartRow(label=res["label"], family=_family_for(base),
                         cost_basis=cost_basis_for(key, row_cost_per_1k_items(res)),
                         cost_per_1k=row_cost_per_1k_items(res), accuracy=res["accuracy"]["point"],
                         ci_lo=res["accuracy"]["lo"], ci_hi=res["accuracy"]["hi"],
                         linked_from=linked,
                         always_label=(key in ALWAYS_LABEL_KEYS or key == ref_key),
            # Without this the label falls back to "did the numbers move",
            # which reads a same-prompt replicate as a GEPA improvement.
            prompt_changed=record_prompt_changed(
                payload.get("optimizations", {}).get(base))))
plot_cost_vs_accuracy(_drop_unpriced(rows), CHART, majority_baseline=payload["task"]["majority_accuracy"],
    accuracy_bar=bar, title="Cheapest model that clears the bar",
    subtitle=(f"{task['label']} — {task['n_test']} held-out test items, "
              f"never seen by the optimizer. Bars are 95% Wilson intervals."),
    footnote=cost_footnote(rows, "2026-09-09"))

payload["results"] = results
json.dump(payload, open(RESULTS, "w"), indent=2, default=str)
print(f"\nmerged {len(merged)} runs into {RESULTS}")
