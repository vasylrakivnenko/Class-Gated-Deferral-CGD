"""Add one or more models to an existing results.json without re-running the rest.

    DS_TASK=banking77 python add_model.py claude-sonnet-4-6 [--gepa]

Reuses the exact same task split (same seed, same sizes) as the original run, so
new rows are scored on the identical 250 test items -- which is what makes the
paired McNemar / non-inferiority statistics against the reference valid.
"""
import json, sys
sys.path.insert(0, "src")

import numpy as np
from downshift import stats
from downshift.chart import ChartRow, cost_footnote, plot_cost_vs_accuracy
from downshift.data import get_task, load_task, verify_or_record_split
from downshift.evaluate import row_cost_per_1k_items, run_eval
from downshift.experiment import _family_for, ALWAYS_LABEL_KEYS, cost_basis_for
from downshift.models import BY_KEY
from downshift.optimize import run_gepa
from downshift.program import build_program, record_prompt_changed

import os




TASK = os.environ.get("DS_TASK", "financial_phrasebank")
RESULTS, CHART = f"runs/{TASK}/results.json", f"runs/{TASK}/cost_vs_accuracy.png"
keys = [a for a in sys.argv[1:] if not a.startswith("--")]
do_gepa = "--gepa" in sys.argv

payload = json.load(open(RESULTS))
cfg, results = payload["config"], payload["results"]
# cfg["task"] is a real key that can hold None (older configs, written before
# this field existed, were back-filled with null rather than the real task) --
# `.get("task", TASK)` only falls back when the KEY is absent, so a stored
# None survives straight into get_task() and blows up with "unknown task
# None". `or TASK` catches the falsy-but-present case too.
task_spec = get_task(cfg.get("task") or TASK)
task = load_task(task_spec, cfg["n_train"], cfg["n_val"], cfg["n_test"], cfg["seed"])
assert len(task.test) == payload["task"]["n_test"]
verify_or_record_split(payload, task, RESULTS)

for key in keys:
    spec = BY_KEY[key]
    print(f"\n=== {spec.label}  (${spec.price_in}/{spec.price_out} per 1M) ===")
    r = run_eval(spec, build_program("direct", task.labels, task_spec.description, task_spec), task.test, task.labels,
                 num_threads=cfg["num_threads"], max_tokens=cfg["max_tokens"])
    results[key] = r.to_dict()
    # cost_per_1k_calls is per BILLED CALL, not per item -- an adapter retry
    # can spend two calls on one classification, so scale by actual calls
    # (recovered via _cost_per_1k_items) rather than len(task.test).
    spend = row_cost_per_1k_items(results[key]) * results[key]["n"] / 1000
    print(f"    actual spend for this eval: ${spend:.3f}")

    if do_gepa:
        reflection = BY_KEY[cfg["reflection_model"]]
        optimized, rec = run_gepa(spec, task, "direct", reflection,
                                  max_metric_calls=cfg["max_metric_calls"],
                                  num_threads=cfg["num_threads"],
                                  max_tokens=cfg["max_tokens"], seed=cfg["seed"])
        print(f"    GEPA val {rec.val_score_before:.3f} -> {rec.val_score_after:.3f}")
        payload["optimizations"][key] = rec.to_dict()
        ro = run_eval(spec, optimized, task.test, task.labels,
                      num_threads=cfg["num_threads"], max_tokens=cfg["max_tokens"])
        ro.label = spec.label + " + GEPA"
        results[key + "+gepa"] = ro.to_dict()

# ── recompute every comparison against the reference ──
ref_key = payload["reference_model"]
ref = results[ref_key]
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

print(f"\nRanked vs {ref['label']} ({ref['accuracy']['point']:.1%}), margin {margin:.0%}:")
for key, res in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]["point"]):
    if key == ref_key:
        print(f"  [ REF] {res['label']:<40} {res['accuracy']['point']:6.1%}  ${row_cost_per_1k_items(res):.5f}/1k items")
        continue
    ni = comparisons[key]["non_inferiority"]
    print(f"  [{'PASS' if ni['passes'] else '----'}] {res['label']:<40} "
          f"{res['accuracy']['point']:6.1%}  ${row_cost_per_1k_items(res):.5f}/1k items  "
          f"diff {ni['diff']:+.1%}  {ni['verdict']}")

# ── re-chart ──
rows = []
for key, res in results.items():
    # The majority-class row is a reference LINE, not a deployable candidate --
    # same as rechart.py, merge_gepa.py and the interactive chart, so it cannot
    # be misread as an option on the menu.
    if key == "majority":
        continue
    base = key.replace("+gepa", "")
    linked = None
    if key.endswith("+gepa") and base in results:
        b = results[base]
        linked = (row_cost_per_1k_items(b), b["accuracy"]["point"])
    rows.append(ChartRow(label=res["label"], family=_family_for(base),
                         cost_per_1k=row_cost_per_1k_items(res), accuracy=res["accuracy"]["point"],
            cost_basis=cost_basis_for(key, row_cost_per_1k_items(res)),
                         ci_lo=res["accuracy"]["lo"], ci_hi=res["accuracy"]["hi"],
                         linked_from=linked,
                         always_label=(key in ALWAYS_LABEL_KEYS or key == ref_key),
            # Without this the label falls back to "did the numbers move",
            # which reads a same-prompt replicate as a GEPA improvement.
            prompt_changed=record_prompt_changed(
                payload.get("optimizations", {}).get(base))))
plot_cost_vs_accuracy(rows, CHART, majority_baseline=payload["task"]["majority_accuracy"],
    accuracy_bar=bar, title="Cheapest model that clears the bar",
    subtitle=(f"{payload['task']['label']} — {payload['task']['n_test']} held-out test items, "
              f"never seen by the optimizer. Bars are 95% Wilson intervals."),
    footnote=cost_footnote(rows, "2026-09-09"))

payload["results"] = results
json.dump(payload, open(RESULTS, "w"), indent=2, default=str)
print(f"\nUpdated {RESULTS} and {CHART}")
