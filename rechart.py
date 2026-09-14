"""Re-render the chart from a finished run, without re-running the models.

An experiment costs tens of minutes; the chart costs a second. Iterating on
labels, the accuracy bar, or which rows to show should not mean paying for the
models again.

    DS_TASK=banking77 python rechart.py [runs/banking77/results.json] [--bar 0.90]
"""
import json
import os
import sys

sys.path.insert(0, "src")

from downshift.chart import ChartRow, cost_footnote, plot_cost_vs_accuracy
from downshift.evaluate import row_cost_per_1k_items
from downshift.program import record_prompt_changed
from downshift.experiment import _family_for, ALWAYS_LABEL_KEYS, cost_basis_for

# Follows DS_TASK the same way add_model.py's default does, so running with no
# path argument re-renders whichever task's run is currently active.
DEFAULT_RESULTS = f"runs/{os.environ.get('DS_TASK', 'financial_phrasebank')}/results.json"




def main(path: str = DEFAULT_RESULTS, bar: float | None = None) -> None:
    payload = json.loads(open(path).read())
    task = payload["task"]
    results = payload["results"]

    ref_key = payload.get("reference_model", "")
    rows = []
    for key, res in results.items():
        # The majority-class row is a reference LINE, not a deployable
        # candidate -- drawn as the horizontal rule only, same as the
        # interactive chart, so it cannot be misread as an option.
        if key == "majority":
            continue
        base_key = key.replace("+gepa", "")
        linked = None
        if key.endswith("+gepa") and base_key in results:
            b = results[base_key]
            linked = (row_cost_per_1k_items(b), b["accuracy"]["point"])
        rows.append(ChartRow(
            label=res["label"], family=_family_for(base_key),
            cost_per_1k=row_cost_per_1k_items(res),
            cost_basis=cost_basis_for(key, row_cost_per_1k_items(res)),
            accuracy=res["accuracy"]["point"],
            ci_lo=res["accuracy"]["lo"], ci_hi=res["accuracy"]["hi"],
            linked_from=linked,
            always_label=(key in ALWAYS_LABEL_KEYS or key == ref_key),
            # Without this the label falls back to "did the numbers move",
            # which reads a same-prompt replicate as a GEPA improvement.
            prompt_changed=record_prompt_changed(
                payload.get("optimizations", {}).get(base_key))))

    out = payload.get("chart") or f"runs/{task['key']}/cost_vs_accuracy.png"
    plot_cost_vs_accuracy(
        rows, out,
        majority_baseline=task["majority_accuracy"],
        accuracy_bar=bar if bar is not None else payload.get("accuracy_bar"),
        title="Cheapest model that clears the bar",
        subtitle=(f"{task['label']} — {task['n_test']} held-out test items, "
                  f"never seen by the optimizer. Bars are 95% Wilson intervals."),
        footnote=cost_footnote(rows, "2026-09-09"))
    print(f"wrote {out} ({len(rows)} rows)")

    print("\nRanked by accuracy:")
    for key, res in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]["point"]):
        a = res["accuracy"]
        print(f"  {res['label']:<38} {a['point']:6.1%} [{a['lo']:.3f},{a['hi']:.3f}]  "
              f"${row_cost_per_1k_items(res):.5f}/1k items  {res['wall_seconds']:6.0f}s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bar = None
    if "--bar" in sys.argv:
        bar = float(sys.argv[sys.argv.index("--bar") + 1])
    main(args[0] if args else DEFAULT_RESULTS, bar)
