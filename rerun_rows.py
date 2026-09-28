"""Re-evaluate rows that already exist in a run, reusing their stored prompts.

Why this exists: a fix to the *scorer* invalidates every accuracy on disk, and
`results.json` cannot be corrected in place because it stores the matched label,
not the model's raw answer -- an answer the old scorer blanked is simply gone.
The rows have to be measured again.

    DS_TASK=banking77 python rerun_rows.py --exclude claude-opus-4-8
    DS_TASK=banking77 python rerun_rows.py --only gpt-5.4-nano --dry-run

One variable at a time. A `+gepa` row is re-run with the instruction GEPA
actually produced, taken verbatim out of `optimizations[...]` and NOT re-pinned
with the format contract, even though `optimize.py` would pin it today. Pinning
is a separate, already-measured change (1.98 -> 1.00 billed calls per item on
banking77 gpt-5.4-nano, accuracy inside the CI); folding it in here would mean
two changes at once and neither would be attributable.

The split is rebuilt from the stored config, so the new rows are scored on the
identical held-out items and stay paired with the reference for McNemar.
"""
import json
import os
import sys
from types import SimpleNamespace
import time

sys.path.insert(0, "src")

import numpy as np

from downshift import stats
from downshift.chart import ChartRow, cost_footnote, plot_cost_vs_accuracy
from downshift.data import get_task, load_task, verify_or_record_split
from downshift.evaluate import row_cost_per_1k_items, run_eval
from downshift.experiment import (ALWAYS_LABEL_KEYS, _budget_for, _family_for,
                                  _profile_for, cost_basis_for)
from downshift.models import BY_KEY
from downshift.program import build_program, record_prompt_changed


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
RESULTS = f"runs/{TASK}/results.json"
CHART = f"runs/{TASK}/cost_vs_accuracy.png"


def _arg(flag):
    if flag in sys.argv:
        return [s for s in sys.argv[sys.argv.index(flag) + 1].split(",") if s]
    return []


exclude, only, dry = set(_arg("--exclude")), set(_arg("--only")), "--dry-run" in sys.argv

payload = json.load(open(RESULTS))
cfg, results, opts = payload["config"], payload["results"], payload.get("optimizations", {})
task_spec = get_task(cfg.get("task") or TASK)

# Only rows that actually called a model. The classical and encoder baselines
# predict by index straight out of the label set and never reach the string
# matcher, so a scorer fix cannot move them -- and re-running the fine-tuned
# encoder would mean 22 minutes of GPU time for a guaranteed-identical number.
targets = [k for k, r in results.items()
           if r["mean_in_tokens"] > 0 and k not in exclude and (not only or k in only)]

print(f"task={TASK}  reference={payload['reference_model']}")
est = sum(row_cost_per_1k_items(results[k]) * results[k]["n"] / 1000 for k in targets)
print(f"{len(targets)} rows to re-run, estimated ${est:.3f} at the rates last measured")
if exclude:
    print(f"excluded: {', '.join(sorted(exclude))}")
if dry:
    for k in sorted(targets, key=lambda k: -row_cost_per_1k_items(results[k])):
        kind = "GEPA prompt" if k.endswith("+gepa") else "stock prompt"
        print(f"  {k:<28} {kind}")
    sys.exit(0)

task = load_task(task_spec, cfg["n_train"], cfg["n_val"], cfg["n_test"], cfg["seed"])
assert len(task.test) == payload["task"]["n_test"]
verify_or_record_split(payload, task, RESULTS)

before = {k: results[k]["accuracy"]["point"] for k in targets}
spend, t0 = 0.0, time.time()

for key in targets:
    base = key.replace("+gepa", "")
    spec = BY_KEY.get(base)
    if spec is None or not spec.available:
        print(f"  {key}: unavailable, left as-is")
        continue
    profile = _profile_for(base)
    # `_budget_for`, not a third copy of its body. The inline version here
    # omitted `requested_max_tokens`'s reasoning-tier bump, so a gpt-5 row
    # recorded and priced mt=1024 while build_lm silently sent 16,000, and it
    # also missed the vendor-reasoning budget rule. Same helper as the screen.
    mt = _budget_for(SimpleNamespace(**{k: cfg[k] for k in
                                       ("max_tokens", "reasoning_max_tokens")}),
                     base, spec)

    instruction = task_spec.description
    if key.endswith("+gepa"):
        rec = opts.get(base) or {}
        instruction = rec.get("optimized_instruction") or instruction

    print(f"\n=== {results[key]['label']} ===", flush=True)
    r = run_eval(spec, build_program(profile, task.labels, instruction, task_spec),
                 task.test, task.labels, num_threads=cfg["num_threads"], max_tokens=mt)
    r.label = results[key]["label"]      # keep the label the chart already uses
    results[key] = r.to_dict()
    spend += r.cost_per_1k_items * r.n / 1000

# ── comparisons against the reference, on the identical items ──
ref_key = payload["reference_model"]
ref = results[ref_key]
margin = cfg["non_inferiority_margin"]
comparisons, raw_p = {}, {}
for key, res in results.items():
    if key == ref_key or res["n"] != ref["n"]:
        continue
    a, b = np.array(res["correct"], bool), np.array(ref["correct"], bool)
    mc, ni = stats.mcnemar_test(a, b), stats.non_inferiority_test(a, b, margin=margin)
    comparisons[key] = {"mcnemar": mc.to_dict(), "non_inferiority": ni.to_dict()}
    raw_p[key] = mc.p_value
for (key, _), rej in zip(raw_p.items(), stats.holm_bonferroni(list(raw_p.values()))):
    comparisons[key]["holm_significant"] = bool(rej)
payload["comparisons"] = comparisons
payload["results"] = results

print(f"\n{'row':<28} {'before':>8} {'after':>8} {'delta':>8}   parse_fail  calls/item")
for key in sorted(targets, key=lambda k: -results[k]["accuracy"]["point"]):
    r = results[key]
    d = (r["accuracy"]["point"] - before[key]) * 100
    print(f"  {key:<26} {before[key]:>7.1%} {r['accuracy']['point']:>8.1%} {d:>+7.1f}pp"
          f"   {r['parse_failures']:>9}  {r['n_calls']/r['n']:>9.3f}")

# The re-run is not a clean A/B on the scorer alone: the gpt-5* tiers reject a
# temperature parameter and therefore sample at 1.0, and cache warmth moves the
# bill between runs. Recording both numbers keeps that visible instead of
# letting the new one quietly replace the old.
# Merged, not overwritten. A second invocation with --only would otherwise
# erase the before/after of every row the first invocation measured, and those
# "before" numbers are the only surviving record of what the old scorer said --
# results.json has already been rewritten with the new ones.
record_path = f"runs/{TASK}/rerun_record.json"
try:
    record = json.load(open(record_path))
except (OSError, ValueError):
    record = {}
record["reason"] = "case-insensitive label matching in metric.score_prediction"
record.setdefault("before_accuracy", {}).update(before)
record.setdefault("after_accuracy", {}).update(
    {k: results[k]["accuracy"]["point"] for k in targets})
# Anything measured is no longer excluded, whichever run measured it.
record["excluded"] = sorted((set(record.get("excluded", [])) | exclude) - set(targets))
record["measured_spend_usd"] = round(record.get("measured_spend_usd", 0.0) + spend, 9)
open(record_path, "w").write(json.dumps(record, indent=2))

# A re-measured row carries no scorer_note (to_dict rebuilt it), so a caveat
# pointing at one is stale the moment the last flagged row is re-run.
if not any(r.get("scorer_note") for r in results.values()):
    payload.pop("caveats", None)

rows = []
for key, res in results.items():
    if key == "majority":
        continue
    base = key.replace("+gepa", "")
    linked = None
    if key.endswith("+gepa") and base in results:
        b = results[base]
        linked = (row_cost_per_1k_items(b), b["accuracy"]["point"])
    rows.append(ChartRow(
        label=res["label"], family=_family_for(base),
        cost_per_1k=row_cost_per_1k_items(res), accuracy=res["accuracy"]["point"],
            cost_basis=cost_basis_for(key, row_cost_per_1k_items(res)),
        ci_lo=res["accuracy"]["lo"], ci_hi=res["accuracy"]["hi"], linked_from=linked,
        always_label=(key in ALWAYS_LABEL_KEYS or key == ref_key),
        prompt_changed=record_prompt_changed(opts.get(base))))
plot_cost_vs_accuracy(_drop_unpriced(rows), CHART, majority_baseline=payload["task"]["majority_accuracy"],
    accuracy_bar=payload["accuracy_bar"], title="Cheapest model that clears the bar",
    subtitle=(f"{payload['task']['label']} - {payload['task']['n_test']} held-out test "
              f"items, never seen by the optimizer. Bars are 95% Wilson intervals."),
    footnote=cost_footnote(rows, "2026-09-09"))

json.dump(payload, open(RESULTS, "w"), indent=2, default=str)
print(f"\nmeasured spend: ${spend:.4f}   wall {time.time()-t0:.0f}s")
print(f"wrote {RESULTS}, {CHART}, runs/{TASK}/rerun_record.json")
