"""Run GEPA for ONE model and write an isolated result file.

Deliberately writes to runs/<task>/gepa/gepa_<key>.json rather than touching
results.json: several of these run concurrently, and a shared read-modify-write
on one JSON file would interleave and silently lose rows. The parent
(merge_gepa.py) merges afterwards.

    DS_TASK=banking77 python gepa_one.py <model-key>
"""
import json, os, sys, time, traceback
sys.path.insert(0, "src")

from downshift.data import get_task, load_task
from downshift.evaluate import row_cost_per_1k_items, run_eval
from downshift.models import BY_KEY
from downshift.optimize import run_gepa

TASK = os.environ.get("DS_TASK", "financial_phrasebank")
key = sys.argv[1]
out_path = f"runs/{TASK}/gepa/gepa_{key.replace('/', '_')}.json"
base = json.load(open(f"runs/{TASK}/results.json"))
cfg = base["config"]




t0 = time.time()
payload = {"key": key, "ok": False}
try:
    # cfg["task"] is a real key that can hold None (older configs, written
    # before this field existed, were back-filled with null rather than the
    # real task) -- `.get("task", TASK)` only falls back when the KEY is
    # absent, so a stored None survives straight into get_task() and blows up
    # with "unknown task None". `or TASK` catches the falsy-but-present case.
    task = load_task(get_task(cfg.get("task") or TASK), cfg["n_train"], cfg["n_val"], cfg["n_test"], cfg["seed"])
    spec = BY_KEY[key]
    reflection = BY_KEY[cfg["reflection_model"]]
    print(f"[{key}] optimizing with {reflection.label} as reflection LM ...", flush=True)

    optimized, rec = run_gepa(spec, task, "direct", reflection,
                              max_metric_calls=cfg["max_metric_calls"],
                              num_threads=cfg["num_threads"],
                              max_tokens=cfg["max_tokens"], seed=cfg["seed"])
    if rec.error:
        raise RuntimeError(rec.error)

    r = run_eval(spec, optimized, task.test, task.labels,
                 num_threads=cfg["num_threads"], max_tokens=cfg["max_tokens"], progress=False)
    r.label = spec.label + " + GEPA"
    rd = r.to_dict()

    stock = base["results"].get(key, {})
    payload = {
        "key": key, "ok": True,
        "result_key": key + "+gepa",
        "result": rd,
        "optimization": rec.to_dict(),
        "stock_acc": stock.get("accuracy", {}).get("point"),
        "gepa_acc": r.accuracy.point,
        "stock_cost": row_cost_per_1k_items(stock) if stock else None,
        "gepa_cost": row_cost_per_1k_items(rd),
        "spend_usd": row_cost_per_1k_items(rd) * rd["n"] / 1000,
        # rec.instruction_changed compares the OPTIMIZABLE BODY: the fixed format
        # contract is re-pinned after every compile, and a trailer both sides carry
        # is not a rewrite.
        "instruction_changed": rec.instruction_changed,
        "wall_seconds": time.time() - t0,
    }
    print(f"[{key}] stock {payload['stock_acc']:.3f} -> gepa {payload['gepa_acc']:.3f} "
          f"| val {rec.val_score_before:.3f} -> {rec.val_score_after:.3f} "
          f"| {payload['wall_seconds']:.0f}s", flush=True)
except Exception as e:
    payload["error"] = f"{type(e).__name__}: {e}"
    payload["traceback"] = traceback.format_exc()[-1500:]
    print(f"[{key}] FAILED: {payload['error']}", flush=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(payload, open(out_path, "w"), indent=2, default=str)
print(f"[{key}] wrote {out_path}", flush=True)
sys.exit(0 if payload["ok"] else 1)
