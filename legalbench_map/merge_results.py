"""Merge a partial per-task rerun into a full results directory.

    python merge_results.py --base results --patch results_diversity_fixed \
        --out results_merged --tasks diversity_1,diversity_2,...

For each task in --tasks, the rows in --base/tasks.csv are replaced (in
place, preserving row order) by that task's rows from --patch/tasks.csv,
and the task's per_class/<task>.csv and configs/<task>.json are taken from
--patch. Every other task is copied through from --base untouched. summary.md
and chart_llm_share.png are regenerated from the merged rows with the same
writers cli.py uses, so they cannot drift from tasks.csv.

Why this exists: the spec calls for runs that are resumable per task, and a
full 37-task run is ~2h. When one task's inputs are corrected (as happened
for diversity_1..6 after the input-column leak was found), rerunning only
those tasks and merging is the honest, cheap path -- provided the merge is
lossless. Run with --self-test to prove it: merging a directory into itself
must reproduce tasks.csv and summary.md byte for byte.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.outputs import write_chart, write_summary_md, write_tasks_csv  # noqa: E402

FLOAT_COLS = {
    "cv_mean_CV_estimated", "cv_ci_low", "cv_ci_high",
    "accuracy_mean_CV_estimated", "macro_f1_mean_CV_estimated",
    "keep_rate_mean", "keep_rate_ci_low", "keep_rate_ci_high",
    "coverage_mean", "coverage_ci_low", "coverage_ci_high",
    "metric_kept_mean_CV_estimated", "metric_kept_ci_low", "metric_kept_ci_high",
    "metric_sent_mean_CV_estimated", "metric_sent_ci_low", "metric_sent_ci_high",
    "est_cascade_metric_ESTIMATED", "published_best_score", "gap_points", "alpha",
}
INT_COLS = {"n_items", "n_classes", "best", "folds", "repeats"}
BOOL_COLS = {"is_reasoning_exception"}


def _typed(col: str, v: str):
    """Invert exactly what csv.DictWriter did to cli.py's row dict, so the
    reconstructed rows feed the same writers and reproduce the same bytes."""
    if col in FLOAT_COLS:
        return None if v == "" else float(v)          # "" was None; "nan" -> nan
    if col in INT_COLS:
        return None if v == "" else int(v)
    if col in BOOL_COLS:
        return v == "True"
    return v


def read_rows(results_dir: Path) -> list[dict]:
    rows = []
    with open(results_dir / "tasks.csv", newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            by_model = {}
            for col, v in raw.items():
                if col.startswith("published__"):
                    if v != "":
                        by_model[col[len("published__"):]] = float(v)
                else:
                    row[col] = _typed(col, v)
            row["published_by_model"] = by_model
            rows.append(row)
    return rows


def merge(base: Path, patch: Path, out: Path, tasks: set[str]) -> list[dict]:
    base_rows = read_rows(base)
    patch_rows = read_rows(patch)
    patch_by_task: dict[str, list[dict]] = {}
    for r in patch_rows:
        patch_by_task.setdefault(r["task"], []).append(r)

    missing = sorted(tasks - set(patch_by_task))
    if missing:
        raise SystemExit(f"--patch has no rows for requested task(s): {missing}")
    unexpected = sorted(set(patch_by_task) - tasks)
    if unexpected:
        raise SystemExit(f"--patch contains tasks not listed in --tasks (refusing to merge silently): {unexpected}")

    merged, emitted = [], set()
    for r in base_rows:
        t = r["task"]
        if t in tasks:
            if t not in emitted:
                merged.extend(patch_by_task[t])
                emitted.add(t)
            continue
        merged.append(r)
    for t in sorted(tasks - emitted):           # task absent from base entirely: append
        merged.extend(patch_by_task[t])

    out.mkdir(parents=True, exist_ok=True)
    write_tasks_csv(merged, out)
    write_summary_md(merged, out)
    write_chart(merged, out)
    for sub, ext in (("per_class", ".csv"), ("configs", ".json")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for r in merged:
            src_dir = patch if r["task"] in tasks else base
            src = src_dir / sub / f"{r['task']}{ext}"
            dst = out / sub / f"{r['task']}{ext}"
            if src.exists() and (not dst.exists() or src.resolve() != dst.resolve()):
                shutil.copyfile(src, dst)
    for aux in ("data_quality.md", "run_errors.log"):
        if (base / aux).exists() and not (out / aux).exists():
            shutil.copyfile(base / aux, out / aux)
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--patch", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tasks", default="")
    p.add_argument("--self-test", action="store_true",
                   help="merge --base into itself for --tasks; tasks.csv and summary.md must round-trip byte-for-byte")
    a = p.parse_args()
    tasks = {t for t in a.tasks.split(",") if t}
    if a.self_test:
        a.patch = a.base
        if not tasks:
            tasks = {r["task"] for r in read_rows(a.base)}
    if a.patch is None:
        raise SystemExit("--patch is required unless --self-test")
    merged = merge(a.base, a.patch, a.out, tasks)
    print(f"merged {len(merged)} rows across {len({r['task'] for r in merged})} tasks -> {a.out}")
    if a.self_test:
        ok = True
        for name in ("tasks.csv", "summary.md"):
            same = (a.base / name).read_bytes() == (a.out / name).read_bytes()
            print(f"  round-trip {name}: {'IDENTICAL' if same else 'DIFFERS  <-- merge is lossy, do not use'}")
            ok &= same
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
