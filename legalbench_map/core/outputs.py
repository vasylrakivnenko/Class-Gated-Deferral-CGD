"""
core/outputs.py

Writes every deliverable under results/:
  - tasks.csv                    one row per task x candidate
  - summary.md                   best-candidate-per-task table + counts + assumptions
  - chart_llm_share.png          horizontal bar chart, sorted by est. LLM share, by verdict
  - per_class/<task>.csv         per-class recall + keep rate for the best candidate
  - configs/<task>.json          best candidate, hyperparams, conformal quantile(s)

Every accuracy/metric number is labeled "CV-estimated" (main metrics) or
"estimated cascade" (the cascade number) SOMEWHERE in the same row/section
-- enforced here by literally naming columns/headers that way rather than
leaving it to convention.

Input: a list of row dicts (one per task x candidate), each expected to
carry the flat scalar fields consumed by tasks.csv/summary.md/chart, plus
three nested fields used only for per_class/<task>.csv and
configs/<task>.json: `per_class_recall`, `per_class_keep_rate`,
`per_class_recall_kept` (dict[class->float]), `fold_thresholds` (list of
float), and `hyperparams` (dict or None). See cli.py for exactly how each
row is assembled from core.cv.CVResult + core.verdict outputs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("legalbench_map.outputs")

VERDICT_COLORS = {
    "match": "#2ca02c",       # green
    "close": "#ffbf00",       # amber
    "short": "#d62728",       # red
    "no_reference": "#7f7f7f",  # gray
}

FLAT_COLUMNS = [
    "task", "candidate", "family", "is_reasoning_exception",
    "n_items", "n_classes", "metric_name",
    "cv_mean_CV_estimated", "cv_ci_low", "cv_ci_high",
    "accuracy_mean_CV_estimated", "macro_f1_mean_CV_estimated",
    "keep_rate_mean", "keep_rate_ci_low", "keep_rate_ci_high",
    "coverage_mean", "coverage_ci_low", "coverage_ci_high",
    "metric_kept_mean_CV_estimated", "metric_kept_ci_low", "metric_kept_ci_high",
    "metric_sent_mean_CV_estimated", "metric_sent_ci_low", "metric_sent_ci_high",
    "est_cascade_metric_ESTIMATED",
    "published_best_score", "published_best_model", "published_best_source_url",
    "gap_points", "verdict", "best",
    "alpha", "folds", "repeats",
]


def _published_model_columns(rows):
    models = set()
    for row in rows:
        models.update((row.get("published_by_model") or {}).keys())
    return sorted(models)


def write_tasks_csv(rows, results_dir: Path):
    import csv as _csv

    model_cols = _published_model_columns(rows)
    columns = FLAT_COLUMNS + [f"published__{m}" for m in model_cols]

    path = Path(results_dir) / "tasks.csv"
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {c: row.get(c, "") for c in FLAT_COLUMNS}
            by_model = row.get("published_by_model") or {}
            for m in model_cols:
                out[f"published__{m}"] = by_model.get(m, "")
            writer.writerow(out)
    return path


def _best_rows(rows):
    """One row per task: the row with best==1 (ties already broken
    upstream, deterministically, by candidate name)."""
    by_task = {}
    for row in rows:
        if row.get("best") == 1:
            by_task[row["task"]] = row
    return by_task


ASSUMPTIONS_TEXT = """\
**CV-vs-published comparability assumption.** Published LLM scores were
computed by their original authors on the fixed LegalBench test split
with their own methodology (typically a single pass, no CV). The
CV-estimated numbers in this report evaluate the same test-split items
via repeated stratified 5-fold cross-validation so every item gets an
out-of-fold prediction. We treat the two as comparable estimates of
"score on the LegalBench test split for this task" -- this is an
assumption, not a guarantee, and every CV-derived number is labeled
"CV-estimated" to keep that assumption visible.

**Estimated-cascade assumption.** `est_cascade_metric` = keep_rate *
metric_on_kept + (1 - keep_rate) * published_best_llm_score. Sent items
are assumed to score at the LLM's published AVERAGE, which may be
optimistic: sent items are exactly the ones the conformal gate flagged
as uncertain, i.e. selected to be harder than a random draw from the
test set. For balanced_accuracy, the estimate is computed per class
(assuming the published task-average score applies uniformly across
classes, since no per-class published number exists) and then averaged.
Every such number is labeled "estimated cascade".
"""


def write_summary_md(rows, results_dir: Path):
    best = _best_rows(rows)
    tasks_sorted = sorted(
        best.values(),
        key=lambda r: (r.get("keep_rate_mean") is None, 1 - (r.get("keep_rate_mean") or 0.0), r["task"]),
    )
    # llm_share ascending == keep_rate descending; recompute explicitly for clarity/sort key.
    tasks_sorted = sorted(best.values(), key=lambda r: (1.0 - (r.get("keep_rate_mean") or 0.0), r["task"]))

    counts = {"match": 0, "close": 0, "short": 0, "no_reference": 0}
    for r in tasks_sorted:
        counts[r.get("verdict", "no_reference")] += 1

    lines = []
    lines.append("# LegalBench Cheap-Model-Map -- Summary\n")
    lines.append(
        "One row per task, showing the BEST cheap candidate for that task "
        "(all numbers below are CV-estimated unless marked estimated "
        "cascade), sorted by estimated LLM share ascending (tasks needing "
        "the LLM least come first).\n"
    )
    lines.append(
        "| task | best candidate | metric | CV-estimated mean | 95% CI | "
        "keep rate | est. LLM share | estimated cascade | published best | source | gap (pts) | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in tasks_sorted:
        kr = r.get("keep_rate_mean")
        llm_share = (1.0 - kr) if kr is not None and kr == kr else float("nan")
        cv_mean = r.get("cv_mean_CV_estimated")
        ci_lo = r.get("cv_ci_low")
        ci_hi = r.get("cv_ci_high")
        est_casc = r.get("est_cascade_metric_ESTIMATED")
        pub = r.get("published_best_score")
        gap = r.get("gap_points")
        # Every published number carries its citation inline -- a number
        # with no traceable source is not usable evidence for this project
        # (see load_published_scores' source_url requirement).
        source = r.get("published_best_source_url") or ("n/a" if pub is None else "MISSING SOURCE")
        lines.append(
            f"| {r['task']} | {r['candidate']} | {r['metric_name']} | "
            f"{cv_mean:.3f} | [{ci_lo:.3f}, {ci_hi:.3f}] | "
            f"{kr:.3f} | {llm_share:.3f} | "
            f"{('%.3f' % est_casc) if isinstance(est_casc, float) else 'n/a'} | "
            f"{('%.3f' % pub) if isinstance(pub, float) else 'n/a'} | "
            f"{source} | "
            f"{('%.2f' % gap) if isinstance(gap, float) else 'n/a'} | "
            f"{r.get('verdict', 'no_reference')} |"
        )

    lines.append("")
    lines.append(
        f"**Verdict counts** (best candidate per task, n={len(tasks_sorted)} tasks): "
        f"match={counts['match']}, close={counts['close']}, short={counts['short']}, "
        f"no_reference={counts['no_reference']}.\n"
    )
    lines.append(ASSUMPTIONS_TEXT)

    path = Path(results_dir) / "summary.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_chart(rows, results_dir: Path):
    best = _best_rows(rows)
    tasks_sorted = sorted(best.values(), key=lambda r: (1.0 - (r.get("keep_rate_mean") or 0.0), r["task"]))
    if not tasks_sorted:
        logger.warning("no best-candidate rows to chart -- skipping chart_llm_share.png")
        return None

    labels = [r["task"] for r in tasks_sorted]
    shares = [1.0 - (r.get("keep_rate_mean") or 0.0) for r in tasks_sorted]
    colors = [VERDICT_COLORS.get(r.get("verdict", "no_reference"), "#7f7f7f") for r in tasks_sorted]

    height = max(3.0, 0.28 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))
    y_pos = range(len(labels))
    ax.barh(y_pos, shares, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated LLM share (1 - keep rate), CV-estimated")
    ax.set_title("Estimated share of items sent to the LLM, by task (best cheap candidate)")
    ax.set_xlim(0, 1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in VERDICT_COLORS.values()]
    ax.legend(handles, list(VERDICT_COLORS.keys()), title="verdict", loc="lower right", fontsize=8)

    fig.tight_layout()
    path = Path(results_dir) / "chart_llm_share.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_per_class_csvs(rows, results_dir: Path):
    import csv as _csv

    best = _best_rows(rows)
    out_dir = Path(results_dir) / "per_class"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for task, row in best.items():
        rec = row.get("per_class_recall") or {}
        kr = row.get("per_class_keep_rate") or {}
        recall_kept = row.get("per_class_recall_kept") or {}
        path = out_dir / f"{task}.csv"
        with open(path, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["class", "recall_CV_estimated", "keep_rate", "recall_on_kept_CV_estimated"])
            for c in sorted(rec.keys()):
                writer.writerow([c, rec.get(c, ""), kr.get(c, ""), recall_kept.get(c, "")])
        paths.append(path)
    return paths


def write_configs(rows, results_dir: Path):
    best = _best_rows(rows)
    out_dir = Path(results_dir) / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for task, row in best.items():
        thresholds = row.get("fold_thresholds") or []
        finite = [t for t in thresholds if t == t and t not in (float("inf"),)]
        config = {
            "task": task,
            "best_candidate": row["candidate"],
            "hyperparameters": row.get("hyperparams"),
            "hyperparameters_note": (
                "The fixed candidate interface (fit_predict_proba -> np.ndarray) "
                "returns only probabilities, not hyperparameters; this field is "
                "populated only if the candidate module optionally exposes "
                "get_last_hyperparams(task_key), else it is null."
            ),
            "conformal_alpha": row.get("alpha"),
            "conformal_quantiles_all_folds": thresholds,
            "conformal_quantile_mean_finite": (sum(finite) / len(finite)) if finite else None,
            "folds": row.get("folds"),
            "repeats": row.get("repeats"),
        }
        path = out_dir / f"{task}.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=False)
        paths.append(path)
    return paths


PER_ITEM_COLUMNS = [
    "task", "candidate", "repeat", "fold", "item_idx", "text_sha256",
    "gold", "pred", "p_pred", "kept", "set_size",
]


def write_per_item_csvs(rows, results_dir, X_test_by_task):
    """Writes the optional per-item dump (Phase 2 "FILE A"):

        <results_dir>/per_item/<task>__<candidate>.csv

    with columns PER_ITEM_COLUMNS in that exact order and one row per
    (repeat, item_idx). Only rows carrying a non-None `per_item` list (set
    by cli.py when --dump-per-item is on, from CVResult.per_item) are
    written; everything else in `rows` is untouched -- this function is
    never called from write_all, so it cannot affect tasks.csv or any
    other Phase 1 output.

    rows            list of row dicts as assembled by cli.py; each must
                    carry "task", "candidate" and (optionally) "per_item".
    results_dir     the run's results directory.
    X_test_by_task  dict task -> list[str], the task's X_test exactly as
                    returned by core.data.load_task_data (post-drop);
                    text_sha256 is sha256(X_test[item_idx].encode("utf-8")).
    Returns the list of paths written.
    """
    import csv as _csv
    import hashlib as _hashlib

    out_dir = Path(results_dir) / "per_item"
    paths = []
    for row in rows:
        per_item = row.get("per_item")
        if per_item is None:
            continue
        task = row["task"]
        candidate = row["candidate"]
        X_test = X_test_by_task[task]
        sha = [_hashlib.sha256(t.encode("utf-8")).hexdigest() for t in X_test]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task}__{candidate}.csv"
        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=PER_ITEM_COLUMNS)
            writer.writeheader()
            for rec in sorted(per_item, key=lambda d: (d["repeat"], d["item_idx"])):
                idx = rec["item_idx"]
                writer.writerow({
                    "task": task,
                    "candidate": candidate,
                    "repeat": rec["repeat"],
                    "fold": rec["fold"],
                    "item_idx": idx,
                    "text_sha256": sha[idx],
                    "gold": rec["gold"],
                    "pred": rec["pred"],
                    "p_pred": rec["p_pred"],
                    "kept": rec["kept"],
                    "set_size": rec["set_size"],
                })
        paths.append(path)
    return paths


def write_all(rows, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks_csv = write_tasks_csv(rows, results_dir)
    summary_md = write_summary_md(rows, results_dir)
    chart_png = write_chart(rows, results_dir)
    per_class = write_per_class_csvs(rows, results_dir)
    configs = write_configs(rows, results_dir)
    return {
        "tasks_csv": tasks_csv,
        "summary_md": summary_md,
        "chart_png": chart_png,
        "per_class": per_class,
        "configs": configs,
    }
