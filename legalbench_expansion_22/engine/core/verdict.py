"""
core/verdict.py

Published-score loading, the "estimated cascade" score, and the per-task
match/close/short/no_reference verdict.

=====================================================================
ESTIMATED CASCADE ASSUMPTION (state this wherever the number is shown)
=====================================================================
    est_cascade_metric = keep_rate * metric_on_kept
                          + (1 - keep_rate) * published_best_llm_score

`published_best_llm_score` is the MAX score across all published models
in data/published_scores.csv for that task (best available LLM score).
This is an ESTIMATE, not a measured cascade: sent items are assumed to be
answered at the LLM's published AVERAGE score, which may be optimistic,
since sent items are exactly the ones the conformal gate flagged as
uncertain -- i.e. selected to be harder, on average, than a random draw
from the test set. A real cascade could score lower on that subset than
the task-average published number suggests.

For balanced_accuracy specifically, the estimate is computed PER CLASS
and then averaged:
    per_class_est[c] = per_class_keep_rate[c] * per_class_recall_kept[c]
                        + (1 - per_class_keep_rate[c]) * published_best_llm_score
    est_cascade_metric = mean_c per_class_est[c]
This assumes the published (task-average) score applies UNIFORMLY across
every class -- we do not have a per-class published number, so this is an
explicit, stated approximation, not a measured per-class LLM score.

=====================================================================
CV-vs-PUBLISHED COMPARABILITY ASSUMPTION
=====================================================================
Published LLM scores (data/published_scores.csv) were computed by their
original authors on the full, fixed LegalBench test split with whatever
methodology they used (typically a single pass, no CV). Our CV-estimated
numbers evaluate the SAME test-split items but via repeated stratified
5-fold cross-validation (see core/cv.py) so every item gets an
out-of-fold prediction. We treat the two as comparable estimates of
"accuracy/metric on the LegalBench test split for this task" -- this is
an assumption, not a guarantee: differences in prompt, exact evaluation
harness, or minor label-normalization choices between the published
source and this pipeline are not accounted for. Every number derived
this way is labeled "CV-estimated" in the outputs to keep this
assumption visible rather than presenting a bare, unqualified number.

=====================================================================
VERDICT RULE (delta points, default delta=1.0)
=====================================================================
Let gap = published_best - cv_mean (candidate's CV-estimated mean on the
task's primary metric, in percentage points, i.e. score * 100).
  * no_reference : no published score exists for the task
  * match         : cv_ci_low >= published_best - delta
  * short         : (not match) and gap > 3
  * close         : (not match) and gap <= 3
    (this also covers the edge case of a small gap with a wide/unstable
    CI that just misses the "match" CI test -- there is no gap in the
    real number line left uncovered by these four buckets)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger("legalbench_map.verdict")


def load_published_scores(path) -> dict:
    """Returns {task: {"best_score": float, "best_model": str,
    "best_source_url": str, "metric": str, "by_model": {model: score},
    "by_model_urls": {model: source_url}, "by_metric": {metric:
    {model: score}}}}.

    "best_source_url" is the citation for "best_score"/"best_model" --
    every published number in this project traces to a concrete source, and
    the number that actually drives verdicts/summary.md is no exception.

    A task can legitimately have MULTIPLE metrics recorded for the same
    model (e.g. the 9 reasoning-exception tasks carry both a manual
    correctness/analysis grading AND balanced_accuracy -- see
    PUBLISHED_SCORES_NOTES.md). Rows are therefore grouped by
    (task, metric) FIRST; by_model/best_score/best_model are then derived
    from only the metric with the WIDEST model coverage for that task
    (ties broken alphabetically by metric name, for determinism) --
    that is the metric comparable to this harness's CV numbers (which are
    always a single fixed metric per task, per the registry), and never
    naively across metrics. This is independent of CSV row order: keying
    by (task, model) alone and overwriting on every row -- picking
    whichever metric happened to be written last for that model --
    would silently conflate incomparable metrics (e.g. a 0-1
    "correctness" grade vs. balanced_accuracy) if the file were ever
    reordered or extended. All other metrics recorded for a task are
    still returned under "by_metric" for transparency/debugging, they are
    just not used to compute best_score/by_model.

    Every row of the CSV MUST have a non-empty source_url -- a row
    missing one raises ValueError immediately (published numbers without
    a traceable source are not usable evidence for this project).
    """
    path = Path(path)
    if not path.exists():
        logger.warning("published scores file not found at %s -- all tasks will be no_reference", path)
        return {}

    # task -> metric -> model -> score
    by_metric_raw: dict[str, dict[str, dict[str, float]]] = {}
    # task -> metric -> model -> source_url (kept alongside, same keys)
    by_metric_urls_raw: dict[str, dict[str, dict[str, str]]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"task", "model", "metric", "score", "source_url"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"published_scores.csv missing required column(s): {missing_cols}")
        for i, row in enumerate(reader):
            source_url = (row.get("source_url") or "").strip()
            if not source_url:
                raise ValueError(
                    f"published_scores.csv row {i} (task={row.get('task')!r}, "
                    f"model={row.get('model')!r}) has an empty/missing source_url"
                )
            task = row["task"].strip()
            model = row["model"].strip()
            metric = row["metric"].strip()
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                raise ValueError(f"published_scores.csv row {i}: non-numeric score {row.get('score')!r}")
            by_metric_raw.setdefault(task, {}).setdefault(metric, {})[model] = score
            by_metric_urls_raw.setdefault(task, {}).setdefault(metric, {})[model] = source_url

    per_task: dict[str, dict] = {}
    for task, by_metric in by_metric_raw.items():
        # widest model coverage wins; alphabetical metric name breaks ties
        # (sorting the metric names first makes max() deterministically
        # pick the alphabetically-first metric among any coverage ties)
        primary_metric = max(sorted(by_metric.keys()), key=lambda m: len(by_metric[m]))
        by_model = dict(by_metric[primary_metric])
        by_model_urls = dict(by_metric_urls_raw[task][primary_metric])
        best_model = max(by_model, key=lambda m: by_model[m])
        per_task[task] = {
            "by_model": by_model,
            "by_model_urls": by_model_urls,
            "metric": primary_metric,
            "by_metric": by_metric,
            "best_model": best_model,
            "best_score": by_model[best_model],
            "best_source_url": by_model_urls[best_model],
        }

    return per_task


def estimate_cascade(cv_result, published_best_score, classes=None):
    """cv_result: core.cv.CVResult. published_best_score: float or None.
    Returns (est_cascade_metric: float|None, assumption_note: str).
    """
    note = (
        "ESTIMATED cascade: assumes sent items score at the LLM's published "
        "AVERAGE, which may be optimistic since sent items are selected as "
        "harder-than-average by the conformal gate."
    )
    if published_best_score is None:
        return None, note

    if cv_result.metric_name == "balanced_accuracy":
        per_class_ests = []
        for c in cv_result.per_class_keep_rate:
            kr = cv_result.per_class_keep_rate[c]
            mk = cv_result.per_class_recall_kept[c]
            if kr != kr:  # keep_rate itself unknown -- genuinely can't estimate this class
                continue
            if kr == 0:
                # Nothing was ever kept for this class, so metric_on_kept is
                # undefined by construction (there's nothing to average) --
                # but the cascade estimate at kr=0 reduces exactly to
                # published_best_score regardless of mk's value, so this is a
                # fully known point, not a gap. (0 * NaN is itself NaN in
                # IEEE 754, so skipping here matters: naively computing
                # kr * mk + (1-kr) * published_best_score would silently
                # poison the whole term with NaN instead of the true 0.)
                per_class_ests.append(published_best_score)
                continue
            if mk != mk:  # kr>0 but recall_kept is NaN for some other reason -- genuinely unknown
                continue
            per_class_ests.append(kr * mk + (1 - kr) * published_best_score)
        if not per_class_ests:
            return None, note
        est = sum(per_class_ests) / len(per_class_ests)
        note += (
            " Per-class variant: computed per class as "
            "keep_rate_c * metric_on_kept_c + (1-keep_rate_c) * published_score, "
            "assuming the published TASK-AVERAGE score applies uniformly across "
            "classes (no per-class published number is available), then averaged "
            "across classes."
        )
        return est, note

    kr = cv_result.keep_rate_mean
    mk = cv_result.metric_kept_mean
    if kr != kr or mk != mk:
        return None, note
    est = kr * mk + (1 - kr) * published_best_score
    return est, note


def compute_verdict(cv_mean, cv_ci_low, published_best_score, delta: float = 1.0):
    """Returns (verdict: str, gap: float|None). Scores are on a 0-1 scale;
    gap is reported in PERCENTAGE POINTS (i.e. *100) to match the "delta
    points" framing in the spec."""
    if published_best_score is None:
        return "no_reference", None
    gap_pts = (published_best_score - cv_mean) * 100.0
    # delta is given in "points" (default 1.0 point == 0.01 on a 0-1 score scale)
    delta_score = delta / 100.0
    if cv_ci_low >= published_best_score - delta_score:
        return "match", gap_pts
    if gap_pts > 3.0:
        return "short", gap_pts
    return "close", gap_pts
