#!/usr/bin/env python
"""
cli.py -- LegalBench cheap-model-map experiment runner.

    /Users/vasyl/zadumai/.venv/bin/python cli.py [flags]

See README.md for the full methodology. Key flags:

  --tasks TASK1,TASK2        filter task_registry by task name (default: all)
  --families FAM1,FAM2       filter task_registry by family (default: all)
  --candidates NAME1,NAME2   candidate keys to run (default: all, including
                             the two baselines). Candidates are discovered
                             dynamically from candidates/*.py.
  --folds N                  stratified CV folds over the test split (5)
  --repeats N                independent fold-partition repeats (3)
  --alpha A                  conformal miscoverage rate (0.05)
  --delta D                  verdict "match" threshold, in points (1.0)
  --include-finetune         allow finetune_encoder to run (off by default);
                              even when set, finetune_encoder is skipped on
                              any task with n_test < 500
  --paired-helm               if set and no HELM per-item data is found,
                              print a note and continue (does not crash)
  --dump-per-item             also write results/per_item/<task>__<candidate>.csv
                              (one row per repeat x test item: gold, conformal
                              point pred, p_pred, kept, set_size, text_sha256).
                              Off by default; when off, every output byte is
                              identical to a run without the flag.
  --seed N                    base seed for full determinism (0)
  --registry PATH             override data/task_registry.json path
  --published-scores PATH     override data/published_scores.csv path
  --results-dir PATH          override results/ output dir
  --cache-dir PATH            override .cache/ dir passed to candidates

Determinism: same --seed, same inputs -> byte-identical results/tasks.csv
across two runs (core.cv derives every RNG seed deterministically from
--seed plus stable string keys; see core/cv.py's module docstring).

A task or candidate failure is logged to results/run_errors.log and does
NOT stop the run for other tasks/candidates.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.cv import run_task_candidate_cv  # noqa: E402
from core.data import load_registry, load_task_data  # noqa: E402
from core.outputs import write_all, write_per_item_csvs  # noqa: E402
from core.verdict import compute_verdict, estimate_cascade, load_published_scores  # noqa: E402

logger = logging.getLogger("legalbench_map.cli")


def discover_candidates(candidates_dir: Path) -> dict:
    """Dynamically imports every candidates/*.py module (except __init__)
    and registers its candidate(s): a module exposing CANDIDATES (dict
    name->callable) contributes every entry; otherwise a module exposing
    NAME + fit_predict_proba contributes that single candidate. Discovery
    order is the sorted filename order, so registration is deterministic.
    """
    registry = {}
    for path in sorted(candidates_dir.glob("*.py")):
        if path.stem in ("__init__",):
            continue
        module_name = f"legalbench_map_candidates.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            logger.exception("failed to import candidate module %s -- skipping it entirely", path)
            continue

        if hasattr(module, "CANDIDATES"):
            for name, fn in module.CANDIDATES.items():
                registry[name] = fn
        elif hasattr(module, "NAME") and hasattr(module, "fit_predict_proba"):
            registry[module.NAME] = module.fit_predict_proba
        else:
            logger.warning(
                "candidate module %s exposes neither CANDIDATES nor NAME+fit_predict_proba -- skipping",
                path,
            )
    return registry


def _parse_comma_list_against_known(raw: str, known_names) -> set:
    """Splits a comma-separated --tasks value, but is aware that at least
    one registry task name (opp115_user_access,_edit_and_deletion)
    contains a literal comma -- a plain `raw.split(",")` would shred it
    into two fragments that match nothing, silently making that task
    unreachable via --tasks. Greedily prefers the LONGEST run of raw
    comma-tokens (re-joined with ",") that exactly matches a known task
    name; falls back to the single raw token when no join matches
    (preserving today's lenient behavior of passing through unknown
    names, which the caller logs/ignores rather than erroring on)."""
    known = set(known_names)
    raw_tokens = [t.strip() for t in raw.split(",")]
    n = len(raw_tokens)
    result = []
    i = 0
    while i < n:
        matched = False
        for j in range(n, i, -1):
            candidate = ",".join(raw_tokens[i:j])
            if candidate in known:
                result.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            result.append(raw_tokens[i])
            i += 1
    return set(result)


FINETUNE_MIN_N_TEST = 500
FINETUNE_NAME = "finetune_encoder"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LegalBench cheap-model-map CV/conformal harness")
    p.add_argument("--tasks", type=str, default=None, help="comma list of task names to include")
    p.add_argument("--families", type=str, default=None, help="comma list of families to include")
    p.add_argument("--candidates", type=str, default=None, help="comma list of candidate names to run")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--include-finetune", action="store_true", default=False)
    p.add_argument("--finetune-min-n-test", type=int, default=FINETUNE_MIN_N_TEST,
                   help="skip finetune_encoder on tasks with fewer test items than this (spec default 500); "
                        "lower it deliberately for an experiment on a smaller task")
    p.add_argument("--paired-helm", action="store_true", default=False)
    p.add_argument("--dump-per-item", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--registry", type=str, default=str(ROOT / "data" / "task_registry.json"))
    p.add_argument("--published-scores", type=str, default=str(ROOT / "data" / "published_scores.csv"))
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--cache-dir", type=str, default=str(ROOT / ".cache"))
    p.add_argument("--candidates-dir", type=str, default=str(ROOT / "candidates"))
    return p




def _open_error_log(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_errors.log"
    # Fresh log every run (kept out of the determinism-tested tasks.csv, but
    # kept free of run-to-run clutter for readability).
    f = open(log_path, "w")
    return f, log_path


def _log_error(log_file, task: str, candidate: str, message: str) -> None:
    log_file.write(f"[task={task}] [candidate={candidate}] {message}\n")
    log_file.flush()
    logger.error("task=%s candidate=%s: %s", task, candidate, message)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    results_dir = Path(args.results_dir)
    error_log, error_log_path = _open_error_log(results_dir)

    if args.paired_helm:
        # No HELM per-item data source is wired into this harness; per
        # spec, note and continue rather than crash.
        print("[--paired-helm] note: no HELM per-item data source is available in this "
              "harness build; continuing without paired-HELM analysis.")

    try:
        registry_entries = load_registry(args.registry)
    except Exception as e:
        _log_error(error_log, "<registry>", "<none>", f"failed to load task registry: {e!r}")
        error_log.close()
        print(f"FATAL: could not load task registry at {args.registry}: {e!r}", file=sys.stderr)
        return 1

    known_task_names = [e.get("task") for e in registry_entries]
    task_filter = _parse_comma_list_against_known(args.tasks, known_task_names) if args.tasks else None
    family_filter = set(f.strip() for f in args.families.split(",")) if args.families else None

    filtered_entries = []
    for entry in registry_entries:
        if task_filter is not None and entry.get("task") not in task_filter:
            continue
        if family_filter is not None and entry.get("family") not in family_filter:
            continue
        filtered_entries.append(entry)

    candidate_registry = discover_candidates(Path(args.candidates_dir))
    candidate_names = sorted(candidate_registry.keys())
    if args.candidates:
        wanted = [c.strip() for c in args.candidates.split(",")]
        unknown = [c for c in wanted if c not in candidate_registry]
        for c in unknown:
            _log_error(error_log, "<candidates>", c, "requested candidate not found among discovered modules")
        candidate_names = [c for c in wanted if c in candidate_registry]

    try:
        published = load_published_scores(args.published_scores)
    except Exception as e:
        _log_error(error_log, "<published_scores>", "<none>", f"failed to load published scores: {e!r}")
        published = {}

    rows = []
    X_test_by_task = {}
    for entry in filtered_entries:
        task_name = entry.get("task", "<unknown>")
        try:
            task_data = load_task_data(entry)
        except Exception as e:
            _log_error(error_log, task_name, "<data_load>", f"failed to load task data: {e!r}")
            continue
        X_test_by_task[task_name] = task_data.X_test

        published_entry = published.get(task_name)
        published_best_score = published_entry["best_score"] if published_entry else None
        published_best_model = published_entry["best_model"] if published_entry else None
        published_by_model = published_entry["by_model"] if published_entry else {}
        published_best_source_url = published_entry["best_source_url"] if published_entry else None

        for cand_name in candidate_names:
            if cand_name == FINETUNE_NAME:
                if not args.include_finetune:
                    continue
                if task_data.spec is not None and len(task_data.X_test) < args.finetune_min_n_test:
                    logger.info(
                        "skipping %s on task=%s: n_test=%d < %d minimum for finetune_encoder",
                        FINETUNE_NAME, task_name, len(task_data.X_test), args.finetune_min_n_test,
                    )
                    continue

            candidate_module = candidate_registry[cand_name]
            try:
                cv_result = run_task_candidate_cv(
                    task_data,
                    _CallableModule(candidate_module),
                    cand_name,
                    folds=args.folds,
                    repeats=args.repeats,
                    alpha=args.alpha,
                    seed=args.seed,
                    cache_dir=args.cache_dir,
                    collect_per_item=args.dump_per_item,
                )
            except Exception as e:
                _log_error(error_log, task_name, cand_name, f"CV run failed: {e!r}")
                continue

            est_cascade, _note = estimate_cascade(cv_result, published_best_score)

            rows.append({
                "task": task_name,
                "candidate": cand_name,
                "family": entry.get("family", ""),
                "is_reasoning_exception": entry.get("is_reasoning_exception", False),
                "n_items": cv_result.n_items,
                "n_classes": cv_result.n_classes,
                "metric_name": cv_result.metric_name,
                "cv_mean_CV_estimated": cv_result.cv_mean,
                "cv_ci_low": cv_result.cv_ci_low,
                "cv_ci_high": cv_result.cv_ci_high,
                "accuracy_mean_CV_estimated": cv_result.accuracy_mean,
                "macro_f1_mean_CV_estimated": cv_result.macro_f1_mean,
                "keep_rate_mean": cv_result.keep_rate_mean,
                "keep_rate_ci_low": cv_result.keep_rate_ci_low,
                "keep_rate_ci_high": cv_result.keep_rate_ci_high,
                "coverage_mean": cv_result.coverage_mean,
                "coverage_ci_low": cv_result.coverage_ci_low,
                "coverage_ci_high": cv_result.coverage_ci_high,
                "metric_kept_mean_CV_estimated": cv_result.metric_kept_mean,
                "metric_kept_ci_low": cv_result.metric_kept_ci_low,
                "metric_kept_ci_high": cv_result.metric_kept_ci_high,
                "metric_sent_mean_CV_estimated": cv_result.metric_sent_mean,
                "metric_sent_ci_low": cv_result.metric_sent_ci_low,
                "metric_sent_ci_high": cv_result.metric_sent_ci_high,
                "est_cascade_metric_ESTIMATED": est_cascade,
                "published_best_score": published_best_score,
                "published_best_model": published_best_model,
                "published_best_source_url": published_best_source_url,
                "published_by_model": published_by_model,
                "alpha": args.alpha,
                "folds": args.folds,
                "repeats": args.repeats,
                "per_class_recall": cv_result.per_class_recall,
                "per_class_keep_rate": cv_result.per_class_keep_rate,
                "per_class_recall_kept": cv_result.per_class_recall_kept,
                "fold_thresholds": cv_result.fold_thresholds,
                "hyperparams": cv_result.hyperparams,
                # None unless --dump-per-item; ignored by every write_all
                # writer (write_tasks_csv emits FLAT_COLUMNS only).
                "per_item": cv_result.per_item,
            })

    # ---- best-candidate-per-task, verdict, gap (deterministic tie-break: candidate name) ----
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    for task_name, task_rows in by_task.items():
        task_rows.sort(key=lambda r: r["candidate"])
        best_row = max(task_rows, key=lambda r: (r["cv_mean_CV_estimated"], ) )
        # deterministic tie-break: among rows sharing the max cv_mean, pick
        # alphabetically-first candidate name.
        max_mean = best_row["cv_mean_CV_estimated"]
        tied = [r for r in task_rows if r["cv_mean_CV_estimated"] == max_mean]
        best_row = min(tied, key=lambda r: r["candidate"])
        for row in task_rows:
            row["best"] = 1 if row is best_row else 0
            verdict, gap = compute_verdict(
                row["cv_mean_CV_estimated"], row["cv_ci_low"], row["published_best_score"], delta=args.delta
            )
            row["verdict"] = verdict
            row["gap_points"] = gap

    rows.sort(key=lambda r: (r["task"], r["candidate"]))

    outputs = write_all(rows, results_dir)
    if args.dump_per_item:
        outputs["per_item"] = write_per_item_csvs(rows, results_dir, X_test_by_task)
    error_log.close()

    print(f"Wrote {len(rows)} task x candidate rows.")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    print(f"  run_errors.log: {error_log_path}")
    return 0


class _CallableModule:
    """Wraps a bare fit_predict_proba callable so it can be passed anywhere
    core.cv expects a "candidate module" (it only ever calls
    .fit_predict_proba(...) and optionally checks for
    get_last_hyperparams -- neither baseline provides the latter, so this
    thin wrapper is enough)."""

    def __init__(self, fn):
        self.fit_predict_proba = fn


if __name__ == "__main__":
    raise SystemExit(main())
