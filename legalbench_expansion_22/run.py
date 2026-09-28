#!/usr/bin/env python3
"""Run only the isolated LegalBench expansion cohort, using local pinned data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
COHORT = "legalbench_expansion_22"
CANDIDATES = ("majority", "tfidf_logreg")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(task, split):
    with (ROOT / "data" / "raw" / task / f"{split}.tsv").open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def prepare_task(entry, splits):
    """Remove exact duplicates without changing the pinned source files."""
    fields = entry["input_columns"]
    key = lambda row: tuple(row[f] for f in fields)
    cleaned = {}
    for split, rows in splits.items():
        seen = {}
        unique = []
        for row in rows:
            text = key(row)
            label = row["answer"].strip().lower()
            if text in seen:
                if seen[text] != label:
                    raise ValueError(f"Conflicting labels for identical input: {entry['task']}/{split}")
                continue
            seen[text] = label
            unique.append(row)
        cleaned[split] = unique
    test_inputs = {key(row) for row in cleaned["test"]}
    cleaned["train"] = [row for row in cleaned["train"] if key(row) not in test_inputs]
    effective = dict(entry)
    effective["n_train"] = len(cleaned["train"])
    effective["n_test"] = len(cleaned["test"])
    effective["class_distribution"] = dict(Counter(
        row["answer"].strip().lower() for row in cleaned["test"]
    ))
    preprocessing = {
        "task": entry["task"],
        "raw_train": len(splits["train"]), "raw_test": len(splits["test"]),
        "effective_train": len(cleaned["train"]), "effective_test": len(cleaned["test"]),
        "removed_train_rows": len(splits["train"]) - len(cleaned["train"]),
        "removed_duplicate_test_rows": len(splits["test"]) - len(cleaned["test"]),
    }
    return effective, cleaned, preprocessing


def verify_registry(registry, selection, original_tasks):
    names = [r["task"] for r in registry]
    if len(names) != 22 or len(set(names)) != 22:
        raise ValueError("The expansion registry must contain exactly 22 distinct tasks.")
    if set(names) & set(original_tasks):
        raise ValueError("Expansion tasks overlap the original cohort.")
    if set(names) != set(selection["tasks"]) or selection["cohort"] != COHORT:
        raise ValueError("Registry differs from the frozen expansion selection.")
    for entry in registry:
        if entry.get("experiment") != COHORT or entry["metric"] != "balanced_accuracy":
            raise ValueError(f"Incorrect experiment or metric for {entry['task']}.")


def select_tasks(registry, requested, smoke):
    names = {r["task"] for r in registry}
    wanted = set(requested.split(",")) if requested else (
        {"function_of_decision_section"} if smoke else names
    )
    if not wanted <= names:
        raise ValueError("Unknown or original-cohort task(s): " + ", ".join(sorted(wanted - names)))
    if smoke and len(wanted) != 1:
        raise ValueError("A smoke check must select exactly one task.")
    return [r for r in registry if r["task"] in wanted]


def reserve_output(root, name, smoke):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
        raise ValueError("Run name must be a simple name, without slashes or path traversal.")
    root = Path(root).resolve()
    base = root / ("smoke_results" if smoke else "results")
    # Reject even a local symlink: the path is dedicated to this experiment.
    if base.is_symlink() or base.resolve().parent != root:
        raise ValueError("Results directory must remain inside this experiment.")
    target = base / name
    if target.is_symlink() or target.resolve().parent != base.resolve():
        raise ValueError("Run output cannot escape the experiment.")
    base.mkdir(exist_ok=True)
    target.mkdir(exist_ok=False)  # Never replace an earlier run.
    return target


def validate():
    registry = read_json(ROOT / "data/task_registry.json")
    selection = read_json(ROOT / "data/selection.json")
    original = read_json(ROOT / "data/original_37_task_names.json")
    verify_registry(registry, selection, original)
    live_original = ROOT.parent / "legalbench_map/data/task_registry.json"
    if live_original.exists():
        verify_registry(registry, selection, [r["task"] for r in read_json(live_original)])
    manifest = read_json(ROOT / "data/source_manifest.json")
    if manifest["errors"] or len(manifest["sources"]) != 66:
        raise ValueError("Expected 44 complete data files and 22 original prompts.")
    for rel, source in manifest["sources"].items():
        if sha256(ROOT / rel) != source["sha256"]:
            raise ValueError(f"Source file has changed: {rel}")
    engine = read_json(ROOT / "data/engine_provenance.json")
    for rel, fingerprint in engine["files_sha256"].items():
        if sha256(ROOT / "engine" / rel) != fingerprint:
            raise ValueError(f"Frozen evaluation code has changed: {rel}")
    placeholders = read_json(ROOT / "data/prompt_placeholders.json")
    details = []
    for entry in registry:
        task = entry["task"]
        prompt = (ROOT / "data/task_instructions" / f"{task}.txt").read_text()
        fields = list(dict.fromkeys(re.findall(r"\{\{\s*([^{}]+?)\s*\}\}", prompt)))
        if set(fields) != set(entry["input_columns"]) or fields != placeholders[task]:
            raise ValueError(f"Input fields differ from the original prompt: {task}")
        splits = {split: read_split(task, split) for split in ("train", "test")}
        for split, rows in splits.items():
            if len(rows) != entry[f"n_{split}"]:
                raise ValueError(f"Wrong row count: {task}/{split}")
            for row in rows:
                if not set(fields) <= row.keys():
                    raise ValueError(f"Missing prompt field: {task}/{split}")
                if row["answer"].strip().lower() not in entry["class_distribution"]:
                    raise ValueError(f"Unexpected target label: {task}/{split}")
        counts = Counter(r["answer"].strip().lower() for r in splits["test"])
        if dict(counts) != entry["class_distribution"] or len(counts) != entry["n_classes"]:
            raise ValueError(f"Target distribution differs from the registry: {task}")
        # Describe limitations of the inherited row-wise CV; do not hide duplicates.
        texts = {
            split: [" [SEP] ".join(r[f] for f in fields) for r in rows]
            for split, rows in splits.items()
        }
        details.append({
            "task": task, "n_train": entry["n_train"], "n_test": entry["n_test"],
            "n_classes": len(counts), "min_test_class_size": min(counts.values()),
            "duplicate_test_text_rows": len(texts["test"]) - len(set(texts["test"])),
            "train_test_shared_texts": len(set(texts["train"]) & set(texts["test"])),
        })
    return registry, {"cohort": COHORT, "task_count": len(registry), "tasks": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List the 22 tasks without fitting.")
    mode.add_argument("--validate", action="store_true", help="Check inputs, labels, hashes and cohort separation.")
    mode.add_argument("--smoke", action="store_true", help="Check wiring on six items per class; not a benchmark score.")
    parser.add_argument("--tasks", help="Comma-separated subset of the 22 expansion tasks.")
    parser.add_argument("--run-name", help="Unique output folder name; existing runs cannot be overwritten.")
    parser.add_argument("--folds", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", default="majority,tfidf_logreg")
    args = parser.parse_args(argv)
    if args.list:
        for row in read_json(ROOT / "data/task_registry.json"):
            print(f"{row['task']}  train={row['n_train']}  test={row['n_test']}")
        return 0
    registry, report = validate()
    print("Validated all 22 expansion tasks; no overlap with the original 37.", flush=True)
    if args.validate:
        print(json.dumps(report, indent=2))
        return 0
    entries = select_tasks(registry, args.tasks, args.smoke)
    candidates = args.candidates.split(",")
    if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(CANDIDATES):
        raise ValueError("Choose majority and/or tfidf_logreg.")
    if args.smoke and (args.folds is not None or args.repeats is not None):
        raise ValueError("Smoke checks use two folds and one repeat.")
    folds = 2 if args.smoke else (args.folds if args.folds is not None else 5)
    repeats = 1 if args.smoke else (args.repeats if args.repeats is not None else 3)
    if folds < 2 or repeats < 1:
        raise ValueError("At least two folds and one repeat are required.")
    local_data = {}
    preprocessing = []
    effective_entries = []
    for entry in entries:
        task = entry["task"]
        entry, local_data[task], preparation = prepare_task(
            entry, {split: read_split(task, split) for split in ("train", "test")}
        )
        preprocessing.append(preparation)
        if args.smoke:
            # Deterministic tiny wiring check after the same deduplication.
            counts = Counter()
            subset = []
            for row in local_data[task]["test"]:
                label = row["answer"].strip().lower()
                if counts[label] < 6:
                    subset.append(row)
                    counts[label] += 1
            local_data[task]["test"] = subset
            entry["n_test"] = len(subset)
            entry["class_distribution"] = dict(counts)
        if min(entry["class_distribution"].values()) < folds:
            raise ValueError(f"Too many folds for the least frequent class: {task}")
        effective_entries.append(entry)
    entries = effective_entries
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = reserve_output(ROOT, name, args.smoke)
    write_json(output / "registry.json", entries)
    write_json(output / "input_validation.json", report)
    run_manifest = {
        "cohort": COHORT, "status": "running", "smoke_only": args.smoke,
        "eligible_for_benchmark_reporting": not args.smoke,
        "protocol": "supervised row-wise CV over benchmark test split; original train examples added to each fold",
        "tasks": [r["task"] for r in entries], "candidates": candidates,
        "folds": folds, "repeats": repeats, "seed": args.seed,
        "registry_sha256": sha256(output / "registry.json"),
        "source_manifest_sha256": sha256(ROOT / "data/source_manifest.json"),
        "engine_provenance_sha256": sha256(ROOT / "data/engine_provenance.json"),
        "runner_sha256": sha256(__file__),
        "preprocessing": preprocessing,
        "llm_comparison": "not_evaluated; no published references added",
        "limitations": [
            "Supervised adaptation uses more labelled examples than the original few-shot benchmark.",
            "Exact input duplicates are removed, but inherited row-wise folds do not isolate source documents or near-duplicate text.",
            "Inherited confidence intervals pool repeated predictions; they are not validated independence-aware intervals.",
            "Inherited per_item predictions are from the conformal model, not the full-training model used for cv_mean.",
        ],
    }
    write_json(output / "manifest.json", run_manifest)
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    sys.path.insert(0, str(ROOT / "engine"))
    try:
        spec = importlib.util.spec_from_file_location("expansion_harness", ROOT / "engine/cli.py")
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        legacy_loader = harness.load_task_data

        def local_loader(entry):
            def get_local(_dataset_name, config):
                return local_data[config]
            return legacy_loader(entry, hf_load_dataset=get_local)

        # Inject local TSVs into the existing loader; no Hub calls or shared caches.
        harness.load_task_data = local_loader
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            code = harness.main([
                "--registry", str(output / "registry.json"),
                "--published-scores", str(ROOT / "data/published_scores.csv"),
                "--results-dir", str(output), "--cache-dir", str(ROOT / ".cache"),
                "--candidates-dir", str(ROOT / "engine/candidates"),
                "--candidates", ",".join(candidates),
                "--folds", str(folds), "--repeats", str(repeats),
                "--seed", str(args.seed), "--dump-per-item",
            ])
        with (output / "tasks.csv").open() as f:
            results = list(csv.DictReader(f))
        expected = {(e["task"], c) for e in entries for c in candidates}
        actual = [(r["task"], r["candidate"]) for r in results]
        if code or len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError("Incomplete run; see run_errors.log. Results are not marked complete.")
        if (output / "run_errors.log").read_text().strip():
            raise RuntimeError("The harness logged errors; results are not marked complete.")
        if not all(math.isfinite(float(r["cv_mean_CV_estimated"])) for r in results):
            raise RuntimeError("Non-finite main score; results are not marked complete.")
        run_manifest["status"] = "complete"
        run_manifest["task_candidate_rows"] = len(results)
        write_json(output / "manifest.json", run_manifest)
        if args.smoke:
            (output / "SMOKE_ONLY.txt").write_text("Wiring check only. These are not full benchmark results.\n")
        print(f"Completed {'smoke check' if args.smoke else 'expansion run'}: {output}")
        return 0
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = str(exc)
        write_json(output / "manifest.json", run_manifest)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
