"""Runs the CLI twice (subprocess, same seed) on a tiny real LegalBench
task (abercrombie: 5 train / 95 test rows, 5 classes) with only fast
candidates (majority + tfidf_logreg) and asserts byte-identical
tasks.csv."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/Users/vasyl/zadumai/.venv/bin/python"


def _write_registry(path):
    registry = [
        {
            "task": "abercrombie",
            "hf_config": "abercrombie",
            "family": "exception",
            "is_reasoning_exception": True,
            "n_train": 5,
            "n_test": 95,
            "input_columns": ["text"],
            "n_classes": 5,
            "class_distribution": {
                "generic": 19, "descriptive": 19, "suggestive": 19,
                "arbitrary": 19, "fanciful": 19,
            },
            "metric": "balanced_accuracy",
            "notes": "small real LegalBench task used for fast/deterministic tests",
        }
    ]
    with open(path, "w") as f:
        json.dump(registry, f)


def _run_cli(tmp_path, tag):
    registry_path = tmp_path / "task_registry.json"
    _write_registry(registry_path)
    published_path = tmp_path / "published_scores.csv"
    published_path.write_text("task,model,metric,score,source_url\n")
    results_dir = tmp_path / f"results_{tag}"
    cache_dir = tmp_path / "cache"

    cmd = [
        PYTHON, str(ROOT / "cli.py"),
        "--registry", str(registry_path),
        "--published-scores", str(published_path),
        "--results-dir", str(results_dir),
        "--cache-dir", str(cache_dir),
        "--candidates", "majority,tfidf_logreg",
        "--folds", "5",
        "--repeats", "2",
        "--seed", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 0, f"cli.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return results_dir / "tasks.csv"


def test_two_runs_same_seed_byte_identical(tmp_path):
    csv_a = _run_cli(tmp_path, "a")
    csv_b = _run_cli(tmp_path, "b")

    bytes_a = csv_a.read_bytes()
    bytes_b = csv_b.read_bytes()
    assert bytes_a == bytes_b, "tasks.csv differs between two runs with the same --seed"
    assert len(bytes_a) > 0
