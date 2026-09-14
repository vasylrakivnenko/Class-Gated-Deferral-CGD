"""Runs the real CLI (subprocess) on the small real 'abercrombie' task with
--dump-per-item and checks the per-item dump (Phase 2 "FILE A") against
the data loader and against tasks.csv; also checks that the flag leaves
tasks.csv byte-identical."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from core.data import load_task_data

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/Users/vasyl/zadumai/.venv/bin/python"
REPEATS = 2
CANDIDATES = ["majority", "tfidf_logreg"]
EXPECTED_COLUMNS = [
    "task", "candidate", "repeat", "fold", "item_idx", "text_sha256",
    "gold", "pred", "p_pred", "kept", "set_size",
]

ENTRY = {
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


def _run_cli(tmp_path, tag, dump: bool):
    registry_path = tmp_path / "task_registry.json"
    with open(registry_path, "w") as f:
        json.dump([ENTRY], f)
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
        "--candidates", ",".join(CANDIDATES),
        "--folds", "5",
        "--repeats", str(REPEATS),
        "--seed", "0",
    ]
    if dump:
        cmd.append("--dump-per-item")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 0, f"cli.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return results_dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("per_item")
    with_dump = _run_cli(tmp_path, "with", dump=True)
    without_dump = _run_cli(tmp_path, "without", dump=False)
    return with_dump, without_dump


@pytest.fixture(scope="module")
def task_data():
    return load_task_data(ENTRY)


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_flag_off_leaves_tasks_csv_byte_identical_and_writes_no_dump(runs):
    with_dump, without_dump = runs
    assert (with_dump / "tasks.csv").read_bytes() == (without_dump / "tasks.csv").read_bytes()
    assert not (without_dump / "per_item").exists()


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_per_item_dump_contents(runs, task_data, candidate):
    with_dump, _ = runs
    n_test = len(task_data.X_test)
    path = with_dump / "per_item" / f"abercrombie__{candidate}.csv"
    assert path.exists(), f"missing per-item dump {path}"

    fieldnames, rows = _read_csv(path)
    assert fieldnames == EXPECTED_COLUMNS
    assert len(rows) == REPEATS * n_test

    # every item_idx appears exactly once per repeat
    per_repeat = Counter((int(r["repeat"]), int(r["item_idx"])) for r in rows)
    assert set(per_repeat) == {(rep, i) for rep in range(REPEATS) for i in range(n_test)}
    assert set(per_repeat.values()) == {1}

    sha_expected = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in task_data.X_test]
    classes = set(task_data.spec.classes)
    for r in rows:
        assert r["task"] == "abercrombie"
        assert r["candidate"] == candidate
        idx = int(r["item_idx"])
        assert r["kept"] in ("0", "1")
        set_size = int(r["set_size"])
        assert set_size >= 0
        assert (r["kept"] == "1") == (set_size == 1)
        assert 0 <= int(r["fold"]) < 5
        assert r["gold"] == task_data.y_test[idx]
        assert r["text_sha256"] == sha_expected[idx]
        assert r["pred"] in classes
        p = float(r["p_pred"])
        assert 0.0 <= p <= 1.0 + 1e-9

    # cross-check: mean(kept) over the dump == keep_rate_mean in tasks.csv
    _, task_rows = _read_csv(with_dump / "tasks.csv")
    agg = [t for t in task_rows if t["task"] == "abercrombie" and t["candidate"] == candidate]
    assert len(agg) == 1
    keep_rate_mean = float(agg[0]["keep_rate_mean"])
    kept_mean = sum(int(r["kept"]) for r in rows) / len(rows)
    assert abs(kept_mean - keep_rate_mean) < 1e-9
