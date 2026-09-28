"""
core/data.py

Loads the task registry and, for a given registered task, the LegalBench
train+test splits from the Hugging Face Hub (`nguha/legalbench`, config =
registry's `hf_config`).

Registry schema (data/task_registry.json), one object per task, either as
a top-level JSON list or as {"tasks": [...]}. Fields used by this module:

    task                    str, task id (registry key / display name)
    hf_config               str, the `nguha/legalbench` config name to load
    family                  str, task family label (informational)
    is_reasoning_exception  bool, informational registry flag; this module
                            and the CV harness do not change behavior based
                            on it -- it is only carried through into
                            outputs, since no instruction ties it to a
                            behavior change.
    n_train / n_test        int, expected split sizes (sanity-checked, not
                            enforced -- a mismatch is logged, not fatal)
    input_columns           list[str], dataset columns concatenated (in
                            this order) with " [SEP] " to build model input
    n_classes               int, expected number of classes (sanity-checked)
    class_distribution      dict[str, int|float], keys are the CANONICAL
                            normalized (strip + lower) label set for the
                            task. Any row whose normalized `answer` is not
                            one of these keys is dropped (and counted).
    metric                  str, one of core.metrics.METRIC_FUNCS keys
    notes                   str, informational

Any other registry fields are ignored by this module but preserved on the
returned TaskSpec.raw dict for downstream use (e.g. outputs/config dumps).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("legalbench_map.data")

SEP = " [SEP] "


@dataclass
class TaskSpec:
    task: str
    hf_config: str
    family: str
    input_columns: list
    classes: list  # sorted, normalized class labels
    metric: str
    n_train_expected: int | None
    n_test_expected: int | None
    is_reasoning_exception: bool
    notes: str
    raw: dict = field(default_factory=dict)


@dataclass
class TaskData:
    spec: TaskSpec
    X_train: list  # list[str]
    y_train: list  # list[str] normalized labels
    X_test: list
    y_test: list
    dropped_train: int
    dropped_test: int


def _normalize_label(x: Any) -> str:
    return str(x).strip().lower()


def load_registry(path) -> list[dict]:
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    elif isinstance(data, list):
        tasks = data
    else:
        raise ValueError(
            f"task_registry.json at {path} must be a JSON list of task objects "
            f"or an object with a 'tasks' key; got {type(data)}"
        )
    return tasks


def parse_task_spec(entry: dict) -> TaskSpec:
    missing = [k for k in ("task", "hf_config", "input_columns", "class_distribution", "metric") if k not in entry]
    if missing:
        raise ValueError(f"task registry entry missing required field(s) {missing}: {entry}")
    class_dist = entry["class_distribution"]
    classes = sorted(_normalize_label(k) for k in class_dist.keys())
    if len(classes) != len(set(classes)):
        raise ValueError(f"task {entry['task']!r}: class_distribution keys collide after normalization")
    return TaskSpec(
        task=entry["task"],
        hf_config=entry["hf_config"],
        family=entry.get("family", ""),
        input_columns=list(entry["input_columns"]),
        classes=classes,
        metric=entry["metric"],
        n_train_expected=entry.get("n_train"),
        n_test_expected=entry.get("n_test"),
        is_reasoning_exception=bool(entry.get("is_reasoning_exception", False)),
        notes=entry.get("notes", ""),
        raw=dict(entry),
    )


def _row_text(row: dict, input_columns: list) -> str:
    parts = []
    for col in input_columns:
        val = row.get(col, "")
        if val is None:
            val = ""
        parts.append(str(val))
    return SEP.join(parts)


def _build_split(dataset_split, spec: TaskSpec, split_name: str):
    X, y = [], []
    dropped = 0
    class_set = set(spec.classes)
    for row in dataset_split:
        label = _normalize_label(row.get("answer", ""))
        if label not in class_set:
            dropped += 1
            continue
        X.append(_row_text(row, spec.input_columns))
        y.append(label)
    if dropped:
        logger.warning(
            "task=%s split=%s dropped %d row(s) with a normalized label not in the "
            "registered class_distribution keys %s",
            spec.task,
            split_name,
            dropped,
            spec.classes,
        )
    return X, y, dropped


def load_task_data(entry: dict, *, hf_load_dataset=None) -> TaskData:
    """entry: one task_registry.json object (already-parsed dict).
    hf_load_dataset: injectable for tests; defaults to
    datasets.load_dataset.
    """
    spec = parse_task_spec(entry)

    if hf_load_dataset is None:
        from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset("nguha/legalbench", spec.hf_config)

    if "train" not in ds or "test" not in ds:
        raise ValueError(
            f"task {spec.task!r} (hf_config={spec.hf_config!r}): expected both 'train' and "
            f"'test' splits, got {list(ds.keys())}"
        )

    X_train, y_train, dropped_train = _build_split(ds["train"], spec, "train")
    X_test, y_test, dropped_test = _build_split(ds["test"], spec, "test")

    if spec.n_train_expected is not None and len(X_train) + dropped_train != spec.n_train_expected:
        logger.warning(
            "task=%s: registry n_train=%s but raw train split has %d rows",
            spec.task,
            spec.n_train_expected,
            len(X_train) + dropped_train,
        )
    if spec.n_test_expected is not None and len(X_test) + dropped_test != spec.n_test_expected:
        logger.warning(
            "task=%s: registry n_test=%s but raw test split has %d rows",
            spec.task,
            spec.n_test_expected,
            len(X_test) + dropped_test,
        )

    if len(X_test) == 0:
        raise ValueError(f"task {spec.task!r}: zero usable test rows after label filtering")

    return TaskData(
        spec=spec,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dropped_train=dropped_train,
        dropped_test=dropped_test,
    )
