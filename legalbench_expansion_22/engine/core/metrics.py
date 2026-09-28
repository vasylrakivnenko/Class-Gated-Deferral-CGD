"""
core/metrics.py

Metric implementations for the LegalBench cheap-model-map harness.

`balanced_accuracy` is implemented by hand here (mirroring
sklearn.metrics.balanced_accuracy_score's own algorithm: mean of per-class
recall over the classes actually present in y_true) because the spec asks
for a from-scratch implementation that is tested against sklearn's. It is
exercised in tests/test_metrics.py on many random trials.

`accuracy` and `macro_f1` are well-defined, unambiguous aggregate metrics
with no reimplementation ambiguity, so we compute them directly via numpy /
sklearn rather than duplicating sklearn's own (already-correct) code for no
reason -- the spec's "implement by hand" instruction is specific to
conformal prediction (core/conformal.py) and to balanced accuracy (this
file), not to every metric.

All metric functions take (y_true, y_pred, labels) where `labels` is the
task's full, fixed class list (so metrics are computed consistently even
when a fold's y_true/y_pred happen not to touch every class).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def balanced_accuracy(y_true, y_pred, labels) -> float:
    """Mean of per-class recall, restricted to classes with nonzero support
    in y_true -- this is exactly what sklearn.metrics.balanced_accuracy_score
    computes (recall_score(..., average='macro') restricted to seen
    classes), reimplemented from the confusion matrix by hand.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(labels)
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    C = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if t not in idx:
            raise KeyError(f"true label {t!r} not in provided labels {labels}")
        if p not in idx:
            raise KeyError(f"predicted label {p!r} not in provided labels {labels}")
        C[idx[t], idx[p]] += 1.0
    row_sums = C.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.diag(C) / row_sums
    per_class_recall = per_class_recall[row_sums > 0]
    if len(per_class_recall) == 0:
        return float("nan")
    return float(np.mean(per_class_recall))


def accuracy(y_true, y_pred, labels=None) -> float:
    return float(accuracy_score(list(y_true), list(y_pred)))


def macro_f1(y_true, y_pred, labels) -> float:
    return float(
        f1_score(list(y_true), list(y_pred), labels=list(labels), average="macro", zero_division=0)
    )


METRIC_FUNCS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
    "f1_macro": macro_f1,
}


def get_metric(name: str):
    if name is None:
        raise KeyError("metric name is None")
    key = name.strip().lower().replace("-", "_")
    if key not in METRIC_FUNCS:
        raise KeyError(f"Unknown metric {name!r}. Known metrics: {sorted(METRIC_FUNCS)}")
    return METRIC_FUNCS[key]


def per_class_recall(y_true, y_pred, labels) -> dict:
    """Recall for every class in `labels` (nan if the class has zero
    support in y_true -- used for the per-class outputs, kept separate from
    balanced_accuracy's "drop unseen classes" behavior since per-class
    tables should show every class, even ones with no support)."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(labels)
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    C = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        C[idx[t], idx[p]] += 1.0
    row_sums = C.sum(axis=1)
    out = {}
    for c in labels:
        i = idx[c]
        out[c] = float(np.diag(C)[i] / row_sums[i]) if row_sums[i] > 0 else float("nan")
    return out
