"""Shared synthetic-data helpers for tests (no network required)."""
from __future__ import annotations

from core.data import TaskData, TaskSpec


class CallableModule:
    """Minimal wrapper so a bare fit_predict_proba function can be passed
    wherever core.cv expects a "candidate module" object."""

    def __init__(self, fn):
        self.fit_predict_proba = fn


def make_synthetic_task_data(n_train=20, n_test=40, n_classes=4, metric="accuracy", seed=0):
    classes = sorted(f"c{i}" for i in range(n_classes))

    def gen(n):
        X, y = [], []
        for i in range(n):
            cls = classes[i % n_classes]
            X.append(f"synthetic document {i} labeled {cls} seed {seed}")
            y.append(cls)
        return X, y

    X_train, y_train = gen(n_train)
    X_test, y_test = gen(n_test)

    spec = TaskSpec(
        task="synthetic_task",
        hf_config="synthetic",
        family="synthetic",
        input_columns=["text"],
        classes=classes,
        metric=metric,
        n_train_expected=n_train,
        n_test_expected=n_test,
        is_reasoning_exception=False,
        notes="synthetic fixture for tests",
        raw={},
    )
    return TaskData(
        spec=spec,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dropped_train=0,
        dropped_test=0,
    )
