"""Our hand-rolled balanced_accuracy must match
sklearn.metrics.balanced_accuracy_score on many random trials."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from core.metrics import balanced_accuracy


@pytest.mark.parametrize("trial", range(200))
def test_balanced_accuracy_matches_sklearn(trial):
    rng = np.random.default_rng(trial)
    n_classes = rng.integers(2, 6)
    n_items = rng.integers(5, 60)
    labels = [f"c{i}" for i in range(n_classes)]

    # Skew class probabilities so some classes may end up with zero true
    # support in y_true, exercising sklearn's "drop unseen classes" path.
    class_p = rng.dirichlet(np.full(n_classes, 0.5))
    y_true = rng.choice(labels, size=n_items, p=class_p).tolist()
    y_pred = rng.choice(labels, size=n_items, p=class_p).tolist()

    ours = balanced_accuracy(y_true, y_pred, labels)
    theirs = balanced_accuracy_score(y_true, y_pred)

    assert np.isclose(ours, theirs, atol=1e-9), f"trial={trial}: {ours} != {theirs}"


def test_balanced_accuracy_perfect_prediction():
    labels = ["a", "b", "c"]
    y_true = ["a", "a", "b", "b", "c", "c"]
    assert np.isclose(balanced_accuracy(y_true, y_true, labels), 1.0)


def test_balanced_accuracy_all_wrong_two_class():
    labels = ["a", "b"]
    y_true = ["a", "a", "b", "b"]
    y_pred = ["b", "b", "a", "a"]
    assert np.isclose(balanced_accuracy(y_true, y_pred, labels), 0.0)
