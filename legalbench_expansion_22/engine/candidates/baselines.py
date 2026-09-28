"""
candidates/baselines.py

Two trivial baseline candidates sharing the fixed
fit_predict_proba(X_train, y_train, X_test, classes, *, seed, task_key,
cache_dir) -> np.ndarray interface, hosted in one module since neither
needs anything beyond y_train's class frequencies.

  - "majority": always predicts the most frequent class in y_train, as a
    one-hot vector for every test row (an honest confidence for a
    constant rule -- it is not overconfident about classes it never
    considered, it is simply certain about the single class it always
    outputs).
  - "stratified_random": predicts each class with probability equal to
    its frequency in y_train, the SAME frequency vector repeated for
    every test row. This is a frequency vector directly, not a sampled
    one-hot draw -- deliberately: it is simplest, avoids adding sampling
    noise/a second RNG dependency, and is an equally valid "stratified
    random" baseline (its expectation under any proper scoring rule
    matches actually sampling one-hot labels at those frequencies, but
    with zero added variance from the harness's side).

Both are exposed via the module-level CANDIDATES dict (name -> callable)
because this one file hosts two distinct candidates; cli.py's dynamic
loader checks for a CANDIDATES dict first and falls back to a single
NAME/fit_predict_proba pair for every other (one-candidate-per-file)
candidate module.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

EPS = 1e-9


def _class_counts(y_train, classes) -> np.ndarray:
    counts = Counter(y_train)
    return np.array([counts.get(c, 0) for c in classes], dtype=np.float64)


def fit_predict_proba(X_train, y_train, X_test, classes, *, seed, task_key, cache_dir):
    """majority: one-hot on the most frequent class in y_train. Ties are
    broken deterministically by (-count, class) order, i.e. the
    alphabetically-first class among those tied for most frequent, so
    output is identical across runs given identical inputs."""
    counts = _class_counts(y_train, classes)
    n = len(classes)
    if counts.sum() == 0:
        # Degenerate: no training labels at all. Rather than crash, fall
        # back to a uniform distribution -- still an "honest" probability,
        # just reflecting total ignorance instead of a majority vote.
        row = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        order = sorted(range(n), key=lambda i: (-counts[i], classes[i]))
        best_i = order[0]
        row = np.full(n, EPS, dtype=np.float64)
        row[best_i] = 1.0
        row = row / row.sum()
    return np.tile(row, (len(X_test), 1))


def fit_predict_proba_stratified_random(X_train, y_train, X_test, classes, *, seed, task_key, cache_dir):
    """stratified_random: the y_train class-frequency vector, repeated for
    every test row (see module docstring for why this is a frequency
    vector rather than an actual sampling draw)."""
    counts = _class_counts(y_train, classes)
    n = len(classes)
    if counts.sum() == 0:
        row = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        row = counts + EPS
        row = row / row.sum()
    return np.tile(row, (len(X_test), 1))


CANDIDATES = {
    "majority": fit_predict_proba,
    "stratified_random": fit_predict_proba_stratified_random,
}

NAME = "majority"  # single-candidate fallback identity for this module


if __name__ == "__main__":
    classes = ["a", "b", "c"]
    y_train = ["a", "a", "b", "c", "c", "c"]
    X_test = ["x1", "x2", "x3"]

    maj = fit_predict_proba([], y_train, X_test, classes, seed=0, task_key="t", cache_dir="/tmp")
    assert maj.shape == (3, 3)
    assert np.allclose(maj.sum(axis=1), 1.0)
    assert np.argmax(maj[0]) == 2  # "c" is most frequent

    strat = fit_predict_proba_stratified_random([], y_train, X_test, classes, seed=0, task_key="t", cache_dir="/tmp")
    assert strat.shape == (3, 3)
    assert np.allclose(strat.sum(axis=1), 1.0)
    assert np.allclose(strat[0], strat[1])  # same vector every row
    print("OK: baselines self-test passed")
