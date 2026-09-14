"""
Candidate: tfidf_logreg

TF-IDF (word 1-2 grams + char 3-5 grams, concatenated as features) feeding a
LogisticRegression(class_weight="balanced"). C is selected from
{0.1, 0.3, 1, 3, 10} via inner stratified k-fold CV (k=3, shrunk down when a
class is too rare for k=3) on X_train/y_train only, maximizing balanced
accuracy.

Truncation
----------
TF-IDF has no architectural token limit the way a transformer does, but we
still cap input length to keep vectorization/CV time and memory bounded on
pathologically long documents. MAX_CHARS (module constant below) is the cap,
applied on raw characters before any tokenization, to both X_train and
X_test. Only truncation of X_test is tracked (per interface spec).

TRUNCATION_STATS readout (for the integration agent)
------------------------------------------------------
Module-level dict `TRUNCATION_STATS: dict[str, float]` maps
task_key -> fraction of X_test examples truncated, ACCUMULATED CUMULATIVELY
across every fit_predict_proba call made for that task_key so far (i.e.
across all folds/repeats seen up to the point you read it -- read it after
the run, or after all 15 calls for a task, for the final "across the run"
fraction). Internally this is backed by a private
`_TRUNCATION_RAW: dict[str, list[int, int]]` of
[cumulative_truncated_count, cumulative_total_count] that
TRUNCATION_STATS[task_key] is recomputed from on every call. Just read
`TRUNCATION_STATS[task_key]` after your run; no other API needed.

Caching
-------
Intentionally NOT implemented for this candidate. TF-IDF fit + transform +
a 5-way (or fewer) x 5-value LogisticRegression grid search on legal-text-
sized folds is fast (seconds), so recomputing per fold/repeat call is not
the bottleneck it would be for an embeddings- or NLI-score-based candidate.
`cache_dir` is accepted (to satisfy the fixed interface) but unused.

Determinism
------------
Every stochastic component is seeded from the `seed` argument:
StratifiedKFold(shuffle=True, random_state=seed) for the inner CV splits,
and LogisticRegression(random_state=seed) for the final classifier (the
default lbfgs solver is itself deterministic given fixed data, but the
random_state is set regardless per spec). No other randomness is used.
GridSearchCV runs with n_jobs=1 so fold execution order cannot introduce
any nondeterminism. Given the same seed and the same X_train/y_train/X_test,
output is byte-identical across runs.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

NAME = "tfidf_logreg"

# --- config -----------------------------------------------------------
MAX_CHARS = 50_000  # raw-character cap applied before vectorization
C_GRID = [0.1, 0.3, 1, 3, 10]
INNER_K = 3  # target inner CV folds; shrunk if a class is too rare
EPS = 1e-9  # near-zero floor for classes absent from y_train / degenerate fits

# --- module-level truncation tracking (see module docstring) ----------
TRUNCATION_STATS: dict[str, float] = {}
_TRUNCATION_RAW: dict[str, list] = {}


def _truncate_batch(texts):
    """Truncate each string to MAX_CHARS; return (truncated_texts, flags)."""
    out = []
    flags = []
    for t in texts:
        if t is None:
            t = ""
        if len(t) > MAX_CHARS:
            out.append(t[:MAX_CHARS])
            flags.append(True)
        else:
            out.append(t)
            flags.append(False)
    return out, flags


def _record_test_truncation(task_key, flags):
    counts = _TRUNCATION_RAW.setdefault(task_key, [0, 0])
    counts[0] += sum(flags)
    counts[1] += len(flags)
    TRUNCATION_STATS[task_key] = (counts[0] / counts[1]) if counts[1] else 0.0


def _build_pipeline(C, seed):
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        random_state=seed,
        max_iter=1000,
        solver="lbfgs",
    )
    return Pipeline([("features", features), ("clf", clf)])


def _fill_full_proba(class_index_map, n_classes, seen_classes, proba):
    n_rows = proba.shape[0]
    full = np.full((n_rows, n_classes), EPS, dtype=np.float64)
    for j, cls in enumerate(seen_classes):
        full[:, class_index_map[cls]] = proba[:, j]
    full = full / full.sum(axis=1, keepdims=True)
    return full


def fit_predict_proba(X_train, y_train, X_test, classes, *, seed, task_key, cache_dir):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    class_index_map = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    X_train_t, _train_flags = _truncate_batch(X_train)
    X_test_t, test_flags = _truncate_batch(X_test)
    _record_test_truncation(task_key, test_flags)

    class_counts = Counter(y_train)
    n_seen_classes = len(class_counts)

    if n_seen_classes < 2:
        # Degenerate fold: only one class present in y_train. A trained
        # discriminative classifier is not meaningful here; the mathematically
        # honest thing to do is a near-one-hot distribution on the single
        # observed class (majority-class-baseline degenerate case), not a
        # crash and not a renormalization over a 1-class support.
        only_class = next(iter(class_counts))
        n_rows = len(X_test_t)
        full = np.full((n_rows, n_classes), EPS, dtype=np.float64)
        full[:, class_index_map[only_class]] = 1.0
        full = full / full.sum(axis=1, keepdims=True)
        return full

    min_count = min(class_counts.values())
    k = min(INNER_K, min_count)

    if k >= 2:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        gs = GridSearchCV(
            estimator=_build_pipeline(C=1.0, seed=seed),
            param_grid={"clf__C": C_GRID},
            scoring="balanced_accuracy",
            cv=skf,
            n_jobs=1,
            refit=True,
        )
        gs.fit(X_train_t, y_train)
        pipe = gs.best_estimator_
    else:
        # Too few examples of the rarest class to run inner CV at all;
        # fall back to the grid's middle value and fit directly.
        pipe = _build_pipeline(C=1.0, seed=seed)
        pipe.fit(X_train_t, y_train)

    seen_classes = pipe.named_steps["clf"].classes_
    proba = pipe.predict_proba(X_test_t)

    return _fill_full_proba(class_index_map, n_classes, seen_classes, proba)


if __name__ == "__main__":
    import random

    rng = random.Random(0)

    SYN_CLASSES = ["alpha", "beta", "gamma"]
    VOCAB = {
        "alpha": ["cat", "dog", "bird", "fish", "mouse", "small pet"],
        "beta": ["car", "truck", "plane", "boat", "train", "vehicle"],
        "gamma": ["apple", "banana", "carrot", "grape", "fruit", "veg"],
    }

    def make_examples(n_per_class, offset):
        texts, labels = [], []
        for cls in SYN_CLASSES:
            words = VOCAB[cls]
            for i in range(n_per_class):
                k = 3 + (i % 3)
                sample = rng.sample(words, min(k, len(words)))
                texts.append(f"example {offset + i} about " + " ".join(sample))
                labels.append(cls)
        return texts, labels

    X_train_syn, y_train_syn = make_examples(13, 0)  # ~39 examples, 3 classes
    # pad to ~40
    X_train_syn.append("a lonely extra alpha cat and dog example")
    y_train_syn.append("alpha")

    X_test_syn, _ = make_examples(4, 100)  # 12 -> trim to 10
    X_test_syn = X_test_syn[:10]

    proba = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        X_test_syn,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_task",
        cache_dir="/tmp/tfidf_logreg_selftest_cache",
    )

    assert proba.shape == (10, 3), f"bad shape: {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"rows don't sum to 1: {row_sums}"
    assert np.all(proba >= 0), "negative probability found"

    # determinism check: same seed + same inputs -> identical output
    proba2 = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        X_test_syn,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_task",
        cache_dir="/tmp/tfidf_logreg_selftest_cache",
    )
    assert np.array_equal(proba, proba2), "non-deterministic output for identical inputs/seed"

    # truncation stats readout sanity check
    assert "selftest_task" in TRUNCATION_STATS
    assert 0.0 <= TRUNCATION_STATS["selftest_task"] <= 1.0

    # single-class-in-fold edge case
    proba_single = fit_predict_proba(
        ["only one class text"] * 5,
        ["alpha"] * 5,
        X_test_syn[:3],
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_singleclass",
        cache_dir="/tmp/tfidf_logreg_selftest_cache",
    )
    assert proba_single.shape == (3, 3)
    assert np.allclose(proba_single.sum(axis=1), 1.0, atol=1e-6)
    assert np.argmax(proba_single, axis=1).tolist() == [0, 0, 0]  # "alpha" is index 0

    print("OK: shape", proba.shape, "row sums ~1.0, deterministic, truncation stats readable")
    print("TRUNCATION_STATS:", TRUNCATION_STATS)
