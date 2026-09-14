"""
Candidate: embed_small_logreg

Thin wrapper around `_embed_common` using sentence-transformers model
BAAI/bge-small-en-v1.5 (384-dim, CPU inference) as a frozen feature
extractor, feeding a LogisticRegression(class_weight="balanced") with C
selected from {0.1, 0.3, 1, 3, 10} via inner CV. ALL of the actual logic --
embedding, disk caching, truncation tracking, the classifier, and the
inner-CV hyperparameter search -- lives in `_embed_common.py`, which this
file and `embed_base_logreg.py` (BAAI/bge-base-en-v1.5) both import; see
that module's docstring for the full pipeline, caching layout, and
determinism argument. This file exists only to bind MODEL_NAME/NAME and
expose the fixed-interface function name.

Truncation
----------
See `_embed_common`'s docstring for the mechanism (the model's own
tokenizer truncates to `model.max_seq_length` -- 512 tokens for this
model -- during `.encode()`; truncation-would-apply is separately detected
per text for bookkeeping).

TRUNCATION_STATS readout (for the integration agent)
-----------------------------------------------------
Module-level dict `TRUNCATION_STATS: dict[str, float]` (bound below to
`_embed_common.get_truncation_stats_dict(MODEL_NAME)`, a live reference
that `_embed_common` mutates in place on every call -- it is NOT a copy)
maps task_key -> fraction of X_test examples truncated, ACCUMULATED
CUMULATIVELY across every fit_predict_proba call made for that task_key so
far. Read `TRUNCATION_STATS[task_key]` after the run (or after all 15
calls for a task) for the final "across the run" fraction. Because the
underlying dict is namespaced by model name inside `_embed_common`, this
candidate's stats are never mixed with embed_base_logreg's even though
both import the same shared module.

Caching
-------
Embeddings are cached to disk under `cache_dir` keyed by
(model_name="BAAI/bge-small-en-v1.5", task_key, sha256(raw text)) -- see
`_embed_common` docstring for the exact layout. The same text seen across
different folds/repeats of the same task is embedded exactly once.

Determinism
------------
See `_embed_common` docstring: CPU encoder inference is deterministic
given fixed weights and input text; `seed` drives the inner
StratifiedKFold splits and the final LogisticRegression's random_state.
Given the same seed and the same X_train/y_train/X_test, output is
byte-identical across runs regardless of cache state.
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import _embed_common as _common  # noqa: E402

NAME = "embed_small_logreg"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Live reference into _embed_common's per-model stats dict -- see
# module docstring above and _embed_common.get_truncation_stats_dict.
TRUNCATION_STATS: dict = _common.get_truncation_stats_dict(MODEL_NAME)


def fit_predict_proba(X_train, y_train, X_test, classes, *, seed, task_key, cache_dir):
    return _common.fit_predict_proba_with_model(
        MODEL_NAME,
        X_train,
        y_train,
        X_test,
        classes,
        seed=seed,
        task_key=task_key,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    import random

    import numpy as np

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
    X_train_syn.append("a lonely extra alpha cat and dog example")
    y_train_syn.append("alpha")

    X_test_syn, _ = make_examples(4, 100)  # 12 -> trim to 10
    X_test_syn = X_test_syn[:10]

    cache_dir = "/tmp/embed_small_logreg_selftest_cache"

    proba = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        X_test_syn,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_task",
        cache_dir=cache_dir,
    )

    assert proba.shape == (10, 3), f"bad shape: {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"rows don't sum to 1: {row_sums}"
    assert np.all(proba >= 0), "negative probability found"

    # determinism check: same seed + same inputs -> identical output,
    # exercised with a warm cache (this call reuses every embedding from
    # the call above).
    proba2 = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        X_test_syn,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_task",
        cache_dir=cache_dir,
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
        cache_dir=cache_dir,
    )
    assert proba_single.shape == (3, 3)
    assert np.allclose(proba_single.sum(axis=1), 1.0, atol=1e-6)
    assert np.argmax(proba_single, axis=1).tolist() == [0, 0, 0]  # "alpha" is index 0

    # a text long enough to force real truncation, to exercise that path
    long_text = "word " * 3000
    proba_long = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        [long_text] * 3,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_longtext",
        cache_dir=cache_dir,
    )
    assert proba_long.shape == (3, 3)
    assert TRUNCATION_STATS["selftest_longtext"] == 1.0

    print("OK: shape", proba.shape, "row sums ~1.0, deterministic, truncation stats readable")
    print("TRUNCATION_STATS:", TRUNCATION_STATS)
