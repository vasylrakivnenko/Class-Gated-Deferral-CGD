"""
Shared implementation for the embed_* candidates (embed_small_logreg,
embed_base_logreg). NOT a candidate itself -- has no NAME constant and is
never registered with the harness directly; the two thin wrapper files
(embed_small_logreg.py, embed_base_logreg.py) each import this module and
call `fit_predict_proba_with_model(MODEL_NAME, ...)` with their own
sentence-transformers model name.

Pipeline
--------
1. Embed X_train/X_test with a frozen sentence-transformers encoder (mean-
   pooled, L2-normalized output via `normalize_embeddings=True` -- the
   usage BAAI recommends for BGE models and a sane default for feeding a
   linear classifier regardless of encoder). No retrieval-style
   instruction prefix ("Represent this sentence for retrieval: ...") is
   added to either side -- this is a plain classification-feature use of
   the encoder, not a query/passage retrieval task, so both train and test
   text are embedded identically and symmetrically.
2. LogisticRegression(class_weight="balanced") on top of the embeddings,
   with C selected from {0.1, 0.3, 1, 3, 10} via inner stratified k-fold CV
   (k=3, shrunk when a class is too rare for k=3) on X_train/y_train only,
   maximizing balanced accuracy -- identical selection procedure to the
   tfidf_logreg candidate, just swapping the feature extractor.

Truncation
----------
Each encoder has a hard token budget (`model.max_seq_length`, read off the
loaded SentenceTransformer -- 512 for both bge-small-en-v1.5 and
bge-base-en-v1.5 as of this writing, but we never hardcode it: we always
read it from the loaded model). We do not hand-slice text ourselves --
slicing on raw characters and re-tokenizing risks cutting a multi-byte
token mid-way and does not match what the model's own tokenizer would keep.
Instead:
  - The SentenceTransformer's own tokenizer truncates to `max_seq_length`
    tokens internally during `.encode()` (HuggingFace `truncation=True`
    under the hood) -- this IS the truncation the interface requires;
    every text longer than the model's max length is truncated before it
    influences the embedding.
  - To detect *whether* a given text would be truncated (for the required
    stats), we separately tokenize each text with
    `add_special_tokens=True, truncation=False` and compare the raw token
    count to `max_seq_length`. This is a cheap side computation (no
    gradient, no pooling) done purely for bookkeeping and does not change
    what gets embedded.

TRUNCATION_STATS readout (for the integration agent)
-----------------------------------------------------
This module keeps truncation stats PER MODEL NAME, not per candidate file,
because both candidates share this code. Each wrapper binds its own
`TRUNCATION_STATS` module attribute at import time via
`get_truncation_stats_dict(MODEL_NAME)`, which returns a dict object that
this module mutates *in place* (never reassigns) on every call -- so
`embed_small_logreg.TRUNCATION_STATS` and `embed_base_logreg.TRUNCATION_STATS`
are live views onto their own model's stats, unaffected by the other
candidate. As with tfidf_logreg, each entry
`TRUNCATION_STATS[task_key]` is the fraction of X_test examples truncated,
ACCUMULATED CUMULATIVELY across every call made for that task_key so far;
read it after the run (or after all 15 calls for a task) for the final
"across the run" fraction. Backed internally by
`_ALL_TRUNCATION_RAW[model_name][task_key] = [cumulative_truncated, cumulative_total]`.

Caching
-------
Caching is the whole point of factoring this out: embeddings are cached to
disk keyed by (model_name, task_key, sha256(raw text)) so the same text
seen in fold 1 is never re-embedded in folds 2-5 or in the other 2 repeats
of the same task. Layout under `cache_dir`:

    <cache_dir>/embeddings/<model_name_sanitized>/<task_key>/<hash[:2]>/<hash>.npy

`<model_name_sanitized>` replaces "/" with "__" so e.g. "BAAI/bge-small-en-v1.5"
becomes a valid path component. Each cache file holds exactly one text's
embedding vector as a float32 .npy array. The hash is over the *raw* input
text (pre-truncation) using UTF-8 bytes -- the embedding stored under that
key is always what the model actually produces for that raw text (i.e.
post the model's own internal truncation), so the cache is a pure function
of (model_name, task_key, raw text) as required, and correctly shared
across every fold/repeat of the task regardless of which fold happened to
compute it first. Writes are atomic (write to a `.tmp` sibling then
`os.replace`) so a killed/concurrent run cannot leave a half-written .npy
that a later read would choke on.

The trained LogisticRegression models and the inner-CV grid search are
NOT cached (unlike the embeddings) -- fitting a 5-way logistic regression
on a few hundred precomputed embedding vectors is milliseconds to low
seconds, nowhere near the bottleneck; only the embedding forward passes
(the actual encoder inference) are worth persisting.

The loaded SentenceTransformer model object itself is also cached
in-process (module-level `_MODEL_CACHE`, keyed by model name) so repeated
`fit_predict_proba` calls across folds/repeats within one process load the
weights from disk exactly once.

Determinism
-----------
Sentence-transformer encoding on CPU in eval mode (no dropout, no
sampling) is a deterministic function of the input text and model weights
-- no seed needed for embedding itself. `StratifiedKFold(shuffle=True,
random_state=seed)` seeds the inner CV splits and
`LogisticRegression(random_state=seed)` seeds the final classifier's
solver (lbfgs is itself deterministic given fixed data, but random_state
is set regardless per spec). GridSearchCV runs with n_jobs=1 so fold
execution order cannot introduce nondeterminism. Given the same seed and
the same X_train/y_train/X_test (and a warm or cold cache -- caching never
changes the numeric result, only how fast it's computed), output is
byte-identical across runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# --- config -------------------------------------------------------------
C_GRID = [0.1, 0.3, 1, 3, 10]
INNER_K = 3  # target inner CV folds; shrunk if a class is too rare
EPS = 1e-9  # near-zero floor for classes absent from y_train / degenerate fits
ENCODE_BATCH_SIZE = 32

# --- in-process model cache (avoid reloading weights every fold/repeat) -
_MODEL_CACHE: dict = {}

# --- per-model truncation tracking (see module docstring) ---------------
_ALL_TRUNCATION_STATS: dict = {}
_ALL_TRUNCATION_RAW: dict = {}


def get_truncation_stats_dict(model_name: str) -> dict:
    """Return the live TRUNCATION_STATS dict for model_name, creating it if
    needed. Callers (the wrapper candidate modules) should bind this ONCE
    at import time as their own module-level `TRUNCATION_STATS` name; this
    module always mutates the dict object in place (never reassigns it),
    so that binding stays live for the life of the process."""
    return _ALL_TRUNCATION_STATS.setdefault(model_name, {})


def _record_test_truncation(model_name: str, task_key: str, flags) -> None:
    raw_for_model = _ALL_TRUNCATION_RAW.setdefault(model_name, {})
    counts = raw_for_model.setdefault(task_key, [0, 0])
    counts[0] += sum(flags)
    counts[1] += len(flags)
    stats = get_truncation_stats_dict(model_name)
    stats[task_key] = (counts[0] / counts[1]) if counts[1] else 0.0


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(os.sep, "__")


def _sha256_text(text: str) -> str:
    if text is None:
        text = ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def get_model(model_name: str):
    """Load (once per process) and cache a SentenceTransformer on CPU."""
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        model.eval()
        _MODEL_CACHE[model_name] = model

        # transformers re-configures its own logging on import (inside the
        # SentenceTransformer/AutoTokenizer constructor above), which would
        # clobber a level set any earlier -- so silence the expected
        # "Token indices sequence length is longer than..." warning here,
        # after that import has definitely happened. We deliberately
        # tokenize with truncation=False in _would_truncate_flags (to
        # measure the raw, pre-truncation token count), which triggers
        # this warning on every over-length text; it's harmless bookkeeping
        # since that raw tokenization is never fed to the model --
        # .encode() truncates independently and correctly. Without this,
        # 15 calls/task x many long-document tasks would flood the
        # harness's logs.
        logging.getLogger("transformers").setLevel(logging.ERROR)
    return model


def _would_truncate_flags(model, texts) -> list:
    """For each text, True if the model's tokenizer would truncate it,
    i.e. its full token count (with special tokens) exceeds
    model.max_seq_length. Pure bookkeeping -- does not itself truncate
    anything; `.encode()` does the real truncation independently."""
    tokenizer = model.tokenizer
    max_len = model.max_seq_length
    flags = []
    for t in texts:
        if t is None:
            t = ""
        n_tokens = len(tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"])
        flags.append(n_tokens > max_len)
    return flags


def _cache_path(cache_dir: str, model_name: str, task_key: str, digest: str) -> str:
    return os.path.join(
        cache_dir,
        "embeddings",
        _sanitize_model_name(model_name),
        task_key,
        digest[:2],
        f"{digest}.npy",
    )


def _load_cached(path: str):
    try:
        return np.load(path)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _save_cached(path: str, vec: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".tmp_embed_", suffix=".npy"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, vec)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def embed_texts(model_name: str, task_key: str, texts, cache_dir: str):
    """Return (embeddings, truncation_flags) for `texts`, using the on-disk
    cache keyed by (model_name, task_key, sha256(raw text)). embeddings is
    a float32 ndarray of shape (len(texts), dim); truncation_flags is a
    list[bool] of the same length, True where the model's tokenizer would
    truncate that text."""
    texts = ["" if t is None else t for t in texts]
    model = get_model(model_name)

    digests = [_sha256_text(t) for t in texts]
    paths = [_cache_path(cache_dir, model_name, task_key, d) for d in digests]

    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    out = np.empty((n, dim), dtype=np.float32)
    hit = np.zeros(n, dtype=bool)

    for i, path in enumerate(paths):
        cached = _load_cached(path)
        if cached is not None and cached.shape == (dim,):
            out[i] = cached
            hit[i] = True

    miss_idx = [i for i in range(n) if not hit[i]]
    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        miss_vecs = model.encode(
            miss_texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        for j, i in enumerate(miss_idx):
            out[i] = miss_vecs[j]
            _save_cached(paths[i], out[i])

    flags = _would_truncate_flags(model, texts)
    return out, flags


def _build_classifier(C: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        random_state=seed,
        max_iter=1000,
        solver="lbfgs",
    )


def _fill_full_proba(class_index_map, n_classes, seen_classes, proba):
    n_rows = proba.shape[0]
    full = np.full((n_rows, n_classes), EPS, dtype=np.float64)
    for j, cls in enumerate(seen_classes):
        full[:, class_index_map[cls]] = proba[:, j]
    full = full / full.sum(axis=1, keepdims=True)
    return full


def fit_predict_proba_with_model(
    model_name: str, X_train, y_train, X_test, classes, *, seed, task_key, cache_dir
):
    """Shared body behind both candidates' fit_predict_proba. See module
    docstring for the pipeline, caching, and truncation-tracking details."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    class_index_map = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    X_train_emb, _train_flags = embed_texts(model_name, task_key, X_train, cache_dir)
    X_test_emb, test_flags = embed_texts(model_name, task_key, X_test, cache_dir)
    _record_test_truncation(model_name, task_key, test_flags)

    class_counts = Counter(y_train)
    n_seen_classes = len(class_counts)

    if n_seen_classes < 2:
        # Degenerate fold: only one class present in y_train. A trained
        # discriminative classifier is not meaningful here; the
        # mathematically honest thing to do is a near-one-hot distribution
        # on the single observed class (majority-class-baseline degenerate
        # case), not a crash and not a renormalization over a 1-class
        # support.
        only_class = next(iter(class_counts))
        n_rows = X_test_emb.shape[0]
        full = np.full((n_rows, n_classes), EPS, dtype=np.float64)
        full[:, class_index_map[only_class]] = 1.0
        full = full / full.sum(axis=1, keepdims=True)
        return full

    min_count = min(class_counts.values())
    k = min(INNER_K, min_count)

    if k >= 2:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        gs = GridSearchCV(
            estimator=_build_classifier(C=1.0, seed=seed),
            param_grid={"C": C_GRID},
            scoring="balanced_accuracy",
            cv=skf,
            n_jobs=1,
            refit=True,
        )
        gs.fit(X_train_emb, y_train)
        clf = gs.best_estimator_
    else:
        # Too few examples of the rarest class to run inner CV at all;
        # fall back to the grid's middle value and fit directly.
        clf = _build_classifier(C=1.0, seed=seed)
        clf.fit(X_train_emb, y_train)

    seen_classes = clf.classes_
    proba = clf.predict_proba(X_test_emb)

    return _fill_full_proba(class_index_map, n_classes, seen_classes, proba)
