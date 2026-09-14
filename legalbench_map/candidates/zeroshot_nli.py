"""
zeroshot_nli candidate for the legalbench_map CV harness.

Approach
--------
MoritzLaurer/deberta-v3-base-zeroshot-v2.0 is a DeBERTa-v3 NLI checkpoint
with a 2-way (entailment / not_entailment) head, trained for zero-shot
classification via the standard "premise, hypothesis" entailment-scoring
pattern. For every (text, candidate label) pair we score the entailment
logit of (premise=text, hypothesis=<generated sentence for that label>).

Two behaviors, both required by spec, both feeding this function's output:

  (a) Pure zero-shot signal: for a given text, softmax-ing its per-label
      entailment logits across `classes` turns them into a mutually
      exclusive distribution (mirrors what transformers' own
      zero-shot-classification pipeline does internally in single-label
      mode). This is the raw NLI signal -- computed (see `_pure_zeroshot`)
      but not returned directly, per spec: it *feeds* (b).

  (b) Stacking (what actually gets returned): the raw per-label entailment
      *logit* vector (dimension = len(classes), one scalar per class) is
      used as the feature vector for a LogisticRegression head
      (class_weight="balanced"), with C selected from C_GRID by inner
      stratified k-fold CV on X_train/y_train ONLY (X_test's text is used
      only to extract NLI features and to predict on -- never its labels,
      which this function is never given). fit_predict_proba returns this
      classifier's predict_proba on X_test's NLI-score features. This is
      real use of the labeled data (not just NLI zero-shot in a costume):
      if only 1 class is present in a fold's y_train (or X_train is empty),
      training a discriminative head is not meaningful, so we fall back to
      the pure-zero-shot softmax distribution (a) for that call instead.

Hypothesis derivation (generic, rule-based -- never hand-written per task)
----------------------------------------------------------------------------
For task_key, /Users/vasyl/zadumai/legalbench_map/data/task_instructions/<task_key>.json
is read (fields: readme_description, base_prompt_preamble, labels -- per
spec, only `readme_description` is used here, and only its first sentence).
If the file is missing, or unreadable, or readme_description is null/empty,
we fall back silently to the label-only template using nothing but
`classes` (never errors).

For every label in `classes` (independent of which label is "under test" --
the frame text never depends on which label a hypothesis is being built
for, only on task-level readme text and the literal label string):

    frame     = "Given the following: {first sentence of readme_description} "
                (omitted entirely when there is no readme_description)
    hypothesis = f"{frame}This example involves {label.replace('_',' ').replace('-',' ')}."

Same shape, every task, every label -- only the inputs (readme text, label
string) change.

Truncation stats (read-back contract for the integration agent)
------------------------------------------------------------------
Module-level dict `TRUNCATION_STATS: dict[str, float]` maps
task_key -> fraction of *unique* X_test texts seen so far (pooled,
deduplicated across every fit_predict_proba call for that task_key, since
X_test is a resampled subset of a roughly-fixed pool across folds/repeats)
whose premise, tokenized alone, would not fit in the room left after
reserving `tokenizer.num_special_tokens_to_add(pair=True)` special tokens
and the longest hypothesis built for that task out of the model's 512-token
limit -- i.e. whether that (premise, some_hypothesis) pair for this task
gets truncated under truncation="only_first". A text is counted at most
once per task_key even if it reappears in later folds/repeats. Read it any
time after a run with:
    from candidates.zeroshot_nli import TRUNCATION_STATS
    TRUNCATION_STATS["<task_key>"]  # float in [0, 1], or absent if never called

Caching (essential -- do not skip when reading this file)
-------------------------------------------------------------
NLI forward passes are the expensive artifact here, so they are cached to
disk keyed by (task_key, sha256(text_utf8), label) under
`<cache_dir>/zeroshot_nli/nli_scores/<task_key>.sqlite3` (one SQLite table
`scores(text_hash, label, score)`, PRIMARY KEY(text_hash, label), storing
the raw entailment *logit* as REAL). Every fit_predict_proba call:
  1. hashes every unique text in X_train + X_test for this call,
  2. looks up which (text_hash, label) pairs already exist in the DB,
  3. runs the model ONLY on the missing pairs, batched,
  4. writes the new scores back (INSERT OR REPLACE), commits, closes.
This means each (text, label) pair is scored by the network exactly once
per task, ever, no matter how many of the up-to-15 fold/repeat calls touch
that text -- the whole point, since folds/repeats resample the same pool.
The heavy model weights are loaded from HF into
`<cache_dir>/zeroshot_nli/hf_models` (also a caching decision made in this
file: the model/tokenizer HF cache is redirected under cache_dir rather
than the global default HF cache) lazily
and only once per process, and (separately) the actual full model
(AutoModelForSequenceClassification) is loaded into memory only when there
is at least one cache miss to compute for a call -- if every needed
(text_hash, label) pair for a call is already cached, the multi-hundred-MB
model is never even instantiated for that call.
Every call logs to stderr, e.g.:
    [zeroshot_nli] task_key=<k> nli cache: 37 computed, 483 reused (this call); totals: 91 computed, 1934 reused (run so far)
so the achieved reuse ratio is verifiable after the fact from run logs.

Determinism
-----------
NLI forward passes involve no randomness (model.eval(), no dropout, fixed
pretrained weights) so they are deterministic given the same text/label/
model regardless of `seed`. `seed` seeds `random`, `numpy`, `torch.manual_seed`
(harmless here) and, materially, `StratifiedKFold(shuffle=True,
random_state=seed)` for the inner-CV fold split and
`LogisticRegression(random_state=seed)` for the final classifier. On CPU
this is byte-identical run to run. This machine also has MPS (Apple
Silicon) available and it is used for the forward passes when present
(pure inference, no accumulation-order-sensitive training step, so results
are stable given fixed weights + fixed inputs on the same hardware/driver
stack) -- see device selection below.

Network
-------
The only network calls this file makes are `from_pretrained(...)` for
MoritzLaurer/deberta-v3-base-zeroshot-v2.0's config/tokenizer/weights,
which resolve to huggingface.co (and its CDN) and nowhere else. No other
host is ever contacted.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

NAME = "zeroshot_nli"

MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
HARD_MAX_LEN = 512  # model architecture cap; also used as a hard fallback
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
INNER_K = 5  # target inner-CV folds; shrunk when a class is too rare
EPS = 1e-6  # near-zero floor for classes absent from y_train
NLI_BATCH_SIZE = 16

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TASK_INSTRUCTIONS_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "data", "task_instructions")
)

# --- module-level truncation tracking (see module docstring) --------------
TRUNCATION_STATS: dict[str, float] = {}
_TRUNC_SEEN: dict[str, dict] = {}  # task_key -> {text_hash: bool truncated}

# --- module-level cache-hit/miss counters (see module docstring) ----------
_CACHE_STATS: dict[str, list] = {}  # task_key -> [computed_total, reused_total]

# --- lazy, process-wide singletons (see Caching section of docstring) -----
_TOKENIZER = None
_TOKENIZER_MAX_LEN = None
_MODEL = None
_DEVICE = None
_ENTAIL_ID = None
_LOAD_LOCK = threading.Lock()

# in-memory memo of parsed task_instructions JSON, to avoid re-reading the
# file on every one of the up-to-15 calls for the same task
_INSTR_CACHE: dict[str, dict] = {}


def _get_tokenizer(hf_cache_dir):
    """Load (once per process) only the tokenizer -- cheap, needed for
    truncation-stat bookkeeping and for hashing hypothesis token lengths
    even on calls that end up being 100% cache hits for NLI scores."""
    global _TOKENIZER, _TOKENIZER_MAX_LEN
    if _TOKENIZER is None:
        with _LOAD_LOCK:
            if _TOKENIZER is None:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=hf_cache_dir)
                model_max = getattr(tok, "model_max_length", None)
                if not model_max or model_max > 100_000:
                    model_max = HARD_MAX_LEN
                _TOKENIZER_MAX_LEN = min(HARD_MAX_LEN, model_max)
                _TOKENIZER = tok
    return _TOKENIZER, _TOKENIZER_MAX_LEN


def _get_full_model(hf_cache_dir):
    """Load (once per process, and only when actually needed -- i.e. some
    NLI score is not already on disk) the full classification model."""
    global _MODEL, _DEVICE, _ENTAIL_ID
    if _MODEL is None:
        with _LOAD_LOCK:
            if _MODEL is None:
                from transformers import AutoModelForSequenceClassification

                _get_tokenizer(hf_cache_dir)
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, cache_dir=hf_cache_dir
                )
                model.eval()
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
                model.to(device)
                label2id = {str(k).lower(): v for k, v in model.config.label2id.items()}
                entail_id = label2id.get("entailment", 0)
                _MODEL = model
                _DEVICE = device
                _ENTAIL_ID = entail_id
    return _MODEL, _DEVICE, _ENTAIL_ID


def _sha256(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _cache_db_path(cache_dir, task_key):
    d = os.path.join(cache_dir, "zeroshot_nli", "nli_scores")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{task_key}.sqlite3")


def _hf_cache_dir(cache_dir):
    d = os.path.join(cache_dir, "zeroshot_nli", "hf_models")
    os.makedirs(d, exist_ok=True)
    return d


def _open_cache_db(cache_dir, task_key):
    conn = sqlite3.connect(_cache_db_path(cache_dir, task_key), timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scores ("
        "text_hash TEXT NOT NULL, label TEXT NOT NULL, score REAL NOT NULL, "
        "PRIMARY KEY (text_hash, label))"
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    return conn


def _load_task_instructions(task_key):
    if task_key in _INSTR_CACHE:
        return _INSTR_CACHE[task_key]
    path = os.path.join(_TASK_INSTRUCTIONS_DIR, f"{task_key}.json")
    info = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f) or {}
        except (OSError, ValueError):
            info = {}
    _INSTR_CACHE[task_key] = info
    return info


def _first_sentence(text):
    """First sentence of `text`, ending at the first '.'/'!'/'?' that is
    followed by whitespace-then-an-uppercase-letter (or by end of string).
    That lookahead is what keeps this from stopping at legal-text
    abbreviations like "e.g.", "i.e.", "Inc.", "No." -- those are almost
    always followed by a lowercase continuation word, not a capitalized new
    sentence, so a plain "split on any period" would misfire on exactly the
    kind of prose LegalBench readme_description fields are full of."""
    if not text:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    m = re.search(r"^(.+?[.!?])(?:\s+(?=[A-Z])|\s*$)", text)
    return (m.group(1).strip() if m else text[:200]).strip()


def _build_hypotheses(classes, task_key):
    """Rule-based, uniform across every task and every label -- see module
    docstring. Returns {label: hypothesis_string}."""
    info = _load_task_instructions(task_key)
    readme = (info or {}).get("readme_description") or ""
    fs = _first_sentence(readme)
    frame = f"Given the following: {fs} " if fs else ""
    hyps = {}
    for label in classes:
        label_text = str(label).replace("_", " ").replace("-", " ").strip()
        hyps[label] = f"{frame}This example involves {label_text}."
    return hyps


def _update_truncation_stats(texts, task_key, tokenizer, max_len, hypotheses):
    seen = _TRUNC_SEEN.setdefault(task_key, {})
    if hypotheses:
        max_hyp_len = max(
            len(tokenizer.encode(h, add_special_tokens=False)) for h in hypotheses.values()
        )
    else:
        max_hyp_len = 0
    try:
        num_special = tokenizer.num_special_tokens_to_add(pair=True)
    except Exception:
        num_special = 3
    premise_budget = max(1, max_len - max_hyp_len - num_special)
    for t in texts:
        th = _sha256(t)
        if th in seen:
            continue
        n_tokens = len(tokenizer.encode(t or "", add_special_tokens=False))
        seen[th] = n_tokens > premise_budget
    if seen:
        frac = sum(1 for v in seen.values() if v) / len(seen)
        TRUNCATION_STATS[task_key] = frac


def _score_pairs_uncached(premises, hyps, model, tokenizer, device, entail_id, max_len):
    scores = []
    with torch.no_grad():
        for i in range(0, len(premises), NLI_BATCH_SIZE):
            batch_p = premises[i : i + NLI_BATCH_SIZE]
            batch_h = hyps[i : i + NLI_BATCH_SIZE]
            enc = tokenizer(
                batch_p,
                batch_h,
                truncation="only_first",
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().to("cpu").numpy()
            for row in logits:
                scores.append(float(row[entail_id]))
    return scores


def _get_nli_scores(texts, task_key, cache_dir, classes, hypotheses):
    """Returns {text: np.ndarray(len(classes))} of raw entailment logits, in
    `classes` order, computing only cache misses (see module docstring)."""
    conn = _open_cache_db(cache_dir, task_key)
    cur = conn.cursor()

    text_hash = {t: _sha256(t) for t in texts}
    all_hashes = list(set(text_hash.values()))
    existing = {}
    CHUNK = 400
    for i in range(0, len(all_hashes), CHUNK):
        chunk = all_hashes[i : i + CHUNK]
        placeholders = ",".join(["?"] * len(chunk))
        q = f"SELECT text_hash, label, score FROM scores WHERE text_hash IN ({placeholders})"
        for th, label, score in cur.execute(q, chunk):
            existing[(th, label)] = score

    result = {t: {} for t in texts}
    to_compute = []  # (text, text_hash, label)
    for t in texts:
        th = text_hash[t]
        for label in classes:
            key = (th, label)
            if key in existing:
                result[t][label] = existing[key]
            else:
                to_compute.append((t, th, label))

    reused = sum(len(result[t]) for t in texts)  # cache hits only; to_compute not yet filled in
    computed = 0
    if to_compute:
        model, device, entail_id = _get_full_model(_hf_cache_dir(cache_dir))
        tokenizer, max_len = _get_tokenizer(_hf_cache_dir(cache_dir))
        premises = [t for (t, _th, _label) in to_compute]
        hyps = [hypotheses[label] for (_t, _th, label) in to_compute]
        scores = _score_pairs_uncached(premises, hyps, model, tokenizer, device, entail_id, max_len)
        rows = []
        for (t, th, label), score in zip(to_compute, scores):
            result[t][label] = score
            rows.append((th, label, score))
        cur.executemany(
            "INSERT OR REPLACE INTO scores (text_hash, label, score) VALUES (?, ?, ?)", rows
        )
        conn.commit()
        computed = len(to_compute)

    conn.close()

    stats = _CACHE_STATS.setdefault(task_key, [0, 0])
    stats[0] += computed
    stats[1] += reused
    print(
        f"[zeroshot_nli] task_key={task_key} nli cache: {computed} computed, {reused} reused "
        f"(this call); totals: {stats[0]} computed, {stats[1]} reused (run so far)",
        file=sys.stderr,
    )

    out = {}
    for t in texts:
        out[t] = np.array([result[t][label] for label in classes], dtype=np.float64)
    return out


def _softmax_rows(scores):
    s = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def _pure_zeroshot(feat):
    """Behavior (a): softmax the raw entailment-logit feature vectors across
    classes. Used as the fallback output when a discriminative head cannot
    meaningfully be trained (see fit_predict_proba)."""
    return _softmax_rows(feat)


def _fill_full_proba(class_index_map, n_classes, seen_classes, proba):
    n_rows = proba.shape[0]
    full = np.full((n_rows, n_classes), EPS, dtype=np.float64)
    for j, cls in enumerate(seen_classes):
        full[:, class_index_map[cls]] = proba[:, j]
    full = full / full.sum(axis=1, keepdims=True)
    return full


def fit_predict_proba(X_train, y_train, X_test, classes, *, seed, task_key, cache_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    classes = list(classes)
    n_classes = len(classes)
    class_index_map = {c: i for i, c in enumerate(classes)}

    X_train = list(X_train)
    y_train = list(y_train)
    X_test = list(X_test)

    hypotheses = _build_hypotheses(classes, task_key)

    tokenizer, max_len = _get_tokenizer(_hf_cache_dir(cache_dir))
    _update_truncation_stats(X_test, task_key, tokenizer, max_len, hypotheses)

    if len(X_test) == 0:
        return np.zeros((0, n_classes), dtype=np.float64)

    unique_texts = list(dict.fromkeys(X_train + X_test))
    score_map = _get_nli_scores(unique_texts, task_key, cache_dir, classes, hypotheses)

    X_test_feat = np.stack([score_map[t] for t in X_test])

    if len(X_train) == 0:
        return _pure_zeroshot(X_test_feat)

    X_train_feat = np.stack([score_map[t] for t in X_train])
    y_train_arr = np.asarray(y_train)

    class_counts = {c: int(np.sum(y_train_arr == c)) for c in classes}
    present_classes = [c for c in classes if class_counts[c] > 0]

    if len(present_classes) < 2:
        # Only one class observed in this fold's y_train (or none at all):
        # a discriminative head is not meaningful here. Fall back to the
        # pure zero-shot signal rather than a degenerate/one-hot fit.
        return _pure_zeroshot(X_test_feat)

    min_count = min(class_counts[c] for c in present_classes)
    best_C = 1.0
    if min_count >= 2 and len(X_train_feat) >= 4:
        k = min(INNER_K, min_count)
        try:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            best_score = -np.inf
            for C in C_GRID:
                fold_scores = []
                for tr_idx, va_idx in skf.split(X_train_feat, y_train_arr):
                    clf = LogisticRegression(
                        C=C, class_weight="balanced", max_iter=2000, random_state=seed
                    )
                    clf.fit(X_train_feat[tr_idx], y_train_arr[tr_idx])
                    fold_scores.append(clf.score(X_train_feat[va_idx], y_train_arr[va_idx]))
                mean_score = float(np.mean(fold_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_C = C
        except ValueError:
            best_C = 1.0

    clf = LogisticRegression(C=best_C, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train_feat, y_train_arr)
    raw_proba = clf.predict_proba(X_test_feat)

    return _fill_full_proba(class_index_map, n_classes, clf.classes_, raw_proba)


if __name__ == "__main__":
    # Fixed (not tempfile-cleaned) cache dir, matching the other candidates'
    # self-test convention -- this lets a re-run of this self-test reuse the
    # already-downloaded ~370MB model weights under
    # <cache_dir>/zeroshot_nli/hf_models instead of re-fetching them from
    # huggingface.co every time this file is executed standalone.
    SELFTEST_CACHE_DIR = "/tmp/zeroshot_nli_selftest_cache"

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

    X_train_syn, y_train_syn = make_examples(13, 0)  # 39 examples, 3 classes
    X_train_syn.append("a lonely extra alpha cat and dog example")
    y_train_syn.append("alpha")  # 40 total

    X_test_syn, _ = make_examples(4, 100)  # 12 -> trim to 10
    X_test_syn = X_test_syn[:10]

    proba = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        X_test_syn,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_task",
        cache_dir=SELFTEST_CACHE_DIR,
    )

    assert proba.shape == (10, 3), f"bad shape: {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"rows don't sum to 1: {row_sums}"
    assert np.all(proba >= 0), "negative probability found"

    # determinism check + cache-hit check: second call, same inputs/seed,
    # must be byte-identical AND must reuse the cache entirely (0 computed).
    stats_before = list(_CACHE_STATS.get("selftest_task", [0, 0]))
    proba2 = fit_predict_proba(
        X_train_syn,
        y_train_syn,
        X_test_syn,
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_task",
        cache_dir=SELFTEST_CACHE_DIR,
    )
    assert np.array_equal(proba, proba2), "non-deterministic output for identical inputs/seed"
    stats_after = _CACHE_STATS["selftest_task"]
    computed_on_second_call = stats_after[0] - stats_before[0]
    assert computed_on_second_call == 0, (
        f"expected 0 fresh NLI computations on repeat call, got {computed_on_second_call}"
    )

    # truncation stats readout sanity check
    assert "selftest_task" in TRUNCATION_STATS
    assert 0.0 <= TRUNCATION_STATS["selftest_task"] <= 1.0

    # missing task_instructions file -> generic fallback template, no error
    assert _load_task_instructions("selftest_task_no_such_key") == {}

    # single-class-in-fold edge case -> falls back to pure zero-shot, no crash
    proba_single = fit_predict_proba(
        ["only one class text"] * 5,
        ["alpha"] * 5,
        X_test_syn[:3],
        classes=SYN_CLASSES,
        seed=42,
        task_key="selftest_singleclass",
        cache_dir=SELFTEST_CACHE_DIR,
    )
    assert proba_single.shape == (3, 3)
    assert np.allclose(proba_single.sum(axis=1), 1.0, atol=1e-6)

    # empty X_train -> pure zero-shot path, no crash
    proba_notrain = fit_predict_proba(
        [], [], X_test_syn[:4], classes=SYN_CLASSES, seed=42,
        task_key="selftest_notrain", cache_dir=SELFTEST_CACHE_DIR,
    )
    assert proba_notrain.shape == (4, 3)
    assert np.allclose(proba_notrain.sum(axis=1), 1.0, atol=1e-6)

    print("OK: shape", proba.shape, "row sums ~1.0, deterministic, cache reused on repeat call")
    print("TRUNCATION_STATS:", TRUNCATION_STATS)
    print("CACHE_STATS:", _CACHE_STATS)
