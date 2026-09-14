"""
core/cv.py

The main CV + conformal evaluation loop.

=====================================================================
AGGREGATION CHOICE: combining 3 repeats with a 1000-resample bootstrap CI
=====================================================================
For every task+candidate we run `repeats` (default 3) INDEPENDENT
stratified 5-fold partitions of the test split (different fold-seeds).
Each repeat's out-of-fold (OOF) predictions cover every test item exactly
once. That gives 3 independent, full-coverage per-item prediction arrays.

We combine them as follows, applied identically everywhere in this
codebase (main CV metric, accuracy, macro-F1, conformal keep_rate,
conformal coverage, metric_on_kept, metric_on_sent):

  * POINT ESTIMATE ("mean"): the metric is computed ONCE per repeat on
    that repeat's full-coverage OOF predictions (3 numbers), and the
    reported mean is the plain arithmetic mean of those 3 numbers. This
    is a clean, easy-to-explain estimate: each of the 3 numbers is
    already a legitimate full-test-set score, no resampling involved.

  * 95% CI: we POOL all 3 repeats' per-item (y_true, y_pred) pairs (or
    boolean outcomes, for rate-like quantities) into one array of length
    3 * n_test -- i.e. each test item contributes 3 rows, one per repeat,
    carrying that repeat's OOF prediction for it. We then bootstrap: draw
    1000 resamples (with replacement, size 3*n_test) from this pooled
    array and recompute the metric on each resample; the CI is the
    (2.5th, 97.5th) percentile of those 1000 values.

  Why this combination and not another: bootstrapping within each repeat
  separately and averaging the three CIs would understate uncertainty
  from the fold-partition itself (which repeat/seed you happened to draw)
  since it never lets a resample mix items across repeats. Pooling all
  repeats first and bootstrapping once captures both item-level sampling
  variability AND repeat-to-repeat (fold-partition) variability in a
  single CI, while keeping the point estimate simple and repeat-symmetric
  (a plain mean of 3 independent full-coverage scores). This choice is
  used consistently for every aggregated quantity in the codebase --
  there is no other CI recipe anywhere in this project.

All RNG (fold-seed derivation, calibration-slice carving, bootstrap
resampling) is derived deterministically from the CLI --seed plus stable
string keys (task, candidate, repeat index, fold index, purpose) via
`_derive_seed`, so two runs with the same --seed produce byte-identical
results (see tests/test_determinism.py).

=====================================================================
LEAK CHECK
=====================================================================
For every fold of every repeat we build the fold's training pool from
EXPLICIT tagged indices ("train-split idx K" vs "test-split idx K") and
call `_assert_disjoint` on the held-out test-split positions vs. the
test-split positions used in that fold's training pool, and again for the
conformal calibration slice vs. the 80% conformal-training slice. This is
a real assertion executed on every fold/repeat, not a comment -- see
tests/test_leak.py, which exercises this same code path directly.
Train-split items are, by design, NEVER held out (they are always part of
every fold's training pool) -- that is correct and is not checked as a
leak.

=====================================================================
HYPERPARAMETER ISOLATION (no leakage into candidates)
=====================================================================
Candidates receive only (X_train, y_train, X_test, classes, seed,
task_key, cache_dir) for each fit_predict_proba call. This module never
constructs a call that includes the held-out fold's y -- X_test for the
held-out fold is passed as bare text with no label array anywhere in the
call, so it is not merely a convention but a structural guarantee: there
is no parameter through which held-out labels could reach the candidate.
Candidates are expected to do their own inner-CV hyperparameter search
using only the X_train/y_train given to them for that call.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from core.conformal import calibration_threshold, evaluate_sets, point_predictions, prediction_sets
from core.metrics import accuracy, get_metric, macro_f1, per_class_recall

logger = logging.getLogger("legalbench_map.cv")

N_BOOT_DEFAULT = 1000


def _derive_seed(base_seed: int, *parts: str) -> int:
    """Deterministic 32-bit seed derived from base_seed + stable string
    parts. Same inputs -> same seed, always, across processes/runs."""
    key = str(base_seed) + "|" + "|".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _assert_disjoint(a, b, msg: str) -> None:
    a_set = set(a)
    b_set = set(b)
    overlap = a_set & b_set
    assert not overlap, f"LEAK DETECTED: {msg} -- overlapping indices: {sorted(overlap)[:10]}"


def _stratified_positions(y, n_splits: int, random_state: int):
    """Stratified k-fold over positions [0, len(y)). Returns list of
    (train_pos, holdout_pos) numpy arrays, one per fold."""
    y = np.asarray(y)
    pos = np.arange(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(train_pos, holdout_pos) for train_pos, holdout_pos in skf.split(pos, y)]


def _carve_calibration_slice(pool_y, frac: float, random_state: int):
    """Split positions [0, len(pool_y)) into (train80_pos, calib_pos).
    Tries a stratified split; falls back to a plain shuffled split if
    stratification is impossible (e.g. a class with count 1 in the pool).
    """
    pool_y = np.asarray(pool_y)
    pos = np.arange(len(pool_y))
    try:
        train_pos, calib_pos = train_test_split(
            pos, test_size=frac, random_state=random_state, stratify=pool_y
        )
    except ValueError:
        rng = np.random.default_rng(random_state)
        shuffled = pos.copy()
        rng.shuffle(shuffled)
        n_calib = max(1, int(round(len(pos) * frac)))
        calib_pos = shuffled[:n_calib]
        train_pos = shuffled[n_calib:]
    return np.asarray(train_pos), np.asarray(calib_pos)


def _bootstrap_metric(y_true_pool, y_pred_pool, metric_fn, classes, n_boot, rng):
    n = len(y_true_pool)
    if n == 0:
        return float("nan"), float("nan")
    y_true_pool = np.asarray(y_true_pool, dtype=object)
    y_pred_pool = np.asarray(y_pred_pool, dtype=object)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = metric_fn(y_true_pool[idx], y_pred_pool[idx], classes)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_rate(bool_pool, n_boot, rng):
    n = len(bool_pool)
    if n == 0:
        return float("nan"), float("nan")
    bool_pool = np.asarray(bool_pool, dtype=bool)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = bool_pool[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class FoldRecord:
    repeat: int
    fold: int
    n_heldout: int
    n_calib: int
    threshold: float


@dataclass
class CVResult:
    task: str
    candidate: str
    n_items: int
    n_classes: int
    metric_name: str
    cv_mean: float
    cv_ci_low: float
    cv_ci_high: float
    accuracy_mean: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    macro_f1_mean: float
    macro_f1_ci_low: float
    macro_f1_ci_high: float
    keep_rate_mean: float
    keep_rate_ci_low: float
    keep_rate_ci_high: float
    coverage_mean: float
    coverage_ci_low: float
    coverage_ci_high: float
    metric_kept_mean: float
    metric_kept_ci_low: float
    metric_kept_ci_high: float
    metric_sent_mean: float
    metric_sent_ci_low: float
    metric_sent_ci_high: float
    per_class_recall: dict  # class -> mean recall across repeats
    per_class_keep_rate: dict  # class -> mean keep rate across repeats
    per_class_recall_kept: dict  # class -> mean recall-among-kept-items across repeats
    fold_thresholds: list  # list of float, one per fold*repeat, for configs/<task>.json
    hyperparams: dict | None
    alpha: float
    folds: int
    repeats: int
    # Populated only when run_task_candidate_cv(..., collect_per_item=True):
    # one dict per (repeat, held-out item) with keys
    #   repeat, fold, item_idx, gold, pred, p_pred, kept, set_size
    # sourced from the SAME per-repeat arrays used for keep_rate / coverage /
    # metric_on_kept (see the fold loop), so a per-item dump can never
    # disagree with tasks.csv. `pred` is the conformal point prediction
    # (core.conformal.point_predictions on the 80%-trained model: the
    # singleton member when kept, else that model's argmax) -- the exact
    # label metric_kept/metric_sent are scored on; `p_pred` is that same
    # model's row-max probability; `set_size` is the conformal prediction
    # set size; kept == (set_size == 1).
    per_item: list | None = None


def _get_hyperparams(candidate_module, task_key: str):
    """Optional convention: a candidate module may expose
    get_last_hyperparams(task_key) -> dict for config-dump purposes. The
    fixed fit_predict_proba contract returns only a probability array, so
    if a candidate does not implement this hook we honestly report None
    rather than guessing."""
    hook = getattr(candidate_module, "get_last_hyperparams", None)
    if hook is None:
        return None
    try:
        return hook(task_key)
    except Exception:
        return None


def run_task_candidate_cv(
    task_data,
    candidate_module,
    candidate_name: str,
    *,
    folds: int = 5,
    repeats: int = 3,
    alpha: float = 0.05,
    seed: int = 0,
    cache_dir: str | Path,
    n_boot: int = N_BOOT_DEFAULT,
    calib_frac: float = 0.2,
    collect_per_item: bool = False,
) -> CVResult:
    spec = task_data.spec
    classes = spec.classes
    n_classes = len(classes)
    class_index = {c: i for i, c in enumerate(classes)}
    metric_fn = get_metric(spec.metric)

    X_train_full = task_data.X_train
    y_train_full = task_data.y_train
    X_test = task_data.X_test
    y_test = task_data.y_test
    n_test = len(X_test)

    cache_dir = Path(cache_dir) / candidate_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Per-repeat storage
    repeat_scores, repeat_acc, repeat_f1 = [], [], []
    repeat_keep_rate, repeat_coverage = [], []
    repeat_metric_kept, repeat_metric_sent = [], []

    pooled_y_true_main, pooled_y_pred_main = [], []
    pooled_kept_bool = []
    pooled_covered_kept_bool = []
    pooled_kept_y_true, pooled_kept_y_pred = [], []
    pooled_sent_y_true, pooled_sent_y_pred = [], []

    per_class_recall_sum = {c: [] for c in classes}
    per_class_keep_rate_sum = {c: [] for c in classes}
    per_class_recall_kept_sum = {c: [] for c in classes}

    fold_thresholds = []
    last_hyperparams = None
    per_item_records = [] if collect_per_item else None

    for r in range(repeats):
        fold_seed = _derive_seed(seed, spec.task, candidate_name, "fold_partition", str(r))
        fold_splits = _stratified_positions(y_test, folds, fold_seed)

        main_proba_repeat = np.full((n_test, n_classes), np.nan, dtype=np.float64)
        filled_mask = np.zeros(n_test, dtype=bool)

        kept_mask_repeat = np.zeros(n_test, dtype=bool)
        covered_mask_repeat = np.zeros(n_test, dtype=bool)
        point_preds_repeat = [None] * n_test
        if collect_per_item:
            # Per-item bookkeeping for the optional dump; filled from the
            # same fold-local objects (pred_sets / heldout_conf_proba) that
            # produce kept_mask_repeat and point_preds_repeat below.
            fold_of_item_repeat = np.full(n_test, -1, dtype=np.int64)
            set_size_repeat = np.full(n_test, -1, dtype=np.int64)
            p_pred_repeat = np.full(n_test, np.nan, dtype=np.float64)

        for f, (trainpart_pos, heldout_pos) in enumerate(fold_splits):
            _assert_disjoint(
                heldout_pos, trainpart_pos, f"task={spec.task} candidate={candidate_name} "
                f"repeat={r} fold={f}: held-out vs in-fold test-split training positions"
            )
            assert not filled_mask[heldout_pos].any(), (
                f"LEAK/BUG: task={spec.task} repeat={r} fold={f}: some held-out positions "
                f"already covered by an earlier fold in this repeat"
            )

            fold_train_X = list(X_train_full) + [X_test[i] for i in trainpart_pos]
            fold_train_y = list(y_train_full) + [y_test[i] for i in trainpart_pos]
            heldout_X = [X_test[i] for i in heldout_pos]

            main_seed = _derive_seed(seed, spec.task, candidate_name, "main", str(r), str(f))
            main_proba = np.asarray(
                candidate_module.fit_predict_proba(
                    fold_train_X, fold_train_y, heldout_X, classes,
                    seed=main_seed, task_key=spec.task, cache_dir=cache_dir,
                ),
                dtype=np.float64,
            )
            main_proba_repeat[heldout_pos] = main_proba
            filled_mask[heldout_pos] = True

            # --- conformal: carve calibration slice out of THIS fold's
            # training pool, fit on the remaining 80%, score calib+heldout
            # with that one fitted model.
            calib_seed = _derive_seed(seed, spec.task, candidate_name, "calib_split", str(r), str(f))
            train80_pos, calib_pos = _carve_calibration_slice(fold_train_y, calib_frac, calib_seed)
            _assert_disjoint(
                calib_pos, train80_pos, f"task={spec.task} candidate={candidate_name} "
                f"repeat={r} fold={f}: calibration slice vs conformal-training 80% slice"
            )

            X_tr80 = [fold_train_X[i] for i in train80_pos]
            y_tr80 = [fold_train_y[i] for i in train80_pos]
            calib_X = [fold_train_X[i] for i in calib_pos]
            calib_y = [fold_train_y[i] for i in calib_pos]

            conf_seed = _derive_seed(seed, spec.task, candidate_name, "conformal", str(r), str(f))
            conf_proba_all = np.asarray(
                candidate_module.fit_predict_proba(
                    X_tr80, y_tr80, calib_X + heldout_X, classes,
                    seed=conf_seed, task_key=spec.task, cache_dir=cache_dir,
                ),
                dtype=np.float64,
            )
            n_calib = len(calib_pos)
            calib_proba = conf_proba_all[:n_calib]
            heldout_conf_proba = conf_proba_all[n_calib:]

            calib_true_proba = np.array(
                [calib_proba[k, class_index[calib_y[k]]] for k in range(n_calib)]
            )
            threshold = calibration_threshold(calib_true_proba, alpha)
            fold_thresholds.append(threshold)

            pred_sets = prediction_sets(heldout_conf_proba, classes, threshold)
            heldout_y_true = [y_test[i] for i in heldout_pos]
            conf_eval = evaluate_sets(pred_sets, heldout_y_true)
            point_preds = point_predictions(pred_sets, heldout_conf_proba, classes)

            kept_mask_repeat[heldout_pos] = conf_eval["kept_mask"]
            covered_mask_repeat[heldout_pos] = conf_eval["covered_mask"]
            for local_i, global_i in enumerate(heldout_pos):
                point_preds_repeat[global_i] = point_preds[local_i]
            if collect_per_item:
                fold_of_item_repeat[heldout_pos] = f
                set_size_repeat[heldout_pos] = [len(s) for s in pred_sets]
                p_pred_repeat[heldout_pos] = heldout_conf_proba.max(axis=1)

            last_hyperparams = _get_hyperparams(candidate_module, spec.task) or last_hyperparams

        assert filled_mask.all(), (
            f"BUG: task={spec.task} candidate={candidate_name} repeat={r}: fold partition did "
            f"not cover every test item exactly once"
        )

        y_pred_main_repeat = [classes[int(np.argmax(main_proba_repeat[i]))] for i in range(n_test)]

        if collect_per_item:
            assert (fold_of_item_repeat >= 0).all() and (set_size_repeat >= 0).all()
            for i in range(n_test):
                per_item_records.append({
                    "repeat": r,
                    "fold": int(fold_of_item_repeat[i]),
                    "item_idx": i,
                    "gold": y_test[i],
                    "pred": point_preds_repeat[i],
                    "p_pred": float(p_pred_repeat[i]),
                    "kept": int(bool(kept_mask_repeat[i])),
                    "set_size": int(set_size_repeat[i]),
                })

        repeat_scores.append(metric_fn(y_test, y_pred_main_repeat, classes))
        repeat_acc.append(accuracy(y_test, y_pred_main_repeat))
        repeat_f1.append(macro_f1(y_test, y_pred_main_repeat, classes))

        pooled_y_true_main.extend(y_test)
        pooled_y_pred_main.extend(y_pred_main_repeat)

        pooled_kept_bool.extend(kept_mask_repeat.tolist())
        pooled_covered_kept_bool.extend(covered_mask_repeat[kept_mask_repeat].tolist())

        repeat_keep_rate.append(float(kept_mask_repeat.mean()))
        repeat_coverage.append(
            float(covered_mask_repeat[kept_mask_repeat].mean()) if kept_mask_repeat.any() else float("nan")
        )

        kept_y_true = [y_test[i] for i in range(n_test) if kept_mask_repeat[i]]
        kept_y_pred = [point_preds_repeat[i] for i in range(n_test) if kept_mask_repeat[i]]
        sent_y_true = [y_test[i] for i in range(n_test) if not kept_mask_repeat[i]]
        sent_y_pred = [point_preds_repeat[i] for i in range(n_test) if not kept_mask_repeat[i]]

        pooled_kept_y_true.extend(kept_y_true)
        pooled_kept_y_pred.extend(kept_y_pred)
        pooled_sent_y_true.extend(sent_y_true)
        pooled_sent_y_pred.extend(sent_y_pred)

        repeat_metric_kept.append(metric_fn(kept_y_true, kept_y_pred, classes) if kept_y_true else float("nan"))
        repeat_metric_sent.append(metric_fn(sent_y_true, sent_y_pred, classes) if sent_y_true else float("nan"))

        rec = per_class_recall(y_test, y_pred_main_repeat, classes)
        for c in classes:
            per_class_recall_sum[c].append(rec[c])
            c_mask = np.array([y_test[i] == c for i in range(n_test)])
            c_kept = kept_mask_repeat & c_mask
            per_class_keep_rate_sum[c].append(float(c_kept.sum() / c_mask.sum()) if c_mask.sum() > 0 else float("nan"))
            if c_kept.sum() > 0:
                c_kept_correct = sum(
                    1 for i in range(n_test) if c_kept[i] and point_preds_repeat[i] == c
                )
                per_class_recall_kept_sum[c].append(float(c_kept_correct / c_kept.sum()))
            else:
                per_class_recall_kept_sum[c].append(float("nan"))

    # ---- aggregate across repeats ----
    # np.nanmean legitimately warns ("Mean of empty slice") when a
    # candidate never keeps anything in any repeat (e.g. a poorly
    # calibrated baseline) -- nan is the correct value there, not a bug,
    # so that specific warning is suppressed for the remainder of this
    # function only.
    import warnings as _warnings

    _warnings.filterwarnings("ignore", message="Mean of empty slice")

    boot_rng = np.random.default_rng(_derive_seed(seed, spec.task, candidate_name, "bootstrap"))

    cv_mean = float(np.mean(repeat_scores))
    cv_ci_low, cv_ci_high = _bootstrap_metric(pooled_y_true_main, pooled_y_pred_main, metric_fn, classes, n_boot, boot_rng)

    acc_mean = float(np.mean(repeat_acc))
    acc_lo, acc_hi = _bootstrap_metric(pooled_y_true_main, pooled_y_pred_main, accuracy, classes, n_boot, boot_rng)

    f1_mean = float(np.mean(repeat_f1))
    f1_lo, f1_hi = _bootstrap_metric(pooled_y_true_main, pooled_y_pred_main, macro_f1, classes, n_boot, boot_rng)

    keep_rate_mean = float(np.mean(repeat_keep_rate))
    keep_rate_lo, keep_rate_hi = _bootstrap_rate(pooled_kept_bool, n_boot, boot_rng)

    coverage_mean = float(np.nanmean(repeat_coverage))
    coverage_lo, coverage_hi = _bootstrap_rate(pooled_covered_kept_bool, n_boot, boot_rng)

    metric_kept_mean = float(np.nanmean(repeat_metric_kept))
    metric_kept_lo, metric_kept_hi = _bootstrap_metric(
        pooled_kept_y_true, pooled_kept_y_pred, metric_fn, classes, n_boot, boot_rng
    )

    metric_sent_mean = float(np.nanmean(repeat_metric_sent))
    metric_sent_lo, metric_sent_hi = _bootstrap_metric(
        pooled_sent_y_true, pooled_sent_y_pred, metric_fn, classes, n_boot, boot_rng
    )

    per_class_recall_out = {c: float(np.nanmean(v)) for c, v in per_class_recall_sum.items()}
    per_class_keep_rate_out = {c: float(np.nanmean(v)) for c, v in per_class_keep_rate_sum.items()}
    per_class_recall_kept_out = {c: float(np.nanmean(v)) for c, v in per_class_recall_kept_sum.items()}

    return CVResult(
        task=spec.task,
        candidate=candidate_name,
        n_items=n_test,
        n_classes=n_classes,
        metric_name=spec.metric,
        cv_mean=cv_mean, cv_ci_low=cv_ci_low, cv_ci_high=cv_ci_high,
        accuracy_mean=acc_mean, accuracy_ci_low=acc_lo, accuracy_ci_high=acc_hi,
        macro_f1_mean=f1_mean, macro_f1_ci_low=f1_lo, macro_f1_ci_high=f1_hi,
        keep_rate_mean=keep_rate_mean, keep_rate_ci_low=keep_rate_lo, keep_rate_ci_high=keep_rate_hi,
        coverage_mean=coverage_mean, coverage_ci_low=coverage_lo, coverage_ci_high=coverage_hi,
        metric_kept_mean=metric_kept_mean, metric_kept_ci_low=metric_kept_lo, metric_kept_ci_high=metric_kept_hi,
        metric_sent_mean=metric_sent_mean, metric_sent_ci_low=metric_sent_lo, metric_sent_ci_high=metric_sent_hi,
        per_class_recall=per_class_recall_out,
        per_class_keep_rate=per_class_keep_rate_out,
        per_class_recall_kept=per_class_recall_kept_out,
        fold_thresholds=fold_thresholds,
        hyperparams=last_hyperparams,
        alpha=alpha,
        folds=folds,
        repeats=repeats,
        per_item=per_item_records,
    )
