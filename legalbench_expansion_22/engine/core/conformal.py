"""
core/conformal.py

Hand-implemented split conformal prediction (no external conformal
library, per spec).

Nonconformity score for an item with candidate-model probability vector p
(aligned to the task's fixed `classes` list) and true label y:

    s(x, y) = 1 - p(y)

Where this sits inside a CV fold (see core/cv.py for the full picture):

  * A fold's "training pool" = (the other 4/5 of the test split) UNION
    (the entire train split).
  * A 20% calibration slice is carved OUT OF that pool. The candidate is
    fit on the REMAINING 80% and used, via a single fit_predict_proba
    call, to score BOTH the calibration slice and the held-out fold
    (X_test = concat(calib_X, held_out_X)) -- so the calibration threshold
    and the held-out prediction sets come from the exact same fitted
    model, per spec ("the calibration slice is scored by the SAME fitted
    model" / "Apply to the held-out fold").
  * Because the calibration threshold and the held-out prediction sets
    are products of this one 80%-trained model, the point predictions
    used for metric_on_kept / metric_on_sent (see core/cv.py) are this
    same model's argmax -- not the separately-fit 100%-trained "main CV"
    model. This keeps two roles cleanly separated: the 100%-trained model
    produces the headline CV mean/CI; the 80%-trained model backs the
    conformal gate and everything computed on kept/sent subsets.

Threshold: standard finite-sample-corrected split-conformal quantile.
For n_calib calibration nonconformity scores, let
    k = ceil((n_calib + 1) * (1 - alpha))
q_hat is the k-th smallest score (1-indexed); if k > n_calib, q_hat = +inf
(the calibration set is too small/extreme to support any exclusion at
this alpha, so every label is kept -> prediction sets are never singleton
-> nothing is gated as "kept"; this is the conservative, correct
behavior, not a bug).

Prediction set for a held-out item = { l in classes : 1 - p(l) <= q_hat }.
kept := |set| == 1 ; sent := otherwise.
"""
from __future__ import annotations

import math

import numpy as np


def calibration_threshold(calib_true_proba, alpha: float) -> float:
    """calib_true_proba: 1D array-like, p_model(true label) for each
    calibration item (already restricted to that item's true class column).
    """
    scores = 1.0 - np.asarray(calib_true_proba, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    scores_sorted = np.sort(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    if k < 1:
        k = 1
    return float(scores_sorted[k - 1])


def prediction_sets(proba, classes, threshold: float, *, eps: float = 1e-9):
    """proba: (n_items, n_classes) array aligned with `classes` (in order).
    Returns a list of frozensets of class labels, one per row.
    """
    classes = list(classes)
    proba = np.asarray(proba, dtype=np.float64)
    out = []
    for row in proba:
        nonconf = 1.0 - row
        members = frozenset(c for c, s in zip(classes, nonconf) if s <= threshold + eps)
        if not members:
            # Numerical edge case only (threshold finite but somehow no
            # label qualifies, e.g. all proba rows sum to slightly < 1 due
            # to float error) -- never leave a prediction set empty.
            members = frozenset([classes[int(np.argmax(row))]])
        out.append(members)
    return out


def evaluate_sets(pred_sets, y_true):
    """Core conformal bookkeeping for one fold's held-out items.

    Returns dict with:
      kept_mask     bool array, True where |set| == 1
      covered_mask  bool array, True where true label is in the set
                    (for kept items this is exactly "prediction correct")
      keep_rate     float, mean(kept_mask)
      coverage_on_kept  float, mean(covered_mask[kept_mask]) (nan if none kept)
    """
    y_true = list(y_true)
    kept_mask = np.array([len(s) == 1 for s in pred_sets], dtype=bool)
    covered_mask = np.array([yt in s for s, yt in zip(pred_sets, y_true)], dtype=bool)
    keep_rate = float(kept_mask.mean()) if len(kept_mask) else float("nan")
    coverage_on_kept = float(covered_mask[kept_mask].mean()) if kept_mask.any() else float("nan")
    return {
        "kept_mask": kept_mask,
        "covered_mask": covered_mask,
        "keep_rate": keep_rate,
        "coverage_on_kept": coverage_on_kept,
    }


def point_predictions(pred_sets, proba, classes):
    """Point prediction per item for metric_on_kept/metric_on_sent
    purposes: for a singleton set, that is trivially the prediction; for a
    larger set, we fall back to the argmax of `proba` (the same
    80%-trained model backing the sets) so every item has a definite
    predicted label to score metric_on_sent against.
    """
    classes = list(classes)
    proba = np.asarray(proba, dtype=np.float64)
    preds = []
    for s, row in zip(pred_sets, proba):
        if len(s) == 1:
            preds.append(next(iter(s)))
        else:
            preds.append(classes[int(np.argmax(row))])
    return preds
