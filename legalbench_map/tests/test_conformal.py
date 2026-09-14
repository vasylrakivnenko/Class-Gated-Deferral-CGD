"""On synthetic data constructed so the true marginal coverage is known
to be ~(1-alpha) (split conformal's guarantee holds for ANY exchangeable
nonconformity-score distribution, so we can draw scores from Uniform(0,1)
directly for both the calibration and test sets), empirical coverage must
land within 1.5 points of 95%."""
from __future__ import annotations

import numpy as np

from core.conformal import calibration_threshold, evaluate_sets, prediction_sets

ALPHA = 0.05


def _make_scored_items(n, n_classes, rng):
    """Returns (proba, true_label) for n items: true label drawn
    uniformly at random, nonconformity score s = 1 - p(true) drawn
    ~ Uniform(0,1), remaining probability mass split evenly across the
    other classes."""
    classes = [f"c{i}" for i in range(n_classes)]
    proba = np.zeros((n, n_classes), dtype=np.float64)
    true_labels = []
    for i in range(n):
        true_idx = rng.integers(0, n_classes)
        true_labels.append(classes[true_idx])
        s = rng.uniform(0.0, 1.0)
        p_true = 1.0 - s
        remainder = s / (n_classes - 1) if n_classes > 1 else 0.0
        row = np.full(n_classes, remainder, dtype=np.float64)
        row[true_idx] = p_true
        proba[i] = row
    return classes, proba, true_labels


def test_split_conformal_empirical_coverage_near_1_minus_alpha():
    rng = np.random.default_rng(12345)
    n_classes = 4
    n_calib = 3000
    n_test = 5000

    classes, calib_proba, calib_true = _make_scored_items(n_calib, n_classes, rng)
    _, test_proba, test_true = _make_scored_items(n_test, n_classes, rng)

    class_index = {c: i for i, c in enumerate(classes)}
    calib_true_proba = np.array([calib_proba[i, class_index[calib_true[i]]] for i in range(n_calib)])

    threshold = calibration_threshold(calib_true_proba, ALPHA)
    pred_sets = prediction_sets(test_proba, classes, threshold)
    result = evaluate_sets(pred_sets, test_true)

    # Marginal coverage over ALL test items (not just "kept"): fraction of
    # items whose true label is in the prediction set.
    marginal_coverage = float(result["covered_mask"].mean())

    assert abs(marginal_coverage - (1 - ALPHA)) <= 0.015, (
        f"marginal coverage {marginal_coverage:.4f} not within 1.5 points of {1 - ALPHA}"
    )


def test_split_conformal_kept_subset_is_at_least_as_well_covered():
    """`kept` (singleton-set) items are, by construction, the ones the
    model was confident about -- coverage restricted to that subset should
    be at least as good as the unconditional marginal guarantee (not
    exactly 1-alpha, which is a property only of the *marginal* coverage
    over all items -- see the other test in this file for that check)."""
    rng = np.random.default_rng(999)
    n_classes = 3
    n_calib = 3000
    n_test = 5000

    classes, calib_proba, calib_true = _make_scored_items(n_calib, n_classes, rng)
    _, test_proba, test_true = _make_scored_items(n_test, n_classes, rng)

    class_index = {c: i for i, c in enumerate(classes)}
    calib_true_proba = np.array([calib_proba[i, class_index[calib_true[i]]] for i in range(n_calib)])

    threshold = calibration_threshold(calib_true_proba, ALPHA)
    pred_sets = prediction_sets(test_proba, classes, threshold)
    result = evaluate_sets(pred_sets, test_true)

    assert 0.0 < result["keep_rate"] < 1.0
    marginal_coverage = float(result["covered_mask"].mean())
    assert result["coverage_on_kept"] >= marginal_coverage - 0.01
