"""Exercises the ACTUAL fold-construction code (core.cv) on a small
synthetic task and asserts held-out/training-fold index sets are
disjoint for every fold of every repeat -- both at the low-level helper
functions and via a full, real run_task_candidate_cv call (which has its
own internal assertions that would raise if a leak were ever introduced)."""
from __future__ import annotations

import numpy as np

from core.cv import _carve_calibration_slice, _derive_seed, _stratified_positions, run_task_candidate_cv
from tests.helpers import CallableModule, make_synthetic_task_data


def test_stratified_positions_disjoint_and_complete():
    y = [f"c{i % 4}" for i in range(40)]
    for repeat in range(3):
        seed = _derive_seed(0, "leaktest", "cand", "fold_partition", str(repeat))
        splits = _stratified_positions(y, n_splits=5, random_state=seed)
        assert len(splits) == 5
        all_heldout = []
        for train_pos, heldout_pos in splits:
            assert set(train_pos.tolist()).isdisjoint(set(heldout_pos.tolist())), (
                f"repeat={repeat}: train/heldout overlap"
            )
            all_heldout.extend(heldout_pos.tolist())
        # every fold's held-out set partitions [0, len(y)) exactly once
        assert sorted(all_heldout) == list(range(len(y)))


def test_carve_calibration_slice_disjoint():
    y = [f"c{i % 3}" for i in range(30)]
    for trial in range(5):
        train80_pos, calib_pos = _carve_calibration_slice(y, 0.2, random_state=trial)
        assert set(train80_pos.tolist()).isdisjoint(set(calib_pos.tolist()))
        assert sorted(train80_pos.tolist() + calib_pos.tolist()) == list(range(len(y)))


def test_full_cv_run_has_no_leak_and_covers_every_item():
    """A real run_task_candidate_cv call on a tiny synthetic task using the
    fast, deterministic 'majority' baseline. If any held-out/training
    overlap were ever introduced, the internal `_assert_disjoint` calls in
    core.cv would raise -- this test simply confirms the real code path
    runs clean, and additionally checks output shape/coverage sanity."""
    import candidates.baselines as baselines

    task_data = make_synthetic_task_data(n_train=12, n_test=40, n_classes=4)
    result = run_task_candidate_cv(
        task_data,
        CallableModule(baselines.fit_predict_proba),
        "majority",
        folds=5,
        repeats=3,
        alpha=0.2,
        seed=0,
        cache_dir="/private/tmp/legalbench_map_test_cache",
    )
    assert result.n_items == 40
    assert 0.0 <= result.cv_mean <= 1.0
    assert 0.0 <= result.keep_rate_mean <= 1.0
