"""Two ways the engine can be silently wrong about which numbers it is holding.

Neither shows up as an exception or a bad-looking chart: the arrays keep their
shape and the pipeline keeps running, it is just describing something other than
what the caller asked for.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from downshift import engine as E  # noqa: E402


# --- embedding cache -------------------------------------------------------
def test_two_corpora_of_the_same_size_do_not_share_a_cache_entry():
    """The old key was (model, row count). Two datasets that happen to have the
    same number of rows would then read each other's vectors -- right shape,
    wrong meaning, and nothing downstream can tell."""
    a = [f"first corpus item {i}" for i in range(500)]
    b = [f"second corpus item {i}" for i in range(500)]
    assert len(a) == len(b)
    assert E._fingerprint(a) != E._fingerprint(b)


def test_the_fingerprint_is_stable_and_order_sensitive():
    a = ["alpha", "beta", "gamma"]
    assert E._fingerprint(a) == E._fingerprint(list(a))
    assert E._fingerprint(a) != E._fingerprint(["beta", "alpha", "gamma"])
    # a boundary that a naive join-on-separator key would collide on
    assert E._fingerprint(["ab", "c"]) != E._fingerprint(["a", "bc"])


def test_a_half_written_cache_file_is_discarded_rather_than_trusted(tmp_path):
    """A run killed mid-write leaves a truncated .npy. Loading it forever is
    worse than paying to recompute once."""
    p = tmp_path / "x.npy"
    E._cache_save(str(p), np.arange(6).reshape(2, 3))
    assert np.array_equal(E._cache_load(str(p)), np.arange(6).reshape(2, 3))
    p.write_bytes(b"not an npy file")
    assert E._cache_load(str(p)) is None
    assert not p.exists()


def test_the_cache_write_is_atomic(tmp_path):
    """np.save straight to the final name is the thing that creates the
    truncated file in the first place."""
    import inspect
    src = inspect.getsource(E._cache_save)
    assert "os.replace" in src and ".tmp" in src


# --- probability column alignment ------------------------------------------
def test_probability_columns_are_placed_by_class_not_by_position():
    """A fold whose training split is missing a rare class -- or, under
    --train-on llm, a class the teacher never predicts -- gives an estimator
    fewer columns than the global class list. Assigning them positionally files
    class 5's scores under class 4 for the whole rest of the pipeline."""
    classes = np.array([0, 1, 2, 3])
    col = {c: j for j, c in enumerate(classes)}
    pb = np.zeros((4, len(classes)))
    te = np.arange(4)
    seen = np.array([0, 2, 3])                 # class 1 absent from this fold
    pr_ = np.array([[.7, .2, .1], [.1, .8, .1], [.2, .2, .6], [.5, .3, .2]])
    for j, c in enumerate(seen):
        pb[te, col[int(c)]] = pr_[:, j]
    assert np.allclose(pb[:, 1], 0.0)          # the missing class stays empty
    assert np.allclose(pb[:, 2], pr_[:, 1])    # class 2 is NOT in column 1
    assert np.allclose(pb[:, 3], pr_[:, 2])
    src = (ROOT / "src" / "downshift" / "engine.py").read_text(encoding="utf-8")
    assert "for j, c in enumerate(seen):" in src
    assert "pb[te] = pr_" not in src
