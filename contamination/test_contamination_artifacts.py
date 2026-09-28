"""Offline check that the committed artifacts still back the published claims.

A reviewer should be able to confirm the numbers in SCOPE.md without a 300k-row
download. These tests read only the JSON in `results/` and the EXPECTED blocks
in the audit scripts, so they run in milliseconds and need no network:

  * every number the docs quote is present in the artifact and matches;
  * the artifact records the dataset revision and file hashes it was produced
    from, so "same data" is checkable;
  * the drift guard actually fires -- an audit that cannot fail is not
    evidence.

The full re-measurement is the two scripts themselves; run those to reproduce
the artifacts from the datasets.

    .venv/bin/python -m pytest contamination/ -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
AUDITS = ("beavertails_contam", "finben_headlines_contam")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as fh:
        return json.load(fh)


def _verify():
    return _load("_provenance").verify


@pytest.mark.parametrize("name", AUDITS)
def test_artifact_matches_published_claim(name):
    """Every value the docs quote is in the artifact, unchanged."""
    mismatches = _verify()(_artifact(name), _load(name).EXPECTED)
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("name", AUDITS)
def test_artifact_declares_its_own_verdict(name):
    art = _artifact(name)
    assert art["verified_against_published_claim"] is True, (
        f"{name} was produced with --no-verify; it is not evidence")
    assert art["reproduces_published_claim"] is True, art["mismatches"]


@pytest.mark.parametrize("name", AUDITS)
def test_artifact_pins_its_data(name):
    """A number without a dataset revision behind it is not reproducible."""
    art = _artifact(name)
    pin = art["dataset"]
    assert len(pin["revision"]) == 40, f"{name}: revision is not a full commit sha"
    assert pin["revision"] == _load(name).REVISION, (
        f"{name}: artifact revision differs from the pin in the script")
    assert pin["snapshot_files"], f"{name}: no file hashes recorded"
    for rel, meta in pin["snapshot_files"].items():
        assert len(meta["sha256"]) == 64, f"{name}: bad sha256 for {rel}"
    assert art["content_fingerprints"], f"{name}: no content fingerprint recorded"
    env = art["environment"]
    assert env["git_commit"], f"{name}: no repo commit recorded"
    assert env.get("audit_code_dirty") is False, (
        f"{name}: produced from uncommitted audit code, so the recorded commit "
        f"does not describe what ran: {env.get('audit_code_uncommitted')}")


@pytest.mark.parametrize("name", AUDITS)
def test_drift_is_detected(name):
    """Perturb the artifact; the guard must catch it. An audit that cannot
    fail proves nothing, so this is the test that makes the others mean
    something."""
    verify, expected = _verify(), _load(name).EXPECTED

    # walk to the first integer leaf in EXPECTED and corrupt the artifact there
    def first_int_path(d, path=()):
        for k, v in d.items():
            if isinstance(v, dict):
                got = first_int_path(v, path + (k,))
                if got:
                    return got
            elif isinstance(v, int) and not isinstance(v, bool):
                return path + (k,)
        return None

    path = first_int_path(expected)
    assert path, f"{name}: EXPECTED has no integer to perturb"

    drifted = copy.deepcopy(_artifact(name))
    node = drifted
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = node[path[-1]] + 1
    mismatches = verify(drifted, expected)
    assert mismatches, f"{name}: guard did not notice {'.'.join(path)} changing"
    assert ".".join(path) in mismatches[0]

    # a missing key must also be caught, not silently skipped
    dropped = copy.deepcopy(_artifact(name))
    node = dropped
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    assert any("MISSING" in m for m in verify(dropped, expected)), (
        f"{name}: guard did not notice {'.'.join(path)} disappearing")


def test_beavertails_headline_numbers():
    """The two figures SCOPE.md actually prints, spelled out."""
    art = _artifact("beavertails_contam")
    k = art["leakage"]["330k"]
    assert round(100 * k["test_rows_whose_prompt_is_in_train_share"], 1) == 99.8
    assert k["test_rows_whose_pair_is_in_train"] == 0, (
        "the claim is that prompts repeat while PAIRS do not")
    assert art["duplicates"]["disagree_pct"]["is_safe"] == 27.7
    assert art["duplicates"]["disagree_pct"]["non_violent_unethical_behavior"] == 23.9


def test_finben_headline_numbers():
    art = _artifact("finben_headlines_contam")
    lk = art["leakage"]
    assert round(100 * lk["total_leak_share"], 1) == 71.5
    assert lk["leak_free_rows_per_subtask"] == 651, (
        "SCOPE.md quotes a leak-free score on 651 rows per sub-task")
    d = art["derivation"]
    assert d["test_rows_agreeing_with_label_type"] == d["n_test_rows"] == 20547, (
        "the per-sub-task split is only sound if it reproduces label_type exactly")
