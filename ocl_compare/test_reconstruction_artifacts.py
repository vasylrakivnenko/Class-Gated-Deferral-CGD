"""Offline checks on the reconstruction artifact.

These never touch the network. They answer the question a reviewer asks first:
does the committed artifact actually show what the README claims, and would it
have noticed if it didn't? The drift tests are the second half of that -- a
verifier that cannot fail is not evidence.

Run:  python -m pytest ocl_compare/ -q
"""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _provenance as P          # noqa: E402
import reconstruct as R          # noqa: E402

ARTIFACT = os.path.join(HERE, "results", "reconstruct.json")


@pytest.fixture(scope="module")
def art():
    assert os.path.exists(ARTIFACT), "run ocl_compare/reconstruct.py first"
    with open(ARTIFACT) as fh:
        return json.load(fh)


def test_artifact_reproduces_every_table1_cell(art):
    assert P.verify(art["measured"], R.EXPECTED) == []


def test_artifact_was_written_in_verifying_mode(art):
    assert art["verified_against_published_claim"] is True
    assert art["reproduces_published_claim"] is True
    assert art["mismatches"] == []


def test_every_upstream_file_is_hashed_and_attributed(art):
    src = art["sources"]
    # one archive + four tasks x two teachers
    assert len(src) == 9, sorted(src)
    for label, pin in src.items():
        assert len(pin["sha256"]) == 64, label
        assert pin["url"].startswith("https://"), label
        assert pin["bytes"] > 0, label


def test_github_side_is_pinned_to_a_commit(art):
    assert len(art["ocl_commit"]) == 40
    for label, pin in art["sources"].items():
        if label != "preprocessed_archive":
            assert art["ocl_commit"] in pin["url"], label


def test_each_stream_carries_a_content_fingerprint(art):
    for task in R.TASKS:
        assert len(art["measured"][task]["content_fingerprint"]) == 64


def test_run_came_from_committed_code(art):
    env = art["environment"]
    assert len(env["git_commit"]) == 40
    assert env["audit_code_dirty"] is False, (
        f"artifact produced from uncommitted ocl_compare code: "
        f"{env['audit_code_uncommitted']}")


# --- the numbers the comparison will be built on, stated explicitly ---------
@pytest.mark.parametrize("task,teacher,correct,rows,paper", [
    ("imdb", "gpt3.5", 23538, 25000, "94.15"),
    ("imdb", "llama2", 23333, 25000, "93.33"),
    ("hatespeech", "gpt3.5", 8920, 10703, "83.34"),
    ("hatespeech", "llama2", 8328, 10703, "77.81"),
    ("isear", "gpt3.5", 5393, 7666, "70.34"),
    ("isear", "llama2", 5231, 7666, "68.23"),
    ("fever", "gpt3.5", 5208, 6512, "79.98"),
    ("fever", "llama2", 5024, 6512, "77.15"),
])
def test_headline_cells(art, task, teacher, correct, rows, paper):
    m = art["measured"][task]
    assert m["n_rows"] == rows
    assert m[teacher]["n_correct"] == correct
    assert m[teacher]["paper_accuracy"] == paper


def test_hatespeech_recall_and_imbalance(art):
    m = art["measured"]["hatespeech"]
    assert m["n_positive"] == 1196                      # 1:7.95, as the paper states
    assert round((m["n_rows"] - 1196) / 1196, 2) == 7.95
    assert m["gpt3.5"]["n_true_positive"] == 996 and m["gpt3.5"]["paper_recall"] == "83.28"
    assert m["llama2"]["n_true_positive"] == 983 and m["llama2"]["paper_recall"] == "82.19"


def test_isear_is_the_only_truncated_task_and_it_matters(art):
    """Guards the one judgement call in the reconstruction. Under rounding both
    ISEAR cells land 0.01 above the paper; under truncation both are exact."""
    assert R.TRUNCATES == {"isear"}
    assert R.display("isear", 5393 / 7666) == "70.34"
    assert f"{5393 / 7666 * 100:.2f}" == "70.35"        # what rounding would give
    assert R.display("isear", 5231 / 7666) == "68.23"
    assert f"{5231 / 7666 * 100:.2f}" == "68.24"
    assert R.display("fever", 5208 / 6512) == "79.98"   # rounding, and it is right
    assert f"{math_floor_pct(5208 / 6512)}" == "79.97"  # truncation would be wrong


def math_floor_pct(x: float) -> str:
    import math
    return f"{math.floor(x * 10000) / 100:.2f}"


def test_malformed_isear_annotations_resolve_not_drop():
    """Two GPT-3.5 ISEAR rows are 'sadnessjoy' and 'sadnessfear'. First match
    over an insertion-ordered dict wins, so they are joy and sadness."""
    assert R.postprocess("isear", "sadnessjoy") == R.ISEAR_TO_ID["joy"]
    assert R.postprocess("isear", "sadnessfear") == R.ISEAR_TO_ID["sadness"]
    assert R.postprocess("isear", "5") == 5             # Llama-2 files are label ids
    assert R.postprocess("isear", "nonsense") == -1


# --- would it notice? ------------------------------------------------------
def test_drift_in_a_count_is_caught(art):
    bad = copy.deepcopy(art["measured"])
    bad["hatespeech"]["gpt3.5"]["n_correct"] += 1
    assert P.verify(bad, R.EXPECTED) != []


def test_drift_in_a_printed_figure_is_caught(art):
    bad = copy.deepcopy(art["measured"])
    bad["isear"]["gpt3.5"]["paper_accuracy"] = "70.35"
    assert P.verify(bad, R.EXPECTED) != []


def test_a_missing_key_is_caught(art):
    bad = copy.deepcopy(art["measured"])
    del bad["fever"]["llama2"]["n_correct"]
    assert any("MISSING" in m for m in P.verify(bad, R.EXPECTED))
