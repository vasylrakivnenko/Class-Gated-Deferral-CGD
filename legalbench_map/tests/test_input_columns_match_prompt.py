"""The classifier may only see what the LLM saw.

Every published LLM score we compare against was produced from the task's
original prompt template (tasks/<task>/base_prompt.txt in the LegalBench
repo). The columns that template exposes -- its {{placeholders}} -- are the
ONLY columns a cheap candidate may be fed. Anything else in the HF dataset
(annotation of intermediate reasoning steps, analysis slices, filenames) is
information the LLM never had, and feeding it makes the comparison invalid.

This bit us once: diversity_1..6 ship gold sub-answers
(parties_are_diverse, aic_is_met) as extra columns. The original prompt is
{{text}} only. A blocklist-based column rule kept them, and the classifier
scored ~100% by learning `answer = diverse AND aic` -- a leak, not a result.

data/prompt_placeholders.json is the ground truth (extracted mechanically
from every selected task's base_prompt.txt). This test pins the registry to
it so the same mistake cannot recur silently for any task, present or future.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "task_registry.json"
PLACEHOLDERS = ROOT / "data" / "prompt_placeholders.json"


def _load():
    reg = json.loads(REGISTRY.read_text())
    ph = json.loads(PLACEHOLDERS.read_text())
    return reg, ph


def test_placeholder_ground_truth_covers_every_registered_task():
    reg, ph = _load()
    missing = [t["task"] for t in reg if t["task"] not in ph]
    assert not missing, (
        f"{len(missing)} registered task(s) have no prompt-placeholder record; "
        f"re-run the placeholder audit before trusting any comparison: {missing}"
    )


@pytest.mark.parametrize("entry", json.loads(REGISTRY.read_text()), ids=lambda e: e["task"])
def test_classifier_input_columns_equal_original_prompt_columns(entry):
    _, ph = _load()
    task = entry["task"]
    if task not in ph:
        pytest.skip("covered by the coverage test above")
    fed = set(entry["input_columns"])
    saw = set(ph[task])
    extra = sorted(fed - saw)
    withheld = sorted(saw - fed)
    assert not extra, (
        f"{task}: classifier is fed {extra}, which the original LLM prompt never "
        f"exposed (prompt placeholders: {sorted(saw)}). This is a leak."
    )
    assert not withheld, (
        f"{task}: original LLM prompt exposed {withheld} but the classifier "
        f"isn't fed it -- the comparison is unfair in the other direction."
    )
