"""Lenient scoring: reported ALONGSIDE exact-match, never instead of it."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase2 import analyze as A  # noqa: E402

CLASSES = ["no", "yes"]


@pytest.mark.parametrize("raw,expected_pred,expected_ok", [
    ("Yes", "yes", 1),
    ("**Label: Yes**\n\nThis post discusses a money issue.", "yes", 1),   # GLM's pilot pattern
    ("Label: No", "no", 1),                                                   # gpt-oss echo pattern
    ("Yes\nYes\nYes\nNo\nNo", "yes", 1),                                     # template re-labeling: earliest wins
    ("The answer is no, because the clause is silent.", "no", 1),
    ("I cannot determine this.", "i cannot determine this", 0),               # no label anywhere -> miss, uncoerced
    ("", "", 0),
    ("<error: NotFoundError>", "error notfounderror", 0),                     # API error never parses (normalize strips punctuation)
    ("Yesterday it rained.", "yesterday it rained", 0),                       # whole-word: 'yes' inside 'yesterday' does not match
])
def test_lenient_pred(raw, expected_pred, expected_ok):
    pred, ok = A.lenient_pred(raw, CLASSES)
    assert (pred, ok) == (expected_pred, expected_ok)


def test_lenient_tie_prefers_longest_label():
    pred, ok = A.lenient_pred("not applicable here", ["not", "not applicable"])
    assert (pred, ok) == ("not applicable", 1)


def test_add_lenient_columns_does_not_touch_exact_columns():
    b = pd.DataFrame({"raw_output": ["**Label: Yes**", "No"], "pred": ["label yes", "no"], "parse_ok": [0, 1]})
    out = A.add_lenient_columns(b, CLASSES)
    assert out["pred"].tolist() == ["label yes", "no"] and out["parse_ok"].tolist() == [0, 1]
    assert out["pred_lenient"].tolist() == ["yes", "no"] and out["parse_ok_lenient"].tolist() == [1, 1]
    assert "pred_lenient" not in b.columns  # original untouched


def test_run_writes_both_scorings(tmp_path):
    """End-to-end on a synthetic pair: exact files keep their names; lenient gets __lenient; summary has both tables."""
    import json, hashlib
    task, cand, mk = "synthetic_task", "tfidf_logreg", "m1"
    n, reps = 20, 2
    texts = [f"item {i}" for i in range(n)]
    shas = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    gold = ["yes" if i % 2 else "no" for i in range(n)]
    per_item = tmp_path / "per_item"; per_item.mkdir()
    rows = []
    for r in range(reps):
        for i in range(n):
            rows.append(dict(task=task, candidate=cand, repeat=r, fold=i % 5, item_idx=i, text_sha256=shas[i],
                             gold=gold[i], pred=gold[i] if i % 3 else ("no" if gold[i] == "yes" else "yes"),
                             p_pred=0.9, kept=int(i % 4 != 0), set_size=1 if i % 4 else 2))
    pd.DataFrame(rows, columns=A.FILE_A_COLS).to_csv(per_item / f"{task}__{cand}.csv", index=False)
    inc = tmp_path / "incumbent"; inc.mkdir()
    # incumbent answers every item correctly but wraps it -> exact scores 0%, lenient scores 100%
    brows = [dict(task=task, model_key=mk, item_idx=i, text_sha256=shas[i], gold=gold[i],
                  raw_output=f"**Label: {gold[i].capitalize()}**\n\nrationale", pred=f"label {gold[i]} rationale",
                  parse_ok=0, in_tokens=100, out_tokens=50, cached_tokens=0, cost_usd=0.0001, latency_s=0.5,
                  ts="2026-09-11T00:00:00Z") for i in range(n)]
    pd.DataFrame(brows, columns=A.FILE_B_COLS).to_csv(inc / f"{task}__{mk}.csv", index=False)
    cfg = tmp_path / "configs"; cfg.mkdir()
    (cfg / f"{task}.json").write_text(json.dumps({"task": task, "best_candidate": cand}))
    out = tmp_path / "out"
    results = A.run(per_item, inc, cfg, out, registry_path=tmp_path / "none.json", published_path=None,
                    n_boot=200, seed=0)
    assert len(results) == 1 and results[0]["scoring"] == "exact"
    assert (out / f"{task}__{mk}.md").exists() and (out / f"{task}__{mk}__lenient.md").exists()
    je = json.loads((out / f"{task}__{mk}.json").read_text())
    jl = json.loads((out / f"{task}__{mk}__lenient.json").read_text())
    assert je["incumbent_alone"]["accuracy"] == 0.0 and je["incumbent_alone"]["parse_fail_rate"] == 1.0
    assert jl["incumbent_alone"]["accuracy"] == 1.0 and jl["incumbent_alone"]["parse_fail_rate"] == 0.0
    summary = (out / "summary.md").read_text()
    assert "## Exact-match scoring" in summary and "## Lenient scoring" in summary
    assert "used for the cascade decision" in summary
