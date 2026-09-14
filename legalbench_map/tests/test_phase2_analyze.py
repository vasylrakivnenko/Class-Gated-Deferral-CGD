"""Tests for phase2/analyze.py -- synthetic FILE A / FILE B, no network, no key."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from core.metrics import balanced_accuracy as core_balanced_accuracy
from phase2 import analyze as A

CLASSES = ["no", "yes"]
N_BOOT = 200  # small for speed; the stats code is downshift's, tested there


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_files(n=50, repeats=3, task="synthetic_task", candidate="embed_base_logreg",
               model_key="glm-5.3-flash", seed=0, cheap_acc=0.8, inc_acc=0.9, keep_p=0.7,
               parse_fail_p=0.0):
    """Synthetic FILE A (all repeats) and FILE B for one task. Returns (a, b, texts)."""
    rng = np.random.default_rng(seed)
    texts = [f"item {i} of {task} seed {seed}" for i in range(n)]
    gold = rng.choice(CLASSES, size=n).tolist()
    rows_a = []
    for r in range(repeats):
        for i in range(n):
            right = rng.random() < cheap_acc
            pred = gold[i] if right else ("yes" if gold[i] == "no" else "no")
            kept = int(rng.random() < keep_p)
            rows_a.append(dict(task=task, candidate=candidate, repeat=r, fold=i % 5, item_idx=i,
                               text_sha256=_sha(texts[i]), gold=gold[i], pred=pred,
                               p_pred=round(float(rng.uniform(0.5, 1.0)), 4), kept=kept,
                               set_size=1 if kept else int(rng.choice([0, 2]))))
    a = pd.DataFrame(rows_a, columns=A.FILE_A_COLS)
    rows_b = []
    for i in range(n):
        right = rng.random() < inc_acc
        pred = gold[i] if right else ("yes" if gold[i] == "no" else "no")
        parse_ok = 1
        if rng.random() < parse_fail_p:
            pred, parse_ok = "i cannot say", 0
        rows_b.append(dict(task=task, model_key=model_key, item_idx=i, text_sha256=_sha(texts[i]),
                           gold=gold[i], raw_output=pred.title() + "\n", pred=pred, parse_ok=parse_ok,
                           in_tokens=int(rng.integers(400, 600)), out_tokens=int(rng.integers(1, 5)),
                           cached_tokens=0, cost_usd=round(float(rng.uniform(1e-5, 5e-5)), 8),
                           latency_s=round(float(rng.uniform(0.2, 1.5)), 3), ts="2026-09-11T00:00:00Z"))
    b = pd.DataFrame(rows_b, columns=A.FILE_B_COLS)
    return a, b, texts


def _roundtrip(df, path):
    df.to_csv(path, index=False)
    return path


def _load_pair(tmp_path, a, b):
    a2 = A.load_file_a(_roundtrip(a, tmp_path / "a.csv"))
    b2 = A.load_file_b(_roundtrip(b, tmp_path / "b.csv"))
    return a2, b2


# ---------------------------------------------------------------- join integrity

def test_join_ok(tmp_path):
    a, b, _ = make_files(n=20, repeats=2)
    a2, b2 = _load_pair(tmp_path, a, b)
    merged, info = A.join_files(a2, b2)
    assert len(merged) == 40 and info["n_joined"] == 20 and not info["pilot"]
    assert info["repeats"] == [0, 1]


def test_join_sha_mismatch_raises(tmp_path):
    a, b, _ = make_files(n=20, repeats=2)
    b.loc[b["item_idx"] == 7, "text_sha256"] = _sha("a different text")
    a2, b2 = _load_pair(tmp_path, a, b)
    with pytest.raises(A.JoinError, match="text_sha256 mismatch"):
        A.join_files(a2, b2)


def test_join_missing_item_in_b_raises(tmp_path):
    a, b, _ = make_files(n=20, repeats=2)
    b = b[b["item_idx"] != 11]  # a hole in the middle, not a --limit prefix
    a2, b2 = _load_pair(tmp_path, a, b)
    with pytest.raises(A.JoinError, match=r"missing item_idx \[11\]"):
        A.join_files(a2, b2)


def test_join_extra_item_in_b_raises(tmp_path):
    a, b, _ = make_files(n=20, repeats=1)
    extra = b.iloc[[0]].copy()
    extra["item_idx"] = 20
    b = pd.concat([b, extra], ignore_index=True)
    a2, b2 = _load_pair(tmp_path, a, b)
    with pytest.raises(A.JoinError, match="not present in FILE A"):
        A.join_files(a2, b2)


def test_join_gold_mismatch_raises(tmp_path):
    a, b, _ = make_files(n=20, repeats=1)
    b.loc[b["item_idx"] == 3, "gold"] = "yes" if b.loc[b["item_idx"] == 3, "gold"].iloc[0] == "no" else "no"
    a2, b2 = _load_pair(tmp_path, a, b)
    with pytest.raises(A.JoinError, match="gold label mismatch"):
        A.join_files(a2, b2)


def test_file_a_duplicate_item_in_repeat_raises(tmp_path):
    a, b, _ = make_files(n=10, repeats=1)
    a = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        A.load_file_a(_roundtrip(a, tmp_path / "a.csv"))


def test_file_wrong_columns_raises(tmp_path):
    a, b, _ = make_files(n=5, repeats=1)
    a = a.drop(columns=["set_size"])
    with pytest.raises(ValueError, match="requires exactly"):
        A.load_file_a(_roundtrip(a, tmp_path / "a.csv"))


# ---------------------------------------------------------------- cascade arithmetic

def test_cascade_arithmetic_hand_built():
    gold = np.array(["yes"] * 5 + ["no"] * 5)
    #                 0     1     2     3     4     5     6     7     8     9
    pred_cheap = np.array(["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"])
    pred_inc = np.array(["yes", "yes", "no", "yes", "no", "no", "no", "no", "no", "yes"])
    kept = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    correct_cheap = gold == pred_cheap  # [1,0,1,0,1,1,0,1,0,1]
    correct_inc = gold == pred_inc      # [1,1,0,1,0,1,1,1,1,0]
    # kept -> cheap: idx 0,1,4,5,8 -> [1,0,1,1,0]; sent -> inc: idx 2,3,6,7,9 -> [0,1,1,1,0]
    expected_correct = np.array([1, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    got = np.where(kept == 1, correct_cheap, correct_inc)
    assert (got == expected_correct).all()
    inc_cost_per_1k = 2.0
    s = A.cascade_stats(gold, pred_cheap, pred_inc, kept, CLASSES, inc_cost_per_1k, 0.01, N_BOOT, 0)
    assert s["accuracy"] == pytest.approx(6 / 10)
    assert s["llm_share"] == pytest.approx(5 / 10)
    assert s["n_sent"] == 5
    assert s["cost_per_1k_items_usd"] == pytest.approx(0.5 * 2.0)
    # cascade balanced accuracy: pred_casc = [yes,no,no,yes,yes | no,no,no,yes,yes]
    # recall yes = 3/5, recall no = 3/5 -> 0.6
    assert s["balanced_accuracy"] == pytest.approx(0.6)
    # discordant counts vs cheap: cascade-only-right = idx 3,7? cheap[7]=1 so no. idx 3,6 ; cheap-only = idx 2,9
    assert s["mcnemar_vs_cheap"]["n_a_only"] == 2
    assert s["mcnemar_vs_cheap"]["n_b_only"] == 2
    # vs incumbent: cascade-only-right = idx 4 ; inc-only-right = idx 1,8
    assert s["mcnemar_vs_incumbent"]["n_a_only"] == 1
    assert s["mcnemar_vs_incumbent"]["n_b_only"] == 2


def test_incumbent_cost_per_1k_and_parse_fail(tmp_path):
    a, b, _ = make_files(n=40, repeats=1, parse_fail_p=0.25, seed=3)
    a2, b2 = _load_pair(tmp_path, a, b)
    inc = A.incumbent_alone_stats(b2, CLASSES)
    assert inc["cost_per_1k_items_usd"] == pytest.approx(b2["cost_usd"].sum() / 40 * 1000)
    assert inc["n_parse_fail"] == int((b2["parse_ok"] == 0).sum())
    assert inc["parse_fail_rate"] == pytest.approx(inc["n_parse_fail"] / 40)
    # unparseable preds are wrong, not coerced
    assert inc["accuracy"] == pytest.approx(float((b2["gold"] == b2["pred"]).mean()))
    assert inc["accuracy_wilson95_lo"] <= inc["accuracy"] <= inc["accuracy_wilson95_hi"]


# ---------------------------------------------------------------- oracle gate

def test_oracle_gate_ceiling():
    rng = np.random.default_rng(1)
    correct_cheap = rng.random(200) < 0.7
    correct_inc = np.where(~correct_cheap, True, rng.random(200) < 0.5)  # incumbent right on every cheap error
    o = A.oracle_gate(correct_cheap, correct_inc)
    assert o["accuracy"] == 1.0
    assert o["llm_share"] == pytest.approx(1.0 - correct_cheap.mean())
    assert o["n_sent"] == int((~correct_cheap).sum())


def test_oracle_gate_not_perfect_when_incumbent_misses():
    correct_cheap = np.array([1, 0, 0, 1], bool)
    correct_inc = np.array([0, 1, 0, 0], bool)
    o = A.oracle_gate(correct_cheap, correct_inc)
    assert o["accuracy"] == pytest.approx(3 / 4)
    assert o["llm_share"] == pytest.approx(0.5)


# ---------------------------------------------------------------- sent subset

def test_sent_subset_uses_only_kept0_rows():
    kept = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    correct_cheap = np.array([1, 1, 1, 0, 0, 1, 0, 1], bool)
    correct_inc = np.array([0, 0, 0, 1, 1, 1, 0, 0], bool)
    s = A.sent_subset_stats(correct_cheap, correct_inc, kept, N_BOOT, 0)
    assert s["n_sent"] == 4
    assert s["sent_share"] == pytest.approx(0.5)
    assert s["incumbent_accuracy"] == pytest.approx(3 / 4)  # inc on idx 3..6 = [1,1,1,0]
    assert s["cheap_accuracy"] == pytest.approx(1 / 4)      # cheap on idx 3..6 = [0,0,1,0]
    assert s["incumbent_accuracy_on_kept"] == pytest.approx(0.0)
    assert s["cheap_accuracy_on_kept"] == pytest.approx(1.0)
    m = s["mcnemar_cheap_vs_incumbent"]
    assert m["n_a_only"] == 0 and m["n_b_only"] == 2 and m["n_both_right"] == 1 and m["n_both_wrong"] == 1


def test_sent_subset_empty():
    s = A.sent_subset_stats([True, False], [True, True], [1, 1], N_BOOT, 0)
    assert s["n_sent"] == 0 and s["mcnemar_cheap_vs_incumbent"] is None
    assert np.isnan(s["incumbent_accuracy"])


# ---------------------------------------------------------------- balanced accuracy

@pytest.mark.parametrize("trial", range(30))
def test_balanced_accuracy_matches_core_metrics(trial):
    rng = np.random.default_rng(trial)
    n_classes = int(rng.integers(2, 5))
    labels = [f"c{i}" for i in range(n_classes)]
    p = rng.dirichlet(np.full(n_classes, 0.6))
    n = int(rng.integers(5, 80))
    gold = rng.choice(labels, size=n, p=p).tolist()
    pred = rng.choice(labels, size=n, p=p).tolist()
    assert A.balanced_accuracy(gold, pred, labels) == pytest.approx(core_balanced_accuracy(gold, pred, labels))


def test_balanced_accuracy_unparseable_pred_counts_wrong_like_sklearn():
    gold = ["yes", "yes", "no", "no", "no"]
    pred = ["yes", "garbage", "no", "no", "i cannot say"]
    with pytest.raises(KeyError):
        core_balanced_accuracy(gold, pred, CLASSES)
    ours = A.balanced_accuracy(gold, pred, CLASSES)
    assert ours == pytest.approx(balanced_accuracy_score(gold, pred))
    assert ours == pytest.approx((0.5 + 2 / 3) / 2)


# ---------------------------------------------------------------- verdict rules

def test_per_repeat_verdict_rules():
    v = A.per_repeat_verdict
    assert v(True, True, True, True) == "cheap-alone"      # cheap-alone wins first
    assert v(True, False, False, False) == "cheap-alone"
    assert v(False, True, True, True) == "earns"
    assert v(False, True, False, True) == "unclear"        # NI ok but not cheaper
    assert v(False, True, True, False) == "unclear"        # NI ok but does not beat cheap after Holm
    assert v(False, False, True, True) == "incumbent"


def test_pair_verdict_unclear_when_repeats_disagree(tmp_path, monkeypatch):
    a, b, _ = make_files(n=30, repeats=2, seed=5)
    a2, b2 = _load_pair(tmp_path, a, b)
    calls = iter(["earns", "incumbent"])
    real = A.repeat_stats

    def fake(*args, **kwargs):
        out = real(*args, **kwargs)
        out["verdict"] = next(calls)
        return out

    monkeypatch.setattr(A, "repeat_stats", fake)
    res = A.analyze_pair(a2, b2, CLASSES, n_boot=N_BOOT)
    assert res["per_repeat_verdicts"] == ["earns", "incumbent"]
    assert res["verdict"] == "unclear"


def test_mean_over_repeats():
    d = [{"x": 1.0, "b": True, "s": "k", "n": {"y": 2, "z": None}, "m": None},
         {"x": 3.0, "b": False, "s": "k", "n": {"y": 4, "z": 1.0}, "m": None}]
    m = A.mean_over_repeats(d)
    assert m["x"] == 2.0 and m["b"] == 0.5 and m["s"] == "k" and m["n"]["y"] == 3.0
    assert m["n"]["z"] == 1.0 and m["m"] is None


# ---------------------------------------------------------------- end to end + PILOT stamp

def _write_dirs(tmp_path, n, repeats, task, model_keys, seed=0, **kw):
    per_item = tmp_path / "per_item"
    inc = tmp_path / "incumbent"
    cfg = tmp_path / "configs"
    for d in (per_item, inc, cfg):
        d.mkdir(exist_ok=True)
    a = None
    for k, mk in enumerate(model_keys):
        a_k, b_k, _ = make_files(n=n, repeats=repeats, task=task, model_key=mk, seed=seed, **kw)
        if a is None:
            a = a_k
            a.to_csv(per_item / f"{task}__embed_base_logreg.csv", index=False)
        # FILE B keeps the same texts/gold (same seed) regardless of model
        b_k["gold"] = a["gold"].iloc[:n].values
        b_k.to_csv(inc / f"{task}__{mk}.csv", index=False)
    (cfg / f"{task}.json").write_text(json.dumps({"task": task, "best_candidate": "embed_base_logreg"}))
    return per_item, inc, cfg


def test_end_to_end_writes_outputs_no_pilot(tmp_path):
    per_item, inc, cfg = _write_dirs(tmp_path, 30, 2, "synthetic_task", ["m1", "m2"])
    out = tmp_path / "out"
    results = A.run(per_item, inc, cfg, out, registry_path=tmp_path / "no_registry.json",
                    published_path=None, n_boot=N_BOOT)
    assert len(results) == 2
    for mk in ("m1", "m2"):
        assert (out / f"synthetic_task__{mk}.md").exists()
        j = json.loads((out / f"synthetic_task__{mk}.json").read_text())
        assert j["pilot"] is False and j["n_items"] == 30
        assert j["verdict"] in {"earns", "cheap-alone", "incumbent", "unclear"}
        assert "0" in j["per_repeat"] and "1" in j["per_repeat"]
        assert set(j["per_repeat"]["0"]["holm"]["survives"]) == set(A.HOLM_FAMILY)
    summary = (out / "summary.md").read_text()
    assert "PILOT" not in summary
    assert "VERDICT RULES" in summary and "cheap-alone :" in summary and "earns       :" in summary
    assert "ASYMMETRY" in summary
    assert "| synthetic_task | m1 |" in summary


def test_pilot_stamp_when_rows_below_n_test(tmp_path):
    # registry says n_test=304 (a real task), FILE A/B have 25 items -> PILOT
    per_item, inc, cfg = _write_dirs(tmp_path, 25, 2, "opp115_data_retention", ["m1"])
    out = tmp_path / "out"
    registry = "/Users/vasyl/zadumai/legalbench_map/data/task_registry.json"
    results = A.run(per_item, inc, cfg, out, registry_path=registry, published_path=None, n_boot=N_BOOT)
    assert results[0]["pilot"] is True
    md = (out / "opp115_data_retention__m1.md").read_text()
    assert md.splitlines()[0] == "# PILOT (n=25) -- not evidence"
    summary = (out / "summary.md").read_text()
    assert summary.splitlines()[0].startswith("# PILOT (n=25) -- not evidence")
    j = json.loads((out / "opp115_data_retention__m1.json").read_text())
    assert j["pilot_stamp"] == "PILOT (n=25) -- not evidence" and j["n_test_expected"] == 304


def test_pilot_when_file_b_is_limit_prefix(tmp_path):
    # FILE A full (40 items), FILE B only item_idx < 15 (runner --limit 15)
    a, b, _ = make_files(n=40, repeats=2)
    b = b[b["item_idx"] < 15]
    a2, b2 = _load_pair(tmp_path, a, b)
    res = A.analyze_pair(a2, b2, CLASSES, n_test_expected=40, n_boot=N_BOOT)
    assert res["pilot"] and res["n_items"] == 15
    assert res["pilot_stamp"] == "PILOT (n=15) -- not evidence"
    assert res["incumbent_alone"]["n"] == 15
    assert all(s["n"] == 15 for s in res["per_repeat"].values())
    assert res["mean_over_repeats"]["n"] == 15


def test_published_2023_is_labeled_contrast_only(tmp_path):
    a, b, _ = make_files(n=20, repeats=1, task="opp115_data_retention")
    a2, b2 = _load_pair(tmp_path, a, b)
    pub = {"opp115_data_retention": {"best_score": 0.9, "best_model": "GPT-4", "metric": "balanced_accuracy",
                                     "best_source_url": "https://example.invalid"}}
    res = A.analyze_pair(a2, b2, CLASSES, n_test_expected=20, published=pub, n_boot=N_BOOT)
    assert res["published_2023"]["label"] == "published-2023"
    md = A.render_pair_md(res)
    assert "published-2023 best for this task (contrast only" in md
    assert "[measured, this run]" in md and "[CV-estimated]" in md


def test_api_error_rows_invalidate_the_pair(tmp_path):
    """Runner '<error: Name>' rows are API failures, not model answers: the pair
    must be stamped INVALID with verdict 'invalid-api-errors', in the json, the
    per-pair md and summary.md -- never silently scored as incumbent misses."""
    per_item, inc, cfg = _write_dirs(tmp_path, 30, 2, "synthetic_task", ["m1", "m2"])
    b_path = inc / "synthetic_task__m2.csv"
    b = pd.read_csv(b_path, dtype=str, keep_default_na=False)
    bad = b["item_idx"].astype(int) < 5
    b.loc[bad, ["raw_output", "pred", "parse_ok", "in_tokens", "out_tokens", "cached_tokens", "cost_usd"]] = \
        ["<error: NotFoundError>", "", "0", "0", "0", "0", "0.0000000000"]
    b.to_csv(b_path, index=False)
    out = tmp_path / "out"
    results = A.run(per_item, inc, cfg, out, registry_path=tmp_path / "no_registry.json",
                    published_path=None, n_boot=N_BOOT)
    by_model = {r["model_key"]: r for r in results}
    assert by_model["m1"]["incumbent_alone"]["n_api_errors"] == 0 and by_model["m1"]["api_error_stamp"] is None
    assert by_model["m1"]["verdict"] in {"earns", "cheap-alone", "incumbent", "unclear"}
    r2 = by_model["m2"]
    assert r2["incumbent_alone"]["n_api_errors"] == 5
    assert r2["verdict"] == "invalid-api-errors"
    assert r2["api_error_stamp"].startswith("INVALID: 5 of 30 incumbent rows are API errors")
    j = json.loads((out / "synthetic_task__m2.json").read_text())
    assert j["verdict"] == "invalid-api-errors" and j["incumbent_alone"]["n_api_errors"] == 5
    md = (out / "synthetic_task__m2.md").read_text()
    assert md.splitlines()[0].startswith("# INVALID: 5 of 30 incumbent rows are API errors")
    assert "API-error rows=5" in md
    md1 = (out / "synthetic_task__m1.md").read_text()
    assert "INVALID" not in md1 and "API-error rows=0" in md1
    summary = (out / "summary.md").read_text()
    assert summary.splitlines()[0].startswith("# INVALID: synthetic_task__m2 (5 API-error rows)")
    assert "**invalid-api-errors**" in summary
